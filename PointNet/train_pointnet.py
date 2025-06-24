import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------------
# Dataset loader for grouped STL folders
# -------------------------------
class STLDataset(Dataset):
    def __init__(self, root_dir, max_points=2048, augment=False):
        self.max_points = max_points
        self.augment = augment
        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.samples = []
        for cls in classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith('.stl'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mesh = trimesh.load_mesh(path)
        verts = mesh.vertices
        max_z = verts[:, 2].max()
        top_pts = verts[verts[:, 2] >= (max_z - 0.01)]
        if len(top_pts) >= self.max_points:
            choice = np.random.choice(len(top_pts), self.max_points, replace=False)
            pts = top_pts[choice]
        else:
            pad = np.zeros((self.max_points - len(top_pts), 3))
            pts = np.vstack([top_pts, pad])
        if self.augment:
            # obrót Z ±10°
            ang = np.random.uniform(-10,10) * np.pi/180
            R = np.array([[np.cos(ang), -np.sin(ang), 0],
                          [np.sin(ang),  np.cos(ang), 0],
                          [0,0,1]])
            pts = pts.dot(R.T)
            # delikatny szum
            pts += np.random.normal(scale=1e-3, size=pts.shape)
        return torch.tensor(pts, dtype=torch.float32), label

# -------------------------------
# Enhanced PointNet classifier
# -------------------------------
class SimplePointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3   = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4   = nn.BatchNorm1d(512)
        # fully connected
        self.fc1 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2)[0]  # global max pool -> (B, 512)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)

# -------------------------------
# Training script with Early Stopping and extended metrics
# -------------------------------
def train_pointnet_es(data_dir,
                      epochs=80,
                      batch_size=32,
                      lr=1e-2,
                      patience=10,
                      min_delta=1e-4):
    # Prepare dataset
    dataset = STLDataset(data_dir, max_points=2048, augment=True)
    num_classes = len(dataset.class_to_idx)
    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePointNet(num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [],  'val_acc': [],
        'train_mse':  [],  'val_mse':  [],
        'train_err':  [],  'val_err':  [],
        'wmean_conv1': [], 'wstd_conv1': [],
        'wmean_conv2': [], 'wstd_conv2': [],
        'wmean_conv3': [], 'wstd_conv3': [],
        'wmean_conv4': [], 'wstd_conv4': [],
        'wmean_fc1':   [], 'wstd_fc1':   [],
        'wmean_fc2':   [], 'wstd_fc2':   []
    }

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        # -------- TRAIN --------
        model.train()
        run_loss, run_corr, run_mse = 0.0, 0, 0.0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pts)
            loss_ce = criterion_ce(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            onehot = F.one_hot(labels, num_classes).float().to(device)
            loss_mse = F.mse_loss(probs, onehot)
            loss = loss_ce
            loss.backward()
            optimizer.step()

            run_loss += loss_ce.item() * pts.size(0)
            run_corr += (outputs.argmax(1) == labels).sum().item()
            run_mse  += loss_mse.item() * pts.size(0)

        avg_train_loss = run_loss / train_size
        train_acc = run_corr / train_size
        avg_train_mse = run_mse / train_size
        train_err = 1 - train_acc

        # -------- VALIDATION --------
        model.eval()
        val_loss, val_corr, val_mse = 0.0, 0, 0.0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts, labels = pts.to(device), labels.to(device)
                outputs = model(pts)
                loss_ce = criterion_ce(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                onehot = F.one_hot(labels, num_classes).float().to(device)
                loss_mse = F.mse_loss(probs, onehot)

                val_loss += loss_ce.item() * pts.size(0)
                val_corr += (outputs.argmax(1) == labels).sum().item()
                val_mse  += loss_mse.item() * pts.size(0)

        avg_val_loss = val_loss / val_size
        val_acc = val_corr / val_size
        avg_val_mse = val_mse / val_size
        val_err = 1 - val_acc

        # record history
        h = history
        h['train_loss'].append(avg_train_loss)
        h['val_loss'].append(avg_val_loss)
        h['train_acc'].append(train_acc)
        h['val_acc'].append(val_acc)
        h['train_mse'].append(avg_train_mse)
        h['val_mse'].append(avg_val_mse)
        h['train_err'].append(train_err)
        h['val_err'].append(val_err)

        # record weight stats
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name == 'conv1.weight':
                    h['wmean_conv1'].append(param.mean().item())
                    h['wstd_conv1'].append(param.std().item())
                if name == 'conv2.weight':
                    h['wmean_conv2'].append(param.mean().item())
                    h['wstd_conv2'].append(param.std().item())
                if name == 'conv3.weight':
                    h['wmean_conv3'].append(param.mean().item())
                    h['wstd_conv3'].append(param.std().item())
                if name == 'conv4.weight':
                    h['wmean_conv4'].append(param.mean().item())
                    h['wstd_conv4'].append(param.std().item())
                if name == 'fc1.weight':
                    h['wmean_fc1'].append(param.mean().item())
                    h['wstd_fc1'].append(param.std().item())
                if name == 'fc2.weight':
                    h['wmean_fc2'].append(param.mean().item())
                    h['wstd_fc2'].append(param.std().item())

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, MSE: {avg_train_mse:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, MSE: {avg_val_mse:.4f}")

        # scheduler and early stopping
        scheduler.step(avg_val_loss)
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_pointnet_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best model
    model.load_state_dict(torch.load('best_pointnet_model.pth'))

    # -------------------------------
    # Plot metrics over epochs
    # -------------------------------
    epochs_range = list(range(1, len(history['train_loss'])+1))

    # 1) Loss & accuracy
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_loss'], label='Train CE Loss')
    plt.plot(epochs_range, history['val_loss'],   label='Val CE Loss')
    plt.title('CrossEntropy Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'],   label='Val Acc')
    plt.title('Accuracy'); plt.legend()
    plt.savefig('metrics_loss_acc.png')
    plt.show()

    # 2) Classification error & MSE
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_err'], label='Train Err')
    plt.plot(epochs_range, history['val_err'],   label='Val Err')
    plt.title('Classification Error'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['train_mse'], label='Train MSE')
    plt.plot(epochs_range, history['val_mse'],   label='Val MSE')
    plt.title('Mean Squared Error'); plt.legend()
    plt.savefig('metrics_err_mse.png')
    plt.show()

    # 3) Weight evolution per layer
    plt.figure(figsize=(12, 8))
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
    for i, lyr in enumerate(layers):
        plt.subplot(3,2,i+1)
        plt.plot(epochs_range, history[f'wmean_{lyr}'], label='mean')
        stds = history[f'wstd_{lyr}']
        means = history[f'wmean_{lyr}']
        plt.fill_between(epochs_range,
                         np.array(means)-np.array(stds),
                         np.array(means)+np.array(stds),
                         alpha=0.3)
        plt.title(f'Weights {lyr}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig('weights_evolution.png')
    plt.show()

    # Confusion matrix on validation
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for pts, labels in val_loader:
            pts = pts.to(device)
            preds = model(pts).argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(cm,
        display_labels=[dataset.idx_to_class[i] for i in range(num_classes)])
    plt.figure(figsize=(8,8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Validation)")
    plt.savefig('confusion_matrix_es.png')
    plt.show()

    return model, dataset.class_to_idx

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    _, _ = train_pointnet_es(
        data_dir=r"C:\Users\szymi\OneDrive\Pulpit\Studia\II_Stopień\AUM\PointNet\dataset",
        epochs=80,
        batch_size=32,
        lr=1e-2,
        patience=10,
        min_delta=1e-4
    )

