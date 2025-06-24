import os
import shutil
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN

def extract_features(path):
    mesh = trimesh.load_mesh(path)
    verts = mesh.vertices
    if verts is None or verts.size == 0:
        return None
    max_z = np.max(verts[:, 2])
    depths = max_z - verts[:, 2]
    cut_pts = verts[depths > 0.01]
    if cut_pts.size == 0:
        return (0.0, 0.0, 0.0)
    xy = cut_pts[:, :2]
    labels = DBSCAN(eps=0.02, min_samples=3).fit_predict(xy)
    clusters = [cut_pts[labels == cid] for cid in set(labels) if cid >= 0]
    depth_list, width_list, curv_list = [], [], []

    if not clusters:
        return (0.0, 0.0, 0.0)
    for cl in clusters:
        d = max_z - cl[:, 2]
        depth_list.append(d.mean())
        xs, ys = cl[:,0], cl[:,1]
        width_list.append(max(xs.max()-xs.min(), ys.max()-ys.min()))
        # approximate curvature: ratio arc length to chord length
        chord = np.linalg.norm(cl[-1] - cl[0])
        arc = np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1))
        curv_list.append((arc - chord) / chord if chord > 0 else 0.0)
    return (float(np.mean(depth_list)),
            float(np.mean(width_list)),
            float(np.mean(curv_list)))

def main():
    stl_folder = r"C:\Users\szymi\OneDrive\Pulpit\Studia\PROJEKT_FIGURKI\Nożyki\Testowe"
    output_folder = r"C:\Users\szymi\OneDrive\Pulpit\Studia\II_Stopień\AUM\PointNet\dataset"
    os.makedirs(output_folder, exist_ok=True)

    features = []
    paths = []
    for fname in os.listdir(stl_folder):
        if not fname.lower().endswith('.stl'):
            continue
        path = os.path.join(stl_folder, fname)
        feat = extract_features(path)
        if feat is None:
            continue
        features.append(feat)
        paths.append(path)

    feats = np.array(features)
    depth_thresholds = float(np.percentile(feats[:,0], 50))
    width_thresholds = float(np.percentile(feats[:,1], 50))
    curv_thresholds  = float(np.percentile(feats[:,2], 50))
    print("Thresholds:")
    print(f"Depth: {depth_thresholds:.4f}")
    print(f"Width: {width_thresholds:.4f}")
    print(f"Curvature: {curv_thresholds:.4f}")

    def assign_class(value, thr):
        return 0 if value < thr else 1

    for (depth, width, curv), path in zip(features, paths):
        d_cls = assign_class(depth, depth_thresholds)
        w_cls = assign_class(width, width_thresholds)
        c_cls = assign_class(curv, curv_thresholds)
        label = f"{d_cls}_{w_cls}_{c_cls}"
        dst_dir = os.path.join(output_folder, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))

    print("Done. Files classified into dataset/ by dynamic thresholds.")

if __name__ == "__main__":
    main()
