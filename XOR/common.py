import tensorflow as tf  # type: ignore[import-untyped]
from matplotlib import pyplot as plt


def create_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(2,)),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model


def plot_weights(model: tf.keras.Sequential) -> None:
    weights = model.get_weights()

    for i in range(len(weights)):
        if len(weights[i].shape) == 2:
            plt.figure(figsize=(10, 5))
            plt.hist(weights[i].flatten(), bins=30)
            plt.title("Weights distribution - Layer {}".format(i))
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
