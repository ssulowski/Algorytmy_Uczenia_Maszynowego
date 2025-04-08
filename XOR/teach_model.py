from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]
import typer

from common import create_model


def inner_main(
    train_files_dir: Path = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to folder containing training files.",
    ),
    val_files_dir: Path = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to folder containing training files.",
    ),
    epochs: int = typer.Option(
        1000,
        help="Number of epochs in learning",
    ),
    verbose: int = typer.Option(
        0,
        help="Verbose value",
    ),
    output_dir: Path = typer.Option(
        "training",
        dir_okay=True,
        file_okay=False,
        help="Path to teaching output folder.",
    ),
    weights: Path = typer.Option(
        None,
        dir_okay=False,
        file_okay=True,
        help="Path to teaching output folder.",
    ),
) -> None:
    inputs_train = np.loadtxt(f"{train_files_dir}/inputs.txt")
    expected_output_train = np.loadtxt(f"{train_files_dir}/expected_output.txt")
    inputs_val = np.loadtxt(f"{val_files_dir}/inputs.txt")
    expected_output_val = np.loadtxt(f"{val_files_dir}/expected_output.txt")
    model = create_model()

    current_date = datetime.now().date()
    checkpoint_path = f"{output_dir}/{current_date}"
    Path(checkpoint_path).mkdir(exist_ok=True, parents=True)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{checkpoint_path}/training.weights.h5",
        save_weights_only=True,
        verbose=verbose,
    )

    log_file = f"{checkpoint_path}/log.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(filename=log_file, separator=",", append=True)

    if weights is not None:
        model.load_weights(weights)

    model.fit(
        inputs_train,
        expected_output_train,
        epochs=epochs,
        validation_data=(inputs_val, expected_output_val),
        callbacks=[cp_callback, csv_logger],
    )

    model.summary()


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
