from pathlib import Path

import numpy as np
import typer

from common import create_model, plot_weights


def inner_main(
    evaluate_files_dir: Path = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to folder containing training files.",
    ),
    model_path: Path = typer.Option(
        "training",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to folder containing tensorflow model.",
    ),
) -> None:
    inputs = np.loadtxt(f"{evaluate_files_dir}/inputs.txt")
    expected_output = np.loadtxt(f"{evaluate_files_dir}/expected_output.txt")
    model = create_model()

    checkpoint_path = f"{model_path}/training.weights.h5"
    model.load_weights(checkpoint_path)
    loss, acc = model.evaluate(inputs, expected_output, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    plot_weights(model)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
