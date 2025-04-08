import random
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray


def save_to_file(
    inputs: NDArray[np.int_],
    expected_output: NDArray[np.int_],
    split_name: str,
    output_dir: Path,
    samples_amout: int,
) -> None:
    current_date = datetime.now().date()
    output_path = output_dir / f"{current_date}.{samples_amout}_samples.{split_name}"
    output_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(f"{output_path}/inputs.txt", inputs, fmt="%d")
    np.savetxt(f"{output_path}/expected_output.txt", expected_output, fmt="%d")
    print(f"Dataset {split_name} saved in {output_path}")


def inner_main(
    samples_amount: int = typer.Argument(
        None,
        help="Number of generated learning samples.",
    ),
    split_name: str = typer.Option(
        "train",
        help="Split name.",
    ),
    output_dir: Path = typer.Option(
        "XOR_data",
        dir_okay=True,
        file_okay=False,
        exists=False,
        help="Output dir of file",
    ),
) -> None:
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = np.array([0, 1, 1, 0])

    if samples_amount is not None:
        indices = [random.randint(0, 3) for _ in range(samples_amount)]
        inputs = xor_inputs[indices]
        expected_output = xor_outputs[indices]
    else:
        inputs = np.array([(i, j) for i in (0, 1) for j in (0, 1)])
        expected_output = np.array([i ^ j for i, j in inputs])

    positive_counter = int(np.sum(expected_output))
    negative_counter = len(expected_output) - positive_counter

    print(f"Generated:\npositive: {positive_counter}\nnegative: {negative_counter}")

    save_to_file(inputs, expected_output, split_name, output_dir, len(inputs))


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
