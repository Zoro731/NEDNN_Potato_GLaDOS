"""Generate a format-reference mean prediction for Challenge 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=ROOT
        / "challenge1"
        / "subject60"
        / "challenge1_dev"
        / "metadata.csv",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ROOT / "results" / "example_submission_mean_baseline.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = ROOT / "challenge1" / "training" / "meg_110ms.npy"
    train = np.load(train_path, mmap_mode="r")
    metadata = pd.read_csv(args.metadata_csv)

    channel_mean = train.mean(axis=0).astype(np.float32)
    prediction = np.repeat(channel_mean[None, :], len(metadata), axis=0)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_file, prediction)
    print(
        f"Saved {args.output_file.resolve()}  "
        f"shape={prediction.shape} dtype={prediction.dtype}"
    )


if __name__ == "__main__":
    main()
