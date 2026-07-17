"""Validate Challenge 1 prediction shape, order assumptions, and values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument(
        "--channel-names",
        type=Path,
        default=ROOT / "challenge1" / "training" / "channel_names.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = np.load(args.submission, mmap_mode="r")
    metadata = pd.read_csv(args.metadata_csv)
    channel_names = [
        line.strip()
        for line in args.channel_names.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_shape = (len(metadata), len(channel_names))

    errors: list[str] = []
    warnings: list[str] = []
    if prediction.shape != expected_shape:
        errors.append(f"shape is {prediction.shape}; expected {expected_shape}")
    if prediction.ndim != 2:
        errors.append("Challenge 1 predictions must be a 2D array")
    if not np.issubdtype(prediction.dtype, np.floating):
        errors.append(f"dtype {prediction.dtype} is not floating point")
    elif prediction.dtype != np.float32:
        warnings.append(f"dtype is {prediction.dtype}; float32 is recommended")
    if not np.isfinite(prediction).all():
        errors.append("array contains NaN or infinite values")
    if len(set(channel_names)) != len(channel_names):
        errors.append("channel_names.txt contains duplicate channel names")

    print(f"Submission : {args.submission.resolve()}")
    print(f"Metadata   : {args.metadata_csv.resolve()}")
    print(f"Shape      : {prediction.shape}")
    print(f"Dtype      : {prediction.dtype}")
    print(f"Channels   : {len(channel_names)}")
    if prediction.size:
        print(
            "Values     : "
            f"min={float(prediction.min()):.6e}, "
            f"max={float(prediction.max()):.6e}, "
            f"mean={float(prediction.mean()):.6e}"
        )

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("VALID: shape and values satisfy the Challenge 1 format contract.")
    print("NOTE: row order is assumed to match the supplied metadata exactly.")


if __name__ == "__main__":
    main()
