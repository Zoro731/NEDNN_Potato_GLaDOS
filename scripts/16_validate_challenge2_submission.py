"""Validate the Challenge 2 ``(fixation, channel, timepoint)`` contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Challenge 2 prediction file")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument(
        "--times-file",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/times.npy"),
    )
    parser.add_argument(
        "--time-ms",
        type=float,
        nargs=6,
        default=[-50.0, 50.0, 75.0, 100.0, 125.0, 150.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = np.load(args.prediction.resolve(), mmap_mode="r")
    metadata_rows = len(pd.read_csv(args.metadata.resolve()))
    if prediction.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {prediction.shape}")
    if prediction.shape != (metadata_rows, 204, 6):
        raise ValueError(
            f"Expected shape ({metadata_rows}, 204, 6), got {prediction.shape}"
        )
    if prediction.dtype != np.float32:
        raise ValueError(f"Expected float32 predictions, got {prediction.dtype}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values")

    times = np.asarray(np.load(args.times_file.resolve()), dtype=np.float64).reshape(-1)
    requested = np.asarray(args.time_ms, dtype=np.float64) / 1000.0
    indices = np.abs(times[:, None] - requested[None, :]).argmin(axis=0)
    actual_ms = times[indices] * 1000.0

    print("Submission:", args.prediction.resolve())
    print("Metadata:", args.metadata.resolve())
    print("Shape:", prediction.shape)
    print("Dtype:", prediction.dtype)
    print("Time indices:", indices.tolist())
    print("Actual times (ms):", actual_ms.round(3).tolist())
    print(
        "Values: min={:.6e}, max={:.6e}, mean={:.6e}".format(
            float(prediction.min()), float(prediction.max()), float(prediction.mean())
        )
    )
    print("VALID: Challenge 2 shape, dtype, time axis, and finite-value contract satisfied.")
    print("NOTE: row order is assumed to match the supplied metadata exactly.")


if __name__ == "__main__":
    main()
