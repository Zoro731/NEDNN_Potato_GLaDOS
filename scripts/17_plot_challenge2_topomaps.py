"""Plot six time-resolved topomaps from a Challenge 2 prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Challenge 2 prediction topomaps")
    parser.add_argument(
        "prediction_file",
        type=Path,
        nargs="?",
        default=Path("results/challenge2/subject60_predictions.npy"),
    )
    parser.add_argument(
        "--info-file",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/sub-01_grad_info.fif"),
    )
    parser.add_argument(
        "--times-file",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/times.npy"),
    )
    parser.add_argument(
        "--metric",
        choices=("rms", "mean_abs", "signed_mean"),
        default="mean_abs",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/challenge2/subject60_topomaps_mean_abs.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = np.asarray(np.load(args.prediction_file.resolve()), dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[1:] != (204, 6):
        raise ValueError(f"Expected prediction shape (n_rows, 204, 6), got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values")

    times = np.asarray(np.load(args.times_file.resolve()), dtype=np.float64).reshape(-1)
    requested_ms = np.asarray([-50.0, 50.0, 75.0, 100.0, 125.0, 150.0])
    indices = np.abs(times[:, None] - requested_ms[None, :] / 1000.0).argmin(axis=0)
    actual_ms = times[indices] * 1000.0

    if args.metric == "rms":
        values = np.sqrt(np.mean(prediction**2, axis=0))
        label = "Predicted RMS activity"
        cmap = "magma"
        vmin, vmax = 0.0, float(values.max())
    elif args.metric == "mean_abs":
        values = np.mean(np.abs(prediction), axis=0)
        label = "Predicted mean absolute activity"
        cmap = "magma"
        vmin, vmax = 0.0, float(values.max())
    else:
        values = np.mean(prediction, axis=0)
        label = "Predicted signed mean activity"
        limit = float(np.max(np.abs(values)))
        cmap = "RdBu_r"
        vmin, vmax = -limit, limit

    info = mne.io.read_info(args.info_file.resolve(), verbose="ERROR")
    if len(info["chs"]) != 204:
        raise ValueError(f"Expected 204 channels in {args.info_file}, got {len(info['chs'])}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    image = None
    for position, axis in enumerate(axes.flat):
        image, _ = mne.viz.plot_topomap(
            values[:, position],
            info,
            ch_type="grad",
            axes=axis,
            sensors=True,
            contours=0,
            cmap=cmap,
            vlim=(vmin, vmax),
            extrapolate="head",
            show=False,
        )
        axis.set_title(f"requested {requested_ms[position]:g} ms\nstored {actual_ms[position]:g} ms")
    fig.suptitle(f"Subject 60 Challenge 2 prediction\n{label}")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, label="prediction units")
    fig.savefig(args.output_file.resolve(), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("CHALLENGE 2 PREDICTION TOPOMAPS")
    print("=" * 72)
    print(f"Prediction: {args.prediction_file.resolve()}")
    print(f"Metric    : {args.metric}")
    print(f"Stored ms : {actual_ms.round(3).tolist()}")
    print(f"Saved     : {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
