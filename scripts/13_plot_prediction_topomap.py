"""Plot a topographic summary of the frozen 204-channel prediction."""

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
    parser = argparse.ArgumentParser(
        description="Plot predicted MEG activity on the 204-channel sensor layout."
    )
    parser.add_argument(
        "prediction_file",
        type=Path,
        nargs="?",
        default=Path(
            "results/subject60/spatial_saccade_scale025_eval_predictions.npy"
        ),
    )
    parser.add_argument(
        "--info-file",
        type=Path,
        default=Path("challenge1/subject60/sub-60_grad_info.fif"),
    )
    parser.add_argument(
        "--metric",
        choices=("rms", "mean_abs", "signed_mean"),
        default="rms",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "results/subject60/spatial_saccade_scale025_eval_topomap_rms.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = np.asarray(np.load(args.prediction_file.resolve()), dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[1] != 204:
        raise ValueError(f"Expected prediction shape (n_rows, 204), got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values")

    if args.metric == "rms":
        values = np.sqrt(np.mean(prediction**2, axis=0))
        label = "Predicted RMS activity"
        cmap = "magma"
        vlim = (float(values.min()), float(values.max()))
    elif args.metric == "mean_abs":
        values = np.mean(np.abs(prediction), axis=0)
        label = "Predicted mean absolute activity"
        cmap = "magma"
        vlim = (float(values.min()), float(values.max()))
    else:
        values = np.mean(prediction, axis=0)
        label = "Predicted signed mean activity"
        limit = float(np.max(np.abs(values)))
        cmap = "RdBu_r"
        vlim = (-limit, limit)

    info = mne.io.read_info(args.info_file.resolve(), verbose="ERROR")
    if len(info["chs"]) != 204:
        raise ValueError(f"Expected 204 channels in {args.info_file}, got {len(info['chs'])}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    image, _ = mne.viz.plot_topomap(
        values,
        info,
        ch_type="grad",
        axes=axis,
        sensors=True,
        contours=0,
        cmap=cmap,
        vlim=vlim,
        extrapolate="head",
        show=False,
    )
    axis.set_title(f"Subject 60 evaluation prediction\n{label}")
    fig.colorbar(image, ax=axis, shrink=0.75, label="prediction units")
    fig.savefig(args.output_file.resolve(), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("PREDICTION TOPOMAP")
    print("=" * 72)
    print(f"Prediction: {args.prediction_file.resolve()}")
    print(f"Info file : {args.info_file.resolve()}")
    print(f"Metric    : {args.metric}")
    print(f"Range     : {values.min():.6e} to {values.max():.6e}")
    print(f"Saved     : {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
