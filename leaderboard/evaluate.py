"""Scoring logic for Challenge 1 submissions. Keep hidden from participants."""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
GT_PATH = REPO_ROOT / "data/brainencoding26/challenge1/subject60/ground_truth/challenge1_dev_meg_110ms.npy"
def _expected_shape() -> tuple[int, int]:
    return tuple(np.load(GT_PATH, mmap_mode="r").shape)


def validate_submission(arr: np.ndarray) -> str | None:
    """Return an error string if invalid, None if valid."""
    expected = _expected_shape()
    if arr.ndim != 2 or arr.shape != expected:
        return f"Expected shape {expected}, got {arr.shape}"
    if not np.isfinite(arr).all():
        return "Submission contains NaN or Inf values"
    return None


def score_submission(pred: np.ndarray) -> dict:
    """Compute per-channel Pearson r and aggregate metrics.

    Returns dict with keys: max_r, mean_r, median_r, worst10_mean, r_per_channel.
    r_per_channel is a (204,) array kept in memory for downstream use; not saved to CSV.
    """
    gt = np.load(GT_PATH)

    # Vectorised Pearson r: centre each column then dot-product
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    gt_c = gt - gt.mean(axis=0, keepdims=True)
    pred_std = pred_c.std(axis=0)
    gt_std = gt_c.std(axis=0)

    # Avoid division by zero for constant channels
    denom = pred_std * gt_std
    safe = denom > 0

    n_channels = gt.shape[1]
    dots = (pred_c * gt_c).mean(axis=0)
    r_per_channel = np.zeros(n_channels)
    r_per_channel[safe] = dots[safe] / denom[safe]

    sorted_r = np.sort(r_per_channel)
    return {
        "max_r": float(r_per_channel.max()),
        "mean_r": float(r_per_channel.mean()),
        "median_r": float(np.median(r_per_channel)),
        "worst10_mean": float(sorted_r[:10].mean()),
        "r_per_channel": r_per_channel,
    }
