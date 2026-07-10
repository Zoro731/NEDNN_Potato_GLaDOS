"""Generate a mean-baseline submission for Challenge 1.

The baseline predicts the per-channel training-set mean for every dev fixation.
Saved to results/example_submission_mean_baseline.npy  (shape 7750 × 204).
"""

from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
TRAIN_MEG = REPO / "data/brainencoding26/challenge1/training/meg_110ms.npy"
DEV_META = REPO / "data/brainencoding26/challenge1/subject60/challenge1_dev/metadata.csv"
OUT_PATH = REPO / "results/example_submission_mean_baseline.npy"

train = np.load(TRAIN_MEG)          # (n_train_fixations, 204)
channel_mean = train.mean(axis=0)   # (204,)

import pandas as pd
dev_meta = pd.read_csv(DEV_META)
n_dev = len(dev_meta)               # 7750

pred = np.tile(channel_mean, (n_dev, 1))   # (7750, 204)

OUT_PATH.parent.mkdir(exist_ok=True)
np.save(OUT_PATH, pred)
print(f"saved: {OUT_PATH}  shape={pred.shape}  dtype={pred.dtype}")
# Note: constant predictions have zero variance → Pearson r = 0 on all channels.
# This file is provided as a format reference, not as a competitive baseline.
