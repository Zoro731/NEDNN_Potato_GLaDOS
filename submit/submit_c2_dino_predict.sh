#!/bin/bash
#SBATCH -p workq
#SBATCH -t 0-08:00:00
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH -o c2_dino_predict-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PATH="/home/student/i/isingh/.local/bin:${PATH}"

uv run scripts/15_predict_challenge2_spatial_saccade.py \
  --train-features challenge2/training/dino_features_vitb14.npy \
  --train-valid-mask challenge2/training/dino_features_vitb14_valid_mask.npy \
  --dev-features challenge2/subject60/challenge2_dev/dino_features_vitb14.npy \
  --dev-valid-mask challenge2/subject60/challenge2_dev/dino_features_vitb14_valid_mask.npy \
  --output results/challenge2/dino_subject60_predictions.npy \
  --summary results/challenge2/dino_subject60_summary.json \
  --pca-components 128 \
  --visual-scale 0.05

uv run scripts/17_plot_challenge2_topomaps.py \
  results/challenge2/dino_subject60_predictions.npy \
  --output-file results/challenge2/dino_subject60_topomaps.png
