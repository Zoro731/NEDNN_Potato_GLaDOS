#!/bin/bash
#SBATCH -p workq
#SBATCH -t 0-08:00:00
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH -o c2_r50_eval-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PATH="/home/student/i/isingh/.local/bin:${PATH}"

uv run scripts/14_evaluate_challenge2_spatial_saccade.py \
  --feature-file challenge2/training/resnet50_features.npy \
  --valid-mask challenge2/training/resnet50_features_valid_mask.npy \
  --pca-components 128 \
  --visual-scale 0.05 \
  --output-csv results/challenge2/resnet50_loso.csv
