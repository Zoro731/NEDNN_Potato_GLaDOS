#!/bin/bash
#SBATCH -p klab-gpu
#SBATCH -t 0-24:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:H100.10gb
#SBATCH -o c2_dino_train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
export PYTHONUNBUFFERED=1
export PATH="/home/student/i/isingh/.local/bin:${PATH}"

uv run scripts/06_extract_dino_features.py \
  --input-file challenge2/training/crops_112.h5 \
  --output-file challenge2/training/dino_features_vitb14.npy \
  --valid-mask-output challenge2/training/dino_features_vitb14_valid_mask.npy \
  --model-name dinov2_vitb14 \
  --repo-dir /home/student/i/isingh/.cache/torch/hub/facebookresearch_dinov2_main \
  --batch-size 64 \
  --device cuda
