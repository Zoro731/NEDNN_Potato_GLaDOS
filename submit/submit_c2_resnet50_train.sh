#!/bin/bash
#SBATCH -p klab-gpu
#SBATCH -t 0-08:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:H100.10gb
#SBATCH -o c2_r50_train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
export PYTHONUNBUFFERED=1
export PATH="/home/student/i/isingh/.local/bin:${PATH}"

uv run scripts/07_extract_resnet_features.py \
  --input-file challenge2/training/crops_112.h5 \
  --output-file challenge2/training/resnet50_features.npy \
  --valid-mask-output challenge2/training/resnet50_features_valid_mask.npy \
  --architecture resnet50 \
  --batch-size 16 \
  --device cuda
