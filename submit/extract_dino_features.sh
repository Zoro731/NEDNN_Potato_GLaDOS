#!/bin/bash
#SBATCH -p klab-gpu
#SBATCH -t 0-06:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:H100.10gb
#SBATCH --job-name=meg-dino
#SBATCH --output=results/slurm-dino-%j.log

set -euo pipefail

spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8

export PYTHONUNBUFFERED=1

mkdir -p results

uv run scripts/06_extract_dino_features.py \
  --input-file challenge1/training/crops_112.h5 \
  --output-file challenge1/training/dino_features_vitb14.npy \
  --valid-mask-output challenge1/training/dino_features_vitb14_valid_mask.npy \
  --batch-size 64 \
  --log-every 20
