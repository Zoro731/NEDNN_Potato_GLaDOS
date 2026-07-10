#!/bin/bash
#SBATCH -p klab-gpu
#SBATCH -t 0-00:30:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:H100.10gb

spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8

# make sure output shows up in logs immediately
export PYTHONUNBUFFERED=1

uv run scripts/hello-gpu.py
