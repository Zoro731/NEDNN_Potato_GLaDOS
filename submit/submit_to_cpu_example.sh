#!/bin/bash
#SBATCH -p workq
#SBATCH -t 0-00:30:00
#SBATCH --mem=64G
#SBATCH -c 8

# --- load git and gpu dependencies (you can remove these for cpu jobs)
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8

echo "Setup complete, running hello-world script ..."
uv run scripts/hello-world.py
echo "job completed!"
