# Submission Scripts

Submission scripts for SLURM jobs on the HPC go here.

There already are two example scripts for:

- [submitting a job to the cpu queue](/submit/submit_to_cpu_example.sh)
- [submitting a job to the gpu queue](/submit/submit_to_gpu_example.sh)

Project job:

- `extract_dino_features.sh` resumes the Challenge 1 DINOv2 training-feature
  extraction on an H100 GPU. Submit it from the repository root with
  `sbatch submit/extract_dino_features.sh` after confirming that the DINOv2
  Torch Hub repository and weights are cached in the HPC account.


# TODO

- point to correct conda env after we have created it
