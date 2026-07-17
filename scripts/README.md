# Project scripts

Scripts are numbered in the approximate order of the workflow. Run commands
from the repository root using `uv run`, or with the project virtual
environment.

## Local GPU environment

The project pins PyTorch 2.5.1 with the official CUDA 12.1 wheels because the
local NVIDIA driver supports CUDA 12.3 while the default PyPI resolution had
installed a CPU-only build. Recreate the environment with:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The expected result includes `2.5.1+cu121`, `True`, and the GTX 1650 device.

## Prepare subject 60 crops

```powershell
uv run scripts/prepare_challenge_crops.py --metadata-csv challenge1/subject60/challenge1_dev/metadata.csv --scenes-dir avs_scenes/avs_scenes --output-file challenge1/subject60/challenge1_dev/crops_112.h5
uv run scripts/prepare_challenge_crops.py --metadata-csv challenge1/subject60/challenge1_eval/metadata.csv --scenes-dir avs_scenes/avs_scenes --output-file challenge1/subject60/challenge1_eval/crops_112.h5
```

## Extract image features

Both extractors resume compatible `.tmp.npy` files by default. Challenge 1 is
run locally. Use `--max-batches` to split extraction into bounded sessions and
`--overwrite` only when intentionally restarting an incompatible run.

```powershell
uv run scripts/07_extract_resnet_features.py --architecture resnet18 --batch-size 64 --max-batches 500
```

Rerun the same command until it reports a final `resnet18_features.npy` and
validity mask. ResNet18 is the first local baseline. The existing partial DINO
ViT-B/14 extraction can also be continued in bounded sessions:

```powershell
uv run scripts/06_extract_dino_features.py --batch-size 16 --max-batches 25
```

For subject 60, pass the corresponding input, output, and mask paths explicitly.

## Evaluate a completed visual representation

```powershell
uv run scripts/08_evaluate_visual_baseline.py challenge1/training/dino_features_vitb14.npy --valid-mask challenge1/training/dino_features_vitb14_valid_mask.npy --include-metadata
```

This runs leave-one-subject-out validation with within-subject MEG
normalization and reports normalized MSE and mean channel correlation.

## Generate and validate a format-reference prediction

```powershell
uv run scripts/make_example_submission.py
uv run scripts/validate_submission.py results/example_submission_mean_baseline.npy challenge1/subject60/challenge1_dev/metadata.csv
```

The mean prediction is a format reference, not a competitive model.
