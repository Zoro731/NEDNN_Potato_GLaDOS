# Project scripts

Scripts are numbered in the approximate order of the workflow. Run commands
from the repository root using `uv run`, or with the project virtual
environment.

## Prepare subject 60 crops

```powershell
uv run scripts/prepare_challenge_crops.py --metadata-csv challenge1/subject60/challenge1_dev/metadata.csv --scenes-dir avs_scenes/avs_scenes --output-file challenge1/subject60/challenge1_dev/crops_112.h5
uv run scripts/prepare_challenge_crops.py --metadata-csv challenge1/subject60/challenge1_eval/metadata.csv --scenes-dir avs_scenes/avs_scenes --output-file challenge1/subject60/challenge1_eval/crops_112.h5
```

## Extract image features

Both extractors resume compatible `.tmp.npy` files by default. Use
`--overwrite` only when intentionally restarting an incompatible run.

```powershell
uv run scripts/06_extract_dino_features.py
uv run scripts/07_extract_resnet_features.py
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
