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

ResNet50 also fits locally with batch size 16. Its corrected extraction was
started with `--overwrite`; all later sessions must omit that flag:

```powershell
uv run scripts/07_extract_resnet_features.py --architecture resnet50 --batch-size 16 --max-batches 500
```

For subject 60, pass the corresponding input, output, and mask paths explicitly.

## Evaluate a completed visual representation

```powershell
uv run scripts/08_evaluate_visual_baseline.py challenge1/training/dino_features_vitb14.npy --valid-mask challenge1/training/dino_features_vitb14_valid_mask.npy --include-metadata
```

This runs leave-one-subject-out validation with within-subject MEG
normalization and reports normalized MSE and mean channel correlation.

## Evaluate fixation and incoming-saccade features

Run the ungrouped ablations first. Each of the 204 sensors selects its own
RidgeCV alpha using only the training side of each leave-one-subject-out fold.

```powershell
uv run scripts/09_evaluate_fixation_aware_ridge.py --feature-set gaze --output-csv results/fixation_aware_ridge/gaze_loso.csv
uv run scripts/09_evaluate_fixation_aware_ridge.py --feature-set saccade --output-csv results/fixation_aware_ridge/saccade_loso.csv
```

Only if incoming-saccade features improve the ungrouped model, test the idea
that repeated scene/location/sequence groups should count once during fitting:

```powershell
uv run scripts/09_evaluate_fixation_aware_ridge.py --feature-set saccade --prototype-bin-px 32 --output-csv results/fixation_aware_ridge/saccade_prototypes32_loso.csv
```

Prototype aggregation applies only to training rows. Every held-out fixation is
still predicted and scored, as required by the challenge.

## Extract spatial crop features and combine them with saccades

After the eye-feature ablations, extract a cheap spatial branch from the crop
itself. It retains grid-pooled RGB means/std and edge summaries, so it preserves
where visual structure occurs inside the fixation crop:

```powershell
uv run scripts/10_extract_spatial_features.py --batch-size 1024 --max-batches 500
```

Repeat until `spatial_features.npy` is finalized, then evaluate it with the
frozen ungrouped incoming-saccade branch and training-fold PCA:

```powershell
uv run scripts/11_evaluate_spatial_saccade_ridge.py `
  challenge1/training/spatial_features.npy `
  --valid-mask challenge1/training/spatial_features_valid_mask.npy `
  --pca-components 128 `
  --output-csv results/spatial_saccade_ridge/loso_results.csv
```

The equal-strength pilot is already complete. The one follow-up worth running
is a prespecified branch-weight test that keeps the saccade branch dominant:

```powershell
uv run scripts/11_evaluate_spatial_saccade_ridge.py `
  challenge1/training/spatial_features.npy `
  --valid-mask challenge1/training/spatial_features_valid_mask.npy `
  --pca-components 128 `
  --visual-scale 0.25 `
  --output-csv results/spatial_saccade_ridge/scale025_loso.csv
```

## Fit the frozen configuration for subject 60 development

The current frozen candidate is PCA-128 spatial features at scale `0.25` plus
the ungrouped incoming-saccade predictors. First extract development spatial
features:

```powershell
uv run scripts/10_extract_spatial_features.py `
  --input-file challenge1/subject60/challenge1_dev/crops_112.h5 `
  --output-file challenge1/subject60/challenge1_dev/spatial_features.npy `
  --valid-mask-output challenge1/subject60/challenge1_dev/spatial_features_valid_mask.npy `
  --batch-size 1024 --max-batches 500
```

Then fit on all five training subjects and save one prediction per development
row. This script does not inspect development ground truth:

```powershell
uv run scripts/12_predict_subject60_spatial_saccade.py
```

The same frozen script can produce the sealed evaluation prediction after its
spatial features are extracted, by changing only the metadata and feature paths:

```powershell
uv run scripts/10_extract_spatial_features.py `
  --input-file challenge1/subject60/challenge1_eval/crops_112.h5 `
  --output-file challenge1/subject60/challenge1_eval/spatial_features.npy `
  --valid-mask-output challenge1/subject60/challenge1_eval/spatial_features_valid_mask.npy `
  --batch-size 1024 --max-batches 500

uv run scripts/12_predict_subject60_spatial_saccade.py `
  --dev-metadata challenge1/subject60/challenge1_eval/metadata.csv `
  --dev-feature-file challenge1/subject60/challenge1_eval/spatial_features.npy `
  --dev-valid-mask challenge1/subject60/challenge1_eval/spatial_features_valid_mask.npy `
  --output-file results/subject60/spatial_saccade_scale025_eval_predictions.npy `
  --summary-file results/subject60/spatial_saccade_scale025_eval_summary.json
```

## Generate and validate a format-reference prediction

```powershell
uv run scripts/make_example_submission.py
uv run scripts/validate_submission.py results/example_submission_mean_baseline.npy challenge1/subject60/challenge1_dev/metadata.csv
```

The mean prediction is a format reference, not a competitive model.

## Plot a prediction topomap

Plot the saved 204-channel evaluation prediction using the subject-60 sensor
geometry. The default metric is predicted RMS activity:

```powershell
uv run scripts/13_plot_prediction_topomap.py
```

Alternative summaries are available with `--metric mean_abs` or
`--metric signed_mean`; pass `--output-file` to choose a different PNG path.
