# Challenge 1 living project report

Last updated: 2026-07-17

## 1. Objective

Predict the 204-channel gradiometer MEG topography at 110 ms after fixation
onset for subject 60. Each prediction row must correspond exactly to the same
row in the supplied subject 60 metadata, and the columns must follow
`challenge1/training/channel_names.txt`.

This document is the experiment record for the project. It should be updated
whenever an experiment changes our understanding, even when the result is
negative.

## 2. Data inventory

| Dataset | Rows | Shape or relevant detail |
|---|---:|---|
| Training metadata | 181,759 | Subjects 1–5 |
| Training MEG | 181,759 | `(181759, 204)`, `float32` |
| Training crops | 181,759 | `(181759, 112, 112, 3)` |
| Valid training crops | 175,425 | 96.52% |
| Invalid training crops | 6,334 | 3.48%; represented by zero crops/features |
| Subject 60 development | 7,750 | Metadata and local ground truth |
| Subject 60 evaluation | 7,909 | Metadata and local ground truth |
| Valid subject 60 development crops | 7,357 | 94.93%; 393 out of bounds |
| Valid subject 60 evaluation crops | 7,519 | 95.07%; 390 out of bounds |

The scene directory contains 4,081 images. Training and subject 60 reference
the same visual scene collection, so scene overlap is expected and is part of
the challenge design.

## 3. Repository audit and repairs

The initial audit found that most project work existed only as local,
uncommitted scripts and generated plots. The following problems were repaired:

- The first channel entry had accidentally become `dir .\scripts\MEG0001`.
  It was restored to `MEG0001` so that channel ordering remains valid.
- The introductory notebook had accidentally been changed to a SQL kernel and
  contained an unrelated connection error. Its Python kernel metadata and clean
  output state were restored.
- `.gitignore` contained duplicate dataset rules and a malformed `*.pyc` rule.
  Generated results are now ignored while `results/README.md` remains tracked.
- The example submission script referenced an obsolete
  `data/brainencoding26/...` directory. It now uses the repository's actual
  `challenge1/...` layout and accepts arbitrary target metadata/output paths.
- A submission validator was added to check shape, number of channels, numeric
  type, finite values, and duplicate channel names before submission.

## 4. Exploratory findings

### 4.1 Strong between-subject scale differences

Raw MEG standard deviations differ substantially:

| Subject | Global MEG standard deviation |
|---:|---:|
| 1 | `3.881e-10` |
| 2 | `5.963e-12` |
| 3 | `5.775e-12` |
| 4 | `1.861e-11` |
| 5 | `2.536e-10` |
| 60 development | `6.016e-12` |
| 60 evaluation | `6.308e-12` |

Subjects 1 and 5 therefore dominate pooled raw MSE. A model can look better or
worse mainly because of subject amplitude rather than because it predicts the
fixation-specific neural pattern.

Decision: model targets are normalized independently within each training
subject and channel during LOSO evaluation. Validation targets are normalized
only to express normalized MSE; Pearson correlation itself is invariant to
per-channel offset and positive scale.

### 4.2 Constant training mean is not useful for subject 60

On subject 60 development data:

- zero prediction MSE: `3.6196e-23`
- pooled training-channel-mean MSE: `3.6414e-23`
- relative change versus zero: `-0.60%`

The pooled training mean is therefore only a submission-format reference.

### 4.3 Metadata-only baseline

Leave-one-subject-out metadata Ridge results:

| Held-out subject | MSE improvement vs training mean | Mean channel correlation |
|---:|---:|---:|
| 1 | `+0.0001%` | `0.0035` |
| 2 | `-30.30%` | `0.0472` |
| 3 | `-30.94%` | `0.0393` |
| 4 | `-2.80%` | `0.0390` |
| 5 | `-0.0022%` | `0.0051` |

Mean channel correlation is approximately `0.0268`. The weak positive
correlation for subjects 2–4 shows that fixation behavior contains some signal,
but raw MSE is unstable because of between-subject scaling.

### 4.4 Scene-mean baseline

Training on subjects 1–4 and validating on subject 5 covered every validation
row's scene, but produced:

- relative MSE improvement: `-3.257%`
- mean channel correlation: `-0.000036`

Lesson: simply memorizing the mean response associated with a scene does not
transfer across subjects. The useful signal likely depends on local fixation
content and/or needs subject-normalized representation learning.

## 5. Feature-extraction incident and recovery

The original extractors preallocated complete arrays but did not support
resume. Both prior GPU runs were interrupted:

| Representation | Completed valid rows | Progress |
|---|---:|---:|
| DINOv2 ViT-B/14 | 41,248 / 175,425 | 23.51% |
| ResNet50 | 175,425 / 175,425 | 100%; finalized and verified |
| ResNet18 | 13,120 / 175,425 | 7.48% and resumable |

The final validity-mask files were never written, and a separate
`dino_features.npy` file is empty/corrupt. The large `.tmp.npy` files are
preallocated, so file size alone does not indicate completion.

Recovery implemented:

- Both extractors resume `.tmp.npy` files by default.
- Completed legacy rows are inferred from non-zero embeddings. Exact all-zero
  embeddings are not expected from either network; invalid/unprocessed rows are
  deliberately zero.
- Arrays are flushed periodically so a crash loses at most a small number of
  batches.
- Shape and representation-width mismatches stop with an explicit instruction
  to restart using `--overwrite`.
- The partial array is renamed to the final output only after every valid crop
  is processed; the validity mask is then saved alongside it.

Challenge 1 is intended to run locally, and this repository is not connected to
an SSH/HPC environment. The available GPU is an NVIDIA GeForce GTX 1650 Max-Q
with 4 GB memory. Conservative batch sizes and bounded, resumable runs are used
so extraction can progress locally without requiring a long uninterrupted job.

The environment audit initially found `torch 2.11.0+cpu`: it had no CUDA
runtime and `torch.cuda.is_available()` was false even though Windows saw the
NVIDIA GPU. The first ResNet18 benchmark therefore ran on six CPU threads and
completed only 128 crops before the three-minute command limit.

The environment now pins the [official PyTorch CUDA 12.1 build](https://docs.pytorch.org/get-started/previous-versions/):
`torch 2.5.1+cu121` and `torchvision 0.20.1+cu121`. CUDA detection and a GPU
matrix multiplication both pass on the GTX 1650. Per-image PIL preprocessing
was then replaced with batched tensor resize, crop, and normalization on the
GPU.

With the corrected pipeline, an initial 320-crop run completed in 14 seconds.
A sustained 6,400-crop session completed in 147 seconds including startup and
partial-file scanning, about 43.5 crops/second. At that measured rate, the
remaining ResNet18 training extraction is approximately 65 minutes and can be
completed in bounded, resumable local sessions.

The first user-run local session then resumed correctly from row 6,720 and
processed another 6,400 valid crops without errors, reaching 13,120/175,425.
This confirms that CUDA selection, partial-file scanning, and cross-session row
alignment all work from the normal activated PowerShell environment. Future
sessions can safely use 500 batches (32,000 crops) to reduce repeated startup
overhead while remaining bounded and resumable.

ResNet50 was then restarted intentionally with the corrected batched
preprocessing, using batch size 16 on the 4 GB GPU. It completed 100 batches
(1,600 crops) without an out-of-memory error. The resulting feature width is
2,048, producing an approximately 1.49 GB training feature matrix. Subsequent
ResNet50 sessions must omit `--overwrite`; bounded 500-batch sessions will add
8,000 crops at a time without increasing peak GPU memory.

Two subsequent 500-batch user sessions resumed from 1,600 and 9,600 rows,
respectively. Both completed without CUDA, memory, or alignment errors and
advanced ResNet50 to 17,600/175,425 valid crops (10.03%). An independent scan of
the 2,048-dimensional temporary array confirmed exactly 17,600 non-zero valid
rows, so the terminal progress and on-disk state agree.

A subsequent 1,000-batch session added 16,000 crops and advanced ResNet50 to
33,600/175,425 valid crops (19.15%). An independent on-disk scan again matched
the terminal count exactly, confirming that the longer bounded session did not
lose or misalign rows.

The next 1,000-batch session advanced ResNet50 to 49,600/175,425 valid crops
(28.27%). A full scan of the temporary `(181759, 2048)` matrix confirmed the
same count, taking the corrected ResNet50 extraction past the quarter mark.

On 2026-07-18, ResNet50 extraction reached 175,425/175,425 valid crops and the
temporary array was finalized as `resnet50_features.npy`. The final integrity
check confirmed shape `(181759, 2048)`, `float32` values, no NaN or infinite
values, non-zero embeddings for every valid crop, zero embeddings for all 6,334
invalid crops, and a matching validity mask. ResNet50 training-feature
extraction is complete.

Subject 60 development extraction was then completed with the identical
ResNet50 preprocessing. The final matrix has shape `(7750, 2048)` and aligns
with all 7,750 metadata rows. Integrity checks confirmed 7,357 finite non-zero
valid embeddings, 393 correctly zero-filled invalid crops, and a matching
validity mask.

The first full-width ResNet50 Ridge memory test held out subject 1 with
`alpha=100` and visual features only. It completed locally on 147,917 training
rows and 33,842 validation rows, confirming that the 2,048-dimensional design
is computationally feasible. Performance was effectively null: normalized MSE
`1.017463`, `-1.752%` improvement over zero, and mean channel correlation
`0.002551`. Subject 1 was also the weakest metadata-only fold, so representation
selection remains deferred until all five LOSO folds are evaluated.

The remaining four folds completed with the following visual-only results:

| Held-out subject | Normalized MSE | Improvement vs zero | Mean channel correlation |
|---:|---:|---:|---:|
| 1 | `1.017463` | `-1.752%` | `0.002551` |
| 2 | `1.011824` | `-1.182%` | `0.017474` |
| 3 | `1.014005` | `-1.400%` | `0.011558` |
| 4 | `1.012990` | `-1.299%` | `0.012076` |
| 5 | `1.019326` | `-1.933%` | `0.000773` |

Across folds, mean normalized MSE is `1.015122`, mean improvement is
`-1.513%`, and mean channel correlation is `0.008886`. The positive but weak
correlation suggests limited visual signal, while MSE above one in every fold
shows that `alpha=100` permits predictions with harmful variance. This model is
not yet the full course recipe: it uses a spatially pooled final ResNet layer,
no PCA, and one shared alpha for every sensor. Its failure rules out continuing
with that exact configuration; it does not rule out linear encoding models.

The same five ResNet50-only folds were then evaluated with stronger shared
regularization, `alpha=10000`:

| Held-out subject | Normalized MSE | Improvement vs zero | Mean channel correlation |
|---:|---:|---:|---:|
| 1 | `1.013436` | `-1.349%` | `0.002717` |
| 2 | `1.008096` | `-0.810%` | `0.019192` |
| 3 | `1.009898` | `-0.990%` | `0.013051` |
| 4 | `1.009309` | `-0.931%` | `0.013356` |
| 5 | `1.014678` | `-1.468%` | `0.000913` |

Across folds, mean normalized MSE improved from `1.015122` to `1.011083`,
mean degradation versus the zero predictor decreased from `-1.513%` to
`-1.110%`, and mean sensor correlation improved from `0.008886` to `0.009846`.
Every fold moved in the correct direction, confirming that `alpha=100` was too
weak. However, all five MSE values remain worse than the zero predictor and the
correlation gain is only `0.000960`. More tuning of this spatially pooled final
ResNet layer is therefore lower priority than the fixation-aware ablation.

On 2026-07-19, the repository's leaderboard evaluator was inspected directly.
The official primary statistic is Pearson correlation across fixations for each
of the 204 sensors, followed by the mean across sensors (`mean_r`). It also
reports median, maximum, and worst-ten-sensor correlation. Our correlation axis
matches that definition.

The metadata-only LOSO baseline averages `0.026820` mean sensor correlation,
compared with `0.009846` for the best completed ResNet50-only configuration.
This is the strongest current evidence that fixation location and eye-movement
state deserve a controlled ablation before another expensive image encoder is
extracted.

A new experiment, `scripts/09_evaluate_fixation_aware_ridge.py`, now implements
that ablation. It derives gaze-relative position, the displacement, distance,
direction and timing of the incoming saccade, and a first-fixation indicator.
It normalizes MEG within each training subject, imputes and scales predictors
using the training fold only, and uses `RidgeCV(alpha_per_target=True)` so each
sensor selects its own regularization. An optional prototype mode averages
training rows sharing scene, nearby gaze bins and nearby fixation-sequence bins.
This makes a repeated group count once during fitting but still predicts and
scores every held-out fixation. No result is claimed until all five LOSO folds
finish.

The ungrouped gaze-only ablation then completed all five LOSO folds. Here,
"gaze-only" includes gaze position, fixation duration, time and sequence within
the trial, and fixation stability (`rms` and `sd`), but no incoming-saccade
features:

| Held-out subject | Normalized MSE | Mean sensor correlation | Median sensor correlation |
|---:|---:|---:|---:|
| 1 | `1.004456` | `0.005106` | `0.005379` |
| 2 | `0.992312` | `0.098344` | `0.102410` |
| 3 | `0.993427` | `0.082101` | `0.081838` |
| 4 | `0.994330` | `0.071751` | `0.059412` |
| 5 | `1.006552` | `0.006287` | `0.006496` |

Mean LOSO correlation is `0.052718` and mean normalized MSE is `0.998215`, the
first tested configuration to improve on the zero-prediction MSE baseline on
average. It also exceeds the earlier metadata baseline (`0.026820`) and the
best ResNet50-only result (`0.009846`). The gain over the old metadata model is
not a pure feature comparison because the new experiment also adds per-sensor
RidgeCV and normalized targets. The upcoming saccade-versus-gaze comparison is
controlled: it uses the same evaluator and changes only the added incoming-eye-
movement predictors.

Subjects 2–4 carry nearly all of the cross-subject signal, while subjects 1 and
5 remain close to zero and slightly worse than the zero predictor in MSE. The
worst-ten-sensor correlation is still negative in every fold, so the result is
a real milestone rather than a finished model. Selected alphas span roughly
`31.6` to `316228` across sensors and folds, directly confirming that one shared
regularization strength was an important weakness of the earlier baselines.

The controlled incoming-saccade ablation then completed with the same LOSO
folds, target normalization, alpha grid and per-sensor RidgeCV. It adds previous
gaze position, incoming displacement, distance, direction, timing, amplitude
and a first-fixation indicator:

| Held-out subject | Normalized MSE | Mean sensor correlation | Median sensor correlation |
|---:|---:|---:|---:|
| 1 | `1.008335` | `0.005504` | `0.004697` |
| 2 | `0.990267` | `0.107647` | `0.109737` |
| 3 | `0.992695` | `0.087980` | `0.084671` |
| 4 | `0.992379` | `0.085584` | `0.070800` |
| 5 | `1.009058` | `0.006502` | `0.005115` |

Mean LOSO correlation increased from `0.052718` to `0.058644`, an absolute gain
of `0.005926` and a relative gain of `11.24%`. Correlation improved in every
fold, with the useful gains again concentrated in subjects 2–4. The average
worst-ten-sensor correlation improved substantially from `-0.017562` to
`-0.002981`; subjects 3 and 4 became positive even on this difficult summary.
This supports the hypothesis that the eye movement landing on a fixation
contains information beyond the fixation location itself.

Mean normalized MSE moved slightly backward from `0.998215` to `0.998547`.
Subjects 1 and 5 gained only tiny correlation while acquiring extra prediction
variance, whereas subjects 2–4 improved on both metrics. Because the official
metric is mean sensor correlation, the saccade model is the current winner, but
the subject-specific MSE tradeoff must be watched. The next experiment tests
32-pixel (approximately one-degree) same-scene fixation prototypes on training
rows only; it does not collapse or omit any validation prediction.

The prespecified 32-pixel prototype test reduced 727,036 total training-fold
rows to 683,916 model rows across the five folds, a reduction of only `5.93%`.
Most groups were singletons (mean prototype size `1.06`–`1.07`), with maximum
group sizes of five or six. Mean correlation decreased slightly from `0.058644`
to `0.058393` (`-0.000250`), and mean normalized MSE moved from `0.998547` to
`0.998558`. Correlation was lower in every fold, although the differences for
subjects 1 and 5 were tiny.

The conclusion is specific: hard averaging of exact-scene, nearby-gaze and
nearby-sequence rows is not useful at this resolution. It does not disprove the
broader idea that semantically similar images or fixation patterns share signal;
that should be represented through visual/spatial features or soft weighting,
not by discarding distinctions through this hard prototype rule. No bin-size
sweep will be performed now, avoiding post-hoc tuning on the same LOSO folds.

As a final metadata boundary check, the `full` feature set added outgoing
saccade fields, `fix_sequence_from_last`, and `caption_task` to the winning
incoming-saccade features. It did not help:

| Configuration | Mean normalized MSE | Mean sensor correlation |
|---|---:|---:|
| Incoming saccade, ungrouped | `0.998547` | `0.058644` |
| Full supplied metadata | `0.998603` | `0.058399` |

The full model's mean correlation fell by `0.000245` and its MSE worsened by
`0.000056`. Subjects 2 and 3 lost correlation, while the small gains in
subjects 1 and 5 were not enough to compensate. We therefore freeze the
ungrouped incoming-saccade feature set as the eye-movement branch. This also
avoids depending on outgoing/future context when we later interpret the model
scientifically.

The first spatial visual-plus-saccade pilot also completed. It used 577 raw
grid features, 128 training-fold PCA components (about `90.9%` explained
variance), and the same per-sensor RidgeCV:

| Configuration | Mean normalized MSE | Mean sensor correlation |
|---|---:|---:|
| Incoming saccade, ungrouped | `0.998547` | `0.058644` |
| Spatial PCA (128) + incoming saccade | `0.999533` | `0.054180` |

The equal-strength concatenation reduced correlation by `0.004464` and
worsened MSE by `0.000986`. This does not prove that spatial appearance is
useless: the visual branch has 128 PCs versus 19 eye predictors, so it can
dominate the shared Ridge penalty. The evaluator now exposes a visual-branch
scale so we can test a single predeclared down-weighted fusion (`0.25`) before
deciding whether the visual branch adds robust signal.

On 2026-07-17, a one-minute local recovery check successfully added 32 DINO
rows. Most of the run was model startup, and inference warned that xFormers was
unavailable. This proved recovery correctness but showed that ViT-B/14 is not
the best first local baseline. The corrected plan is to finish a lighter
ResNet18 representation locally first, evaluate it, and return to DINO only if
its expected benefit justifies the longer runtime.

## 6. Evaluation protocol

The current protocol is:

1. Hold out one entire training subject.
2. Normalize every training subject independently per MEG channel using only
   that subject's training fold values.
3. Fit image-feature Ridge, optionally adding robustly scaled fixation metadata
   and an explicit valid-crop indicator.
4. Evaluate normalized MSE and mean Pearson correlation across 204 channels.
5. Repeat for all five subjects and retain fold-level results.
6. Choose representation, metadata use, and regularization before evaluating a
   single frozen model on subject 60 development.
7. Do not inspect subject 60 evaluation ground truth during model selection.

Why LOSO: the actual challenge is cross-subject generalization. Random
fixation-level splits would mix subjects and would substantially overstate
generalization performance.

Why two metrics: mean sensor correlation is the confirmed official statistic.
Normalized MSE remains a diagnostic that catches badly calibrated or needlessly
variable predictions even when their correlation is slightly positive.

## 7. Planned model ladder

Run the smallest defensible comparison before adding complexity:

1. zero and pooled-mean format baselines;
2. existing metadata-only and ResNet50-only LOSO baselines;
3. gaze-only versus incoming-saccade RidgeCV, with per-sensor alpha selection;
4. repeat the best eye-feature model with training-only fixation prototypes;
5. reproduce the course-style easy within-subject split only as a diagnostic,
   never as the challenge estimate;
6. add variance filtering and PCA to spatial Gabor or intermediate DNN features;
7. combine the winning visual representation with the winning eye features;
8. optional target PCA or reduced-rank regression if compute/memory becomes the
   limiting factor;
9. only then consider full-scene context, multi-crop features, or neural models.

Each experiment must record feature version, crop size, validity policy,
normalization, alpha, fold metrics, runtime, and output path.

## 8. Immediate execution checklist

- [x] Repair channel names, notebook metadata, ignore rules, and example paths.
- [x] Make DINO and ResNet extraction resumable.
- [x] Add submission validation.
- [x] Add normalized LOSO visual-baseline evaluation.
- [x] Prepare subject 60 development and evaluation crop files.
- [x] Install and verify a CUDA-enabled local PyTorch environment.
- [x] Replace per-image preprocessing with batched GPU preprocessing.
- [x] Finish and verify the local ResNet50 training representation; retain the
      partial ResNet18 and DINO files for later comparison if needed.
- [x] Extract and verify ResNet50 features for subject 60 development.
- [x] Run five-fold ResNet50 visual-only LOSO at `alpha=100`.
- [x] Run five-fold ResNet50 visual-only LOSO at `alpha=10000` and consolidate
      the interrupted run into one result file.
- [x] Confirm the official mean-sensor-correlation scoring implementation.
- [x] Implement gaze/incoming-saccade ablations, per-sensor RidgeCV, and optional
      training-only fixation prototypes.
- [x] Run five-fold ungrouped gaze-only LOSO with per-sensor RidgeCV.
- [x] Run five-fold ungrouped incoming-saccade LOSO with the identical evaluator.
- [x] Test the prespecified 32-pixel prototype aggregation after the ungrouped
      ablations; reject it because it slightly reduced every fold's correlation.
- [x] Test the full supplied-metadata feature set; reject it because it did not
      improve the incoming-saccade branch.
- [x] Implement the spatial crop feature extractor and PCA+saccade evaluator.
- [x] Extract spatial crop features locally.
- [x] Run spatial visual plus incoming-saccade LOSO at equal branch scale; retain
      the eye-only model as the current winner.
- [x] Run one down-weighted spatial fusion at visual scale `0.25`; it is the
      current best official-correlation configuration (`mean_r=0.059205`).
- [x] Implement frozen subject-60 prediction using the selected configuration.
- [x] Extract subject-60 development spatial features.
- [x] Fit the frozen configuration and generate subject-60 development output.
- [x] Extract subject-60 evaluation spatial features.
- [x] Fit the frozen configuration and generate the sealed evaluation output.
- [x] Generate and validate the final evaluation prediction without using the
      evaluation ground truth for tuning.

## 9. Frozen final output

The selected configuration is PCA-128 spatial crop features scaled by `0.25`,
concatenated with the ungrouped incoming-saccade feature set, an explicit visual
validity indicator, subject-wise target normalization, and per-sensor RidgeCV.
Its five-fold LOSO result is mean sensor correlation `0.059205` and mean
normalized MSE `0.998702`.

The sealed subject-60 evaluation prediction is saved at
`results/subject60/spatial_saccade_scale025_eval_predictions.npy`. It has shape
`(7909, 204)`, finite `float32` values, and passed the repository's submission
validator. No subject-60 evaluation ground truth was used during model
selection or prediction generation. The file is ready for the challenge's
submission mechanism; an external leaderboard score is not available locally.

## 10. Challenge 2 development result

Challenge 2 uses the shared HPC dataset rather than copying the multi-gigabyte
target into Git. The training target is `meg_c2.npy` with shape
`(181759, 204, 61)` and `float32` values. The supplied time axis contains
61 samples from `-50` to `250` ms. The official six-timepoint order is
`[-50, 50, 75, 100, 125, 150]` ms; the available samples nearest to 75 and
125 ms are 74 and 124 ms, respectively (indices `[0, 20, 25, 30, 35, 40]`).

The Challenge 2 training crops contain 181,759 rows, of which 175,425 are
valid. The spatial extractor produced a `(181759, 577)` feature matrix. The
development metadata contains 6,354 fixations, with 6,076 valid visual crops.

The frozen best model is the spatial+saccade configuration with PCA-128 spatial
features at visual scale `0.05`, incoming-saccade metadata, an explicit
visual-validity indicator, subject-wise target normalization, and per-target
RidgeCV. A scale sweep (`0.25`, `0.1`, `0.05`, `0.025`, and `0`) selected
`0.05` by the official mean sensor correlation. Five-subject LOSO evaluation
across all six timepoints produced:

| Requested time | Stored sample | Mean r | Normalized MSE |
|---:|---:|---:|---:|
| -50 ms | -50 ms | 0.046260 | 0.998739 |
| 50 ms | 50 ms | 0.051325 | 0.998552 |
| 75 ms | 74 ms | 0.050536 | 0.998677 |
| 100 ms | 100 ms | 0.059925 | 0.998462 |
| 125 ms | 124 ms | 0.055872 | 0.998700 |
| 150 ms | 150 ms | 0.049505 | 0.998709 |

Overall mean LOSO correlation is `0.052237` and mean normalized MSE is
`0.998640`. A reduced-rank target-PCA experiment was also tested, but its mean
correlation (`0.049941`) was lower, so it was not selected. The subject-60
development prediction has shape `(6354, 204, 6)`, finite `float32` values, and
passed the Challenge 2 validator.

The prediction and summary are generated on HPC at
`results/challenge2/subject60_predictions_scale005.npy` and
`results/challenge2/subject60_summary_scale005.json`. A six-panel mean-absolute
topomap is available at
`results/challenge2/subject60_topomaps_scale005.png`. The supplied data package
contains no separate `challenge2_eval` or `challenge2_final` metadata; therefore
the validated `challenge2_dev` prediction is the final available Challenge 2
output for this project.

## 11. Open questions and risks

- Local subject 60 evaluation ground truth creates a leakage risk. It should be
  treated as sealed data even though the file is accessible.
- Invalid crops are currently zero-feature rows plus a validity indicator. If
  they are common in subject 60, a full-scene or metadata fallback may help.
- DINOv2 requires the pretrained repository and weights to remain available in
  the local Torch Hub cache.
- The CUDA-enabled PyTorch build is deliberately pinned to 2.5.1/cu121 for the
  installed driver. Future dependency upgrades must verify CUDA detection
  instead of assuming the default PyPI wheel uses the GPU.
- Dense Ridge on 181,759 × 768/2,048 features may be memory intensive. DINO is
  the better first experiment; target PCA or incremental solvers are fallback
  options if resource use is excessive.

## 12. Experiment log template

Copy this block for every new run:

```text
Date:
Question:
Representation and input:
Training/validation split:
Target normalization:
Model and hyperparameters:
Runtime/hardware:
Fold results:
Subject 60 development result (only after selection):
Interpretation:
Decision and next experiment:
Artifacts:
```
