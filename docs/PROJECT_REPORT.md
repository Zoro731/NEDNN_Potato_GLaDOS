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
shows that `alpha=100` permits predictions with harmful variance. The next
experiment must test substantially stronger regularization before visual
features are combined with fixation metadata.

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

Why two metrics: the official metric is not yet recorded in the repository.
Correlation measures fixation-specific pattern prediction without being
dominated by amplitude scale; normalized MSE catches badly calibrated or noisy
predictions.

## 7. Planned model ladder

Run the smallest defensible comparison before adding complexity:

1. zero and pooled-mean format baselines;
2. metadata-only Ridge;
3. ResNet18 fixation-crop features as the first practical local baseline;
4. DINOv2 fixation-crop features;
5. best visual representation plus fixation metadata;
6. regularization sweep on the best representation;
7. optional target PCA or reduced-rank regression if compute/memory becomes the
   limiting factor;
8. only then consider full-scene context, multi-crop features, or neural models.

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
- [ ] Run five-fold visual-plus-metadata LOSO.
- [ ] Select alpha using training subjects only.
- [ ] Freeze the configuration and evaluate once on subject 60 development.
- [ ] Generate and validate the final evaluation prediction without using the
      evaluation ground truth for tuning.

## 9. Open questions and risks

- The official scoring metric must be confirmed. This affects whether raw scale
  calibration matters or correlation is sufficient.
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

## 10. Experiment log template

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
