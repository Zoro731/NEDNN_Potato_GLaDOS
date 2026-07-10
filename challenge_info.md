# Active Vision Brain Encoding Challenge

## Overview

The human brain processes visual scenes through sequences of rapid, targeted fixations — each one triggering a cascade of neural responses that build our scene understanding over time. In this challenge, you will build **brain encoding models** that predict fixation-aligned MEG activity from image content and eye movement behavior. Using training data from 5 subjects in the Active Visiual Semantics (AVS) dataset, your models must generalise to a held-out participant ("subject 60") — predicting the neural response to each fixation solely from what was seen and where the eyes landed.

The challenge is structured in two stages of increasing difficulty.

**Dataset reference:** Sulewski, P., Amme, C., Hebart, M., König, P., & Kietzmann, T. (2025). Why we linger: Memory encoding, rather than visual processing demand, drives fixation timing on natural scenes — evidence from a large-scale MEG dataset.

---

## The AVS Dataset

The Active Vision on Scenes (AVS) dataset comprises simultaneous MEG and eye tracking recordings from 5 participants who freely explored 4,080 natural scenes (sourced from the NSD stimulus set; Allen et al., 2022) across 10 sessions each. Each scene was presented for 4 seconds during which participants made self-paced eye movements. On 25% of trials, participants verbally described the preceding scene.

**MEG acquisition:** Data were recorded with a 306-channel Elekta Neuromag TRIUX system (102 magnetometers, 204 planar gradiometers) at 1000 Hz, preprocessed with tSSS movement compensation, bandpass filtered (0.2–200 Hz), resampled to 500 Hz, and ICA-corrected for ocular artifacts. Epochs are time-locked to fixation onsets. For this challenge, only the 204 gradiometer channels are relevant.

**Scale:** The dataset contains over 200,000 fixation epochs in total (~40,000 per subject), each associated with the image content at the gaze location, fixation metadata, and simultaneous MEG recordings. Stimuli were sampled to ensure uniform semantic coverage across 60 content clusters.

Previous analyses have shown that single fixation epochs carry content-specific information in MEG sensor patterns, with encoding accuracy peaking at ~110 ms post fixation onset at posterior sensors (see Sulewski et al., 2025 for details).

---

## Challenge Structure

### Challenge 1 — Single-Timepoint Topography Prediction

**Goal:** Predict the MEG gradiometer topography at 110 ms post fixation onset for subject 60.

**What you predict:** For each fixation, a vector of gradiometer channel amplitudes at a single timepoint (110 ms).

**Submission format:** A NumPy file (`.npy`) with shape `(n_fixations, n_channels)`, where rows correspond to fixations in the order given by the metadata and columns correspond to gradiometer channels in the order provided in the training data.

**Phases:**

| Phase | Data | Purpose |
|-------|------|---------|
| Development | Metadata for 25% of subject 60's scenes | Build and iterate on your model. Submit predictions repeatedly for implementation feedback. |
| Final evaluation | Metadata for a different 25% of subject 60's scenes | Released shortly before the deadline. Generate final predictions on this set. |

---

### Challenge 2 — Multi-Timepoint Prediction

**Goal:** Extend your encoding model to predict MEG topographies at multiple timepoints between fixation onset: −50 to 250 ms.

**Submission format:** A NumPy file (`.npy`) with shape `(n_fixations, n_channels, n_timepoints)`, where the timepoint dimension follows the order listed above.

**Phases:**

| Phase | Data | Purpose |
|-------|------|---------|
| Development | Metadata for 25% of subject 60's scenes (disjoint from Challenge 1) | Build and iterate. |
| Final evaluation | Metadata for the remaining 25% of subject 60's scenes | Generate final predictions. |

> **Note:** The four 25% scene splits across both challenges are mutually exclusive and together cover all scenes for subject 60.

---

## Provided Materials

### Training data (subjects 1–5)

- **MEG data:** Fixation-aligned MEG epochs for all gradiometer channels across all timepoints listed above.
- **Stimulus images:** Original full scene images and pre-extracted fixation patches (112 × 112 px square crops centered on fixation location).
- **Fixation metadata:** Tables containing per-fixation information, including (but not limited to):
  - Gaze position (x, y)
  - Time in trial
  - Fixation sequence position (ordinal rank within the trial)
  - Scene identifier

### Test data (subject 60)

- Fixation metadata only — released in phases as described above. No MEG data for subject 60 is provided; that is what you predict.
- The same stimulus images (scenes and patches) are shared across all subjects.

---

## Evaluation

Evaluation metrics will be specified separately. Predictions will be evaluated by the organizers.

During the development phase of each challenge, you may submit `.npy` prediction files repeatedly and receive implementation feedback (e.g., format checks, baseline comparisons). Final evaluation predictions are submitted once, shortly after the final metadata release.

---

## Timeline

| Event | Date |
|-------|------|
| Training data release | TBD |
| Challenge 1 — development metadata release | TBD |
| Challenge 1 — final metadata release | TBD |
| Challenge 1 — submission deadline | TBD |
| Challenge 2 — development metadata release | TBD |
| Challenge 2 — final metadata release | TBD |
| Challenge 2 — submission deadline | TBD |

---

## Submission Format

All submissions are NumPy `.npy` files.

| Challenge | Shape | Axis order |
|-----------|-------|------------|
| 1 | `(n_fixations, n_channels)` | fixations × channels |
| 2 | `(n_fixations, n_channels, n_timepoints)` | fixations × channels × timepoints |

Fixation order must match the order in the provided metadata. Channel order must match the order in the training data. For Challenge 2, timepoint order is: −50, 50, 75, 100, 125, 150 ms.

---

## Rules and Constraints

- Models must generalise from subjects 1–5 to subject 60. You may not use any external MEG data for subject 60.
- You may use any computational approach (classical regression, ANN encoding, etc.).
- You may extract your own image features from the provided scenes and patches, or use pretrained model representations.
- Collaboration policy: TBD.
- Late submissions: TBD.

---

## Getting Started

1. Explore the training MEG data and metadata for subjects 1–5.
2. Extract or compute features from the stimulus patches and/or full scenes.
3. Fit an encoding model that maps stimulus and fixation features to MEG channel amplitudes.
4. When development metadata for subject 60 is released, generate predictions and submit for feedback.
5. Refine your model, then generate final predictions when the final metadata is released.

---

## Contact

For questions about the challenge, data, or submissions, contact Dan & Philip.
