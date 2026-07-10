# Metadata column reference

Each row in `metadata.csv` describes one fixation event. The file covers all fixations from the training subjects (1–5) or, for the challenge splits, from subject 60. Only rows where `type == "fixation"` and `recording == "scene"` are relevant for the encoding challenge — other rows are saccades/blinks or events recorded during the verbal captioning period.

---

## Gaze position & event geometry

| Column | Unit | Description |
|--------|------|-------------|
| `mean_gx` | px | Mean horizontal gaze position during the fixation, in screen pixels (screen width = 1024 px, origin at left edge). |
| `mean_gy` | px | Mean vertical gaze position during the fixation, in screen pixels (screen height = 768 px, origin at top edge). |
| `start_gx` / `start_gy` | px | Gaze position at fixation onset. |
| `end_gx` / `end_gy` | px | Gaze position at fixation offset. |
| `duration` | s | Duration of the fixation event. |
| `start_time` / `end_time` | samples | Timestamps of fixation onset and offset, in EyeLink samples (1000 Hz). |
| `rms` | px | Root-mean-square deviation of gaze samples within the fixation — a measure of fixation stability. |
| `sd` | px | Standard deviation of gaze samples within the fixation. |

> **Tip:** `mean_gx` / `mean_gy` are the most useful columns for locating the fixation on the image. The pre-extracted stimulus crops are centred on these coordinates.

---

## Surrounding saccade context

The metadata contains only fixation events, so columns that belong to the saccade event itself (`amplitude`, `peak_velocity`) are entirely empty. The columns below describe the saccades surrounding each fixation and are mostly populated.

| Column | Unit | Description |
|--------|------|-------------|
| `amplitude_pre` | dva | Amplitude of the saccade that landed on this fixation (i.e. the incoming saccade). |
| `amplitude_post` | dva | Amplitude of the saccade that left this fixation (i.e. the outgoing saccade). |
| `duration_pre` | s | Duration of the fixation immediately before this one. |
| `duration_post` | s | Duration of the fixation immediately after this one. |
| `multi_saccade` | — | Whether the incoming saccade was a multi-step saccade (glissadic overshoot). Values: `"no"` or a description of the pattern. |

> **Note:** `amplitude_pre`/`amplitude_post` are NaN for the first and last fixation of a trial respectively (~4% of rows).

### Computing saccade distance yourself

The pre-computed amplitudes are in degrees of visual angle. If you prefer to work with pixel distances, or want to compute the distance from the *previous fixation's gaze position* directly, group by `subject` and `sceneID` (or `subject`, `session`, `trial`) and use the shift-diff pattern:

```python
df = df.sort_values(['subject', 'session', 'trial', 'fix_sequence'])

grp = df.groupby(['subject', 'session', 'trial'])
dx = df['mean_gx'] - grp['mean_gx'].shift(1)
dy = df['mean_gy'] - grp['mean_gy'].shift(1)
df['saccade_dist_px'] = (dx**2 + dy**2) ** 0.5
```

Convert pixels to degrees of visual angle using the screen geometry. The stimulus subtends 28.54° × 21.61° and is centred on the screen with grey padding — it occupies `SCREEN_USAGE = 0.925` of the screen height (768 px). The full screen is therefore slightly larger in DVA than the stimulus:

```python
from megcourselib.constants import SCREEN_H, SCREEN_USAGE

STIM_DVA_H = 21.61  # stimulus height [°]
STIM_DVA_W = 28.54  # stimulus width  [°]

# Stimulus occupies SCREEN_USAGE * SCREEN_H pixels vertically = STIM_DVA_H degrees
# → px/° is the same everywhere on screen (grey borders have the same pixel density)
PX_PER_DEG = (SCREEN_H * SCREEN_USAGE) / STIM_DVA_H  # ≈ 32.9 px/°

df['saccade_dist_dva'] = df['saccade_dist_px'] / PX_PER_DEG
```

---

## Sequence position within the trial

| Column | Description |
|--------|-------------|
| `fix_sequence` | Ordinal position of this fixation within the trial, counting from the start (0-indexed). |
| `fix_sequence_from_last` | Ordinal position counting from the *last* fixation (−1 = last fixation, −2 = second-to-last, …). The last fixation is always interrupted by the scene offset — consider excluding rows where `fix_sequence_from_last == -1`. |
| `sac_sequence` | Ordinal position of the associated saccade, counting from trial start. |
| `sac_sequence_from_last` | Ordinal position of the associated saccade, counting from the last. |
| `time_in_trial` | Time from scene onset to fixation onset, in seconds. |

---

## Trial & session bookkeeping

| Column | Description |
|--------|-------------|
| `subject` | Subject identifier (1–5 for training data; 60 for the test subject). |
| `session` | Recording session number (1–10 per subject). |
| `trial` | Trial number within the session. |
| `block` | Block number (14 blocks/session, except session 1 which has 10). |
| `trial_per_block` | Trial number within its block. |
| `sceneID` | MS-COCO image ID of the stimulus scene. Use this to look up the corresponding image file. |
| `recording` | Phase of the trial: `"scene"` = stimulus viewing (4 s), `"microphone"` = captioning prompt, `"caption"` = verbal description recording. **Use only `"scene"` rows.** |
| `caption_task` | Whether this trial was followed by a verbal captioning prompt (`1`) or not (`0`). |

---

## Columns you can ignore

| Column | Note |
|--------|------|
| `type` | Always `"fixation"` in the challenge data (blinks and saccades are excluded). |
| `amplitude` | Saccade amplitude — empty because only fixation events are included. |
| `peak_velocity` | Saccade peak velocity — empty for the same reason. Use `amplitude_pre`/`amplitude_post` or compute distance from gaze positions instead. |
| `blink_id` | Links blink events to nearby fixations — not relevant here. |
| `end_point` | Internal processing artefact, no meaningful value. |
