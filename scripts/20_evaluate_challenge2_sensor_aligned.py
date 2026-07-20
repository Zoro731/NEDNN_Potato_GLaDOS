"""Evaluate Challenge 2 after aligning MEG targets into each fold's sensor space.

The training targets use the same 204-channel ordering, but each subject has a
subject-specific gradiometer geometry in its FIF file.  This script builds a
small inverse-distance interpolation from every source subject's sensor
locations to the held-out subject's locations before fitting Ridge.  It also
retests cross-subject fixation prototypes after alignment.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXATION = load_module("09_evaluate_fixation_aware_ridge.py", "fixation_aware_ridge")
EVALUATOR = load_module("14_evaluate_challenge2_spatial_saccade.py", "challenge2_evaluator")
PROTOTYPE = load_module("19_evaluate_challenge2_fixation_prototypes.py", "prototype")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO sensor-aligned Challenge 2 evaluation")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/brainencoding26/challenge2/training/metadata.csv"))
    parser.add_argument("--target-file", type=Path, default=Path("data/brainencoding26/challenge2/training/meg_c2.npy"))
    parser.add_argument("--times-file", type=Path, default=Path("data/brainencoding26/challenge2/training/times.npy"))
    parser.add_argument("--feature-file", type=Path, default=Path("challenge2/training/spatial_features.npy"))
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--channel-names", type=Path, default=Path("challenge2/training/channel_names.txt"))
    parser.add_argument("--grad-info-dir", type=Path, default=Path("challenge2/training"))
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--visual-scale", type=float, default=0.05)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--time-ms", type=float, nargs=6, default=[-50.0, 50.0, 75.0, 100.0, 125.0, 150.0])
    parser.add_argument("--alphas", type=float, nargs="+", default=np.logspace(-3, 6, 19).tolist())
    parser.add_argument("--blend-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "challenge2" / "sensor_aligned_loso.csv")
    return parser.parse_args()


def read_positions(path: Path, channel_names: list[str]) -> np.ndarray:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read gradiometer geometry") from exc
    info = mne.io.read_info(path.resolve(), verbose="ERROR")
    by_name = {
        channel["ch_name"]: np.asarray(channel["loc"][:3], dtype=np.float64)
        for channel in info["chs"]
        if channel["ch_name"]
    }
    missing = [name for name in channel_names if name not in by_name]
    if not missing:
        positions = np.stack([by_name[name] for name in channel_names])
    else:
        # The challenge anonymizes target columns as MEG0001...MEG0204,
        # whereas the FIF retains physical Elekta names (MEG0113, etc.).
        # The target columns follow the ordered planar-gradiometer list in the
        # grad-info file, so use that order when anonymized labels are detected.
        import mne

        grad_picks = mne.pick_types(info, meg="grad", eeg=False, exclude=[])
        if len(grad_picks) != len(channel_names):
            raise ValueError(
                f"{path} has {len(grad_picks)} planar gradiometers, expected "
                f"{len(channel_names)}; missing labels={missing[:3]}"
            )
        positions = np.stack(
            [np.asarray(info["chs"][pick]["loc"][:3], dtype=np.float64) for pick in grad_picks]
        )
        print(
            f"Using ordered FIF planar-gradiometer positions for {path.name}; "
            f"{len(missing)} anonymized labels do not occur in the FIF"
        )
    if not np.isfinite(positions).all() or np.allclose(positions, 0):
        raise ValueError(f"Invalid sensor positions in {path}")
    return positions


def interpolation_matrix(source: np.ndarray, target: np.ndarray, neighbors: int) -> np.ndarray:
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=2)
    k = min(max(1, neighbors), source.shape[0])
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    weights = np.zeros_like(nearest, dtype=np.float64)
    for row in range(target.shape[0]):
        d = distances[row, nearest[row]]
        if np.min(d) < 1e-12:
            weights[row, np.argmin(d)] = 1.0
        else:
            inv = 1.0 / np.maximum(d, 1e-12) ** 2
            weights[row] = inv / inv.sum()
    matrix = np.zeros((target.shape[0], source.shape[0]), dtype=np.float32)
    rows = np.arange(target.shape[0])[:, None]
    matrix[rows, nearest] = weights.astype(np.float32)
    return matrix


def align_targets(
    targets: np.ndarray,
    subjects: np.ndarray,
    indices: np.ndarray,
    held_out: int,
    positions: dict[int, np.ndarray],
    neighbors: int,
) -> np.ndarray:
    output = np.empty((len(indices),) + targets.shape[1:], dtype=np.float32)
    target_positions = positions[held_out]
    local_subjects = subjects[indices]
    for subject in np.unique(local_subjects):
        local = np.flatnonzero(local_subjects == subject)
        source_positions = positions[int(subject)]
        matrix = interpolation_matrix(source_positions, target_positions, neighbors)
        values = targets[indices[local]]
        output[local] = np.einsum("oc,nct->not", matrix, values, optimize=True)
    return output


def design_matrices(metadata, features, valid_mask, train_indices, validation_indices, args):
    train_valid = valid_mask[train_indices]
    visual_scaler = StandardScaler()
    visual_valid = np.asarray(features[train_indices[train_valid]], dtype=np.float32)
    visual_scaler.fit(visual_valid)
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=0)
    pca.fit(visual_scaler.transform(visual_valid))
    visual_train = pca.transform(visual_scaler.transform(np.asarray(features[train_indices], dtype=np.float32))).astype(np.float32)
    visual_validation = pca.transform(visual_scaler.transform(np.asarray(features[validation_indices], dtype=np.float32))).astype(np.float32)
    visual_train *= args.visual_scale
    visual_validation *= args.visual_scale
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    eye_train = scaler.fit_transform(imputer.fit_transform(metadata.iloc[train_indices][FIXATION.SACCADE_FEATURES])).astype(np.float32)
    eye_validation = scaler.transform(imputer.transform(metadata.iloc[validation_indices][FIXATION.SACCADE_FEATURES])).astype(np.float32)
    x_train = np.concatenate([visual_train, eye_train, train_valid[:, None].astype(np.float32)], axis=1)
    x_validation = np.concatenate([visual_validation, eye_validation, valid_mask[validation_indices, None].astype(np.float32)], axis=1)
    return x_train, x_validation, float(pca.explained_variance_ratio_.sum())


def main() -> None:
    args = parse_args()
    if args.pca_components < 1 or args.visual_scale < 0 or args.neighbors < 1:
        raise ValueError("PCA, visual scale, and neighbor count must be valid")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All Ridge alphas must be positive")
    if any(weight < 0 or weight > 1 for weight in args.blend_weights):
        raise ValueError("Blend weights must be between 0 and 1")

    metadata = FIXATION.add_eye_movement_features(pd.read_csv(args.metadata_csv.resolve()))
    subjects = metadata["subject"].to_numpy()
    raw_targets = np.load(args.target_file.resolve(), mmap_mode="r")
    times = np.load(args.times_file.resolve())
    time_indices, actual_ms = EVALUATOR.select_time_indices(times, args.time_ms)
    targets = EVALUATOR.normalize_targets_by_subject_time(raw_targets, subjects, time_indices)
    features = np.load(args.feature_file.resolve(), mmap_mode="r")
    valid_mask = np.asarray(np.load(args.valid_mask.resolve()), dtype=bool) if args.valid_mask else np.any(features != 0, axis=1)
    if raw_targets.shape[:2] != (len(metadata), 204) or features.shape != (len(metadata), 577) or valid_mask.shape != (len(metadata),):
        raise ValueError("Input shapes do not align with metadata")
    channel_names = [line.strip() for line in args.channel_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(channel_names) != 204:
        raise ValueError(f"Expected 204 channel names, got {len(channel_names)}")
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    positions = {
        subject: read_positions(args.grad_info_dir / f"sub-{subject:02d}_grad_info.fif", channel_names)
        for subject in fold_subjects
    }
    exact_keys = PROTOTYPE.composite_key(metadata, include_fixation=True)
    scene_keys = PROTOTYPE.composite_key(metadata, include_fixation=False)
    result_rows = []
    print("=" * 72)
    print("CHALLENGE 2 SENSOR-ALIGNED RIDGE + PROTOTYPES · LOSO")
    print("=" * 72)
    print(f"Neighbors: {args.neighbors}; blend weights: {list(args.blend_weights)}")
    print(f"Actual times (ms): {actual_ms.round(3).tolist()}")

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        x_train, x_validation, pca_variance = design_matrices(metadata, features, valid_mask, train_indices, validation_indices, args)
        y_train = align_targets(targets, subjects, train_indices, held_out, positions, args.neighbors)
        y_validation = targets[validation_indices]
        model = RidgeCV(alphas=np.asarray(args.alphas, dtype=np.float64), alpha_per_target=True, gcv_mode="svd")
        model.fit(x_train, y_train.transpose(0, 2, 1).reshape(len(train_indices), -1))
        ridge = model.predict(x_validation).reshape(len(validation_indices), len(time_indices), 204).transpose(0, 2, 1).astype(np.float32)
        prototypes, match_counts = PROTOTYPE.prototype_predict(
            y_train, exact_keys[train_indices], scene_keys[train_indices], exact_keys[validation_indices], scene_keys[validation_indices]
        )
        print(f"\nHeld-out subject {held_out}; PCA variance={pca_variance:.3f}; matches={match_counts}")
        for weight in args.blend_weights:
            prediction = (1.0 - weight) * ridge + weight * prototypes
            fold_rs = []
            for time_position, requested_ms in enumerate(args.time_ms):
                truth = y_validation[:, :, time_position]
                pred = prediction[:, :, time_position]
                metrics = FIXATION.score_channels(truth, pred)
                fold_rs.append(metrics["mean_r"])
                result_rows.append({
                    "validation_subject": held_out,
                    "prototype_weight": float(weight),
                    "ridge_weight": float(1.0 - weight),
                    "requested_time_ms": float(requested_ms),
                    "actual_time_ms": float(actual_ms[time_position]),
                    "time_index": int(time_indices[time_position]),
                    "pca_components": int(args.pca_components),
                    "visual_scale": float(args.visual_scale),
                    "sensor_neighbors": int(args.neighbors),
                    "explained_variance": pca_variance,
                    "normalized_mse": float(mean_squared_error(truth, pred)),
                    "match_exact_rows": match_counts["exact"],
                    "match_scene_rows": match_counts["scene"],
                    "match_global_rows": match_counts["global"],
                    **metrics,
                })
            print(f"  prototype weight={weight:g}: mean r={np.mean(fold_rs):.6f}")

    result = pd.DataFrame(result_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary = result.groupby("prototype_weight", as_index=False)[["mean_r", "normalized_mse"]].mean()
    print("\n" + "=" * 72)
    print(summary.to_string(index=False))
    best = summary.sort_values("mean_r", ascending=False).iloc[0]
    print(f"Best sensor-aligned prototype weight: {best.prototype_weight:g}; mean r={best.mean_r:.6f}; MSE={best.normalized_mse:.6f}")
    print(f"Saved results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
