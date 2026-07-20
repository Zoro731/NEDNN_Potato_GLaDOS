"""Evaluate the Challenge 2 spatial+saccade model across six MEG timepoints.

The target is stored as ``(fixation, channel, full_time_axis)``.  This script
selects the six official output times (using the nearest available samples),
normalizes targets within subject, and evaluates a PCA + Ridge model with
leave-one-subject-out folds.
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


def load_fixation_module():
    path = Path(__file__).with_name("09_evaluate_fixation_aware_ridge.py")
    spec = importlib.util.spec_from_file_location("fixation_aware_ridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXATION = load_fixation_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO Ridge evaluation for Challenge 2 six-timepoint targets"
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/metadata.csv"),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/meg_c2.npy"),
    )
    parser.add_argument(
        "--times-file",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/times.npy"),
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=Path("challenge2/training/spatial_features.npy"),
    )
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--visual-scale", type=float, default=0.25)
    parser.add_argument(
        "--time-ms",
        type=float,
        nargs=6,
        default=[-50.0, 50.0, 75.0, 100.0, 125.0, 150.0],
        metavar=("T0", "T1", "T2", "T3", "T4", "T5"),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=np.logspace(-3, 6, 19).tolist(),
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "challenge2" / "spatial_saccade_loso.csv",
    )
    return parser.parse_args()


def select_time_indices(times: np.ndarray, requested_ms: list[float]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    requested = np.asarray(requested_ms, dtype=np.float64) / 1000.0
    indices = np.abs(times[:, None] - requested[None, :]).argmin(axis=0)
    actual_ms = times[indices] * 1000.0
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Requested times map to duplicate samples: {indices.tolist()}")
    return indices.astype(np.int64), actual_ms


def normalize_targets_by_subject_time(
    raw_targets: np.ndarray, subjects: np.ndarray, time_indices: np.ndarray
) -> np.ndarray:
    selected = np.stack(
        [np.asarray(raw_targets[:, :, index], dtype=np.float32) for index in time_indices],
        axis=2,
    )
    normalized = np.empty_like(selected)
    for subject in np.unique(subjects):
        mask = subjects == subject
        values = selected[mask]
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-20] = 1.0
        normalized[mask] = (values - mean) / std
    return normalized


def main() -> None:
    args = parse_args()
    if args.pca_components < 1 or args.visual_scale < 0:
        raise ValueError("PCA components and visual scale must be valid")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All Ridge alphas must be positive")

    metadata = pd.read_csv(args.metadata_csv.resolve())
    metadata = FIXATION.add_eye_movement_features(metadata)
    subjects = metadata["subject"].to_numpy()

    raw_targets = np.load(args.target_file.resolve(), mmap_mode="r")
    times = np.load(args.times_file.resolve())
    time_indices, actual_ms = select_time_indices(times, args.time_ms)
    if raw_targets.ndim != 3 or raw_targets.shape[:2] != (len(metadata), 204):
        raise ValueError(
            f"Expected target shape ({len(metadata)}, 204, time), got {raw_targets.shape}"
        )
    if np.max(time_indices) >= raw_targets.shape[2]:
        raise ValueError("Selected time index exceeds target time axis")

    features = np.load(args.feature_file.resolve(), mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(metadata):
        raise ValueError(
            f"Expected a 2-D feature matrix with {len(metadata)} rows, got {features.shape}"
        )
    if args.pca_components > features.shape[1]:
        raise ValueError("Requested PCA width exceeds feature width")

    if args.valid_mask:
        valid_mask = np.asarray(np.load(args.valid_mask.resolve()), dtype=bool)
    else:
        valid_mask = np.any(features != 0, axis=1)
    if valid_mask.shape != (len(metadata),):
        raise ValueError("Validity mask does not align with metadata")

    targets = normalize_targets_by_subject_time(raw_targets, subjects, time_indices)
    target_width = 204 * len(time_indices)
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    result_rows: list[dict[str, float | int]] = []

    print("=" * 72)
    print("CHALLENGE 2 SPATIAL VISUAL + INCOMING SACCADE RIDGECV · LOSO")
    print("=" * 72)
    print(f"Metadata rows: {len(metadata)}")
    print(f"Target shape: {raw_targets.shape}")
    print(f"Requested times (ms): {list(args.time_ms)}")
    print(f"Selected indices: {time_indices.tolist()}")
    print(f"Actual times (ms): {actual_ms.round(3).tolist()}")
    print(f"Spatial features: {features.shape}; PCA={args.pca_components}")

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        train_valid = valid_mask[train_indices]
        if not train_valid.any():
            raise ValueError(f"No valid visual rows in training fold {held_out}")

        visual_scaler = StandardScaler()
        visual_train_valid = np.asarray(features[train_indices[train_valid]], dtype=np.float32)
        visual_scaler.fit(visual_train_valid)
        pca = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=0,
        )
        pca.fit(visual_scaler.transform(visual_train_valid))
        visual_train = pca.transform(
            visual_scaler.transform(np.asarray(features[train_indices], dtype=np.float32))
        ).astype(np.float32)
        visual_validation = pca.transform(
            visual_scaler.transform(np.asarray(features[validation_indices], dtype=np.float32))
        ).astype(np.float32)
        visual_train *= args.visual_scale
        visual_validation *= args.visual_scale

        eye_imputer = SimpleImputer(strategy="median")
        eye_scaler = RobustScaler()
        eye_train = eye_imputer.fit_transform(
            metadata.iloc[train_indices][FIXATION.SACCADE_FEATURES]
        )
        eye_validation = eye_imputer.transform(
            metadata.iloc[validation_indices][FIXATION.SACCADE_FEATURES]
        )
        eye_train = eye_scaler.fit_transform(eye_train).astype(np.float32)
        eye_validation = eye_scaler.transform(eye_validation).astype(np.float32)

        X_train = np.concatenate(
            [visual_train, eye_train, train_valid[:, None].astype(np.float32)], axis=1
        )
        X_validation = np.concatenate(
            [
                visual_validation,
                eye_validation,
                valid_mask[validation_indices, None].astype(np.float32),
            ],
            axis=1,
        )
        y_train = targets[train_indices].transpose(0, 2, 1).reshape(-1, target_width)
        y_validation = targets[validation_indices]

        model = RidgeCV(
            alphas=np.asarray(args.alphas, dtype=np.float64),
            alpha_per_target=True,
            gcv_mode="svd",
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation).reshape(
            len(validation_indices), len(time_indices), 204
        ).transpose(0, 2, 1).astype(np.float32)

        print(f"\nHeld-out subject {held_out}")
        print(
            f"Rows: {len(train_indices)} train / {len(validation_indices)} validation; "
            f"valid visual train rows={int(train_valid.sum())}"
        )
        print(
            f"PCA variance={pca.explained_variance_ratio_.sum():.3f}; "
            f"visual scale={args.visual_scale:g}"
        )
        selected_alpha = np.asarray(model.alpha_, dtype=np.float64)
        for time_position, requested_ms in enumerate(args.time_ms):
            truth = y_validation[:, :, time_position]
            pred = prediction[:, :, time_position]
            metrics = FIXATION.score_channels(truth, pred)
            normalized_mse = mean_squared_error(truth, pred)
            print(
                f"  {requested_ms:g} ms (sample {actual_ms[time_position]:g} ms): "
                f"normalized MSE={normalized_mse:.6f}; mean r={metrics['mean_r']:.6f}"
            )
            alpha_slice = selected_alpha[time_position * 204 : (time_position + 1) * 204]
            result_rows.append(
                {
                    "validation_subject": held_out,
                    "requested_time_ms": float(requested_ms),
                    "actual_time_ms": float(actual_ms[time_position]),
                    "time_index": int(time_indices[time_position]),
                    "pca_components": int(pca.n_components_),
                    "visual_scale": float(args.visual_scale),
                    "explained_variance": float(pca.explained_variance_ratio_.sum()),
                    "training_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "normalized_mse": float(normalized_mse),
                    **metrics,
                    "alpha_min": float(alpha_slice.min()),
                    "alpha_median": float(np.median(alpha_slice)),
                    "alpha_max": float(alpha_slice.max()),
                }
            )

    result = pd.DataFrame(result_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary = result.groupby("requested_time_ms", as_index=False)[
        ["mean_r", "normalized_mse"]
    ].mean()
    print("\n" + "=" * 72)
    print(
        f"Mean LOSO r={result['mean_r'].mean():.6f}; "
        f"mean normalized MSE={result['normalized_mse'].mean():.6f}"
    )
    print("Per-timepoint mean r:")
    for row in summary.itertuples(index=False):
        print(f"  {row.requested_time_ms:g} ms: r={row.mean_r:.6f}; MSE={row.normalized_mse:.6f}")
    print(f"Saved results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
