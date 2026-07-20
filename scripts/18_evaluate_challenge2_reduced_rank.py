"""Evaluate reduced-rank Ridge for the Challenge 2 six-timepoint target."""

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
BASELINE = load_module(
    "14_evaluate_challenge2_spatial_saccade.py", "challenge2_baseline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO reduced-rank Ridge evaluation for Challenge 2"
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
    parser.add_argument("--target-pca-components", type=int, default=128)
    parser.add_argument("--visual-scale", type=float, default=0.25)
    parser.add_argument(
        "--time-ms",
        type=float,
        nargs=6,
        default=[-50.0, 50.0, 75.0, 100.0, 125.0, 150.0],
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
        default=ROOT / "results" / "challenge2" / "reduced_rank_loso.csv",
    )
    return parser.parse_args()


def make_design_matrices(
    metadata: pd.DataFrame,
    features: np.ndarray,
    valid_mask: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    pca_components: int,
    visual_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_valid = valid_mask[train_indices]
    if not train_valid.any():
        raise ValueError("No valid visual rows in training fold")

    visual_scaler = StandardScaler()
    visual_train_valid = np.asarray(features[train_indices[train_valid]], dtype=np.float32)
    visual_scaler.fit(visual_train_valid)
    visual_pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=0)
    visual_pca.fit(visual_scaler.transform(visual_train_valid))
    visual_train = visual_pca.transform(
        visual_scaler.transform(np.asarray(features[train_indices], dtype=np.float32))
    ).astype(np.float32)
    visual_validation = visual_pca.transform(
        visual_scaler.transform(np.asarray(features[validation_indices], dtype=np.float32))
    ).astype(np.float32)
    visual_train *= visual_scale
    visual_validation *= visual_scale

    eye_imputer = SimpleImputer(strategy="median")
    eye_scaler = RobustScaler()
    eye_train = eye_scaler.fit_transform(
        eye_imputer.fit_transform(metadata.iloc[train_indices][FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)
    eye_validation = eye_scaler.transform(
        eye_imputer.transform(metadata.iloc[validation_indices][FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)

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
    return X_train, X_validation, float(visual_pca.explained_variance_ratio_.sum())


def main() -> None:
    args = parse_args()
    if args.pca_components < 1 or args.target_pca_components < 1:
        raise ValueError("PCA component counts must be positive")
    if args.visual_scale < 0 or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("Visual scale and Ridge alphas must be valid")

    metadata = FIXATION.add_eye_movement_features(pd.read_csv(args.metadata_csv.resolve()))
    subjects = metadata["subject"].to_numpy()
    raw_targets = np.load(args.target_file.resolve(), mmap_mode="r")
    times = np.load(args.times_file.resolve())
    time_indices, actual_ms = BASELINE.select_time_indices(times, args.time_ms)
    targets = BASELINE.normalize_targets_by_subject_time(raw_targets, subjects, time_indices)
    features = np.load(args.feature_file.resolve(), mmap_mode="r")
    if features.shape != (len(metadata), 577):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    if args.valid_mask:
        valid_mask = np.asarray(np.load(args.valid_mask.resolve()), dtype=bool)
    else:
        valid_mask = np.any(features != 0, axis=1)
    if valid_mask.shape != (len(metadata),):
        raise ValueError("Validity mask does not align with metadata")

    target_width = 204 * len(time_indices)
    if args.target_pca_components > target_width:
        raise ValueError("Target PCA width exceeds target width")
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    result_rows: list[dict[str, float | int]] = []

    print("=" * 72)
    print("CHALLENGE 2 REDUCED-RANK SPATIAL + SACCADE RIDGECV · LOSO")
    print("=" * 72)
    print(f"Target PCA components: {args.target_pca_components}")
    print(f"Requested times (ms): {list(args.time_ms)}")
    print(f"Actual times (ms): {actual_ms.round(3).tolist()}")

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        X_train, X_validation, visual_variance = make_design_matrices(
            metadata,
            features,
            valid_mask,
            train_indices,
            validation_indices,
            args.pca_components,
            args.visual_scale,
        )
        y_train = targets[train_indices].transpose(0, 2, 1).reshape(-1, target_width)
        y_validation = targets[validation_indices]

        target_pca = PCA(
            n_components=args.target_pca_components,
            svd_solver="randomized",
            random_state=0,
        )
        y_train_scores = target_pca.fit_transform(y_train).astype(np.float32)
        model = RidgeCV(
            alphas=np.asarray(args.alphas, dtype=np.float64),
            alpha_per_target=True,
            gcv_mode="svd",
        )
        model.fit(X_train, y_train_scores)
        prediction_flat = target_pca.inverse_transform(model.predict(X_validation))
        prediction = prediction_flat.reshape(
            len(validation_indices), len(time_indices), 204
        ).transpose(0, 2, 1).astype(np.float32)

        print(f"\nHeld-out subject {held_out}")
        print(
            f"Rows: {len(train_indices)} train / {len(validation_indices)} validation; "
            f"visual PCA variance={visual_variance:.3f}; "
            f"target PCA variance={target_pca.explained_variance_ratio_.sum():.3f}"
        )
        for time_position, requested_ms in enumerate(args.time_ms):
            truth = y_validation[:, :, time_position]
            pred = prediction[:, :, time_position]
            metrics = FIXATION.score_channels(truth, pred)
            normalized_mse = mean_squared_error(truth, pred)
            print(
                f"  {requested_ms:g} ms (sample {actual_ms[time_position]:g} ms): "
                f"normalized MSE={normalized_mse:.6f}; mean r={metrics['mean_r']:.6f}"
            )
            result_rows.append(
                {
                    "validation_subject": held_out,
                    "requested_time_ms": float(requested_ms),
                    "actual_time_ms": float(actual_ms[time_position]),
                    "time_index": int(time_indices[time_position]),
                    "visual_pca_components": int(args.pca_components),
                    "target_pca_components": int(args.target_pca_components),
                    "visual_scale": float(args.visual_scale),
                    "visual_explained_variance": visual_variance,
                    "target_explained_variance": float(target_pca.explained_variance_ratio_.sum()),
                    "training_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "normalized_mse": float(normalized_mse),
                    **metrics,
                }
            )

    result = pd.DataFrame(result_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 72)
    print(
        f"Mean LOSO r={result['mean_r'].mean():.6f}; "
        f"mean normalized MSE={result['normalized_mse'].mean():.6f}"
    )
    print(f"Saved results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
