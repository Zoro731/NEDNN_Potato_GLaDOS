"""Fit the Challenge 2 spatial+saccade model and predict subject 60 dev data."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
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
EVALUATOR = load_module(
    "14_evaluate_challenge2_spatial_saccade.py", "challenge2_evaluator"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Challenge 2 subject 60 development fixations"
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("data/brainencoding26/challenge2/training/metadata.csv"),
    )
    parser.add_argument(
        "--dev-metadata",
        type=Path,
        default=Path(
            "data/brainencoding26/challenge2/subject60/challenge2_dev/metadata.csv"
        ),
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
        "--train-features",
        type=Path,
        default=Path("challenge2/training/spatial_features.npy"),
    )
    parser.add_argument(
        "--train-valid-mask",
        type=Path,
        default=Path("challenge2/training/spatial_features_valid_mask.npy"),
    )
    parser.add_argument(
        "--dev-features",
        type=Path,
        default=Path("challenge2/subject60/challenge2_dev/spatial_features.npy"),
    )
    parser.add_argument(
        "--dev-valid-mask",
        type=Path,
        default=Path(
            "challenge2/subject60/challenge2_dev/spatial_features_valid_mask.npy"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "challenge2" / "subject60_predictions.npy",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "challenge2" / "subject60_summary.json",
    )
    parser.add_argument("--pca-components", type=int, default=128)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pca_components < 1 or args.visual_scale < 0:
        raise ValueError("PCA components and visual scale must be valid")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All Ridge alphas must be positive")

    train_metadata = FIXATION.add_eye_movement_features(
        pd.read_csv(args.train_metadata.resolve())
    )
    dev_metadata = FIXATION.add_eye_movement_features(
        pd.read_csv(args.dev_metadata.resolve())
    )
    train_subjects = train_metadata["subject"].to_numpy()
    raw_targets = np.load(args.target_file.resolve(), mmap_mode="r")
    times = np.load(args.times_file.resolve())
    time_indices, actual_ms = EVALUATOR.select_time_indices(times, args.time_ms)
    targets = EVALUATOR.normalize_targets_by_subject_time(
        raw_targets, train_subjects, time_indices
    )

    train_features = np.load(args.train_features.resolve(), mmap_mode="r")
    dev_features = np.load(args.dev_features.resolve(), mmap_mode="r")
    if train_features.ndim != 2 or train_features.shape[0] != len(train_metadata):
        raise ValueError(f"Unexpected training feature shape: {train_features.shape}")
    if dev_features.ndim != 2 or dev_features.shape[0] != len(dev_metadata):
        raise ValueError(f"Unexpected development feature shape: {dev_features.shape}")
    if train_features.shape[1] != dev_features.shape[1]:
        raise ValueError("Training and development feature widths differ")
    train_valid = np.asarray(np.load(args.train_valid_mask.resolve()), dtype=bool)
    dev_valid = np.asarray(np.load(args.dev_valid_mask.resolve()), dtype=bool)
    if train_valid.shape != (len(train_metadata),):
        raise ValueError("Training validity mask does not align")
    if dev_valid.shape != (len(dev_metadata),):
        raise ValueError("Development validity mask does not align")

    visual_scaler = StandardScaler()
    visual_scaler.fit(np.asarray(train_features[train_valid], dtype=np.float32))
    pca = PCA(
        n_components=args.pca_components,
        svd_solver="randomized",
        random_state=0,
    )
    pca.fit(visual_scaler.transform(np.asarray(train_features[train_valid], dtype=np.float32)))
    visual_train = pca.transform(
        visual_scaler.transform(np.asarray(train_features, dtype=np.float32))
    ).astype(np.float32)
    visual_dev = pca.transform(
        visual_scaler.transform(np.asarray(dev_features, dtype=np.float32))
    ).astype(np.float32)
    visual_train *= args.visual_scale
    visual_dev *= args.visual_scale

    eye_imputer = SimpleImputer(strategy="median")
    eye_scaler = RobustScaler()
    eye_train = eye_scaler.fit_transform(
        eye_imputer.fit_transform(train_metadata[FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)
    eye_dev = eye_scaler.transform(
        eye_imputer.transform(dev_metadata[FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)

    X_train = np.concatenate(
        [visual_train, eye_train, train_valid[:, None].astype(np.float32)], axis=1
    )
    X_dev = np.concatenate(
        [visual_dev, eye_dev, dev_valid[:, None].astype(np.float32)], axis=1
    )
    target_width = 204 * len(time_indices)
    y_train = targets.transpose(0, 2, 1).reshape(len(train_metadata), target_width)

    model = RidgeCV(
        alphas=np.asarray(args.alphas, dtype=np.float64),
        alpha_per_target=True,
        gcv_mode="svd",
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_dev).reshape(
        len(dev_metadata), len(time_indices), 204
    ).transpose(0, 2, 1).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.resolve(), prediction)
    selected_alpha = np.asarray(model.alpha_, dtype=np.float64)
    summary = {
        "train_rows": int(len(train_metadata)),
        "dev_rows": int(len(dev_metadata)),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "train_valid_visual_rows": int(train_valid.sum()),
        "dev_valid_visual_rows": int(dev_valid.sum()),
        "requested_time_ms": [float(value) for value in args.time_ms],
        "actual_time_ms": [float(value) for value in actual_ms],
        "time_indices": [int(value) for value in time_indices],
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "visual_scale": float(args.visual_scale),
        "alpha_min": float(selected_alpha.min()),
        "alpha_median": float(np.median(selected_alpha)),
        "alpha_max": float(selected_alpha.max()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=" * 72)
    print("CHALLENGE 2 SUBJECT 60 DEVELOPMENT PREDICTION")
    print("=" * 72)
    print(f"Prediction file: {args.output.resolve()}")
    print(f"Shape: {prediction.shape}; dtype={prediction.dtype}")
    print(f"Valid development visual rows: {int(dev_valid.sum())}/{len(dev_valid)}")
    print(f"PCA variance={pca.explained_variance_ratio_.sum():.3f}")
    print(f"Actual timepoints (ms): {actual_ms.round(3).tolist()}")
    print(f"Summary file: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
