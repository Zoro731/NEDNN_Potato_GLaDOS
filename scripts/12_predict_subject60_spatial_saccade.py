"""Fit the frozen spatial+saccade model and predict subject 60 rows."""

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
        description="Fit frozen LOSO configuration and predict subject 60."
    )
    parser.add_argument(
        "--train-feature-file",
        type=Path,
        default=Path("challenge1/training/spatial_features.npy"),
    )
    parser.add_argument(
        "--train-valid-mask",
        type=Path,
        default=Path("challenge1/training/spatial_features_valid_mask.npy"),
    )
    parser.add_argument(
        "--dev-feature-file",
        type=Path,
        default=Path("challenge1/subject60/challenge1_dev/spatial_features.npy"),
    )
    parser.add_argument(
        "--dev-metadata",
        type=Path,
        default=Path("challenge1/subject60/challenge1_dev/metadata.csv"),
    )
    parser.add_argument(
        "--dev-valid-mask",
        type=Path,
        default=Path(
            "challenge1/subject60/challenge1_dev/spatial_features_valid_mask.npy"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "results/subject60/spatial_saccade_scale025_predictions.npy"
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("results/subject60/spatial_saccade_scale025_summary.json"),
    )
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--visual-scale", type=float, default=0.25)
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
        raise ValueError("PCA components must be positive and visual scale valid")

    train_metadata = FIXATION.add_eye_movement_features(
        pd.read_csv(ROOT / "challenge1" / "training" / "metadata.csv")
    )
    dev_metadata = FIXATION.add_eye_movement_features(
        pd.read_csv(args.dev_metadata.resolve())
    )
    targets = np.load(
        ROOT / "challenge1" / "training" / "meg_110ms.npy", mmap_mode="r"
    )
    train_features = np.load(args.train_feature_file.resolve(), mmap_mode="r")
    dev_features = np.load(args.dev_feature_file.resolve(), mmap_mode="r")
    train_valid = np.asarray(np.load(args.train_valid_mask.resolve()), dtype=bool)
    dev_valid = np.asarray(np.load(args.dev_valid_mask.resolve()), dtype=bool)

    if len(train_features) != len(train_metadata) or len(train_valid) != len(train_metadata):
        raise ValueError("Training features/mask do not align with metadata")
    if len(dev_features) != len(dev_metadata) or len(dev_valid) != len(dev_metadata):
        raise ValueError("Development features/mask do not align with metadata")
    if train_features.shape[1] != dev_features.shape[1]:
        raise ValueError("Training and development feature widths differ")
    if args.pca_components > train_features.shape[1]:
        raise ValueError("PCA components exceed spatial feature width")

    train_subjects = train_metadata["subject"].to_numpy()
    normalized_targets = FIXATION.normalize_targets_by_subject(targets, train_subjects)
    valid_visual = np.asarray(train_features[train_valid], dtype=np.float32)
    visual_scaler = StandardScaler()
    visual_scaler.fit(valid_visual)
    pca = PCA(
        n_components=args.pca_components,
        svd_solver="randomized",
        random_state=0,
    )
    pca.fit(visual_scaler.transform(valid_visual))

    train_visual = pca.transform(
        visual_scaler.transform(np.asarray(train_features, dtype=np.float32))
    ).astype(np.float32)
    dev_visual = pca.transform(
        visual_scaler.transform(np.asarray(dev_features, dtype=np.float32))
    ).astype(np.float32)
    train_visual *= args.visual_scale
    dev_visual *= args.visual_scale

    eye_imputer = SimpleImputer(strategy="median")
    eye_scaler = RobustScaler()
    train_eye = eye_imputer.fit_transform(
        train_metadata[FIXATION.SACCADE_FEATURES]
    )
    dev_eye = eye_imputer.transform(dev_metadata[FIXATION.SACCADE_FEATURES])
    train_eye = eye_scaler.fit_transform(train_eye).astype(np.float32)
    dev_eye = eye_scaler.transform(dev_eye).astype(np.float32)

    X_train = np.concatenate(
        [train_visual, train_eye, train_valid[:, None].astype(np.float32)], axis=1
    )
    X_dev = np.concatenate(
        [dev_visual, dev_eye, dev_valid[:, None].astype(np.float32)], axis=1
    )
    model = RidgeCV(
        alphas=np.asarray(args.alphas, dtype=np.float64),
        alpha_per_target=True,
        gcv_mode="svd",
    )
    model.fit(X_train, normalized_targets)
    prediction = model.predict(X_dev).astype(np.float32)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_file, prediction)
    summary = {
        "configuration": "spatial_pca128_plus_incoming_saccade",
        "visual_scale": args.visual_scale,
        "pca_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "training_rows": int(len(train_metadata)),
        "training_valid_visual_rows": int(train_valid.sum()),
        "development_rows": int(len(dev_metadata)),
        "development_valid_visual_rows": int(dev_valid.sum()),
        "prediction_shape": list(prediction.shape),
        "alpha_min": float(np.min(model.alpha_)),
        "alpha_median": float(np.median(model.alpha_)),
        "alpha_max": float(np.max(model.alpha_)),
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=" * 72)
    print("SUBJECT 60 DEVELOPMENT PREDICTION")
    print("=" * 72)
    print(f"Prediction file: {args.output_file.resolve()}")
    print(f"Shape: {prediction.shape}; dtype={prediction.dtype}")
    print(f"Valid development visual rows: {int(dev_valid.sum())}/{len(dev_valid)}")
    print(
        f"PCA variance={summary['explained_variance']:.3f}; "
        f"selected alpha median={summary['alpha_median']:g}"
    )
    print(f"Summary file: {args.summary_file.resolve()}")


if __name__ == "__main__":
    main()
