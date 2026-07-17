"""Evaluate image embeddings with leakage-safe leave-one-subject-out folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
METADATA_FEATURES = [
    "duration",
    "mean_gx",
    "mean_gy",
    "rms",
    "sd",
    "time_in_trial",
    "fix_sequence",
    "fix_sequence_from_last",
    "amplitude_pre",
    "amplitude_post",
    "duration_pre",
    "duration_post",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_file", type=Path)
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "visual_baseline" / "loso_results.csv",
    )
    return parser.parse_args()


def normalize_targets_by_subject(
    targets: np.ndarray, subjects: np.ndarray
) -> np.ndarray:
    normalized = np.empty(targets.shape, dtype=np.float32)
    for subject in np.unique(subjects):
        mask = subjects == subject
        values = np.asarray(targets[mask], dtype=np.float32)
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-20] = 1.0
        normalized[mask] = (values - mean) / std
    return normalized


def mean_channel_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=0)
        * np.sum(pred_centered**2, axis=0)
    )
    valid = denominator > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def make_design(
    features: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    valid_mask: np.ndarray,
    *,
    image_scaler: StandardScaler,
    metadata_imputer: SimpleImputer | None,
    metadata_scaler: RobustScaler | None,
    fit: bool,
) -> np.ndarray:
    image = np.asarray(features[indices], dtype=np.float32).copy()
    image = (
        image_scaler.fit_transform(image)
        if fit
        else image_scaler.transform(image)
    )
    valid_column = valid_mask[indices, None].astype(np.float32)
    parts = [image, valid_column]

    if metadata_imputer is not None and metadata_scaler is not None:
        raw = metadata.iloc[indices][METADATA_FEATURES]
        values = (
            metadata_imputer.fit_transform(raw)
            if fit
            else metadata_imputer.transform(raw)
        )
        values = (
            metadata_scaler.fit_transform(values)
            if fit
            else metadata_scaler.transform(values)
        ).astype(np.float32)
        parts.append(values)
    return np.concatenate(parts, axis=1)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(ROOT / "challenge1" / "training" / "metadata.csv")
    targets = np.load(
        ROOT / "challenge1" / "training" / "meg_110ms.npy", mmap_mode="r"
    )
    features = np.load(args.feature_file, mmap_mode="r")
    if features.ndim != 2 or len(features) != len(metadata):
        raise ValueError(
            f"Feature shape {features.shape} does not align with {len(metadata)} rows"
        )
    if targets.shape != (len(metadata), 204):
        raise ValueError(f"Unexpected MEG shape: {targets.shape}")

    if args.valid_mask:
        valid_mask = np.asarray(np.load(args.valid_mask), dtype=bool)
    else:
        valid_mask = np.any(features != 0, axis=1)
    if valid_mask.shape != (len(metadata),):
        raise ValueError("Validity mask does not align with metadata")
    incomplete_valid = 0
    for start in range(0, len(features), 4096):
        stop = min(start + 4096, len(features))
        nonzero = np.any(features[start:stop] != 0, axis=1)
        incomplete_valid += int((valid_mask[start:stop] & ~nonzero).sum())
    if incomplete_valid:
        raise ValueError(
            f"Feature file is incomplete: {incomplete_valid} valid rows still "
            "contain all-zero embeddings"
        )

    subjects = metadata["subject"].to_numpy()
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    rows: list[dict[str, float | int]] = []

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        image_scaler = StandardScaler(with_mean=False, copy=False)
        imputer = SimpleImputer(strategy="median") if args.include_metadata else None
        metadata_scaler = RobustScaler() if args.include_metadata else None

        print(f"\nHeld-out subject {held_out}")
        print(f"Training rows: {len(train_indices)}; validation rows: {len(validation_indices)}")
        X_train = make_design(
            features,
            metadata,
            train_indices,
            valid_mask,
            image_scaler=image_scaler,
            metadata_imputer=imputer,
            metadata_scaler=metadata_scaler,
            fit=True,
        )
        X_validation = make_design(
            features,
            metadata,
            validation_indices,
            valid_mask,
            image_scaler=image_scaler,
            metadata_imputer=imputer,
            metadata_scaler=metadata_scaler,
            fit=False,
        )
        y_train = normalize_targets_by_subject(
            targets[train_indices], subjects[train_indices]
        )
        y_validation = normalize_targets_by_subject(
            targets[validation_indices], subjects[validation_indices]
        )

        model = Ridge(alpha=args.alpha, solver="lsqr", tol=1e-4)
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation).astype(np.float32)
        mse = mean_squared_error(y_validation, prediction)
        zero_mse = mean_squared_error(y_validation, np.zeros_like(y_validation))
        correlation = mean_channel_correlation(y_validation, prediction)
        improvement = (zero_mse - mse) / zero_mse * 100
        print(
            f"normalized MSE={mse:.6f}, improvement={improvement:.3f}%, "
            f"mean channel correlation={correlation:.6f}"
        )
        rows.append(
            {
                "validation_subject": held_out,
                "training_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "alpha": args.alpha,
                "normalized_mse": float(mse),
                "relative_improvement_percent": float(improvement),
                "mean_channel_correlation": correlation,
            }
        )

    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
