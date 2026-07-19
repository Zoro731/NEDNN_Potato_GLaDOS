"""Evaluate gaze and incoming-saccade features with leakage-safe LOSO folds.

This experiment deliberately stays small: it tests whether fixation geometry and
eye-movement dynamics explain cross-subject MEG before another expensive visual
representation is extracted.  Optional prototype aggregation gives each repeated
scene/location/sequence group one training vote while still scoring every
validation fixation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]

GAZE_FEATURES = [
    "gaze_x_centered",
    "gaze_y_centered",
    "gaze_radius",
    "duration",
    "time_in_trial",
    "fix_sequence",
    "rms",
    "sd",
]

SACCADE_FEATURES = GAZE_FEATURES + [
    "amplitude_pre",
    "duration_pre",
    "previous_gaze_x",
    "previous_gaze_y",
    "incoming_dx",
    "incoming_dy",
    "incoming_distance_px",
    "incoming_direction_cos",
    "incoming_direction_sin",
    "inter_fixation_gap",
    "is_first_fixation",
]

FULL_FEATURES = SACCADE_FEATURES + [
    "amplitude_post",
    "duration_post",
    "fix_sequence_from_last",
    "caption_task",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO RidgeCV for fixation-aware Challenge 1 features"
    )
    parser.add_argument(
        "--feature-set",
        choices=("gaze", "saccade", "full"),
        default="saccade",
        help="Ablation to evaluate; saccade excludes outgoing/future context.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Held-out subjects to run (default: every training subject).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=np.logspace(-3, 6, 19).tolist(),
        help="Ridge candidates; selected independently for each MEG sensor.",
    )
    parser.add_argument(
        "--prototype-bin-px",
        type=float,
        default=0.0,
        help=(
            "If positive, average training rows within scene/location/sequence "
            "groups using this gaze-grid width. Validation rows are never collapsed."
        ),
    )
    parser.add_argument(
        "--prototype-sequence-bin",
        type=int,
        default=2,
        help="Number of adjacent fixation positions per prototype sequence bin.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "fixation_aware_ridge" / "loso_results.csv",
    )
    return parser.parse_args()


def add_eye_movement_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """Derive incoming movement without crossing trial or subject boundaries."""
    derived = metadata.copy()
    order = derived.sort_values(
        ["subject", "session", "trial", "fix_sequence"],
        kind="stable",
    ).index
    ordered = derived.loc[order].copy()
    group_columns = ["subject", "session", "trial"]
    grouped = ordered.groupby(group_columns, sort=False, dropna=False)

    previous_x = grouped["mean_gx"].shift(1)
    previous_y = grouped["mean_gy"].shift(1)
    previous_end = grouped["end_time"].shift(1)
    incoming_dx = ordered["mean_gx"] - previous_x
    incoming_dy = ordered["mean_gy"] - previous_y
    distance = np.hypot(incoming_dx, incoming_dy)
    safe_distance = distance.where(distance > 0)

    ordered["gaze_x_centered"] = (ordered["mean_gx"] - 512.0) / 512.0
    ordered["gaze_y_centered"] = (ordered["mean_gy"] - 384.0) / 384.0
    ordered["gaze_radius"] = np.hypot(
        ordered["gaze_x_centered"], ordered["gaze_y_centered"]
    )
    ordered["previous_gaze_x"] = (previous_x - 512.0) / 512.0
    ordered["previous_gaze_y"] = (previous_y - 384.0) / 384.0
    ordered["incoming_dx"] = incoming_dx / 512.0
    ordered["incoming_dy"] = incoming_dy / 384.0
    ordered["incoming_distance_px"] = distance
    ordered["incoming_direction_cos"] = incoming_dx / safe_distance
    ordered["incoming_direction_sin"] = incoming_dy / safe_distance
    ordered["inter_fixation_gap"] = ordered["start_time"] - previous_end
    ordered["is_first_fixation"] = previous_x.isna().astype(np.float32)

    return ordered.sort_index()


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


def score_channels(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Match the official leaderboard's per-channel Pearson aggregation."""
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    numerator = np.mean(true_centered * pred_centered, axis=0)
    denominator = pred_centered.std(axis=0) * true_centered.std(axis=0)
    correlations = np.zeros(y_true.shape[1], dtype=np.float64)
    safe = denominator > 0
    correlations[safe] = numerator[safe] / denominator[safe]
    sorted_correlations = np.sort(correlations)
    return {
        "mean_r": float(correlations.mean()),
        "median_r": float(np.median(correlations)),
        "max_r": float(correlations.max()),
        "worst10_mean_r": float(sorted_correlations[:10].mean()),
    }


def collapse_training_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    bin_px: float,
    sequence_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average similar training fixations so repeated groups count once."""
    if bin_px <= 0:
        return X, y, np.ones(len(X), dtype=np.int32)
    if sequence_bin < 1:
        raise ValueError("--prototype-sequence-bin must be at least 1")

    scene = metadata["sceneID"].fillna(-1).to_numpy()
    gaze_x_bin = np.floor(metadata["mean_gx"].fillna(-1).to_numpy() / bin_px)
    gaze_y_bin = np.floor(metadata["mean_gy"].fillna(-1).to_numpy() / bin_px)
    sequence = metadata["fix_sequence"].fillna(-1).to_numpy()
    sequence_group = np.floor(sequence / sequence_bin)
    keys = pd.MultiIndex.from_arrays(
        [scene, gaze_x_bin, gaze_y_bin, sequence_group],
        names=["scene", "gaze_x_bin", "gaze_y_bin", "sequence_bin"],
    )
    codes, unique_keys = pd.factorize(keys, sort=False)
    counts = np.bincount(codes, minlength=len(unique_keys)).astype(np.int32)

    X_sum = np.zeros((len(unique_keys), X.shape[1]), dtype=np.float64)
    y_sum = np.zeros((len(unique_keys), y.shape[1]), dtype=np.float64)
    np.add.at(X_sum, codes, X)
    np.add.at(y_sum, codes, y)
    X_mean = (X_sum / counts[:, None]).astype(np.float32)
    y_mean = (y_sum / counts[:, None]).astype(np.float32)
    return X_mean, y_mean, counts


def selected_features(feature_set: str) -> list[str]:
    return {
        "gaze": GAZE_FEATURES,
        "saccade": SACCADE_FEATURES,
        "full": FULL_FEATURES,
    }[feature_set]


def main() -> None:
    args = parse_args()
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All ridge alphas must be positive")

    metadata = pd.read_csv(ROOT / "challenge1" / "training" / "metadata.csv")
    metadata = add_eye_movement_features(metadata)
    raw_targets = np.load(
        ROOT / "challenge1" / "training" / "meg_110ms.npy", mmap_mode="r"
    )
    if raw_targets.shape != (len(metadata), 204):
        raise ValueError(
            f"Expected {(len(metadata), 204)} MEG targets, got {raw_targets.shape}"
        )

    subjects = metadata["subject"].to_numpy()
    targets = normalize_targets_by_subject(raw_targets, subjects)
    features = selected_features(args.feature_set)
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    result_rows: list[dict[str, float | int | str]] = []

    print("=" * 72)
    print("FIXATION-AWARE RIDGECV · LEAVE ONE SUBJECT OUT")
    print("=" * 72)
    print(f"Feature set: {args.feature_set} ({len(features)} columns)")
    print(f"Per-sensor alpha candidates: {len(args.alphas)}")
    print(f"Prototype gaze bin: {args.prototype_bin_px:g} px")

    for held_out in fold_subjects:
        train_mask = subjects != held_out
        validation_mask = subjects == held_out
        train_metadata = metadata.loc[train_mask]
        validation_metadata = metadata.loc[validation_mask]

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_train = imputer.fit_transform(train_metadata[features]).astype(np.float32)
        X_validation = imputer.transform(validation_metadata[features]).astype(
            np.float32
        )
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_validation = scaler.transform(X_validation).astype(np.float32)
        y_train = targets[train_mask]
        y_validation = targets[validation_mask]

        model_X, model_y, prototype_counts = collapse_training_prototypes(
            X_train,
            y_train,
            train_metadata,
            args.prototype_bin_px,
            args.prototype_sequence_bin,
        )

        print(f"\nHeld-out subject {held_out}")
        print(
            f"Rows: {len(X_train)} train -> {len(model_X)} model rows; "
            f"{len(X_validation)} validation"
        )
        if args.prototype_bin_px > 0:
            print(
                "Prototype size: "
                f"mean={prototype_counts.mean():.2f}, "
                f"max={prototype_counts.max()}, "
                f"repeated={(prototype_counts > 1).sum()} groups"
            )

        model = RidgeCV(
            alphas=np.asarray(args.alphas, dtype=np.float64),
            alpha_per_target=True,
            gcv_mode="svd",
        )
        model.fit(model_X, model_y)
        prediction = model.predict(X_validation).astype(np.float32)
        normalized_mse = mean_squared_error(y_validation, prediction)
        metrics = score_channels(y_validation, prediction)
        selected_alpha = np.atleast_1d(model.alpha_).astype(np.float64)

        print(
            f"normalized MSE={normalized_mse:.6f}; "
            f"mean r={metrics['mean_r']:.6f}; "
            f"median r={metrics['median_r']:.6f}"
        )
        print(
            "selected alpha: "
            f"min={selected_alpha.min():g}, "
            f"median={np.median(selected_alpha):g}, "
            f"max={selected_alpha.max():g}"
        )

        result_rows.append(
            {
                "validation_subject": held_out,
                "feature_set": args.feature_set,
                "prototype_bin_px": args.prototype_bin_px,
                "training_rows": len(X_train),
                "model_rows": len(model_X),
                "validation_rows": len(X_validation),
                "normalized_mse": float(normalized_mse),
                **metrics,
                "alpha_min": float(selected_alpha.min()),
                "alpha_median": float(np.median(selected_alpha)),
                "alpha_max": float(selected_alpha.max()),
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
