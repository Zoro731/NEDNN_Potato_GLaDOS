"""Evaluate leakage-safe cross-subject fixation prototypes for Challenge 2.

For each LOSO fold, prototypes are built only from the four training subjects.
The primary key is ``sceneID + fix_sequence``.  If that key is unavailable in
the training subjects, the predictor falls back to the mean response for the
same scene, then to the training-fold global mean.  A ridge prediction is also
fit from the existing spatial+saccade features, and fixed prototype/ridge
blends are reported for later model selection.
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
EVALUATOR = load_module(
    "14_evaluate_challenge2_spatial_saccade.py", "challenge2_evaluator"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO Challenge 2 fixation prototypes and ridge blends"
    )
    parser.add_argument(
        "--metadata-csv", type=Path,
        default=Path("data/brainencoding26/challenge2/training/metadata.csv"),
    )
    parser.add_argument(
        "--target-file", type=Path,
        default=Path("data/brainencoding26/challenge2/training/meg_c2.npy"),
    )
    parser.add_argument(
        "--times-file", type=Path,
        default=Path("data/brainencoding26/challenge2/training/times.npy"),
    )
    parser.add_argument(
        "--feature-file", type=Path,
        default=Path("challenge2/training/spatial_features.npy"),
    )
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--visual-scale", type=float, default=0.05)
    parser.add_argument(
        "--time-ms", type=float, nargs=6,
        default=[-50.0, 50.0, 75.0, 100.0, 125.0, 150.0],
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=np.logspace(-3, 6, 19).tolist(),
    )
    parser.add_argument(
        "--blend-weights", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Prototype weight; 0 is ridge-only and 1 is prototype-only.",
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument(
        "--output-csv", type=Path,
        default=ROOT / "results" / "challenge2" / "fixation_prototype_loso.csv",
    )
    return parser.parse_args()


def composite_key(metadata: pd.DataFrame, include_fixation: bool) -> np.ndarray:
    scene = pd.to_numeric(metadata["sceneID"], errors="coerce").fillna(-1)
    scene = np.rint(scene).astype(np.int64).to_numpy()
    if not include_fixation:
        return scene
    fixation = pd.to_numeric(metadata["fix_sequence"], errors="coerce").fillna(-1)
    fixation = np.rint(fixation).astype(np.int64).to_numpy()
    # The dataset's fixation sequence is small; this keeps scene and sequence
    # keys collision-free without converting 180k rows to Python strings.
    return scene * 1000 + fixation


def build_prototypes(keys: np.ndarray, targets: np.ndarray):
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(unique_keys),) + targets.shape[1:], dtype=np.float32)
    np.add.at(sums, inverse, targets)
    counts = np.bincount(inverse).astype(np.float32)
    sums /= counts.reshape((-1,) + (1,) * (targets.ndim - 1))
    lookup = {int(key): index for index, key in enumerate(unique_keys)}
    return lookup, sums


def prototype_predict(
    train_targets: np.ndarray,
    train_exact_keys: np.ndarray,
    train_scene_keys: np.ndarray,
    validation_exact_keys: np.ndarray,
    validation_scene_keys: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    exact_lookup, exact_values = build_prototypes(train_exact_keys, train_targets)
    scene_lookup, scene_values = build_prototypes(train_scene_keys, train_targets)
    global_mean = train_targets.mean(axis=0, dtype=np.float32)

    prediction = np.empty(
        (len(validation_exact_keys),) + train_targets.shape[1:], dtype=np.float32
    )
    counts = {"exact": 0, "scene": 0, "global": 0}
    for row, (exact_key, scene_key) in enumerate(
        zip(validation_exact_keys, validation_scene_keys)
    ):
        exact_index = exact_lookup.get(int(exact_key))
        if exact_index is not None:
            prediction[row] = exact_values[exact_index]
            counts["exact"] += 1
            continue
        scene_index = scene_lookup.get(int(scene_key))
        if scene_index is not None:
            prediction[row] = scene_values[scene_index]
            counts["scene"] += 1
            continue
        prediction[row] = global_mean
        counts["global"] += 1
    return prediction, counts


def fit_ridge_predictions(
    metadata: pd.DataFrame,
    features: np.ndarray,
    valid_mask: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_valid = valid_mask[train_indices]
    visual_scaler = StandardScaler()
    visual_train_valid = np.asarray(features[train_indices[train_valid]], dtype=np.float32)
    visual_scaler.fit(visual_train_valid)
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=0)
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
    eye_train = eye_scaler.fit_transform(
        eye_imputer.fit_transform(metadata.iloc[train_indices][FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)
    eye_validation = eye_scaler.transform(
        eye_imputer.transform(metadata.iloc[validation_indices][FIXATION.SACCADE_FEATURES])
    ).astype(np.float32)
    x_train = np.concatenate(
        [visual_train, eye_train, train_valid[:, None].astype(np.float32)], axis=1
    )
    x_validation = np.concatenate(
        [visual_validation, eye_validation, valid_mask[validation_indices, None].astype(np.float32)],
        axis=1,
    )
    target_width = targets.shape[1] * targets.shape[2]
    y_train = targets[train_indices].transpose(0, 2, 1).reshape(-1, target_width)
    model = RidgeCV(
        alphas=np.asarray(args.alphas, dtype=np.float64),
        alpha_per_target=True,
        gcv_mode="svd",
    )
    model.fit(x_train, y_train)
    ridge = model.predict(x_validation).reshape(
        len(validation_indices), targets.shape[2], targets.shape[1]
    ).transpose(0, 2, 1).astype(np.float32)
    return ridge, model.alpha_, float(pca.explained_variance_ratio_.sum())


def main() -> None:
    args = parse_args()
    if args.pca_components < 1 or args.visual_scale < 0:
        raise ValueError("PCA components and visual scale must be valid")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All Ridge alphas must be positive")
    if any(weight < 0 or weight > 1 for weight in args.blend_weights):
        raise ValueError("Blend weights must be between 0 and 1")

    metadata = pd.read_csv(args.metadata_csv.resolve())
    metadata = FIXATION.add_eye_movement_features(metadata)
    subjects = metadata["subject"].to_numpy()
    raw_targets = np.load(args.target_file.resolve(), mmap_mode="r")
    times = np.load(args.times_file.resolve())
    time_indices, actual_ms = EVALUATOR.select_time_indices(times, args.time_ms)
    if raw_targets.shape[:2] != (len(metadata), 204):
        raise ValueError(f"Unexpected target shape: {raw_targets.shape}")
    targets = EVALUATOR.normalize_targets_by_subject_time(raw_targets, subjects, time_indices)
    features = np.load(args.feature_file.resolve(), mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(metadata):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    valid_mask = (
        np.asarray(np.load(args.valid_mask.resolve()), dtype=bool)
        if args.valid_mask else np.any(features != 0, axis=1)
    )
    if valid_mask.shape != (len(metadata),):
        raise ValueError("Validity mask does not align with metadata")

    exact_keys = composite_key(metadata, include_fixation=True)
    scene_keys = composite_key(metadata, include_fixation=False)
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    result_rows: list[dict[str, float | int]] = []

    print("=" * 72)
    print("CHALLENGE 2 FIXATION PROTOTYPES + RIDGE BLENDS · LOSO")
    print("=" * 72)
    print(f"Prototype key: sceneID + fix_sequence; fallback: same scene, then global")
    print(f"Blend weights: {list(args.blend_weights)}")
    print(f"Requested times (ms): {list(args.time_ms)}")
    print(f"Actual times (ms): {actual_ms.round(3).tolist()}")

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        train_valid = valid_mask[train_indices]
        ridge, alpha, pca_variance = fit_ridge_predictions(
            metadata, features, valid_mask, targets, train_indices, validation_indices, args
        )
        prototypes, match_counts = prototype_predict(
            targets[train_indices],
            exact_keys[train_indices],
            scene_keys[train_indices],
            exact_keys[validation_indices],
            scene_keys[validation_indices],
        )
        truth = targets[validation_indices]
        print(f"\nHeld-out subject {held_out}")
        print(
            f"Rows: {len(train_indices)} train / {len(validation_indices)} validation; "
            f"matches exact={match_counts['exact']}, scene={match_counts['scene']}, "
            f"global={match_counts['global']}"
        )
        print(f"PCA variance={pca_variance:.3f}; visual scale={args.visual_scale:g}")
        for weight in args.blend_weights:
            prediction = (1.0 - weight) * ridge + weight * prototypes
            for time_position, requested_ms in enumerate(args.time_ms):
                metrics = FIXATION.score_channels(
                    truth[:, :, time_position], prediction[:, :, time_position]
                )
                normalized_mse = mean_squared_error(
                    truth[:, :, time_position], prediction[:, :, time_position]
                )
                result_rows.append(
                    {
                        "validation_subject": held_out,
                        "prototype_weight": float(weight),
                        "ridge_weight": float(1.0 - weight),
                        "requested_time_ms": float(requested_ms),
                        "actual_time_ms": float(actual_ms[time_position]),
                        "time_index": int(time_indices[time_position]),
                        "pca_components": int(args.pca_components),
                        "visual_scale": float(args.visual_scale),
                        "explained_variance": pca_variance,
                        "training_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        "match_exact_rows": match_counts["exact"],
                        "match_scene_rows": match_counts["scene"],
                        "match_global_rows": match_counts["global"],
                        "normalized_mse": float(normalized_mse),
                        **metrics,
                    }
                )
            summary_r = np.mean(
                [row["mean_r"] for row in result_rows[-len(args.time_ms):]]
            )
            print(f"  prototype weight={weight:g}: mean r={summary_r:.6f}")

    result = pd.DataFrame(result_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary = result.groupby("prototype_weight", as_index=False)[
        ["mean_r", "normalized_mse"]
    ].mean()
    print("\n" + "=" * 72)
    print(summary.to_string(index=False))
    best = summary.sort_values("mean_r", ascending=False).iloc[0]
    print(
        f"Best LOSO prototype weight by mean r={best.prototype_weight:g}: "
        f"r={best.mean_r:.6f}; MSE={best.normalized_mse:.6f}"
    )
    print(f"Saved results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
