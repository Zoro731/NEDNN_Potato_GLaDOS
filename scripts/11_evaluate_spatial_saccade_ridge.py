"""Evaluate spatial crop features plus incoming-saccade predictors with LOSO."""

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
        description="LOSO RidgeCV for spatial crop features plus saccades"
    )
    parser.add_argument(
        "feature_file",
        type=Path,
        nargs="?",
        default=Path("challenge1/training/spatial_features.npy"),
    )
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Training-fold PCA width applied only to valid visual rows.",
    )
    parser.add_argument(
        "--visual-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to PCA visual features before concatenation.",
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
        default=ROOT / "results" / "spatial_saccade_ridge" / "loso_results.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.pca_components < 1
        or args.visual_scale < 0
        or any(alpha <= 0 for alpha in args.alphas)
    ):
        raise ValueError("PCA components, visual scale, and alphas must be valid")

    metadata = FIXATION.add_eye_movement_features(
        pd.read_csv(ROOT / "challenge1" / "training" / "metadata.csv")
    )
    raw_targets = np.load(
        ROOT / "challenge1" / "training" / "meg_110ms.npy", mmap_mode="r"
    )
    features = np.load(args.feature_file.resolve(), mmap_mode="r")
    if features.ndim != 2 or len(features) != len(metadata):
        raise ValueError(
            f"Feature shape {features.shape} does not align with {len(metadata)} rows"
        )
    if args.pca_components > features.shape[1]:
        raise ValueError(
            f"Requested {args.pca_components} PCA components, but features have "
            f"width {features.shape[1]}"
        )
    if raw_targets.shape != (len(metadata), 204):
        raise ValueError(f"Unexpected MEG shape: {raw_targets.shape}")

    if args.valid_mask:
        valid_mask = np.asarray(np.load(args.valid_mask.resolve()), dtype=bool)
    else:
        valid_mask = np.any(features != 0, axis=1)
    if valid_mask.shape != (len(metadata),):
        raise ValueError("Validity mask does not align with metadata")

    subjects = metadata["subject"].to_numpy()
    targets = FIXATION.normalize_targets_by_subject(raw_targets, subjects)
    fold_subjects = args.subjects or sorted(int(s) for s in np.unique(subjects))
    result_rows: list[dict[str, float | int]] = []

    print("=" * 72)
    print("SPATIAL VISUAL + INCOMING SACCADE RIDGECV · LEAVE ONE SUBJECT OUT")
    print("=" * 72)
    print(f"Feature file: {args.feature_file.resolve()}")
    print(f"Raw visual width: {features.shape[1]}")
    print(f"PCA components: {args.pca_components}")
    print(f"Visual branch scale: {args.visual_scale:g}")

    for held_out in fold_subjects:
        train_indices = np.flatnonzero(subjects != held_out)
        validation_indices = np.flatnonzero(subjects == held_out)
        train_valid = valid_mask[train_indices]
        if not train_valid.any():
            raise ValueError(f"No valid visual rows in training fold {held_out}")

        visual_scaler = StandardScaler()
        visual_train_valid = np.asarray(
            features[train_indices[train_valid]], dtype=np.float32
        )
        visual_scaler.fit(visual_train_valid)
        pca = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=0,
        )
        pca.fit(visual_scaler.transform(visual_train_valid))

        visual_train = pca.transform(
            visual_scaler.transform(
                np.asarray(features[train_indices], dtype=np.float32)
            )
        ).astype(np.float32)
        visual_validation = pca.transform(
            visual_scaler.transform(
                np.asarray(features[validation_indices], dtype=np.float32)
            )
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
            [visual_train, eye_train, train_valid[:, None].astype(np.float32)],
            axis=1,
        )
        X_validation = np.concatenate(
            [
                visual_validation,
                eye_validation,
                valid_mask[validation_indices, None].astype(np.float32),
            ],
            axis=1,
        )

        model = RidgeCV(
            alphas=np.asarray(args.alphas, dtype=np.float64),
            alpha_per_target=True,
            gcv_mode="svd",
        )
        model.fit(X_train, targets[train_indices])
        prediction = model.predict(X_validation).astype(np.float32)
        normalized_mse = mean_squared_error(targets[validation_indices], prediction)
        metrics = FIXATION.score_channels(targets[validation_indices], prediction)
        selected_alpha = np.atleast_1d(model.alpha_).astype(np.float64)

        print(f"\nHeld-out subject {held_out}")
        print(
            f"Rows: {len(train_indices)} train / {len(validation_indices)} validation; "
            f"valid visual train rows={int(train_valid.sum())}"
        )
        print(
            f"PCA retained {pca.n_components_} components; "
            f"explained variance={pca.explained_variance_ratio_.sum():.3f}"
        )
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
                "pca_components": int(pca.n_components_),
                "visual_scale": args.visual_scale,
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "valid_visual_training_rows": int(train_valid.sum()),
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
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
