from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "session",
    "caption_task",
    "multi_saccade",
]


def mean_channel_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    correlations = []

    for channel_index in range(y_true.shape[1]):
        true_channel = y_true[:, channel_index]
        predicted_channel = y_pred[:, channel_index]

        true_std = np.std(true_channel)
        predicted_std = np.std(predicted_channel)

        if true_std == 0 or predicted_std == 0:
            continue

        correlation = np.corrcoef(
            true_channel,
            predicted_channel,
        )[0, 1]

        if np.isfinite(correlation):
            correlations.append(correlation)

    if not correlations:
        return float("nan")

    return float(np.mean(correlations))


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "ridge",
                Ridge(alpha=100.0),
            ),
        ]
    )


def run_fold(
    metadata: pd.DataFrame,
    meg: np.ndarray,
    validation_subject: int,
) -> dict:
    train_mask = (
        metadata["subject"] != validation_subject
    ).to_numpy()

    validation_mask = (
        metadata["subject"] == validation_subject
    ).to_numpy()

    selected_features = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    X_train = metadata.loc[
        train_mask,
        selected_features,
    ].copy()

    X_validation = metadata.loc[
        validation_mask,
        selected_features,
    ].copy()

    y_train = np.asarray(
        meg[train_mask],
        dtype=np.float32,
    )

    y_validation = np.asarray(
        meg[validation_mask],
        dtype=np.float32,
    )

    model = build_model()

    print(
        f"\nTraining fold with subject "
        f"{validation_subject} held out..."
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")

    model.fit(X_train, y_train)

    predictions = model.predict(
        X_validation
    ).astype(np.float32)

    global_mean = y_train.mean(
        axis=0,
        keepdims=True,
    )

    mean_predictions = np.repeat(
        global_mean,
        repeats=len(y_validation),
        axis=0,
    )

    mean_mse = mean_squared_error(
        y_validation,
        mean_predictions,
    )

    model_mse = mean_squared_error(
        y_validation,
        predictions,
    )

    improvement = (
        (mean_mse - model_mse)
        / mean_mse
        * 100
    )

    correlation = mean_channel_correlation(
        y_validation,
        predictions,
    )

    print(f"Mean baseline MSE: {mean_mse:.6e}")
    print(f"Model MSE:         {model_mse:.6e}")
    print(f"Improvement:       {improvement:.4f}%")
    print(f"Mean correlation:  {correlation:.6f}")

    return {
        "validation_subject": validation_subject,
        "training_samples": len(X_train),
        "validation_samples": len(X_validation),
        "mean_baseline_mse": mean_mse,
        "model_mse": model_mse,
        "relative_improvement_percent": improvement,
        "mean_channel_correlation": correlation,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    metadata_path = (
        root
        / "challenge1"
        / "training"
        / "metadata.csv"
    )

    meg_path = (
        root
        / "challenge1"
        / "training"
        / "meg_110ms.npy"
    )

    output_dir = (
        root
        / "results"
        / "loso_metadata_baseline"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        metadata_path
    )

    meg = np.load(
        meg_path,
        mmap_mode="r",
    )

    subjects = sorted(
        metadata["subject"].unique()
    )

    print("=" * 60)
    print("LEAVE-ONE-SUBJECT-OUT EVALUATION")
    print("=" * 60)

    print(f"Subjects: {subjects}")
    print(f"MEG shape: {meg.shape}")

    fold_results = []

    for validation_subject in subjects:
        result = run_fold(
            metadata=metadata,
            meg=meg,
            validation_subject=int(
                validation_subject
            ),
        )

        fold_results.append(result)

    results_df = pd.DataFrame(
        fold_results
    )

    results_path = (
        output_dir
        / "loso_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("LOSO SUMMARY")
    print("=" * 60)

    print(
        results_df[
            [
                "validation_subject",
                "relative_improvement_percent",
                "mean_channel_correlation",
            ]
        ].to_string(index=False)
    )

    print("\nAverage results")

    print(
        "Mean relative improvement: "
        f"{results_df['relative_improvement_percent'].mean():.4f}%"
    )

    print(
        "Mean channel correlation:  "
        f"{results_df['mean_channel_correlation'].mean():.6f}"
    )

    print(
        "Mean model MSE:             "
        f"{results_df['model_mse'].mean():.6e}"
    )

    print(f"\nResults saved to:\n{results_path}")


if __name__ == "__main__":
    main()