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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    metadata_path = (
        project_root / "challenge1" / "training" / "metadata.csv"
    )
    meg_path = (
        project_root / "challenge1" / "training" / "meg_110ms.npy"
    )

    metadata = pd.read_csv(metadata_path)
    meg = np.load(meg_path, mmap_mode="r")

    train_subjects = [1, 2, 3, 4]
    validation_subject = 5

    train_mask = metadata["subject"].isin(train_subjects).to_numpy()
    validation_mask = (
        metadata["subject"] == validation_subject
    ).to_numpy()

    X_train = metadata.loc[
        train_mask,
        NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    ].copy()

    X_validation = metadata.loc[
        validation_mask,
        NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    ].copy()

    y_train = np.asarray(meg[train_mask], dtype=np.float32)
    y_validation = np.asarray(meg[validation_mask], dtype=np.float32)

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

    model = Pipeline(
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

    print("Training metadata-only Ridge baseline...")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Target shape:       {y_train.shape}")

    model.fit(X_train, y_train)

    print("Generating validation predictions...")
    predictions = model.predict(X_validation).astype(np.float32)

    model_mse = mean_squared_error(
        y_validation,
        predictions,
    )

    mean_prediction = np.repeat(
        y_train.mean(axis=0, keepdims=True),
        repeats=len(y_validation),
        axis=0,
    )

    mean_baseline_mse = mean_squared_error(
        y_validation,
        mean_prediction,
    )

    print("\n=== Validation results: subject 5 ===")
    print(f"Mean baseline MSE:     {mean_baseline_mse:.6e}")
    print(f"Metadata Ridge MSE:    {model_mse:.6e}")

    improvement = (
        mean_baseline_mse - model_mse
    ) / mean_baseline_mse * 100

    print(f"Relative improvement:  {improvement:.2f}%")
    print(f"Prediction shape:      {predictions.shape}")
    print(f"Prediction mean:       {predictions.mean():.6e}")
    print(f"Prediction std:        {predictions.std():.6e}")


if __name__ == "__main__":
    main()