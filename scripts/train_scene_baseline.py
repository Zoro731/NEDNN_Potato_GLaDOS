from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def mean_channel_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    correlations = []

    for channel in range(y_true.shape[1]):
        true_channel = y_true[:, channel]
        pred_channel = y_pred[:, channel]

        if np.std(true_channel) == 0 or np.std(pred_channel) == 0:
            continue

        correlation = np.corrcoef(
            true_channel,
            pred_channel,
        )[0, 1]

        if np.isfinite(correlation):
            correlations.append(correlation)

    return float(np.mean(correlations))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    metadata_path = (
        project_root
        / "challenge1"
        / "training"
        / "metadata.csv"
    )

    meg_path = (
        project_root
        / "challenge1"
        / "training"
        / "meg_110ms.npy"
    )

    metadata = pd.read_csv(metadata_path)
    meg = np.load(meg_path, mmap_mode="r")

    train_mask = metadata["subject"].isin([1, 2, 3, 4]).to_numpy()
    validation_mask = (metadata["subject"] == 5).to_numpy()

    train_metadata = metadata.loc[train_mask].copy()
    validation_metadata = metadata.loc[validation_mask].copy()

    y_train = np.asarray(meg[train_mask], dtype=np.float32)
    y_validation = np.asarray(meg[validation_mask], dtype=np.float32)

    print("Computing scene-level MEG averages...")

    global_mean = y_train.mean(axis=0)

    scene_sums: dict[int, np.ndarray] = {}
    scene_counts: dict[int, int] = {}

    train_scene_ids = train_metadata["sceneID"].to_numpy()

    for scene_id, response in zip(train_scene_ids, y_train):
        scene_id = int(scene_id)

        if scene_id not in scene_sums:
            scene_sums[scene_id] = np.zeros(
                y_train.shape[1],
                dtype=np.float64,
            )
            scene_counts[scene_id] = 0

        scene_sums[scene_id] += response
        scene_counts[scene_id] += 1

    scene_means = {
        scene_id: (
            scene_sums[scene_id] / scene_counts[scene_id]
        ).astype(np.float32)
        for scene_id in scene_sums
    }

    validation_scene_ids = validation_metadata["sceneID"].to_numpy()

    predictions = np.stack(
        [
            scene_means.get(int(scene_id), global_mean)
            for scene_id in validation_scene_ids
        ]
    ).astype(np.float32)

    mean_predictions = np.repeat(
        global_mean[None, :],
        len(y_validation),
        axis=0,
    )

    mean_mse = mean_squared_error(
        y_validation,
        mean_predictions,
    )

    scene_mse = mean_squared_error(
        y_validation,
        predictions,
    )

    scene_correlation = mean_channel_correlation(
        y_validation,
        predictions,
    )

    improvement = (
        (mean_mse - scene_mse)
        / mean_mse
        * 100
    )

    covered_scenes = sum(
        int(scene_id) in scene_means
        for scene_id in validation_scene_ids
    )

    print("\n=== Scene baseline: subject 5 ===")
    print(f"Training scenes:          {len(scene_means)}")
    print(
        f"Validation rows covered: {covered_scenes}"
        f"/{len(validation_scene_ids)}"
    )
    print(f"Mean baseline MSE:        {mean_mse:.6e}")
    print(f"Scene baseline MSE:       {scene_mse:.6e}")
    print(f"Relative improvement:     {improvement:.3f}%")
    print(f"Mean channel correlation: {scene_correlation:.6f}")
    print(f"Prediction shape:         {predictions.shape}")


if __name__ == "__main__":
    main()