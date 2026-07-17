from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    training_dir = project_root / "challenge1" / "training"
    subject60_dev_dir = (
        project_root / "challenge1" / "subject60" / "challenge1_dev"
    )
    subject60_eval_dir = (
        project_root / "challenge1" / "subject60" / "challenge1_eval"
    )

    training_metadata_path = training_dir / "metadata.csv"
    training_meg_path = training_dir / "meg_110ms.npy"
    channel_names_path = training_dir / "channel_names.txt"
    times_path = training_dir / "times.npy"

    dev_metadata_path = subject60_dev_dir / "metadata.csv"
    eval_metadata_path = subject60_eval_dir / "metadata.csv"

    required_files = [
        training_metadata_path,
        training_meg_path,
        channel_names_path,
        times_path,
        dev_metadata_path,
        eval_metadata_path,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        print("Missing files:")
        for path in missing_files:
            print(f"  - {path}")
        raise FileNotFoundError("Some required Challenge 1 files are missing.")

    training_metadata = pd.read_csv(training_metadata_path)
    dev_metadata = pd.read_csv(dev_metadata_path)
    eval_metadata = pd.read_csv(eval_metadata_path)

    # Memory mapping avoids loading the entire MEG array into RAM immediately.
    training_meg = np.load(training_meg_path, mmap_mode="r")
    times = np.load(times_path)

    with channel_names_path.open("r", encoding="utf-8") as file:
        channel_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print("\n=== Challenge 1 Training Data ===")
    print(f"Training metadata shape: {training_metadata.shape}")
    print(f"Training MEG shape:      {training_meg.shape}")
    print(f"Number of channels:      {len(channel_names)}")
    print(f"Times array shape:       {times.shape}")
    print(f"Times:                   {times}")

    print("\nTraining metadata columns:")
    for column in training_metadata.columns:
        print(f"  - {column}")

    print("\nFirst 5 training metadata rows:")
    print(training_metadata.head())

    print("\n=== Subject 60 Test Data ===")
    print(f"Development metadata shape: {dev_metadata.shape}")
    print(f"Evaluation metadata shape:  {eval_metadata.shape}")

    print("\nFirst 5 development rows:")
    print(dev_metadata.head())

    print("\n=== Alignment Checks ===")

    if len(training_metadata) == training_meg.shape[0]:
        print("✓ Training metadata rows match MEG fixations.")
    else:
        print(
            "✗ Mismatch: "
            f"{len(training_metadata)} metadata rows vs "
            f"{training_meg.shape[0]} MEG rows."
        )

    if training_meg.shape[1] == len(channel_names):
        print("✓ MEG channels match channel_names.txt.")
    else:
        print(
            "✗ Mismatch: "
            f"{training_meg.shape[1]} MEG channels vs "
            f"{len(channel_names)} channel names."
        )

    print("\n=== MEG Sample Statistics ===")

    # Inspect only a subset first for speed.
    sample_size = min(5000, training_meg.shape[0])
    meg_sample = np.asarray(training_meg[:sample_size])

    print(f"Sample size: {sample_size}")
    print(f"Minimum:     {np.nanmin(meg_sample):.6f}")
    print(f"Maximum:     {np.nanmax(meg_sample):.6f}")
    print(f"Mean:        {np.nanmean(meg_sample):.6f}")
    print(f"Std:         {np.nanstd(meg_sample):.6f}")
    print(f"NaN count:   {np.isnan(meg_sample).sum()}")

    print("\nData inspection completed successfully.")


if __name__ == "__main__":
    main()