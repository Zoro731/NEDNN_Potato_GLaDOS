from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    training_path = project_root / "challenge1" / "training" / "metadata.csv"
    dev_path = (
        project_root
        / "challenge1"
        / "subject60"
        / "challenge1_dev"
        / "metadata.csv"
    )

    output_dir = project_root / "results" / "metadata_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(training_path)
    dev = pd.read_csv(dev_path)

    print("\n=== Subjects ===")
    print(train["subject"].value_counts().sort_index())

    print("\n=== Missing-value percentages ===")
    missing = train.isna().mean().sort_values(ascending=False) * 100
    print(missing.to_string())

    print("\n=== Important feature summary ===")

    important_columns = [
        "duration",
        "mean_gx",
        "mean_gy",
        "time_in_trial",
        "fix_sequence",
        "amplitude_pre",
        "amplitude_post",
        "duration_pre",
        "duration_post",
    ]

    available_columns = [
        column for column in important_columns if column in train.columns
    ]

    print(train[available_columns].describe().T)

    print("\n=== Unique values ===")
    for column in ["subject", "session", "sceneID", "trial", "caption_task"]:
        if column in train.columns:
            print(f"{column}: {train[column].nunique()} unique values")

    # Fixation position scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(
        train["mean_gx"].iloc[::20],
        train["mean_gy"].iloc[::20],
        s=4,
        alpha=0.25,
    )
    plt.xlabel("Mean gaze x")
    plt.ylabel("Mean gaze y")
    plt.title("Training fixation locations")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "training_fixation_locations.png", dpi=150)
    plt.close()

    # Fixation duration
    plt.figure(figsize=(8, 5))
    plt.hist(
        train["duration"].dropna(),
        bins=80,
    )
    plt.xlabel("Fixation duration (seconds)")
    plt.ylabel("Count")
    plt.title("Training fixation duration distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "fixation_duration_distribution.png", dpi=150)
    plt.close()

    # Time in trial
    plt.figure(figsize=(8, 5))
    plt.hist(
        train["time_in_trial"].dropna(),
        bins=80,
    )
    plt.xlabel("Time in trial (seconds)")
    plt.ylabel("Count")
    plt.title("Time-in-trial distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "time_in_trial_distribution.png", dpi=150)
    plt.close()

    # Number of fixations per subject
    subject_counts = train["subject"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    subject_counts.plot(kind="bar")
    plt.xlabel("Subject")
    plt.ylabel("Number of fixations")
    plt.title("Training fixations per subject")
    plt.tight_layout()
    plt.savefig(output_dir / "fixations_per_subject.png", dpi=150)
    plt.close()

    # Compare training and subject 60 gaze distributions
    plt.figure(figsize=(8, 5))
    plt.hist(
        train["mean_gx"].dropna(),
        bins=60,
        alpha=0.5,
        density=True,
        label="Training subjects",
    )
    plt.hist(
        dev["mean_gx"].dropna(),
        bins=60,
        alpha=0.5,
        density=True,
        label="Subject 60 development",
    )
    plt.xlabel("Mean gaze x")
    plt.ylabel("Density")
    plt.title("Horizontal gaze-position comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_vs_subject60_gaze_x.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()