from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    project_root = Path(__file__).resolve().parents[1]

    metadata = pd.read_csv(
        project_root / "challenge1" / "training" / "metadata.csv"
    )

    meg = np.load(
        project_root / "challenge1" / "training" / "meg_110ms.npy",
        mmap_mode="r",
    )

    output_dir = project_root / "results" / "meg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MEG DATASET")
    print("=" * 60)

    print(f"Shape: {meg.shape}")
    print(f"Number of fixations: {meg.shape[0]}")
    print(f"Number of channels: {meg.shape[1]}")

    # -------------------------------------------------
    # Channel statistics
    # -------------------------------------------------

    mean_channel = np.mean(meg, axis=0)
    std_channel = np.std(meg, axis=0)

    print("\nOverall MEG statistics")
    print(f"Global mean : {np.mean(meg):.6e}")
    print(f"Global std  : {np.std(meg):.6e}")
    print(f"Minimum     : {np.min(meg):.6e}")
    print(f"Maximum     : {np.max(meg):.6e}")

    # Mean response

    plt.figure(figsize=(12,4))
    plt.plot(mean_channel)
    plt.title("Mean MEG response per channel")
    plt.xlabel("Channel")
    plt.ylabel("Mean")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_channel_response.png")
    plt.close()

    # Standard deviation

    plt.figure(figsize=(12,4))
    plt.plot(std_channel)
    plt.title("Channel standard deviation")
    plt.xlabel("Channel")
    plt.ylabel("Standard deviation")
    plt.tight_layout()
    plt.savefig(output_dir / "channel_standard_deviation.png")
    plt.close()

    # -------------------------------------------------
    # Random fixation responses
    # -------------------------------------------------

    rng = np.random.default_rng(42)

    plt.figure(figsize=(12,6))

    for i in rng.choice(len(meg), size=5, replace=False):
        plt.plot(meg[i], alpha=0.8)

    plt.title("Example fixation responses")
    plt.xlabel("Channel")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_dir / "example_fixation_responses.png")
    plt.close()

    # -------------------------------------------------
    # Subject statistics
    # -------------------------------------------------

    print("\nSubject statistics")

    rows = []

    for subject in sorted(metadata.subject.unique()):

        mask = metadata.subject == subject

        data = np.asarray(meg[mask])

        rows.append({
            "subject": subject,
            "mean": data.mean(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "fixations": len(data),
        })

        print(
            f"Subject {subject}: "
            f"mean={data.mean():.3e}, "
            f"std={data.std():.3e}"
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "subject_statistics.csv",
        index=False,
    )

    # -------------------------------------------------
    # PCA
    # -------------------------------------------------

    print("\nRunning PCA...")

    sample = np.asarray(meg[:50000])

    pca = PCA()
    pca.fit(sample)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8,5))
    plt.plot(cumulative)
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance.png")
    plt.close()

    for threshold in [0.80,0.90,0.95,0.99]:
        n = np.argmax(cumulative >= threshold)+1
        print(f"{threshold:.0%} variance -> {n} components")

    print("\nResults saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()