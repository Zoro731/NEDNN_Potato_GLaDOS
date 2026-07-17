from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    metadata_path = root / "challenge1" / "training" / "metadata.csv"
    meg_path = root / "challenge1" / "training" / "meg_110ms.npy"

    output_dir = root / "results" / "normalized_meg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    meg = np.load(meg_path, mmap_mode="r")

    normalized_meg = np.empty(meg.shape, dtype=np.float32)

    subject_rows = []

    print("=" * 60)
    print("SUBJECT-WISE MEG NORMALIZATION")
    print("=" * 60)

    for subject in sorted(metadata["subject"].unique()):
        mask = metadata["subject"].to_numpy() == subject
        subject_data = np.asarray(meg[mask], dtype=np.float32)

        # Compute one mean and standard deviation for each channel
        channel_mean = subject_data.mean(axis=0)
        channel_std = subject_data.std(axis=0)

        # Avoid division by zero
        channel_std[channel_std < 1e-20] = 1.0

        subject_normalized = (
            subject_data - channel_mean
        ) / channel_std

        normalized_meg[mask] = subject_normalized

        subject_rows.append(
            {
                "subject": subject,
                "raw_mean": float(subject_data.mean()),
                "raw_std": float(subject_data.std()),
                "normalized_mean": float(subject_normalized.mean()),
                "normalized_std": float(subject_normalized.std()),
                "fixations": int(mask.sum()),
            }
        )

        print(
            f"Subject {subject}: "
            f"raw std={subject_data.std():.3e}, "
            f"normalized mean={subject_normalized.mean():.3e}, "
            f"normalized std={subject_normalized.std():.3f}"
        )

    stats = pd.DataFrame(subject_rows)

    stats.to_csv(
        output_dir / "subject_normalization_statistics.csv",
        index=False,
    )

    print("\nNormalized global statistics")
    print(f"Mean: {normalized_meg.mean():.6e}")
    print(f"Std:  {normalized_meg.std():.6f}")
    print(f"Min:  {normalized_meg.min():.6f}")
    print(f"Max:  {normalized_meg.max():.6f}")

    # Plot normalized channel means
    normalized_channel_mean = normalized_meg.mean(axis=0)
    normalized_channel_std = normalized_meg.std(axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(normalized_channel_mean)
    plt.xlabel("Channel index")
    plt.ylabel("Normalized mean")
    plt.title("Mean normalized MEG response per channel")
    plt.tight_layout()
    plt.savefig(
        output_dir / "normalized_channel_mean.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(normalized_channel_std)
    plt.xlabel("Channel index")
    plt.ylabel("Normalized standard deviation")
    plt.title("Standard deviation after subject-wise normalization")
    plt.tight_layout()
    plt.savefig(
        output_dir / "normalized_channel_std.png",
        dpi=150,
    )
    plt.close()

    # Use a balanced sample from each subject for PCA
    rng = np.random.default_rng(42)
    samples = []

    sample_per_subject = 10000

    for subject in sorted(metadata["subject"].unique()):
        subject_indices = np.flatnonzero(
            metadata["subject"].to_numpy() == subject
        )

        selected = rng.choice(
            subject_indices,
            size=min(sample_per_subject, len(subject_indices)),
            replace=False,
        )

        samples.append(normalized_meg[selected])

    pca_sample = np.concatenate(samples, axis=0)

    print(f"\nBalanced PCA sample shape: {pca_sample.shape}")
    print("Running PCA on normalized MEG...")

    pca = PCA()
    pca.fit(pca_sample)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    plt.figure(figsize=(9, 5))
    plt.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
    )
    plt.axhline(0.80, linestyle="--")
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.axhline(0.99, linestyle="--")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA after subject-wise channel normalization")
    plt.tight_layout()
    plt.savefig(
        output_dir / "normalized_pca_explained_variance.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        np.arange(1, 31),
        explained[:30],
        marker="o",
    )
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Variance explained by the first 30 components")
    plt.tight_layout()
    plt.savefig(
        output_dir / "first_30_pca_components.png",
        dpi=150,
    )
    plt.close()

    print("\nPCA results after normalization")

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        components = int(np.searchsorted(cumulative, threshold) + 1)

        print(
            f"{threshold:.0%} variance -> "
            f"{components} components"
        )

    print("\nFirst 10 component variance ratios:")

    for index, value in enumerate(explained[:10], start=1):
        print(f"PC {index:2d}: {value:.4%}")

    print(f"\nResults saved to:\n{output_dir}")


if __name__ == "__main__":
    main()