"""Preview DINOv2 crop embeddings as an image grid and a 2D PCA projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = np.load(args.features, mmap_mode="r")
    # Extraction proceeds in order, so stop scanning once we have enough rows.
    # This keeps previews fast even when the feature memmap is several GB.
    completed_chunks = []
    for start in range(0, len(features), 4096):
        stop = min(start + 4096, len(features))
        rows = np.flatnonzero(np.any(features[start:stop] != 0, axis=1)) + start
        completed_chunks.append(rows)
        if sum(map(len, completed_chunks)) >= args.sample_size:
            break
    indices = np.concatenate(completed_chunks)
    if len(indices) < 2:
        raise ValueError("Need at least two completed DINO embeddings to visualize.")

    rng = np.random.default_rng(0)
    sample = rng.choice(indices, size=min(args.sample_size, len(indices)), replace=False)
    embedding_sample = np.asarray(features[sample], dtype=np.float32)
    projection = PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(
        embedding_sample
    )

    preview = rng.choice(sample, size=min(12, len(sample)), replace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 7)
    scatter_axis = figure.add_subplot(grid[:, :3])
    scatter_axis.scatter(projection[:, 0], projection[:, 1], s=3, alpha=0.35)
    scatter_axis.set_title(f"DINOv2 embeddings (PCA; {len(sample):,} completed crops)")
    scatter_axis.set_xlabel("PC 1")
    scatter_axis.set_ylabel("PC 2")

    with h5py.File(args.crops, "r") as h5_file:
        crops = h5_file["crops"]
        for position, index in enumerate(preview):
            axis = figure.add_subplot(grid[position // 4, 3 + position % 4])
            axis.imshow(crops[index])
            axis.set_title(f"crop {index}", fontsize=8)
            axis.axis("off")

    figure.savefig(args.output, dpi=160)
    print(f"Saved {args.output} (preview uses {len(sample):,} completed embeddings).")


if __name__ == "__main__":
    main()
