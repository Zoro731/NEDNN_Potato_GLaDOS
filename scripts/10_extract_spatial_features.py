"""Extract compact spatial appearance features from fixation crops.

This is a cheap, spatially explicit visual branch for the next encoding test.
It keeps pooled RGB means and texture/edge summaries on a regular grid instead
of collapsing each crop to one global ImageNet embedding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from feature_extraction_utils import (
    extract_resumable,
    finalize_features,
    prepare_outputs,
    save_array_safely,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract grid-pooled spatial and texture features from crops."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("challenge1/training/crops_112.h5"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("challenge1/training/spatial_features.npy"),
    )
    parser.add_argument(
        "--valid-mask-output",
        type=Path,
        default=Path("challenge1/training/spatial_features_valid_mask.npy"),
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def spatial_features(images: np.ndarray, grid_size: int) -> np.ndarray:
    """Return grid-pooled RGB, texture and edge summaries for each crop."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB images, got {images.shape}")
    _, height, width, _ = images.shape
    if height % grid_size or width % grid_size:
        raise ValueError(
            f"Crop shape {(height, width)} is not divisible by grid {grid_size}"
        )

    rgb = images.astype(np.float32) / 255.0
    block_h = height // grid_size
    block_w = width // grid_size

    pooled = rgb.reshape(
        len(rgb), grid_size, block_h, grid_size, block_w, 3
    )
    rgb_mean = pooled.mean(axis=(2, 4))
    rgb_std = pooled.std(axis=(2, 4))

    gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    grad_x = np.diff(gray, axis=2, prepend=gray[:, :, :1])
    grad_y = np.diff(gray, axis=1, prepend=gray[:, :1, :])
    edge = np.abs(grad_x) + np.abs(grad_y)
    gradient = np.stack((grad_x, grad_y, edge), axis=-1)
    gradient_pooled = gradient.reshape(
        len(rgb), grid_size, block_h, grid_size, block_w, 3
    ).mean(axis=(2, 4))

    features = np.concatenate(
        [
            rgb_mean.reshape(len(rgb), -1),
            rgb_std.reshape(len(rgb), -1),
            gradient_pooled.reshape(len(rgb), -1),
            np.ones((len(rgb), 1), dtype=np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()
    mask_file = args.valid_mask_output.resolve()
    if args.grid_size < 1 or args.batch_size < 1:
        raise ValueError("grid-size and batch-size must be positive")

    temp_file = prepare_outputs(
        output_file,
        mask_file,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    print("=" * 72)
    print("SPATIAL CROP FEATURE EXTRACTION")
    print("=" * 72)
    print(f"Input file : {input_file}")
    print(f"Output file: {output_file}")
    print(f"Grid size  : {args.grid_size} x {args.grid_size}")
    print(f"Batch size : {args.batch_size}")

    with h5py.File(input_file, "r") as h5_file:
        if "crops" not in h5_file or "valid_mask" not in h5_file:
            raise KeyError("Expected HDF5 datasets 'crops' and 'valid_mask'.")
        crops = h5_file["crops"]
        valid_mask = np.asarray(h5_file["valid_mask"][:], dtype=bool)
        if crops.shape[0] != len(valid_mask):
            raise ValueError("crops and valid_mask row counts do not match")

        def infer_batch(batch_indices: np.ndarray) -> np.ndarray:
            images = np.asarray(crops[batch_indices.tolist()])
            return spatial_features(images, args.grid_size)

        feature_shape, feature_dtype, extracted, complete = extract_resumable(
            valid_mask=valid_mask,
            temp_file=temp_file,
            infer_batch=infer_batch,
            batch_size=args.batch_size,
            log_every=args.log_every,
            max_batches=args.max_batches,
        )

    if complete:
        finalize_features(temp_file, output_file)
        save_array_safely(mask_file, valid_mask)

    print("\nFeature matrix:")
    print(f"  {output_file if complete else temp_file}")
    print(f"  shape={feature_shape}, dtype={feature_dtype}")
    print(f"  rows extracted this run={extracted}")
    if complete:
        print(f"Saved validity mask: {mask_file}")
    else:
        print("Partial run finished cleanly; rerun the same command to resume.")


if __name__ == "__main__":
    main()
