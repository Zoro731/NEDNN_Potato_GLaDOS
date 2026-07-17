"""Shared, crash-safe helpers for fixation-crop feature extraction."""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np


def temporary_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def prepare_outputs(
    output_file: Path,
    mask_file: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    """Validate outputs and return the temporary feature path."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = temporary_output_path(output_file)

    if overwrite:
        for path in (output_file, mask_file, temp_file):
            if path.exists():
                path.unlink()
        return temp_file

    finished = [path for path in (output_file, mask_file) if path.exists()]
    if finished:
        paths = "\n".join(f"  - {path}" for path in finished)
        raise FileExistsError(
            "Finished output already exists. Use --overwrite to replace it:\n"
            f"{paths}"
        )

    if temp_file.exists() and not resume:
        raise FileExistsError(
            f"Partial output exists: {temp_file}\n"
            "Use --resume to continue it or --overwrite to restart."
        )

    return temp_file


def save_array_safely(path: Path, array: np.ndarray) -> None:
    temp_file = temporary_output_path(path)
    if temp_file.exists():
        temp_file.unlink()
    np.save(temp_file, array)
    temp_file.replace(path)


def _completed_rows(
    features: np.ndarray,
    valid_mask: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Infer completed rows in an interrupted array.

    DINO and ResNet embeddings are never exactly all-zero, while unprocessed and
    invalid rows are initialized to zero. This lets legacy partial arrays resume
    even though they were created before explicit progress tracking existed.
    """
    completed = np.zeros(len(valid_mask), dtype=bool)
    for start in range(0, len(valid_mask), chunk_size):
        stop = min(start + chunk_size, len(valid_mask))
        completed[start:stop] = np.any(features[start:stop] != 0, axis=1)
    completed &= valid_mask
    return completed


def extract_resumable(
    *,
    valid_mask: np.ndarray,
    temp_file: Path,
    infer_batch: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    log_every: int,
) -> tuple[tuple[int, int], np.dtype, int]:
    """Extract features, resuming zero-initialized .npy files when present."""
    num_samples = len(valid_mask)
    feature_memmap: np.memmap | None = None

    if temp_file.exists():
        feature_memmap = np.load(temp_file, mmap_mode="r+")
        if feature_memmap.ndim != 2 or feature_memmap.shape[0] != num_samples:
            raise ValueError(
                f"Partial feature shape {feature_memmap.shape} does not match "
                f"{num_samples} input rows. Use --overwrite to restart."
            )
        completed = _completed_rows(feature_memmap, valid_mask)
        print(
            f"Resuming partial output: {int(completed.sum())}/"
            f"{int(valid_mask.sum())} valid rows already complete"
        )
    else:
        completed = np.zeros(num_samples, dtype=bool)

    pending_indices = np.flatnonzero(valid_mask & ~completed)
    initially_complete = int(completed.sum())
    total_valid = int(valid_mask.sum())
    total_batches = (len(pending_indices) + batch_size - 1) // batch_size

    if total_valid == 0:
        raise ValueError("No valid crops were found in valid_mask.")

    for batch_number, start in enumerate(
        range(0, len(pending_indices), batch_size), start=1
    ):
        batch_indices = pending_indices[start : start + batch_size]
        batch_features = np.asarray(infer_batch(batch_indices), dtype=np.float32)
        if batch_features.ndim != 2 or len(batch_features) != len(batch_indices):
            raise ValueError(
                "Model output must be a 2D array with one row per input crop."
            )

        if feature_memmap is None:
            feature_memmap = np.lib.format.open_memmap(
                temp_file,
                mode="w+",
                dtype=np.float32,
                shape=(num_samples, batch_features.shape[1]),
            )
            feature_memmap[:] = 0.0
        elif feature_memmap.shape[1] != batch_features.shape[1]:
            raise ValueError(
                f"Model produced {batch_features.shape[1]} features, but partial "
                f"output has {feature_memmap.shape[1]}. Use --overwrite to restart."
            )

        feature_memmap[batch_indices] = batch_features

        if (
            batch_number == 1
            or batch_number % max(1, log_every) == 0
            or batch_number == total_batches
        ):
            feature_memmap.flush()
            processed = initially_complete + min(
                start + len(batch_indices), len(pending_indices)
            )
            print(
                f"Processed {processed}/{total_valid} valid crops "
                f"({batch_number}/{total_batches} remaining batches)"
            )

    if feature_memmap is None:
        # This happens only when a partial file was somehow removed after scanning.
        raise RuntimeError("No feature array was available after extraction.")

    shape = tuple(feature_memmap.shape)
    dtype = feature_memmap.dtype
    feature_memmap.flush()
    del feature_memmap
    gc.collect()
    return shape, dtype, len(pending_indices)


def finalize_features(temp_file: Path, output_file: Path) -> None:
    if not temp_file.exists():
        raise FileNotFoundError(f"Missing temporary feature file: {temp_file}")
    temp_file.replace(output_file)
