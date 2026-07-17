"""Extract resumable ResNet50 embeddings from fixation crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import models

from feature_extraction_utils import (
    extract_resumable,
    finalize_features,
    prepare_outputs,
    save_array_safely,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet50 features from fixation crops stored in HDF5."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("challenge1/training/crops_112.h5"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("challenge1/training/resnet50_features.npy"),
    )
    parser.add_argument(
        "--valid-mask-output",
        type=Path,
        default=Path("challenge1/training/resnet50_features_valid_mask.npy"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible .tmp.npy file (default: enabled).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def main() -> None:
    args = parse_args()
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()
    mask_file = args.valid_mask_output.resolve()
    device = get_device(args.device)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    temp_file = prepare_outputs(
        output_file,
        mask_file,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    print("=" * 72)
    print("RESNET50 FEATURE EXTRACTION")
    print("=" * 72)
    print(f"Input file  : {input_file}")
    print(f"Output file : {output_file}")
    print(f"Mask file   : {mask_file}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Device      : {device}")

    model, transform = load_model(device)

    with h5py.File(input_file, "r") as h5_file:
        if "crops" not in h5_file or "valid_mask" not in h5_file:
            raise KeyError("Expected HDF5 datasets 'crops' and 'valid_mask'.")
        crops = h5_file["crops"]
        valid_mask = np.asarray(h5_file["valid_mask"][:], dtype=bool)
        if crops.shape[0] != len(valid_mask):
            raise ValueError("crops and valid_mask row counts do not match")

        print(f"Crops shape   : {crops.shape}")
        print(f"Valid crops   : {int(valid_mask.sum())}/{len(valid_mask)}")
        print(f"Invalid crops : {int((~valid_mask).sum())}")

        def infer_batch(batch_indices: np.ndarray) -> np.ndarray:
            images = [
                transform(Image.fromarray(crops[index], mode="RGB"))
                for index in batch_indices
            ]
            inputs = torch.stack(images).to(device, non_blocking=True)
            with torch.inference_mode():
                return model(inputs).detach().cpu().numpy()

        feature_shape, feature_dtype, extracted = extract_resumable(
            valid_mask=valid_mask,
            temp_file=temp_file,
            infer_batch=infer_batch,
            batch_size=args.batch_size,
            log_every=args.log_every,
        )

    finalize_features(temp_file, output_file)
    save_array_safely(mask_file, valid_mask)

    print("\nSaved feature matrix:")
    print(f"  {output_file}")
    print(f"  shape={feature_shape}, dtype={feature_dtype}")
    print(f"  rows extracted this run={extracted}")
    print("Saved validity mask:")
    print(f"  {mask_file}")


if __name__ == "__main__":
    main()
