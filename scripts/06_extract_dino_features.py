from __future__ import annotations

import argparse
import gc
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from feature_extraction_utils import (
    extract_resumable,
    finalize_features,
    prepare_outputs,
    save_array_safely as save_mask_safely,
)


DEFAULT_REPO_DIR = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
DEFAULT_MODEL_NAME = "dinov2_vitb14"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LOG_EVERY = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features from fixation crops stored in HDF5."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("challenge1/training/crops_112.h5"),
        help="Input HDF5 file containing crops and valid_mask.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("challenge1/training/dino_features_vitb14.npy"),
        help="Output .npy file for extracted feature vectors.",
    )
    parser.add_argument(
        "--valid-mask-output",
        type=Path,
        default=Path("challenge1/training/dino_features_vitb14_valid_mask.npy"),
        help="Output .npy file to save the validity mask alongside the features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Torch Hub DINOv2 model name, for example dinov2_vitb14 or dinov2_vits14.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Local Torch Hub repository directory for DINOv2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, for example cuda, cuda:0, or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Print progress every N batches.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible .tmp.npy file (default: enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def temporary_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def check_output_paths(output_file: Path, mask_file: Path, overwrite: bool) -> None:
    existing = [path for path in (output_file, mask_file) if path.exists()]
    if existing and not overwrite:
        existing_text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output path already exists. Use --overwrite to replace it:\n"
            f"{existing_text}"
        )


def save_array_safely(path: Path, array: np.ndarray, overwrite: bool) -> None:
    temp_path = temporary_output_path(path)

    if temp_path.exists():
        temp_path.unlink()

    np.save(temp_path, array)

    if overwrite and path.exists():
        path.unlink()

    temp_path.replace(path)


def load_model(repo_dir: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Local DINOv2 repo not found: {repo_dir}\n"
            "Either cache the repository first or pass --repo-dir."
        )

    model = torch.hub.load(
        str(repo_dir),
        model_name,
        source="local",
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def extract_features(
    crops: h5py.Dataset,
    valid_mask: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    output_file: Path,
    batch_size: int,
    log_every: int,
    overwrite: bool,
) -> tuple[Path, tuple[int, int], np.dtype]:
    temp_output = temporary_output_path(output_file)
    def infer_batch(batch_indices: np.ndarray) -> np.ndarray:
        batch_tensors = [
            transform(Image.fromarray(crops[index], mode="RGB"))
            for index in batch_indices
        ]
        inputs = torch.stack(batch_tensors, dim=0).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            return model(inputs).detach().cpu().numpy()

    feature_shape, feature_dtype, _ = extract_resumable(
        valid_mask=valid_mask,
        temp_file=temp_output,
        infer_batch=infer_batch,
        batch_size=batch_size,
        log_every=log_every,
    )
    finalize_features(temp_output, output_file)
    return output_file, feature_shape, feature_dtype


def main() -> None:
    args = parse_args()

    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()
    mask_output = args.valid_mask_output.resolve()
    repo_dir = args.repo_dir.resolve()
    device = resolve_device(args.device)

    if not input_file.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_file}")

    prepare_outputs(
        output_file,
        mask_output,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    print("=" * 72)
    print("DINOv2 FEATURE EXTRACTION")
    print("=" * 72)
    print(f"Input file   : {input_file}")
    print(f"Output file  : {output_file}")
    print(f"Mask file    : {mask_output}")
    print(f"Model        : {args.model_name}")
    print(f"Repo dir     : {repo_dir}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Device       : {device}")

    transform = build_transform()
    model = load_model(repo_dir=repo_dir, model_name=args.model_name, device=device)

    with h5py.File(input_file, "r") as h5_file:
        if "crops" not in h5_file or "valid_mask" not in h5_file:
            raise KeyError("Expected datasets 'crops' and 'valid_mask' in the HDF5 file.")

        crops = h5_file["crops"]
        valid_mask = np.asarray(h5_file["valid_mask"][:], dtype=bool)

        print(f"Crops shape   : {crops.shape}")
        print(f"Valid crops   : {int(valid_mask.sum())}/{len(valid_mask)}")
        print(f"Invalid crops : {int((~valid_mask).sum())}")

        feature_file, feature_shape, feature_dtype = extract_features(
            crops=crops,
            valid_mask=valid_mask,
            model=model,
            transform=transform,
            device=device,
            output_file=output_file,
            batch_size=args.batch_size,
            log_every=max(1, args.log_every),
            overwrite=args.overwrite,
        )

    save_mask_safely(mask_output, valid_mask)

    print("\nSaved feature matrix:")
    print(f"  {feature_file}")
    print(f"  shape={feature_shape}, dtype={feature_dtype}")
    print("Saved validity mask:")
    print(f"  {mask_output}")
    print(f"  shape={valid_mask.shape}, dtype={valid_mask.dtype}")


if __name__ == "__main__":
    main()
