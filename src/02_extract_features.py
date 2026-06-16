"""Stage 2: extract frozen ResNet18 image features from ALOHA frames.

The script reads images from LeRobotDataset, extracts 512-D ImageNet ResNet18
features from ``observation.images.top``, and saves arrays for later stages.
It does not train an MLP or run any coreset algorithm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from utils import ensure_dir, get_project_root, save_json, set_seed, to_numpy


REQUIRED_OUTPUT_FILES = (
    "features.npy",
    "actions.npy",
    "episode_ids.npy",
    "frame_ids.npy",
    "timestamps.npy",
    "split_info.json",
    "feature_info.json",
)


def parse_args() -> argparse.Namespace:
    """Parse Stage 2 feature extraction arguments."""
    parser = argparse.ArgumentParser(description="Extract ResNet18 features from ALOHA images.")
    parser.add_argument(
        "--dataset_name",
        default="lerobot/aloha_sim_transfer_cube_human",
        help="LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--cache_dir",
        default="data/aloha",
        help="Dataset cache directory relative to the project root.",
    )
    parser.add_argument(
        "--image_field",
        default="observation.images.top",
        help="Image field to read from each LeRobotDataset sample.",
    )
    parser.add_argument(
        "--action_field",
        default="action",
        help="Action field to read from each sample.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/features",
        help="Directory for extracted feature files.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Feature extraction batch size.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows to avoid multiprocessing issues.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for ResNet18 inference.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional debug limit. Example: --max_samples 100.",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Recompute and overwrite existing feature files.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def output_files_exist(output_dir: Path) -> bool:
    """Return True when all Stage 2 output files already exist."""
    return all((output_dir / filename).exists() for filename in REQUIRED_OUTPUT_FILES)


def print_existing_shapes(output_dir: Path) -> None:
    """Print shapes of existing NumPy feature artifacts."""
    for filename in REQUIRED_OUTPUT_FILES:
        path = output_dir / filename
        if path.suffix == ".npy" and path.exists():
            array = np.load(path, mmap_mode="r")
            print(f"{filename}: shape={array.shape}, dtype={array.dtype}")
        elif path.exists():
            print(f"{filename}: exists")


def choose_device(device_arg: str) -> torch.device:
    """Choose the requested torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' was requested, but CUDA is not available.")
    return torch.device(device_arg)


def build_resnet18_feature_extractor(device: torch.device) -> tuple[torch.nn.Module, str]:
    """Create a frozen ImageNet-pretrained ResNet18 with an identity fc layer."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load ResNet18 ImageNet weights. "
            "If this is the first run, make sure the environment can download "
            "torchvision model weights or pre-cache them locally."
        ) from exc

    model.fc = torch.nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.to(device)
    return model, "ResNet18_Weights.IMAGENET1K_V1"


def prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Resize [B, 3, H, W] images to 224x224 and apply ImageNet normalization."""
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B, 3, H, W], got shape {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3-channel images, got shape {tuple(images.shape)}")

    images = images.to(device=device, dtype=torch.float32)
    if images.max() > 2.0:
        images = images / 255.0

    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def action_to_7d(action: Any, sample_index: int) -> torch.Tensor:
    """Convert action to a 7-D single-arm label."""
    action_tensor = torch.as_tensor(to_numpy(action), dtype=torch.float32).reshape(-1)
    if action_tensor.shape[-1] == 14:
        return action_tensor[:7]
    if action_tensor.shape[-1] == 7:
        return action_tensor
    raise ValueError(
        f"Sample {sample_index}: action dimension must be 7 or 14, "
        f"got {action_tensor.shape[-1]}."
    )


def scalar_to_number(value: Any) -> int | float:
    """Convert tensor/list/numpy scalar metadata into a Python number."""
    array = np.asarray(to_numpy(value)).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot convert empty metadata field to scalar.")
    item = array[0].item()
    if isinstance(item, np.generic):
        item = item.item()
    return item


class AlohaFeatureDataset(Dataset):
    """Thin wrapper that exposes image, 7-D action, and frame metadata."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        image_field: str,
        action_field: str,
        max_samples: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.image_field = image_field
        self.action_field = action_field
        self.length = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]

        if self.image_field not in sample:
            available = ", ".join(sample.keys())
            raise KeyError(
                f"Sample {index}: image field '{self.image_field}' was not found. "
                f"Available keys: {available}"
            )
        if self.action_field not in sample:
            available = ", ".join(sample.keys())
            raise KeyError(
                f"Sample {index}: action field '{self.action_field}' was not found. "
                f"Available keys: {available}"
            )

        image = torch.as_tensor(sample[self.image_field], dtype=torch.float32)
        action_7 = action_to_7d(sample[self.action_field], index)

        return {
            "image": image,
            "action": action_7,
            "episode_index": torch.tensor(scalar_to_number(sample["episode_index"])),
            "frame_index": torch.tensor(scalar_to_number(sample["frame_index"])),
            "timestamp": torch.tensor(scalar_to_number(sample["timestamp"]), dtype=torch.float64),
        }


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    """Detach a tensor and move it to CPU NumPy."""
    return value.detach().cpu().numpy()


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run frozen ResNet18 inference and collect arrays."""
    feature_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    episode_batches: list[np.ndarray] = []
    frame_batches: list[np.ndarray] = []
    timestamp_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting ResNet18 features"):
            images = prepare_images(batch["image"], device)
            features = model(images)
            if features.ndim != 2 or features.shape[1] != 512:
                raise RuntimeError(f"Expected model output [B, 512], got {tuple(features.shape)}")

            feature_batches.append(tensor_to_numpy(features).astype(np.float32))
            action_batches.append(tensor_to_numpy(batch["action"]).astype(np.float32))
            episode_batches.append(tensor_to_numpy(batch["episode_index"]).reshape(-1))
            frame_batches.append(tensor_to_numpy(batch["frame_index"]).reshape(-1))
            timestamp_batches.append(tensor_to_numpy(batch["timestamp"]).reshape(-1))

    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(action_batches, axis=0),
        np.concatenate(episode_batches, axis=0),
        np.concatenate(frame_batches, axis=0),
        np.concatenate(timestamp_batches, axis=0),
    )


def build_split_info(episode_ids: np.ndarray) -> dict[str, Any]:
    """Build deterministic episode split metadata for later stages."""
    sorted_episode_ids = sorted(int(episode_id) for episode_id in np.unique(episode_ids))
    train_count = int(len(sorted_episode_ids) * 0.8)
    train_episodes = sorted_episode_ids[:train_count]
    test_episodes = sorted_episode_ids[train_count:]
    return {
        "sorted_episode_ids": sorted_episode_ids,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "train_ratio": 0.8,
        "split_rule": "sort episode_index, first 80% train, last 20% test",
    }


def save_outputs(
    output_dir: Path,
    features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    split_info: dict[str, Any],
    feature_info: dict[str, Any],
) -> None:
    """Save all Stage 2 feature artifacts."""
    ensure_dir(output_dir)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "episode_ids.npy", episode_ids)
    np.save(output_dir / "frame_ids.npy", frame_ids)
    np.save(output_dir / "timestamps.npy", timestamps)
    save_json(split_info, output_dir / "split_info.json")
    save_json(feature_info, output_dir / "feature_info.json")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cache_dir = resolve_project_path(args.cache_dir)
    output_dir = ensure_dir(resolve_project_path(args.output_dir))

    if output_files_exist(output_dir) and not args.force_extract:
        print("Feature files already exist. Skip extraction.")
        print_existing_shapes(output_dir)
        return

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading LeRobotDataset: {args.dataset_name}")
    print(f"Cache directory: {cache_dir}")

    dataset = LeRobotDataset(repo_id=args.dataset_name, root=cache_dir)
    wrapped_dataset = AlohaFeatureDataset(
        dataset=dataset,
        image_field=args.image_field,
        action_field=args.action_field,
        max_samples=args.max_samples,
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Samples to extract: {len(wrapped_dataset)}")

    if len(wrapped_dataset) == 0:
        raise RuntimeError("No samples selected for feature extraction.")

    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, weight_name = build_resnet18_feature_extractor(device)
    features, actions, episode_ids, frame_ids, timestamps = extract_features(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    if features.shape[1] != 512:
        raise RuntimeError(f"features.npy must have shape [N, 512], got {features.shape}")
    if actions.shape[1] != 7:
        raise RuntimeError(f"actions.npy must have shape [N, 7], got {actions.shape}")

    split_info = build_split_info(episode_ids)
    feature_info = {
        "dataset_name": args.dataset_name,
        "image_field": args.image_field,
        "action_field": args.action_field,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "model": "torchvision.models.resnet18 with fc=Identity",
        "pretrained_weights": weight_name,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "max_samples": args.max_samples,
    }

    save_outputs(
        output_dir=output_dir,
        features=features,
        actions=actions,
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        timestamps=timestamps,
        split_info=split_info,
        feature_info=feature_info,
    )

    print("\nFeature extraction finished.")
    print(f"features.npy shape: {features.shape}")
    print(f"actions.npy shape: {actions.shape}")
    print(f"episode_ids.npy shape: {episode_ids.shape}")
    print(f"frame_ids.npy shape: {frame_ids.shape}")
    print(f"timestamps.npy shape: {timestamps.shape}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
