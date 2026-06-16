"""Stage 1: check ALOHA data loading with LeRobotDataset first.

This script only verifies dataset access, image fields, and action labels. It
does not extract ResNet18 features, train an MLP, or run coreset selection.
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image

from utils import ensure_dir, get_project_root, set_seed, to_numpy


IMAGE_FIELD_PRIORITY = (
    "observation.images.top",
    "observation.images.cam_high",
    "observation.images.front",
    "observation.images.overhead",
    "observation.image",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Stage 1 dataset check."""
    parser = argparse.ArgumentParser(description="Check ALOHA dataset loading.")
    parser.add_argument(
        "--dataset_name",
        default="lerobot/aloha_sim_transfer_cube_human",
        help="LeRobot/Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--cache_dir",
        default="data/aloha",
        help="Dataset cache directory relative to the project root.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to inspect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible inspection.",
    )
    return parser.parse_args()


def resolve_cache_dir(cache_dir: str) -> Path:
    """Resolve cache_dir relative to the project root unless it is absolute."""
    path = Path(cache_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    return ensure_dir(path)


def flatten_dict(sample: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into dot-separated keys.

    LeRobotDataset samples are often already flat, for example
    ``observation.images.top``. Some loaders may return nested dictionaries.
    This helper supports both formats with one lookup path.
    """
    flattened: dict[str, Any] = {}
    if not isinstance(sample, Mapping):
        return flattened

    for key, value in sample.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        flattened[field] = value
        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, field))

    return flattened


def keys_containing(flattened: Mapping[str, Any], text: str) -> list[str]:
    """Return keys containing a case-insensitive substring."""
    text = text.lower()
    return [key for key in flattened.keys() if text in key.lower()]


def print_value_summary(name: str, value: Any, max_chars: int = 1000) -> None:
    """Print a short summary for potentially large dataset attributes."""
    text = repr(value)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    print(f"{name}: {text}")


def print_dataset_basic_info(dataset: Any) -> None:
    """Print dataset type, length, metadata, features, and first sample keys."""
    print("\nLeRobotDataset basic information:")
    print(f"dataset type: {type(dataset)}")
    try:
        print(f"len(dataset): {len(dataset)}")
    except Exception as exc:
        print(f"WARNING: failed to compute len(dataset): {exc}")

    if hasattr(dataset, "meta"):
        print_value_summary("dataset.meta", getattr(dataset, "meta"))
    else:
        print("dataset.meta: not found")

    if hasattr(dataset, "features"):
        print_value_summary("dataset.features", getattr(dataset, "features"))
    else:
        print("dataset.features: not found")

    first_sample = dataset[0]
    flattened = flatten_dict(first_sample)
    print("\nFirst sample keys:")
    for key in flattened.keys():
        print(f"  - {key}")


def import_lerobot_dataset() -> Any | None:
    """Import LeRobotDataset, returning None when lerobot is unavailable."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        print(f"WARNING: failed to import LeRobotDataset: {exc}")
        print(
            "Fallback to datasets.load_dataset. This may only load tabular "
            "metadata and may not include decoded images."
        )
        return None


def instantiate_lerobot_dataset(dataset_cls: Any, repo_id: str, root: Path) -> Any | None:
    """Instantiate LeRobotDataset and print constructor details on failure."""
    try:
        return dataset_cls(repo_id=repo_id, root=root)
    except TypeError as exc:
        print(f"WARNING: LeRobotDataset(repo_id=..., root=...) failed: {exc}")
        try:
            signature = inspect.signature(dataset_cls)
            print(f"LeRobotDataset constructor signature: {signature}")
        except Exception as sig_exc:
            print(f"WARNING: failed to inspect LeRobotDataset signature: {sig_exc}")
            return None

        kwargs: dict[str, Any] = {}
        params = signature.parameters
        if "repo_id" in params:
            kwargs["repo_id"] = repo_id
        elif "dataset_name" in params:
            kwargs["dataset_name"] = repo_id
        elif "repo" in params:
            kwargs["repo"] = repo_id
        else:
            print("ERROR: no recognized dataset repo parameter in LeRobotDataset.")
            return None

        if "root" in params:
            kwargs["root"] = root
        elif "cache_dir" in params:
            kwargs["cache_dir"] = root

        try:
            print(f"Retrying LeRobotDataset with kwargs: {kwargs}")
            return dataset_cls(**kwargs)
        except Exception as retry_exc:
            print(f"ERROR: failed to instantiate LeRobotDataset after signature retry: {retry_exc}")
            return None
    except Exception as exc:
        print(f"ERROR: failed to instantiate LeRobotDataset: {exc}")
        return None


def print_image_value_info(field: str, value: Any) -> None:
    """Print type-specific information for a selected image field."""
    print(f"  selected image field: {field}")
    print(f"  image type: {type(value)}")

    if isinstance(value, torch.Tensor):
        print(f"  image shape: {tuple(value.shape)}")
        print(f"  image dtype: {value.dtype}")
        if value.numel() > 0:
            image_float = value.detach().cpu().float()
            print(f"  image min/max: {image_float.min().item():.6f} / {image_float.max().item():.6f}")
        return

    if isinstance(value, Image.Image):
        print(f"  image size: {value.size}")
        print(f"  image mode: {value.mode}")
        return

    if isinstance(value, np.ndarray):
        print(f"  image shape: {value.shape}")
        print(f"  image dtype: {value.dtype}")
        return

    summary = str(value)
    print(f"  image summary: {summary[:300]}")


def select_image_field(flattened: Mapping[str, Any]) -> str | None:
    """Select one camera/image field using the requested priority order."""
    for candidate in IMAGE_FIELD_PRIORITY:
        if candidate in flattened and not isinstance(flattened[candidate], Mapping):
            return candidate

    for key, value in flattened.items():
        if "image" in key.lower() and not isinstance(value, Mapping):
            return key

    return None


def print_action_info(action_value: Any) -> Any:
    """Print action shape/preview and return the single-arm 7-DoF label."""
    if isinstance(action_value, torch.Tensor):
        action = action_value.detach().cpu()
        flat = action.reshape(-1)
        preview = flat[: min(5, flat.numel())].tolist()
        print(f"  action type: {type(action_value)}")
        print(f"  action shape: {tuple(action.shape)}")
        print(f"  action first values: {preview}")
        last_dim = action.shape[-1] if action.ndim > 0 else 1
        if last_dim == 14:
            print("Use action[:7] as single-arm 7-DoF action label.")
            return action[..., :7]
        if last_dim == 7:
            return action
        print(f"WARNING: action last dimension is neither 7 nor 14. Detected: {last_dim}")
        return action[..., : min(7, last_dim)] if action.ndim > 0 else action

    action = to_numpy(action_value)
    flat = action.reshape(-1)
    preview = flat[: min(5, flat.size)]
    print(f"  action type: {type(action_value)}")
    print(f"  action shape: {action.shape}")
    print(f"  action first values: {np.array2string(preview, precision=4, separator=', ')}")

    last_dim = action.shape[-1] if action.ndim > 0 else 1
    if last_dim == 14:
        print("Use action[:7] as single-arm 7-DoF action label.")
        return action[..., :7]
    if last_dim == 7:
        return action

    print(f"WARNING: action last dimension is neither 7 nor 14. Detected: {last_dim}")
    return action[..., : min(7, last_dim)] if action.ndim > 0 else action


def inspect_lerobot_samples(dataset: Any, max_samples: int) -> tuple[str | None, str | None]:
    """Inspect the first samples from LeRobotDataset."""
    detected_image_field: str | None = None
    detected_action_field: str | None = None
    inspect_count = min(max_samples, len(dataset))

    for sample_index in range(inspect_count):
        sample = dataset[sample_index]
        flattened = flatten_dict(sample)

        image_keys = sorted(set(keys_containing(flattened, "image") + keys_containing(flattened, "images")))
        action_keys = keys_containing(flattened, "action")
        observation_keys = keys_containing(flattened, "observation")

        print(f"\nSample index: {sample_index}")
        print("  sample keys:")
        for key in flattened.keys():
            print(f"    - {key}")
        print(f"  episode_index: {flattened.get('episode_index', 'not found')}")
        print(f"  frame_index: {flattened.get('frame_index', 'not found')}")
        print(f"  timestamp: {flattened.get('timestamp', 'not found')}")
        print(f"  fields containing image/images: {image_keys if image_keys else 'none'}")
        print(f"  fields containing action: {action_keys if action_keys else 'none'}")
        print(f"  fields containing observation: {observation_keys if observation_keys else 'none'}")

        image_field = select_image_field(flattened)
        if image_field is not None:
            detected_image_field = detected_image_field or image_field
            print_image_value_info(image_field, flattened[image_field])

        if "action" in flattened:
            detected_action_field = "action"
            action_7 = print_action_info(flattened["action"])
            if hasattr(action_7, "shape"):
                print(f"  single-arm action shape: {tuple(action_7.shape)}")
        else:
            print("WARNING: action field was not found in this sample.")

    return detected_image_field, detected_action_field


def fallback_load_dataset(dataset_name: str, cache_dir: Path, max_samples: int) -> None:
    """Fallback to datasets.load_dataset for tabular metadata inspection."""
    print("\nFallback to datasets.load_dataset. This may only load tabular metadata and may not include decoded images.")
    dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))

    print("\ndatasets.load_dataset split information:")
    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} samples")
        split = dataset[next(iter(dataset.keys()))]
    else:
        print(f"  - default: {len(dataset)} samples")
        split = dataset

    if len(split) == 0:
        print("WARNING: fallback dataset split is empty.")
        return

    inspect_count = min(max_samples, len(split))
    for sample_index in range(inspect_count):
        sample = split[sample_index]
        flattened = flatten_dict(sample)
        print(f"\nFallback sample index: {sample_index}")
        print("  keys:")
        for key in flattened.keys():
            print(f"    - {key}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_dir = resolve_cache_dir(args.cache_dir)

    dataset_cls = import_lerobot_dataset()
    if dataset_cls is None:
        fallback_load_dataset(args.dataset_name, cache_dir, args.max_samples)
        return

    dataset = instantiate_lerobot_dataset(dataset_cls, args.dataset_name, cache_dir)
    if dataset is None:
        fallback_load_dataset(args.dataset_name, cache_dir, args.max_samples)
        return

    print_dataset_basic_info(dataset)
    image_field, action_field = inspect_lerobot_samples(dataset, args.max_samples)

    print("\nDataset check finished.")
    if image_field is not None:
        print("LeRobotDataset image loading succeeded.")
        print(f"Detected image field: {image_field}")
        print(f"Detected action field: {action_field if action_field else 'not found'}")
    else:
        print("WARNING: LeRobotDataset loaded, but no image field was detected.")
        print("Please print all keys above for debugging.")


if __name__ == "__main__":
    main()
