"""Common utility helpers for the VLA coreset project."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Keep utility imports usable before the full environment is installed.
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """Return the project root without relying on a machine-specific path."""
    return Path(__file__).resolve().parents[1]


def save_json(data: Any, path: str | Path) -> None:
    """Save data as a JSON file, creating the parent directory automatically."""
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Load a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_arrays(feature_dir: str | Path) -> dict[str, Any]:
    """Load Stage 2 feature arrays and split metadata from a feature directory."""
    feature_path = Path(feature_dir)
    required_files = {
        "features": feature_path / "features.npy",
        "actions": feature_path / "actions.npy",
        "episode_ids": feature_path / "episode_ids.npy",
        "frame_ids": feature_path / "frame_ids.npy",
        "timestamps": feature_path / "timestamps.npy",
        "split_info": feature_path / "split_info.json",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 2 feature files are missing. Run feature extraction first. "
            f"Missing files: {missing}"
        )

    return {
        "features": np.load(required_files["features"]),
        "actions": np.load(required_files["actions"]),
        "episode_ids": np.load(required_files["episode_ids"]),
        "frame_ids": np.load(required_files["frame_ids"]),
        "timestamps": np.load(required_files["timestamps"]),
        "split_info": load_json(required_files["split_info"]),
    }


def get_train_test_masks(
    episode_ids: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Create fixed episode-level train/test masks.

    Episodes are sorted by id. The first train_ratio fraction is used for
    training, and the remaining episodes are reserved for testing. Test samples
    must never be used for selection or clustering.
    """
    sorted_episode_ids = sorted(int(episode_id) for episode_id in np.unique(episode_ids))
    if not sorted_episode_ids:
        raise ValueError("No episode ids were found.")

    train_count = int(len(sorted_episode_ids) * train_ratio)
    if train_count <= 0 or train_count >= len(sorted_episode_ids):
        raise ValueError(
            f"Invalid train/test split with {len(sorted_episode_ids)} episodes "
            f"and train_ratio={train_ratio}."
        )

    train_episodes = sorted_episode_ids[:train_count]
    test_episodes = sorted_episode_ids[train_count:]
    train_mask = np.isin(episode_ids, train_episodes)
    test_mask = np.isin(episode_ids, test_episodes)
    return train_mask, test_mask, train_episodes, test_episodes


def compute_action_change_scores(actions: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Compute within-episode L2 action changes for adjacent frames.

    Each episode's first frame receives score 0. Different episode boundaries
    are never subtracted. This score corresponds to prediction error / surprise
    in predictive coding and helps filter temporally redundant frames.
    """
    if len(actions) != len(episode_ids):
        raise ValueError(
            f"actions and episode_ids must have the same length, got "
            f"{len(actions)} and {len(episode_ids)}."
        )

    scores = np.zeros(len(actions), dtype=np.float32)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if len(indices) <= 1:
            continue
        diffs = actions[indices[1:]] - actions[indices[:-1]]
        scores[indices[1:]] = np.linalg.norm(diffs, axis=1).astype(np.float32)
    return scores


def allocate_cluster_quotas(cluster_ids: np.ndarray, total_budget: int) -> dict[int, int]:
    """Allocate an exact sample budget across non-empty clusters.

    Quotas are proportional to cluster size. When possible, every non-empty
    cluster receives at least one sample, then rounding leftovers are assigned
    by largest fractional remainder. The returned quotas sum to total_budget.
    """
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}.")
    if total_budget == 0:
        return {}

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    if len(unique_clusters) == 0:
        return {}

    exact = counts / counts.sum() * total_budget
    quotas = np.floor(exact).astype(int)

    if total_budget >= len(unique_clusters):
        quotas = np.maximum(quotas, 1)

    while quotas.sum() > total_budget:
        candidates = np.flatnonzero(quotas > 0)
        if total_budget >= len(unique_clusters):
            candidates = np.flatnonzero(quotas > 1)
        remove_at = candidates[np.argmin(exact[candidates] - np.floor(exact[candidates]))]
        quotas[remove_at] -= 1

    remainders = exact - np.floor(exact)
    while quotas.sum() < total_budget:
        candidates = np.flatnonzero(quotas < counts)
        add_at = candidates[np.argmax(remainders[candidates])]
        quotas[add_at] += 1
        remainders[add_at] = -1.0

    return {int(cluster): int(quota) for cluster, quota in zip(unique_clusters, quotas) if quota > 0}


def infer_shape(value: Any) -> tuple[int, ...] | str:
    """Infer a readable shape for tensors, arrays, PIL images, or lists."""
    if hasattr(value, "shape"):
        return tuple(value.shape)

    if hasattr(value, "size") and hasattr(value, "mode"):
        width, height = value.size
        return (height, width, len(value.getbands()))

    if isinstance(value, (list, tuple)):
        return np.asarray(value).shape

    return "unknown"


def to_numpy(value: Any) -> np.ndarray:
    """Convert common sample values to a NumPy array without assuming a backend."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    if hasattr(value, "__array__"):
        return np.asarray(value)

    return np.asarray(value)
