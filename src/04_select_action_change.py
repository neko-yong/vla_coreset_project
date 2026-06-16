"""Stage 3b: Action-Change Coreset selection.

The action-change score is the within-episode L2 distance between adjacent
actions. It corresponds to prediction error / surprise in predictive coding and
is used to filter temporally redundant frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    compute_action_change_scores,
    ensure_dir,
    get_project_root,
    get_train_test_masks,
    load_feature_arrays,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse Action-Change Coreset arguments."""
    parser = argparse.ArgumentParser(description="Select high action-change training frames.")
    parser.add_argument("--feature_dir", default="outputs/features", help="Stage 2 feature directory.")
    parser.add_argument("--output_dir", default="outputs/results", help="Directory for selection files.")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of train frames to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed recorded for reproducibility.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def selection_budget(num_train_samples: int, sample_ratio: float) -> int:
    """Return the exact number of samples selected from the training set."""
    if not 0 < sample_ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}.")
    return max(1, int(round(num_train_samples * sample_ratio)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_dir = resolve_project_path(args.feature_dir)
    output_dir = ensure_dir(resolve_project_path(args.output_dir))
    arrays = load_feature_arrays(feature_dir)

    actions = arrays["actions"]
    episode_ids = arrays["episode_ids"]
    train_mask, test_mask, train_episodes, test_episodes = get_train_test_masks(episode_ids)
    train_indices = np.flatnonzero(train_mask)
    budget = selection_budget(len(train_indices), args.sample_ratio)

    scores = np.zeros(len(episode_ids), dtype=np.float32)
    train_scores = compute_action_change_scores(actions[train_indices], episode_ids[train_indices])
    scores[train_indices] = train_scores
    order = np.lexsort((train_indices, -train_scores))
    selected_indices = np.sort(train_indices[order[:budget]])

    scores_path = output_dir / "action_change_scores.npy"
    selected_path = output_dir / "selected_indices_action_change.npy"
    info_path = output_dir / "action_change_selection_info.json"

    np.save(scores_path, scores.astype(np.float32))
    np.save(selected_path, selected_indices.astype(np.int64))

    info: dict[str, Any] = {
        "method": "action_change",
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "num_total_samples": int(len(episode_ids)),
        "num_train_samples": int(train_mask.sum()),
        "num_test_samples": int(test_mask.sum()),
        "num_selected_samples": int(len(selected_indices)),
        "score_definition": "L2 norm of adjacent action difference within the same episode",
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
    }
    save_json(info, info_path)

    print("Action-Change Coreset selection finished.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Saved action scores: {scores_path}")
    print(f"Saved selected indices: {selected_path}")
    print(f"Saved selection info: {info_path}")


if __name__ == "__main__":
    main()
