"""Stage 3c: Coverage-aware Action-Surprise Coreset Selection.

The method clusters training visual features for state coverage, then selects
high action-change frames inside each cluster. Action change corresponds to
prediction error / action surprise, while clustering avoids selecting only a
small set of visually similar high-motion frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils import (
    allocate_cluster_quotas,
    compute_action_change_scores,
    ensure_dir,
    get_project_root,
    get_train_test_masks,
    load_feature_arrays,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse Fusion Coreset arguments."""
    parser = argparse.ArgumentParser(
        description="Run Coverage-aware Action-Surprise Coreset Selection."
    )
    parser.add_argument("--feature_dir", default="outputs/features", help="Stage 2 feature directory.")
    parser.add_argument("--output_dir", default="outputs/results", help="Directory for selection files.")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of train frames to select.")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of visual feature clusters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def select_by_cluster_quota(
    train_indices: np.ndarray,
    train_cluster_ids: np.ndarray,
    train_scores: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Select high-score samples within proportional cluster quotas."""
    quotas = allocate_cluster_quotas(train_cluster_ids, budget)
    selected_parts: list[np.ndarray] = []

    for cluster_id, quota in quotas.items():
        local_indices = np.flatnonzero(train_cluster_ids == cluster_id)
        local_scores = train_scores[local_indices]
        local_global_indices = train_indices[local_indices]
        order = np.lexsort((local_global_indices, -local_scores))
        selected_parts.append(local_global_indices[order[:quota]])

    if selected_parts:
        selected = np.concatenate(selected_parts).astype(np.int64)
    else:
        selected = np.empty(0, dtype=np.int64)

    selected_set = set(int(index) for index in selected)
    if len(selected) < budget:
        order = np.lexsort((train_indices, -train_scores))
        fillers = [int(train_indices[i]) for i in order if int(train_indices[i]) not in selected_set]
        selected = np.concatenate(
            [selected, np.asarray(fillers[: budget - len(selected)], dtype=np.int64)]
        )

    if len(selected) > budget:
        selected_scores = train_scores[np.searchsorted(train_indices, selected)]
        order = np.lexsort((selected, -selected_scores))
        selected = selected[order[:budget]]

    return np.sort(selected.astype(np.int64))


def build_sample_table(
    train_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    cluster_ids_full: np.ndarray,
    action_scores_full: np.ndarray,
    selected_indices: np.ndarray,
) -> pd.DataFrame:
    """Build the Fusion Coreset sample table for training candidates."""
    selected_set = set(int(index) for index in selected_indices)
    return pd.DataFrame(
        {
            "global_index": train_indices.astype(int),
            "episode_id": episode_ids[train_indices].astype(int),
            "frame_id": frame_ids[train_indices].astype(int),
            "timestamp": timestamps[train_indices],
            "cluster_id": cluster_ids_full[train_indices].astype(int),
            "action_score": action_scores_full[train_indices],
            "selected": [int(index) in selected_set for index in train_indices],
        }
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {args.num_clusters}.")

    feature_dir = resolve_project_path(args.feature_dir)
    output_dir = ensure_dir(resolve_project_path(args.output_dir))
    arrays = load_feature_arrays(feature_dir)

    features = arrays["features"]
    actions = arrays["actions"]
    episode_ids = arrays["episode_ids"]
    frame_ids = arrays["frame_ids"]
    timestamps = arrays["timestamps"]

    train_mask, test_mask, train_episodes, test_episodes = get_train_test_masks(episode_ids)
    train_indices = np.flatnonzero(train_mask)
    budget = selection_budget(len(train_indices), args.sample_ratio)

    if args.num_clusters > len(train_indices):
        raise ValueError(
            f"num_clusters={args.num_clusters} cannot exceed train samples={len(train_indices)}."
        )

    train_features = features[train_indices]
    print("Fitting KMeans on training features only.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Num clusters: {args.num_clusters}")
    print(f"Selection budget: {budget}")

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(train_features)

    action_scores_full = np.zeros(len(episode_ids), dtype=np.float32)
    train_scores = compute_action_change_scores(actions[train_indices], episode_ids[train_indices])
    action_scores_full[train_indices] = train_scores

    cluster_ids_full = np.full(len(episode_ids), fill_value=-1, dtype=np.int32)
    cluster_ids_full[train_indices] = train_cluster_ids.astype(np.int32)

    selected_indices = select_by_cluster_quota(
        train_indices=train_indices,
        train_cluster_ids=train_cluster_ids,
        train_scores=train_scores,
        budget=budget,
    )

    if len(selected_indices) != budget:
        raise RuntimeError(
            f"Fusion selection must return exactly {budget} samples, got {len(selected_indices)}."
        )
    if np.any(test_mask[selected_indices]):
        raise RuntimeError("Fusion selection error: test samples were selected.")

    cluster_path = output_dir / "fusion_cluster_ids.npy"
    score_path = output_dir / "fusion_action_scores.npy"
    selected_path = output_dir / "selected_indices_fusion.npy"
    info_path = output_dir / "fusion_selection_info.json"
    table_path = output_dir / "fusion_sample_table.csv"

    np.save(cluster_path, cluster_ids_full)
    np.save(score_path, action_scores_full)
    np.save(selected_path, selected_indices.astype(np.int64))

    sample_table = build_sample_table(
        train_indices=train_indices,
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        timestamps=timestamps,
        cluster_ids_full=cluster_ids_full,
        action_scores_full=action_scores_full,
        selected_indices=selected_indices,
    )
    sample_table.to_csv(table_path, index=False)

    info: dict[str, Any] = {
        "method": "fusion_coreset",
        "algorithm_name": "Coverage-aware Action-Surprise Coreset Selection",
        "sample_ratio": args.sample_ratio,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "num_total_samples": int(len(episode_ids)),
        "num_train_samples": int(train_mask.sum()),
        "num_test_samples": int(test_mask.sum()),
        "num_selected_samples": int(len(selected_indices)),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "explanation": (
            "visual clustering ensures state coverage; action change corresponds "
            "to prediction error / action surprise."
        ),
    }
    save_json(info, info_path)

    print("Fusion Coreset selection finished.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Saved cluster ids: {cluster_path}")
    print(f"Saved action scores: {score_path}")
    print(f"Saved selected indices: {selected_path}")
    print(f"Saved sample table: {table_path}")
    print(f"Saved selection info: {info_path}")


if __name__ == "__main__":
    main()
