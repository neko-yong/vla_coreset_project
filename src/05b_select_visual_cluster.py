"""Stage 3d：Visual-Cluster Only 消融采样。

输入：Stage 2 的 ResNet18 特征和 episode/frame/timestamp 元信息。
输出：visual_cluster 聚类 id、选中样本索引、JSON 说明和样本表。

Visual-Cluster Only 是 Fusion 的消融实验，只保留“视觉状态覆盖”这一部分，
不使用动作变化分数。它通过训练集特征 KMeans 聚类分配采样预算，簇内随机选样。
该方法用于验证：仅靠视觉状态覆盖是否已经能提升核心集质量。
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
    ensure_dir,
    get_project_root,
    get_train_test_masks,
    load_feature_arrays,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析 Visual-Cluster Only 采样参数。"""
    parser = argparse.ArgumentParser(description="Run Visual-Cluster Only Coreset selection.")
    parser.add_argument("--feature_dir", default="outputs/features", help="Stage 2 feature directory.")
    parser.add_argument("--output_dir", default="outputs/results", help="Directory for selection files.")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of train frames to select.")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of visual feature clusters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """解析项目路径；相对路径按项目根目录解释。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def selection_budget(num_train_samples: int, sample_ratio: float) -> int:
    """根据训练集规模和采样比例计算精确采样数量。"""
    if not 0 < sample_ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}.")
    return max(1, int(round(num_train_samples * sample_ratio)))


def select_random_by_cluster_quota(
    train_indices: np.ndarray,
    train_cluster_ids: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """在每个视觉簇内按配额随机采样训练样本。

    与 Fusion 的区别在于：这里簇内不按 action_change_score 排序，
    因而可以作为“视觉覆盖本身”的消融对照。
    """
    quotas = allocate_cluster_quotas(train_cluster_ids, budget)
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    cluster_sizes: dict[int, int] = {}
    selected_counts: dict[int, int] = {}

    for cluster_id in sorted(int(cluster) for cluster in np.unique(train_cluster_ids)):
        local_positions = np.flatnonzero(train_cluster_ids == cluster_id)
        cluster_sizes[cluster_id] = int(len(local_positions))
        quota = int(quotas.get(cluster_id, 0))
        if quota <= 0:
            selected_counts[cluster_id] = 0
            continue
        local_global_indices = train_indices[local_positions]
        chosen = rng.choice(local_global_indices, size=min(quota, len(local_global_indices)), replace=False)
        selected_parts.append(chosen.astype(np.int64))
        selected_counts[cluster_id] = int(len(chosen))

    selected = (
        np.concatenate(selected_parts).astype(np.int64)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    selected_set = set(int(index) for index in selected)

    if len(selected) < budget:
        remaining = np.array(
            [int(index) for index in train_indices if int(index) not in selected_set],
            dtype=np.int64,
        )
        fillers = rng.choice(remaining, size=budget - len(selected), replace=False)
        selected = np.concatenate([selected, fillers.astype(np.int64)])
        for filler in fillers:
            cluster_id = int(train_cluster_ids[np.flatnonzero(train_indices == int(filler))[0]])
            selected_counts[cluster_id] = selected_counts.get(cluster_id, 0) + 1

    if len(selected) > budget:
        keep = rng.choice(selected, size=budget, replace=False)
        selected = keep.astype(np.int64)
        selected_set = set(int(index) for index in selected)
        selected_counts = {}
        for cluster_id in sorted(int(cluster) for cluster in np.unique(train_cluster_ids)):
            local_global_indices = train_indices[train_cluster_ids == cluster_id]
            selected_counts[cluster_id] = int(
                sum(int(index) in selected_set for index in local_global_indices)
            )

    return np.sort(selected.astype(np.int64)), cluster_sizes, selected_counts


def build_sample_table(
    train_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    cluster_ids_full: np.ndarray,
    selected_indices: np.ndarray,
) -> pd.DataFrame:
    """构建 Visual-Cluster 的训练候选样本表。"""
    selected_set = set(int(index) for index in selected_indices)
    return pd.DataFrame(
        {
            "global_index": train_indices.astype(int),
            "episode_id": episode_ids[train_indices].astype(int),
            "frame_id": frame_ids[train_indices].astype(int),
            "timestamp": timestamps[train_indices],
            "cluster_id": cluster_ids_full[train_indices].astype(int),
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

    # KMeans 只 fit 训练集特征；测试集不能参与聚类、采样或配额分配。
    print("Fitting KMeans on training features only.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Target selected samples: {budget}")
    print(f"Num clusters: {args.num_clusters}")

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(features[train_indices])

    selected_indices, cluster_sizes, selected_counts = select_random_by_cluster_quota(
        train_indices=train_indices,
        train_cluster_ids=train_cluster_ids,
        budget=budget,
        seed=args.seed,
    )

    if len(selected_indices) != budget:
        raise RuntimeError(
            f"Visual-Cluster selection must return exactly {budget} samples, "
            f"got {len(selected_indices)}."
        )
    if np.any(test_mask[selected_indices]):
        raise RuntimeError("Visual-Cluster selection error: test samples were selected.")

    cluster_ids_full = np.full(len(episode_ids), fill_value=-1, dtype=np.int32)
    cluster_ids_full[train_indices] = train_cluster_ids.astype(np.int32)

    cluster_path = output_dir / "visual_cluster_ids.npy"
    selected_path = output_dir / "selected_indices_visual_cluster.npy"
    info_path = output_dir / "visual_cluster_selection_info.json"
    table_path = output_dir / "visual_cluster_sample_table.csv"

    np.save(cluster_path, cluster_ids_full)
    np.save(selected_path, selected_indices.astype(np.int64))

    sample_table = build_sample_table(
        train_indices=train_indices,
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        timestamps=timestamps,
        cluster_ids_full=cluster_ids_full,
        selected_indices=selected_indices,
    )
    sample_table.to_csv(table_path, index=False)

    info: dict[str, Any] = {
        "method": "visual_cluster",
        "algorithm_name": "Visual-Cluster Only Coreset",
        "sample_ratio": args.sample_ratio,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "num_total_samples": int(len(episode_ids)),
        "num_train_samples": int(train_mask.sum()),
        "num_test_samples": int(test_mask.sum()),
        "num_selected_samples": int(len(selected_indices)),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "explanation": "visual clustering preserves state coverage without using action-change scores.",
    }
    save_json(info, info_path)

    print("\nVisual-Cluster Only selection finished.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Target selected samples: {budget}")
    print(f"Actual selected samples: {len(selected_indices)}")
    print("Cluster sample counts:")
    for cluster_id in sorted(cluster_sizes):
        print(
            f"  cluster {cluster_id}: "
            f"samples={cluster_sizes[cluster_id]}, selected={selected_counts.get(cluster_id, 0)}"
        )
    print(f"Saved cluster ids: {cluster_path}")
    print(f"Saved selected indices: {selected_path}")
    print(f"Saved sample table: {table_path}")
    print(f"Saved selection info: {info_path}")


if __name__ == "__main__":
    main()
