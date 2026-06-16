"""Stage 3e：Fusion + Temporal Neighbor Coreset。

输入：Stage 2 的 ResNet18 特征、动作标签和 episode/frame 元信息。
输出：fusion_neighbor 聚类 id、动作分数、选中样本索引、JSON 说明和样本表。

该方法先按 Fusion 思路选择高动作惊奇度 anchor frame，再加入同一 episode 内
的局部邻域帧。机械臂动作具有局部时间连续性，关键帧前后的 t-1/t+1 可能包含
有用动作上下文。为公平比较，总预算仍固定为训练集 10%；邻域扩展不能跨 episode，
也不能加入测试集。若超预算则优先保留 anchor，再保留 neighbor；若不足预算则
从训练集未选样本中按动作惊奇度补齐。
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
    """解析 Fusion + Temporal Neighbor 采样参数。"""
    parser = argparse.ArgumentParser(description="Run Fusion + Temporal Neighbor Coreset selection.")
    parser.add_argument("--feature_dir", default="outputs/features", help="Stage 2 feature directory.")
    parser.add_argument("--output_dir", default="outputs/results", help="Directory for selection files.")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of train frames to select.")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of visual feature clusters.")
    parser.add_argument("--neighbor_window", type=int, default=1, help="Temporal neighbor radius around anchors.")
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


def anchor_budget_from_neighbor_window(total_budget: int, neighbor_window: int) -> int:
    """根据邻域窗口设置 anchor 预算。

    如果直接选择 1600 个 anchor，再扩展邻域会全部被裁剪回 anchor，
    方法会退化为 Fusion。因此先减少 anchor 数量，为 t-1/t+1 邻域留出预算。
    """
    neighbor_span = 2 * neighbor_window + 1
    if neighbor_span <= 1:
        return total_budget
    return max(1, int(np.ceil(total_budget / neighbor_span)))


def select_fusion_anchors(
    train_indices: np.ndarray,
    train_cluster_ids: np.ndarray,
    train_scores: np.ndarray,
    budget: int,
) -> np.ndarray:
    """按 Fusion 规则选择 anchor：簇配额控制覆盖，动作分数控制信息量。"""
    quotas = allocate_cluster_quotas(train_cluster_ids, budget)
    selected_parts: list[np.ndarray] = []

    for cluster_id, quota in quotas.items():
        local_positions = np.flatnonzero(train_cluster_ids == cluster_id)
        local_scores = train_scores[local_positions]
        local_global_indices = train_indices[local_positions]
        order = np.lexsort((local_global_indices, -local_scores))
        selected_parts.append(local_global_indices[order[:quota]])

    anchors = (
        np.concatenate(selected_parts).astype(np.int64)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    if len(anchors) > budget:
        anchor_positions = np.searchsorted(train_indices, anchors)
        anchor_scores = train_scores[anchor_positions]
        order = np.lexsort((anchors, -anchor_scores))
        anchors = anchors[order[:budget]]
    return np.sort(anchors.astype(np.int64))


def build_episode_frame_lookup(
    train_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
) -> dict[tuple[int, int], int]:
    """建立训练集内 (episode_id, frame_id) 到全局索引的映射。"""
    lookup: dict[tuple[int, int], int] = {}
    for index in train_indices:
        lookup[(int(episode_ids[index]), int(frame_ids[index]))] = int(index)
    return lookup


def expand_neighbors(
    anchors: np.ndarray,
    train_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    neighbor_window: int,
) -> np.ndarray:
    """在同一训练 episode 内加入 anchor 的时间邻域。

    查找使用 (episode_id, frame_id)，因此不会跨 episode；lookup 只由训练集
    样本构建，因此测试集帧不可能被加入。
    """
    lookup = build_episode_frame_lookup(train_indices, episode_ids, frame_ids)
    neighbors: set[int] = set()
    for anchor in anchors.astype(int):
        episode_id = int(episode_ids[anchor])
        frame_id = int(frame_ids[anchor])
        for offset in range(-neighbor_window, neighbor_window + 1):
            candidate = lookup.get((episode_id, frame_id + offset))
            if candidate is not None and candidate != anchor:
                neighbors.add(candidate)
    return np.asarray(sorted(neighbors), dtype=np.int64)


def finalize_selection(
    anchors: np.ndarray,
    neighbors: np.ndarray,
    train_indices: np.ndarray,
    train_scores: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 anchor、neighbor 和补齐样本合并为严格预算大小的选择集。

    优先级为 anchor > neighbor > 高 action_change_score 补齐样本。
    这样既保留关键帧，又尽量加入局部上下文，同时保证所有方法样本数一致。
    """
    anchor_set = set(int(index) for index in anchors)
    ordered_neighbors = np.asarray(
        [int(index) for index in neighbors if int(index) not in anchor_set],
        dtype=np.int64,
    )
    if len(ordered_neighbors) > 0:
        neighbor_positions = np.searchsorted(train_indices, ordered_neighbors)
        neighbor_scores = train_scores[neighbor_positions]
        order = np.lexsort((ordered_neighbors, -neighbor_scores))
        ordered_neighbors = ordered_neighbors[order]

    selected: list[int] = []
    selected.extend(int(index) for index in anchors[:budget])

    if len(selected) < budget:
        remaining_slots = budget - len(selected)
        selected.extend(int(index) for index in ordered_neighbors[:remaining_slots])

    selected_set = set(selected)
    if len(selected) < budget:
        order = np.lexsort((train_indices, -train_scores))
        for position in order:
            candidate = int(train_indices[position])
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) == budget:
                    break

    if len(selected) != budget:
        raise RuntimeError(f"Expected {budget} selected samples, got {len(selected)}.")

    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    anchors_kept = np.asarray(sorted(index for index in anchors if int(index) in selected_set), dtype=np.int64)
    neighbors_kept = np.asarray(
        sorted(index for index in ordered_neighbors if int(index) in selected_set),
        dtype=np.int64,
    )
    return selected_array, anchors_kept, neighbors_kept


def build_sample_table(
    train_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    cluster_ids_full: np.ndarray,
    action_scores_full: np.ndarray,
    selected_indices: np.ndarray,
    anchors_kept: np.ndarray,
    neighbors_kept: np.ndarray,
) -> pd.DataFrame:
    """构建 Fusion + Neighbor 的训练候选样本表。"""
    selected_set = set(int(index) for index in selected_indices)
    anchor_set = set(int(index) for index in anchors_kept)
    neighbor_set = set(int(index) for index in neighbors_kept)
    return pd.DataFrame(
        {
            "global_index": train_indices.astype(int),
            "episode_id": episode_ids[train_indices].astype(int),
            "frame_id": frame_ids[train_indices].astype(int),
            "timestamp": timestamps[train_indices],
            "cluster_id": cluster_ids_full[train_indices].astype(int),
            "action_score": action_scores_full[train_indices],
            "selected": [int(index) in selected_set for index in train_indices],
            "is_anchor": [int(index) in anchor_set for index in train_indices],
            "is_neighbor": [int(index) in neighbor_set for index in train_indices],
        }
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {args.num_clusters}.")
    if args.neighbor_window < 0:
        raise ValueError(f"neighbor_window must be non-negative, got {args.neighbor_window}.")

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
    anchor_budget = anchor_budget_from_neighbor_window(budget, args.neighbor_window)
    if args.num_clusters > len(train_indices):
        raise ValueError(
            f"num_clusters={args.num_clusters} cannot exceed train samples={len(train_indices)}."
        )

    print("Fitting KMeans on training features only.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Target selected samples: {budget}")
    print(f"Anchor budget before neighbor expansion: {anchor_budget}")
    print(f"Neighbor window: {args.neighbor_window}")

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(features[train_indices])

    train_scores = compute_action_change_scores(actions[train_indices], episode_ids[train_indices])
    action_scores_full = np.zeros(len(episode_ids), dtype=np.float32)
    action_scores_full[train_indices] = train_scores

    cluster_ids_full = np.full(len(episode_ids), fill_value=-1, dtype=np.int32)
    cluster_ids_full[train_indices] = train_cluster_ids.astype(np.int32)

    anchors = select_fusion_anchors(train_indices, train_cluster_ids, train_scores, anchor_budget)
    neighbors = expand_neighbors(
        anchors=anchors,
        train_indices=train_indices,
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        neighbor_window=args.neighbor_window,
    )
    selected_indices, anchors_kept, neighbors_kept = finalize_selection(
        anchors=anchors,
        neighbors=neighbors,
        train_indices=train_indices,
        train_scores=train_scores,
        budget=budget,
    )

    if np.any(test_mask[selected_indices]):
        raise RuntimeError("Fusion + Neighbor selection error: test samples were selected.")

    cluster_path = output_dir / "fusion_neighbor_cluster_ids.npy"
    score_path = output_dir / "fusion_neighbor_action_scores.npy"
    selected_path = output_dir / "selected_indices_fusion_neighbor.npy"
    info_path = output_dir / "fusion_neighbor_selection_info.json"
    table_path = output_dir / "fusion_neighbor_sample_table.csv"

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
        anchors_kept=anchors_kept,
        neighbors_kept=neighbors_kept,
    )
    sample_table.to_csv(table_path, index=False)

    info: dict[str, Any] = {
        "method": "fusion_neighbor",
        "algorithm_name": "Fusion + Temporal Neighbor Coreset",
        "sample_ratio": args.sample_ratio,
        "num_clusters": args.num_clusters,
        "neighbor_window": args.neighbor_window,
        "seed": args.seed,
        "num_total_samples": int(len(episode_ids)),
        "num_train_samples": int(train_mask.sum()),
        "num_test_samples": int(test_mask.sum()),
        "anchor_budget": int(anchor_budget),
        "num_anchor_candidates": int(len(anchors)),
        "num_anchors_kept": int(len(anchors_kept)),
        "num_neighbors_kept": int(len(neighbors_kept)),
        "num_selected_samples": int(len(selected_indices)),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "explanation": (
            "Fusion anchors are selected by visual cluster coverage and action surprise; "
            "temporal neighbors are added only within the same training episode."
        ),
    }
    save_json(info, info_path)

    print("\nFusion + Temporal Neighbor selection finished.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Target selected samples: {budget}")
    print(f"Anchor budget before neighbor expansion: {anchor_budget}")
    print(f"Actual selected samples: {len(selected_indices)}")
    print(f"Anchors kept: {len(anchors_kept)}")
    print(f"Neighbors kept: {len(neighbors_kept)}")
    print(f"Saved cluster ids: {cluster_path}")
    print(f"Saved action scores: {score_path}")
    print(f"Saved selected indices: {selected_path}")
    print(f"Saved sample table: {table_path}")
    print(f"Saved selection info: {info_path}")


if __name__ == "__main__":
    main()
