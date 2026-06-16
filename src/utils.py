"""项目通用工具函数。

本文件不对应单独实验阶段，而是为 Stage 1-9 提供公共能力：
- 固定随机种子，保证样本选择和训练结果可复现；
- 统一创建目录、读写 JSON、定位项目根目录；
- 读取 Stage 2 生成的特征数组；
- 生成固定 train/test episode 划分；
- 计算动作变化分数与簇采样预算。

这些函数集中放置，可以减少各阶段脚本中的重复代码，并降低不同方法
在数据划分或分数计算上出现不一致的风险。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。

    核心集选择和 MLP 训练都涉及随机过程。固定 seed=42 可以保证课程设计
    实验在同一环境下可复现，便于不同采样方法做公平比较。
    """
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
        # 即使完整依赖尚未安装，也允许工具函数被导入使用。
        pass


def ensure_dir(path: str | Path) -> Path:
    """目录不存在时自动创建，并以 Path 形式返回。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """返回项目根目录，避免依赖本机绝对路径。"""
    return Path(__file__).resolve().parents[1]


def save_json(data: Any, path: str | Path) -> None:
    """保存 JSON 文件，并自动创建父目录。"""
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """从磁盘读取 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_arrays(feature_dir: str | Path) -> dict[str, Any]:
    """读取 Stage 2 生成的特征数组和划分元信息。"""
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
    """创建固定的 episode 级训练/测试划分。

    Episodes 按 id 排序，前 80% 作为训练集，后 20% 作为测试集。
    在核心集选择任务中，测试集只能用于最终评估，不能参与采样、聚类、
    标准化拟合或预算分配，避免信息泄漏。
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
    """计算同一 episode 内相邻帧动作变化分数。

    每个 episode 的第一帧分数设为 0，不同 episode 之间不能相减。
    该分数可理解为预测编码中的 prediction error / surprise：
    相邻动作变化越大，该时刻包含的控制信息通常越强。
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
    """按簇大小分配严格的采样预算。

    配额与簇内训练样本数成比例；在预算允许时，每个非空簇至少保留 1 个样本。
    这用于“视觉状态覆盖”思想：大簇应获得更多名额，小簇也尽量不被完全忽略。
    返回的所有配额之和严格等于 total_budget。
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
    """推断 tensor、array、PIL 图像或列表的可读 shape。"""
    if hasattr(value, "shape"):
        return tuple(value.shape)

    if hasattr(value, "size") and hasattr(value, "mode"):
        width, height = value.size
        return (height, width, len(value.getbands()))

    if isinstance(value, (list, tuple)):
        return np.asarray(value).shape

    return "unknown"


def to_numpy(value: Any) -> np.ndarray:
    """将常见样本值转换为 NumPy 数组，兼容 torch / numpy / list。"""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    if hasattr(value, "__array__"):
        return np.asarray(value)

    return np.asarray(value)
