"""Stage 3a：Random 10% 基准采样。

输入：`outputs/features/` 中的特征、动作和 episode id。
输出：`selected_indices_random.npy` 与 `random_selection_info.json`。

Random 是最重要的基准方法之一，用来衡量“没有认知筛选”的 10% 数据表现。
采样只允许发生在训练 episode 内，测试 episode 绝不能参与采样。固定 seed=42
用于保证随机选择可复现。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    ensure_dir,
    get_project_root,
    get_train_test_masks,
    load_feature_arrays,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析 Random 基准采样参数。"""
    parser = argparse.ArgumentParser(description="Select random training frames.")
    parser.add_argument("--feature_dir", default="outputs/features", help="Stage 2 feature directory.")
    parser.add_argument("--output_dir", default="outputs/results", help="Directory for selection files.")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of train frames to select.")
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_dir = resolve_project_path(args.feature_dir)
    output_dir = ensure_dir(resolve_project_path(args.output_dir))
    arrays = load_feature_arrays(feature_dir)

    episode_ids = arrays["episode_ids"]
    train_mask, test_mask, train_episodes, test_episodes = get_train_test_masks(episode_ids)
    train_indices = np.flatnonzero(train_mask)
    budget = selection_budget(len(train_indices), args.sample_ratio)

    # 只从训练集 frame 中随机选择；测试集保留到最终 MSE 评估，避免信息泄漏。
    rng = np.random.default_rng(args.seed)
    selected_indices = np.sort(rng.choice(train_indices, size=budget, replace=False))

    selected_path = output_dir / "selected_indices_random.npy"
    info_path = output_dir / "random_selection_info.json"
    np.save(selected_path, selected_indices.astype(np.int64))

    info: dict[str, Any] = {
        "method": "random",
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "num_total_samples": int(len(episode_ids)),
        "num_train_samples": int(train_mask.sum()),
        "num_test_samples": int(test_mask.sum()),
        "num_selected_samples": int(len(selected_indices)),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
    }
    save_json(info, info_path)

    print("Random selection finished.")
    print(f"Train samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Saved selected indices: {selected_path}")
    print(f"Saved selection info: {info_path}")


if __name__ == "__main__":
    main()
