"""Stage 1：ALOHA 数据集读取检查。

输入：Hugging Face / LeRobot 数据集 `lerobot/aloha_sim_transfer_cube_human`。
输出：仅在终端打印数据集类型、字段名、图像信息和 action 信息，不写训练结果。

本阶段的目标是确认课程指定数据集能够通过 LeRobotDataset 正确读取图像。
普通 `datasets.load_dataset` 在本数据集上通常只能看到 state/action 等表格字段，
无法直接获得解码图像；后续需要提取单视角图像特征，因此主读取方式使用
`LeRobotDataset`。本脚本只做数据读取 smoke test，不做训练、特征提取或核心集选择。
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
    """解析 Stage 1 数据集检查脚本的命令行参数。"""
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
    """解析缓存目录；相对路径按项目根目录解释。"""
    path = Path(cache_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    return ensure_dir(path)


def flatten_dict(sample: Any, prefix: str = "") -> dict[str, Any]:
    """将嵌套样本展开为点号 key。

    LeRobotDataset 有时返回扁平 key，如 `observation.images.top`；
    其他加载方式可能返回嵌套 dict。统一展开后，字段检测逻辑更稳健。
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
    """返回包含指定子串的字段名，忽略大小写。"""
    text = text.lower()
    return [key for key in flattened.keys() if text in key.lower()]


def print_value_summary(name: str, value: Any, max_chars: int = 1000) -> None:
    """打印较大对象的简短摘要，避免终端输出过长。"""
    text = repr(value)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    print(f"{name}: {text}")


def print_dataset_basic_info(dataset: Any) -> None:
    """打印数据集类型、长度、元信息、features 和第一个样本字段。"""
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
    """导入 LeRobotDataset。

    LeRobotDataset 能读取视频/图像字段，是本项目后续视觉特征提取的基础。
    如果导入失败，才退回到 datasets.load_dataset，并明确提示该方式可能只有表格字段。
    """
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
    """实例化 LeRobotDataset；失败时打印构造函数签名便于排查版本差异。"""
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
    """根据图像字段类型打印 shape、dtype、范围或摘要信息。"""
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
    """按优先级自动选择一个单视角图像字段。"""
    for candidate in IMAGE_FIELD_PRIORITY:
        if candidate in flattened and not isinstance(flattened[candidate], Mapping):
            return candidate

    for key, value in flattened.items():
        if "image" in key.lower() and not isinstance(value, Mapping):
            return key

    return None


def print_action_info(action_value: Any) -> Any:
    """打印 action 信息，并返回单臂 7 自由度动作标签。

    ALOHA 原始 action 为 14 维，通常对应双臂动作。本课程设计为了降低任务
    复杂度，仅预测单臂 7 自由度动作，因此使用 action[:7]。
    """
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
    """检查 LeRobotDataset 前若干个样本的字段结构和图像/action 信息。"""
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
    """退回 datasets.load_dataset，仅用于查看表格元数据字段。"""
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
