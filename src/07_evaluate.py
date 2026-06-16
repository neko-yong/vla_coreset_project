"""Stage 4 辅助脚本：加载 checkpoint 并重新评估测试集 MSE。

输入：`outputs/checkpoints/mlp_{method}.pt` 与 Stage 2 特征数组。
输出：更新对应 `eval_{method}.json` 和 `results.csv` 中该方法的评估行。

该脚本不重新训练模型，只复用已保存的 MLP 权重和训练阶段保存的 scaler 参数。
评估时必须使用与训练相同的固定 train/test episode 划分，且测试集仍不能参与
标准化拟合。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import torch

from utils import ensure_dir, get_project_root, save_json, set_seed


def load_train_module() -> Any:
    """动态加载训练模块。

    文件名 `06_train_mlp.py` 以数字开头，不能用普通 import 语法直接导入。
    这里通过 importlib 复用 MLP 类、scaler 处理和 evaluate_model，避免评估逻辑重复。
    """
    module_path = Path(__file__).resolve().parent / "06_train_mlp.py"
    spec = importlib.util.spec_from_file_location("stage4_train_mlp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """解析独立评估脚本参数。"""
    parser = argparse.ArgumentParser(description="Evaluate a saved MLP checkpoint.")
    parser.add_argument(
        "--method",
        default="random",
        choices=("random", "action_change", "visual_cluster", "fusion", "fusion_neighbor", "full"),
    )
    parser.add_argument("--feature_dir", default="outputs/features")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints")
    parser.add_argument("--result_dir", default="outputs/results")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """解析项目路径；相对路径按项目根目录解释。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def main() -> None:
    args = parse_args()
    set_seed(42)
    train_module = load_train_module()

    feature_dir = resolve_project_path(args.feature_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    result_dir = ensure_dir(resolve_project_path(args.result_dir))
    checkpoint_path = checkpoint_dir / f"mlp_{args.method}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = train_module.choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data = train_module.load_arrays_and_split(feature_dir)
    selected_train_indices = train_module.get_method_train_indices(
        method=args.method,
        train_indices=data["train_indices"],
        train_mask=data["train_mask"],
        result_dir=result_dir,
    )

    x_test_raw = data["features"][data["test_indices"]]
    y_test = data["actions"][data["test_indices"]]
    x_test = train_module.apply_checkpoint_scaler(x_test_raw, checkpoint)
    test_loader = train_module.make_loader(
        x_test,
        y_test,
        batch_size=int(checkpoint.get("batch_size", 128)),
        shuffle=False,
    )

    model = train_module.MLPRegressor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = train_module.evaluate_model(model, test_loader, device)

    sample_ratio = checkpoint.get(
        "sample_ratio",
        train_module.sample_ratio_for_method(
            args.method,
            num_train_samples=len(selected_train_indices),
            full_train_count=len(data["train_indices"]),
        ),
    )
    eval_info = {
        "method": args.method,
        "sample_ratio": float(sample_ratio),
        "num_train_samples": int(checkpoint.get("num_train_samples", len(selected_train_indices))),
        "num_test_samples": int(len(data["test_indices"])),
        **metrics,
        "epochs": int(checkpoint.get("epochs", 0)),
        "batch_size": int(checkpoint.get("batch_size", 128)),
        "lr": float(checkpoint.get("lr", 0.0)),
        "seed": int(checkpoint.get("seed", 42)),
        "checkpoint_path": str(checkpoint_path),
        "standardize_features": bool(checkpoint.get("standardize_features", True)),
    }
    eval_path = result_dir / f"eval_{args.method}.json"
    save_json(eval_info, eval_path)
    train_module.update_results_csv(result_dir, eval_info)

    print("Evaluation finished.")
    print(f"Method: {args.method}")
    print(f"Test samples: {len(data['test_indices'])}")
    print(f"Test MSE: {metrics['test_mse']:.8f}")
    print(f"Saved eval json: {eval_path}")


if __name__ == "__main__":
    main()
