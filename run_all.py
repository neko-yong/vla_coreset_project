"""项目统一运行入口。

该文件将课程设计流程按 stage 封装为命令行入口，便于按阶段复现实验：
数据读取、特征提取、样本选择、MLP 训练评估、可视化和归档。

默认 stage 为 `load`，只做轻量数据读取检查。耗时或可能覆盖结果的任务
（如完整特征提取、训练、归档覆盖）都需要用户显式选择对应 stage，
避免误触发长时间实验或覆盖已有结果。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析 stage 选择参数，保证按阶段安全执行。"""
    parser = argparse.ArgumentParser(description="Run selected VLA coreset project stage.")
    parser.add_argument(
        "--stage",
        default="load",
        choices=(
            "load",
            "extract",
            "extract-full",
            "select-random",
            "select-action",
            "select-visual",
            "select-fusion",
            "select-fusion-neighbor",
            "select-all",
            "select-all-plus",
            "train-random",
            "train-action",
            "train-visual",
            "train-fusion",
            "train-fusion-neighbor",
            "train-full",
            "train-all",
            "train-all-plus",
            "eval-visual",
            "eval-fusion-neighbor",
            "visualize",
            "archive-baseline",
            "archive-visual",
            "archive-fusion-neighbor",
        ),
        help="Stage to run. Default is the lightweight dataset loading check.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Sample limit for --stage extract. Defaults to 100.",
    )
    return parser.parse_args()


def build_commands(project_root: Path, args: argparse.Namespace) -> list[list[str]]:
    """根据 stage 构造要执行的子命令。

    每个 stage 对应实验流程中的一个可复现步骤。使用 subprocess 调用独立脚本，
    可以保持各阶段职责清晰，也便于单独调试。
    """
    if args.stage == "load":
        return [[sys.executable, str(project_root / "src" / "01_load_dataset.py")]]

    if args.stage == "extract":
        max_samples = 100 if args.max_samples is None else args.max_samples
        return [
            [
                sys.executable,
                str(project_root / "src" / "02_extract_features.py"),
                "--max_samples",
                str(max_samples),
            ]
        ]

    if args.stage == "extract-full":
        return [
            [
                sys.executable,
                str(project_root / "src" / "02_extract_features.py"),
                "--force_extract",
            ]
        ]

    selection_scripts = {
        "select-random": ["03_select_random.py"],
        "select-action": ["04_select_action_change.py"],
        "select-visual": ["05b_select_visual_cluster.py"],
        "select-fusion": ["05_select_fusion_coreset.py"],
        "select-fusion-neighbor": ["05c_select_fusion_neighbor.py"],
        "select-all": [
            "03_select_random.py",
            "04_select_action_change.py",
            "05_select_fusion_coreset.py",
        ],
        "select-all-plus": [
            "03_select_random.py",
            "04_select_action_change.py",
            "05_select_fusion_coreset.py",
            "05b_select_visual_cluster.py",
        ],
    }
    if args.stage in selection_scripts:
        return [
            [sys.executable, str(project_root / "src" / script_name)]
            for script_name in selection_scripts[args.stage]
        ]

    if args.stage == "visualize":
        return [[sys.executable, str(project_root / "src" / "08_visualize.py")]]

    if args.stage == "archive-baseline":
        return [[sys.executable, str(project_root / "src" / "09_archive_experiment.py")]]

    if args.stage == "archive-visual":
        return [
            [
                sys.executable,
                str(project_root / "src" / "09_archive_experiment.py"),
                "--experiment_name",
                "baseline_v2_add_visual_cluster",
            ]
        ]

    if args.stage == "archive-fusion-neighbor":
        return [
            [
                sys.executable,
                str(project_root / "src" / "09_archive_experiment.py"),
                "--experiment_name",
                "baseline_v3_add_fusion_neighbor",
            ]
        ]

    train_methods = {
        "train-random": ["random"],
        "train-action": ["action_change"],
        "train-visual": ["visual_cluster"],
        "train-fusion": ["fusion"],
        "train-fusion-neighbor": ["fusion_neighbor"],
        "train-full": ["full"],
        "train-all": ["random", "action_change", "fusion"],
        "train-all-plus": ["random", "action_change", "fusion", "visual_cluster"],
    }
    if args.stage == "eval-visual":
        return [
            [
                sys.executable,
                str(project_root / "src" / "07_evaluate.py"),
                "--method",
                "visual_cluster",
            ]
        ]
    if args.stage == "eval-fusion-neighbor":
        return [
            [
                sys.executable,
                str(project_root / "src" / "07_evaluate.py"),
                "--method",
                "fusion_neighbor",
            ]
        ]

    return [
        [
            sys.executable,
            str(project_root / "src" / "06_train_mlp.py"),
            "--method",
            method,
        ]
        for method in train_methods[args.stage]
    ]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    commands = build_commands(project_root, args)

    print(f"Running stage: {args.stage}")
    for command in commands:
        print(" ".join(command))
        try:
            subprocess.run(command, cwd=project_root, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\nERROR: stage '{args.stage}' failed with exit code {exc.returncode}.")
            print("Please review the messages above.")
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
