"""Project entry point for staged course-design steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the stage selector for safe staged execution."""
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
            "select-all",
            "select-all-plus",
            "train-random",
            "train-action",
            "train-visual",
            "train-fusion",
            "train-full",
            "train-all",
            "train-all-plus",
            "eval-visual",
            "visualize",
            "archive-baseline",
            "archive-visual",
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
    """Build subprocess commands for the selected stage."""
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

    train_methods = {
        "train-random": ["random"],
        "train-action": ["action_change"],
        "train-visual": ["visual_cluster"],
        "train-fusion": ["fusion"],
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
