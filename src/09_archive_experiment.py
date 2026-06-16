"""Stage 6：实验版本归档脚本。

输入：当前 `outputs/results/`、`outputs/figures/`、`outputs/checkpoints/`、
`report_assets/` 和 README。
输出：`experiments/{experiment_name}/` 下的结果副本、experiment_note.md、
file_manifest.json 和 README_snapshot.md。

该脚本只复制和整理已有结果，不重新运行训练、特征提取或样本选择。归档用于保存
baseline_v1/v2/v3 等不同实验版本，便于后续优化实验与历史结果做可追溯比较。
`file_manifest.json` 记录归档文件清单和大小，`experiment_note.md` 记录实验目的、
方法差异和自动生成的结果观察。默认不覆盖已有归档，只有传入 --overwrite 才会重建。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils import ensure_dir, get_project_root


DEFAULT_EXPERIMENT_NAME = "baseline_v1_random_action_fusion"
VISUAL_CLUSTER_EXPERIMENT_NAME = "baseline_v2_add_visual_cluster"
FUSION_NEIGHBOR_EXPERIMENT_NAME = "baseline_v3_add_fusion_neighbor"


def parse_args() -> argparse.Namespace:
    """解析归档脚本参数。"""
    parser = argparse.ArgumentParser(description="Archive current baseline experiment results.")
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing archive directory and recreate it.",
    )
    return parser.parse_args()


def copy_directory(src: Path, dst: Path) -> bool:
    """复制目录树；源目录缺失时只警告并继续。

    归档脚本应尽量稳健：某些可选目录不存在时，不应影响其他结果的保存。
    """
    if not src.exists():
        print(f"WARNING: source directory does not exist, skipped: {src}")
        return False
    shutil.copytree(src, dst)
    print(f"Copied directory: {src} -> {dst}")
    return True


def copy_file(src: Path, dst: Path) -> bool:
    """复制单个文件；源文件缺失时只警告并继续。"""
    if not src.exists():
        print(f"WARNING: source file does not exist, skipped: {src}")
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    print(f"Copied file: {src} -> {dst}")
    return True


def format_float(value: Any) -> str:
    """将数值格式化为报告表格中的字符串。"""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """生成简单 Markdown 表格，避免依赖可选的 tabulate 包。"""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def build_results_table(results: pd.DataFrame) -> str:
    """从 results.csv 生成 Markdown 结果表，并将 MSE 保留 6 位小数。"""
    columns = [
        "method",
        "sample_ratio",
        "num_train_samples",
        "num_test_samples",
        "test_mse",
        "joint_1_mse",
        "joint_2_mse",
        "joint_3_mse",
        "joint_4_mse",
        "joint_5_mse",
        "joint_6_mse",
        "joint_7_mse",
    ]
    missing = [column for column in columns if column not in results.columns]
    if missing:
        raise ValueError(f"results.csv is missing required columns: {missing}")

    table = results[columns].copy()
    mse_columns = ["test_mse"] + [f"joint_{idx}_mse" for idx in range(1, 8)]
    for column in mse_columns:
        table[column] = table[column].map(format_float)
    return dataframe_to_markdown(table)


def get_common_setting(results: pd.DataFrame, column: str, default: str = "not recorded") -> str:
    """从 results.csv 中读取通用训练设置；不存在时返回默认文本。"""
    if column not in results.columns or results.empty:
        return default
    values = results[column].dropna().unique()
    if len(values) == 0:
        return default
    if len(values) == 1:
        value = values[0]
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return ", ".join(str(value) for value in values)


def build_observation(results: pd.DataFrame) -> str:
    """根据 results.csv 自动生成 baseline_v1 观察结论。"""
    if "method" not in results.columns or "test_mse" not in results.columns:
        raise ValueError("results.csv must contain method and test_mse columns.")

    best_row = results.loc[results["test_mse"].idxmin()]
    worst_row = results.loc[results["test_mse"].idxmax()]
    lines = [
        f"- Lowest test MSE: `{best_row['method']}` ({float(best_row['test_mse']):.6f}).",
        f"- Highest test MSE: `{worst_row['method']}` ({float(worst_row['test_mse']):.6f}).",
    ]

    by_method = results.set_index("method")
    if {"fusion", "action_change"}.issubset(by_method.index):
        fusion = float(by_method.loc["fusion", "test_mse"])
        action = float(by_method.loc["action_change", "test_mse"])
        verdict = "improves over" if fusion < action else "does not improve over"
        lines.append(
            f"- Fusion {verdict} Action-Change "
            f"({fusion:.6f} vs. {action:.6f})."
        )
    if {"fusion", "random"}.issubset(by_method.index):
        fusion = float(by_method.loc["fusion", "test_mse"])
        random = float(by_method.loc["random", "test_mse"])
        verdict = "improves over" if fusion < random else "does not improve over"
        lines.append(f"- Fusion {verdict} Random ({fusion:.6f} vs. {random:.6f}).")

        if fusion >= random:
            lines.append(
                "- Fusion not outperforming Random is not an experimental failure. "
                "It suggests that, in the current relatively regular dataset, random "
                "10% sampling may already cover a broad state distribution, while "
                "frozen ResNet18 features and KMeans clusters may not fully encode "
                "the robot manipulation state relevant to action prediction."
            )

    return "\n".join(lines)


def compare_methods(results: pd.DataFrame, left: str, right: str) -> str | None:
    """当两个方法都存在时，比较二者 test MSE。"""
    by_method = results.set_index("method")
    if not {left, right}.issubset(by_method.index):
        return None
    left_mse = float(by_method.loc[left, "test_mse"])
    right_mse = float(by_method.loc[right, "test_mse"])
    verdict = "is better than" if left_mse < right_mse else "is not better than"
    return f"- `{left}` {verdict} `{right}` ({left_mse:.6f} vs. {right_mse:.6f})."


def build_visual_cluster_observation(results: pd.DataFrame) -> str:
    """根据当前 results.csv 生成 baseline_v2 观察结论。"""
    ten_percent = results[results["sample_ratio"].astype(float) < 1.0].copy()
    if ten_percent.empty:
        raise ValueError("No 10% methods were found in results.csv.")

    best_10 = ten_percent.loc[ten_percent["test_mse"].idxmin()]
    lines = [
        f"- Among 10% methods, the lowest test MSE is `{best_10['method']}` "
        f"({float(best_10['test_mse']):.6f})."
    ]

    for other in ("random", "action_change", "fusion"):
        comparison = compare_methods(results, "visual_cluster", other)
        if comparison is not None:
            lines.append(comparison)

    by_method = results.set_index("method")
    if {"visual_cluster", "random"}.issubset(by_method.index):
        visual = float(by_method.loc["visual_cluster", "test_mse"])
        random = float(by_method.loc["random", "test_mse"])
        if visual < random:
            lines.append(
                "- Visual-Cluster Only is better than Random 10%, suggesting that "
                "state coverage from ResNet18 feature clustering can improve coreset quality."
            )
        else:
            lines.append(
                "- Visual-Cluster Only does not outperform Random 10%, suggesting that "
                "random 10% sampling may already cover the regular state distribution well."
            )

    if {"fusion", "visual_cluster"}.issubset(by_method.index):
        fusion = float(by_method.loc["fusion", "test_mse"])
        visual = float(by_method.loc["visual_cluster", "test_mse"])
        if fusion > visual:
            lines.append(
                "- Fusion does not exceed Visual-Cluster. In this task, action-change "
                "scores may overemphasize abrupt action frames and introduce distribution "
                "bias; preserving visual state coverage alone appears more stable."
            )
        else:
            lines.append(
                "- Fusion exceeds Visual-Cluster, indicating that adding action surprise "
                "to visual coverage provides extra value in the current setting."
            )

    if "full" in by_method.index:
        full_mse = float(by_method.loc["full", "test_mse"])
        ten_percent_best_mse = float(best_10["test_mse"])
        verdict = "is better than" if full_mse < ten_percent_best_mse else "is not better than"
        lines.append(
            f"- Full Data 100% {verdict} all 10% methods "
            f"({full_mse:.6f} vs. best 10% {ten_percent_best_mse:.6f})."
        )

    return "\n".join(lines)


def build_method_explanations(results: pd.DataFrame) -> str:
    """解释 results.csv 中当前存在的所有方法。"""
    descriptions = {
        "random": "Random: random 10% baseline.",
        "action_change": "Action-Change: action surprise from adjacent action changes only.",
        "fusion": "Fusion: visual clustering coverage plus action surprise.",
        "visual_cluster": "Visual-Cluster: visual clustering coverage only, with random sampling inside each cluster.",
        "fusion_neighbor": (
            "Fusion-Neighbor: adds t-1 / t+1 temporal neighbors from the same episode "
            "around high action-surprise Fusion anchors."
        ),
        "full": "Full: 100% training set upper-bound reference.",
    }
    return "\n".join(
        f"- {descriptions[method]}"
        for method in results["method"].tolist()
        if method in descriptions
    )


def generate_v1_note(results: pd.DataFrame, note_path: Path) -> None:
    """生成 Baseline V1 实验说明文档。"""
    results_table = build_results_table(results)
    observation = build_observation(results)

    note = f"""# Baseline V1: Random / Action-Change / Fusion

## 1. Experiment Purpose

This version is the first complete closed-loop baseline experiment. It is archived as a comparison reference for later optimization experiments.

## 2. Dataset

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Total episodes: 50
- Total frames: 20000
- Image field: `observation.images.top`
- Action label: `action[:7]`

## 3. Feature Extractor

- Frozen ImageNet-pretrained ResNet18
- Feature dimension: 512
- Input image tensor shape: `3 x 480 x 640`
- Saved feature file: `outputs/features/features.npy`

## 4. Train/Test Split

- Train episodes: first 40 episodes
- Test episodes: last 10 episodes
- Train samples: 16000
- Test samples: 4000
- Test set is fixed and never used for sampling or scaler fitting.

## 5. Sampling Methods

- Random 10%: random baseline sampled from the training frames.
- Action-Change Coreset 10%: uses adjacent action difference within the same episode as action surprise.
- Fusion Coreset 10%: uses visual KMeans clustering for state coverage, then selects high action-surprise samples inside each cluster.

## 6. MLP Setting

- Model: `512 -> 256 -> 128 -> 7`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Epochs: {get_common_setting(results, "epochs")}
- Batch size: {get_common_setting(results, "batch_size")}
- Learning rate: {get_common_setting(results, "lr")}
- Seed: {get_common_setting(results, "seed")}

## 7. Main Results

{results_table}

## 8. Current Observation

{observation}

## 9. Files Archived

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
"""
    note_path.write_text(note, encoding="utf-8")
    print(f"Generated experiment note: {note_path}")


def generate_v2_note(results: pd.DataFrame, note_path: Path) -> None:
    """生成包含 Visual-Cluster 消融的 Baseline V2 实验说明文档。"""
    results_table = build_results_table(results)
    observation = build_visual_cluster_observation(results)
    methods = build_method_explanations(results)
    full_note = (
        "- Full Data 100% is included as an upper-bound reference."
        if "full" in set(results["method"])
        else "- Full Data 100% is not included in the current results.csv."
    )

    note = f"""# Baseline V2: Add Visual-Cluster Only Coreset

## 1. Experiment Purpose

This version extends `baseline_v1_random_action_fusion` with a Visual-Cluster Only ablation. The goal is to verify whether visual state coverage alone helps coreset selection.

## 2. Difference from Baseline V1

- Baseline V1 contains Random 10%, Action-Change 10%, and Fusion 10%.
- Baseline V2 adds Visual-Cluster Only 10%.
{full_note}

## 3. Dataset and Feature Setting

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Image field: `observation.images.top`
- Action label: `action[:7]`
- Feature extractor: frozen ImageNet-pretrained ResNet18
- Feature dimension: 512

## 4. Methods

{methods}

## 5. Main Results

{results_table}

## 6. Observation

{observation}

## 7. Archived Files

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `visual_cluster_sample_table.csv`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
"""
    note_path.write_text(note, encoding="utf-8")
    print(f"Generated experiment note: {note_path}")


def build_fusion_neighbor_observation(results: pd.DataFrame) -> str:
    """根据当前 results.csv 生成 Baseline V3 观察结论。"""
    if "method" not in results.columns or "test_mse" not in results.columns:
        raise ValueError("results.csv must contain method and test_mse columns.")

    best_all = results.loc[results["test_mse"].idxmin()]
    ten_percent = results[results["sample_ratio"].astype(float) < 1.0].copy()
    if ten_percent.empty:
        raise ValueError("No 10% methods were found in results.csv.")
    best_10 = ten_percent.loc[ten_percent["test_mse"].idxmin()]

    lines = [
        f"- Lowest test MSE among all methods: `{best_all['method']}` "
        f"({float(best_all['test_mse']):.6f}).",
        f"- Lowest test MSE among 10% methods: `{best_10['method']}` "
        f"({float(best_10['test_mse']):.6f}).",
    ]

    for other in ("random", "fusion", "visual_cluster"):
        comparison = compare_methods(results, "fusion_neighbor", other)
        if comparison is not None:
            lines.append(comparison)

    visual_comparison = compare_methods(results, "visual_cluster", "random")
    if visual_comparison is not None:
        lines.append(visual_comparison)

    by_method = results.set_index("method")
    if str(best_10["method"]) == "visual_cluster":
        lines.append(
            "- Visual-Cluster Only is the best 10% method, suggesting that visual "
            "state coverage is more important than pure action-change emphasis in "
            "the current task."
        )

    if {"fusion_neighbor", "fusion"}.issubset(by_method.index):
        neighbor = float(by_method.loc["fusion_neighbor", "test_mse"])
        fusion = float(by_method.loc["fusion", "test_mse"])
        if neighbor >= fusion:
            lines.append(
                "- Fusion + Temporal Neighbor does not improve Fusion. Under a fixed "
                "10% budget, temporal expansion preserves local context but consumes "
                "samples that could otherwise cover more anchors or visual states."
            )
        else:
            lines.append(
                "- Fusion + Temporal Neighbor improves Fusion, indicating that local "
                "temporal context around action-surprise anchors is useful here."
            )

    if {"fusion_neighbor", "visual_cluster"}.issubset(by_method.index):
        neighbor = float(by_method.loc["fusion_neighbor", "test_mse"])
        visual = float(by_method.loc["visual_cluster", "test_mse"])
        if neighbor >= visual:
            lines.append(
                "- Fusion + Temporal Neighbor does not improve Visual-Cluster. This "
                "suggests that, with single-frame ResNet18 features, preserving visual "
                "state coverage is more effective than expanding neighborhoods around "
                "action-change frames."
            )
        else:
            lines.append(
                "- Fusion + Temporal Neighbor improves Visual-Cluster, suggesting that "
                "local temporal context adds useful information beyond state coverage."
            )

    if "full" in by_method.index:
        full_mse = float(by_method.loc["full", "test_mse"])
        best_10_mse = float(best_10["test_mse"])
        verdict = "is better than" if full_mse < best_10_mse else "is not better than"
        lines.append(
            f"- Full Data 100% {verdict} all 10% methods "
            f"({full_mse:.6f} vs. best 10% {best_10_mse:.6f})."
        )

    return "\n".join(lines)


def generate_v3_note(results: pd.DataFrame, note_path: Path) -> None:
    """生成包含 Fusion + Temporal Neighbor 的 Baseline V3 实验说明文档。"""
    results_table = build_results_table(results)
    observation = build_fusion_neighbor_observation(results)
    methods = build_method_explanations(results)
    full_note = (
        "- Full Data 100% is included as an upper-bound reference."
        if "full" in set(results["method"])
        else "- Full Data 100% is not included in the current results.csv."
    )

    note = f"""# Baseline V3: Add Fusion + Temporal Neighbor Coreset

## 1. Experiment Purpose

This version extends baseline_v2 with Fusion + Temporal Neighbor Coreset. The goal is to test whether local temporal context around high action-surprise key frames helps coreset selection.

## 2. Difference from Previous Versions

- Baseline V1 contains Random 10%, Action-Change 10%, and Fusion 10%.
- Baseline V2 adds Visual-Cluster Only 10%.
- Baseline V3 adds Fusion + Temporal Neighbor 10%.
{full_note}

## 3. Dataset and Feature Setting

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Image field: `observation.images.top`
- Action label: `action[:7]`
- Feature extractor: frozen ImageNet-pretrained ResNet18
- Feature dimension: 512
- Train episodes: first 40 episodes
- Test episodes: last 10 episodes
- Train samples: 16000
- Test samples: 4000

## 4. Methods

{methods}

## 5. Main Results

{results_table}

## 6. Observation

{observation}

## 7. Archived Files

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `fusion_neighbor_sample_table.csv`
- `fusion_neighbor_selection_info.json`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
"""
    note_path.write_text(note, encoding="utf-8")
    print(f"Generated experiment note: {note_path}")


def generate_experiment_note(results_csv: Path, note_path: Path, experiment_name: str) -> None:
    """根据归档后的 results.csv 生成 experiment_note.md。"""
    if not results_csv.exists():
        raise FileNotFoundError(f"Cannot generate experiment note; missing: {results_csv}")

    results = pd.read_csv(results_csv)
    if experiment_name == FUSION_NEIGHBOR_EXPERIMENT_NAME:
        generate_v3_note(results, note_path)
    elif experiment_name == VISUAL_CLUSTER_EXPERIMENT_NAME:
        generate_v2_note(results, note_path)
    else:
        generate_v1_note(results, note_path)


def build_manifest(archive_dir: Path) -> list[dict[str, Any]]:
    """递归扫描归档目录，生成文件清单。

    manifest 记录相对路径、文件大小和修改时间，便于以后确认归档内容是否完整。
    """
    records: list[dict[str, Any]] = []
    for path in sorted(archive_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        records.append(
            {
                "relative_path": path.relative_to(archive_dir).as_posix(),
                "file_size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return records


def write_manifest(archive_dir: Path, manifest_path: Path) -> None:
    """为归档目录写出 file_manifest.json。"""
    records = build_manifest(archive_dir)
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated file manifest: {manifest_path}")


def prepare_archive_dir(archive_dir: Path, overwrite: bool) -> None:
    """创建归档目录，并遵守默认不覆盖规则。

    已归档的 baseline 是实验基准，不应被误覆盖；只有显式传入 --overwrite 时才删除重建。
    """
    if archive_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Archive directory already exists: {archive_dir}\n"
                "Use --overwrite to delete the old archive and recreate it."
            )
        shutil.rmtree(archive_dir)
        print(f"Removed existing archive directory: {archive_dir}")
    ensure_dir(archive_dir)


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    archive_dir = project_root / "experiments" / args.experiment_name

    prepare_archive_dir(archive_dir, args.overwrite)

    copy_directory(project_root / "outputs" / "results", archive_dir / "results")
    copy_directory(project_root / "outputs" / "figures", archive_dir / "figures")
    copy_directory(project_root / "outputs" / "checkpoints", archive_dir / "checkpoints")
    copy_directory(project_root / "report_assets", archive_dir / "report_assets")
    copy_file(project_root / "README.md", archive_dir / "README_snapshot.md")

    results_csv = archive_dir / "results" / "results.csv"
    note_path = archive_dir / "experiment_note.md"
    generate_experiment_note(results_csv, note_path, args.experiment_name)

    manifest_path = archive_dir / "file_manifest.json"
    write_manifest(archive_dir, manifest_path)

    print("\nArchive finished.")
    print(f"Archive directory: {archive_dir}")


if __name__ == "__main__":
    main()
