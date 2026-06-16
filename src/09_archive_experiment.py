"""Archive the current baseline experiment outputs for future comparison."""

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


def parse_args() -> argparse.Namespace:
    """Parse archive options."""
    parser = argparse.ArgumentParser(description="Archive current baseline experiment results.")
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing archive directory and recreate it.",
    )
    return parser.parse_args()


def copy_directory(src: Path, dst: Path) -> bool:
    """Copy a directory tree, warning and continuing if the source is missing."""
    if not src.exists():
        print(f"WARNING: source directory does not exist, skipped: {src}")
        return False
    shutil.copytree(src, dst)
    print(f"Copied directory: {src} -> {dst}")
    return True


def copy_file(src: Path, dst: Path) -> bool:
    """Copy one file, warning and continuing if the source is missing."""
    if not src.exists():
        print(f"WARNING: source file does not exist, skipped: {src}")
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    print(f"Copied file: {src} -> {dst}")
    return True


def format_float(value: Any) -> str:
    """Format numeric values for report tables."""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Create a simple Markdown table without optional tabulate dependency."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def build_results_table(results: pd.DataFrame) -> str:
    """Build a Markdown table from results.csv with MSE values rounded."""
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
    """Read a common training setting from results.csv when present."""
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
    """Generate observations directly from results.csv."""
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


def generate_experiment_note(results_csv: Path, note_path: Path) -> None:
    """Generate experiment_note.md from the archived results.csv."""
    if not results_csv.exists():
        raise FileNotFoundError(f"Cannot generate experiment note; missing: {results_csv}")

    results = pd.read_csv(results_csv)
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


def build_manifest(archive_dir: Path) -> list[dict[str, Any]]:
    """Recursively scan archive files and return manifest records."""
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
    """Write file_manifest.json for the archive directory."""
    records = build_manifest(archive_dir)
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated file manifest: {manifest_path}")


def prepare_archive_dir(archive_dir: Path, overwrite: bool) -> None:
    """Create the archive directory while respecting overwrite behavior."""
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
    generate_experiment_note(results_csv, note_path)

    manifest_path = archive_dir / "file_manifest.json"
    write_manifest(archive_dir, manifest_path)

    print("\nArchive finished.")
    print(f"Archive directory: {archive_dir}")


if __name__ == "__main__":
    main()
