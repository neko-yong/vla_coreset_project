"""Stage 5: generate report-ready figures and result tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import ensure_dir, get_project_root, get_train_test_masks, set_seed


REQUIRED_METHODS = ("random", "action_change", "fusion")
METHOD_ORDER = ("random", "action_change", "visual_cluster", "fusion", "full")
METHOD_LABELS = {
    "random": "Random",
    "action_change": "Action-Change",
    "visual_cluster": "Visual-Cluster",
    "fusion": "Fusion",
    "full": "Full",
}
METHOD_COLORS = {
    "random": "#9D755D",
    "action_change": "#F58518",
    "visual_cluster": "#B279A2",
    "fusion": "#54A24B",
    "full": "#4C78A8",
}
METHOD_MARKERS = {
    "random": "o",
    "action_change": "^",
    "visual_cluster": "D",
    "fusion": "s",
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage 5 report figures.")
    parser.add_argument("--feature_dir", default="outputs/features")
    parser.add_argument("--result_dir", default="outputs/results")
    parser.add_argument("--figure_dir", default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_to_plot", type=int, default=0)
    parser.add_argument("--max_pca_samples", type=int, default=5000)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def require_file(path: Path) -> Path:
    """Raise a clear error when an expected input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    return path


def load_stage_data(feature_dir: Path, result_dir: Path) -> dict[str, Any]:
    """Load feature arrays, result CSV, scores, and selected index files."""
    data = {
        "features": np.load(require_file(feature_dir / "features.npy")),
        "actions": np.load(require_file(feature_dir / "actions.npy")),
        "episode_ids": np.load(require_file(feature_dir / "episode_ids.npy")),
        "frame_ids": np.load(require_file(feature_dir / "frame_ids.npy")),
        "results": pd.read_csv(require_file(result_dir / "results.csv")),
        "action_scores": np.load(require_file(result_dir / "action_change_scores.npy")),
        "selected_random": np.load(require_file(result_dir / "selected_indices_random.npy")),
        "selected_action": np.load(require_file(result_dir / "selected_indices_action_change.npy")),
        "selected_fusion": np.load(require_file(result_dir / "selected_indices_fusion.npy")),
    }
    visual_path = result_dir / "selected_indices_visual_cluster.npy"
    data["selected_visual_cluster"] = np.load(visual_path) if visual_path.exists() else None

    missing_methods = [m for m in REQUIRED_METHODS if m not in set(data["results"]["method"])]
    if missing_methods:
        raise ValueError(
            f"results.csv must contain methods {REQUIRED_METHODS}. Missing: {missing_methods}"
        )
    return data


def available_result_methods(results: pd.DataFrame) -> list[str]:
    """Return known result methods in a stable visual order."""
    result_methods = set(results["method"])
    return [method for method in METHOD_ORDER if method in result_methods]


def save_figure(path: Path) -> None:
    """Save the active matplotlib figure at report quality."""
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def plot_mse_comparison(results: pd.DataFrame, figure_dir: Path) -> None:
    """Generate the test MSE bar chart."""
    methods = available_result_methods(results)
    plot_df = results[results["method"].isin(methods)].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=methods, ordered=True)
    plot_df = plot_df.sort_values("method")
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS)

    plt.figure(figsize=(max(7.5, 1.7 * len(plot_df)), 5.0))
    bars = plt.bar(
        plot_df["method_label"],
        plot_df["test_mse"],
        color=[METHOD_COLORS.get(str(method), "#4C78A8") for method in plot_df["method"]],
    )
    plt.ylabel("Test MSE")
    plt.xlabel("Sampling Method")
    plt.title("Test MSE Comparison of 10% Sampling Methods")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    ymax = float(plot_df["test_mse"].max())
    for bar, value in zip(bars, plot_df["test_mse"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0, ymax * 1.18)
    save_figure(figure_dir / "mse_comparison.png")


def choose_episode_for_action_plot(
    requested_episode: int,
    episode_ids: np.ndarray,
    train_episodes: list[int],
    selected_sets: list[set[int]],
) -> int:
    """Choose a training episode that has selected points for the action plot."""
    def has_selected_points(episode_id: int) -> bool:
        episode_indices = set(np.flatnonzero(episode_ids == episode_id).astype(int))
        return any(bool(episode_indices & selected_set) for selected_set in selected_sets)

    if requested_episode in train_episodes and has_selected_points(requested_episode):
        return requested_episode

    print(
        f"Episode {requested_episode} is not a training episode or has no selected points. "
        "Automatically choosing a training episode for plotting."
    )
    for episode_id in train_episodes:
        if has_selected_points(episode_id):
            print(f"Using episode {episode_id} for action-change visualization.")
            return episode_id

    fallback = train_episodes[0]
    print(f"No training episode has selected points; using episode {fallback}.")
    return fallback


def plot_action_change_selected(
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    action_scores: np.ndarray,
    selected_random: np.ndarray,
    selected_action: np.ndarray,
    selected_fusion: np.ndarray,
    selected_visual_cluster: np.ndarray | None,
    train_episodes: list[int],
    requested_episode: int,
    figure_dir: Path,
) -> None:
    """Plot action-change scores and selected frames for one training episode."""
    selected_sets = [
        set(selected_random.astype(int)),
        set(selected_action.astype(int)),
        set(selected_fusion.astype(int)),
    ]
    if selected_visual_cluster is not None:
        selected_sets.append(set(selected_visual_cluster.astype(int)))
    episode_id = choose_episode_for_action_plot(
        requested_episode,
        episode_ids,
        train_episodes,
        selected_sets,
    )
    episode_indices = np.flatnonzero(episode_ids == episode_id)

    plt.figure(figsize=(10.0, 5.0))
    plt.plot(
        frame_ids[episode_indices],
        action_scores[episode_indices],
        color="#4C78A8",
        linewidth=1.8,
        label="Action-change score",
    )

    marker_specs = [
        ("Random selected", selected_random, METHOD_MARKERS["random"], METHOD_COLORS["random"]),
        (
            "Action-Change selected",
            selected_action,
            METHOD_MARKERS["action_change"],
            METHOD_COLORS["action_change"],
        ),
        (
            "Visual-Cluster selected",
            selected_visual_cluster,
            METHOD_MARKERS["visual_cluster"],
            METHOD_COLORS["visual_cluster"],
        ),
        ("Fusion selected", selected_fusion, METHOD_MARKERS["fusion"], METHOD_COLORS["fusion"]),
    ]
    episode_set = set(episode_indices.astype(int))
    for label, selected, marker, color in marker_specs:
        if selected is None:
            continue
        selected_in_episode = np.array(
            [idx for idx in selected.astype(int) if idx in episode_set],
            dtype=int,
        )
        if len(selected_in_episode) == 0:
            continue
        plt.scatter(
            frame_ids[selected_in_episode],
            action_scores[selected_in_episode],
            s=42,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            label=label,
            zorder=3,
        )

    plt.xlabel("Frame ID")
    plt.ylabel("Action Change Score")
    plt.title("Action Change Scores and Selected Coreset Frames")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)
    save_figure(figure_dir / "action_change_selected.png")


def plot_pca_feature_distribution(
    features: np.ndarray,
    episode_ids: np.ndarray,
    selected_random: np.ndarray,
    selected_fusion: np.ndarray,
    selected_visual_cluster: np.ndarray | None,
    train_mask: np.ndarray,
    max_pca_samples: int,
    seed: int,
    figure_dir: Path,
) -> None:
    """Plot PCA of training ResNet18 features with selected samples highlighted."""
    if max_pca_samples <= 0:
        raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}.")

    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) > max_pca_samples:
        background_indices = np.sort(
            rng.choice(train_indices, size=max_pca_samples, replace=False)
        )
    else:
        background_indices = train_indices

    pca = PCA(n_components=2, random_state=seed)
    background_xy = pca.fit_transform(features[background_indices])

    random_train = selected_random[train_mask[selected_random]]
    fusion_train = selected_fusion[train_mask[selected_fusion]]
    visual_train = (
        selected_visual_cluster[train_mask[selected_visual_cluster]]
        if selected_visual_cluster is not None
        else None
    )
    random_xy = pca.transform(features[random_train])
    fusion_xy = pca.transform(features[fusion_train])
    visual_xy = pca.transform(features[visual_train]) if visual_train is not None else None

    plt.figure(figsize=(8.2, 6.2))
    plt.scatter(
        background_xy[:, 0],
        background_xy[:, 1],
        s=8,
        color="#B8B8B8",
        alpha=0.45,
        label="Training samples",
    )
    plt.scatter(
        random_xy[:, 0],
        random_xy[:, 1],
        s=16,
        color=METHOD_COLORS["random"],
        alpha=0.75,
        label="Random selected",
    )
    plt.scatter(
        fusion_xy[:, 0],
        fusion_xy[:, 1],
        s=18,
        color=METHOD_COLORS["fusion"],
        alpha=0.85,
        marker="^",
        label="Fusion selected",
    )
    if visual_xy is not None:
        plt.scatter(
            visual_xy[:, 0],
            visual_xy[:, 1],
            s=18,
            color=METHOD_COLORS["visual_cluster"],
            alpha=0.70,
            marker=METHOD_MARKERS["visual_cluster"],
            label="Visual-Cluster selected",
        )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization of ResNet18 Features and Selected Samples")
    plt.legend(markerscale=1.5)
    plt.grid(linestyle="--", alpha=0.25)
    save_figure(figure_dir / "pca_feature_distribution.png")


def plot_selected_frame_distribution(
    episode_ids: np.ndarray,
    selected_random: np.ndarray,
    selected_action: np.ndarray,
    selected_fusion: np.ndarray,
    selected_visual_cluster: np.ndarray | None,
    train_episodes: list[int],
    figure_dir: Path,
) -> None:
    """Plot selected frame counts per training episode for each method."""
    counts: dict[str, list[int]] = {}
    method_arrays = {
        "random": selected_random,
        "action_change": selected_action,
        "visual_cluster": selected_visual_cluster,
        "fusion": selected_fusion,
    }
    method_arrays = {method: selected for method, selected in method_arrays.items() if selected is not None}
    for method, selected in method_arrays.items():
        selected_episode_ids = episode_ids[selected.astype(int)]
        counts[method] = [
            int(np.sum(selected_episode_ids == episode_id)) for episode_id in train_episodes
        ]

    x = np.arange(len(train_episodes))
    width = min(0.22, 0.8 / max(len(method_arrays), 1))
    plt.figure(figsize=(12.0, 5.2))
    offsets = (np.arange(len(method_arrays)) - (len(method_arrays) - 1) / 2) * width
    for offset, method in zip(offsets, method_arrays.keys()):
        plt.bar(
            x + offset,
            counts[method],
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    plt.xlabel("Episode Index")
    plt.ylabel("Selected Frame Count")
    plt.title("Selected Frame Distribution across Training Episodes")
    plt.xticks(x, train_episodes, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(figure_dir / "selected_frame_distribution.png")


def plot_joint_mse_comparison(results: pd.DataFrame, figure_dir: Path) -> None:
    """Generate per-joint MSE grouped bar chart."""
    methods = available_result_methods(results)
    plot_df = results[results["method"].isin(methods)].copy()
    joints = [f"joint_{idx}_mse" for idx in range(1, 8)]
    x = np.arange(len(joints))
    width = min(0.18, 0.8 / max(len(methods), 1))

    plt.figure(figsize=(max(10.0, 1.5 * len(methods) + 6), 5.4))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for offset, method in zip(offsets, methods):
        row = plot_df[plot_df["method"] == method].iloc[0]
        plt.bar(
            x + offset,
            [row[joint] for joint in joints],
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS.get(method, "#4C78A8"),
        )

    plt.xlabel("Action Dimension")
    plt.ylabel("MSE")
    plt.title("Per-Joint MSE Comparison")
    plt.xticks(x, [f"joint_{idx}" for idx in range(1, 8)])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(figure_dir / "joint_mse_comparison.png")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Create a dependency-free Markdown table."""
    headers = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows) + "\n"


def write_report_tables(results: pd.DataFrame, table_dir: Path) -> None:
    """Write CSV and Markdown summary tables for the report."""
    ensure_dir(table_dir)
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

    methods = available_result_methods(results)
    summary = results[results["method"].isin(methods)][columns].copy()
    csv_path = table_dir / "results_summary.csv"
    md_path = table_dir / "results_summary.md"
    summary.to_csv(csv_path, index=False)

    md_summary = summary.copy()
    mse_columns = ["test_mse"] + [f"joint_{idx}_mse" for idx in range(1, 8)]
    for column in mse_columns:
        md_summary[column] = md_summary[column].map(lambda value: f"{float(value):.6f}")
    md_path.write_text(dataframe_to_markdown(md_summary), encoding="utf-8")
    print(f"Saved table: {csv_path}")
    print(f"Saved table: {md_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_dir = resolve_project_path(args.feature_dir)
    result_dir = resolve_project_path(args.result_dir)
    figure_dir = ensure_dir(resolve_project_path(args.figure_dir))
    table_dir = ensure_dir(get_project_root() / "report_assets" / "result_tables")

    data = load_stage_data(feature_dir, result_dir)
    train_mask, _, train_episodes, _ = get_train_test_masks(data["episode_ids"])

    plot_mse_comparison(data["results"], figure_dir)
    plot_action_change_selected(
        episode_ids=data["episode_ids"],
        frame_ids=data["frame_ids"],
        action_scores=data["action_scores"],
        selected_random=data["selected_random"],
        selected_action=data["selected_action"],
        selected_fusion=data["selected_fusion"],
        selected_visual_cluster=data["selected_visual_cluster"],
        train_episodes=train_episodes,
        requested_episode=args.episode_to_plot,
        figure_dir=figure_dir,
    )
    plot_pca_feature_distribution(
        features=data["features"],
        episode_ids=data["episode_ids"],
        selected_random=data["selected_random"],
        selected_fusion=data["selected_fusion"],
        selected_visual_cluster=data["selected_visual_cluster"],
        train_mask=train_mask,
        max_pca_samples=args.max_pca_samples,
        seed=args.seed,
        figure_dir=figure_dir,
    )
    plot_selected_frame_distribution(
        episode_ids=data["episode_ids"],
        selected_random=data["selected_random"],
        selected_action=data["selected_action"],
        selected_fusion=data["selected_fusion"],
        selected_visual_cluster=data["selected_visual_cluster"],
        train_episodes=train_episodes,
        figure_dir=figure_dir,
    )
    plot_joint_mse_comparison(data["results"], figure_dir)
    write_report_tables(data["results"], table_dir)

    print("\nStage 5 visualization finished.")


if __name__ == "__main__":
    main()
