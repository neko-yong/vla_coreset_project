"""Stage 4: train a unified MLP and evaluate fixed test-set MSE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    ensure_dir,
    get_project_root,
    get_train_test_masks,
    load_feature_arrays,
    save_json,
    set_seed,
)


METHOD_TO_INDEX_FILE = {
    "random": "selected_indices_random.npy",
    "action_change": "selected_indices_action_change.npy",
    "visual_cluster": "selected_indices_visual_cluster.npy",
    "fusion": "selected_indices_fusion.npy",
    "fusion_neighbor": "selected_indices_fusion_neighbor.npy",
}


class MLPRegressor(nn.Module):
    """Shared MLP architecture for every sampling method."""

    def __init__(self, input_dim: int = 512, output_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_bool(value: str | bool) -> bool:
    """Parse a CLI boolean value."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    """Parse Stage 4 training arguments."""
    parser = argparse.ArgumentParser(description="Train MLP on selected VLA features.")
    parser.add_argument("--feature_dir", default="outputs/features")
    parser.add_argument("--result_dir", default="outputs/results")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints")
    parser.add_argument(
        "--method",
        default="random",
        choices=("random", "action_change", "visual_cluster", "fusion", "fusion_neighbor", "full"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--standardize_features", type=parse_bool, default=True)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved
    return resolved


def choose_device(device_arg: str) -> torch.device:
    """Choose the requested torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' was requested, but CUDA is not available.")
    return torch.device(device_arg)


def get_method_train_indices(
    method: str,
    train_indices: np.ndarray,
    train_mask: np.ndarray,
    result_dir: Path,
) -> np.ndarray:
    """Return training indices for one method and verify they are train-only."""
    if method == "full":
        return train_indices.astype(np.int64)

    index_path = result_dir / METHOD_TO_INDEX_FILE[method]
    if not index_path.exists():
        raise FileNotFoundError(
            f"Selection file not found for method '{method}': {index_path}. "
            "Run Stage 3 selection first."
        )

    selected_indices = np.load(index_path).astype(np.int64)
    if selected_indices.ndim != 1:
        raise ValueError(f"Selected indices must be 1-D, got shape {selected_indices.shape}.")
    if len(selected_indices) == 0:
        raise ValueError(f"No selected samples were found in {index_path}.")
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(train_mask)):
        raise ValueError(f"Selected indices in {index_path} are out of range.")
    if not np.all(train_mask[selected_indices]):
        bad_count = int((~train_mask[selected_indices]).sum())
        raise RuntimeError(
            f"Selected indices for method '{method}' contain {bad_count} test samples. "
            "Test samples must never participate in training."
        )
    return selected_indices


def sample_ratio_for_method(method: str, num_train_samples: int, full_train_count: int) -> float:
    """Compute the selected fraction relative to the full training split."""
    if method == "full":
        return 1.0
    return float(num_train_samples / full_train_count)


def standardize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit StandardScaler on train features only, then transform train/test."""
    if not enabled:
        return train_features.astype(np.float32), test_features.astype(np.float32), {
            "standardize_features": False,
            "scaler_mean": None,
            "scaler_scale": None,
        }

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    return train_scaled, test_scaled, {
        "standardize_features": True,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }


def apply_checkpoint_scaler(features: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    """Transform features using scaler parameters saved in a checkpoint."""
    if not checkpoint.get("standardize_features", True):
        return features.astype(np.float32)

    mean = checkpoint.get("scaler_mean")
    scale = checkpoint.get("scaler_scale")
    if mean is None or scale is None:
        raise RuntimeError("Checkpoint is missing scaler_mean/scaler_scale.")
    mean_array = np.asarray(mean, dtype=np.float32)
    scale_array = np.asarray(scale, dtype=np.float32)
    return ((features.astype(np.float32) - mean_array) / scale_array).astype(np.float32)


def make_loader(
    features: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a tensor DataLoader for MLP regression."""
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(actions.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    """Train the MLP with MSE loss and Adam."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logs: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for features, actions in train_loader:
            features = features.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        logs.append({"epoch": epoch, "train_loss": train_loss})
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.8f}")

    return logs


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate overall and per-joint MSE on the fixed test set."""
    model.eval()
    squared_error_sum = torch.zeros(7, dtype=torch.float64)
    total_samples = 0

    with torch.no_grad():
        for features, actions in test_loader:
            features = features.to(device)
            actions = actions.to(device)
            predictions = model(features)
            squared_error = (predictions - actions) ** 2
            squared_error_sum += squared_error.sum(dim=0).cpu().double()
            total_samples += features.shape[0]

    if total_samples == 0:
        raise RuntimeError("Test loader has no samples.")

    joint_mse = (squared_error_sum / total_samples).numpy()
    metrics = {"test_mse": float(joint_mse.mean())}
    for joint_idx, value in enumerate(joint_mse, start=1):
        metrics[f"joint_{joint_idx}_mse"] = float(value)
    return metrics


def update_results_csv(result_dir: Path, eval_info: dict[str, Any]) -> None:
    """Insert or replace one method row in outputs/results/results.csv."""
    results_path = result_dir / "results.csv"
    row_fields = [
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
        "epochs",
        "batch_size",
        "lr",
        "seed",
    ]
    new_row = {field: eval_info[field] for field in row_fields}

    if results_path.exists():
        df = pd.read_csv(results_path)
        df = df[df["method"] != eval_info["method"]]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    order = {
        "random": 0,
        "action_change": 1,
        "visual_cluster": 2,
        "fusion": 3,
        "fusion_neighbor": 4,
        "full": 5,
    }
    df["_order"] = df["method"].map(order).fillna(99)
    df = df.sort_values(["_order", "method"]).drop(columns=["_order"])
    df.to_csv(results_path, index=False)
    print(f"Updated results summary: {results_path}")


def load_arrays_and_split(feature_dir: Path) -> dict[str, Any]:
    """Load features/actions and rebuild the fixed train/test split."""
    arrays = load_feature_arrays(feature_dir)
    features = arrays["features"].astype(np.float32)
    actions = arrays["actions"].astype(np.float32)
    episode_ids = arrays["episode_ids"]

    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f"Expected features shape [N, 512], got {features.shape}.")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape [N, 7], got {actions.shape}.")
    if len(features) != len(actions) or len(features) != len(episode_ids):
        raise ValueError("features, actions, and episode_ids must have the same length.")

    train_mask, test_mask, train_episodes, test_episodes = get_train_test_masks(episode_ids)
    return {
        "features": features,
        "actions": actions,
        "episode_ids": episode_ids,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "train_indices": np.flatnonzero(train_mask),
        "test_indices": np.flatnonzero(test_mask),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_dir = resolve_project_path(args.feature_dir)
    result_dir = ensure_dir(resolve_project_path(args.result_dir))
    checkpoint_dir = ensure_dir(resolve_project_path(args.checkpoint_dir))
    device = choose_device(args.device)

    data = load_arrays_and_split(feature_dir)
    selected_train_indices = get_method_train_indices(
        method=args.method,
        train_indices=data["train_indices"],
        train_mask=data["train_mask"],
        result_dir=result_dir,
    )
    test_indices = data["test_indices"]

    x_train_raw = data["features"][selected_train_indices]
    y_train = data["actions"][selected_train_indices]
    x_test_raw = data["features"][test_indices]
    y_test = data["actions"][test_indices]

    x_train, x_test, scaler_state = standardize_features(
        x_train_raw,
        x_test_raw,
        enabled=args.standardize_features,
    )

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    print(f"Method: {args.method}")
    print(f"Device: {device}")
    print(f"Train samples: {len(selected_train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Standardize features: {args.standardize_features}")

    model = MLPRegressor().to(device)
    logs = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    metrics = evaluate_model(model, test_loader, device)

    checkpoint_path = checkpoint_dir / f"mlp_{args.method}.pt"
    checkpoint = {
        "method": args.method,
        "model_state_dict": model.state_dict(),
        "input_dim": 512,
        "output_dim": 7,
        "standardize_features": bool(args.standardize_features),
        "scaler_mean": scaler_state["scaler_mean"],
        "scaler_scale": scaler_state["scaler_scale"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "sample_ratio": sample_ratio_for_method(
            args.method,
            num_train_samples=len(selected_train_indices),
            full_train_count=len(data["train_indices"]),
        ),
        "num_train_samples": int(len(selected_train_indices)),
        "num_test_samples": int(len(test_indices)),
    }
    torch.save(checkpoint, checkpoint_path)

    log_path = result_dir / f"train_log_{args.method}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)

    eval_info: dict[str, Any] = {
        "method": args.method,
        "sample_ratio": checkpoint["sample_ratio"],
        "num_train_samples": int(len(selected_train_indices)),
        "num_test_samples": int(len(test_indices)),
        **metrics,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "standardize_features": bool(args.standardize_features),
        "checkpoint_path": str(checkpoint_path),
    }
    eval_path = result_dir / f"eval_{args.method}.json"
    save_json(eval_info, eval_path)
    update_results_csv(result_dir, eval_info)

    print("\nTraining and evaluation finished.")
    print(f"Test MSE: {metrics['test_mse']:.8f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved train log: {log_path}")
    print(f"Saved eval json: {eval_path}")


if __name__ == "__main__":
    main()
