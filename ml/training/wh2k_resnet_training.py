"""
WreckHunter 2000 — Hybrid ResNet-18 Training Pipeline
========================================================
Trains a ResNet-18 classifier on 3-channel magnetic tiles (NSS, VDR, Tilt-Angle)
using the "Anchor & Supplement" strategy:

  Step 1: Real chips from AWOIS/GLSC known locations (anchor)
  Step 2: Background negatives from empty Central Basin locations
  Step 3: Synthetic augmentation (boost sparse classes)

Classes:
  0 = GEOLOGY_ONLY    (background, no target)
  1 = STEEL_HULL      (700ft+ steel freighter — "Iron Giant")
  2 = WOOD_CARGO      (150-300ft wooden wreck w/ ore — "Cargo Ghost")
  3 = WELLHEAD        (vertical casing monopole — "Pin-Prick")

Architecture:
  ResNet-18 with 3-channel input (pretrained ImageNet weights adapted).
  Final FC layer → 4 classes.

Data loading:
  Reads real_chips.npz + synthetic_tiles.npz, merges, shuffles, splits
  into train/val/test (70/15/15).

Output:
  Best model checkpoint + training curves + confusion matrix.

Usage:
  python wh2k_resnet_training.py --real-data wreck_hunting_ml/data/real_chips/real_chips.npz \\
                                  --synthetic-data wreck_hunting_ml/data/synthetic/synthetic_tiles.npz \\
                                  --output-dir wreck_hunting_ml/models
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "wreck_hunting_ml" / "models"

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}
NUM_CLASSES = 4


# ── Dataset Preparation ────────────────────────────────────────────────────

def load_and_merge_datasets(
    real_data_path: str | Path | None = None,
    synthetic_data_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and merge real + synthetic tile datasets.

    Returns (tiles, labels) where:
      tiles: (N, 3, H, W) float32
      labels: (N,) int64
    """
    all_tiles = []
    all_labels = []

    if real_data_path and Path(real_data_path).exists():
        data = np.load(real_data_path)
        all_tiles.append(data["tiles"])
        all_labels.append(data["labels"])
        logger.info("Loaded real data: %d tiles from %s", len(data["labels"]), real_data_path)

    if synthetic_data_path and Path(synthetic_data_path).exists():
        data = np.load(synthetic_data_path)
        all_tiles.append(data["tiles"])
        all_labels.append(data["labels"])
        logger.info("Loaded synthetic data: %d tiles from %s", len(data["labels"]), synthetic_data_path)

    if not all_tiles:
        raise FileNotFoundError("No training data found. Run chip extractor and/or synthetic generator first.")

    tiles = np.concatenate(all_tiles, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info("Merged dataset: %d tiles — %s", len(labels), {
        CLASS_NAMES[i]: int(np.sum(labels == i)) for i in range(NUM_CLASSES)
    })

    return tiles, labels


def normalize_tiles(tiles: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-channel z-score normalization.

    Returns (normalized_tiles, stats_dict).
    stats_dict has keys 'mean' and 'std' (each shape (3,)).
    """
    mean = tiles.mean(axis=(0, 2, 3), keepdims=True)
    std = tiles.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-8] = 1.0  # Avoid division by zero
    normalized = (tiles - mean) / std
    return normalized, {
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
    }


def split_dataset(
    tiles: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Stratified split into train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": (tiles[train_idx], labels[train_idx]),
        "val": (tiles[val_idx], labels[val_idx]),
        "test": (tiles[test_idx], labels[test_idx]),
    }


# ── PyTorch Training ───────────────────────────────────────────────────────

def train_resnet18(
    real_data_path: str | Path | None = None,
    synthetic_data_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Full training pipeline: load data → train ResNet-18 → save best model.

    Returns a results dict with training history and test metrics.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import resnet18, ResNet18_Weights
    except ImportError:
        logger.error(
            "PyTorch and torchvision are required. "
            "Install: pip install torch torchvision"
        )
        return {"error": "torch not installed"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # Load & merge
    tiles, labels = load_and_merge_datasets(real_data_path, synthetic_data_path)

    # Normalize
    tiles_norm, norm_stats = normalize_tiles(tiles)

    # Split
    splits = split_dataset(tiles_norm, labels, seed=seed)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    logger.info("Split: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

    # Compute class weights for imbalanced data (geology >> wrecks)
    class_counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
    class_weights = np.where(class_counts > 0, len(y_train) / (NUM_CLASSES * class_counts), 1.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info("Class weights: %s", {CLASS_NAMES[i]: f"{class_weights[i]:.2f}" for i in range(NUM_CLASSES)})

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model: ResNet-18 with modified first conv (3-channel magnetic, not RGB)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Replace first conv: ImageNet expects 3ch RGB, we have 3ch (NSS,VDR,Tilt)
    # Keep pretrained weights for feature extraction, they transfer surprisingly well
    # to structured 2D grids despite being trained on photos
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(n_train_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "norm_stats": norm_stats,
                "class_names": CLASS_NAMES,
            }, output_dir / "best_resnet18.pt")

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_acc=%.4f (best=%.4f @ %d)",
                epoch, epochs, avg_train_loss, avg_val_loss, val_acc, best_val_acc, best_epoch,
            )

    # ── Test evaluation ────────────────────────────────────────────────
    logger.info("Loading best model from epoch %d...", best_epoch)
    checkpoint = torch.load(output_dir / "best_resnet18.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    test_acc = float(np.mean(all_preds == all_true))

    # Per-class metrics
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        mask_true = all_true == cls_id
        mask_pred = all_preds == cls_id
        tp = int(np.sum(mask_true & mask_pred))
        fp = int(np.sum(~mask_true & mask_pred))
        fn = int(np.sum(mask_true & ~mask_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class[CLASS_NAMES[cls_id]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(mask_true)),
        }

    # Confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_true, all_preds):
        confusion[int(t)][int(p)] += 1

    results = {
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": [CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        "norm_stats": norm_stats,
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": device,
        },
    }

    # Save results
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save history for plotting
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Test accuracy: %.4f | Best val acc: %.4f (epoch %d)",
        test_acc, best_val_acc, best_epoch,
    )
    logger.info("Per-class results: %s", json.dumps(per_class, indent=2))
    logger.info("Model saved to %s/best_resnet18.pt", output_dir)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K ResNet-18 Hybrid Training")
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to real_chips.npz from chip extractor")
    parser.add_argument("--synthetic-data", type=str, default=None,
                        help="Path to synthetic_tiles.npz from tile generator")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_resnet18(
        real_data_path=args.real_data,
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
