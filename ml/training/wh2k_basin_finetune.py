"""
wh2k_basin_finetune.py
======================
Fine-tunes the epoch-48 ResNet-18 checkpoint into per-basin Lake Erie
expert models using *real* aeromagnetic tiles from wh2k_extract_real_tiles.py.

Inherits all Phase 2 machinery (FVD, off-axis penalty, ReduceLROnPlateau)
but uses a lower LR=1e-6 (fine-tuning regime) and mixes real + synthetic
tiles to prevent catastrophic forgetting.

Usage
-----
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_basin_finetune.py ^
      --basin west ^
      --checkpoint wreck_hunting_ml\\models\\best_resnet18.pt ^
      --real-tiles wreck_hunting_ml\\data\\real_tiles\\real_tiles_basin_west.npz ^
      --synthetic wreck_hunting_ml\\data\\synthetic\\synthetic_tiles.npz ^
      --output-dir wreck_hunting_ml\\models ^
      --epochs 25 --batch-size 32 --device cuda

  # Run all three basins back-to-back:
  For %B in (west central east) do (
      C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_basin_finetune.py ^
          --basin %B --device cuda
  )

Outputs
-------
  wreck_hunting_ml/models/basin_{name}/best_resnet18.pt
  wreck_hunting_ml/models/basin_{name}/training_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}
NUM_CLASSES = 4
TILE_DX_M   = 8.928571          # metres/pixel — matches synthetic tile generator
NE_SW_STRIKE_DEG = 45.0

BASINS = {
    "west":    {"lon_min": -83.50, "lat_min": 41.30, "lon_max": -82.00, "lat_max": 42.20},
    "central": {"lon_min": -82.00, "lat_min": 41.50, "lon_max": -80.30, "lat_max": 42.80},
    "east":    {"lon_min": -80.30, "lat_min": 42.00, "lon_max": -78.85, "lat_max": 42.95},
}

# Mix ratio: how many synthetic tiles to keep per real tile (to prevent forgetting)
SYNTH_PER_REAL = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("basin_finetune")


# ── GPU helpers (same as Phase 2) ─────────────────────────────────────────────

def get_gpu_temp() -> Optional[float]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def thermal_guard(threshold=80.0, sleep_sec=1.5, soft_threshold=78.0,
                  soft_sleep_sec=0.5, batch_num=0, soft_every_n=5):
    temp = get_gpu_temp()
    if temp is None:
        return
    if temp >= threshold:
        log.warning("GPU %.0f°C >= %.0f°C (HARD) — sleeping %.1fs", temp, threshold, sleep_sec)
        time.sleep(sleep_sec)
    elif temp >= soft_threshold and batch_num % soft_every_n == 0:
        time.sleep(soft_sleep_sec)


# ── FVD transform + off-axis weight (copy from wh2k_phase2_gpu_training) ──────

class FVDTransform:
    def __init__(self, dx_m=TILE_DX_M):
        self.dx_m = dx_m
        self._k_cache: dict = {}

    def _get_k(self, H, W, device):
        import torch
        key = (H, W)
        if key not in self._k_cache:
            kx = torch.fft.fftfreq(W, d=self.dx_m) * (2 * math.pi)
            ky = torch.fft.fftfreq(H, d=self.dx_m) * (2 * math.pi)
            KY, KX = torch.meshgrid(ky, kx, indexing="ij")
            self._k_cache[key] = torch.sqrt(KX ** 2 + KY ** 2)
        return self._k_cache[key].to(device)

    def __call__(self, x):
        import torch
        _, _, H, W = x.shape
        K = self._get_k(H, W, x.device)
        nss = x[:, 0]
        fvd = torch.real(torch.fft.ifft2(torch.fft.fft2(nss) * K))
        out = x.clone()
        out[:, 0] = fvd
        return out


def compute_off_axis_weights(tiles):
    import torch
    nss = tiles[:, 0]
    dy = nss[:, 1:, :] - nss[:, :-1, :]
    dx = nss[:, :, 1:] - nss[:, :, :-1]
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    mean_dy = dy.mean(dim=(1, 2))
    mean_dx = dx.mean(dim=(1, 2))
    angle = torch.atan2(mean_dx, mean_dy + 1e-10) * 180 / math.pi % 360
    offset = (angle - NE_SW_STRIKE_DEG).abs()
    offset = torch.where(offset > 180, 360 - offset, offset)
    offset = torch.where(offset > 90,  180 - offset, offset)
    return torch.where(offset >= 45, torch.full_like(offset, 3.0), torch.ones_like(offset))


# ── Dataset mixing ─────────────────────────────────────────────────────────────

def build_dataset(
    real_npz: Path,
    synth_npz: Optional[Path],
    synth_per_real: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tiles, labels) as float32 / int64 arrays.
    Mixes all real tiles with a proportional sample of synthetic tiles.
    """
    rng = np.random.default_rng(seed)

    real_data = np.load(str(real_npz))
    real_tiles  = real_data["tiles"].astype(np.float32)
    real_labels = real_data["labels"].astype(np.int64)
    log.info("Real tiles: %d  classes=%s", len(real_labels),
             {CLASS_NAMES[k]: int(np.sum(real_labels == k)) for k in range(NUM_CLASSES)})

    if synth_npz is None or not synth_npz.exists():
        log.warning("No synthetic data — training on real tiles only.")
        return real_tiles, real_labels

    n_synth_want = len(real_tiles) * synth_per_real
    synth_data   = np.load(str(synth_npz))
    n_synth_avail = len(synth_data["labels"])

    if n_synth_want >= n_synth_avail:
        synth_tile_all  = synth_data["tiles"].astype(np.float32)
        synth_label_all = synth_data["labels"].astype(np.int64)
    else:
        idx = rng.choice(n_synth_avail, n_synth_want, replace=False)
        synth_tile_all  = synth_data["tiles"][idx].astype(np.float32)
        synth_label_all = synth_data["labels"][idx].astype(np.int64)

    log.info("Synthetic tiles sampled: %d / %d", len(synth_label_all), n_synth_avail)

    tiles  = np.concatenate([real_tiles,  synth_tile_all],  axis=0)
    labels = np.concatenate([real_labels, synth_label_all], axis=0)

    perm = rng.permutation(len(tiles))
    log.info("Combined dataset: %d tiles  classes=%s", len(labels),
             {CLASS_NAMES[k]: int(np.sum(labels[perm] == k)) for k in range(NUM_CLASSES)})
    return tiles[perm], labels[perm]


# ── Main fine-tune function ───────────────────────────────────────────────────

def finetune_basin(
    basin_name: str,
    checkpoint_path: Path,
    real_npz: Path,
    synth_npz: Optional[Path],
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    weight_decay: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
    gpu_temp_threshold: float = 80.0,
    epoch_cooldown_sec: float = 2.0,
) -> dict:

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import resnet18
    except ImportError as exc:
        log.error("PyTorch required: %s", exc)
        return {"error": str(exc)}

    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA unavailable — using CPU")
        device = "cpu"

    if device == "cuda":
        log.info("GPU: %s | VRAM: %.1fGB | CUDA: %s",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9,
                 torch.version.cuda)

    out_basin_dir = output_dir / f"basin_{basin_name}"
    out_basin_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tiles ────────────────────────────────────────────────────────
    tiles_raw, labels_raw = build_dataset(real_npz, synth_npz, SYNTH_PER_REAL, seed)

    # ── Load checkpoint + norm_stats ─────────────────────────────────────
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        return {"error": f"checkpoint missing: {checkpoint_path}"}

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    start_epoch  = ckpt.get("epoch", 48)
    prev_val_acc = ckpt.get("val_acc", 0.0)
    log.info("Resuming checkpoint epoch=%d  val_acc=%.4f", start_epoch, prev_val_acc)

    # Recompute norm stats from the combined real+synth dataset
    log.info("Computing norm stats from combined dataset (%d tiles) ...", len(tiles_raw))
    sample_idx = np.random.default_rng(seed).choice(len(tiles_raw), min(2000, len(tiles_raw)),
                                                     replace=False)
    sample = tiles_raw[sample_idx]
    norm_mean = sample.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    norm_std  = sample.std( axis=(0, 2, 3), keepdims=True).astype(np.float32)
    norm_std[norm_std < 1e-8] = 1.0
    del sample

    tiles_raw = (tiles_raw - norm_mean) / norm_std

    # ── Train/val split ───────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n   = len(labels_raw)
    idx = rng.permutation(n)
    n_train = int(n * 0.80)   # Use 80/20 split (less val needed, more train)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    X_train = torch.from_numpy(tiles_raw[train_idx].copy())
    X_val   = torch.from_numpy(tiles_raw[val_idx  ].copy())
    y_train = torch.tensor(labels_raw[train_idx], dtype=torch.long)
    y_val   = torch.tensor(labels_raw[val_idx],   dtype=torch.long)
    del tiles_raw

    log.info("Split: train=%d  val=%d", len(y_train), len(y_val))

    counts = np.bincount(labels_raw[train_idx].astype(int), minlength=NUM_CLASSES)
    cw = np.where(counts > 0, len(train_idx) / (NUM_CLASSES * counts), 1.0)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    log.info("Class weights: %s", {CLASS_NAMES[i]: round(float(cw[i]), 3) for i in range(NUM_CLASSES)})

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              pin_memory=(device == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────
    model = resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5
    )

    fvd = FVDTransform(dx_m=TILE_DX_M)

    history: dict[str, list] = {k: [] for k in ["train_loss", "val_loss", "val_acc", "lr"]}
    best_val_acc = prev_val_acc
    best_epoch   = start_epoch

    log.info("=== Basin %s fine-tuning: epochs %d→%d | LR=%.1e ===",
             basin_name.upper(), start_epoch + 1, start_epoch + epochs, learning_rate)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch = fvd(X_batch.to(device, non_blocking=True))
            y_batch = y_batch.to(device, non_blocking=True)
            ow = compute_off_axis_weights(X_batch).to(device)
            optimizer.zero_grad()
            per_loss = criterion(model(X_batch), y_batch)
            loss = (per_loss * ow).mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
            thermal_guard(threshold=gpu_temp_threshold, batch_num=n_batches)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = fvd(X_batch.to(device, non_blocking=True))
                y_batch = y_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                val_loss_sum += criterion(logits, y_batch).mean().item()
                val_correct  += (logits.argmax(1) == y_batch).sum().item()
                val_total    += len(y_batch)

        val_acc    = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(n_batches, 1)
        val_loss   = val_loss_sum  / max(len(val_loader), 1)
        scheduler.step(val_acc)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss,   6))
        history["val_acc"].append(round(val_acc,     6))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        log.info("Epoch %3d | train=%.4f | val=%.4f | acc=%.4f | lr=%.2e | GPU=%.0f°C",
                 epoch, train_loss, val_loss, val_acc,
                 optimizer.param_groups[0]["lr"], get_gpu_temp() or 0)

        if device == "cuda" and epoch_cooldown_sec > 0:
            time.sleep(epoch_cooldown_sec)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            ckpt_out = {
                "epoch":             epoch,
                "basin":             basin_name,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":           val_acc,
                "norm_stats": {
                    "mean": norm_mean.squeeze().tolist(),
                    "std":  norm_std.squeeze().tolist(),
                },
            }
            torch.save(ckpt_out, str(out_basin_dir / "best_resnet18.pt"))
            log.info("  ✓ Saved best  val_acc=%.4f → %s", val_acc, out_basin_dir / "best_resnet18.pt")

        # Always save latest
        torch.save({
            "epoch":            epoch,
            "basin":            basin_name,
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
            "norm_stats": {"mean": norm_mean.squeeze().tolist(), "std": norm_std.squeeze().tolist()},
        }, str(out_basin_dir / "latest_resnet18.pt"))

    results = {
        "basin":             basin_name,
        "resumed_from_epoch": start_epoch,
        "final_epoch":       start_epoch + epochs,
        "best_epoch":        best_epoch,
        "best_val_acc":      round(best_val_acc, 6),
        "history":           history,
        "norm_stats": {"mean": norm_mean.squeeze().tolist(), "std": norm_std.squeeze().tolist()},
    }
    results_path = out_basin_dir / "training_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Results saved: %s", results_path)
    log.info("=== Basin %s DONE — best val_acc=%.4f at epoch %d ===",
             basin_name.upper(), best_val_acc, best_epoch)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Per-basin fine-tuning of Lake Erie magnetic expert")
    ap.add_argument("--basin", required=True, choices=["west", "central", "east"],
                    help="Basin name")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "models" / "best_resnet18.pt"),
                    help="Base checkpoint to fine-tune from")
    ap.add_argument("--real-tiles",
                    help="Path to real-tiles .npz (default: wreck_hunting_ml/data/real_tiles/real_tiles_basin_{basin}.npz)")
    ap.add_argument("--synthetic",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "data" / "synthetic" / "synthetic_tiles.npz"),
                    help="Synthetic tiles for catastrophic-forgetting prevention")
    ap.add_argument("--output-dir",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "models"),
                    help="Output directory (basin sub-folder created automatically)")
    ap.add_argument("--epochs",      type=int,   default=25)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-6)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-cooldown", action="store_true",
                    help="Disable inter-epoch GPU cooldown (faster but hotter)")
    args = ap.parse_args()

    # Default real-tiles path
    real_tiles_path = args.real_tiles or str(
        REPO_ROOT / "wreck_hunting_ml" / "data" / "real_tiles"
        / f"real_tiles_basin_{args.basin}.npz"
    )

    real_npz  = Path(real_tiles_path)
    synth_npz = Path(args.synthetic) if args.synthetic else None

    if not real_npz.exists():
        log.error("Real tiles NPZ not found: %s", real_npz)
        log.error("Run wh2k_extract_real_tiles.py --basin %s first.", args.basin)
        sys.exit(1)

    finetune_basin(
        basin_name     = args.basin,
        checkpoint_path= Path(args.checkpoint),
        real_npz       = real_npz,
        synth_npz      = synth_npz,
        output_dir     = Path(args.output_dir),
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        learning_rate  = args.lr,
        device         = args.device,
        seed           = args.seed,
        epoch_cooldown_sec = 0.0 if args.no_cooldown else 2.0,
    )


if __name__ == "__main__":
    main()
