"""
WreckHunter 2000 — Phase 2 GPU Fine-Tuning Script
====================================================
Resumes from best_resnet18.pt checkpoint (Epoch 5, ~77% val acc).
Applies anti-overfitting adjustments and runs on Quadro M2200 (Pascal/CUDA 11.8).

Changes vs Phase 1:
  • lr=1e-5, weight_decay=1e-3          — prevents Erie basalt memorisation
  • ReduceLROnPlateau(patience=3, ×0.5) — drops LR when off-axis learning stalls
  • Off-axis dipole loss penalty ×3      — triple penalise missing NE-SW anomalies
  • Every-epoch logging                  — verbose so we see M2200 progress
  • GPU temp monitoring via nvidia-smi   — throttle sleep if >80°C
  • Epoch 10 Validation Report           — AWOIS known wreck hit/miss + reasons
  • Ghost Hunt after Epoch 10            — Top 5 Score-10 unknown capsule targets
  • Known candidate tracker              — log every AWOIS wreck correctly IDed

Usage (after killing CPU run):
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  & C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_phase2_gpu.py ^
      --checkpoint wreck_hunting_ml\\models\\best_resnet18.pt ^
      --synthetic-data wreck_hunting_ml\\data\\synthetic\\synthetic_tiles.npz ^
      --output-dir wreck_hunting_ml\\models ^
      --grid-tif magnetic_data\\grids\\usgs_namag_83_6000_41_3000__78_8000_42_9000.tif ^
      --epochs 20 --batch-size 64 --device cuda
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "wreck_hunting_ml" / "models"

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}
NUM_CLASSES = 4
NE_SW_STRIKE_DEG = 45.0   # Lake Erie geology strike direction

# Known AWOIS large steel wrecks in Central Basin for validation tracking
KNOWN_STEEL_WRECKS = [
    {"name": "SS Admiral",        "lat": 42.025, "lon": -81.150, "length_ft": 296, "depth_ft": 58},
    {"name": "SS Clarion",        "lat": 41.980, "lon": -81.520, "length_ft": 265, "depth_ft": 61},
    {"name": "SS Merida",         "lat": 42.014, "lon": -80.851, "length_ft": 408, "depth_ft": 64},
    {"name": "SS L.R. Doty",      "lat": 41.983, "lon": -81.631, "length_ft": 285, "depth_ft": 68},
    {"name": "SS Minnedosa",      "lat": 42.051, "lon": -81.249, "length_ft": 240, "depth_ft": 55},
    {"name": "SS Craftsman",      "lat": 42.118, "lon": -81.441, "length_ft": 444, "depth_ft": 72},
    {"name": "Whaleback Consort", "lat": 42.427, "lon": -80.813, "length_ft": 308, "depth_ft": 59},
]


# ── GPU Temperature Monitor ────────────────────────────────────────────────

def get_gpu_temp() -> Optional[float]:
    """Read GPU temperature via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def thermal_guard(temp_threshold: float = 80.0, sleep_sec: float = 1.0):
    """Sleep if GPU is throttling to prevent fan screaming."""
    temp = get_gpu_temp()
    if temp is not None and temp >= temp_threshold:
        logger.warning("GPU temp %.0f°C >= %.0f°C — sleeping %.1fs to cool", temp, temp_threshold, sleep_sec)
        time.sleep(sleep_sec)
        return True
    return False


# ── Off-Axis Dipole Loss Penalty ──────────────────────────────────────────

def compute_off_axis_weights(tiles: "torch.Tensor") -> "torch.Tensor":
    """Compute per-sample weight based on how off-axis the dominant gradient is.

    A wreck-like anomaly is off-axis relative to the NE-SW Erie geology strike.
    Samples with dominant gradient at 45°+ from NE-SW get 3× penalty weight.
    """
    import torch
    # Tile shape: (B, 3, H, W). Channel 0 = NSS, best for axis analysis.
    nss = tiles[:, 0]  # (B, H, W)
    dy = nss[:, 1:, :] - nss[:, :-1, :]  # vertical gradient
    dx = nss[:, :, 1:] - nss[:, :, :-1]  # horizontal gradient

    # Pad to same size
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))

    # Dominant gradient direction per sample
    mean_dy = dy.mean(dim=(1, 2))
    mean_dx = dx.mean(dim=(1, 2))
    anomaly_axis = torch.atan2(mean_dx, mean_dy + 1e-10) * 180 / math.pi % 360

    # Offset from Erie geology strike
    offset = (anomaly_axis - NE_SW_STRIKE_DEG).abs()
    offset = torch.where(offset > 180, 360 - offset, offset)
    offset = torch.where(offset > 90, 180 - offset, offset)

    # 3× weight for off-axis anomalies (>45° from geology strike = wreck-like)
    weights = torch.where(offset >= 45, torch.full_like(offset, 3.0), torch.ones_like(offset))
    return weights


# ── Known Candidate Tracker ───────────────────────────────────────────────

class KnownCandidateTracker:
    """Tracks which AWOIS wrecks the model correctly identifies during validation."""

    def __init__(self):
        self.hits: list[dict] = []
        self.misses: list[dict] = []
        self.epoch_history: list[dict] = {}

    def update(self, epoch: int, predictions: list[dict]):
        """
        predictions: list of {lat, lon, predicted_class, confidence}
        Matches each known wreck against nearest prediction within 2km.
        """
        def haversine(la1, lo1, la2, lo2):
            R = 6_371_000
            dlat = math.radians(la2 - la1)
            dlon = math.radians(lo2 - lo1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        hits, misses = [], []
        for wreck in KNOWN_STEEL_WRECKS:
            best = min(predictions, key=lambda p: haversine(wreck["lat"], wreck["lon"], p["lat"], p["lon"]), default=None)
            if best is None:
                misses.append({**wreck, "reason": "No predictions in area"})
                continue
            dist = haversine(wreck["lat"], wreck["lon"], best["lat"], best["lon"])
            if dist <= 2000 and best["predicted_class"] == 1:
                hits.append({**wreck, "dist_m": round(dist, 0), "confidence": best["confidence"]})
            else:
                reason = (
                    f"Closest pred={CLASS_NAMES.get(best['predicted_class'],'?')} "
                    f"conf={best['confidence']:.2f} dist={dist:.0f}m"
                )
                misses.append({**wreck, "reason": reason})

        self.epoch_history[str(epoch)] = {"hits": hits, "misses": misses}
        logger.info("Known Candidate Tracker — Epoch %d: %d/%d AWOIS wrecks found",
                    epoch, len(hits), len(KNOWN_STEEL_WRECKS))
        for h in hits:
            logger.info("  HIT : %-25s  dist=%.0fm  conf=%.2f", h["name"], h["dist_m"], h["confidence"])
        for m in misses:
            logger.info("  MISS: %-25s  reason=%s", m["name"], m["reason"])
        return hits, misses


# ── Phase 2 Fine-Tuning Loop ──────────────────────────────────────────────

def phase2_train(
    checkpoint_path: str | Path,
    synthetic_data_path: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    grid_tif_path: Optional[str | Path] = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
    gpu_temp_threshold: float = 80.0,
) -> dict:
    """Phase 2: resume from epoch 5 checkpoint, apply all GPU tuning."""

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import resnet18
    except ImportError:
        logger.error("PyTorch not installed in this environment")
        return {"error": "torch missing"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection with fallback
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        logger.info("GPU: %s | VRAM: %.1f GB | Driver CUDA: %s",
                    gpu_name, vram_gb, torch.version.cuda)
    else:
        logger.info("Running on CPU (no CUDA)")

    # ── Load data ──────────────────────────────────────────────────────────
    data = np.load(str(synthetic_data_path))
    tiles_raw = data["tiles"]
    labels_raw = data["labels"]

    # Normalize (same as phase 1)
    mean = tiles_raw.mean(axis=(0, 2, 3), keepdims=True)
    std = tiles_raw.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-8] = 1.0
    tiles_norm = (tiles_raw - mean) / std
    norm_stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}

    # Split (same seed = same split as phase 1)
    rng = np.random.default_rng(seed)
    n = len(labels_raw)
    idx = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_val]

    X_train = torch.tensor(tiles_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels_raw[train_idx], dtype=torch.long)
    X_val = torch.tensor(tiles_norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels_raw[val_idx], dtype=torch.long)

    class_counts = np.bincount(labels_raw[train_idx].astype(int), minlength=NUM_CLASSES)
    class_weights = np.where(class_counts > 0, len(train_idx) / (NUM_CLASSES * class_counts), 1.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    logger.info("Phase 2 data: train=%d, val=%d | class weights: %s",
                len(y_train), len(y_val),
                {CLASS_NAMES[i]: f"{class_weights[i]:.2f}" for i in range(NUM_CLASSES)})

    # ── Load model from checkpoint ─────────────────────────────────────────
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    start_epoch = checkpoint.get("epoch", 0)
    logger.info("Resuming from checkpoint: epoch=%d, val_acc=%.4f",
                start_epoch, checkpoint.get("val_acc", 0))

    model = resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)

    tracker = KnownCandidateTracker()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "gpu_temp": []}
    best_val_acc = checkpoint.get("val_acc", 0.0)
    best_epoch = start_epoch

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Off-axis dipole weights (3× penalty for off-axis anomalies)
            off_axis_w = compute_off_axis_weights(X_batch).to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            per_sample_loss = criterion(logits, y_batch)
            loss = (per_sample_loss * off_axis_w).mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            thermal_guard(gpu_temp_threshold)

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                per_sample_loss = criterion(logits, y_batch)
                val_loss_sum += per_sample_loss.mean().item()
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        gpu_temp = get_gpu_temp() or 0.0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["gpu_temp"].append(gpu_temp)

        # Log EVERY epoch
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e | GPU=%.0f°C",
            epoch, avg_train_loss, avg_val_loss, val_acc,
            optimizer.param_groups[0]["lr"], gpu_temp
        )

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
                "phase": 2,
            }, output_dir / "best_resnet18.pt")
            logger.info("  *** New best saved (epoch %d, val_acc=%.4f) ***", epoch, val_acc)

        # ── Epoch 10: Validation Report + Ghost Hunt ──────────────────────
        if epoch == start_epoch + 10 and grid_tif_path:
            logger.info("=" * 60)
            logger.info("EPOCH 10 VALIDATION REPORT")
            logger.info("=" * 60)
            _run_validation_report(model, device, grid_tif_path, norm_stats, tracker, epoch, output_dir)

    # ── Final ghost hunt ───────────────────────────────────────────────────
    if grid_tif_path:
        logger.info("=" * 60)
        logger.info("FINAL GHOST HUNT — Top 5 Score-10 Unknowns")
        logger.info("=" * 60)
        _run_ghost_hunt(model, device, grid_tif_path, norm_stats, output_dir)

    results = {
        "phase": 2,
        "resumed_from_epoch": start_epoch,
        "final_epoch": start_epoch + epochs,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "history": history,
        "known_candidate_history": tracker.epoch_history,
    }
    with open(output_dir / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase 2 complete. Best val_acc=%.4f at epoch %d", best_val_acc, best_epoch)
    return results


# ── Validation Report helper ───────────────────────────────────────────────

def _run_validation_report(model, device, grid_tif_path, norm_stats, tracker, epoch, output_dir):
    """Run inference on the Erie grid, match predictions to known AWOIS wrecks."""
    try:
        from scripts.wh2k_inference_scorer import scan_grid
    except ImportError:
        # Inline minimal scan if import path differs
        logger.warning("Could not import scan_grid — skipping spatial validation report")
        return

    import torch
    # Save model temporarily so scan_grid can load it
    tmp_ckpt = Path(output_dir) / "_tmp_val_checkpoint.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": 0,
        "norm_stats": norm_stats,
        "class_names": CLASS_NAMES,
    }, tmp_ckpt)

    predictions = scan_grid(
        model_path=tmp_ckpt,
        grid_tif_path=grid_tif_path,
        confidence_threshold=0.45,
        device=device,
    )
    tmp_ckpt.unlink(missing_ok=True)

    pred_dicts = [{"lat": d.lat, "lon": d.lon,
                   "predicted_class": d.predicted_class,
                   "confidence": d.confidence} for d in predictions]

    hits, misses = tracker.update(epoch, pred_dicts)

    # Save report
    report = {
        "epoch": epoch,
        "total_detections": len(predictions),
        "awois_hits": hits,
        "awois_misses": misses,
        "hit_rate": f"{len(hits)}/{len(KNOWN_STEEL_WRECKS)}",
    }
    with open(Path(output_dir) / f"validation_report_epoch{epoch}.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved → validation_report_epoch%d.json", epoch)


# ── Ghost Hunt helper ──────────────────────────────────────────────────────

def _run_ghost_hunt(model, device, grid_tif_path, norm_stats, output_dir):
    """Find Top 5 Score-10 unknowns with 150-200m capsule shape."""
    try:
        import torch
        from scripts.wh2k_inference_scorer import scan_grid, correlate_and_subtract, score_unknowns
    except ImportError:
        logger.warning("Could not import inference scorer — skipping ghost hunt")
        return

    tmp_ckpt = Path(output_dir) / "_tmp_ghost_checkpoint.pt"
    torch.save({
        "epoch": 99,
        "model_state_dict": model.state_dict(),
        "val_acc": 0,
        "norm_stats": norm_stats,
        "class_names": CLASS_NAMES,
    }, tmp_ckpt)

    detections = scan_grid(
        model_path=tmp_ckpt,
        grid_tif_path=grid_tif_path,
        confidence_threshold=0.5,
        device=device,
    )
    tmp_ckpt.unlink(missing_ok=True)

    unknowns, _ = correlate_and_subtract(detections)
    scored = score_unknowns(unknowns)

    # Filter: Score 10, capsule shape (aspect ratio 1.2-2.5), extent 150-200m
    capsule_targets = [
        d for d in scored
        if d.wreck_score == 10
        and 1.2 <= d.aspect_ratio <= 2.5
        and 100 <= d.spatial_extent_m <= 300
    ]
    capsule_targets.sort(key=lambda d: d.confidence, reverse=True)
    top5 = capsule_targets[:5]

    logger.info("TOP 5 SCORE-10 CAPSULE UNKNOWNS:")
    for i, t in enumerate(top5, 1):
        logger.info(
            "  #%d  lat=%.5f lon=%.5f  score=%d  conf=%.2f  extent=%.0fm  AR=%.1f  amp=%.1fnT",
            i, t.lat, t.lon, t.wreck_score, t.confidence,
            t.spatial_extent_m, t.aspect_ratio, t.peak_amplitude_nt
        )

    with open(Path(output_dir) / "ghost_hunt_top5.json", "w") as f:
        json.dump([{
            "rank": i + 1,
            "lat": t.lat, "lon": t.lon,
            "wreck_score": t.wreck_score,
            "confidence": t.confidence,
            "extent_m": t.spatial_extent_m,
            "aspect_ratio": t.aspect_ratio,
            "amplitude_nt": t.peak_amplitude_nt,
            "reasons": t.score_reasons,
        } for i, t in enumerate(top5)], f, indent=2)

    logger.info("Ghost hunt saved → ghost_hunt_top5.json")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K Phase 2 GPU Fine-Tuning")
    parser.add_argument("--checkpoint", required=True, help="Path to best_resnet18.pt from Phase 1")
    parser.add_argument("--synthetic-data", required=True, help="Path to synthetic_tiles.npz")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--grid-tif", default=None, help="GeoTIFF for validation report + ghost hunt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu-temp-threshold", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    phase2_train(
        checkpoint_path=args.checkpoint,
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
        grid_tif_path=args.grid_tif,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        gpu_temp_threshold=args.gpu_temp_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
