"""
WreckHunter 2000 — Phase 2 GPU Fine-Tuning (Full Feature Build)
================================================================
Resumes training from best_resnet18.pt (Phase 1 checkpoint, typically epoch 5 or later).

Phase 2 additions vs Phase 1:
  • FVD preprocessing  — First Vertical Derivative replaces raw NSS channel.
                         Fourier-domain |k| filter sharpens compact wreck anomalies
                         and suppresses broad NE-SW geology field (computed on-GPU).
  • lr=1e-5, wd=1e-3   — tight anti-overfitting hyperparameters
  • ReduceLROnPlateau  — drops LR ×0.5 when off-axis stalls (patience=3)
  • Off-axis 3× penalty— triples loss weight for NE-SW off-axis dipole samples
  • Every-epoch logging — verbose progress with GPU temp monitoring
  • Discovery Report   — at (start_epoch + 10): AWOIS wreck hit/miss spatial audit
  • Ghost Hunt         — post-training Top-5 Score-10 capsule unknown targets
  • Known Candidate Tracker — records every AWOIS wreck correctly ID-ed per epoch

Usage (GPU):
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_phase2_gpu_training.py ^
    --checkpoint wreck_hunting_ml\\models\\best_resnet18.pt ^
    --synthetic-data wreck_hunting_ml\\data\\synthetic\\synthetic_tiles.npz ^
    --output-dir wreck_hunting_ml\\models ^
    --epochs 20 --batch-size 32 --device cuda

Usage (CPU fallback):
  ... --device cpu --batch-size 16
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
NE_SW_STRIKE_DEG = 45.0           # Lake Erie geology strike
TILE_DX_M = 8.928571              # meters per pixel (2000m / 224px)

# ── Negative-bias guard ────────────────────────────────────────────────────
# If the model predicts GEOLOGY for more than GEO_BIAS_THRESHOLD fraction of
# validation tiles, STEEL_HULL class weight is boosted by GEO_BOOST_FACTOR
# each epoch until the bias subsides, capped at GEO_MAX_BOOST_X × initial.
GEO_BIAS_THRESHOLD = 0.70   # fraction of val preds that are GEOLOGY_ONLY
GEO_BOOST_FACTOR   = 1.30   # multiply STEEL_HULL class weight by this amount
GEO_MAX_BOOST_X    = 5.0    # absolute cap: weight ≤ initial × GEO_MAX_BOOST_X


# ── Known AWOIS Large Steel Wrecks — Central Basin Validation Set ─────────

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
    """Read GPU temp via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def thermal_guard(
    threshold: float = 80.0,
    sleep_sec: float = 1.5,
    soft_threshold: float = 78.0,
    soft_sleep_sec: float = 0.5,
    batch_num: int = 0,
    soft_every_n_batches: int = 5,
) -> bool:
    """
    Two-tier GPU thermal protection:
      • Hard tier (>= threshold, default 80°C): sleep sleep_sec every batch.
      • Soft tier (>= soft_threshold, default 78°C): sleep soft_sleep_sec
        every soft_every_n_batches batches to stay under 80°C.
    Returns True if any sleep occurred.
    """
    temp = get_gpu_temp()
    if temp is None:
        return False
    if temp >= threshold:
        logger.warning("GPU %.0f°C >= %.0f°C (HARD) — sleeping %.1fs",
                       temp, threshold, sleep_sec)
        time.sleep(sleep_sec)
        return True
    if temp >= soft_threshold and batch_num % soft_every_n_batches == 0:
        logger.debug("GPU %.0f°C >= %.0f°C (SOFT) — micro-sleep %.1fs (batch %d)",
                     temp, soft_threshold, soft_sleep_sec, batch_num)
        time.sleep(soft_sleep_sec)
        return True
    return False


# ── FVD Preprocessing (Fourier-domain First Vertical Derivative) ──────────

class FVDTransform:
    """
    First Vertical Derivative preprocessing for magnetic tiles.

    Replaces channel 0 (NSS raw field) with FVD(NSS):
        FVD_spec(k) = |k| · NSS_spec(k)

    Effect on wreck detection:
      • Compact wreck anomalies (high spatial freq) → AMPLIFIED
      • Broad NE-SW geology strike (low spatial freq) → SUPPRESSED
      • VDR and Tilt channels unchanged (already derivative-enhanced)

    Model compatibility: output is still 3-channel — checkpoint loads fine.
    Runs on GPU using torch.fft.  Batched for speed; wavenumber grid cached.
    """

    def __init__(self, dx_m: float = TILE_DX_M):
        self.dx_m = dx_m
        self._k_cache: dict[tuple[int, int], "torch.Tensor"] = {}

    def _get_k(self, H: int, W: int, device) -> "torch.Tensor":
        key = (H, W)
        if key not in self._k_cache:
            import torch
            kx = torch.fft.fftfreq(W, d=self.dx_m) * (2 * math.pi)   # rad/m
            ky = torch.fft.fftfreq(H, d=self.dx_m) * (2 * math.pi)
            KY, KX = torch.meshgrid(ky, kx, indexing="ij")             # (H, W)
            K = torch.sqrt(KX ** 2 + KY ** 2)                          # |k|
            self._k_cache[key] = K
        return self._k_cache[key].to(device)

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x : (B, 3, H, W)  float32  on same device
        returns same shape with ch0 = FVD(NSS)
        """
        import torch
        B, C, H, W = x.shape
        K = self._get_k(H, W, x.device)          # (H, W)

        nss = x[:, 0]                              # (B, H, W)
        spec = torch.fft.fft2(nss)                 # complex (B, H, W)
        fvd = torch.real(torch.fft.ifft2(spec * K))

        out = x.clone()
        out[:, 0] = fvd
        return out


# ── Off-Axis Dipole Loss Penalty ──────────────────────────────────────────

def compute_off_axis_weights(tiles: "torch.Tensor") -> "torch.Tensor":
    """Per-sample loss weight: 3× for off-axis anomalies (>45° from geology strike)."""
    import torch
    nss = tiles[:, 0]                              # (B, H, W) uses FVD(NSS) if applied
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

    weights = torch.where(offset >= 45, torch.full_like(offset, 3.0), torch.ones_like(offset))
    return weights


# ── Known Candidate Tracker ───────────────────────────────────────────────

class KnownCandidateTracker:
    """Records AWOIS wreck hits/misses each epoch during spatial validation."""

    def __init__(self):
        self.epoch_history: dict[str, dict] = {}

    @staticmethod
    def _haversine(la1: float, lo1: float, la2: float, lo2: float) -> float:
        R = 6_371_000
        dlat = math.radians(la2 - la1)
        dlon = math.radians(lo2 - lo1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def update(self, epoch: int, predictions: list[dict]) -> tuple[list, list]:
        """
        predictions: list of {lat, lon, predicted_class, confidence}
        Returns (hits, misses).
        """
        hits, misses = [], []
        for wreck in KNOWN_STEEL_WRECKS:
            if not predictions:
                misses.append({**wreck, "reason": "No predictions in area"})
                continue
            best = min(predictions,
                       key=lambda p: self._haversine(wreck["lat"], wreck["lon"], p["lat"], p["lon"]))
            dist = self._haversine(wreck["lat"], wreck["lon"], best["lat"], best["lon"])
            if dist <= 2000 and best["predicted_class"] == 1:
                hits.append({**wreck, "dist_m": round(dist), "confidence": best["confidence"]})
            else:
                reason = (
                    f"best_pred={CLASS_NAMES.get(best['predicted_class'], '?')} "
                    f"conf={best['confidence']:.2f} dist={dist:.0f}m"
                )
                misses.append({**wreck, "reason": reason})

        self.epoch_history[str(epoch)] = {"hits": hits, "misses": misses}
        logger.info("KnownCandidateTracker — Epoch %d: %d/%d AWOIS wrecks found",
                    epoch, len(hits), len(KNOWN_STEEL_WRECKS))
        for h in hits:
            logger.info("  HIT : %-25s  dist=%4.0fm  conf=%.2f", h["name"], h["dist_m"], h["confidence"])
        for m in misses:
            logger.info("  MISS: %-25s  %s", m["name"], m["reason"])
        return hits, misses


# ── Training Loop ─────────────────────────────────────────────────────────

def phase2_train(
    checkpoint_path: str | Path,
    synthetic_data_path: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    grid_tif_path: Optional[str | Path] = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
    gpu_temp_threshold: float = 80.0,
    epoch_cooldown_sec: float = 2.0,
    real_tiles_dir: Optional[str | Path] = None,
) -> dict:
    """
    Phase 2 fine-tune:
      1. Load synthetic tiles + apply FVD preprocessing
      2. Resume from Phase 1 checkpoint
      3. Train with off-axis triple penalty + ReduceLROnPlateau
      4. Discovery Report at (start_epoch + 10)
      5. Ghost Hunt (Top-5 capsule unknowns) after final epoch
    """

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import resnet18
    except ImportError as exc:
        logger.error("PyTorch not installed: %s", exc)
        return {"error": "torch missing"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        logger.info("GPU: %s | VRAM: %.1fGB | CUDA: %s | torch: %s",
                    gpu_name, vram_gb, torch.version.cuda, torch.__version__)
    else:
        logger.info("Device: CPU")

    # ── Load norm_stats from checkpoint (avoid full-array .std() call) ───
    #    Phase 1 stores per-channel mean/std in the checkpoint — reuse them.
    #    Only recompute from a small sample if checkpoint predates Phase 1.
    ckpt_for_norm = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "norm_stats" in ckpt_for_norm and ckpt_for_norm["norm_stats"]:
        ns = ckpt_for_norm["norm_stats"]
        norm_mean = np.array(ns["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        norm_std  = np.array(ns["std"],  dtype=np.float32).reshape(1, 3, 1, 1)
        norm_std[norm_std < 1e-8] = 1.0
        norm_stats = ns
        logger.info("Reusing Phase 1 norm_stats from checkpoint")
    else:
        # Fallback: compute from a 500-tile sample — never load whole array
        logger.info("Checkpoint has no norm_stats — computing from 500-tile sample")
        _probe = np.load(str(synthetic_data_path))
        pidx = np.random.default_rng(seed).choice(len(_probe["labels"]),
                                                   min(500, len(_probe["labels"])),
                                                   replace=False)
        sample = _probe["tiles"][pidx].astype(np.float32)
        del _probe
        norm_mean = sample.mean(axis=(0, 2, 3), keepdims=True)
        norm_std  = sample.std( axis=(0, 2, 3), keepdims=True)
        norm_std[norm_std < 1e-8] = 1.0
        norm_stats = {"mean": norm_mean.squeeze().tolist(), "std": norm_std.squeeze().tolist()}
        del sample

    # ── Load full tile array then normalise in-place (peak RAM = 1× array) ─
    #    In-place ops avoid creating a second 3.9 GB temp array.
    logger.info("Loading synthetic tiles: %s", synthetic_data_path)
    _npz = np.load(str(synthetic_data_path))
    tiles_raw  = _npz["tiles"].astype(np.float32)   # (N, 3, 224, 224)
    labels_raw = _npz["labels"].astype(np.int64)
    del _npz                                          # release ZIP handle
    logger.info("Dataset loaded: %d tiles — classes %s", len(labels_raw),
                {CLASS_NAMES[i]: int(np.sum(labels_raw == i)) for i in range(NUM_CLASSES)})

    tiles_raw -= norm_mean      # in-place, no extra 3.9 GB copy
    tiles_raw /= norm_std       # in-place

    # ── Load and mix real wreck tiles (optional) ──────────────────────────
    if real_tiles_dir is not None:
        real_dir = Path(real_tiles_dir)
        real_npzs = sorted(real_dir.glob("real_tiles_basin_*.npz"))
        if not real_npzs:
            logger.warning("--real-tiles-dir: no real_tiles_basin_*.npz found in %s", real_dir)
        else:
            r_parts_t, r_parts_l = [], []
            for npz_f in real_npzs:
                _r = np.load(str(npz_f), allow_pickle=True)
                r_parts_t.append(_r["tiles"].astype(np.float32))
                r_parts_l.append(_r["labels"].astype(np.int64))
                cl_dist = {
                    CLASS_NAMES[i]: int(np.sum(_r["labels"] == i))
                    for i in range(NUM_CLASSES) if np.sum(_r["labels"] == i) > 0
                }
                logger.info("  Real tiles loaded: %s — %d tiles %s",
                            npz_f.name, len(_r["labels"]), cl_dist)
            r_tiles  = np.concatenate(r_parts_t, axis=0)
            r_labels = np.concatenate(r_parts_l, axis=0)
            del r_parts_t, r_parts_l
            # Independently z-score real tiles channel-wise so they sit in
            # the same numerical range as the z-scored synthetic tiles.
            r_mean = r_tiles.mean(axis=(0, 2, 3), keepdims=True)
            r_std  = r_tiles.std( axis=(0, 2, 3), keepdims=True)
            r_std[r_std < 1e-8] = 1.0
            r_tiles -= r_mean
            r_tiles /= r_std
            tiles_raw  = np.concatenate([tiles_raw,  r_tiles],  axis=0)
            labels_raw = np.concatenate([labels_raw, r_labels], axis=0)
            del r_tiles
            logger.info(
                "Combined dataset: %d tiles (synthetic + real) — classes %s",
                len(labels_raw),
                {CLASS_NAMES[i]: int(np.sum(labels_raw == i)) for i in range(NUM_CLASSES)},
            )

    # ── Build train/val splits ─────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n = len(labels_raw)
    idx = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train: n_train + n_val]

    logger.info("Converting to tensors (train=%d, val=%d)...", len(train_idx), len(val_idx))
    X_train = torch.from_numpy(tiles_raw[train_idx].copy())
    X_val   = torch.from_numpy(tiles_raw[val_idx].copy())
    y_train = torch.tensor(labels_raw[train_idx], dtype=torch.long)
    y_val   = torch.tensor(labels_raw[val_idx],   dtype=torch.long)
    del tiles_raw   # free the 3.9 GB array; tensors hold their own copies
    logger.info("Tensors ready — train=%d val=%d", len(y_train), len(y_val))

    counts = np.bincount(labels_raw[train_idx].astype(int), minlength=NUM_CLASSES)
    cw = np.where(counts > 0, len(train_idx) / (NUM_CLASSES * counts), 1.0)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    initial_steel_cw = float(cw[1])   # remember initial STEEL_HULL weight for boost cap

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              pin_memory=(device == "cuda"))

    logger.info("Split: train=%d val=%d | class weights %s",
                len(y_train), len(y_val),
                {CLASS_NAMES[i]: f"{cw[i]:.2f}" for i in range(NUM_CLASSES)})

    # ── FVD transform (on-GPU) ────────────────────────────────────────────
    fvd_transform = FVDTransform(dx_m=TILE_DX_M)
    logger.info("FVD preprocessing enabled — NSS channel → FVD(NSS) on-GPU")

    # ── Load checkpoint ───────────────────────────────────────────────────
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return {"error": f"checkpoint missing: {checkpoint_path}"}

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    start_epoch = ckpt.get("epoch", 5)
    prev_val_acc = ckpt.get("val_acc", 0.0)
    logger.info("Resuming from checkpoint: epoch=%d, val_acc=%.4f", start_epoch, prev_val_acc)

    model = resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Override LR to Phase 2 value after loading
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    tracker = KnownCandidateTracker()
    history: dict[str, list] = {k: [] for k in ["train_loss", "val_loss", "val_acc", "lr", "gpu_temp", "geology_frac"]}
    best_val_acc = prev_val_acc
    best_epoch   = start_epoch
    discovery_report_epoch = start_epoch + 10

    # ── Main training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        train_loss_sum, n_batches = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # ① FVD preprocessing — enhance compact anomalies on GPU
            X_batch = fvd_transform(X_batch)

            # ② Off-axis weight (3× penalty for off-axis dipoles)
            off_axis_w = compute_off_axis_weights(X_batch).to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            per_sample_loss = criterion(logits, y_batch)
            loss = (per_sample_loss * off_axis_w).mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            thermal_guard(
                threshold=gpu_temp_threshold,
                soft_threshold=78.0,
                soft_sleep_sec=0.5,
                batch_num=n_batches,
                soft_every_n_batches=5,
            )

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_correct, val_total, val_geo_preds = 0.0, 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                X_batch = fvd_transform(X_batch)
                logits = model(X_batch)
                preds  = logits.argmax(1)
                per_sample_loss = criterion(logits, y_batch)
                val_loss_sum  += per_sample_loss.mean().item()
                val_correct   += (preds == y_batch).sum().item()
                val_geo_preds += (preds == 0).sum().item()
                val_total     += len(y_batch)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        # ── Negative-bias monitor ──────────────────────────────────────────
        geo_frac = val_geo_preds / max(val_total, 1)
        history["geology_frac"].append(round(geo_frac, 4))
        if geo_frac > GEO_BIAS_THRESHOLD:
            if cw[1] < initial_steel_cw * GEO_MAX_BOOST_X:
                cw[1] = min(cw[1] * GEO_BOOST_FACTOR,
                            initial_steel_cw * GEO_MAX_BOOST_X)
                cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=cw_tensor, reduction="none")
                logger.warning(
                    "Epoch %d  NEGATIVE BIAS — geo_frac=%.1f%% > %.0f%%  |"
                    "  STEEL_HULL cw boosted → %.2f (cap %.2f)",
                    epoch, geo_frac * 100, GEO_BIAS_THRESHOLD * 100,
                    cw[1], initial_steel_cw * GEO_MAX_BOOST_X,
                )
            else:
                logger.info(
                    "Epoch %d  bias=%.1f%% (at weight cap %.2f — no further boost)",
                    epoch, geo_frac * 100, cw[1],
                )

        gpu_temp = get_gpu_temp() or 0.0
        history["train_loss"].append(round(avg_train_loss, 6))
        history["val_loss"].append(round(avg_val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["gpu_temp"].append(gpu_temp)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | "
            "geo=%.1f%% | lr=%.2e | GPU=%.0f°C",
            epoch, avg_train_loss, avg_val_loss, val_acc,
            geo_frac * 100, optimizer.param_groups[0]["lr"], gpu_temp,
        )

        # Extra inter-epoch cooldown to reduce sustained thermal load.
        # Per-batch thermal_guard already throttles hot batches; this is a
        # deterministic pause between epochs for additional safety.
        if device == "cuda" and epoch_cooldown_sec > 0:
            logger.info("Epoch %3d cooldown: sleeping %.1fs for thermal safety",
                        epoch, epoch_cooldown_sec)
            time.sleep(epoch_cooldown_sec)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":             val_acc,
                "norm_stats":          norm_stats,
                "class_names":         CLASS_NAMES,
                "phase":               2,
                "fvd_preprocessing":   True,
            }, output_dir / "best_resnet18.pt")
            logger.info("  ★ New best — epoch %d, val_acc=%.4f", epoch, val_acc)

        # ── Discovery Report at (start_epoch + 10) ──────────────────────
        if epoch == discovery_report_epoch and grid_tif_path:
            logger.info("=" * 65)
            logger.info("DISCOVERY REPORT — Epoch %d", epoch)
            logger.info("=" * 65)
            _run_discovery_report(model, device, grid_tif_path, norm_stats,
                                  fvd_transform, tracker, epoch, output_dir)

    # ── Ghost Hunt (final) ────────────────────────────────────────────────
    if grid_tif_path:
        logger.info("=" * 65)
        logger.info("GHOST HUNT — Top-5 Score-10 Capsule Unknowns")
        logger.info("=" * 65)
        _run_ghost_hunt(model, device, grid_tif_path, norm_stats, fvd_transform, output_dir)

    results = {
        "phase":                 2,
        "fvd_preprocessing":     True,
        "resumed_from_epoch":    start_epoch,
        "final_epoch":           start_epoch + epochs,
        "best_epoch":            best_epoch,
        "best_val_acc":          round(best_val_acc, 4),
        "history":               history,
        "known_candidate_history": tracker.epoch_history,
    }
    out_path = output_dir / "phase2_training_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Phase 2 complete — best val_acc=%.4f at epoch %d → %s",
                best_val_acc, best_epoch, out_path)
    return results


# ── Discovery Report ──────────────────────────────────────────────────────

def _run_discovery_report(
    model, device, grid_tif_path, norm_stats, fvd_transform,
    tracker: KnownCandidateTracker, epoch: int, output_dir: Path,
) -> None:
    """Spatial validation: match model detections against AWOIS known wrecks."""
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wh2k_inference_scorer import scan_grid
    except ImportError:
        logger.warning("wh2k_inference_scorer not importable — skipping Discovery Report")
        return

    import torch
    tmp_ckpt = output_dir / "_tmp_discovery.pt"
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "val_acc":          0,
        "norm_stats":       norm_stats,
        "class_names":      CLASS_NAMES,
        "fvd_preprocessing": True,
    }, tmp_ckpt)

    try:
        predictions = scan_grid(
            model_path=tmp_ckpt,
            grid_tif_path=grid_tif_path,
            confidence_threshold=0.45,
            device=device,
        )
        pred_dicts = [
            {"lat": d.lat, "lon": d.lon,
             "predicted_class": d.predicted_class,
             "confidence": d.confidence}
            for d in predictions
        ]
        hits, misses = tracker.update(epoch, pred_dicts)
        report = {
            "epoch":             epoch,
            "total_detections":  len(predictions),
            "awois_hits":        hits,
            "awois_misses":      misses,
            "hit_rate":          f"{len(hits)}/{len(KNOWN_STEEL_WRECKS)}",
        }
        out = output_dir / f"discovery_report_epoch{epoch}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Discovery Report → %s  (hit_rate %s)", out, report["hit_rate"])
    finally:
        tmp_ckpt.unlink(missing_ok=True)


# ── Ghost Hunt ────────────────────────────────────────────────────────────

def _run_ghost_hunt(
    model, device, grid_tif_path, norm_stats, fvd_transform, output_dir: Path,
) -> None:
    """Top-5 Score-10 unknown capsule targets (150-300m extent, AR 1.2-2.5)."""
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wh2k_inference_scorer import scan_grid, correlate_and_subtract, score_unknowns
    except ImportError:
        logger.warning("wh2k_inference_scorer not importable — skipping Ghost Hunt")
        return

    import torch
    tmp_ckpt = output_dir / "_tmp_ghost.pt"
    torch.save({
        "epoch":            999,
        "model_state_dict": model.state_dict(),
        "val_acc":          0,
        "norm_stats":       norm_stats,
        "class_names":      CLASS_NAMES,
        "fvd_preprocessing": True,
    }, tmp_ckpt)

    try:
        detections = scan_grid(
            model_path=tmp_ckpt,
            grid_tif_path=grid_tif_path,
            confidence_threshold=0.50,
            device=device,
        )
        unknowns, _ = correlate_and_subtract(detections)
        scored = score_unknowns(unknowns)

        # Filter: Score 10, capsule shape, 150-300m extent
        caps = [
            d for d in scored
            if d.wreck_score == 10
            and 1.2 <= d.aspect_ratio <= 2.5
            and 100 <= d.spatial_extent_m <= 300
        ]
        caps.sort(key=lambda d: d.confidence, reverse=True)
        top5 = caps[:5]

        logger.info("TOP 5 GHOST HUNT CAPSULES:")
        for i, t in enumerate(top5, 1):
            logger.info(
                "  #%d  lat=%.5f  lon=%.5f  score=%d  conf=%.2f  extent=%.0fm  AR=%.1f  amp=%.1fnT",
                i, t.lat, t.lon, t.wreck_score, t.confidence,
                t.spatial_extent_m, t.aspect_ratio, t.peak_amplitude_nt,
            )

        out = output_dir / "ghost_hunt_top5.json"
        with open(out, "w") as f:
            json.dump([{
                "rank":         i + 1,
                "lat":          t.lat,
                "lon":          t.lon,
                "wreck_score":  t.wreck_score,
                "confidence":   t.confidence,
                "extent_m":     t.spatial_extent_m,
                "aspect_ratio": t.aspect_ratio,
                "amplitude_nt": t.peak_amplitude_nt,
                "reasons":      t.score_reasons,
            } for i, t in enumerate(top5)], f, indent=2)
        logger.info("Ghost Hunt saved → %s", out)
    finally:
        tmp_ckpt.unlink(missing_ok=True)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(REPO_ROOT / "wreck_hunting_ml" / "phase2_gpu_training.log",
                                mode="a", encoding="utf-8"),
        ],
    )

    p = argparse.ArgumentParser(description="WH2K Phase 2 GPU Training — FVD + Off-Axis Penalty")
    p.add_argument("--checkpoint",      required=True,
                   help="Path to best_resnet18.pt from Phase 1")
    p.add_argument("--synthetic-data",  required=True,
                   help="Path to synthetic_tiles.npz")
    p.add_argument("--output-dir",      default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--grid-tif",        default=None,
                   help="GeoTIFF for Discovery Report + Ghost Hunt (optional)")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--weight-decay",    type=float, default=1e-3)
    p.add_argument("--device",          default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gpu-temp",        type=float, default=80.0,
                   help="Temperature threshold (°C) for thermal throttle")
    p.add_argument("--epoch-cooldown",  type=float, default=2.0,
                   help="Seconds to sleep between epochs (extra thermal guard)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--real-tiles-dir",  default=None,
                   help="Directory containing real_tiles_basin_*.npz from "
                        "wh2k_extract_real_tiles.py (optional)")
    args = p.parse_args()

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
        gpu_temp_threshold=args.gpu_temp,
        epoch_cooldown_sec=args.epoch_cooldown,
        seed=args.seed,
        real_tiles_dir=args.real_tiles_dir,
    )


if __name__ == "__main__":
    main()
