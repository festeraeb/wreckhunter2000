"""
WreckHunter 2000 — Lake Huron From-Scratch Training
=====================================================
Trains a ResNet-18 from ImageNet pretrained (or random) init using
Huron-physics synthetic tiles.  Does NOT fine-tune from Erie checkpoint.

Key differences vs Phase 2 (Erie fine-tune):
  • Model: fresh ImageNet-pretrained ResNet-18 (--random-init for truly random)
  • norm_stats: computed from Huron synthetic data, not from Erie checkpoint
  • Strike angle: 5° (N-S Canadian Shield) for off-axis penalty
  • Known wrecks: 9 Huron steel wrecks for candidate tracking
  • Learning rate: 1e-4 (higher — no pre-learned features from Erie)
  • Epochs: 100 (more iterations needed for from-scratch convergence)
  • Single-phase: no Phase 1→2 split (all mechanisms active from start)

Output: huron_agent.pt in --output-dir

Usage:
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_huron_train_from_scratch.py ^
    --synthetic-data wreck_hunting_ml\\data\\synthetic\\huron_synthetic_tiles.npz ^
    --epochs 100 --batch-size 32 --device cuda
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

# ── Huron-specific parameters ─────────────────────────────────────────────
NS_STRIKE_DEG = 5.0              # N-S Canadian Shield geological strike
TILE_DX_M = 8.928571             # meters per pixel (2000m / 224px)

# Negative-bias guard
GEO_BIAS_THRESHOLD = 0.70
GEO_BOOST_FACTOR   = 1.30
GEO_MAX_BOOST_X    = 5.0

# ── Known Huron Steel Wrecks — Validation Set ─────────────────────────────
# Found steel/iron wrecks with precise dive-site coordinates.
# Sources: wrecks.db, Sanilac Shores Preserve, Thumb Area Preserve.
KNOWN_STEEL_WRECKS = [
    {"name": "SS Cedarville",             "lat": 45.9035, "lon": -84.7300, "length_ft": 588},
    {"name": "SS Daniel J. Morrell (stern)", "lat": 44.2580, "lon": -82.8348, "length_ft": 580},  # Thumb dive
    {"name": "SS Daniel J. Morrell (bow)",   "lat": 44.3053, "lon": -82.7527, "length_ft": 580},  # Thumb dive
    {"name": "SS James Carruthers",        "lat": 44.1736, "lon": -81.6406, "length_ft": 529},
    {"name": "SS Charles S. Price",         "lat": 43.1529, "lon": -82.3529, "length_ft": 524},  # Sanilac
    {"name": "SS John McGean",             "lat": 43.9533, "lon": -82.5286, "length_ft": 432},  # Thumb dive — 1913 Storm
    {"name": "SS Hydrus",                  "lat": 43.2700, "lon": -82.5300, "length_ft": 416},
    {"name": "SS Argus",                   "lat": 44.1736, "lon": -81.6406, "length_ft": 416},
    {"name": "SS Canisteo",                "lat": 43.2357, "lon": -82.3049, "length_ft": 416},  # Sanilac
    {"name": "SS Glenorchy",               "lat": 43.8097, "lon": -82.5299, "length_ft": 365},  # Thumb dive
    {"name": "SS North Star",              "lat": 43.3993, "lon": -82.4420, "length_ft": 300},  # Sanilac
    {"name": "SS Regina",                  "lat": 43.3411, "lon": -82.4483, "length_ft": 269},  # Sanilac
    {"name": "SS Albany",                  "lat": 44.1059, "lon": -82.7003, "length_ft": 267},  # Thumb dive
]

# ── Known Huron Wood Wrecks — Steamers/Tugs with Boilers ─────────────────
# Wooden vessels with significant iron machinery (boilers, engines, shafts).
# Cast-iron boilers have thermoremanent magnetization (TRM) from heating/
# cooling cycles — much stronger than CRM from iron fittings alone.
# These create localized magnetic anomalies overlapping with small steel hulls.
KNOWN_WOOD_WRECKS = [
    # Large wooden steamers — boiler + engine + propeller shaft
    {"name": "Gov. Smith",      "lat": 44.1556, "lon": -82.7000, "length_ft": 240, "note": "wooden steamer, upright intact 175ft"},
    {"name": "Philadelphia",    "lat": 44.0687, "lon": -82.7165, "length_ft": 236, "note": "steamer + cargo of cast-iron stoves"},
    {"name": "City of Detroit", "lat": 44.2079, "lon": -83.0140, "length_ft": 167, "note": "arched propeller, intact upright 176ft"},
    {"name": "Jacob Bertschy",  "lat": 44.0572, "lon": -82.8846, "length_ft": 139, "note": "steamer, 8ft depth"},
    {"name": "Iron Chief",      "lat": 44.0939, "lon": -82.7098, "length_ft": 129, "note": "wooden steamer + coal cargo"},
    # Steamers with confirmed boiler/engine in dive description
    {"name": "Troy",            "lat": 44.1442, "lon": -83.0323, "length_ft": 100, "note": "steeple engine + boiler visible"},
    {"name": "Goliath",         "lat": 43.7835, "lon": -82.5454, "length_ft": 100, "note": "engine + boiler + early propellers, 1848"},
    {"name": "E.P. Dorr",       "lat": 44.1462, "lon": -82.7330, "length_ft": 120, "note": "salvage tug + steam pumps + iron salvage"},
    {"name": "Waverly",         "lat": 43.7645, "lon": -82.5136, "length_ft": 100, "note": "steamer, broken at 124ft"},
    {"name": "Fred Lee",        "lat": 44.2071, "lon": -82.7600, "length_ft":  70, "note": "wooden tug, intact 196ft depth"},
]


# ── GPU Temperature Monitor ────────────────────────────────────────────────

def get_gpu_temp() -> Optional[float]:
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


# ── FVD Preprocessing ──────────────────────────────────────────────────────

class FVDTransform:
    """First Vertical Derivative: |k| filter on channel 0 (NSS → FVD)."""

    def __init__(self, dx_m: float = TILE_DX_M):
        self.dx_m = dx_m
        self._k_cache: dict[tuple[int, int], "torch.Tensor"] = {}

    def _get_k(self, H: int, W: int, device) -> "torch.Tensor":
        key = (H, W)
        if key not in self._k_cache:
            import torch
            kx = torch.fft.fftfreq(W, d=self.dx_m) * (2 * math.pi)
            ky = torch.fft.fftfreq(H, d=self.dx_m) * (2 * math.pi)
            KY, KX = torch.meshgrid(ky, kx, indexing="ij")
            K = torch.sqrt(KX ** 2 + KY ** 2)
            self._k_cache[key] = K
        return self._k_cache[key].to(device)

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch
        B, C, H, W = x.shape
        K = self._get_k(H, W, x.device)
        nss = x[:, 0]
        spec = torch.fft.fft2(nss)
        fvd = torch.real(torch.fft.ifft2(spec * K))
        out = x.clone()
        out[:, 0] = fvd
        return out


# ── Off-Axis Dipole Loss Penalty ──────────────────────────────────────────

def compute_off_axis_weights(tiles: "torch.Tensor") -> "torch.Tensor":
    """3× loss weight for anomalies >45° from Huron N-S strike (5°)."""
    import torch
    nss = tiles[:, 0]
    dy = nss[:, 1:, :] - nss[:, :-1, :]
    dx = nss[:, :, 1:] - nss[:, :, :-1]
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))

    mean_dy = dy.mean(dim=(1, 2))
    mean_dx = dx.mean(dim=(1, 2))
    angle = torch.atan2(mean_dx, mean_dy + 1e-10) * 180 / math.pi % 360

    offset = (angle - NS_STRIKE_DEG).abs()
    offset = torch.where(offset > 180, 360 - offset, offset)
    offset = torch.where(offset > 90,  180 - offset, offset)

    weights = torch.where(offset >= 45, torch.full_like(offset, 3.0), torch.ones_like(offset))
    return weights


# ── Known Candidate Tracker ───────────────────────────────────────────────

class KnownCandidateTracker:
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
        hits, misses = [], []

        # Check steel wrecks (expect predicted_class == 1)
        for wreck in KNOWN_STEEL_WRECKS:
            self._check_wreck(wreck, predictions, expected_class=1, hits=hits, misses=misses)

        # Check wooden steamers with boilers (expect predicted_class == 2)
        for wreck in KNOWN_WOOD_WRECKS:
            self._check_wreck(wreck, predictions, expected_class=2, hits=hits, misses=misses)

        total = len(KNOWN_STEEL_WRECKS) + len(KNOWN_WOOD_WRECKS)
        self.epoch_history[str(epoch)] = {"hits": hits, "misses": misses}
        logger.info("KnownCandidateTracker — Epoch %d: %d/%d Huron wrecks found",
                    epoch, len(hits), total)
        for h in hits:
            logger.info("  HIT : %-25s  dist=%4.0fm  conf=%.2f", h["name"], h["dist_m"], h["confidence"])
        for m in misses:
            logger.info("  MISS: %-25s  %s", m["name"], m["reason"])
        return hits, misses

    def _check_wreck(self, wreck, predictions, expected_class, hits, misses):
        if not predictions:
            misses.append({**wreck, "reason": "No predictions in area"})
            return
        best = min(predictions,
                   key=lambda p: self._haversine(wreck["lat"], wreck["lon"], p["lat"], p["lon"]))
        dist = self._haversine(wreck["lat"], wreck["lon"], best["lat"], best["lon"])
        if dist <= 2000 and best["predicted_class"] == expected_class:
            hits.append({**wreck, "dist_m": round(dist), "confidence": best["confidence"]})
        else:
            reason = (
                f"best_pred={CLASS_NAMES.get(best['predicted_class'], '?')} "
                f"conf={best['confidence']:.2f} dist={dist:.0f}m"
            )
            misses.append({**wreck, "reason": reason})


# ── Training ──────────────────────────────────────────────────────────────

def train_huron_from_scratch(
    synthetic_data_path: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    grid_tif_path: Optional[str | Path] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
    gpu_temp_threshold: float = 80.0,
    epoch_cooldown_sec: float = 2.0,
    random_init: bool = False,
    real_tiles_dir: Optional[str | Path] = None,
) -> dict:
    """
    From-scratch Huron training:
      1. Fresh ResNet-18 (ImageNet pretrained or random init)
      2. Compute norm_stats from Huron synthetic data
      3. Train with FVD + N-S off-axis penalty + negative-bias guard
      4. Checkpoint as huron_agent.pt
    """

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import resnet18, ResNet18_Weights
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

    # ── Compute norm_stats from Huron synthetic data ──────────────────────
    # No Erie checkpoint to pull stats from — compute from a 500-tile sample.
    logger.info("Computing norm_stats from 500-tile sample of Huron data...")
    _probe = np.load(str(synthetic_data_path))
    probe_n = min(500, len(_probe["labels"]))
    pidx = np.random.default_rng(seed).choice(len(_probe["labels"]), probe_n, replace=False)
    sample = _probe["tiles"][pidx].astype(np.float32)
    del _probe
    norm_mean = sample.mean(axis=(0, 2, 3), keepdims=True)
    norm_std  = sample.std( axis=(0, 2, 3), keepdims=True)
    norm_std[norm_std < 1e-8] = 1.0
    norm_stats = {"mean": norm_mean.squeeze().tolist(), "std": norm_std.squeeze().tolist()}
    del sample
    logger.info("Norm stats: mean=%s  std=%s", norm_stats["mean"], norm_stats["std"])

    # ── Load full tile array + normalise in-place ─────────────────────────
    logger.info("Loading Huron synthetic tiles: %s", synthetic_data_path)
    _npz = np.load(str(synthetic_data_path))
    tiles_raw  = _npz["tiles"].astype(np.float32)
    labels_raw = _npz["labels"].astype(np.int64)
    del _npz
    logger.info("Dataset: %d tiles — classes %s", len(labels_raw),
                {CLASS_NAMES[i]: int(np.sum(labels_raw == i)) for i in range(NUM_CLASSES)})

    tiles_raw -= norm_mean
    tiles_raw /= norm_std

    # ── Load and mix real tiles (optional) ────────────────────────────────
    if real_tiles_dir is not None:
        real_dir = Path(real_tiles_dir)
        real_npzs = sorted(real_dir.glob("real_tiles_basin_*.npz"))
        if not real_npzs:
            logger.warning("--real-tiles-dir: no real_tiles_basin_*.npz in %s", real_dir)
        else:
            r_parts_t, r_parts_l = [], []
            for npz_f in real_npzs:
                _r = np.load(str(npz_f), allow_pickle=True)
                r_parts_t.append(_r["tiles"].astype(np.float32))
                r_parts_l.append(_r["labels"].astype(np.int64))
                logger.info("  Real tiles: %s — %d tiles", npz_f.name, len(_r["labels"]))
            r_tiles  = np.concatenate(r_parts_t, axis=0)
            r_labels = np.concatenate(r_parts_l, axis=0)
            del r_parts_t, r_parts_l
            r_mean = r_tiles.mean(axis=(0, 2, 3), keepdims=True)
            r_std  = r_tiles.std( axis=(0, 2, 3), keepdims=True)
            r_std[r_std < 1e-8] = 1.0
            r_tiles -= r_mean
            r_tiles /= r_std
            tiles_raw  = np.concatenate([tiles_raw,  r_tiles],  axis=0)
            labels_raw = np.concatenate([labels_raw, r_labels], axis=0)
            del r_tiles
            logger.info("Combined: %d tiles (synthetic + real)", len(labels_raw))

    # ── Train/val split (70/15/15) ────────────────────────────────────────
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
    del tiles_raw
    logger.info("Tensors ready — train=%d val=%d", len(y_train), len(y_val))

    # Class weights — inverse frequency
    counts = np.bincount(labels_raw[train_idx].astype(int), minlength=NUM_CLASSES)
    cw = np.where(counts > 0, len(train_idx) / (NUM_CLASSES * counts), 1.0)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    initial_steel_cw = float(cw[1])

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              pin_memory=(device == "cuda"))

    logger.info("Split: train=%d val=%d | class weights %s",
                len(y_train), len(y_val),
                {CLASS_NAMES[i]: f"{cw[i]:.2f}" for i in range(NUM_CLASSES)})

    # ── FVD transform ─────────────────────────────────────────────────────
    fvd_transform = FVDTransform(dx_m=TILE_DX_M)
    logger.info("FVD preprocessing enabled — NSS → FVD(NSS) on-GPU")

    # ── Fresh model (NOT from Erie checkpoint) ────────────────────────────
    if random_init:
        logger.info("Model: ResNet-18 with RANDOM initialization (no pretrained weights)")
        model = resnet18(weights=None, num_classes=NUM_CLASSES)
    else:
        logger.info("Model: ResNet-18 with ImageNet pretrained weights")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace classifier head — ImageNet has 1000 classes, we need 4
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    # Adapt input layer for 3-channel mag data (same shape as RGB, but different domain)
    # ImageNet conv1 expects RGB — keep the weight init, it still extracts useful features
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s (%.1f M)", f"{total_params:,}", total_params / 1e6)

    # ── Optimizer + scheduler (fresh, no checkpoint state) ────────────────
    criterion = nn.CrossEntropyLoss(weight=cw_tensor, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    tracker = KnownCandidateTracker()
    history: dict[str, list] = {
        k: [] for k in ["train_loss", "val_loss", "val_acc", "lr", "gpu_temp", "geology_frac"]
    }
    best_val_acc = 0.0
    best_epoch   = 0

    # ── Main training loop ────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, n_batches = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # FVD preprocessing
            X_batch = fvd_transform(X_batch)

            # Off-axis penalty (N-S strike, 3× for perpendicular)
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

        # ── Negative-bias monitor ──────────────────────────────────────
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
                    "Epoch %d  bias=%.1f%% (at weight cap %.2f)",
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

        # Inter-epoch cooldown
        if device == "cuda" and epoch_cooldown_sec > 0:
            time.sleep(epoch_cooldown_sec)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            import torch as _torch
            _torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":             val_acc,
                "norm_stats":          norm_stats,
                "class_names":         CLASS_NAMES,
                "phase":               "huron_from_scratch",
                "fvd_preprocessing":   True,
                "random_init":         random_init,
                "strike_deg":          NS_STRIKE_DEG,
                "known_wrecks":        [w["name"] for w in KNOWN_STEEL_WRECKS],
            }, output_dir / "huron_agent.pt")
            logger.info("  ★ New best — epoch %d, val_acc=%.4f → huron_agent.pt", epoch, val_acc)

        # ── Discovery Report at epoch 10 ──────────────────────────────
        if epoch == 10 and grid_tif_path:
            logger.info("=" * 65)
            logger.info("DISCOVERY REPORT — Epoch %d", epoch)
            logger.info("=" * 65)
            _run_discovery_report(model, device, grid_tif_path, norm_stats,
                                  fvd_transform, tracker, epoch, output_dir)

    # ── Ghost Hunt (final) ────────────────────────────────────────────────
    if grid_tif_path:
        logger.info("=" * 65)
        logger.info("GHOST HUNT — Top-5 Score-10 Huron Capsule Unknowns")
        logger.info("=" * 65)
        _run_ghost_hunt(model, device, grid_tif_path, norm_stats, fvd_transform, output_dir)

    results = {
        "lake":                  "huron",
        "phase":                 "from_scratch",
        "random_init":           random_init,
        "fvd_preprocessing":     True,
        "strike_deg":            NS_STRIKE_DEG,
        "final_epoch":           epochs,
        "best_epoch":            best_epoch,
        "best_val_acc":          round(best_val_acc, 4),
        "history":               history,
        "known_candidate_history": tracker.epoch_history,
    }
    out_path = output_dir / "huron_training_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Huron training complete — best val_acc=%.4f at epoch %d → %s",
                best_val_acc, best_epoch, out_path)
    return results


# ── Discovery Report ──────────────────────────────────────────────────────

def _run_discovery_report(
    model, device, grid_tif_path, norm_stats, fvd_transform,
    tracker: KnownCandidateTracker, epoch: int, output_dir: Path,
) -> None:
    try:
        import sys, torch
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wh2k_inference_scorer import scan_grid
    except ImportError:
        logger.warning("wh2k_inference_scorer not importable — skipping Discovery Report")
        return

    tmp_ckpt = output_dir / "_tmp_huron_discovery.pt"
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
        out = output_dir / f"huron_discovery_report_epoch{epoch}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Discovery Report → %s  (hit_rate %s)", out, report["hit_rate"])
    finally:
        tmp_ckpt.unlink(missing_ok=True)


# ── Ghost Hunt ────────────────────────────────────────────────────────────

def _run_ghost_hunt(
    model, device, grid_tif_path, norm_stats, fvd_transform, output_dir: Path,
) -> None:
    try:
        import sys, torch
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wh2k_inference_scorer import scan_grid, correlate_and_subtract, score_unknowns
    except ImportError:
        logger.warning("wh2k_inference_scorer not importable — skipping Ghost Hunt")
        return

    tmp_ckpt = output_dir / "_tmp_huron_ghost.pt"
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

        caps = [
            d for d in scored
            if d.wreck_score == 10
            and 1.2 <= d.aspect_ratio <= 2.5
            and 100 <= d.spatial_extent_m <= 300
        ]
        caps.sort(key=lambda d: d.confidence, reverse=True)
        top5 = caps[:5]

        logger.info("TOP 5 HURON GHOST HUNT CAPSULES:")
        for i, t in enumerate(top5, 1):
            logger.info(
                "  #%d  lat=%.5f  lon=%.5f  score=%d  conf=%.2f  extent=%.0fm  AR=%.1f  amp=%.1fnT",
                i, t.lat, t.lon, t.wreck_score, t.confidence,
                t.spatial_extent_m, t.aspect_ratio, t.peak_amplitude_nt,
            )

        out = output_dir / "huron_ghost_hunt_top5.json"
        with open(out, "w", encoding="utf-8") as f:
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
            logging.FileHandler(REPO_ROOT / "wreck_hunting_ml" / "huron_training.log",
                                mode="a", encoding="utf-8"),
        ],
    )

    p = argparse.ArgumentParser(
        description="WH2K — Lake Huron From-Scratch Training (ResNet-18 + FVD + N-S Off-Axis)"
    )
    p.add_argument("--synthetic-data", required=True,
                   help="Path to huron_synthetic_tiles.npz")
    p.add_argument("--output-dir",     default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--grid-tif",       default=None,
                   help="Huron GeoTIFF for Discovery Report + Ghost Hunt (optional)")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-3)
    p.add_argument("--device",         default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gpu-temp",       type=float, default=80.0)
    p.add_argument("--epoch-cooldown", type=float, default=2.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--random-init",    action="store_true",
                   help="Use random weights instead of ImageNet pretrained")
    p.add_argument("--real-tiles-dir", default=None,
                   help="Dir with real_tiles_basin_*.npz for mixing (optional)")
    args = p.parse_args()

    train_huron_from_scratch(
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
        random_init=args.random_init,
        real_tiles_dir=args.real_tiles_dir,
    )


if __name__ == "__main__":
    main()
