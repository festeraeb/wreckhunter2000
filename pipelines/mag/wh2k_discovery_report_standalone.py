"""
WreckHunter 2000 — Standalone Discovery Report
================================================
Runs the Epoch 20 Discovery Report without any training.
Loads best_resnet18.pt, scans the Erie grid (CPU or GPU),
matches against AWOIS known wrecks, prints hit/miss summary,
and writes discovery_report_standalone.json.

Usage:
  cd C:/Users/thomf/programming/Bagrecovery
  python -W ignore scripts/wh2k_discovery_report_standalone.py
    --checkpoint wreck_hunting_ml/models/best_resnet18.pt
    --grid-tif magnetic_data/grids/usgs_namag_83_6000_41_3000__78_8000_42_9000.tif
    --device cpu
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            REPO_ROOT / "wreck_hunting_ml" / "discovery_report_standalone.log",
            mode="a", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# Known AWOIS steel wrecks — same list as training script
KNOWN_STEEL_WRECKS = [
    {"name": "SS Admiral",        "lat": 42.025, "lon": -81.150, "length_ft": 296, "depth_ft": 58},
    {"name": "SS Clarion",        "lat": 41.980, "lon": -81.520, "length_ft": 265, "depth_ft": 61},
    {"name": "SS Merida",         "lat": 42.014, "lon": -80.851, "length_ft": 408, "depth_ft": 64},
    {"name": "SS L.R. Doty",      "lat": 41.983, "lon": -81.631, "length_ft": 285, "depth_ft": 68},
    {"name": "SS Minnedosa",      "lat": 42.051, "lon": -81.249, "length_ft": 240, "depth_ft": 55},
    {"name": "SS Craftsman",      "lat": 42.118, "lon": -81.441, "length_ft": 444, "depth_ft": 72},
    {"name": "Whaleback Consort", "lat": 42.427, "lon": -80.813, "length_ft": 308, "depth_ft": 59},
]

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}


def haversine(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6_371_000
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def run_discovery_report(
    checkpoint_path: Path,
    grid_tif_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.45,
    device: str = "cpu",
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("STANDALONE DISCOVERY REPORT")
    logger.info("  checkpoint : %s", checkpoint_path)
    logger.info("  grid       : %s", grid_tif_path)
    logger.info("  device     : %s", device)
    logger.info("  conf thresh: %.2f", confidence_threshold)
    logger.info("=" * 65)

    # ── Scan grid ─────────────────────────────────────────────────────────
    # The NAMAG Erie grid is 0.02°/px ≈ 2.2 km/px — far coarser than the
    # 9 m/px training tiles.  Standard 224-px tiling produces 0 chips.
    # Coarse-grid strategy:
    #   • Use full grid, no bbox clip.
    #   • For each pixel, extract a 5×5 neighbourhood (≈11 km × 11 km window).
    #   • Upsample to 224×224 via bilinear interpolation — the scale the model
    #     was trained on (2 km × 2 km feature context).
    #   • Run classifier.  Report all STEEL_HULL hits.
    try:
        import rasterio
        import torch
        import torch.nn.functional as F
        from torchvision.models import resnet18
        from scipy import ndimage as _nd
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return {"error": str(e)}

    # Load model
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ns = ckpt.get("norm_stats", {"mean": [0, 0, 0], "std": [1, 1, 1]})
    ch_mean = torch.tensor(ns["mean"], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.tensor(ns["std"],  dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.where(ch_std < 1e-8, torch.ones_like(ch_std), ch_std)

    model = resnet18(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load grid (full, no clip)
    with rasterio.open(str(grid_tif_path)) as src:
        grid = src.read(1).astype(np.float32)
        transform = src.transform
        if src.nodata is not None:
            grid[grid == src.nodata] = np.nan
        H, W = grid.shape
        logger.info("Full grid: %d×%d px  res=%.4f°/px  bounds=%s",
                    H, W, abs(transform.a), src.bounds)

    if np.any(np.isnan(grid)):
        grid[np.isnan(grid)] = float(np.nanmedian(grid))

    # Compute 3 derived channels over the full grid
    def _nss(g):
        dx = np.gradient(g.astype(np.float64), axis=1)
        dy = np.gradient(g.astype(np.float64), axis=0)
        dz = _nd.laplace(g.astype(np.float64))
        return np.sqrt(dx**2 + dy**2 + dz**2).astype(np.float32)

    def _vdr(g):
        f = np.fft.fft2(g.astype(np.float64))
        ny, nx = g.shape
        ky = np.fft.fftfreq(ny).reshape(-1, 1)
        kx = np.fft.fftfreq(nx).reshape(1, -1)
        k  = np.sqrt(kx**2 + ky**2)
        k[0, 0] = 1e-10
        return np.real(np.fft.ifft2(f * k * 2 * np.pi)).astype(np.float32)

    def _tilt(g):
        dx  = np.gradient(g.astype(np.float64), axis=1)
        dy  = np.gradient(g.astype(np.float64), axis=0)
        thdr = np.sqrt(dx**2 + dy**2)
        return np.arctan2(_vdr(g), thdr + 1e-12).astype(np.float32)

    logger.info("Computing NSS / VDR / Tilt layers...")
    nss_g  = _nss(grid)
    vdr_g  = _vdr(grid)
    tilt_g = _tilt(grid)

    # ── Coarse-grid scan  ─────────────────────────────────────────────────
    # neighbourhood radius (pixels): 2 → 5×5 window ≈ 11 km at 0.02°/px
    RADIUS    = 2
    PAD       = RADIUS
    TARGET_PX = 224   # model input size

    nss_p  = np.pad(nss_g,  PAD, mode="reflect")
    vdr_p  = np.pad(vdr_g,  PAD, mode="reflect")
    tilt_p = np.pad(tilt_g, PAD, mode="reflect")

    logger.info("Scanning %d×%d grid (5×5 neighbourhood → %dpx, stride 1)…",
                H, W, TARGET_PX)

    from rasterio.transform import xy as _xy

    detections_raw = []

    with torch.no_grad():
        for r in range(H):
            for c in range(W):
                rp, cp = r + PAD, c + PAD
                n_chip  = nss_p [rp - RADIUS: rp + RADIUS + 1,
                                  cp - RADIUS: cp + RADIUS + 1]
                v_chip  = vdr_p [rp - RADIUS: rp + RADIUS + 1,
                                  cp - RADIUS: cp + RADIUS + 1]
                t_chip  = tilt_p[rp - RADIUS: rp + RADIUS + 1,
                                  cp - RADIUS: cp + RADIUS + 1]

                tile = np.stack([n_chip, v_chip, t_chip], axis=0)  # (3, 5, 5)
                t    = torch.tensor(tile[np.newaxis], dtype=torch.float32).to(device)
                # Upsample to 224×224
                t    = F.interpolate(t, size=(TARGET_PX, TARGET_PX),
                                     mode="bilinear", align_corners=False)
                t    = (t - ch_mean) / ch_std

                logits = model(t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class  = int(np.argmax(probs))
                confidence  = float(probs[pred_class])

                if pred_class == 1 and confidence >= confidence_threshold:
                    # rasterio xy: row, col → lon, lat
                    lon, lat = _xy(transform, r, c)
                    detections_raw.append({
                        "lat": lat, "lon": lon,
                        "predicted_class": pred_class,
                        "confidence": round(confidence, 3),
                        "class_name": "STEEL_HULL",
                        "all_probs": {
                            "GEOLOGY_ONLY": round(float(probs[0]), 3),
                            "STEEL_HULL":   round(float(probs[1]), 3),
                            "WOOD_CARGO":   round(float(probs[2]), 3),
                            "WELLHEAD":     round(float(probs[3]), 3),
                        },
                    })

    logger.info("Scan complete: %d STEEL_HULL detections (conf >= %.2f)",
                len(detections_raw), confidence_threshold)

    if not detections_raw:
        logger.warning("Zero detections. The model may need lower confidence "
                       "threshold — try --confidence 0.30")

    # ── Match AWOIS wrecks ─────────────────────────────────────────────────
    MATCH_RADIUS_M = 3000  # wider 3 km window for coarse grid

    hits, misses = [], []
    for wreck in KNOWN_STEEL_WRECKS:
        if not detections_raw:
            misses.append({**wreck, "reason": "No detections produced by scan"})
            continue
        best = min(detections_raw,
                   key=lambda p: haversine(wreck["lat"], wreck["lon"], p["lat"], p["lon"]))
        dist = haversine(wreck["lat"], wreck["lon"], best["lat"], best["lon"])

        if dist <= MATCH_RADIUS_M:
            entry = {**wreck, "dist_m": round(dist), "confidence": best["confidence"],
                     "nearest_lat": best["lat"], "nearest_lon": best["lon"]}
            hits.append(entry)
            logger.info("  ✔ HIT : %-25s  dist=%4.0fm  conf=%.2f",
                        wreck["name"], dist, best["confidence"])
        else:
            reason = (f"nearest STEEL_HULL conf={best['confidence']:.2f}  "
                      f"dist={dist:.0f}m  @ {best['lat']:.4f},{best['lon']:.4f}")
            misses.append({**wreck, "reason": reason})
            logger.info("  ✘ MISS: %-25s  %s", wreck["name"], reason)

    hit_rate = f"{len(hits)}/{len(KNOWN_STEEL_WRECKS)}"
    logger.info("─" * 65)
    logger.info("RESULT: %s AWOIS wrecks found", hit_rate)
    logger.info("─" * 65)

    detections_raw.sort(key=lambda d: d["confidence"], reverse=True)
    logger.info("Top STEEL_HULL detections:")
    for i, d in enumerate(detections_raw[:20], 1):
        logger.info("  #%-2d  lat=%.4f  lon=%.4f  conf=%.2f",
                    i, d["lat"], d["lon"], d["confidence"])

    # ── Write report ───────────────────────────────────────────────────────
    report = {
        "checkpoint":             str(checkpoint_path),
        "grid_tif":               str(grid_tif_path),
        "grid_shape":             [H, W],
        "device":                 device,
        "confidence_threshold":   confidence_threshold,
        "match_radius_m":         MATCH_RADIUS_M,
        "neighbourhood_px":       2 * RADIUS + 1,
        "total_steel_detections": len(detections_raw),
        "awois_hit_rate":         hit_rate,
        "hits":                   hits,
        "misses":                 misses,
        "top_detections":         detections_raw[:50],
    }

    out_path = output_dir / "discovery_report_standalone.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", out_path)

    return report


def main() -> None:
    import argparse

    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    p = argparse.ArgumentParser(description="WH2K Standalone Discovery Report")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--grid-tif",    required=True)
    p.add_argument("--output-dir",  default="wreck_hunting_ml/models")
    p.add_argument("--confidence",  type=float, default=0.45)
    p.add_argument("--device",      default=_default_device(), choices=["cpu", "cuda"])
    args = p.parse_args()

    run_discovery_report(
        checkpoint_path=Path(args.checkpoint),
        grid_tif_path=Path(args.grid_tif),
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence,
        device=args.device,
    )


if __name__ == "__main__":
    main()
