"""
wh2k_satellite_mag_validate.py
================================
Validates per-basin Lake Erie expert models at known wreck coordinates by
extracting a 224×224-pixel chip centred on each wreck from the high-res
GeoTIFF (GSC Erie preferred; EMAG2 fallback) and running inference.

Two outputs:
  sat_mag_validation.csv      — one row per wreck: name, lat, lon, basin,
                                 predicted_class, confidence, top4_probs,
                                 dist_m_nearest_anomaly, tif_source
  sat_mag_validation.geojson  — same data for QGIS / web map

Wrecks are sourced from wrecks.db (table `features`, 9607 rows).
Each wreck is assigned to a basin based on its coordinates.  Models are
loaded lazily (one per basin, cached between wrecks).

Usage
-----
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe scripts\\wh2k_satellite_mag_validate.py

  # Override paths:
  python scripts\\wh2k_satellite_mag_validate.py \\
      --model-dir   wreck_hunting_ml/models \\
      --tif         magnetic_data/tier_2_aero_lowalt/local/gsc_erie_highres_0_001.tif \\
      --fallback-tif magnetic_data/tier_3_aero_regional/emag2/local_EMAG2_bessemer_erie_subset_92_5000_41_0000__75_0000_49_0000.tif \\
      --output-dir  wreck_hunting_ml/runs/satellite_validation \\
      --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}
NUM_CLASSES = 4

# Lake Erie bounding box (wrecks outside this are skipped)
ERIE_BBOX = {"lat_min": 41.2, "lat_max": 43.0, "lon_min": -84.0, "lon_max": -78.5}

BASINS: dict[str, dict] = {
    "west":    {"lon_min": -83.50, "lat_min": 41.30, "lon_max": -82.00, "lat_max": 42.20},
    "central": {"lon_min": -82.00, "lat_min": 41.50, "lon_max": -80.30, "lat_max": 42.80},
    "east":    {"lon_min": -80.30, "lat_min": 42.00, "lon_max": -78.85, "lat_max": 42.95},
}

CHIP_PX = 224

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sat_mag_validate")


# ── Geo helpers ───────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _assign_basin(lat: float, lon: float) -> Optional[str]:
    """Assign wreck to a basin; returns None if outside all basins."""
    for name, bb in BASINS.items():
        if bb["lat_min"] <= lat <= bb["lat_max"] and bb["lon_min"] <= lon <= bb["lon_max"]:
            return name
    return None


# ── Wreck loading ─────────────────────────────────────────────────────────────

def _load_erie_wrecks(db_path: Path) -> list[dict]:
    """Load Lake Erie wrecks from wrecks.db."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, latitude, longitude, "
        "  COALESCE(vessel_type, type, '') AS vessel_type "
        "FROM features "
        "WHERE latitude  BETWEEN ? AND ? "
        "  AND longitude BETWEEN ? AND ? "
        "  AND latitude IS NOT NULL AND longitude IS NOT NULL",
        (ERIE_BBOX["lat_min"], ERIE_BBOX["lat_max"],
         ERIE_BBOX["lon_min"], ERIE_BBOX["lon_max"]),
    ).fetchall()
    conn.close()

    wrecks = []
    for r in rows:
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        basin = _assign_basin(lat, lon)
        wrecks.append({
            "name":        r["name"] or "UNKNOWN",
            "lat":         lat,
            "lon":         lon,
            "vessel_type": r["vessel_type"] or "",
            "basin":       basin,
        })

    in_basin = sum(1 for w in wrecks if w["basin"])
    log.info("Loaded %d Lake Erie wrecks from db (%d in defined basins)", len(wrecks), in_basin)
    return wrecks


# ── Feature computation (mirrors wh2k_inference_scorer) ──────────────────────

def _compute_features(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import ndimage as _nd  # type: ignore[import]

    if np.any(np.isnan(grid)):
        grid = grid.copy()
        grid[np.isnan(grid)] = float(np.nanmedian(grid))

    def _nss(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        return np.sqrt(dx ** 2 + dy ** 2 + _nd.laplace(g) ** 2)

    def _vdr(g):
        fft = np.fft.fft2(g)
        ky = np.fft.fftfreq(g.shape[0]).reshape(-1, 1)
        kx = np.fft.fftfreq(g.shape[1]).reshape(1, -1)
        k = np.sqrt(kx ** 2 + ky ** 2)
        k[0, 0] = 1e-10
        return np.real(np.fft.ifft2(fft * k * 2 * np.pi))

    def _tilt(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        return np.arctan2(_vdr(g), np.sqrt(dx ** 2 + dy ** 2) + 1e-12)

    return (_nss(grid).astype(np.float32),
            _vdr(grid).astype(np.float32),
            _tilt(grid).astype(np.float32))


def _normalise_chip(chip: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(chip, 2), np.percentile(chip, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(chip, dtype=np.float32)
    return np.clip((chip - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ── TIF helpers ───────────────────────────────────────────────────────────────

class TifReader:
    """Lazy-loads a GeoTIFF and computes feature layers once."""

    def __init__(self, tif_path: Path):
        import rasterio  # type: ignore[import]
        self.path = tif_path
        with rasterio.open(str(tif_path)) as src:
            self.grid       = src.read(1).astype(np.float32)
            self.transform  = src.transform
            self.nodata     = src.nodata
            self.shape      = self.grid.shape   # (rows, cols)

        if self.nodata is not None:
            self.grid[self.grid == self.nodata] = np.nan

        # Replace NaN with median before FFT
        nan_mask = np.isnan(self.grid)
        if nan_mask.any():
            self.grid[nan_mask] = float(np.nanmedian(self.grid))

        log.info("TIF loaded: %s  shape=%s  transform=%s", tif_path.name, self.shape, self.transform)
        log.info("Computing feature layers (NSS, VDR, Tilt) on full grid ...")
        self.nss, self.vdr, self.tilt = _compute_features(self.grid)
        log.info("Feature layers ready.")

    def chip_at(self, lat: float, lon: float, chip_px: int = CHIP_PX
                ) -> Optional[tuple[np.ndarray, dict]]:
        """
        Returns ((3, chip_px, chip_px) float32 normalised chip, metadata)
        or None if the wreck centre is outside the TIF or too close to the edge.
        """
        from rasterio.transform import rowcol  # type: ignore[import]
        try:
            row, col = rowcol(self.transform, lon, lat)
        except Exception:
            return None

        half = chip_px // 2
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half

        if r0 < 0 or r1 > self.shape[0] or c0 < 0 or c1 > self.shape[1]:
            return None  # Too close to TIF edge

        nss_c  = self.nss[r0:r1,  c0:c1]
        vdr_c  = self.vdr[r0:r1,  c0:c1]
        tilt_c = self.tilt[r0:r1, c0:c1]

        if nss_c.shape != (chip_px, chip_px):
            return None

        tile = np.stack([
            _normalise_chip(nss_c),
            _normalise_chip(vdr_c),
            _normalise_chip(tilt_c),
        ], axis=0)  # (3, 224, 224)

        # Pixel resolution in metres at this latitude
        res_lon = abs(self.transform.a)    # degrees per pixel (lon)
        res_m   = res_lon * 111_000 * math.cos(math.radians(lat))

        meta = {
            "row": row, "col": col,
            "chip_r0": r0, "chip_c0": c0,
            "pixel_res_m": round(res_m, 1),
        }
        return tile, meta


# ── Model cache ───────────────────────────────────────────────────────────────

class ModelCache:
    def __init__(self, model_dir: Path, device: str):
        self.model_dir = model_dir
        self.device    = device
        self._models: dict[str, tuple] = {}   # basin -> (model, ch_mean, ch_std)

    def get(self, basin: str) -> Optional[tuple]:
        if basin in self._models:
            return self._models[basin]

        # Prefer basin-specific model; fall back to base model
        basin_path = self.model_dir / f"basin_{basin}" / "best_resnet18.pt"
        base_path  = self.model_dir / "best_resnet18.pt"

        ckpt_path = basin_path if basin_path.exists() else base_path
        if not ckpt_path.exists():
            log.warning("No model found for basin '%s' (checked %s and %s)", basin, basin_path, base_path)
            return None

        try:
            import torch
            from torchvision.models import resnet18
        except ImportError as exc:
            log.error("PyTorch required: %s", exc)
            return None

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        ns   = ckpt.get("norm_stats", {"mean": [0, 0, 0], "std": [1, 1, 1]})
        ch_mean = np.array(ns["mean"], dtype=np.float32).reshape(3, 1, 1)
        ch_std  = np.array(ns["std"],  dtype=np.float32).reshape(3, 1, 1)
        ch_std[ch_std < 1e-8] = 1.0

        model = resnet18(num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        model_type = "basin-expert" if basin_path.exists() else "base-model"
        log.info("Loaded %s for basin '%s' from %s (epoch=%d, val_acc=%.4f)",
                 model_type, basin, ckpt_path.name,
                 ckpt.get("epoch", 0), ckpt.get("val_acc", 0))

        entry = (model, ch_mean, ch_std, str(ckpt_path))
        self._models[basin] = entry
        return entry


# ── Inference on a single chip ───────────────────────────────────────────────

def _infer_chip(chip_3hw: np.ndarray, model_entry: tuple, device: str) -> dict:
    """Run model forward pass. Returns probs dict and predicted class."""
    import torch

    model, ch_mean, ch_std, ckpt_path = model_entry
    tile = (chip_3hw - ch_mean) / ch_std       # (3, 224, 224)
    x = torch.from_numpy(tile[np.newaxis]).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

    pred_cls = int(np.argmax(probs))
    return {
        "predicted_class":    pred_cls,
        "predicted_class_name": CLASS_NAMES[pred_cls],
        "confidence":         round(float(probs[pred_cls]), 4),
        "prob_geology":       round(float(probs[0]), 4),
        "prob_steel_hull":    round(float(probs[1]), 4),
        "prob_wood_cargo":    round(float(probs[2]), 4),
        "prob_wellhead":      round(float(probs[3]), 4),
        "model_checkpoint":   str(Path(ckpt_path).name),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def validate(
    model_dir: Path,
    tif_path: Path,
    fallback_tif_path: Optional[Path],
    db_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load wrecks
    wrecks = _load_erie_wrecks(db_path)
    if not wrecks:
        log.error("No wrecks loaded; check wrecks.db path")
        sys.exit(1)

    # Decide which TIF to use
    active_tif = tif_path if tif_path.exists() else None
    tif_source  = "gsc_highres"
    if active_tif is None:
        if fallback_tif_path and fallback_tif_path.exists():
            active_tif = fallback_tif_path
            tif_source = "emag2_fallback"
            log.warning("Primary TIF not found — using EMAG2 fallback: %s", fallback_tif_path)
        else:
            log.error("Neither primary TIF nor EMAG2 fallback found.  Cannot validate.")
            sys.exit(1)

    reader = TifReader(active_tif)
    cache  = ModelCache(model_dir, device)

    rows = []
    skipped_edge   = 0
    skipped_nobasin = 0
    skipped_nomodel = 0

    total = len(wrecks)
    log.info("Running inference on %d wrecks ...", total)

    for i, wreck in enumerate(wrecks, 1):
        if i % 500 == 0:
            log.info("  Progress: %d / %d", i, total)

        basin = wreck["basin"]
        if basin is None:
            skipped_nobasin += 1
            continue

        model_entry = cache.get(basin)
        if model_entry is None:
            skipped_nomodel += 1
            continue

        chip_result = reader.chip_at(wreck["lat"], wreck["lon"])
        if chip_result is None:
            skipped_edge += 1
            continue

        chip, chip_meta = chip_result
        inference_result = _infer_chip(chip, model_entry, device)

        row = {
            "name":             wreck["name"],
            "lat":              wreck["lat"],
            "lon":              wreck["lon"],
            "vessel_type":      wreck["vessel_type"],
            "basin":            basin,
            "tif_source":       tif_source,
            "pixel_res_m":      chip_meta["pixel_res_m"],
            **inference_result,
        }
        rows.append(row)

    log.info("Inference complete: %d results  (skipped: edge=%d, no-basin=%d, no-model=%d)",
             len(rows), skipped_edge, skipped_nobasin, skipped_nomodel)

    # ── Write CSV ────────────────────────────────────────────────────────
    csv_path = output_dir / "sat_mag_validation.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        log.info("CSV:     %s  (%d rows)", csv_path, len(rows))

    # ── Write GeoJSON ────────────────────────────────────────────────────
    geojson_path = output_dir / "sat_mag_validation.geojson"
    features = []
    for row in rows:
        props = {k: v for k, v in row.items() if k not in ("lat", "lon")}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": props,
        })
    gj = {"type": "FeatureCollection", "features": features}
    with open(geojson_path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh)
    log.info("GeoJSON: %s  (%d features)", geojson_path, len(features))

    # ── Quick summary ─────────────────────────────────────────────────────
    _print_summary(rows)

    return csv_path


def _print_summary(rows: list[dict]):
    if not rows:
        log.info("No results to summarise.")
        return

    total = len(rows)
    hits  = [r for r in rows if r["predicted_class"] != 0]
    log.info("\n=== Validation Summary ===")
    log.info("Total wrecks evaluated: %d", total)
    log.info("Non-geology detections: %d  (%.1f%%)", len(hits), 100.0 * len(hits) / total)

    # By basin
    for basin in BASINS:
        basin_rows = [r for r in rows if r["basin"] == basin]
        basin_hits = [r for r in basin_rows if r["predicted_class"] != 0]
        if basin_rows:
            log.info("  Basin %s: %d wrecks, %d detected (%.1f%%)",
                     basin, len(basin_rows), len(basin_hits),
                     100.0 * len(basin_hits) / len(basin_rows))

    # By predicted class
    for cls_id, cls_name in CLASS_NAMES.items():
        count = sum(1 for r in rows if r["predicted_class"] == cls_id)
        log.info("  Class %-15s : %d  (%.1f%%)", cls_name, count, 100.0 * count / total)

    # Top 10 highest-confidence non-geology
    non_geo = sorted(
        [r for r in rows if r["predicted_class"] != 0],
        key=lambda r: r["confidence"], reverse=True,
    )[:10]
    if non_geo:
        log.info("\nTop 10 highest-confidence non-geology wrecks:")
        for r in non_geo:
            log.info("  %-40s  basin=%-8s  class=%-12s  conf=%.3f  lat=%.4f  lon=%.4f",
                     r["name"][:40], r["basin"], r["predicted_class_name"],
                     r["confidence"], r["lat"], r["lon"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Validate basin mag models at known wreck sites")
    ap.add_argument("--model-dir",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "models"),
                    help="Directory containing basin_*/best_resnet18.pt")
    ap.add_argument("--tif",
                    default=str(REPO_ROOT / "magnetic_data" / "tier_2_aero_lowalt" / "local"
                                / "gsc_erie_highres_0_001.tif"),
                    help="Primary high-res GeoTIFF (GSC Erie)")
    ap.add_argument("--fallback-tif",
                    default=str(REPO_ROOT / "magnetic_data" / "tier_3_aero_regional" / "emag2"
                                / "local_EMAG2_bessemer_erie_subset_92_5000_41_0000__75_0000_49_0000.tif"),
                    help="EMAG2 fallback if primary TIF absent")
    ap.add_argument("--wreck-db",
                    default=str(REPO_ROOT / "db" / "wrecks.db"))
    ap.add_argument("--output-dir",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "runs" / "satellite_validation"))
    ap.add_argument("--device", default="cpu",
                    help="Inference device: cpu (default) or cuda")
    args = ap.parse_args()

    validate(
        model_dir        = Path(args.model_dir),
        tif_path         = Path(args.tif),
        fallback_tif_path= Path(args.fallback_tif),
        db_path          = Path(args.wreck_db),
        output_dir       = Path(args.output_dir),
        device           = args.device,
    )


if __name__ == "__main__":
    main()
