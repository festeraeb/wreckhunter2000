"""
wh2k_extract_real_tiles.py
==========================
Extracts labelled 224×224-pixel aeromagnetic patches per Lake Erie basin
from a GeoTIFF grid, using wrecks.db (AWOIS + Swayze, 9607 entries) to
produce positive (STEEL_HULL / WOOD_CARGO / WELLHEAD) and negative
(GEOLOGY_ONLY) training examples for the ResNet-18 basin experts.

Each output .npz contains:
  tiles  : float32 (N, 3, 224, 224) — channels: NSS, VDR, Tilt (same as
            the inference_scorer pipeline)
  labels : int64   (N,)             — 0=GEOLOGY_ONLY 1=STEEL_HULL
                                       2=WOOD_CARGO  3=WELLHEAD
  meta   : object  (N,)             — dict per sample with lat, lon, wreck_name

Basins:
  west    lon [-83.50, -82.00]  lat [41.30, 42.20]
  central lon [-82.00, -80.30]  lat [41.50, 42.80]
  east    lon [-80.30, -78.85]  lat [42.00, 42.95]

Usage
-----
  python scripts/wh2k_extract_real_tiles.py \\
      --tif magnetic_data/tier_2_aero_lowalt/local/gsc_erie_highres_0_001.tif \\
      --basin all [west|central|east] \\
      --neg-ratio 5 \\
      --out-dir wreck_hunting_ml/data/real_tiles
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

CHIP_PX = 224
WRECK_LABEL_RADIUS_M = 3_000.0   # tag chip positive if any wreck within this radius
NEG_RATIO = 5                     # negative samples per positive sample
OVERLAP_FRAC = 0.50               # stride = CHIP_PX * (1 - OVERLAP_FRAC)

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}

BASINS: dict[str, dict] = {
    "west":    {"lon_min": -83.50, "lat_min": 41.30, "lon_max": -82.00, "lat_max": 42.20},
    "central": {"lon_min": -82.00, "lat_min": 41.50, "lon_max": -80.30, "lat_max": 42.80},
    "east":    {"lon_min": -80.30, "lat_min": 42.00, "lon_max": -78.85, "lat_max": 42.95},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_real_tiles")


# ── geo helpers ───────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── 3-channel feature extraction (mirrors wh2k_inference_scorer) ─────────────

def _compute_features(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nss, vdr, tilt) full-grid arrays from a raw anomaly grid."""
    import scipy.ndimage as _nd  # type: ignore[import]

    # Impute NaN
    if np.any(np.isnan(grid)):
        grid = grid.copy()
        grid[np.isnan(grid)] = float(np.nanmedian(grid))

    def _nss(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        dz = _nd.laplace(g)
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def _vdr(g):
        fft = np.fft.fft2(g)
        ny, nx = g.shape
        ky = np.fft.fftfreq(ny).reshape(-1, 1)
        kx = np.fft.fftfreq(nx).reshape(1, -1)
        k = np.sqrt(kx ** 2 + ky ** 2)
        k[0, 0] = 1e-10
        return np.real(np.fft.ifft2(fft * k * 2 * np.pi))

    def _tilt(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        thdr = np.sqrt(dx ** 2 + dy ** 2)
        return np.arctan2(_vdr(g), thdr + 1e-12)

    nss = _nss(grid).astype(np.float32)
    vdr = _vdr(grid).astype(np.float32)
    tilt = _tilt(grid).astype(np.float32)
    return nss, vdr, tilt


def _normalise_channel(arr: np.ndarray) -> np.ndarray:
    """Robust [0,1] normalisation per chip channel."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ── wreck DB ──────────────────────────────────────────────────────────────────

def _load_wrecks(db_path: Path, basin: dict) -> list[dict]:
    """Load wrecks from wrecks.db clipped to a basin bbox (with margin)."""
    margin = 0.5  # degrees — load slightly outside basin for edge chips
    lon_min = basin["lon_min"] - margin
    lon_max = basin["lon_max"] + margin
    lat_min = basin["lat_min"] - margin
    lat_max = basin["lat_max"] + margin

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, latitude, longitude, "
        "  COALESCE(hull_material, '') AS hull_material, "
        "  COALESCE(feature_type, '') AS feature_type, "
        "  COALESCE(is_steel_freighter, 0) AS is_steel_freighter, "
        "  COALESCE(is_iron_ore_carrier, 0) AS is_iron_ore_carrier "
        "FROM features "
        "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? "
        "  AND latitude IS NOT NULL AND longitude IS NOT NULL",
        (lat_min, lat_max, lon_min, lon_max),
    ).fetchall()
    conn.close()

    wrecks = []
    for r in rows:
        name = r["name"] or ""
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        mat = (r["hull_material"] or "").upper()
        ftype = (r["feature_type"] or "").upper()
        is_steel = int(r["is_steel_freighter"] or 0)
        is_ore   = int(r["is_iron_ore_carrier"] or 0)
        # Assign class based on available hints
        if is_steel or is_ore or any(t in mat for t in ("STEEL", "IRON")):
            cls = 1  # STEEL_HULL
        elif any(t in mat for t in ("WOOD", "OAK", "ELM")):
            cls = 2  # WOOD_CARGO
        elif "STEAM" in mat or "STEAM" in ftype:
            cls = 1  # STEEL_HULL (steam = metal hull era)
        elif "WELL" in ftype or "OIL" in ftype:
            cls = 3  # WELLHEAD
        else:
            cls = 1  # default unknown → STEEL_HULL (max interest)
        wrecks.append({"name": name, "lat": lat, "lon": lon, "cls": cls})

    log.info("Loaded %d wreck records for basin [%s]", len(wrecks), basin)
    return wrecks


def _also_load_hardcoded_wrecks(basin: dict) -> list[dict]:
    """Pull the 47 hardcoded NDA + ShipwreckWorld wrecks (erie_wellhead_discriminator.py)."""
    try:
        from erie_wellhead_discriminator import get_all_known_wrecks  # type: ignore[import]
        raw = get_all_known_wrecks()
        result = []
        for w in raw:
            lat, lon = float(w.lat), float(w.lon)
            if (basin["lat_min"] - 0.5 <= lat <= basin["lat_max"] + 0.5 and
                    basin["lon_min"] - 0.5 <= lon <= basin["lon_max"] + 0.5):
                result.append({"name": w.name, "lat": lat, "lon": lon, "cls": 1})
        log.info("Appended %d hardcoded NDA wrecks for basin", len(result))
        return result
    except Exception as e:
        log.debug("erie_wellhead_discriminator not available: %s", e)
        return []


# ── chip extraction ───────────────────────────────────────────────────────────

def _chip_centers(grid_shape, chip_px: int, overlap: float, transform) -> list[tuple]:
    """Yield (row, col, lat, lon) for each chip in the grid."""
    from affine import Affine  # type: ignore[import]
    stride = int(chip_px * (1.0 - overlap))
    nrows, ncols = grid_shape
    centers = []
    for r in range(0, nrows - chip_px + 1, stride):
        for c in range(0, ncols - chip_px + 1, stride):
            cr = r + chip_px // 2
            cc = c + chip_px // 2
            lon, lat = transform * (cc + 0.5, cr + 0.5)
            centers.append((r, c, lat, lon))
    return centers


def _nearest_wreck_dist(lat, lon, wrecks: list[dict]) -> tuple[float, dict | None]:
    if not wrecks:
        return 9_999_999.0, None
    dists = [(_haversine_m(lat, lon, w["lat"], w["lon"]), w) for w in wrecks]
    return min(dists, key=lambda x: x[0])


# ── orchestration ─────────────────────────────────────────────────────────────

def extract_basin(
    tif_path: Path,
    basin_name: str,
    basin: dict,
    wrecks: list[dict],
    out_dir: Path,
    neg_ratio: int,
    rng: np.random.Generator,
) -> Path:
    import rasterio  # type: ignore[import]

    log.info("=== Basin: %s ===", basin_name.upper())

    with rasterio.open(str(tif_path)) as src:
        from rasterio.windows import from_bounds as fb  # type: ignore[import]
        window = fb(
            basin["lon_min"], basin["lat_min"],
            basin["lon_max"], basin["lat_max"],
            src.transform,
        )
        grid = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window)

    log.info("Basin grid shape: %d×%d", grid.shape[0], grid.shape[1])

    if grid.shape[0] < CHIP_PX or grid.shape[1] < CHIP_PX:
        log.warning("Basin grid too small (%s) for %dpx chips — skipping.", grid.shape, CHIP_PX)
        return None

    log.info("Computing NSS, VDR, Tilt feature layers ...")
    nss_full, vdr_full, tilt_full = _compute_features(grid)

    log.info("Scanning chips (overlap=%.0f%%) ...", OVERLAP_FRAC * 100)
    centers = _chip_centers(grid.shape, CHIP_PX, OVERLAP_FRAC, transform)
    log.info("Total chips: %d", len(centers))

    positives = []
    negatives = []

    for r, c, lat, lon in centers:
        # Clip to basin
        if not (basin["lat_min"] <= lat <= basin["lat_max"] and
                basin["lon_min"] <= lon <= basin["lon_max"]):
            continue

        dist_m, nearest_wreck = _nearest_wreck_dist(lat, lon, wrecks)

        # Extract 3-channel chip
        nss_c = nss_full[r: r + CHIP_PX, c: c + CHIP_PX].copy()
        vdr_c = vdr_full[r: r + CHIP_PX, c: c + CHIP_PX].copy()
        tilt_c = tilt_full[r: r + CHIP_PX, c: c + CHIP_PX].copy()

        # Per-chip robust normalise
        tile = np.stack([
            _normalise_channel(nss_c),
            _normalise_channel(vdr_c),
            _normalise_channel(tilt_c),
        ], axis=0)  # (3, 224, 224)

        meta = {"lat": float(lat), "lon": float(lon),
                "nearest_wreck_m": float(dist_m),
                "wreck_name": nearest_wreck["name"] if nearest_wreck else ""}

        if dist_m <= WRECK_LABEL_RADIUS_M and nearest_wreck is not None:
            positives.append((tile, nearest_wreck["cls"], meta))
        else:
            negatives.append((tile, 0, meta))  # GEOLOGY_ONLY

    n_pos = len(positives)
    n_neg_keep = min(len(negatives), n_pos * neg_ratio)
    log.info("Positives: %d   Negatives available: %d   Keeping: %d",
             n_pos, len(negatives), n_neg_keep)

    if n_pos == 0:
        log.warning("No positive samples found in basin %s — "
                    "all chips will be GEOLOGY_ONLY; training value is limited.", basin_name)
        n_neg_keep = min(len(negatives), 500)

    # Random subsample negatives
    neg_idx = rng.choice(len(negatives), size=n_neg_keep, replace=False) if n_neg_keep > 0 else []
    neg_samples = [negatives[i] for i in neg_idx]
    all_samples = positives + neg_samples
    rng.shuffle(all_samples)  # type: ignore[arg-type]

    tiles_arr = np.stack([s[0] for s in all_samples], axis=0).astype(np.float32)
    labels_arr = np.array([s[1] for s in all_samples], dtype=np.int64)
    meta_list = [s[2] for s in all_samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"real_tiles_basin_{basin_name}.npz"
    np.savez_compressed(
        str(npz_path),
        tiles=tiles_arr,
        labels=labels_arr,
        meta=np.array(meta_list, dtype=object),
    )

    log.info("Saved %d tiles → %s", len(all_samples), npz_path)
    _save_manifest(out_dir, basin_name, basin, n_pos, n_neg_keep, len(all_samples), str(npz_path))
    return npz_path


def _save_manifest(out_dir, basin_name, basin, n_pos, n_neg, n_total, npz_path):
    manifest = {
        "basin": basin_name,
        "bbox": basin,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_total": n_total,
        "wreck_label_radius_m": WRECK_LABEL_RADIUS_M,
        "chip_px": CHIP_PX,
        "overlap_frac": OVERLAP_FRAC,
        "channels": ["NSS", "VDR", "Tilt"],
        "class_names": CLASS_NAMES,
        "npz_path": npz_path,
    }
    json_path = out_dir / f"manifest_{basin_name}.json"
    with open(json_path, "w") as fh:
        import json
        json.dump(manifest, fh, indent=2)
    log.info("Manifest: %s", json_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="Input GeoTIFF (Lake Erie high-res grid)")
    ap.add_argument("--basin", default="all",
                    choices=["west", "central", "east", "all"],
                    help="Basin to process (default: all)")
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO,
                    help="Negative samples per positive (default 5)")
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "wreck_hunting_ml" / "data" / "real_tiles"),
                    help="Output directory for .npz files")
    ap.add_argument("--wreck-db",
                    default=str(REPO_ROOT / "db" / "wrecks.db"),
                    help="Path to wrecks.db SQLite database")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.exists():
        log.error("TIF not found: %s", tif_path)
        sys.exit(1)

    db_path = Path(args.wreck_db)
    if not db_path.exists():
        log.error("wrecks.db not found: %s", db_path)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    basins_to_run = BASINS if args.basin == "all" else {args.basin: BASINS[args.basin]}

    for name, basin in basins_to_run.items():
        # Add scripts/ to path for hardcoded wrecks import
        scripts_dir = str(REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        wrecks = _load_wrecks(db_path, basin)
        wrecks += _also_load_hardcoded_wrecks(basin)

        # Deduplicate by name+lat+lon
        seen = set()
        unique = []
        for w in wrecks:
            key = (round(w["lat"], 4), round(w["lon"], 4))
            if key not in seen:
                seen.add(key)
                unique.append(w)
        log.info("Unique wreck positions for %s basin: %d", name, len(unique))

        extract_basin(tif_path, name, basin, unique, out_dir, args.neg_ratio, rng)

    print("\nDone.  Files in:", out_dir)


if __name__ == "__main__":
    main()
