"""
WreckHunter 2000 — Real Magnetic Chip Extractor ("Knowns" Baseline)
=====================================================================
Extracts 2km × 2km "chips" from GeoTIFF grids at every known AWOIS wreck
and GLSC well location.  Computes the 3-channel representation
(NSS, VDR, Tilt-Angle) identical to the synthetic tile format so real and
synthetic data can be mixed in the same ResNet-18 training loop.

Labels:
  - REAL_STEEL_WRECK    (label_id=1) — AWOIS wrecks with steel/iron hull
  - REAL_WOOD_CARGO     (label_id=2) — AWOIS wrecks with wood hull + cargo
  - REAL_WELLHEAD       (label_id=3) — GLSC wells / OGSr petroleum wells
  - GEOLOGY_ONLY        (label_id=0) — Random background locations, no known sites

Data flow:
  catalog.json → pick best-resolution GeoTIFF per location → crop chip →
  compute NSS/VDR/Tilt → label → save as NPZ.

NEVER modify the wreck database.  Read-only extraction.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAG_DATA_DIR = REPO_ROOT / "magnetic_data"
GRIDS_DIR = MAG_DATA_DIR / "grids"
CATALOG_PATH = MAG_DATA_DIR / "catalog.json"
DB_PATH = REPO_ROOT / "db" / "wrecks.db"

# Chip size
DEFAULT_CHIP_EXTENT_M = 2000.0  # 2 km × 2 km
DEFAULT_CHIP_PX = 224           # Match ResNet-18 input

# ── Derived-layer computation (same as wh2k_synthetic_tiles) ────────────────

def _compute_nss(grid: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    dz = ndimage.laplace(grid)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def _compute_vdr(grid: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = 1e-10
    vdr = np.real(np.fft.ifft2(fft * k_mag * 2 * np.pi))
    return vdr


def _compute_tilt_angle(grid: np.ndarray) -> np.ndarray:
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    thdr = np.sqrt(dx**2 + dy**2)
    vdr = _compute_vdr(grid)
    return np.arctan2(vdr, thdr + 1e-12)


# ── GeoTIFF chip extraction ────────────────────────────────────────────────

def extract_chip_from_tif(
    tif_path: str | Path,
    lat: float,
    lon: float,
    chip_extent_m: float = DEFAULT_CHIP_EXTENT_M,
    chip_px: int = DEFAULT_CHIP_PX,
) -> Optional[np.ndarray]:
    """Extract a chip centred on (lat, lon) from a GeoTIFF.

    Returns a (chip_px, chip_px) float64 array, or None if the point
    is outside the raster or has insufficient valid data.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
        from rasterio.windows import Window
    except ImportError:
        logger.error("rasterio is required for chip extraction")
        return None

    with rasterio.open(str(tif_path)) as src:
        # Check bounds
        if not (src.bounds.left <= lon <= src.bounds.right and
                src.bounds.bottom <= lat <= src.bounds.top):
            return None

        row, col = rowcol(src.transform, lon, lat)
        row, col = int(row), int(col)

        # Compute radius in pixels
        res_deg_x = abs(src.transform.a)
        res_deg_y = abs(src.transform.e)
        center_lat = (src.bounds.top + src.bounds.bottom) / 2
        m_per_deg_lon = 111_320 * math.cos(math.radians(center_lat))
        m_per_deg_lat = 111_320.0
        m_per_px_x = res_deg_x * m_per_deg_lon
        m_per_px_y = res_deg_y * m_per_deg_lat

        half_extent_px_x = int(math.ceil(chip_extent_m / 2 / m_per_px_x))
        half_extent_px_y = int(math.ceil(chip_extent_m / 2 / m_per_px_y))

        # Window
        r0 = max(0, row - half_extent_px_y)
        r1 = min(src.height, row + half_extent_px_y)
        c0 = max(0, col - half_extent_px_x)
        c1 = min(src.width, col + half_extent_px_x)

        if r1 - r0 < 5 or c1 - c0 < 5:
            return None

        window = Window.from_slices((r0, r1), (c0, c1))
        patch = src.read(1, window=window).astype(np.float64)

        # Handle nodata
        if src.nodata is not None:
            patch[patch == src.nodata] = np.nan

        # Reject if >30% NaN
        if np.sum(np.isnan(patch)) / patch.size > 0.3:
            return None

        # Fill NaN with local median for clean derivatives
        if np.any(np.isnan(patch)):
            median_val = np.nanmedian(patch)
            patch[np.isnan(patch)] = median_val

        # Resize to target pixel count
        from scipy.ndimage import zoom
        zoom_y = chip_px / patch.shape[0]
        zoom_x = chip_px / patch.shape[1]
        chip = zoom(patch, (zoom_y, zoom_x), order=3)

        return chip


def chip_to_3channel(chip: np.ndarray) -> np.ndarray:
    """Convert a raw anomaly chip to 3-channel (NSS, VDR, Tilt) tile.

    Returns shape (3, H, W) float32.
    """
    nss = _compute_nss(chip)
    vdr = _compute_vdr(chip)
    tilt = _compute_tilt_angle(chip)
    return np.stack([nss, vdr, tilt], axis=0).astype(np.float32)


# ── Catalog-aware best-grid finder ──────────────────────────────────────────

def find_best_grid_for_point(
    lat: float,
    lon: float,
    catalog: dict,
    grids_dir: Path,
    prefer_tiers: list[int] | None = None,
) -> Optional[Path]:
    """Find the highest-resolution GeoTIFF that covers (lat, lon).

    Searches catalog.json for files whose bbox contains the point,
    then picks the one with the smallest resolution_m (best resolution).
    Optionally filters by tier preference (1=marine, 2=aero, 3=regional, 4=satellite).
    """
    if prefer_tiers is None:
        prefer_tiers = [1, 2, 3, 4]  # Prefer high-res first

    candidates = []
    for fid, entry in catalog.get("files", {}).items():
        bbox = entry.get("bbox")
        tier = entry.get("tier", 4)
        if bbox is None or tier not in prefer_tiers:
            continue

        lon_min, lat_min, lon_max, lat_max = bbox
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            res_m = entry.get("resolution_m", 999999)
            rel_path = entry.get("rel_path", "")
            # Check for GeoTIFF in grids dir
            tif_name = Path(rel_path).stem + ".tif"
            tif_path = grids_dir / tif_name
            if not tif_path.exists():
                # Try original path
                tif_path = grids_dir / rel_path.replace(".csv", ".tif")
            candidates.append((res_m, tier, tif_path, fid))

    if not candidates:
        return None

    # Sort by resolution (smallest = best), then tier (lowest = best)
    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    return best[2] if best[2].exists() else None


# ── Known-location loaders ──────────────────────────────────────────────────

def load_awois_wrecks(db_path: Path = DB_PATH) -> list[dict]:
    """Load known wrecks from the wrecks.db with coordinates and hull material."""
    if not db_path.exists():
        logger.warning("Wrecks DB not found at %s", db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT name, latitude, longitude, hull_material,
                   length_ft, depth_ft, gross_tons, vessel_type
            FROM features
            WHERE latitude IS NOT NULL
              AND longitude IS NOT NULL
              AND latitude != 0
              AND longitude != 0
        """).fetchall()
        wrecks = [dict(r) for r in rows]
        logger.info("Loaded %d wrecks with coordinates from %s", len(wrecks), db_path)
        return wrecks
    except Exception as e:
        logger.error("Failed to load wrecks: %s", e)
        return []
    finally:
        conn.close()


def load_well_locations(wells_csv: str | Path) -> list[dict]:
    """Load well locations from OGSr CSV (or similar).

    Expected columns: well_id, name, latitude, longitude
    """
    import csv
    wells = []
    wells_csv = Path(wells_csv)
    if not wells_csv.exists():
        logger.warning("Wells CSV not found: %s", wells_csv)
        return wells

    with open(wells_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("latitude") or row.get("lat") or 0)
                lon = float(row.get("longitude") or row.get("lon") or 0)
                if lat == 0 or lon == 0:
                    continue
                wells.append({
                    "well_id": row.get("well_id", ""),
                    "name": row.get("name", row.get("well_name", "")),
                    "latitude": lat,
                    "longitude": lon,
                })
            except (ValueError, TypeError):
                continue

    logger.info("Loaded %d wells from %s", len(wells), wells_csv)
    return wells


def generate_background_locations(
    n: int = 5000,
    bbox: tuple[float, float, float, float] = (-82.0, 41.6, -80.0, 42.6),
    known_locations: list[tuple[float, float]] | None = None,
    min_distance_m: float = 2000.0,
    seed: int = 99,
) -> list[dict]:
    """Generate random background locations in the Central Basin.

    Ensures each point is at least min_distance_m from any known wreck/well.
    """
    rng = np.random.default_rng(seed)
    lon_min, lat_min, lon_max, lat_max = bbox

    known = set()
    if known_locations:
        for lat, lon in known_locations:
            known.add((round(lat, 4), round(lon, 4)))

    def _haversine(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    locations = []
    attempts = 0
    max_attempts = n * 10

    while len(locations) < n and attempts < max_attempts:
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        attempts += 1

        # Check distance from knowns
        too_close = False
        if known_locations:
            for klat, klon in known_locations:
                if _haversine(lat, lon, klat, klon) < min_distance_m:
                    too_close = True
                    break

        if not too_close:
            locations.append({"name": f"bg_{len(locations)}", "latitude": lat, "longitude": lon})

    logger.info("Generated %d background locations (%d attempts)", len(locations), attempts)
    return locations


# ── Main Chip Extraction Pipeline ──────────────────────────────────────────

def extract_all_chips(
    chip_px: int = DEFAULT_CHIP_PX,
    chip_extent_m: float = DEFAULT_CHIP_EXTENT_M,
    n_background: int = 5000,
    wells_csv: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Full extraction pipeline: real wrecks + wells + background.

    1. Load AWOIS wrecks → extract chips → label REAL_STEEL_WRECK or REAL_WOOD_CARGO
    2. Load wells → extract chips → label REAL_WELLHEAD
    3. Generate background → extract chips → label GEOLOGY_ONLY

    Returns dict with tiles, labels, metadata (same format as synthetic generator).
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "wreck_hunting_ml" / "data" / "real_chips"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog
    if not CATALOG_PATH.exists():
        logger.error("catalog.json not found at %s", CATALOG_PATH)
        return {"tiles": np.array([]), "labels": np.array([]), "metadata": []}

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    # Find all available GeoTIFFs
    available_tifs = list(GRIDS_DIR.glob("*.tif"))
    logger.info("Found %d GeoTIFF grids in %s", len(available_tifs), GRIDS_DIR)

    # Load locations
    wrecks = load_awois_wrecks()
    wells = load_well_locations(wells_csv) if wells_csv else []

    # Gather all known coords for background exclusion
    known_coords = [(w["latitude"], w["longitude"]) for w in wrecks if w["latitude"] and w["longitude"]]
    known_coords += [(w["latitude"], w["longitude"]) for w in wells]

    backgrounds = generate_background_locations(
        n=n_background,
        known_locations=known_coords,
    )

    all_tiles = []
    all_labels = []
    all_meta = []

    def _extract_and_add(locations: list[dict], default_label_id: int, label_resolver=None):
        for loc in locations:
            lat = loc.get("latitude", 0)
            lon = loc.get("longitude", 0)
            if lat == 0 or lon == 0:
                continue

            # Find best grid
            tif_path = find_best_grid_for_point(lat, lon, catalog, GRIDS_DIR)
            if tif_path is None:
                # Fall back to any covering TIF
                for tif in available_tifs:
                    chip = extract_chip_from_tif(tif, lat, lon, chip_extent_m, chip_px)
                    if chip is not None:
                        tif_path = tif
                        break
                else:
                    continue

            chip = extract_chip_from_tif(tif_path, lat, lon, chip_extent_m, chip_px)
            if chip is None:
                continue

            tile = chip_to_3channel(chip)

            label_id = default_label_id
            if label_resolver:
                label_id = label_resolver(loc)

            all_tiles.append(tile)
            all_labels.append(label_id)
            all_meta.append({
                "name": loc.get("name", ""),
                "lat": lat,
                "lon": lon,
                "label_id": label_id,
                "source_tif": str(tif_path.name) if tif_path else "",
                "real_data": True,
            })

    # Label resolver for wrecks
    def _wreck_label(w: dict) -> int:
        mat = (w.get("hull_material") or "").lower()
        if "steel" in mat or "iron" in mat:
            return 1  # REAL_STEEL_WRECK
        else:
            return 2  # REAL_WOOD_CARGO (default for non-steel)

    logger.info("Extracting chips for %d wrecks...", len(wrecks))
    _extract_and_add(wrecks, 1, _wreck_label)

    logger.info("Extracting chips for %d wells...", len(wells))
    _extract_and_add(wells, 3)

    logger.info("Extracting chips for %d background locations...", len(backgrounds))
    _extract_and_add(backgrounds, 0)

    tiles_arr = np.array(all_tiles, dtype=np.float32) if all_tiles else np.empty((0, 3, chip_px, chip_px), dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64) if all_labels else np.empty((0,), dtype=np.int64)

    logger.info("Extracted %d real chips: steel=%d, wood=%d, well=%d, bg=%d",
                len(tiles_arr),
                int(np.sum(labels_arr == 1)),
                int(np.sum(labels_arr == 2)),
                int(np.sum(labels_arr == 3)),
                int(np.sum(labels_arr == 0)))

    # Save
    np.savez_compressed(
        output_dir / "real_chips.npz",
        tiles=tiles_arr,
        labels=labels_arr,
    )
    with open(output_dir / "real_chips_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2, default=str)

    logger.info("Saved to %s", output_dir)

    return {"tiles": tiles_arr, "labels": labels_arr, "metadata": all_meta}


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K Real Chip Extractor")
    parser.add_argument("--chip-px", type=int, default=224)
    parser.add_argument("--chip-extent-m", type=float, default=2000.0)
    parser.add_argument("--n-background", type=int, default=5000)
    parser.add_argument("--wells-csv", type=str, default=None,
                        help="Path to OGSr wells CSV")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    extract_all_chips(
        chip_px=args.chip_px,
        chip_extent_m=args.chip_extent_m,
        n_background=args.n_background,
        wells_csv=args.wells_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
