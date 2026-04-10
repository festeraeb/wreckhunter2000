"""
wh2k_ingest_gsc_erie.py
=======================
Ingests the GSC (Geological Survey of Canada) aeromagnetic survey CSV for Lake Erie
into a high-resolution GeoTIFF grid, tagged for use as Phase-3 training data.

CSV format: X (lon), Y (lat), TIME, MAGLEV, SRVMGLEV, MAGRAW, RALT, FLIGHT, ...
Comment lines begin with '/'.  MAGRAW is the raw total-field intensity in nT.

Output: magnetic_data/tier_2_aero_lowalt/local/gsc_erie_highres_grid.tif
        (also copies into magnetic_data/grids/ for pipeline compatibility)

Usage
-----
  python scripts/wh2k_ingest_gsc_erie.py [--res-deg 0.001] [--output-dir <path>]

The default 0.001° ≈ 100 m per pixel at 42 °N, sufficient to resolve
individual anomalies from the survey's ~200 m flight-line spacing.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GSC_ERIE_CSV = (
    REPO_ROOT
    / "magnetic_data"
    / "new data to digest"
    / "extracted_csv"
    / "Erie__Lake_-_CSV_Point_Data_-_CSV_Donn_es_ponctuelles"
    / "gsc_erie.csv"
)

# Lake Erie tight bbox (NAD27 ≈ WGS84 within a meter at these scales)
ERIE_BBOX = (-83.50, 41.30, -78.85, 42.95)  # lonmin, latmin, lonmax, latmax

DEFAULT_RES_DEG = 0.001  # ~100 m at 42 °N

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_gsc_erie")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_gsc_erie(csv_path: Path, bbox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lons, lats, magraw) arrays clipped to bbox, skipping comment lines."""
    lonmin, latmin, lonmax, latmax = bbox
    lons, lats, vals = [], [], []

    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        header_found = False
        col_x = col_y = col_mag = -1
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("/"):
                continue
            if not header_found:
                # First non-comment line is the header
                cols = [c.strip().upper() for c in line.split(",")]
                try:
                    col_x = cols.index("X")
                    col_y = cols.index("Y")
                except ValueError:
                    log.error("Could not find X/Y columns in header: %s", cols)
                    sys.exit(1)
                # Prefer order: MAGLEV (leveled), then SRVMGLEV, then MAGRAW
                for candidate in ("MAGLEV", "SRVMGLEV", "MAGRAW"):
                    if candidate in cols:
                        col_mag = cols.index(candidate)
                        log.info("Using column '%s' (index %d) as anomaly source", candidate, col_mag)
                        break
                if col_mag == -1:
                    log.error("No usable magnetic column found in header: %s", cols)
                    sys.exit(1)
                header_found = True
                continue

            parts = line.split(",")
            try:
                x = float(parts[col_x])
                y = float(parts[col_y])
                v = float(parts[col_mag])
            except (ValueError, IndexError):
                continue

            # Skip GSC null value (500000.00)
            if abs(v) > 100_000 or v == 500_000.0:
                continue
            if lonmin <= x <= lonmax and latmin <= y <= latmax:
                lons.append(x)
                lats.append(y)
                vals.append(v)

    log.info("Loaded %d points within Erie bbox", len(vals))
    return np.array(lons, np.float32), np.array(lats, np.float32), np.array(vals, np.float32)


def _block_average_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    vals: np.ndarray,
    bbox,
    res_deg: float,
) -> np.ndarray:
    """Bin-average points into a regular grid; fill gaps with nearest-neighbour."""
    lonmin, latmin, lonmax, latmax = bbox
    xi = np.arange(lonmin, lonmax + res_deg, res_deg)
    yi = np.arange(latmax, latmin - res_deg, -res_deg)   # top → bottom
    nrows, ncols = len(yi), len(xi)

    col_idx = np.clip(((lons - lonmin) / res_deg).astype(int), 0, ncols - 1)
    row_idx = np.clip(((latmax - lats) / res_deg).astype(int), 0, nrows - 1)

    grid_sum = np.zeros((nrows, ncols), np.float64)
    grid_cnt = np.zeros((nrows, ncols), np.int32)
    np.add.at(grid_sum, (row_idx, col_idx), vals)
    np.add.at(grid_cnt, (row_idx, col_idx), 1)

    with np.errstate(invalid="ignore"):
        grid = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan).astype(np.float32)

    n_empty = np.sum(np.isnan(grid))
    n_total = nrows * ncols
    log.info(
        "Grid shape %d×%d · %d filled · %d empty (%.1f%%)",
        nrows, ncols, n_total - n_empty, n_empty, 100.0 * n_empty / n_total,
    )

    if n_empty > 0:
        log.info("Filling gaps with nearest-neighbour interpolation ...")
        from scipy.interpolate import NearestNDInterpolator  # type: ignore[import]

        known_mask = ~np.isnan(grid)
        row_kn, col_kn = np.where(known_mask)
        interp = NearestNDInterpolator(
            list(zip(col_kn, row_kn)),  # x, y
            grid[row_kn, col_kn],
        )
        row_em, col_em = np.where(~known_mask)
        grid[row_em, col_em] = interp(col_em, row_em)

    return grid


def _write_tif(grid: np.ndarray, bbox, res_deg: float, out_path: Path) -> None:
    import rasterio  # type: ignore[import]
    from rasterio.transform import from_origin  # type: ignore[import]

    lonmin, _, _, latmax = bbox
    transform = from_origin(lonmin, latmax, res_deg, res_deg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(grid, 1)
    log.info("Wrote GeoTIFF: %s  (%d×%d)", out_path, grid.shape[1], grid.shape[0])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ingest GSC Erie aeromag CSV → GeoTIFF")
    ap.add_argument("--csv", default=str(GSC_ERIE_CSV), help="Path to gsc_erie.csv")
    ap.add_argument("--res-deg", type=float, default=DEFAULT_RES_DEG,
                    help="Grid resolution in degrees (default 0.001 ≈ 100 m)")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "magnetic_data" / "tier_2_aero_lowalt" / "local"),
                    help="Directory for output GeoTIFF")
    ap.add_argument("--also-copy-to-grids", action="store_true",
                    help="Additionally copy the output into magnetic_data/grids/")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading GSC Erie CSV: %s  (%.1f MB)", csv_path.name, csv_path.stat().st_size / 1e6)
    lons, lats, vals = _load_gsc_erie(csv_path, ERIE_BBOX)

    if len(vals) == 0:
        log.error("No valid points found — check bbox or column names.")
        sys.exit(1)

    log.info("Gridding to %.4f° resolution ...", args.res_deg)
    grid = _block_average_grid(lons, lats, vals, ERIE_BBOX, args.res_deg)

    # Subtract regional mean so we have an anomaly field (removes IGRF background)
    regional_mean = np.nanmean(grid)
    grid -= regional_mean
    log.info("Removed regional mean (%.1f nT); anomaly range [%.1f, %.1f] nT",
             regional_mean, float(np.nanmin(grid)), float(np.nanmax(grid)))

    res_str = str(args.res_deg).replace(".", "_")
    out_name = f"gsc_erie_highres_{res_str}.tif"
    out_path = Path(args.output_dir) / out_name
    _write_tif(grid, ERIE_BBOX, args.res_deg, out_path)

    if args.also_copy_to_grids:
        import shutil
        grids_dir = REPO_ROOT / "magnetic_data" / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        dest = grids_dir / out_name
        shutil.copy2(str(out_path), str(dest))
        log.info("Copied to grids/: %s", dest)

    print(f"\nOutput: {out_path}")
    print(f"Shape:  {grid.shape[0]} rows × {grid.shape[1]} cols")
    print(f"Res:    {args.res_deg}° ≈ {args.res_deg * 111_000:.0f} m/pixel at 42°N")


if __name__ == "__main__":
    main()
