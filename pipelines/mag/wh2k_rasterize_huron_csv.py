"""
wh2k_rasterize_huron_csv.py
============================
Rasterize the raw GSC/NRCAN Lake Huron aeromagnetic CSV into a GeoTIFF
suitable for wh2k_extract_real_tiles.py and scan_grid inference.

Source:  magnetic_data/new data to digest/extracted_csv/
         Huron__Lake_-_CSV_Point_Data_.../gsc_huron.csv
         ~1,018,241 raw points
         Columns (positional, after skipping '/' comment lines):
           0=lon  1=lat  2=TIME  3=MAGLEV  4=SRVMGLEV  5=MAGRAW  6=RALT
           7=FLIGHT  8=maglev_used  9=LINE  10=LINETYPE  11=LINENAME

Output:  magnetic_data/tier_2_aero_lowalt/local/gsc_huron_highres_0_001.tif
         0.001° pixel  (~111 m N-S, ~74 m E-W at 44°N)
         RALT filter: −50 ft to 1500 ft (survey-grade only — Huron flights may be higher)
         Interpolation: scipy griddata linear + nearest-neighbour edge fill

Usage
-----
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  python scripts\\wh2k_rasterize_huron_csv.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rasterize_huron")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = (
    REPO_ROOT
    / "magnetic_data"
    / "new data to digest"
    / "extracted_csv"
    / "Huron__Lake_-_CSV_Point_Data_-_CSV_Donn_es_ponctuelles"
    / "gsc_huron.csv"
)

DEFAULT_OUT = (
    REPO_ROOT
    / "magnetic_data"
    / "tier_2_aero_lowalt"
    / "local"
    / "gsc_huron_highres_0_001.tif"
)

DEFAULT_PIXEL_DEG = 0.001   # ~111 m N-S per pixel at 44°N
RALT_MIN_FT = -50.0
RALT_MAX_FT = 1_500.0       # Huron flights may be higher altitude


# ── CSV parser ────────────────────────────────────────────────────────────────

def parse_gsc_huron_csv(path: Path) -> np.ndarray:
    """
    Return float64 array (N, 4): [lon, lat, magraw, ralt].

    File format identical to gsc_erie.csv:
    Lines starting with '/' are comments (header metadata).
    Data lines: comma-separated, positional — lon=0, lat=1, magraw=5, ralt=6.
    """
    rows: list[tuple[float, float, float, float]] = []
    n_bad = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("/"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                lon    = float(parts[0])
                lat    = float(parts[1])
                magraw = float(parts[5])
                ralt   = float(parts[6])
            except ValueError:
                n_bad += 1
                continue
            # Huron bounds: lon ~-85 to -79, lat ~43 to 47
            if not (-90.0 <= lon <= -75.0 and 42.0 <= lat <= 48.0):
                n_bad += 1
                continue
            rows.append((lon, lat, magraw, ralt))

    log.info("Parsed %d valid rows  (%d skipped)", len(rows), n_bad)
    return np.array(rows, dtype=np.float64)


# ── rasterization ─────────────────────────────────────────────────────────────

def rasterize(
    data: np.ndarray,
    pixel_deg: float,
    ralt_min: float,
    ralt_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (grid_float32, lon_arr, lat_arr) where grid is stored
    with rasterio convention: row 0 = northernmost latitude.
    """
    from scipy.interpolate import griddata

    mask = (data[:, 3] >= ralt_min) & (data[:, 3] <= ralt_max)
    kept = data[mask]
    log.info(
        "RALT filter (%.0f – %.0f ft): kept %d / %d points (%.1f%%)",
        ralt_min, ralt_max, mask.sum(), len(data), 100 * mask.sum() / len(data),
    )

    if len(kept) == 0:
        log.error("No points survived RALT filter!")
        sys.exit(1)

    lon_min = kept[:, 0].min()
    lon_max = kept[:, 0].max()
    lat_min = kept[:, 1].min()
    lat_max = kept[:, 1].max()
    log.info(
        "Extent after filter: lon %.4f – %.4f   lat %.4f – %.4f",
        lon_min, lon_max, lat_min, lat_max,
    )

    # Build target grid
    lon_arr = np.arange(lon_min, lon_max + pixel_deg * 0.5, pixel_deg)
    lat_arr = np.arange(lat_min, lat_max + pixel_deg * 0.5, pixel_deg)
    LON, LAT = np.meshgrid(lon_arr, lat_arr)

    n_grid = len(lon_arr) * len(lat_arr)
    log.info(
        "Grid: %d × %d = %d pixels  (~%.0f m/px N-S  ~%.0f m/px E-W)",
        len(lon_arr), len(lat_arr), n_grid,
        pixel_deg * 111_000,
        pixel_deg * 111_000 * np.cos(np.radians((lat_min + lat_max) / 2)),
    )
    log.info("Interpolating %d source points → %d grid pixels (linear) …", len(kept), n_grid)

    grid = griddata(
        (kept[:, 0], kept[:, 1]),
        kept[:, 2],
        (LON, LAT),
        method="linear",
        fill_value=np.nan,
    )
    log.info("Linear interpolation done.  NaN coverage: %.1f%%", 100 * np.isnan(grid).mean())

    nan_mask = np.isnan(grid)
    if nan_mask.any():
        log.info("Filling %d NaN pixels with nearest-neighbour …", nan_mask.sum())
        grid[nan_mask] = griddata(
            (kept[:, 0], kept[:, 1]),
            kept[:, 2],
            (LON[nan_mask], LAT[nan_mask]),
            method="nearest",
        )

    grid_ns = grid[::-1, :].copy().astype(np.float32)
    lat_arr_ns = lat_arr[::-1].copy()

    log.info("Grid ready: %d rows × %d cols", grid_ns.shape[0], grid_ns.shape[1])
    return grid_ns, lon_arr, lat_arr_ns


# ── GeoTIFF export ────────────────────────────────────────────────────────────

def save_geotiff(
    grid: np.ndarray,
    lon_arr: np.ndarray,
    lat_arr_ns: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError:
        log.error("rasterio not installed — cannot save GeoTIFF")
        sys.exit(1)

    nrows, ncols = grid.shape
    west  = float(lon_arr[0])
    east  = float(lon_arr[-1])
    south = float(lat_arr_ns[-1])
    north = float(lat_arr_ns[0])

    transform = from_bounds(west, south, east, north, ncols, nrows)
    nodata = np.float32(-99999.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=np.float32,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=float(nodata),
        compress="lzw",
    ) as dst:
        dst.write(grid, 1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    pixel_m_ns = abs(lat_arr_ns[0] - lat_arr_ns[-1]) / nrows * 111_000
    pixel_m_ew = abs(lon_arr[-1] - lon_arr[0]) / ncols * 111_000 * np.cos(
        np.radians((north + south) / 2)
    )
    log.info(
        "Saved → %s\n  Size: %.1f MB   |  %d × %d px  |  "
        "~%.0f m/px N-S  ~%.0f m/px E-W",
        out_path, size_mb, ncols, nrows, pixel_m_ns, pixel_m_ew,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rasterize gsc_huron.csv → GeoTIFF for wh2k tile extraction",
    )
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pixel-deg", type=float, default=DEFAULT_PIXEL_DEG)
    ap.add_argument("--ralt-min", type=float, default=RALT_MIN_FT)
    ap.add_argument("--ralt-max", type=float, default=RALT_MAX_FT)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("=== wh2k_rasterize_huron_csv ===")
    log.info("Source CSV : %s", csv_path)
    log.info("Output TIF : %s", out_path)
    log.info("Pixel size : %.4f° (~%.0f m)", args.pixel_deg, args.pixel_deg * 111_000)
    log.info("RALT range : %.0f – %.0f ft", args.ralt_min, args.ralt_max)

    data = parse_gsc_huron_csv(csv_path)
    if len(data) == 0:
        log.error("No valid data rows found — check CSV path and format")
        sys.exit(1)
    log.info("Total raw points: %d", len(data))

    grid, lon_arr, lat_arr_ns = rasterize(data, args.pixel_deg, args.ralt_min, args.ralt_max)
    save_geotiff(grid, lon_arr, lat_arr_ns, out_path)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
