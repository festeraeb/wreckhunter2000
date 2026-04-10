#!/usr/bin/env python3
"""Magnetic Anomaly Data Acquisition & Detection Pipeline.

This is the REAL pipeline for magnetic anomaly-based wreck hunting.
It downloads aerial/satellite magnetic datasets, grids them, runs anomaly
detection via trained models, and cross-references detections against the
wrecks database.

Data sources (verified working 2025-06):
  1. USGS MRData    — North American aeromagnetic anomaly grids (NAmag)  [315 MB]
  2. USGS USmag     — US-only aeromagnetic anomaly continuation          [46 MB]
  3. WDMAM v2       — World Digital Magnetic Anomaly Map (satellite)     [~150 MB]
  4. NCEI GridExtract — NOAA NCEI on-demand grid extraction API
  5. ESA Swarm       — Satellite magnetic field data (free, VirES API)

Retired sources (URLs dead as of 2025-06):
  - EMAG2v3 (NOAA/NCEI) — all download URLs return 404
  - NRCan ftp.maps.canada.ca — entire path removed

Persistent data lake:
  All downloaded data is cached permanently under magnetic_data/ at the
  repository root.  The pipeline checks the cache first and only downloads
  when files are missing or stale.  Optional AWS S3 sync keeps a cloud
  backup of the data lake.

Stages:
  Stage 1 — Download raw data
  Stage 2 — Ingest & grid to GeoTIFF (subset to region of interest)
  Stage 3 — Tile into patches for model inference
  Stage 4 — Run anomaly detection (trained models)
  Stage 5 — Cross-reference against known wreck positions
  Stage 6 — Export candidate list + probability rasters

Usage:
  # Full pipeline
  python scripts/mag_data_pipeline.py --stages all --bbox -92.5 41.0 -75.0 49.0

  # Download only
  python scripts/mag_data_pipeline.py --stages download --sources emag2,usgs,nrcan

  # Detect from existing grids
  python scripts/mag_data_pipeline.py --stages detect --grids-dir magnetic_data/grids

  # Called from wrecks_api as a stage
  from scripts.mag_data_pipeline import run_pipeline
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import tempfile
import time
import traceback
import zipfile
from urllib.parse import parse_qsl, urlparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("mag_pipeline")

# ── Great Lakes default bounding box ─────────────────────────────────────────
DEFAULT_BBOX = (-92.5, 41.0, -75.0, 49.0)  # lonmin, latmin, lonmax, latmax
DEFAULT_RES_DEG = 2.0 / 60.0  # 2 arc-minutes
DEFAULT_PATCH_PX = 256
REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Persistent data lake ─────────────────────────────────────────────────────
# Downloaded data is cached here permanently and reused across pipeline runs.
MAG_DATA_DIR = REPO_ROOT / "magnetic_data"
MAG_DATA_RAW = MAG_DATA_DIR / "raw"
MAG_DATA_GRIDS = MAG_DATA_DIR / "grids"
MAG_DATA_PATCHES = MAG_DATA_DIR / "patches"
MAG_DATA_META = MAG_DATA_DIR / "meta"
LOCAL_MAGE_CSV_DIR = MAG_DATA_RAW / "local_mage_csv"

# Download configuration
DOWNLOAD_TIMEOUT = 600  # 10 minutes — these are 300 MB+ files
DOWNLOAD_CHUNK = 1024 * 1024  # 1 MB chunks


# ── Data source catalog ─────────────────────────────────────────────────────
@dataclass
class MagSource:
    key: str
    name: str
    urls: list[str]
    source_type: str  # "csv_zip", "arcgrid_zip", "geotiff", "api"
    paid: bool = False
    description: str = ""


NRCAN_SOURCE_KEYS = [
    "nrcan_ca_1km_rtf",
    "nrcan_ca_200m_rtf",
    "nrcan_ca_1km_vd",
    "nrcan_ca_200m_vd",
]


SOURCES = [
    MagSource(
        key="usgs_namag",
        name="USGS NAmag (MRData)",
        urls=[
            "https://mrdata.usgs.gov/magnetic/NAmag_origmrg.zip",
            "https://mrdata.usgs.gov/magnetic/NAmag_hp500.zip",
        ],
        source_type="arcgrid_zip",
        description="North American aeromagnetic anomaly grid — original merge (315 MB)",
    ),
    MagSource(
        key="usgs_usmag",
        name="USGS USmag (MRData)",
        urls=[
            "https://mrdata.usgs.gov/magnetic/USmag_origmrg.zip",
        ],
        source_type="arcgrid_zip",
        description="US aeromagnetic anomaly grid — original merge (46 MB)",
    ),
    MagSource(
        key="nrcan_ca_1km_rtf",
        name="NRCan Canada 1km MAG Residual Total Field",
        urls=[
            "https://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=129",
            "http://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=129",
        ],
        source_type="arcgrid_zip",
        description="Canada-wide aeromagnetic residual total field (1 km) via NRCan DAP landing endpoint",
    ),
    MagSource(
        key="nrcan_ca_200m_rtf",
        name="NRCan Canada 200m MAG Residual Total Field",
        urls=[
            "https://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=140",
            "http://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=140",
        ],
        source_type="arcgrid_zip",
        description="Canada-wide aeromagnetic residual total field (200 m) via NRCan DAP landing endpoint",
    ),
    MagSource(
        key="nrcan_ca_1km_vd",
        name="NRCan Canada 1km MAG 1st Vertical Derivative",
        urls=[
            "https://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=130",
            "http://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=130",
        ],
        source_type="arcgrid_zip",
        description="Canada-wide first vertical derivative grid (1 km) via NRCan DAP landing endpoint",
    ),
    MagSource(
        key="nrcan_ca_200m_vd",
        name="NRCan Canada 200m MAG 1st Vertical Derivative",
        urls=[
            "https://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=138",
            "http://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=138",
        ],
        source_type="arcgrid_zip",
        description="Canada-wide first vertical derivative grid (200 m) via NRCan DAP landing endpoint",
    ),
    MagSource(
        key="wdmam",
        name="WDMAM v2 (World Digital Magnetic Anomaly Map)",
        urls=[
            "https://wdmam.org/WDMAM2_v2_XYZ.zip",
            "https://wdmam.org/download/WDMAM2_xyz.zip",
        ],
        source_type="csv_zip",
        description="Global 3-arc-minute satellite-derived total-field anomaly map — BROKEN: site now serves HTML SPA, not direct downloads (Mar 2026)",
    ),
    MagSource(
        key="ncei_extract",
        name="NOAA NCEI Grid Extract API",
        urls=[],
        source_type="api",
        description="NOAA NCEI on-demand grid extraction — can produce EMAG2 subsets for bounding box",
    ),
    MagSource(
        key="swarm",
        name="ESA Swarm (VirES API)",
        urls=[],
        source_type="api",
        description="ESA Swarm satellite magnetic field data via VirES for Swarm API (free registration)",
    ),
]

SOURCE_MAP = {s.key: s for s in SOURCES}


def _expand_source_keys(source_keys: list[str]) -> list[str]:
    """Expand aliases and keep source order stable while de-duplicating keys."""
    expanded: list[str] = []
    for key in source_keys:
        k = key.strip()
        if not k:
            continue
        if k in ("nrcan", "nrcan_aeromag", "canada_aeromag"):
            expanded.extend(NRCAN_SOURCE_KEYS)
        else:
            expanded.append(k)

    out: list[str] = []
    seen = set()
    for key in expanded:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _filename_from_url(url: str, fallback_key: str, expected_zip: bool = False) -> str:
    """Generate a Windows-safe filename from URL path + query parameters."""
    parsed = urlparse(url)
    base = Path(parsed.path).name or fallback_key
    if parsed.query:
        tokens = [f"{k}-{v}" for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
        if tokens:
            base = f"{base}_{'_'.join(tokens)}"

    # Replace characters that are invalid on Windows filesystems.
    for bad in '<>:"/\\|?*':
        base = base.replace(bad, "_")
    if expected_zip and not base.lower().endswith(".zip"):
        base = f"{base}.zip"
    return base


def _is_zip_file(path: Path) -> bool:
    """Quick signature check to avoid treating HTML landing pages as ZIP payloads."""
    try:
        with path.open("rb") as fh:
            return fh.read(4).startswith(b"PK")
    except OSError:
        return False


def _list_local_mage_csvs(min_size_bytes: int = 1024) -> list[Path]:
    """Return locally staged MAGE CSV files that look non-trivial in size."""
    if not LOCAL_MAGE_CSV_DIR.exists():
        return []
    return sorted(
        p for p in LOCAL_MAGE_CSV_DIR.glob("*.csv")
        if p.is_file() and p.stat().st_size > min_size_bytes
    )


def _safe_key(name: str) -> str:
    """Build filesystem-safe keys for derived output names."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _bbox_token(bbox: tuple[float, float, float, float]) -> str:
    """Return a stable token for bbox-specific cache artifacts."""
    return _safe_key("_".join(f"{v:.4f}" for v in bbox))


# ── Progress tracking ────────────────────────────────────────────────────────
@dataclass
class StageProgress:
    stage: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    message: str = ""
    started: float = 0.0
    ended: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    stages: list[StageProgress] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    grids_produced: list[str] = field(default_factory=list)
    patches_produced: int = 0
    detections: int = 0
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "stages": [asdict(s) for s in self.stages],
            "candidates_count": len(self.candidates),
            "grids_produced": self.grids_produced,
            "patches_produced": self.patches_produced,
            "detections": self.detections,
        }


# ── Stage 1: Download (with persistent cache) ───────────────────────────────
def _download_with_resume(url: str, dest: Path, progress_callback=None) -> bool:
    """Download a file with resume support and progress tracking."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    # Resume from partial download if it exists
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        log.info("Resuming download from %d bytes: %s", existing_bytes, url)

    try:
        r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers)
        if r.status_code == 416:
            # Range not satisfiable — file is complete
            if partial.exists():
                partial.rename(dest)
            return True
        r.raise_for_status()

        total_size = int(r.headers.get("Content-Length", 0))
        if r.status_code == 200:
            # Server doesn't support range — start fresh
            existing_bytes = 0
            mode = "wb"
        else:
            mode = "ab"

        downloaded = existing_bytes
        with open(partial, mode) as fh:
            for chunk in r.iter_content(DOWNLOAD_CHUNK):
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size:
                    pct = downloaded * 100 // (total_size + existing_bytes)
                    progress_callback(f"Downloading... {downloaded // (1024*1024)} MB ({pct}%)")

        partial.rename(dest)
        log.info("Downloaded %s (%d MB)", dest.name, downloaded // (1024 * 1024))
        return True

    except Exception as e:
        log.warning("Download failed %s: %s", url, e)
        return False


def stage_download(
    data_dir: Path,
    source_keys: list[str],
    progress_callback=None,
) -> StageProgress:
    """Download magnetic datasets — checks persistent cache first."""
    sp = StageProgress(stage="download", status="running", started=time.time())
    downloaded = {}
    cache_dir = MAG_DATA_RAW  # persistent cache

    source_keys = _expand_source_keys(source_keys)

    for key in source_keys:
        src = SOURCE_MAP.get(key)
        if not src:
            log.warning("Unknown source: %s", key)
            continue
        if src.source_type == "api":
            log.info("Skipping API-based source %s (handled in dedicated stage)", key)
            continue

        # Check persistent cache first
        cached_dir = cache_dir / key
        if cached_dir.exists():
            cached_files = [f for f in cached_dir.iterdir() if f.is_file() and f.stat().st_size > 1_000 and not f.suffix.endswith(".partial")]
            if cached_files:
                log.info("Using cached data for %s: %s", key, cached_files[0])
                downloaded[key] = str(cached_files[0])
                if progress_callback:
                    progress_callback(f"{src.name} — cached")
                continue

        # Also check the per-run data dir (legacy location)
        legacy_dir = data_dir / "raw" / key
        if legacy_dir.exists():
            legacy_files = [f for f in legacy_dir.iterdir() if f.is_file() and f.stat().st_size > 1_000]
            if legacy_files:
                # Migrate to persistent cache
                cached_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                dest = cached_dir / legacy_files[0].name
                shutil.copy2(str(legacy_files[0]), str(dest))
                log.info("Migrated cached data for %s to persistent lake: %s", key, dest)
                downloaded[key] = str(dest)
                continue

        # Download needed
        success = False
        for url in src.urls:
            fname = _filename_from_url(
                url,
                fallback_key=key,
                expected_zip=(src.source_type in ("arcgrid_zip", "csv_zip")),
            )
            dest = cached_dir / fname
            if progress_callback:
                progress_callback(f"Downloading {src.name}...")

            if _download_with_resume(url, dest, progress_callback):
                if src.source_type in ("arcgrid_zip", "csv_zip") and not _is_zip_file(dest):
                    log.warning("Downloaded payload is not a ZIP archive for %s: %s", key, url)
                    try:
                        dest.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue

                downloaded[key] = str(dest)
                # Write metadata
                meta_dir = MAG_DATA_META
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta = {
                    "source": key,
                    "url": url,
                    "file": str(dest),
                    "size_bytes": dest.stat().st_size,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(meta_dir / f"{key}.json", "w") as f:
                    json.dump(meta, f, indent=2)

                success = True
                break

        if not success:
            log.error("Could not download %s from any URL", key)

    sp.status = "completed" if downloaded else "failed"
    sp.ended = time.time()
    sp.details = {"downloaded": downloaded}
    sp.message = f"Downloaded {len(downloaded)}/{len(source_keys)} datasets"
    return sp


# ── Stage 2: Ingest & grid to GeoTIFF ───────────────────────────────────────
def _grid_csv_to_geotiff(csv_path: Path, out_tif: Path, bbox, res_deg=DEFAULT_RES_DEG):
    """Grid a CSV of lon,lat,anomaly points into a GeoTIFF within bbox."""
    import numpy as np
    from scipy.interpolate import griddata as scipy_griddata
    import rasterio
    from rasterio.transform import from_origin

    lonmin, latmin, lonmax, latmax = bbox
    lons, lats, vals = [], [], []

    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        first = next(reader)
        header = [c.lower().strip() for c in first]

        # Detect columns
        if any("lon" in h for h in header):
            i_lon = next(i for i, k in enumerate(header) if "lon" in k)
            i_lat = next(i for i, k in enumerate(header) if "lat" in k)
            i_val = next(
                (i for i, k in enumerate(header) if any(t in k for t in ("anom", "value", "mag", "tmi"))),
                2,
            )
        else:
            i_lon, i_lat, i_val = 0, 1, 2
            # First row might be data
            try:
                lon, lat, v = float(first[0]), float(first[1]), float(first[2])
                if lonmin <= lon <= lonmax and latmin <= lat <= latmax:
                    lons.append(lon); lats.append(lat); vals.append(v)
            except (ValueError, IndexError):
                pass

        for row in reader:
            try:
                lon, lat, v = float(row[i_lon]), float(row[i_lat]), float(row[i_val])
            except (ValueError, IndexError):
                continue
            if lonmin <= lon <= lonmax and latmin <= lat <= latmax:
                lons.append(lon); lats.append(lat); vals.append(v)

    if not vals:
        raise RuntimeError(f"No points found in {csv_path} within bbox {bbox}")

    pts = np.vstack([lons, lats]).T
    xi = np.arange(lonmin, lonmax + res_deg, res_deg)
    yi = np.arange(latmax, latmin - res_deg, -res_deg)
    grid_x, grid_y = np.meshgrid(xi, yi)
    log.info("Gridding %d points to %s", len(vals), grid_x.shape)

    grid_z = scipy_griddata(pts, np.array(vals), (grid_x, grid_y), method="linear")
    # Keep out-of-support cells as NaN so sparse surveys do not get extrapolated
    # across the full bbox via nearest-neighbor filling.

    transform = from_origin(xi[0], yi[0], res_deg, res_deg)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_tif), "w", driver="GTiff",
        height=grid_z.shape[0], width=grid_z.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(grid_z.astype("float32"), 1)
    log.info("Wrote GeoTIFF: %s", out_tif)
    return out_tif


def _convert_arcgrid_to_geotiff(raw_path: Path, out_tif: Path, bbox):
    """Convert an ArcGrid zip to a GeoTIFF subset in EPSG:4326."""
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as warp_transform
    from rasterio.windows import from_bounds

    extract_dir = raw_path.parent / (raw_path.stem + "_extracted")
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(raw_path), "r") as z:
            z.extractall(str(extract_dir))

    # Find the raster dataset (could be hdr.adf or similar)
    candidates = list(extract_dir.rglob("hdr.adf")) + list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.adf"))
    if not candidates:
        raise RuntimeError(f"No raster files found in {extract_dir}")

    src_path = candidates[0]
    lonmin, latmin, lonmax, latmax = bbox
    dst_crs = "EPSG:4326"

    with rasterio.open(str(src_path)) as src:
        # If already in 4326, just window-read
        if src.crs and src.crs.to_epsg() == 4326:
            left, bottom, right, top = lonmin, latmin, lonmax, latmax
            left = max(left, src.bounds.left)
            right = min(right, src.bounds.right)
            bottom = max(bottom, src.bounds.bottom)
            top = min(top, src.bounds.top)
            if left >= right or bottom >= top:
                raise RuntimeError("Bbox does not intersect source grid")
            window = from_bounds(left, bottom, right, top, src.transform)
            arr = src.read(1, window=window).astype("float32")
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                "transform": src.window_transform(window), "count": 1, "dtype": "float32",
                "crs": dst_crs,
            })
            out_tif.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(str(out_tif), "w", **meta) as dst:
                dst.write(arr, 1)
        else:
            # Reproject from source CRS to EPSG:4326, clipped to bbox
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height,
                *src.bounds,
            )
            # Clip the output to bbox
            from rasterio.transform import from_bounds as tf_from_bounds
            res_x = (lonmax - lonmin) / max(width, 1)
            res_y = (latmax - latmin) / max(height, 1)
            # Use source resolution in degrees (approximate)
            out_width = int((lonmax - lonmin) / abs(transform[0]))
            out_height = int((latmax - latmin) / abs(transform[4]))
            out_width = max(out_width, 1)
            out_height = max(out_height, 1)
            out_transform = tf_from_bounds(lonmin, latmin, lonmax, latmax, out_width, out_height)

            arr = np.empty((out_height, out_width), dtype="float32")
            out_tif.parent.mkdir(parents=True, exist_ok=True)

            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

            # Replace nodata with NaN
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

            meta = {
                "driver": "GTiff", "height": out_height, "width": out_width,
                "count": 1, "dtype": "float32", "crs": dst_crs,
                "transform": out_transform,
            }
            with rasterio.open(str(out_tif), "w", **meta) as dst:
                dst.write(arr, 1)

    log.info("Wrote GeoTIFF (EPSG:4326): %s (%dx%d)", out_tif, out_width if 'out_width' in dir() else '?', out_height if 'out_height' in dir() else '?')
    return out_tif


def stage_ingest(
    data_dir: Path,
    grids_dir: Path,
    downloaded: dict,
    bbox=DEFAULT_BBOX,
    progress_callback=None,
) -> StageProgress:
    """Ingest raw downloads into region-clipped GeoTIFFs."""
    sp = StageProgress(stage="ingest", status="running", started=time.time())
    produced = []

    bbox_tag = _bbox_token(tuple(bbox))

    for key, raw_path_str in downloaded.items():
        raw_path = Path(raw_path_str)
        out_tif = grids_dir / f"{key}_{bbox_tag}.tif"
        if out_tif.exists() and out_tif.stat().st_size > 1_000:
            log.info("Grid already exists: %s", out_tif)
            produced.append(str(out_tif))
            continue

        try:
            if progress_callback:
                progress_callback(f"Gridding {key}...")
            src = SOURCE_MAP.get(key)
            if src and src.source_type == "csv_zip":
                # Extract CSV from zip
                extract_dir = raw_path.parent / (raw_path.stem + "_extracted")
                if not extract_dir.exists():
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(str(raw_path), "r") as z:
                        z.extractall(str(extract_dir))
                csv_files = list(extract_dir.rglob("*.csv"))
                if not csv_files:
                    raise RuntimeError(f"No CSV found in {raw_path}")
                _grid_csv_to_geotiff(csv_files[0], out_tif, bbox)
            elif src and src.source_type == "arcgrid_zip":
                _convert_arcgrid_to_geotiff(raw_path, out_tif, bbox)
            elif raw_path.suffix.lower() in (".tif", ".tiff"):
                # Already a GeoTIFF — just subset
                import rasterio
                from rasterio.windows import from_bounds
                lonmin, latmin, lonmax, latmax = bbox
                with rasterio.open(str(raw_path)) as s:
                    window = from_bounds(lonmin, latmin, lonmax, latmax, s.transform)
                    arr = s.read(1, window=window)
                    meta = s.meta.copy()
                    meta.update({"height": arr.shape[0], "width": arr.shape[1], "transform": s.window_transform(window)})
                    with rasterio.open(str(out_tif), "w", **meta) as d:
                        d.write(arr.astype("float32"), 1)
            else:
                log.warning("Unknown format for %s: %s", key, raw_path)
                continue
            produced.append(str(out_tif))
        except Exception as e:
            log.error("Failed to ingest %s: %s", key, e)
            sp.details.setdefault("errors", []).append({"source": key, "error": str(e)})

    # Also ingest locally staged MAGE CSVs copied into the persistent data lake.
    local_csvs = _list_local_mage_csvs()
    local_ingested = 0
    for csv_path in local_csvs:
        local_key = _safe_key(csv_path.stem)
        out_tif = grids_dir / f"local_{local_key}_{bbox_tag}.tif"
        if out_tif.exists() and out_tif.stat().st_size > 1_000:
            produced.append(str(out_tif))
            continue

        try:
            if progress_callback:
                progress_callback(f"Gridding local CSV {csv_path.name}...")
            _grid_csv_to_geotiff(csv_path, out_tif, bbox)
            produced.append(str(out_tif))
            local_ingested += 1
        except Exception as e:
            log.warning("Skipped local CSV %s: %s", csv_path.name, e)
            sp.details.setdefault("errors", []).append({"source": csv_path.name, "error": str(e)})

    sp.status = "completed" if produced else "failed"
    sp.ended = time.time()
    sp.details["grids_produced"] = produced
    sp.details["local_csv_found"] = len(local_csvs)
    sp.details["local_csv_ingested"] = local_ingested
    sp.message = f"Produced {len(produced)} GeoTIFF grids"
    return sp


# ── Stage 3: Tile into patches ──────────────────────────────────────────────
def stage_tile(
    grids_dir: Path,
    patches_dir: Path,
    patch_px: int = DEFAULT_PATCH_PX,
    stride_px: int = None,
    tif_files: list[Path] = None,
    progress_callback=None,
) -> StageProgress:
    """Tile GeoTIFFs into numpy patches for model inference."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    stride_px = stride_px or patch_px
    sp = StageProgress(stage="tile", status="running", started=time.time())
    total_patches = 0

    tif_files = sorted(tif_files) if tif_files else sorted(grids_dir.glob("*.tif"))
    if not tif_files:
        sp.status = "skipped"
        sp.message = "No GeoTIFF files found to tile"
        sp.ended = time.time()
        return sp

    for tif_path in tif_files:
        key = tif_path.stem
        out_dir = patches_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(f"Tiling {tif_path.name}...")

        with rasterio.open(str(tif_path)) as src:
            h, w = src.height, src.width
            count = 0
            y = 0
            while y + patch_px <= h:
                x = 0
                while x + patch_px <= w:
                    win = Window(x, y, patch_px, patch_px)
                    arr = src.read(1, window=win).astype("float32")
                    # Skip mostly-nodata patches
                    valid_frac = np.count_nonzero(~np.isnan(arr)) / arr.size
                    if valid_frac < 0.3:
                        x += stride_px
                        continue
                    np.save(out_dir / f"{key}_{y}_{x}.npy", arr)
                    count += 1
                    x += stride_px
                y += stride_px
            log.info("Tiled %s: %d patches", tif_path.name, count)
            total_patches += count

    sp.status = "completed"
    sp.ended = time.time()
    sp.details["total_patches"] = total_patches
    sp.message = f"Produced {total_patches} patches from {len(tif_files)} grids"
    return sp


# ── Stage 4: Anomaly detection ──────────────────────────────────────────────
def stage_detect(
    patches_dir: Path,
    models_dir: Path,
    output_dir: Path,
    grids_dir: Path,
    bbox=DEFAULT_BBOX,
    threshold: float = 0.5,
    progress_callback=None,
) -> StageProgress:
    """Run trained models on patches to detect magnetic anomalies.

    Works in two complementary ways:
      A) If trained UNet / ML models exist in models_dir, run inference per-patch
      B) Statistical anomaly detection (gradient, RMS, peak-to-peak) as fallback
    """
    import numpy as np

    sp = StageProgress(stage="detect", status="running", started=time.time())
    detections = []

    # Gather all patches
    patch_files = sorted(patches_dir.rglob("*.npy"))
    if not patch_files:
        sp.status = "skipped"
        sp.message = "No patches to analyze"
        sp.ended = time.time()
        return sp

    if progress_callback:
        progress_callback(f"Analyzing {len(patch_files)} patches...")

    # B) Statistical anomaly detection (always runs — no model deps required)
    # First pass: compute raw metrics for all patches to enable adaptive scoring
    patch_metrics = []
    for pf in patch_files:
        arr = np.load(pf).astype("float64")  # use float64 to avoid overflow
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.std() < 1e-6:
            patch_metrics.append(None)
            continue

        patch_h, patch_w = arr.shape[:2]

        grad_y, grad_x = np.gradient(arr)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        patch_metrics.append({
            "rms": float(np.sqrt(np.mean(arr**2))),
            "peak_to_peak": float(arr.max() - arr.min()),
            "max_gradient": float(grad_mag.max()),
            "mean_gradient": float(grad_mag.mean()),
            "patch_h": int(patch_h),
            "patch_w": int(patch_w),
        })

    # Compute adaptive normalization from data distribution (mean + 2*std)
    valid_metrics = [m for m in patch_metrics if m is not None]
    if valid_metrics:
        mg_vals = np.array([m["max_gradient"] for m in valid_metrics])
        p2p_vals = np.array([m["peak_to_peak"] for m in valid_metrics])
        rms_vals = np.array([m["rms"] for m in valid_metrics])
        meangrad_vals = np.array([m["mean_gradient"] for m in valid_metrics])

        # Normalizers: use p95 of the dataset so only the top outliers score high
        norm_mg = float(np.percentile(mg_vals, 95)) or 1.0
        norm_p2p = float(np.percentile(p2p_vals, 95)) or 1.0
        norm_rms = float(np.percentile(rms_vals, 95)) or 1.0
        norm_meangrad = float(np.percentile(meangrad_vals, 95)) or 1.0
    else:
        norm_mg = norm_p2p = norm_rms = norm_meangrad = 1.0

    log.info("Adaptive normalizers: max_grad=%.1f  p2p=%.1f  rms=%.1f  mean_grad=%.1f",
             norm_mg, norm_p2p, norm_rms, norm_meangrad)

    # Second pass: score each patch using adaptive normalization
    for pf, metrics in zip(patch_files, patch_metrics):
        if metrics is None:
            continue

        rms = metrics["rms"]
        peak_to_peak = metrics["peak_to_peak"]
        max_gradient = metrics["max_gradient"]
        mean_gradient = metrics["mean_gradient"]

        # Anomaly score: normalized relative to dataset distribution
        # Patches with metrics above the 95th percentile score > 1.0 before clamping
        score = (
            0.35 * min(max_gradient / norm_mg, 1.0) +
            0.25 * min(peak_to_peak / norm_p2p, 1.0) +
            0.20 * min(rms / norm_rms, 1.0) +
            0.20 * min(mean_gradient / norm_meangrad, 1.0)
        )

        if score >= threshold:
            # Extract approximate location from patch filename
            parts = pf.stem.split("_")
            try:
                row_idx = int(parts[-2])
                col_idx = int(parts[-1])
            except (ValueError, IndexError):
                row_idx, col_idx = 0, 0

            # Try to get geo-coordinates from the parent grid
            grid_key = pf.parent.name
            tif_path = grids_dir / f"{grid_key}.tif"
            lat, lon = None, None
            if tif_path.exists():
                try:
                    import rasterio
                    with rasterio.open(str(tif_path)) as src:
                        center_row = row_idx + int(metrics.get("patch_h", DEFAULT_PATCH_PX)) // 2
                        center_col = col_idx + int(metrics.get("patch_w", DEFAULT_PATCH_PX)) // 2
                        lon, lat = src.xy(center_row, center_col)
                except Exception:
                    pass

            detections.append({
                "patch_file": str(pf),
                "source_grid": grid_key,
                "anomaly_score": round(score, 4),
                "rms": round(rms, 2),
                "peak_to_peak": round(peak_to_peak, 2),
                "max_gradient": round(max_gradient, 2),
                "mean_gradient": round(mean_gradient, 4),
                "lat": lat,
                "lon": lon,
                "method": "statistical",
            })

    # Sort by score descending
    detections.sort(key=lambda d: d["anomaly_score"], reverse=True)

    # A) Model-based inference (if models exist)
    model_detections = []
    model_files = list(Path(models_dir).glob("*.pt")) + list(Path(models_dir).glob("*.pth"))
    if model_files:
        try:
            import torch
            for model_path in model_files:
                if progress_callback:
                    progress_callback(f"Running model {model_path.name}...")
                try:
                    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
                    if hasattr(model, "eval"):
                        model.eval()
                    # Run inference on top statistical candidates
                    for det in detections[:100]:
                        patch = np.load(det["patch_file"]).astype("float32")
                        patch = np.nan_to_num(patch, nan=0.0)
                        patch = (patch - patch.mean()) / (patch.std() + 1e-9)
                        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            out = model(tensor)
                            if isinstance(out, tuple):
                                out = out[0]
                            prob = torch.sigmoid(out).mean().item()
                        model_detections.append({
                            **det,
                            "model": model_path.name,
                            "model_prob": round(prob, 4),
                            "method": "model+statistical",
                        })
                except Exception as e:
                    log.warning("Model %s failed: %s", model_path.name, e)
        except ImportError:
            log.info("PyTorch not available — using statistical detection only")

    # Merge model detections with statistical
    if model_detections:
        # Re-score with model probabilities
        for md in model_detections:
            md["anomaly_score"] = round(
                0.6 * md.get("model_prob", 0) + 0.4 * md["anomaly_score"], 4
            )
        model_detections.sort(key=lambda d: d["anomaly_score"], reverse=True)
        final_detections = model_detections
    else:
        final_detections = detections

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)
    det_json = output_dir / "detections.json"
    with open(det_json, "w") as f:
        json.dump(final_detections, f, indent=2)

    det_csv = output_dir / "detections.csv"
    if final_detections:
        with open(det_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_detections[0].keys())
            writer.writeheader()
            writer.writerows(final_detections)

    sp.status = "completed"
    sp.ended = time.time()
    sp.details = {
        "total_patches_analyzed": len(patch_files),
        "statistical_detections": len(detections),
        "model_detections": len(model_detections),
        "final_detections": len(final_detections),
        "detections_json": str(det_json),
        "detections_csv": str(det_csv),
        "models_used": [m.name for m in model_files],
    }
    sp.message = f"{len(final_detections)} anomaly detections from {len(patch_files)} patches"
    return sp


# ── Stage 5: Cross-reference with wrecks DB ─────────────────────────────────
def stage_cross_reference(
    detections_json: Path,
    db_path: Path,
    search_radius_km: float = 5.0,
    progress_callback=None,
) -> StageProgress:
    """Match anomaly detections to known wrecks within search radius."""
    sp = StageProgress(stage="cross_reference", status="running", started=time.time())

    if not detections_json.exists():
        sp.status = "skipped"
        sp.message = "No detections file found"
        sp.ended = time.time()
        return sp

    with open(detections_json) as f:
        detections = json.load(f)

    # Filter to detections with coordinates
    geo_dets = [d for d in detections if d.get("lat") is not None and d.get("lon") is not None]
    if not geo_dets:
        sp.status = "completed"
        sp.message = "No geo-located detections to cross-reference"
        sp.ended = time.time()
        return sp

    if progress_callback:
        progress_callback(f"Cross-referencing {len(geo_dets)} detections against wrecks DB...")

    from math import radians, cos, sin, asin, sqrt

    def haversine_km(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371 * 2 * asin(sqrt(a))

    # Load wrecks with coordinates — table is 'features' in wrecks.db
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    # Detect table name (features vs wrecks)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    tbl = "features" if "features" in tables else "wrecks"
    rows = conn.execute(
        f"SELECT id, name, latitude, longitude, feature_type, hull_material, "
        f"magnetic_potential, salvage_status FROM {tbl} "
        f"WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    conn.close()

    # Disposition values that mean the wreck is no longer on the bottom
    RAISED_DISPOSITIONS = {"raised_scrapped", "raised", "salvaged", "removed", "refloated"}

    wrecks = [dict(r) for r in rows]
    matches = []
    disposition_filtered = 0

    for det in geo_dets:
        det_lat, det_lon = det["lat"], det["lon"]
        nearby = []
        for w in wrecks:
            dist = haversine_km(det_lat, det_lon, w["latitude"], w["longitude"])
            if dist <= search_radius_km:
                w_entry = {**w, "distance_km": round(dist, 3)}
                # Flag wrecks with raised/scrapped disposition
                salvage = (w.get("salvage_status") or "").lower().strip()
                mag_pot = (w.get("magnetic_potential") or "").lower().strip()
                if salvage in RAISED_DISPOSITIONS or mag_pot == "geological_false_positive":
                    w_entry["disposition_flag"] = "raised_or_false_positive"
                nearby.append(w_entry)
        if nearby:
            nearby.sort(key=lambda w: w["distance_km"])
            closest = nearby[0]
            match_entry = {
                "detection": det,
                "nearby_wrecks": nearby,
                "closest_wreck": closest["name"],
                "closest_distance_km": closest["distance_km"],
            }
            # If the closest match is a raised/scrapped wreck, flag as geological
            if closest.get("disposition_flag"):
                match_entry["likely_false_positive"] = True
                match_entry["false_positive_reason"] = (
                    f"{closest['name']} was {closest.get('salvage_status', 'removed')} — "
                    f"any magnetic signature is likely geological"
                )
                disposition_filtered += 1
            matches.append(match_entry)

    # Also find "unmatched" high-score detections = potential NEW wreck candidates
    matched_patches = {m["detection"]["patch_file"] for m in matches}
    new_candidates = [
        d for d in geo_dets
        if d["patch_file"] not in matched_patches and d["anomaly_score"] >= 0.6
    ]

    sp.status = "completed"
    sp.ended = time.time()
    sp.details = {
        "matched": len(matches),
        "disposition_filtered": disposition_filtered,
        "unmatched_high_score": len(new_candidates),
        "total_wrecks_in_db": len(wrecks),
        "matches": matches[:50],  # Top 50 for serialization
        "new_candidates": new_candidates[:50],
    }
    sp.message = (
        f"{len(matches)} matched to known wrecks ({disposition_filtered} flagged as likely false positive), "
        f"{len(new_candidates)} potential new candidates"
    )
    return sp


# ── Stage 6: Export ──────────────────────────────────────────────────────────
def stage_export(
    output_dir: Path,
    result: PipelineResult,
    progress_callback=None,
) -> StageProgress:
    """Write final pipeline results to JSON and CSV."""
    sp = StageProgress(stage="export", status="running", started=time.time())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    sp.status = "completed"
    sp.ended = time.time()
    sp.details = {"summary": str(summary_path)}
    sp.message = f"Exported pipeline summary to {summary_path}"
    return sp


# ── Stage 7: Refine (optional — feeds detections into training) ──────────────
def stage_refine(
    detections_dir: Path,
    patches_dir: Path,
    models_dir: Path,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-4,
    progress_callback=None,
) -> StageProgress:
    """Fine-tune models using confirmed detections as new positive samples.

    Looks for a ``confirmed_detections.json`` file in detections_dir containing
    detection objects with a ``"label"`` field (1=wreck, 0=not-wreck).  These are
    converted to image/label numpy pairs and added to the training dataset.
    """
    sp = StageProgress(stage="refine", status="running", started=time.time())

    confirmed_path = detections_dir / "confirmed_detections.json"
    if not confirmed_path.exists():
        sp.status = "skipped"
        sp.message = "No confirmed_detections.json found — run detection first and label results"
        sp.ended = time.time()
        return sp

    with open(confirmed_path) as f:
        confirmed = json.load(f)

    if not confirmed:
        sp.status = "skipped"
        sp.message = "confirmed_detections.json is empty"
        sp.ended = time.time()
        return sp

    import numpy as np

    # Prepare training patches from confirmed detections
    train_dir = output_dir / "refine_training"
    train_dir.mkdir(parents=True, exist_ok=True)
    n_prepared = 0

    for det in confirmed:
        patch_file = det.get("patch_file")
        label = det.get("label")  # 1 = wreck, 0 = not-wreck
        if patch_file is None or label is None:
            continue
        pf = Path(patch_file)
        if not pf.exists():
            continue

        arr = np.load(pf).astype("float32")
        arr = np.nan_to_num(arr, nan=0.0)
        # Normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-9)

        # Save as image/label pair compatible with training scripts
        stem = pf.stem
        np.save(train_dir / f"{stem}_image.npy", arr[np.newaxis, :, :] if arr.ndim == 2 else arr)
        # Label mask: all-ones for wreck, all-zeros for background
        label_mask = np.full(arr.shape[-2:], int(label), dtype=np.int64)
        np.save(train_dir / f"{stem}_label.npy", label_mask)
        n_prepared += 1

    if n_prepared == 0:
        sp.status = "skipped"
        sp.message = "No valid confirmed detections with patch files found"
        sp.ended = time.time()
        return sp

    if progress_callback:
        progress_callback(f"Prepared {n_prepared} training samples, starting fine-tuning...")

    # Attempt fine-tuning with PyTorch
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader

        class _PatchDS(Dataset):
            def __init__(self, d):
                self.files = [(f, d / (f.stem.replace("_image", "_label") + ".npy"))
                              for f in d.glob("*_image.npy")
                              if (d / (f.stem.replace("_image", "_label") + ".npy")).exists()]
            def __len__(self):
                return len(self.files)
            def __getitem__(self, idx):
                img = np.load(self.files[idx][0]).astype("float32")
                lab = np.load(self.files[idx][1]).astype("int64")
                if img.ndim == 2:
                    img = img[np.newaxis]
                return torch.from_numpy(img), torch.from_numpy(lab)

        ds = _PatchDS(train_dir)
        if len(ds) == 0:
            sp.status = "skipped"
            sp.message = "No training pairs produced"
            sp.ended = time.time()
            return sp

        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Find best existing model to fine-tune
        model_files = sorted(Path(models_dir).glob("*.pt")) + sorted(Path(models_dir).glob("*.pth"))
        if not model_files:
            sp.status = "skipped"
            sp.message = "No base models found for fine-tuning"
            sp.ended = time.time()
            return sp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model_path = model_files[0]
        model = torch.load(str(base_model_path), map_location=device, weights_only=False)
        if hasattr(model, "train"):
            model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, labs in dl:
                imgs, labs = imgs.to(device), labs.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                if isinstance(out, tuple):
                    out = out[0]
                loss = loss_fn(out, labs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if progress_callback:
                progress_callback(f"Refine epoch {epoch+1}/{epochs}, loss={epoch_loss/len(dl):.4f}")

        # Save refined model
        refined_path = Path(models_dir) / f"refined_{base_model_path.stem}.pt"
        torch.save(model, str(refined_path))

        sp.status = "completed"
        sp.ended = time.time()
        sp.details = {
            "samples": n_prepared,
            "base_model": base_model_path.name,
            "refined_model": str(refined_path),
            "epochs": epochs,
        }
        sp.message = f"Fine-tuned {base_model_path.name} with {n_prepared} confirmed samples → {refined_path.name}"

    except ImportError:
        sp.status = "completed"
        sp.ended = time.time()
        sp.details = {"samples_prepared": n_prepared, "training_dir": str(train_dir)}
        sp.message = f"Prepared {n_prepared} training samples (PyTorch not available for fine-tuning)"

    return sp


# ── Master orchestrator ─────────────────────────────────────────────────────
def run_pipeline(
    output_dir: str = "mag_pipeline_output",
    source_keys: list[str] = None,
    bbox: tuple = DEFAULT_BBOX,
    models_dir: str = "bagfilework/training/models",
    db_path: str = "db/wrecks.db",
    stages: str = "all",
    threshold: float = 0.3,
    patch_px: int = DEFAULT_PATCH_PX,
    progress_callback=None,
) -> dict:
    """Run the full magnetic anomaly data pipeline.

    Args:
        output_dir: Where to write all pipeline output
        source_keys: Which data sources to download (default: all free ones)
        bbox: (lonmin, latmin, lonmax, latmax) bounding box
        models_dir: Path to trained model weights
        db_path: Path to wrecks SQLite database
        stages: Comma-separated list or "all"
        threshold: Anomaly score threshold for detection
        progress_callback: Optional callable(message: str) for live updates
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if source_keys is None:
        source_keys = ["usgs_namag", "usgs_usmag"]
    else:
        source_keys = _expand_source_keys(source_keys)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    grids_dir = MAG_DATA_GRIDS  # persistent cache for grids
    patches_dir = MAG_DATA_PATCHES / _bbox_token(tuple(bbox))  # bbox-scoped patch cache
    detections_dir = output / "detections"

    requested = set(stages.split(",")) if stages != "all" else {"download", "ingest", "tile", "detect", "xref", "export"}
    # "refine" is opt-in — only runs if explicitly requested (stages="all,refine" or "refine")
    result = PipelineResult(status="running")

    try:
        run_grid_paths = []

        # Stage 1: Download
        if "download" in requested:
            sp = stage_download(data_dir, source_keys, progress_callback)
            result.stages.append(sp)
            downloaded = sp.details.get("downloaded", {})
        else:
            downloaded = {}
            # Check for existing raw files
            raw_dir = data_dir / "raw"
            if raw_dir.exists():
                for key in source_keys:
                    key_dir = raw_dir / key
                    if key_dir.exists():
                        files = list(key_dir.glob("*"))
                        if files:
                            downloaded[key] = str(files[0])

        # Stage 2: Ingest
        local_csvs_available = bool(_list_local_mage_csvs())
        if "ingest" in requested and (downloaded or local_csvs_available):
            sp = stage_ingest(data_dir, grids_dir, downloaded, bbox, progress_callback)
            result.stages.append(sp)
            result.grids_produced = sp.details.get("grids_produced", [])
            run_grid_paths = [Path(p) for p in result.grids_produced]
        elif "ingest" not in requested:
            # Use existing grids
            result.grids_produced = [str(p) for p in grids_dir.glob("*.tif")] if grids_dir.exists() else []
            run_grid_paths = [Path(p) for p in result.grids_produced]

        # Stage 3: Tile
        if "tile" in requested and (grids_dir.exists() or result.grids_produced):
            sp = stage_tile(
                grids_dir,
                patches_dir,
                patch_px,
                tif_files=run_grid_paths if run_grid_paths else None,
                progress_callback=progress_callback,
            )
            result.stages.append(sp)
            result.patches_produced = sp.details.get("total_patches", 0)

        # Stage 4: Detect
        if "detect" in requested and patches_dir.exists():
            sp = stage_detect(
                patches_dir, Path(models_dir), detections_dir, grids_dir,
                bbox, threshold, progress_callback,
            )
            result.stages.append(sp)
            result.detections = sp.details.get("final_detections", 0)

        # Stage 5: Cross-reference
        det_json = detections_dir / "detections.json"
        if "xref" in requested and det_json.exists():
            sp = stage_cross_reference(det_json, Path(db_path), progress_callback=progress_callback)
            result.stages.append(sp)
            result.candidates = sp.details.get("new_candidates", [])

        # Stage 6: Export
        if "export" in requested:
            sp = stage_export(output, result, progress_callback)
            result.stages.append(sp)

        # Stage 7: Refine (optional — only if confirmed detections exist)
        if "refine" in requested:
            sp = stage_refine(
                detections_dir, patches_dir, Path(models_dir), output,
                progress_callback=progress_callback,
            )
            result.stages.append(sp)

        result.status = "completed"

    except Exception as e:
        log.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        result.status = "failed"
        result.stages.append(StageProgress(
            stage="error", status="failed", message=str(e), ended=time.time(),
        ))

    return result.to_dict()


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Magnetic Anomaly Data Acquisition & Detection Pipeline")
    p.add_argument("--stages", default="all", help="Comma-separated stages: download,ingest,tile,detect,xref,export or 'all'")
    p.add_argument(
        "--sources",
        default="usgs_namag,usgs_usmag",
        help=(
            "Comma-separated source keys. Aliases: "
            "nrcan|nrcan_aeromag expands to all four NRCan Canada aeromagnetic datasets"
        ),
    )
    p.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX), help="lonmin latmin lonmax latmax")
    p.add_argument("--output-dir", default="mag_pipeline_output")
    p.add_argument("--models-dir", default="bagfilework/training/models")
    p.add_argument("--db-path", default="db/wrecks.db")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--patch-px", type=int, default=DEFAULT_PATCH_PX)
    p.add_argument("--grids-dir", help="Use existing grids directory instead of downloading")
    args = p.parse_args()

    result = run_pipeline(
        output_dir=args.output_dir,
        source_keys=args.sources.split(","),
        bbox=tuple(args.bbox),
        models_dir=args.models_dir,
        db_path=args.db_path,
        stages=args.stages,
        threshold=args.threshold,
        patch_px=args.patch_px,
        progress_callback=lambda msg: print(f"  >> {msg}"),
    )
    print("\n" + "=" * 60)
    print("Pipeline Result:", result["status"])
    for s in result.get("stages", []):
        print(f"  {s['stage']:20s}  {s['status']:12s}  {s.get('message', '')}")
    print(f"  Grids: {len(result.get('grids_produced', []))}")
    print(f"  Patches: {result.get('patches_produced', 0)}")
    print(f"  Detections: {result.get('detections', 0)}")
    print(f"  Candidates: {result.get('candidates_count', 0)}")


if __name__ == "__main__":
    main()
