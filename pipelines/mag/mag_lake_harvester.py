#!/usr/bin/env python3
"""Mag-Lake Harvester — Tiered magnetic data acquisition, cataloging, and
satellite-baseline normalization for Great Lakes wreck hunting.

Builds on the existing mag_data_pipeline.py infrastructure but adds:
  - Tiered folder organization (Tier 1–4 by resolution)
  - catalog.json master index with bounding box + resolution scores
  - Repository search (NOAA NCEI ADS, USGS ScienceBase, U-Mich Deep Blue)
  - Smart Download (gap-fill + resolution-upgrade logic)
  - Header metadata extraction (sensor altitude, line spacing)
  - Retry logic + zstd compression for processed ASCII
  - Tier 4 satellite baseline normalization (Swarm/EMAG2)
  - Gap interpolation with EXPERIMENTAL_SAT_FILL flagging
  - ML confidence annotations (low-confidence wreck / high-confidence geology)

Tier Definitions
----------------
  Tier 1: Marine/bottom-tow magnetometer (<100 m line spacing)
  Tier 2: Low-altitude aeromagnetic (100 m–1 km line spacing, alt <500 m)
  Tier 3: Regional aeromagnetic compilations (1–5 km effective resolution)
  Tier 4: Satellite (Swarm, EMAG2, WDMAM — >5 km effective resolution)

Usage:
  # Initialize — scan existing data, build tiered folders + catalog
  python scripts/mag_lake_harvester.py init

  # Search remote repositories for new data
  python scripts/mag_lake_harvester.py search

  # Smart download — fetch only upgrades/gap-fills
  python scripts/mag_lake_harvester.py fetch

  # Build Tier 4 reference baseline
  python scripts/mag_lake_harvester.py baseline

  # Full pipeline
  python scripts/mag_lake_harvester.py run

  # Show catalog summary
  python scripts/mag_lake_harvester.py status
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

log = logging.getLogger("mag_lake_harvester")

# ── Constants ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
HARVEST_ROOT = REPO_ROOT / "magnetic_data"
CATALOG_PATH = HARVEST_ROOT / "catalog.json"
MASTER_INDEX_PATH = HARVEST_ROOT / "master_index.json"

# Tiered folder roots
TIER_DIRS = {
    1: HARVEST_ROOT / "tier_1_marine",
    2: HARVEST_ROOT / "tier_2_aero_lowalt",
    3: HARVEST_ROOT / "tier_3_aero_regional",
    4: HARVEST_ROOT / "tier_4_satellite",
}
TIER4_REFERENCE = HARVEST_ROOT / "tier_4_reference"

# Great Lakes bounding box (mutable so CLI can override)
_CONFIG = {"bbox": (-92.5, 41.0, -75.5, 49.0)}

def _gl_bbox():
    return _CONFIG["bbox"]

# Resolution thresholds (meters) — used to auto-classify tiers
TIER_THRESHOLDS = {
    1: (0, 100),        # 0–100 m effective resolution
    2: (100, 1_000),    # 100 m–1 km
    3: (1_000, 5_000),  # 1–5 km
    4: (5_000, 999_999),  # >5 km (satellite)
}

# Download configuration
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0  # exponential backoff seconds
DOWNLOAD_TIMEOUT = 600
DOWNLOAD_CHUNK = 1024 * 1024

# zstd compression level (1–22, 3 is default, good speed/ratio)
ZSTD_LEVEL = 3


# ===============================================================================
# SECTION 1: CATALOG SCHEMA
# ===============================================================================

@dataclass
class CatalogEntry:
    """Single file entry in catalog.json."""
    file_id: str                         # SHA-256 of relative path
    rel_path: str                        # path relative to HARVEST_ROOT
    original_source: str                 # e.g. "usgs_namag", "ncei_ads", "local"
    tier: int                            # 1–4
    bbox: list[float]                    # [lonmin, latmin, lonmax, latmax]
    resolution_m: float                  # effective resolution in meters
    resolution_score: float              # 1/resolution_m (higher=better)
    format: str                          # csv, geotiff, xyz, grd, etc.
    sensor_altitude_m: Optional[float] = None
    line_spacing_m: Optional[float] = None
    acquisition_date: Optional[str] = None
    file_size_bytes: int = 0
    compressed: bool = False             # True if .zst compressed
    compressed_path: Optional[str] = None
    sampling_rate_hz: Optional[float] = None   # sensor sampling rate
    quality_flags: list[str] = field(default_factory=list)
    ingested_at: str = ""
    checksum_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MasterIndexEntry:
    """Extended metadata for the Master Index (superset of catalog)."""
    file_id: str
    rel_path: str
    tier: int
    bbox: list[float]
    resolution_m: float
    sensor_altitude_m: Optional[float] = None
    line_spacing_m: Optional[float] = None
    line_count: Optional[int] = None
    point_count: Optional[int] = None
    data_min: Optional[float] = None
    data_max: Optional[float] = None
    data_mean: Optional[float] = None
    crs: str = "EPSG:4326"
    ml_confidence: str = "standard"      # standard | low-confidence | high-confidence
    fill_type: Optional[str] = None      # None | EXPERIMENTAL_SAT_FILL
    baseline_normalized: bool = False
    notes: str = ""


@dataclass
class RemoteDataset:
    """A dataset found by searching remote repositories."""
    source_repo: str          # "ncei_ads", "usgs_sciencebase", "umich_deepblue"
    dataset_id: str
    title: str
    bbox: list[float]
    resolution_m: float
    download_url: str
    format: str
    file_size_bytes: int = 0
    description: str = ""
    acquisition_date: str = ""
    metadata_url: str = ""
    sampling_rate_hz: Optional[float] = None
    researcher: str = ""       # researcher name (for UMich categorization)
    doi: str = ""              # DOI (for UMich categorization)


# ===============================================================================
# SECTION 2: CATALOG MANAGEMENT
# ===============================================================================

class Catalog:
    """Load, query, and persist catalog.json."""

    def __init__(self, path: Path = CATALOG_PATH):
        self.path = path
        self.entries: dict[str, CatalogEntry] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for fid, edict in raw.get("files", {}).items():
                    self.entries[fid] = CatalogEntry(**{
                        k: v for k, v in edict.items()
                        if k in CatalogEntry.__dataclass_fields__
                    })
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("Could not load catalog: %s — starting fresh", exc)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "great_lakes_bbox": list(_gl_bbox()),
            "tier_thresholds_m": {str(k): list(v) for k, v in TIER_THRESHOLDS.items()},
            "file_count": len(self.entries),
            "files": {fid: e.to_dict() for fid, e in self.entries.items()},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)
        log.info("Catalog saved: %d entries -> %s", len(self.entries), self.path)

    def add(self, entry: CatalogEntry):
        self.entries[entry.file_id] = entry

    def remove(self, file_id: str):
        self.entries.pop(file_id, None)

    def get_by_path(self, rel_path: str) -> Optional[CatalogEntry]:
        for e in self.entries.values():
            if e.rel_path == rel_path:
                return e
        return None

    def get_coverage_for_tier(self, tier: int) -> list[CatalogEntry]:
        return [e for e in self.entries.values() if e.tier == tier]

    def best_resolution_at(self, lon: float, lat: float) -> Optional[CatalogEntry]:
        """Return the highest-resolution entry whose bbox covers (lon, lat)."""
        candidates = []
        for e in self.entries.values():
            bx = e.bbox
            if bx[0] <= lon <= bx[2] and bx[1] <= lat <= bx[3]:
                candidates.append(e)
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.resolution_m)

    def find_gaps(self, bbox: list[float], max_resolution_m: float = 5000) -> list[dict]:
        """Find geographic areas within bbox not covered by any entry below
        max_resolution_m.  Returns list of gap-bboxes on a coarse 0.5° grid."""
        lonmin, latmin, lonmax, latmax = bbox
        step = 0.5
        gaps = []
        lon = lonmin
        while lon < lonmax:
            lat = latmin
            while lat < latmax:
                cell = [lon, lat, lon + step, lat + step]
                covered = any(
                    e.resolution_m <= max_resolution_m
                    and _bbox_overlaps(e.bbox, cell)
                    for e in self.entries.values()
                )
                if not covered:
                    gaps.append({"bbox": cell, "max_needed_m": max_resolution_m})
                lat += step
            lon += step
        return gaps

    def summary(self) -> dict:
        by_tier = {}
        for e in self.entries.values():
            by_tier.setdefault(e.tier, []).append(e)
        return {
            "total_files": len(self.entries),
            "tiers": {
                t: {
                    "count": len(entries),
                    "total_mb": round(sum(e.file_size_bytes for e in entries) / 1e6, 1),
                    "best_res_m": min(e.resolution_m for e in entries) if entries else None,
                }
                for t, entries in sorted(by_tier.items())
            },
        }


def _bbox_overlaps(a: list[float], b: list[float]) -> bool:
    """True if two [lonmin,latmin,lonmax,latmax] boxes overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _file_id(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode()).hexdigest()[:16]


def _file_sha256(filepath: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ===============================================================================
# SECTION 3: TIER CLASSIFICATION & INITIALIZATION
# ===============================================================================

# Known source -> tier mappings for existing data
SOURCE_TIER_MAP = {
    # Tier 4 -- satellite
    "wdmam": 4,
    "swarm": 4,
    "esa_swarm": 4,         # ESA Swarm Level 2 crustal field
    # Tier 3 -- regional compilations
    "usgs_namag": 3,
    "usgs_usmag": 3,
    "nrcan_ca_1km_rtf": 3,
    "nrcan_ca_1km_vd": 3,
    # Tier 2 -- mid-resolution aeromagnetic
    "nrcan_ca_200m_rtf": 2,
    "nrcan_ca_200m_vd": 2,
    "nrcan_gdr": 2,          # NRCan Geophysical Data Repository grids
    # Tier 1 -- high-res (marine, AUV, boat mag)
    "umich_deepblue": 1,     # UMich AUV magnetometer (AI4Shipwrecks)
    "usgs_glsc": 1,          # USGS GLSC dense boat magnetometer
    # local MAGE CSVs and bottom-tow surveys
}

# Resolution estimates for known sources (meters)
SOURCE_RESOLUTION_MAP = {
    "wdmam": 5_556,          # 3 arc-min ~ 5.6 km
    "usgs_namag": 1_000,     # 1 km grid
    "usgs_usmag": 1_000,     # 1 km grid
    "nrcan_ca_1km_rtf": 1_000,
    "nrcan_ca_1km_vd": 1_000,
    "nrcan_ca_200m_rtf": 200,
    "nrcan_ca_200m_vd": 200,
    "ncei_extract": 3_704,   # 2 arc-min ~ 3.7 km (EMAG2v3)
    "swarm": 50_000,         # ~50 km effective
    "local_mage_csv": 50,    # assumes dense marine survey
    "emag2": 3_704,
    # New high-priority sources
    "nrcan_gdr": 200,        # NRCan Geophysical Data Repository (200m grid)
    "umich_deepblue": 5,     # UMich Deep Blue AUV magnetometer (~5m)
    "esa_swarm": 40_000,     # ESA Swarm Level 2 crustal (~40 km)
    "usgs_glsc": 10,         # USGS Great Lakes Science Center (dense boat mag)
}


def classify_tier(resolution_m: float) -> int:
    """Classify a file's tier based on its effective resolution."""
    for tier, (lo, hi) in sorted(TIER_THRESHOLDS.items()):
        if lo <= resolution_m < hi:
            return tier
    return 4  # default to satellite if unknown


def estimate_resolution_from_geotiff(tif_path: Path) -> float:
    """Read GeoTIFF pixel size and return effective resolution in meters."""
    try:
        import rasterio
        with rasterio.open(str(tif_path)) as ds:
            # pixel size in degrees -> meters (rough: 1° lat ~ 111 km)
            res_deg = abs(ds.res[0])
            return res_deg * 111_000
    except Exception:
        return 5_000  # fallback


def estimate_resolution_from_csv(csv_path: Path, sample_lines: int = 500) -> float:
    """Estimate effective resolution from point spacing in a CSV."""
    import math
    lons, lats = [], []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return 5_000
            hdr = [c.lower().strip() for c in header]
            i_lon = next((i for i, h in enumerate(hdr) if "lon" in h), 0)
            i_lat = next((i for i, h in enumerate(hdr) if "lat" in h), 1)
            for i, row in enumerate(reader):
                if i >= sample_lines:
                    break
                try:
                    lons.append(float(row[i_lon]))
                    lats.append(float(row[i_lat]))
                except (ValueError, IndexError):
                    continue
    except Exception:
        return 5_000

    if len(lons) < 2:
        return 5_000

    # Median nearest-neighbor distance
    dists = []
    for i in range(min(len(lons), 200)):
        min_d = float("inf")
        for j in range(len(lons)):
            if i == j:
                continue
            dx = (lons[i] - lons[j]) * 111_000 * math.cos(math.radians(lats[i]))
            dy = (lats[i] - lats[j]) * 111_000
            d = math.sqrt(dx * dx + dy * dy)
            if 0 < d < min_d:
                min_d = d
        if min_d < float("inf"):
            dists.append(min_d)

    if not dists:
        return 5_000
    dists.sort()
    return dists[len(dists) // 2]


def estimate_bbox_from_geotiff(tif_path: Path) -> list[float]:
    try:
        import rasterio
        with rasterio.open(str(tif_path)) as ds:
            b = ds.bounds
            return [b.left, b.bottom, b.right, b.top]
    except Exception:
        return list(_gl_bbox())


def estimate_bbox_from_csv(csv_path: Path, sample_lines: int = 10_000) -> list[float]:
    lons, lats = [], []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return list(_gl_bbox())
            hdr = [c.lower().strip() for c in header]
            i_lon = next((i for i, h in enumerate(hdr) if "lon" in h), 0)
            i_lat = next((i for i, h in enumerate(hdr) if "lat" in h), 1)
            for i, row in enumerate(reader):
                if i >= sample_lines:
                    break
                try:
                    lons.append(float(row[i_lon]))
                    lats.append(float(row[i_lat]))
                except (ValueError, IndexError):
                    continue
    except Exception:
        return list(_gl_bbox())
    if not lons:
        return list(_gl_bbox())
    return [min(lons), min(lats), max(lons), max(lats)]


def _detect_source_key(filepath: Path) -> str:
    """Guess the source key from file path components."""
    parts = filepath.parts
    name = filepath.stem.lower()
    for key in SOURCE_TIER_MAP:
        if key in name:
            return key
    for p in parts:
        pl = p.lower()
        for key in SOURCE_TIER_MAP:
            if key in pl:
                return key
    if "emag2" in name or "emag" in name:
        return "emag2"
    if "wdmam" in name:
        return "wdmam"
    if "swarm" in name:
        return "swarm"
    if "nrcan" in name or "gsc" in name:
        return "nrcan_ca_200m_rtf"
    if "namag" in name:
        return "usgs_namag"
    if "usmag" in name:
        return "usgs_usmag"
    return "local"


def scan_and_catalog_file(filepath: Path, catalog: Catalog) -> Optional[CatalogEntry]:
    """Analyze a single file and add it to the catalog."""
    rel = str(filepath.relative_to(HARVEST_ROOT)).replace("\\", "/")
    fid = _file_id(rel)

    # Skip if already cataloged and unchanged
    existing = catalog.entries.get(fid)
    if existing and existing.file_size_bytes == filepath.stat().st_size:
        return existing

    source_key = _detect_source_key(filepath)
    suffix = filepath.suffix.lower()

    # Determine resolution and bbox
    if suffix in (".tif", ".tiff"):
        resolution_m = estimate_resolution_from_geotiff(filepath)
        bbox = estimate_bbox_from_geotiff(filepath)
        fmt = "geotiff"
    elif suffix == ".csv":
        resolution_m = estimate_resolution_from_csv(filepath)
        bbox = estimate_bbox_from_csv(filepath)
        fmt = "csv"
    elif suffix in (".xyz", ".dat", ".txt"):
        resolution_m = SOURCE_RESOLUTION_MAP.get(source_key, 5_000)
        bbox = list(_gl_bbox())
        fmt = suffix.lstrip(".")
    elif suffix == ".grd":
        resolution_m = SOURCE_RESOLUTION_MAP.get(source_key, 1_000)
        bbox = list(_gl_bbox())
        fmt = "grd"
    else:
        resolution_m = SOURCE_RESOLUTION_MAP.get(source_key, 5_000)
        bbox = list(_gl_bbox())
        fmt = suffix.lstrip(".") or "unknown"

    # Override resolution if we know the source
    if source_key in SOURCE_RESOLUTION_MAP:
        resolution_m = SOURCE_RESOLUTION_MAP[source_key]

    tier = SOURCE_TIER_MAP.get(source_key, classify_tier(resolution_m))

    entry = CatalogEntry(
        file_id=fid,
        rel_path=rel,
        original_source=source_key,
        tier=tier,
        bbox=bbox,
        resolution_m=resolution_m,
        resolution_score=round(1.0 / max(resolution_m, 1), 8),
        format=fmt,
        file_size_bytes=filepath.stat().st_size,
        ingested_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    catalog.add(entry)
    return entry


def init_tiered_structure():
    """Create the tier directory structure."""
    for tier_dir in TIER_DIRS.values():
        tier_dir.mkdir(parents=True, exist_ok=True)
    TIER4_REFERENCE.mkdir(parents=True, exist_ok=True)
    log.info("Tier directories initialized under %s", HARVEST_ROOT)


def scan_existing_data() -> Catalog:
    """Scan all existing magnetic data and build/update the catalog."""
    catalog = Catalog()
    init_tiered_structure()

    scan_dirs = [
        HARVEST_ROOT / "raw",
        HARVEST_ROOT / "grids",
        HARVEST_ROOT / "patches",
        HARVEST_ROOT / "meta",
    ]
    # Also scan tier dirs in case they already have data
    scan_dirs.extend(TIER_DIRS.values())
    scan_dirs.append(TIER4_REFERENCE)

    data_extensions = {".tif", ".tiff", ".csv", ".xyz", ".dat", ".txt", ".grd", ".asc"}
    scanned = 0
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for filepath in scan_dir.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in data_extensions:
                continue
            if filepath.stat().st_size < 512:
                continue
            try:
                scan_and_catalog_file(filepath, catalog)
                scanned += 1
            except Exception as exc:
                log.warning("Could not catalog %s: %s", filepath, exc)

    catalog.save()
    log.info("Scanned %d files, catalog has %d entries", scanned, len(catalog.entries))
    return catalog


def organize_into_tiers(catalog: Catalog, copy: bool = True):
    """Symlink or copy cataloged files into tier directories.
    Uses copy=True by default so tier folders are self-contained.
    Original files in raw/grids are preserved."""
    moved = 0
    for entry in catalog.entries.values():
        src = HARVEST_ROOT / entry.rel_path
        if not src.exists():
            continue
        # Already in a tier directory?
        for td in TIER_DIRS.values():
            try:
                src.relative_to(td)
                break
            except ValueError:
                continue
        else:
            # Not in a tier dir — place it
            tier_dir = TIER_DIRS.get(entry.tier, TIER_DIRS[4])
            dest = tier_dir / entry.original_source / src.name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(str(src), str(dest))
            else:
                # Symlink (Unix) or junction (Windows)
                try:
                    dest.symlink_to(src)
                except OSError:
                    shutil.copy2(str(src), str(dest))
            # Update catalog entry path
            entry.rel_path = str(dest.relative_to(HARVEST_ROOT)).replace("\\", "/")
            moved += 1

    catalog.save()
    log.info("Organized %d files into tier directories", moved)


# ===============================================================================
# SECTION 4: REPOSITORY SEARCH
# ===============================================================================

def _build_ncei_ads_query(bbox: tuple = None, data_type: str = "aeromagnetic") -> dict:
    if bbox is None:
        bbox = _gl_bbox()
    """Build NOAA NCEI Aeromagnetic Data System query parameters.

    NCEI ADS REST endpoint (trackline geophysical data):
      https://www.ngdc.noaa.gov/geomag/aeromag/survey_data.shtml

    The NCEI marine/aero trackline search uses:
      https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/trackline_combined_dynamic/MapServer

    For bulk CSV: the NCEI geophysical data index:
      https://www.ngdc.noaa.gov/mgg/geodata/trackline/

    For EMAG2/grid extraction:
      https://www.ngdc.noaa.gov/geomag/emag2.html
      WCS: https://gis.ngdc.noaa.gov/arcgis/services/geophysical/EMAG2_V3/ImageServer/WCSServer
    """
    lonmin, latmin, lonmax, latmax = bbox

    return {
        # ArcGIS REST query for tracklines in bbox (Layer 0 = combined)
        "trackline_query": {
            "url": "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/trackline_combined_dynamic/MapServer/0/query",
            "params": {
                "where": "1=1",
                "geometry": f"{lonmin},{latmin},{lonmax},{latmax}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "json",
                "resultRecordCount": 500,
            },
        },
        # EMAG2 WCS GetCoverage for the bbox
        "emag2_wcs": {
            "url": "https://gis.ngdc.noaa.gov/arcgis/services/geophysical/EMAG2_V3/ImageServer/WCSServer",
            "params": {
                "service": "WCS",
                "version": "1.1.1",
                "request": "GetCoverage",
                "identifier": "EMAG2_V3",
                "format": "GeoTIFF",
                "boundingbox": f"{latmin},{lonmin},{latmax},{lonmax},urn:ogc:def:crs:EPSG::4326",
                "GridBaseCRS": "urn:ogc:def:crs:EPSG::4326",
            },
        },
        # NCEI geophysical metadata catalog search
        "catalog_search": {
            "url": "https://www.ngdc.noaa.gov/geomag/aeromag/surveys.json",
            "note": "Fallback — may not exist as JSON endpoint; use trackline_query instead",
        },
    }


def _build_sciencebase_query(bbox: tuple = None) -> dict:
    if bbox is None:
        bbox = _gl_bbox()
    """Build USGS ScienceBase catalog API query for magnetic data."""
    lonmin, latmin, lonmax, latmax = bbox
    return {
        "url": "https://www.sciencebase.gov/catalog/items",
        "params": {
            "q": "aeromagnetic magnetic",
            "bbox": f"{lonmin},{latmin},{lonmax},{latmax}",
            "fields": "title,summary,spatial,webLinks,files,dates",
            "format": "json",
            "max": 100,
        },
    }


def _build_deepblue_query(bbox: tuple = None) -> dict:
    if bbox is None:
        bbox = _gl_bbox()
    """Build U-Mich Deep Blue Data search query."""
    return {
        "url": "https://deepblue.lib.umich.edu/data/catalog.json",
        "params": {
            "q": "Great Lakes magnetic aeromagnetic geophysical survey",
            "rows": 50,
            "sort": "score desc",
        },
        "note": "Deep Blue uses Samvera/Hyrax; spatial filtering requires post-processing",
    }


def search_ncei(bbox: tuple = None) -> list[RemoteDataset]:
    if bbox is None:
        bbox = _gl_bbox()
    """Query NOAA NCEI for aeromagnetic trackline surveys."""
    import requests

    results = []
    query = _build_ncei_ads_query(bbox)

    tl = query["trackline_query"]
    log.info("Querying NCEI trackline: %s", tl["url"])

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                tl["url"], params=tl["params"], timeout=60,
                headers={"User-Agent": "MagLakeHarvester/1.0 (Great Lakes research)"}
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("NCEI query attempt %d failed: %s — retrying in %.0fs", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        log.error("NCEI query failed after %d retries", MAX_RETRIES)
        return results

    features = data.get("features", [])
    log.info("NCEI returned %d trackline features", len(features))

    for feat in features:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {})

        survey_id = attrs.get("SURVEY_ID", "unknown")
        name = attrs.get("SURVEY_NAME") or attrs.get("PROJECT") or survey_id
        year = attrs.get("SURVEY_YEAR") or attrs.get("START_YR") or attrs.get("YEAR", "")
        total_km = attrs.get("TOTAL_KM", 0)

        # Estimate bbox from attributes or geometry
        if all(k in attrs for k in ("LON_LEFT", "LAT_BOTTOM", "LON_RIGHT", "LAT_TOP")):
            feat_bbox = [attrs["LON_LEFT"], attrs["LAT_BOTTOM"],
                         attrs["LON_RIGHT"], attrs["LAT_TOP"]]
        else:
            rings = geom.get("rings", geom.get("paths", []))
            if rings:
                all_pts = [pt for ring in rings for pt in ring]
                xs = [p[0] for p in all_pts]
                ys = [p[1] for p in all_pts]
                feat_bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                feat_bbox = list(bbox)

        # Use DOWNLOAD_URL from attributes if available, else construct
        dl_url = attrs.get("DOWNLOAD_URL", "")
        if not dl_url:
            dl_url = f"http://www.ngdc.noaa.gov/trackline/request/?surveyIds={survey_id}"

        # NCEI trackline request URLs are form-based, not direct downloads;
        # tag as metadata_url and use a constructable MGD77T URL pattern
        metadata_url = dl_url
        direct_dl = f"https://www.ngdc.noaa.gov/mgg/geodata/trackline/geophysical/{survey_id}/{survey_id}.m77t"

        results.append(RemoteDataset(
            source_repo="ncei_ads",
            dataset_id=survey_id,
            title=name,
            bbox=feat_bbox,
            resolution_m=200,  # aeromagnetic tracklines ~200 m typical
            download_url=direct_dl,
            format="m77t",
            description=f"NCEI survey {survey_id}, {total_km} km, year {year}, platform={attrs.get('PLATFORM','')}",
            acquisition_date=str(int(year)) if year else "",
            metadata_url=metadata_url,
        ))

    return results


def search_sciencebase(bbox: tuple = None) -> list[RemoteDataset]:
    if bbox is None:
        bbox = _gl_bbox()
    """Query USGS ScienceBase for magnetic survey data."""
    import requests

    results = []
    query = _build_sciencebase_query(bbox)

    log.info("Querying USGS ScienceBase: %s", query["url"])

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                query["url"], params=query["params"], timeout=60,
                headers={"User-Agent": "MagLakeHarvester/1.0"}
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("ScienceBase attempt %d failed: %s — retrying in %.0fs", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        log.error("ScienceBase query failed after %d retries", MAX_RETRIES)
        return results

    items = data.get("items", [])
    log.info("ScienceBase returned %d items", len(items))

    for item in items:
        item_id = item.get("id", "")
        title = item.get("title", "Untitled")

        # Extract spatial extent — only include if ScienceBase returned real bbox
        spatial = item.get("spatial", {})
        has_real_bbox = False
        item_bbox = list(bbox)
        if "boundingBox" in spatial:
            bb = spatial["boundingBox"]
            if all(k in bb for k in ("minX", "minY", "maxX", "maxY")):
                item_bbox = [bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]]
                has_real_bbox = True

        # Skip items without real spatial extent — cannot verify they're in our area
        if not has_real_bbox:
            log.debug("ScienceBase item %s (%s) has no spatial data, skipping",
                      item_id, title[:50])
            continue

        # Find downloadable files
        files = item.get("files", [])
        weblinks = item.get("webLinks", [])
        dl_url = ""
        fmt = "unknown"
        file_size = 0

        for f in files:
            fname = f.get("name", "").lower()
            if any(ext in fname for ext in (".csv", ".tif", ".xyz", ".grd", ".zip")):
                dl_url = f.get("url", "")
                fmt = Path(fname).suffix.lstrip(".")
                file_size = f.get("size", 0)
                break

        if not dl_url:
            for wl in weblinks:
                if wl.get("type") == "download":
                    dl_url = wl.get("uri", "")
                    break

        if not dl_url:
            continue

        results.append(RemoteDataset(
            source_repo="usgs_sciencebase",
            dataset_id=item_id,
            title=title,
            bbox=item_bbox,
            resolution_m=1_000,  # default estimate; refine from metadata
            download_url=dl_url,
            format=fmt,
            file_size_bytes=file_size,
            description=str(item.get("summary", ""))[:500],
            metadata_url=f"https://www.sciencebase.gov/catalog/item/{item_id}",
        ))

    return results


def search_deepblue(bbox: tuple = None) -> list[RemoteDataset]:
    if bbox is None:
        bbox = _gl_bbox()
    """Query University of Michigan Deep Blue Data for magnetic datasets.
    Note: DeepBlue JSON API may return 403; we degrade gracefully."""
    import requests

    results = []
    query = _build_deepblue_query(bbox)

    log.info("Querying U-Mich Deep Blue: %s", query["url"])

    for attempt in range(2):  # Only 2 retries — if blocked, move on
        try:
            resp = requests.get(
                query["url"], params=query["params"], timeout=60,
                headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"}
            )
            if resp.status_code == 403:
                log.warning("Deep Blue returned 403 Forbidden — API may be restricted; skipping")
                return results
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("Deep Blue attempt %d failed: %s — retrying in %.0fs", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        log.warning("Deep Blue query failed — service may be unavailable; skipping")
        return results

    docs = data.get("response", {}).get("docs", [])
    if not docs:
        docs = data.get("data", [])
    log.info("Deep Blue returned %d results", len(docs))

    for doc in docs:
        doc_id = doc.get("id", "")
        title = doc.get("title_tesim", [doc.get("title", "Untitled")])
        if isinstance(title, list):
            title = title[0] if title else "Untitled"

        # Deep Blue spatial filtering must be done client-side
        desc = " ".join(str(v) for v in doc.values()).lower()
        gl_keywords = ("great lakes", "lake erie", "lake huron", "lake michigan",
                       "lake superior", "lake ontario", "magnetic", "aeromagnetic")
        if not any(kw in desc for kw in gl_keywords):
            continue

        dl_url = f"https://deepblue.lib.umich.edu/data/concern/data_sets/{doc_id}"

        results.append(RemoteDataset(
            source_repo="umich_deepblue",
            dataset_id=str(doc_id),
            title=title,
            bbox=list(bbox),
            resolution_m=1_000,
            download_url=dl_url,
            format="unknown",
            description=str(doc.get("description_tesim", "")),
            metadata_url=dl_url,
        ))

    return results


# ── NRCan Geophysical Data Repository ─────────────────────────────────────


def _build_nrcan_csw_query(bbox: tuple) -> dict:
    """Build NRCan CSW (Catalogue Service for the Web) GetRecords request."""
    lonmin, latmin, lonmax, latmax = bbox
    return {
        "url": "https://gdr.agg.nrcan.gc.ca/csw",
        "params": {
            "service": "CSW",
            "version": "2.0.2",
            "request": "GetRecords",
            "typeNames": "csw:Record",
            "resultType": "results",
            "maxRecords": "100",
            "outputFormat": "application/json",
            "elementSetName": "full",
            "constraintLanguage": "CQL_TEXT",
            "constraint": (
                f"AnyText LIKE '%magnetic%' AND "
                f"BBOX(ows:BoundingBox, {latmin}, {lonmin}, {latmax}, {lonmax})"
            ),
        },
    }


def search_nrcan(bbox: tuple = None) -> list[RemoteDataset]:
    """Search NRCan for Canadian aeromagnetic grids.

    Uses Canada Open Data Portal (CKAN API) which is the most reliable endpoint.
    Falls back to CSW if CKAN is unavailable.  Results are tagged source_repo="nrcan_gdr".
    """
    if bbox is None:
        bbox = _gl_bbox()
    import requests

    results = []
    lonmin, latmin, lonmax, latmax = bbox

    # Strategy 1: Canada Open Data Portal (CKAN) -- most reliable
    ckan_url = "https://open.canada.ca/data/api/3/action/package_search"
    search_terms = ["aeromagnetic", "magnetic total field", "magnetic anomaly"]

    seen_ids = set()

    for term in search_terms:
        params = {
            "q": term,
            "fq": "organization:nrcan-rncan",
            "rows": 50,
        }
        log.info("Querying NRCan via Canada Open Data: %s", term)

        try:
            resp = requests.get(ckan_url, params=params, timeout=60,
                                headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
            if resp.status_code != 200:
                log.warning("NRCan CKAN returned %d", resp.status_code)
                continue

            data = resp.json()
            items = data.get("result", {}).get("results", [])

            for item in items:
                pkg_id = item.get("id", "")
                if pkg_id in seen_ids:
                    continue
                seen_ids.add(pkg_id)

                title = item.get("title", "Untitled")

                # Extract spatial extent if available
                item_bbox = list(bbox)
                has_real_bbox = False
                spatial_str = item.get("spatial", "")
                if spatial_str:
                    try:
                        spatial_json = json.loads(spatial_str)
                        coords = spatial_json.get("coordinates", [])
                        if coords:
                            # Polygon or bbox coordinates
                            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                                all_pts = [pt for ring in coords for pt in ring]
                            else:
                                all_pts = coords
                            xs = [p[0] for p in all_pts]
                            ys = [p[1] for p in all_pts]
                            item_bbox = [min(xs), min(ys), max(xs), max(ys)]
                            has_real_bbox = True
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass

                # Check if this dataset overlaps our target bbox
                if has_real_bbox and not _bbox_overlaps(item_bbox, list(bbox)):
                    continue

                # Find downloadable resources (prefer GeoTIFF, ASCII Grid, CSV)
                resources = item.get("resources", [])
                best_resource = None
                preferred_formats = ["geotiff", "tif", "tiff", "ascii", "csv",
                                     "grd", "xyz", "netcdf", "nc"]

                for res in resources:
                    fmt = (res.get("format", "") or "").lower()
                    if any(f in fmt for f in preferred_formats):
                        best_resource = res
                        break
                if best_resource is None and resources:
                    # Take first resource as fallback
                    best_resource = resources[0]

                dl_url = best_resource.get("url", "") if best_resource else ""
                fmt = (best_resource.get("format", "geotiff") if best_resource
                       else "geotiff").lower()

                # Determine resolution from title/notes
                notes = item.get("notes", "")
                res_m = 200  # default NRCan aeromagnetic
                if "1 km" in title.lower() or "1km" in title.lower():
                    res_m = 1000
                elif "200 m" in title.lower() or "200m" in title.lower():
                    res_m = 200

                results.append(RemoteDataset(
                    source_repo="nrcan_gdr",
                    dataset_id=pkg_id,
                    title=title,
                    bbox=item_bbox,
                    resolution_m=res_m,
                    download_url=dl_url,
                    format=fmt,
                    description=(notes or str(item.get("notes_translated", "")))[:500],
                    metadata_url=f"https://open.canada.ca/data/en/dataset/{pkg_id}",
                ))

        except Exception as exc:
            log.error("NRCan CKAN query failed for '%s': %s", term, exc)

    log.info("NRCan: found %d datasets", len(results))

    # Strategy 2: If CKAN returned nothing, try CSW fallback
    if not results:
        csw = _build_nrcan_csw_query(bbox)
        log.info("NRCan CKAN empty; trying CSW fallback: %s", csw["url"])
        try:
            resp = requests.get(csw["url"], params=csw["params"], timeout=60,
                                headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    records = data.get("csw:GetRecordsResponse", {}).get(
                        "csw:SearchResults", {}).get("csw:Record", [])
                    if isinstance(records, dict):
                        records = [records]
                    for rec in records:
                        rid = rec.get("dc:identifier", "")
                        results.append(RemoteDataset(
                            source_repo="nrcan_gdr",
                            dataset_id=rid,
                            title=rec.get("dc:title", "NRCan CSW"),
                            bbox=list(bbox),
                            resolution_m=200,
                            download_url=f"https://gdr.agg.nrcan.gc.ca/gdrdap/e/dap/{rid}",
                            format="geotiff",
                            description=rec.get("dc:description", "")[:500],
                            metadata_url=f"https://gdr.agg.nrcan.gc.ca/gdrdap/e/dap/{rid}",
                        ))
                    log.info("NRCan CSW: found %d records", len(records))
                except Exception:
                    log.debug("NRCan CSW returned non-JSON")
        except Exception as exc:
            log.warning("NRCan CSW fallback failed: %s", exc)

    return results


# ── U-Mich Deep Blue: AI4Shipwrecks & Magnetometer datasets ──────────────


def search_deepblue_ai4shipwrecks(bbox: tuple = None) -> list[RemoteDataset]:
    """Search for AI4Shipwrecks and magnetometer datasets via DataCite DOI API.

    The Deep Blue Solr JSON API returns 403, so we use DataCite to find datasets
    published by UMich Deep Blue, then categorize by researcher and DOI.
    """
    if bbox is None:
        bbox = _gl_bbox()
    import requests

    results = []

    # DataCite API: search for UMich Deep Blue magnetometer / shipwreck datasets
    # Note: client-id filter doesn't work; search broadly and filter by publisher
    queries = [
        "AI4Shipwrecks",
        "magnetometer Great Lakes shipwreck",
        "AUV magnetometer survey lake",
        "side scan sonar Great Lakes shipwreck",
        "Great Lakes sonar magnetometer survey data",
    ]

    seen_dois = set()

    for query_str in queries:
        url = "https://api.datacite.org/dois"
        params = {
            "query": query_str,
            "page[size]": "25",
            "sort": "-created",
        }
        log.info("Querying DataCite for UMich Deep Blue: %s", query_str)

        for attempt in range(2):
            try:
                resp = requests.get(url, params=params, timeout=60,
                                    headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 404:
                    log.debug("DataCite returned 404 for query '%s'", query_str)
                    data = {"data": []}
                    break
                else:
                    log.warning("DataCite returned %d for '%s'", resp.status_code, query_str)
                    data = {"data": []}
                    break
            except Exception as exc:
                wait = RETRY_BACKOFF_BASE ** attempt
                log.warning("DataCite attempt %d failed: %s -- retrying in %.0fs",
                            attempt + 1, exc, wait)
                time.sleep(wait)
                data = {"data": []}

        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            doi = attrs.get("doi", "")
            if doi in seen_dois:
                continue

            # Filter: only include datasets from UMich Deep Blue or relevant repos
            publisher = (attrs.get("publisher") or "").lower()
            container = str(attrs.get("container", {}).get("title", "")).lower()
            umich_publishers = ["university of michigan", "deep blue", "umich"]
            if not any(p in publisher or p in container for p in umich_publishers):
                # Also accept if title/description clearly references Great Lakes data
                all_titles = " ".join(t.get("title", "") for t in attrs.get("titles", []))
                if not any(kw in all_titles.lower() for kw in
                           ["great lakes", "shipwreck", "lake erie", "lake michigan"]):
                    continue

            seen_dois.add(doi)

            title = attrs.get("titles", [{}])[0].get("title", "Untitled")
            description = ""
            descs = attrs.get("descriptions", [])
            if descs:
                description = descs[0].get("description", "")[:500]

            # Extract researcher info
            creators = attrs.get("creators", [])
            researcher = "; ".join(
                c.get("name", c.get("familyName", ""))
                for c in creators[:3]
            )

            # Extract publication date
            pub_year = str(attrs.get("publicationYear", ""))

            # Extract geo info if available
            geo = attrs.get("geoLocations", [])
            item_bbox = list(bbox)
            has_geo = False
            for g in geo:
                gb = g.get("geoLocationBox", {})
                if all(k in gb for k in ("westBoundLongitude", "southBoundLatitude",
                                          "eastBoundLongitude", "northBoundLatitude")):
                    item_bbox = [
                        gb["westBoundLongitude"], gb["southBoundLatitude"],
                        gb["eastBoundLongitude"], gb["northBoundLatitude"],
                    ]
                    has_geo = True
                    break

            # Skip items without geo if they don't mention Great Lakes in title/desc
            if not has_geo:
                combined = (title + " " + description).lower()
                gl_keywords = ["great lakes", "lake erie", "lake michigan", "lake huron",
                               "lake superior", "lake ontario", "shipwreck"]
                if not any(kw in combined for kw in gl_keywords):
                    log.debug("Skipping non-GL DataCite item: %s", title[:60])
                    continue

            # Build download URL: DataCite DOI resolves to Deep Blue landing page
            dl_url = f"https://doi.org/{doi}"

            # Estimate resolution: AUV data ~5m, sonar ~1m, general ~50m
            title_lower = title.lower()
            if "auv" in title_lower or "autonomous" in title_lower:
                res_m = 5.0
                sampling_hz = 10.0
            elif "sonar" in title_lower or "side-scan" in title_lower:
                res_m = 1.0
                sampling_hz = 50.0
            elif "magnetometer" in title_lower or "mag " in title_lower:
                res_m = 10.0
                sampling_hz = 5.0
            else:
                res_m = 50.0
                sampling_hz = 1.0

            # Detect format from descriptions or known patterns
            fmt = "unknown"
            for ext in ["csv", "xyz", "geotiff", "tif", "mat", "hdf5", "netcdf"]:
                if ext in description.lower() or ext in title_lower:
                    fmt = ext
                    break

            results.append(RemoteDataset(
                source_repo="umich_deepblue",
                dataset_id=doi,
                title=title,
                bbox=item_bbox,
                resolution_m=res_m,
                download_url=dl_url,
                format=fmt,
                description=description,
                acquisition_date=pub_year,
                metadata_url=dl_url,
                sampling_rate_hz=sampling_hz,
                researcher=researcher,
                doi=doi,
            ))

    log.info("UMich Deep Blue (DataCite): found %d datasets across %d queries",
             len(results), len(queries))
    return results


# ── ESA Swarm Level 2 Crustal Field Models ────────────────────────────────


def search_esa_swarm(bbox: tuple = None) -> list[RemoteDataset]:
    """Search for ESA Swarm Level 2 crustal field models (MLI products).

    These are the long-wavelength baseline for Tier 4 experimental layer.
    Uses the Swarm dissemination server index to find latest baselines.
    """
    if bbox is None:
        bbox = _gl_bbox()
    import requests

    results = []

    # Swarm dissemination server: FTP-over-HTTPS listing for Level 2 MLI products
    # MLI = Magnetic Lithospheric Inversion (crustal field model)
    swarm_base = "https://swarm-diss.eo.esa.int"
    mli_paths = [
        "/Level2daily/Latest_baselines/MLI/SW_OPER_MLI_SHA_2C",
        "/Level2daily/Latest_baselines/MLI/SW_OPER_MLI_SHA_2D",
        "/Level2daily/Latest_baselines/MLI/SW_OPER_MLI_SHA_2E",
    ]

    # Also try the VirES capabilities endpoint
    vires_url = "https://vires.services/ows"
    vires_params = {
        "service": "WPS",
        "request": "GetCapabilities",
    }

    log.info("Querying ESA Swarm dissemination server and VirES")

    # Approach 1: Try the VirES web service to find available MLI models
    try:
        resp = requests.get(vires_url, params=vires_params, timeout=60,
                            headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
        if resp.status_code == 200:
            log.debug("VirES capabilities retrieved (%d bytes)", len(resp.content))
            # Parse XML for MLI product references (basic extraction)
            content = resp.text
            # Find MLI model references in capabilities
            mli_refs = re.findall(
                r'(SW_OPER_MLI_SHA_2[A-Z]_\d{8}T\d{6}_\d{8}T\d{6}_\d{4})',
                content
            )
            for ref in set(mli_refs):
                # Parse date from reference
                date_match = re.search(r'(\d{8})T\d{6}_(\d{8})', ref)
                start_date = date_match.group(1) if date_match else ""

                results.append(RemoteDataset(
                    source_repo="esa_swarm",
                    dataset_id=ref,
                    title=f"ESA Swarm Level 2 MLI: {ref}",
                    bbox=[-180, -90, 180, 90],  # global product
                    resolution_m=40_000,
                    download_url=f"{swarm_base}/Level2daily/Latest_baselines/MLI/{ref}",
                    format="cdf",  # Common Data Format
                    description="Swarm Level 2 Magnetic Lithospheric Inversion model - "
                                "long-wavelength crustal field baseline",
                    acquisition_date=start_date[:4] if start_date else "",
                    metadata_url="https://earth.esa.int/eogateway/missions/swarm",
                ))
    except Exception as exc:
        log.warning("VirES query failed: %s", exc)

    # Approach 2: Known Swarm MLI product URLs (latest known baselines)
    # These are updated periodically by ESA
    known_products = [
        {
            "id": "SW_OPER_MLI_SHA_2C_latest",
            "title": "Swarm Level 2 MLI (latest Comprehensive Inversion)",
            "url": f"{swarm_base}/Level2daily/Latest_baselines/MLI/",
            "desc": "Latest Swarm comprehensive magnetic lithospheric inversion model. "
                    "Provides long-wavelength (>300km) crustal magnetic field baseline.",
        },
        {
            "id": "CHAOS-7_MMA",
            "title": "CHAOS-7 Magnetospheric Model (via Swarm)",
            "url": "https://spacecenter.dk/files/magnetic-models/CHAOS-7/",
            "desc": "CHAOS-7 model: core + crustal + magnetospheric separation. "
                    "Use crustal component (n >= 15) as Tier 4 baseline reference.",
        },
    ]

    for prod in known_products:
        # Check if already in results from VirES
        if any(r.dataset_id == prod["id"] for r in results):
            continue
        results.append(RemoteDataset(
            source_repo="esa_swarm",
            dataset_id=prod["id"],
            title=prod["title"],
            bbox=[-180, -90, 180, 90],  # global products
            resolution_m=40_000,  # ~40 km effective resolution
            download_url=prod["url"],
            format="cdf",
            description=prod["desc"],
            metadata_url=prod["url"],
        ))

    log.info("ESA Swarm: found %d Level 2 products", len(results))
    return results


# ── USGS Great Lakes Science Center: FAN (Field Activity Number) Scraper ──


def search_glsc_infobank(bbox: tuple = None) -> list[RemoteDataset]:
    """Scrape USGS InfoBank/CMGDS for recent Great Lakes Science Center
    field activities with magnetometer data (2020-2024).

    Targets FANs (Field Activity Numbers) from GLSC missions that contain
    raw magnetometer logs that haven't been gridded into national maps yet.
    """
    if bbox is None:
        bbox = _gl_bbox()
    import requests

    results = []

    # CMGDS data catalog search API
    cmgds_base = "https://cmgds.marine.usgs.gov"

    # Search patterns for GLSC magnetometer field activities
    search_terms = [
        "Great Lakes Science Center magnetics",
        "Lake Michigan magnetometer",
        "Lake Erie magnetometer",
        "Lake Huron magnetometer",
        "Lake Superior magnetometer",
        "Lake Ontario magnetometer",
        "Great Lakes magnetic survey",
    ]

    # Also search the InfoBank directly
    infobank_url = "https://walrus.wr.usgs.gov/infobank/programs/html/search/search.html"

    # Strategy 1: ScienceBase API (CMGDS publishes here too)
    sb_url = "https://www.sciencebase.gov/catalog/items"
    for term in search_terms[:3]:  # limit to avoid rate-limiting
        params = {
            "q": term,
            "fields": "title,summary,spatial,webLinks,files,dates",
            "format": "json",
            "max": 20,
            "filter0": "browseCategory=Data",
        }
        log.info("Querying CMGDS/ScienceBase for GLSC: %s", term)

        try:
            resp = requests.get(sb_url, params=params, timeout=60,
                                headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
            if resp.status_code != 200:
                log.warning("GLSC ScienceBase query returned %d", resp.status_code)
                continue

            data = resp.json()
            items = data.get("items", [])

            for item in items:
                item_id = item.get("id", "")
                title = item.get("title", "Untitled")
                title_lower = title.lower()

                # Filter: must reference magnetics/magnetometer AND Great Lakes
                mag_keywords = ["magnet", "mag survey", "geophysi"]
                gl_keywords = ["great lakes", "glsc", "lake michigan", "lake erie",
                               "lake huron", "lake superior", "lake ontario"]

                has_mag = any(k in title_lower for k in mag_keywords)
                has_gl = any(k in title_lower for k in gl_keywords)

                # Also check summary
                summary = str(item.get("summary", ""))[:500].lower()
                if not has_mag:
                    has_mag = any(k in summary for k in mag_keywords)
                if not has_gl:
                    has_gl = any(k in summary for k in gl_keywords)

                if not (has_mag and has_gl):
                    continue

                # Extract spatial data
                spatial = item.get("spatial", {})
                has_real_bbox = False
                item_bbox = list(bbox)
                if "boundingBox" in spatial:
                    bb = spatial["boundingBox"]
                    if all(k in bb for k in ("minX", "minY", "maxX", "maxY")):
                        item_bbox = [bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]]
                        has_real_bbox = True

                if not has_real_bbox:
                    continue

                # Extract date from item
                dates = item.get("dates", [])
                acq_date = ""
                for d in dates:
                    if d.get("type", "") in ("Start", "Publication"):
                        acq_date = d.get("dateString", "")[:4]
                        break

                # Extract FAN from title if present (e.g., "2023-012-GL")
                fan_match = re.search(r'(\d{4}-\d{3}-[A-Z]{2})', title)
                fan = fan_match.group(1) if fan_match else ""

                # Find downloadable files
                files = item.get("files", [])
                dl_files = [f for f in files
                            if any(f.get("name", "").lower().endswith(ext)
                                   for ext in [".csv", ".txt", ".log", ".xyz", ".dat"])]

                # Also check webLinks for download URLs
                links = item.get("webLinks", [])
                dl_links = [l.get("uri", "") for l in links
                            if l.get("type", "") == "download"]

                if dl_files:
                    dl_url = dl_files[0].get("url", "")
                    fmt = dl_files[0].get("name", "").rsplit(".", 1)[-1] if dl_files else "csv"
                elif dl_links:
                    dl_url = dl_links[0]
                    fmt = "csv"
                else:
                    dl_url = f"https://www.sciencebase.gov/catalog/item/{item_id}"
                    fmt = "unknown"

                # Extract data steward from contacts if available
                steward = ""
                contacts = item.get("contacts", [])
                for c in contacts:
                    if "data" in c.get("type", "").lower():
                        steward = c.get("email", c.get("name", ""))
                        break

                results.append(RemoteDataset(
                    source_repo="usgs_glsc",
                    dataset_id=fan if fan else item_id,
                    title=title,
                    bbox=item_bbox,
                    resolution_m=10,  # GLSC boat surveys are very dense
                    download_url=dl_url,
                    format=fmt,
                    file_size_bytes=sum(f.get("size", 0) for f in dl_files),
                    description=(
                        f"USGS GLSC field activity. "
                        f"FAN: {fan}. "
                        f"Data steward: {steward}. "
                        + summary[:300]
                    ),
                    acquisition_date=acq_date,
                    metadata_url=f"https://www.sciencebase.gov/catalog/item/{item_id}",
                    sampling_rate_hz=10.0,  # typical boat mag ~10 Hz
                ))

        except Exception as exc:
            log.error("GLSC ScienceBase query failed for '%s': %s", term, exc)

    # Strategy 2: Try CMGDS direct API for FAN search
    log.info("Querying CMGDS FAN database for recent (2020-2024) GL activities")
    for year in range(2020, 2025):
        fan_url = f"{cmgds_base}/fan_info.php"
        params = {"fan": f"{year}-*-GL"}
        try:
            resp = requests.get(fan_url, params=params, timeout=30,
                                headers={"User-Agent": "MagLakeHarvester/1.0 (academic research)"})
            if resp.status_code == 200 and "magneti" in resp.text.lower():
                # Parse HTML for data links and steward info
                text = resp.text
                # Extract any HTTPS download links
                dl_urls = re.findall(
                    r'href=["\']?(https?://[^"\'>\s]+\.(?:csv|txt|log|xyz|dat))',
                    text, re.IGNORECASE
                )
                # Extract FAN numbers
                fans = re.findall(r'(\d{4}-\d{3}-[A-Z]{2})', text)
                # Extract email addresses (data steward)
                emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)

                for fan in set(fans):
                    # Only include if not already found
                    if any(r.dataset_id == fan for r in results):
                        continue
                    if not fan.startswith(str(year)):
                        continue

                    results.append(RemoteDataset(
                        source_repo="usgs_glsc",
                        dataset_id=fan,
                        title=f"USGS GLSC Field Activity: {fan}",
                        bbox=list(bbox),
                        resolution_m=10,
                        download_url=dl_urls[0] if dl_urls else f"{cmgds_base}/fan_info.php?fan={fan}",
                        format="csv",
                        description=(
                            f"USGS GLSC FAN {fan}. "
                            f"Data steward: {emails[0] if emails else 'unknown'}. "
                            f"Raw magnetometer sensor log from Great Lakes field mission."
                        ),
                        acquisition_date=str(year),
                        metadata_url=f"{cmgds_base}/fan_info.php?fan={fan}",
                        sampling_rate_hz=10.0,
                    ))
        except Exception as exc:
            log.debug("CMGDS FAN query for %d failed: %s", year, exc)

    log.info("USGS GLSC InfoBank: found %d field activities", len(results))
    return results


def search_all_repositories(bbox: tuple = None) -> list[RemoteDataset]:
    if bbox is None:
        bbox = _gl_bbox()
    """Search all repositories and return combined results."""
    all_results = []
    for searcher, name in [
        (search_ncei, "NCEI ADS"),
        (search_sciencebase, "USGS ScienceBase"),
        (search_deepblue, "U-Mich Deep Blue (legacy)"),
        (search_nrcan, "NRCan Geophysical Data Repository"),
        (search_deepblue_ai4shipwrecks, "U-Mich Deep Blue (AI4Shipwrecks/DataCite)"),
        (search_esa_swarm, "ESA Swarm Level 2"),
        (search_glsc_infobank, "USGS GLSC InfoBank"),
    ]:
        try:
            results = searcher(bbox)
            all_results.extend(results)
            log.info("%s: found %d datasets", name, len(results))
        except Exception as exc:
            log.error("%s search failed: %s", name, exc)
    return all_results


# ===============================================================================
# SECTION 5: SMART DOWNLOAD (GAP-FILL + RESOLUTION UPGRADE)
# ===============================================================================

def should_download(remote: RemoteDataset, catalog: Catalog,
                    target_bbox: Optional[list[float]] = None) -> tuple[bool, str]:
    """Decide if a remote dataset should be downloaded.

    Returns (should_download, reason).
    Only download if:
      1. Its bbox actually overlaps the target region, AND
      2. It covers a geographic gap (no local data in that bbox), OR
      3. It provides better resolution than existing local data in that area.
    Never duplicate existing data.
    """
    # Require the remote dataset to actually overlap our target region
    if target_bbox is not None and not _bbox_overlaps(remote.bbox, target_bbox):
        return False, f"outside-region: bbox does not overlap target"

    # Skip datasets that are way too large relative to our target (>10x area)
    # Exception: ESA Swarm Level 2 products are global by design (Tier 4 baseline)
    if target_bbox is not None and remote.source_repo != "esa_swarm":
        target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        remote_area = (remote.bbox[2] - remote.bbox[0]) * (remote.bbox[3] - remote.bbox[1])
        if target_area > 0 and remote_area > target_area * 10:
            return False, f"too-broad: remote bbox area {remote_area:.1f} >> target {target_area:.1f}"

    # Check for exact duplicate by dataset ID
    for entry in catalog.entries.values():
        if remote.dataset_id in entry.rel_path or remote.dataset_id in entry.original_source:
            return False, f"duplicate: already have {entry.rel_path}"

    # Check geographic coverage overlap with LOCAL data
    overlapping = [
        e for e in catalog.entries.values()
        if _bbox_overlaps(e.bbox, remote.bbox)
    ]

    if not overlapping:
        return True, "gap-fill: no local coverage in this area"

    # Check if remote has better resolution than all overlapping locals
    best_local = min(e.resolution_m for e in overlapping)
    if remote.resolution_m < best_local * 0.8:  # 20% improvement threshold
        return True, f"resolution-upgrade: {remote.resolution_m:.0f}m vs local best {best_local:.0f}m"

    # Check if remote has better sampling rate than local data for same area.
    # If a UMich dataset has higher sampling than a NOAA trackline, prioritize it.
    if remote.sampling_rate_hz is not None and remote.sampling_rate_hz > 0:
        # Check if any overlapping local entry has known sampling rate
        local_max_rate = 0
        for e in overlapping:
            if hasattr(e, "sampling_rate_hz") and e.sampling_rate_hz:
                local_max_rate = max(local_max_rate, e.sampling_rate_hz)
        if local_max_rate > 0 and remote.sampling_rate_hz > local_max_rate * 1.5:
            return True, (f"sampling-rate-upgrade: remote {remote.sampling_rate_hz:.1f} Hz "
                          f"vs local best {local_max_rate:.1f} Hz "
                          f"(source: {remote.source_repo})")

    # Check if remote covers MORE area than any single local entry
    remote_area = (remote.bbox[2] - remote.bbox[0]) * (remote.bbox[3] - remote.bbox[1])
    for e in overlapping:
        local_area = (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])
        # Compute overlap area
        olap_lonmin = max(remote.bbox[0], e.bbox[0])
        olap_latmin = max(remote.bbox[1], e.bbox[1])
        olap_lonmax = min(remote.bbox[2], e.bbox[2])
        olap_latmax = min(remote.bbox[3], e.bbox[3])
        if olap_lonmax > olap_lonmin and olap_latmax > olap_latmin:
            overlap_area = (olap_lonmax - olap_lonmin) * (olap_latmax - olap_latmin)
        else:
            overlap_area = 0
        # If less than 70% of remote area is already covered -> gap-fill
        if overlap_area < remote_area * 0.7:
            return True, f"partial-gap-fill: only {overlap_area / max(remote_area, 1):.0%} overlapped"

    return False, f"skip: area covered at {best_local:.0f}m resolution"


def download_with_retry(url: str, dest: Path, max_retries: int = MAX_RETRIES) -> bool:
    """Download with exponential backoff retry logic."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    for attempt in range(max_retries):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "MagLakeHarvester/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers)

            if resp.status_code == 416:
                if partial.exists():
                    partial.replace(dest)
                return True

            resp.raise_for_status()

            mode = "ab" if resp.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0

            with partial.open(mode) as fh:
                for chunk in resp.iter_content(DOWNLOAD_CHUNK):
                    fh.write(chunk)

            partial.replace(dest)
            log.info("Downloaded %s (%d MB)", dest.name, dest.stat().st_size // (1024 * 1024))
            return True

        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("Download attempt %d/%d failed for %s: %s — retry in %.0fs",
                        attempt + 1, max_retries, url, exc, wait)
            time.sleep(wait)

    log.error("Download failed after %d retries: %s", max_retries, url)
    return False


def smart_fetch(catalog: Catalog, remote_datasets: list[RemoteDataset],
                target_bbox: Optional[list[float]] = None) -> list[CatalogEntry]:
    """Download only new data that provides better resolution or fills gaps."""
    if target_bbox is None:
        target_bbox = list(_gl_bbox())
    new_entries = []
    skipped = 0
    fetched = 0

    for remote in remote_datasets:
        do_dl, reason = should_download(remote, catalog, target_bbox=target_bbox)
        if not do_dl:
            log.info("SKIP %s — %s", remote.title, reason)
            skipped += 1
            continue

        log.info("FETCH %s — %s", remote.title, reason)

        # Determine tier and destination
        tier = classify_tier(remote.resolution_m)
        tier_dir = TIER_DIRS.get(tier, TIER_DIRS[4])
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', remote.dataset_id)
        ext = f".{remote.format}" if remote.format != "unknown" else ""
        dest = tier_dir / remote.source_repo / f"{safe_name}{ext}"

        if download_with_retry(remote.download_url, dest):
            entry = CatalogEntry(
                file_id=_file_id(str(dest.relative_to(HARVEST_ROOT))),
                rel_path=str(dest.relative_to(HARVEST_ROOT)).replace("\\", "/"),
                original_source=f"{remote.source_repo}:{remote.dataset_id}",
                tier=tier,
                bbox=remote.bbox,
                resolution_m=remote.resolution_m,
                resolution_score=round(1.0 / max(remote.resolution_m, 1), 8),
                format=remote.format,
                file_size_bytes=dest.stat().st_size if dest.exists() else 0,
                acquisition_date=remote.acquisition_date,
                ingested_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            catalog.add(entry)
            new_entries.append(entry)
            fetched += 1

    catalog.save()
    log.info("Smart fetch complete: %d fetched, %d skipped", fetched, skipped)
    return new_entries


# ===============================================================================
# SECTION 6: METADATA EXTRACTION (SENSOR ALTITUDE & LINE SPACING)
# ===============================================================================

# Common header patterns in aeromagnetic data files
_RE_ALTITUDE = re.compile(
    r"(?:sensor|flight|drape|terrain)\s*(?:altitude|height|clearance)\s*[:=]?\s*([\d.]+)\s*(m|ft|meters?|feet)?",
    re.IGNORECASE,
)
_RE_LINE_SPACING = re.compile(
    r"(?:line|flight.?line|traverse)\s*(?:spacing|separation|interval)\s*[:=]?\s*([\d.]+)\s*(m|km|ft|meters?|kilometers?|feet)?",
    re.IGNORECASE,
)
_RE_DATE = re.compile(
    r"(?:date|acquired|flown|survey.?date)\s*[:=]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4})",
    re.IGNORECASE,
)


def extract_header_metadata(filepath: Path) -> dict:
    """Extract sensor altitude, line spacing, and dates from file headers.

    Reads the first 200 lines looking for structured header comments.
    Common in NCEI .m77t, USGS .xyz, and NRCan .csv files.
    """
    meta = {
        "sensor_altitude_m": None,
        "line_spacing_m": None,
        "acquisition_date": None,
    }

    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 200:
                    break
                line = line.strip()
                if not line:
                    continue

                # Sensor altitude
                m = _RE_ALTITUDE.search(line)
                if m and meta["sensor_altitude_m"] is None:
                    val = float(m.group(1))
                    unit = (m.group(2) or "m").lower()
                    if unit.startswith("ft") or unit.startswith("feet"):
                        val *= 0.3048
                    elif unit.startswith("km"):
                        val *= 1000
                    meta["sensor_altitude_m"] = round(val, 1)

                # Line spacing
                m = _RE_LINE_SPACING.search(line)
                if m and meta["line_spacing_m"] is None:
                    val = float(m.group(1))
                    unit = (m.group(2) or "m").lower()
                    if unit.startswith("km"):
                        val *= 1000
                    elif unit.startswith("ft") or unit.startswith("feet"):
                        val *= 0.3048
                    meta["line_spacing_m"] = round(val, 1)

                # Date
                m = _RE_DATE.search(line)
                if m and meta["acquisition_date"] is None:
                    meta["acquisition_date"] = m.group(1)

    except Exception as exc:
        log.debug("Could not parse headers from %s: %s", filepath, exc)

    return meta


def enrich_catalog_metadata(catalog: Catalog):
    """Run header extraction on all catalog entries and update metadata."""
    enriched = 0
    for entry in catalog.entries.values():
        filepath = HARVEST_ROOT / entry.rel_path
        if not filepath.exists():
            continue

        # Skip binary formats that won't have text headers
        if entry.format in ("geotiff",):
            continue

        meta = extract_header_metadata(filepath)

        if meta["sensor_altitude_m"] is not None:
            entry.sensor_altitude_m = meta["sensor_altitude_m"]
        if meta["line_spacing_m"] is not None:
            entry.line_spacing_m = meta["line_spacing_m"]
            # Refine resolution estimate from actual line spacing
            entry.resolution_m = meta["line_spacing_m"]
            entry.resolution_score = round(1.0 / max(entry.resolution_m, 1), 8)
            entry.tier = classify_tier(entry.resolution_m)
        if meta["acquisition_date"] is not None:
            entry.acquisition_date = meta["acquisition_date"]

        if any(v is not None for v in meta.values()):
            enriched += 1

    catalog.save()
    log.info("Enriched metadata for %d entries", enriched)


def build_master_index(catalog: Catalog):
    """Build the Master Index from catalog + extended metadata."""
    index = []
    for entry in catalog.entries.values():
        mi = MasterIndexEntry(
            file_id=entry.file_id,
            rel_path=entry.rel_path,
            tier=entry.tier,
            bbox=entry.bbox,
            resolution_m=entry.resolution_m,
            sensor_altitude_m=entry.sensor_altitude_m,
            line_spacing_m=entry.line_spacing_m,
        )

        # Read raster stats if GeoTIFF
        filepath = HARVEST_ROOT / entry.rel_path
        if filepath.exists() and entry.format == "geotiff":
            try:
                import rasterio
                import numpy as np
                with rasterio.open(str(filepath)) as ds:
                    band = ds.read(1)
                    valid = band[~np.isnan(band)] if hasattr(band, '__len__') else band
                    if len(valid) > 0:
                        mi.data_min = float(np.nanmin(valid))
                        mi.data_max = float(np.nanmax(valid))
                        mi.data_mean = float(np.nanmean(valid))
            except Exception:
                pass

        index.append(asdict(mi))

    MASTER_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry_count": len(index),
        "entries": index,
    }
    MASTER_INDEX_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Master index written: %d entries -> %s", len(index), MASTER_INDEX_PATH)


# ===============================================================================
# SECTION 7: ZSTD COMPRESSION FOR PROCESSED ASCII
# ===============================================================================

def compress_ascii_files(catalog: Catalog, min_size_bytes: int = 1_000_000):
    """Compress large ASCII data files with zstd to save space.
    Original files are kept until the compressed version is verified.
    The catalog is updated with compressed paths."""
    try:
        import zstandard as zstd
    except ImportError:
        log.warning("zstandard not installed — skipping compression (pip install zstandard)")
        return

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed_count = 0

    ascii_formats = {"csv", "xyz", "dat", "txt", "m77t", "asc"}

    for entry in list(catalog.entries.values()):
        if entry.compressed:
            continue
        if entry.format not in ascii_formats:
            continue

        filepath = HARVEST_ROOT / entry.rel_path
        if not filepath.exists():
            continue
        if filepath.stat().st_size < min_size_bytes:
            continue

        zst_path = filepath.with_suffix(filepath.suffix + ".zst")
        try:
            with filepath.open("rb") as fin, zst_path.open("wb") as fout:
                compressor.copy_stream(fin, fout)

            # Verify compressed file is readable
            decompressor = zstd.ZstdDecompressor()
            with zst_path.open("rb") as f:
                # Read first 4KB to verify
                reader = decompressor.stream_reader(f)
                reader.read(4096)
                reader.close()

            entry.compressed = True
            entry.compressed_path = str(zst_path.relative_to(HARVEST_ROOT)).replace("\\", "/")
            compressed_count += 1

            orig_mb = filepath.stat().st_size / 1e6
            comp_mb = zst_path.stat().st_size / 1e6
            ratio = comp_mb / max(orig_mb, 0.001) * 100
            log.info("Compressed %s: %.1f MB -> %.1f MB (%.0f%%)", filepath.name, orig_mb, comp_mb, ratio)

        except Exception as exc:
            log.warning("Compression failed for %s: %s", filepath, exc)
            if zst_path.exists():
                zst_path.unlink(missing_ok=True)

    catalog.save()
    log.info("Compressed %d ASCII files with zstd", compressed_count)


def decompress_for_query(entry: CatalogEntry) -> Path:
    """Decompress a zstd-compressed file to a temp path for querying."""
    if not entry.compressed or not entry.compressed_path:
        return HARVEST_ROOT / entry.rel_path

    try:
        import zstandard as zstd
    except ImportError:
        # Fall back to original if still exists
        orig = HARVEST_ROOT / entry.rel_path
        if orig.exists():
            return orig
        raise ImportError("zstandard required to decompress")

    zst_path = HARVEST_ROOT / entry.compressed_path
    if not zst_path.exists():
        return HARVEST_ROOT / entry.rel_path

    import tempfile
    tmp = Path(tempfile.mktemp(suffix=Path(entry.rel_path).suffix))
    decompressor = zstd.ZstdDecompressor()
    with zst_path.open("rb") as fin, tmp.open("wb") as fout:
        decompressor.copy_stream(fin, fout)
    return tmp


# ===============================================================================
# SECTION 8: TIER 4 SATELLITE BASELINE (SWARM / EMAG2)
# ===============================================================================

def fetch_emag2_baseline(bbox: tuple = None) -> Optional[Path]:
    if bbox is None:
        bbox = _gl_bbox()
    """Download EMAG2v3 grid for the full Great Lakes basin via WCS.
    Stored in tier_4_reference/ as the persistent normalization baseline."""
    import requests

    dest = TIER4_REFERENCE / "emag2_v3_great_lakes_baseline.tif"
    if dest.exists() and dest.stat().st_size > 10_000:
        log.info("EMAG2 baseline already cached: %s", dest)
        return dest

    query = _build_ncei_ads_query(bbox)
    wcs = query["emag2_wcs"]

    log.info("Fetching EMAG2v3 baseline via WCS for bbox %s", bbox)
    TIER4_REFERENCE.mkdir(parents=True, exist_ok=True)

    if download_with_retry(f"{wcs['url']}?{urlencode(wcs['params'])}", dest):
        return dest

    log.warning("EMAG2 WCS download failed — trying ArcGIS ImageServer export")
    # Fallback: ArcGIS ImageServer export
    fallback_url = (
        "https://gis.ngdc.noaa.gov/arcgis/rest/services/geophysical/EMAG2_V3/ImageServer/exportImage"
    )
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": "4326",
        "size": "2048,1024",
        "format": "tiff",
        "f": "image",
    }
    if download_with_retry(f"{fallback_url}?{urlencode(params)}", dest):
        return dest

    log.error("Could not download EMAG2 baseline")
    return None


def fetch_swarm_baseline(bbox: tuple = None) -> Optional[Path]:
    if bbox is None:
        bbox = _gl_bbox()
    """Download Swarm lithospheric field model data via VirES.

    VirES for Swarm: https://vires.services
    Uses the viresclient Python package if available, otherwise falls back
    to direct REST query.
    """
    dest = TIER4_REFERENCE / "swarm_litho_great_lakes_baseline.csv"
    if dest.exists() and dest.stat().st_size > 1_000:
        log.info("Swarm baseline already cached: %s", dest)
        return dest

    TIER4_REFERENCE.mkdir(parents=True, exist_ok=True)

    # Try viresclient first
    try:
        from viresclient import SwarmRequest

        request = SwarmRequest("https://vires.services/ows")
        request.set_collection("SW_OPER_MLIT_LOSR_RP:MLI_SHA_2E")
        request.set_products(
            measurements=["F"],
            sampling_step="PT60S",
        )
        # VirES uses time filters; get latest available data
        request.set_range_filter("Latitude", bbox[1], bbox[3])
        request.set_range_filter("Longitude", bbox[0], bbox[2])

        data = request.get_between("2024-01-01", "2025-12-31")
        df = data.as_dataframe()
        df.to_csv(str(dest), index=False)
        log.info("Swarm lithospheric data saved: %d points -> %s", len(df), dest)
        return dest

    except ImportError:
        log.info("viresclient not installed — using REST fallback")
    except Exception as exc:
        log.warning("viresclient failed: %s — using REST fallback", exc)

    # REST fallback: VirES WPS or direct download of Swarm MLI model
    # The Swarm MLI (Magnetic field of the Lithosphere) product SHA file
    import requests
    mli_url = "https://swarm-diss.eo.esa.int/Latest_baselines/MCO/SW_OPER_MCO_SHA_2X_20131125T000000_20250101T000000_0801.ZIP"
    zip_dest = TIER4_REFERENCE / "swarm_mco_sha.zip"

    if download_with_retry(mli_url, zip_dest):
        log.info("Swarm MCO SHA model downloaded: %s", zip_dest)
        # We'll process this in the baseline normalization step
        return zip_dest

    log.warning("Swarm baseline download failed — will use EMAG2 only")
    return None


def build_tier4_reference_grid(bbox: tuple = None) -> Optional[Path]:
    if bbox is None:
        bbox = _gl_bbox()
    """Build or verify the Tier 4 reference grid for baseline normalization.

    This combines EMAG2 + Swarm into a single long-wavelength reference that
    all Tier 1-3 data gets normalized against.
    """
    ref_grid = TIER4_REFERENCE / "combined_baseline_grid.tif"
    if ref_grid.exists() and ref_grid.stat().st_size > 10_000:
        log.info("Combined baseline grid exists: %s", ref_grid)
        return ref_grid

    emag2_path = fetch_emag2_baseline(bbox)
    swarm_path = fetch_swarm_baseline(bbox)

    if not emag2_path:
        log.error("Cannot build baseline without EMAG2 data")
        return None

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Load EMAG2 as primary baseline
        with rasterio.open(str(emag2_path)) as ds:
            emag2_data = ds.read(1).astype(np.float64)
            transform = ds.transform
            crs = ds.crs
            height, width = emag2_data.shape

        # If Swarm data available, blend long-wavelength components
        if swarm_path and swarm_path.suffix == ".csv":
            try:
                swarm_pts = []
                with swarm_path.open("r") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        try:
                            lon = float(row.get("Longitude", row.get("lon", 0)))
                            lat = float(row.get("Latitude", row.get("lat", 0)))
                            f = float(row.get("F", row.get("value", 0)))
                            swarm_pts.append((lon, lat, f))
                        except (ValueError, KeyError):
                            continue

                if swarm_pts:
                    from scipy.interpolate import griddata
                    pts = np.array([(p[0], p[1]) for p in swarm_pts])
                    vals = np.array([p[2] for p in swarm_pts])

                    # Create matching grid coordinates
                    cols = np.arange(width)
                    rows = np.arange(height)
                    xs = transform[2] + cols * transform[0]
                    ys = transform[5] + rows * transform[4]
                    gx, gy = np.meshgrid(xs, ys)

                    swarm_grid = griddata(pts, vals, (gx, gy), method="linear")
                    valid = ~np.isnan(swarm_grid) & ~np.isnan(emag2_data)

                    if np.any(valid):
                        # Weight: 80% EMAG2, 20% Swarm for long-wavelength blend
                        blended = emag2_data.copy()
                        blended[valid] = 0.8 * emag2_data[valid] + 0.2 * swarm_grid[valid]
                        emag2_data = blended
                        log.info("Blended Swarm data into baseline (%d valid pixels)", np.sum(valid))

            except Exception as exc:
                log.warning("Swarm blending failed: %s — using EMAG2 only", exc)

        # Write combined baseline
        with rasterio.open(
            str(ref_grid), "w", driver="GTiff",
            height=height, width=width, count=1,
            dtype="float64", crs=crs, transform=transform,
        ) as dst:
            dst.write(emag2_data, 1)

        log.info("Combined baseline grid written: %s", ref_grid)
        return ref_grid

    except ImportError as exc:
        log.error("Missing dependency for baseline grid: %s", exc)
        return None


def normalize_to_baseline(data_path: Path, baseline_path: Path, output_path: Path) -> bool:
    """Remove long-wavelength regional field from Tier 1-3 data using the
    Tier 4 satellite baseline.

    Subtracts the upward-continued satellite field from higher-resolution data
    to isolate short-wavelength anomalies (the ones that could be wrecks).
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling

        with rasterio.open(str(data_path)) as ds_data:
            data = ds_data.read(1).astype(np.float64)
            data_meta = ds_data.meta.copy()

        with rasterio.open(str(baseline_path)) as ds_base:
            # Reproject baseline to match data grid
            baseline_resampled = np.empty_like(data)
            reproject(
                source=rasterio.band(ds_base, 1),
                destination=baseline_resampled,
                src_transform=ds_base.transform,
                src_crs=ds_base.crs,
                dst_transform=ds_data.transform,
                dst_crs=ds_data.crs,
                resampling=Resampling.bilinear,
            )

        # Subtract baseline (regional removal)
        valid = ~np.isnan(data) & ~np.isnan(baseline_resampled)
        residual = data.copy()
        residual[:] = np.nan
        residual[valid] = data[valid] - baseline_resampled[valid]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_meta.update({"dtype": "float64"})
        with rasterio.open(str(output_path), "w", **data_meta) as dst:
            dst.write(residual, 1)

        log.info("Baseline-normalized: %s -> %s", data_path.name, output_path.name)
        return True

    except Exception as exc:
        log.error("Baseline normalization failed for %s: %s", data_path, exc)
        return False


# ===============================================================================
# SECTION 9: GAP INTERPOLATION WITH EXPERIMENTAL_SAT_FILL
# ===============================================================================

def identify_flight_line_gaps(
    data_path: Path,
    max_spacing_ratio: float = 2.0,
) -> Optional[dict]:
    """Identify gaps in aeromagnetic data where flight line spacing exceeds
    the acceptable ratio (default 2:1).

    Returns a mask array and gap statistics, or None if no gaps detected.
    The gap mask is True where data is missing AND the spacing between
    adjacent flight lines exceeds max_spacing_ratio × nominal line spacing.

    NOTE: This is where the future training pipeline can learn to pull
    larger wrecks from the gaps — not just geological filler, but actual
    anomaly detection at satellite resolution. The EXPERIMENTAL_SAT_FILL
    pixels become training targets once we have enough high-res ground truth
    to validate against.
    """
    try:
        import numpy as np
        import rasterio

        with rasterio.open(str(data_path)) as ds:
            data = ds.read(1)
            transform = ds.transform
            pixel_size_m = abs(transform[0]) * 111_000

        # NaN mask = data gaps
        nan_mask = np.isnan(data)
        if not np.any(nan_mask):
            return None

        # Estimate nominal line spacing from data density per row
        data_per_row = np.sum(~nan_mask, axis=1)
        valid_rows = data_per_row[data_per_row > 0]
        if len(valid_rows) == 0:
            return None

        median_pts_per_row = float(np.median(valid_rows))
        nominal_spacing_px = data.shape[1] / max(median_pts_per_row, 1)
        nominal_spacing_m = nominal_spacing_px * pixel_size_m

        # Identify gap columns: consecutive NaN columns wider than threshold
        gap_mask = np.zeros_like(data, dtype=bool)
        threshold_px = int(nominal_spacing_px * max_spacing_ratio)

        for row_idx in range(data.shape[0]):
            row = nan_mask[row_idx]
            gap_start = None
            for col_idx in range(data.shape[1]):
                if row[col_idx]:
                    if gap_start is None:
                        gap_start = col_idx
                else:
                    if gap_start is not None:
                        gap_width = col_idx - gap_start
                        if gap_width > threshold_px:
                            gap_mask[row_idx, gap_start:col_idx] = True
                        gap_start = None
            # Handle trailing gap
            if gap_start is not None:
                gap_width = data.shape[1] - gap_start
                if gap_width > threshold_px:
                    gap_mask[row_idx, gap_start:] = True

        gap_pixels = int(np.sum(gap_mask))
        total_pixels = data.size
        gap_fraction = gap_pixels / max(total_pixels, 1)

        return {
            "gap_mask": gap_mask,
            "gap_pixels": gap_pixels,
            "total_pixels": total_pixels,
            "gap_fraction": gap_fraction,
            "nominal_line_spacing_m": nominal_spacing_m,
            "threshold_spacing_m": nominal_spacing_m * max_spacing_ratio,
            "transform": transform,
            "shape": data.shape,
        }

    except Exception as exc:
        log.error("Gap identification failed for %s: %s", data_path, exc)
        return None


def fill_gaps_with_satellite(
    data_path: Path,
    baseline_path: Path,
    output_path: Path,
    max_spacing_ratio: float = 2.0,
) -> dict:
    """Fill flight-line gaps with satellite data and flag as EXPERIMENTAL_SAT_FILL.

    The filled pixels are marked in a companion mask raster so downstream ML
    can treat them differently:
      - LOW-CONFIDENCE for wreck identification (satellite too coarse for small targets)
      - HIGH-CONFIDENCE for regional geological noise removal (satellite captures
        the crustal field well)
      - FUTURE TRAINING TARGET: the gap regions become candidates for the
        aero->wreck ML model that aims to detect large wrecks (>50m steel vessels)
        even at 200m line spacing, using the satellite-fill as initial estimate
        and the surrounding high-res data as context for the model.

    Returns statistics dict.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    stats = {
        "filled_pixels": 0,
        "total_gap_pixels": 0,
        "fill_type": "EXPERIMENTAL_SAT_FILL",
        "ml_wreck_confidence": "low-confidence",
        "ml_geology_confidence": "high-confidence",
        "training_target": True,
    }

    gap_info = identify_flight_line_gaps(data_path, max_spacing_ratio)
    if gap_info is None:
        log.info("No significant gaps in %s", data_path.name)
        return stats

    gap_mask = gap_info["gap_mask"]
    stats["total_gap_pixels"] = gap_info["gap_pixels"]

    with rasterio.open(str(data_path)) as ds_data:
        data = ds_data.read(1).astype(np.float64)
        data_meta = ds_data.meta.copy()

    with rasterio.open(str(baseline_path)) as ds_base:
        baseline_resampled = np.empty_like(data)
        reproject(
            source=rasterio.band(ds_base, 1),
            destination=baseline_resampled,
            src_transform=ds_base.transform,
            src_crs=ds_base.crs,
            dst_transform=ds_data.transform,
            dst_crs=ds_data.crs,
            resampling=Resampling.bilinear,
        )

    # Fill gaps with satellite values
    fill_pixels = gap_mask & ~np.isnan(baseline_resampled)
    data[fill_pixels] = baseline_resampled[fill_pixels]
    stats["filled_pixels"] = int(np.sum(fill_pixels))

    # Write filled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_meta.update({"dtype": "float64"})
    with rasterio.open(str(output_path), "w", **data_meta) as dst:
        dst.write(data, 1)

    # Write companion mask: 0=original, 1=EXPERIMENTAL_SAT_FILL
    mask_path = output_path.with_name(output_path.stem + "_fill_mask.tif")
    mask_data = np.zeros_like(data, dtype=np.uint8)
    mask_data[fill_pixels] = 1

    mask_meta = data_meta.copy()
    mask_meta.update({"dtype": "uint8", "count": 1})
    with rasterio.open(str(mask_path), "w", **mask_meta) as dst:
        dst.write(mask_data, 1)

    log.info(
        "Gap-filled %s: %d/%d pixels filled with satellite data, "
        "mask -> %s (wreck-confidence=LOW, geology-confidence=HIGH, training-target=YES)",
        data_path.name, stats["filled_pixels"], stats["total_gap_pixels"],
        mask_path.name,
    )

    return stats


# ===============================================================================
# SECTION 10: ML CONFIDENCE ANNOTATIONS
# ===============================================================================

@dataclass
class MLConfidenceConfig:
    """Configuration for how the ML model should treat different data tiers
    and fill types."""

    # Per-tier confidence for wreck identification
    wreck_confidence = {
        1: "highest",          # Marine bottom-tow: best for wrecks
        2: "high",             # Low-alt aero: good for large wrecks
        3: "medium",           # Regional aero: marginal for wrecks
        4: "baseline-only",    # Satellite: not useful for direct wreck ID
    }

    # Per-tier confidence for geological noise removal
    geology_confidence = {
        1: "low",              # Too local for regional geology
        2: "medium",           # OK for intermediate-wavelength geology
        3: "high",             # Good for regional crustal structure
        4: "highest",          # Satellite captures regional geology best
    }

    # Experimental fill handling
    sat_fill_wreck_confidence = "low-confidence"
    sat_fill_geology_confidence = "high-confidence"

    # Future training pipeline hints
    # The gap pixels filled with satellite data are interesting because:
    # 1. Large wrecks (>50m steel freighters) have magnetic signatures
    #    detectable even at 200m aeromagnetic line spacing
    # 2. The satellite fill provides a baseline estimate that the model
    #    can learn to refine using surrounding high-res context
    # 3. The model should learn to distinguish wreck dipoles from
    #    geological features at this resolution
    large_wreck_threshold_m = 50  # minimum wreck size for gap detection
    training_use_sat_fill = True
    training_sat_fill_label = "EXPERIMENTAL_SAT_FILL"


def generate_ml_confidence_layer(
    data_path: Path,
    fill_mask_path: Optional[Path],
    tier: int,
    output_path: Path,
) -> Path:
    """Generate a per-pixel ML confidence raster for the wreck detection model.

    Output bands:
      Band 1: Wreck detection confidence (0–255, higher=more confident)
      Band 2: Geology noise removal confidence (0–255)
      Band 3: Training target flag (0=standard, 1=experimental fill, 2=gap-edge context)
    """
    import numpy as np
    import rasterio

    config = MLConfidenceConfig()
    confidence_map = {"highest": 255, "high": 200, "medium": 128, "low": 64, "baseline-only": 16}

    with rasterio.open(str(data_path)) as ds:
        shape = (ds.height, ds.width)
        meta = ds.meta.copy()
        data = ds.read(1)

    wreck_conf = np.full(shape, confidence_map.get(config.wreck_confidence.get(tier, "medium"), 128), dtype=np.uint8)
    geol_conf = np.full(shape, confidence_map.get(config.geology_confidence.get(tier, "medium"), 128), dtype=np.uint8)
    training_flag = np.zeros(shape, dtype=np.uint8)

    # Where data is NaN, confidence = 0
    nan_mask = np.isnan(data)
    wreck_conf[nan_mask] = 0
    geol_conf[nan_mask] = 0

    # Apply fill mask if present
    if fill_mask_path and fill_mask_path.exists():
        with rasterio.open(str(fill_mask_path)) as ds_mask:
            fill_mask = ds_mask.read(1)

        sat_filled = fill_mask == 1
        wreck_conf[sat_filled] = confidence_map["low"]
        geol_conf[sat_filled] = confidence_map["highest"]
        training_flag[sat_filled] = 1  # EXPERIMENTAL_SAT_FILL

        # Mark edge pixels adjacent to fills as training context
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(sat_filled, iterations=3) & ~sat_filled & ~nan_mask
        training_flag[edge] = 2  # gap-edge context for training

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update({"dtype": "uint8", "count": 3})
    with rasterio.open(str(output_path), "w", **meta) as dst:
        dst.write(wreck_conf, 1)
        dst.write(geol_conf, 2)
        dst.write(training_flag, 3)

    log.info("ML confidence layer written: %s (3 bands)", output_path.name)
    return output_path


# ===============================================================================
# SECTION 11: FULL PIPELINE ORCHESTRATION
# ===============================================================================

def run_init():
    """Stage: Initialize — scan data, build catalog, organize tiers."""
    log.info("=== INIT: Scanning existing data and building catalog ===")
    catalog = scan_existing_data()
    organize_into_tiers(catalog)
    enrich_catalog_metadata(catalog)
    build_master_index(catalog)
    print(f"\nCatalog: {len(catalog.entries)} files")
    summary = catalog.summary()
    for tier, info in summary.get("tiers", {}).items():
        print(f"  Tier {tier}: {info['count']} files, {info['total_mb']:.1f} MB, best {info['best_res_m']:.0f}m")
    return catalog


def run_search():
    """Stage: Search remote repositories."""
    log.info("=== SEARCH: Querying remote repositories ===")
    results = search_all_repositories()
    # Save search results
    search_out = HARVEST_ROOT / "search_results.json"
    payload = {
        "searched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bbox": list(_gl_bbox()),
        "results": [asdict(r) for r in results],
    }
    search_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFound {len(results)} remote datasets -> {search_out}")
    for r in results[:10]:
        print(f"  [{r.source_repo}] {r.title} ({r.resolution_m:.0f}m)")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
    return results


def run_fetch(catalog: Optional[Catalog] = None, remote: Optional[list] = None):
    """Stage: Smart download — only fetch upgrades/gap-fills."""
    if catalog is None:
        catalog = Catalog()
    if remote is None:
        search_path = HARVEST_ROOT / "search_results.json"
        if search_path.exists():
            data = json.loads(search_path.read_text(encoding="utf-8"))
            remote = [
                RemoteDataset(**{
                    k: v for k, v in r.items()
                    if k in RemoteDataset.__dataclass_fields__
                })
                for r in data.get("results", [])
            ]
        else:
            log.warning("No search results found — run 'search' first")
            remote = []

    log.info("=== FETCH: Smart download of %d candidates ===", len(remote))
    new_entries = smart_fetch(catalog, remote, target_bbox=list(_gl_bbox()))
    if new_entries:
        enrich_catalog_metadata(catalog)
        build_master_index(catalog)
    print(f"\nFetched {len(new_entries)} new datasets")
    return new_entries


def run_baseline():
    """Stage: Build Tier 4 satellite reference baseline."""
    log.info("=== BASELINE: Building Tier 4 satellite reference ===")
    ref_grid = build_tier4_reference_grid()
    if ref_grid:
        print(f"\nBaseline grid ready: {ref_grid}")
    else:
        print("\nWARNING: Could not build baseline grid")
    return ref_grid


def run_compress(catalog: Optional[Catalog] = None):
    """Stage: Compress processed ASCII files with zstd."""
    if catalog is None:
        catalog = Catalog()
    log.info("=== COMPRESS: zstd compression of ASCII data ===")
    compress_ascii_files(catalog)
    return catalog


def run_normalize(catalog: Optional[Catalog] = None):
    """Stage: Normalize Tier 1-3 data to satellite baseline."""
    if catalog is None:
        catalog = Catalog()

    baseline = TIER4_REFERENCE / "combined_baseline_grid.tif"
    if not baseline.exists():
        baseline_result = build_tier4_reference_grid()
        if not baseline_result:
            log.error("No baseline available for normalization")
            return

    log.info("=== NORMALIZE: Baseline-normalizing Tier 1-3 data ===")
    normalized = 0
    for entry in catalog.entries.values():
        if entry.tier >= 4:
            continue
        if entry.format != "geotiff":
            continue

        src = HARVEST_ROOT / entry.rel_path
        if not src.exists():
            continue

        out = src.with_name(src.stem + "_baseline_normalized.tif")
        if out.exists():
            continue

        if normalize_to_baseline(src, baseline, out):
            normalized += 1

    log.info("Normalized %d GeoTIFF files", normalized)


def run_gap_fill(catalog: Optional[Catalog] = None):
    """Stage: Fill flight-line gaps with satellite data."""
    if catalog is None:
        catalog = Catalog()

    baseline = TIER4_REFERENCE / "combined_baseline_grid.tif"
    if not baseline.exists():
        log.error("No baseline available for gap filling — run 'baseline' first")
        return

    log.info("=== GAP-FILL: Interpolating flight-line gaps ===")

    for entry in catalog.entries.values():
        if entry.tier >= 4:
            continue
        if entry.format != "geotiff":
            continue

        src = HARVEST_ROOT / entry.rel_path
        if not src.exists():
            continue

        out = src.with_name(src.stem + "_gap_filled.tif")
        if out.exists():
            continue

        stats = fill_gaps_with_satellite(src, baseline, out)
        if stats["filled_pixels"] > 0:
            # Generate ML confidence layer
            mask = out.with_name(out.stem.replace("_gap_filled", "") + "_fill_mask.tif")
            conf_out = out.with_name(out.stem + "_ml_confidence.tif")
            generate_ml_confidence_layer(out, mask, entry.tier, conf_out)


def run_status():
    """Show catalog and data lake status."""
    catalog = Catalog()
    summary = catalog.summary()

    print(f"\n{'=' * 60}")
    print(f"  MAG-LAKE HARVESTER STATUS")
    print(f"{'=' * 60}")
    print(f"  Data root: {HARVEST_ROOT}")
    print(f"  Catalog:   {CATALOG_PATH}")
    print(f"  Files:     {summary['total_files']}")
    print()

    for tier, info in sorted(summary.get("tiers", {}).items()):
        tier_names = {1: "Marine/Bottom-Tow", 2: "Low-Alt Aero", 3: "Regional Aero", 4: "Satellite"}
        print(f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}):")
        print(f"    Files: {info['count']}")
        print(f"    Size:  {info['total_mb']:.1f} MB")
        print(f"    Best:  {info['best_res_m']:.0f} m resolution")
        print()

    # Check Tier 4 reference
    ref = TIER4_REFERENCE / "combined_baseline_grid.tif"
    if ref.exists():
        print(f"  Tier 4 Reference: {ref.stat().st_size / 1e6:.1f} MB (ready)")
    else:
        print(f"  Tier 4 Reference: NOT BUILT (run 'baseline')")

    # Check gaps
    gaps = catalog.find_gaps(list(_gl_bbox()), max_resolution_m=2_000)
    if gaps:
        print(f"\n  Geographic gaps (< 2km resolution): {len(gaps)} cells")
    else:
        print(f"\n  No geographic gaps detected at 2km threshold")

    print(f"{'=' * 60}\n")


def run_full_pipeline():
    """Run the full Mag-Lake Harvester pipeline."""
    catalog = run_init()
    remote = run_search()
    run_fetch(catalog, remote)
    run_baseline()
    run_compress(catalog)
    run_normalize(catalog)
    run_gap_fill(catalog)
    run_status()


# ===============================================================================
# SECTION 12: CLI ENTRY POINT
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mag-Lake Harvester: Tiered magnetic data acquisition for Great Lakes wreck hunting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init      — Scan existing data, build catalog, organize into tiers
  search    — Query NCEI ADS, USGS ScienceBase, U-Mich Deep Blue
  fetch     — Smart download (only upgrades/gap-fills)
  baseline  — Build Tier 4 satellite reference
  compress  — zstd compress processed ASCII files
  normalize — Normalize Tier 1-3 to satellite baseline
  gapfill   — Fill flight-line gaps with satellite data
  status    — Show data lake status
  run       — Full pipeline (all stages)
""",
    )
    parser.add_argument("command", choices=[
        "init", "search", "fetch", "baseline", "compress",
        "normalize", "gapfill", "status", "run",
    ], help="Pipeline command to run")
    parser.add_argument("--bbox", nargs=4, type=float, default=list(_gl_bbox()),
                        metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"),
                        help="Bounding box (default: Great Lakes)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Update module-level bbox if custom bbox provided
    _CONFIG["bbox"] = tuple(args.bbox)

    commands = {
        "init": run_init,
        "search": run_search,
        "fetch": run_fetch,
        "baseline": run_baseline,
        "compress": run_compress,
        "normalize": run_normalize,
        "gapfill": run_gap_fill,
        "status": run_status,
        "run": run_full_pipeline,
    }

    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
