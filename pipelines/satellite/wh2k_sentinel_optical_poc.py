"""
WreckHunter 2000 — Sentinel-2 Optical Wreck-Proxy POC
=======================================================
Three independent optical-band detection concepts for finding wreck candidates
on Lake Erie using Sentinel-2 L2A imagery.

Concept A — Shadow / Surface Roughness (B08 NIR, 10m)
  Wrecks that break the surface or sit in shallow water cast narrow NIR-dark
  shadows.  Even completely submerged, small hull-height anomalies create edge
  gradients in NIR backscatter detectable as linear high-gradient blobs.
  Method: Sobel gradient magnitude of B08 → cluster analysis → score by
  elongation (hull-like) and contrast vs local background.

Concept B — Zebra Mussel Clarity (B02/B03/B04, 10m)
  Zebra/quagga mussels colonise hard substrate — wrecks first, then spread.
  Dense mussel colonies dramatically clarify water locally (Secchi depth can
  triple).  Sentinel-2 clarity proxy:
     Secchi_proxy = 3.9*(B02/B04)^0.5 + 0.55   (after Lee et al. 2015)
  High-clarity spatial outliers vs the basin mean → mussel-colonised substrate
  → flag coordinates for cross-reference with wreck database.

Concept C — Post-Storm Sediment Plume (B04/B03 NDTI)
  After a storm, the lake re-clears over ~3-7 days but wrecks act as local
  sediment traps + re-suspension sites.  NDTI = (B04-B03)/(B04+B03).  Pixels
  with anomalously HIGH NDTI after surrounding water has already cleared
  indicate lingering turbidity plumes anchored to bottom features.
  Requires a pair: clear-water baseline scene + post-storm scene.

Usage
-----
  One-shot all three concepts:
    python scripts/wh2k_sentinel_optical_poc.py \\
        --concept all \\
        --output-dir wreck_hunting_ml/runs/sentinel_optical \\
        --max-cloud 20

  Single concept, custom date window:
    python scripts/wh2k_sentinel_optical_poc.py \\
        --concept zebra_clarity \\
        --date-start 2024-06-01 --date-end 2024-09-30 \\
        --output-dir wreck_hunting_ml/runs/sentinel_optical

  Use locally cached band TIFs (skip network) from a previous run:
    python scripts/wh2k_sentinel_optical_poc.py \\
        --concept all \\
        --band-cache wreck_hunting_ml/sentinel_bands \\
        --no-fetch

Data source
-----------
  Sentinel-2 L2A Cloud-Optimised GeoTIFFs (COGs) from AWS Open Data:
    STAC endpoint : https://earth-search.aws.element84.com/v1
    Collection    : sentinel-2-l2a
    AOI           : Central Lake Erie (see CENTRAL_ERIE_BBOX below)
    Bands used    : B02, B03, B04, B08
    Auth          : None required (requester-pays on S3; HTTP STAC is public)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Area of interest  ─────────────────────────────────────────────────────
# Central + Eastern Erie basin — where Phase-2 model is focused.
# Western Erie is very shallow (<10m) and prone to total cloud cover reflectance
# corruption; keep main AOI in the central/east until we have better masks.

CENTRAL_ERIE_BBOX = {
    "west":  -82.5,
    "south":  41.5,
    "east":  -78.8,
    "north":  42.9,
}

# Visual sanity — Erie extends from ~-83.5W to -78.8W, 41.4N to 42.9N
# Our strip covers ~3.7° × 1.4° = roughly 300km × 140km

STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Band names as they appear in the STAC asset keys for sentinel-2-l2a
BAND_KEYS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
}

DEFAULT_DATE_START = "2023-06-01"
DEFAULT_DATE_END   = "2024-09-30"
MAX_CLOUD_PCT      = 20.0

# ── Candidate output ─────────────────────────────────────────────────────

@dataclass
class OpticalCandidate:
    """A single potential wreck target from optical analysis."""
    lat:          float
    lon:          float
    concept:      str          # shadow_roughness | zebra_clarity | sediment_plume
    score:        float        # 0-10 optical confidence
    wreck_score:  int          # integer 0-10 after cross-ref
    scene_date:   str          # ISO date of source image
    metric:       float        # raw signal (gradient, clarity, NDTI)
    metric_zscore: float       # z-score vs local lake background
    note:         str = ""
    known_wreck_nearby: bool = False
    nearest_known_m:    float = 99999.0


# ── KML writer (mirrors wh2k_sentinel_cpu.py style) ──────────────────────

def _kml_color_by_concept(concept: str) -> tuple[str, str]:
    """Return (style_id, AABBGGRR) for a concept."""
    return {
        "shadow_roughness": ("sr", "ff0000ff"),   # red
        "zebra_clarity":    ("zc", "ff00ff00"),   # green
        "sediment_plume":   ("sp", "ff00aaff"),   # orange
    }.get(concept, ("xx", "ff00ffff"))


def write_candidates_kml(candidates: list[OpticalCandidate], out_path: Path, title: str) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f"<name>{title}</name>",
    ]
    seen_styles: set[str] = set()
    for c in candidates:
        sid, color = _kml_color_by_concept(c.concept)
        if sid not in seen_styles:
            lines += [
                f"<Style id='{sid}'>",
                f"<IconStyle><color>{color}</color><scale>1.2</scale>",
                "<Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon>",
                "</IconStyle></Style>",
            ]
            seen_styles.add(sid)

    for c in candidates:
        sid, _ = _kml_color_by_concept(c.concept)
        known_tag = " [known wreck nearby]" if c.known_wreck_nearby else ""
        desc = (
            f"Concept: {c.concept}{known_tag}<br/>"
            f"Optical score: {c.score:.2f} | Z: {c.metric_zscore:.2f} | "
            f"Metric: {c.metric:.4f}<br/>Scene: {c.scene_date}"
        )
        lines += [
            "<Placemark>",
            f"<name>{c.concept.replace('_', ' ').title()} {c.score:.1f}</name>",
            f"<description><![CDATA[{desc}]]></description>",
            f"<styleUrl>#{sid}</styleUrl>",
            "<Point>",
            f"<coordinates>{c.lon:.6f},{c.lat:.6f},0</coordinates>",
            "</Point></Placemark>",
        ]
    lines += ["</Document></kml>"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("KML written -> %s (%d placemarks)", out_path, len(candidates))


# ── Sentinel-2 STAC catalog + COG download ───────────────────────────────

def _stac_search(
    bbox: dict,
    date_start: str,
    date_end: str,
    max_cloud: float,
    limit: int = 15,
) -> list[dict]:
    """
    Query the Element84 Earth Search STAC API.
    Returns a list of STAC item dicts sorted by cloud coverage ascending.
    Uses plain urllib — no pystac_client dependency.
    """
    url = f"{STAC_URL}/collections/{S2_COLLECTION}/items"
    params = {
        "bbox": f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "limit": limit,
        "filter": f"eo:cloud_cover <= {max_cloud}",
        "filter-lang": "cql2-text",
    }
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())

    try:
        with urllib.request.urlopen(f"{url}?{query}", timeout=20) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("features", [])
        items.sort(key=lambda x: x.get("properties", {}).get("eo:cloud_cover", 100))
        logger.info("STAC returned %d scenes (<=%.0f%% cloud)", len(items), max_cloud)
        return items
    except Exception as exc:
        logger.warning("STAC query failed: %s", exc)
        return []


def _asset_href(item: dict, band: str) -> Optional[str]:
    """Extract the download URL for a given band from a STAC item."""
    assets = item.get("assets", {})
    # Element84 STAC uses asset keys like "B02", "B03", etc.
    asset = assets.get(band) or assets.get(band.lower())
    if asset:
        return asset.get("href")
    # Fallback: search by band common name
    common = BAND_KEYS.get(band, "").lower()
    for _key, val in assets.items():
        if val.get("eo:bands"):
            for b in val["eo:bands"]:
                if b.get("common_name", "").lower() == common:
                    return val.get("href")
    return None


def _download_band_window(
    href: str,
    bbox: dict,
    cache_path: Path,
    timeout: int = 60,
) -> Optional[np.ndarray]:
    """
    Download a windowed subset of a COG band using rasterio.
    The URL must be publicly accessible (Sentinel-2 Open Data on AWS is public
    via HTTPS; no credentials needed for STAC metadata, though S3://URLs
    require requester-pays which we avoid by using the HTTPS link).
    """
    if cache_path.exists():
        logger.info("  Band cache hit: %s", cache_path.name)
        try:
            arr = np.load(str(cache_path))
            if arr.size > 100:
                return arr.astype(np.float32)
            else:
                logger.info("  Cache entry is empty (shape %s) — re-downloading", arr.shape)
        except Exception as exc:
            logger.warning("  Cache read failed (%s) — re-downloading", exc)

    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        # rasterio can open remote HTTPS COGs natively
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          CPL_VSIL_CURL_CACHE_SIZE=20000000):
            with rasterio.open(href) as src:
                # Sentinel-2 COGs are in UTM (not EPSG:4326); reproject bbox
                src_crs = src.crs
                wgs84 = CRS.from_epsg(4326)
                if src_crs != wgs84:
                    left, bottom, right, top = transform_bounds(
                        wgs84, src_crs,
                        bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    )
                else:
                    left, bottom, right, top = (
                        bbox["west"], bbox["south"], bbox["east"], bbox["north"]
                    )
                win = from_bounds(left, bottom, right, top, transform=src.transform)
                data = src.read(1, window=win).astype(np.float32)
                # Sentinel-2 L2A surface reflectance is stored as DN/10000
                data = np.where(data == 0, np.nan, data / 10000.0)

        # Cache locally
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), data)
        logger.info("  Downloaded %s -> %s (shape %s)", href.split("/")[-1], cache_path.name, data.shape)
        return data

    except Exception as exc:
        logger.warning("  COG download failed for %s: %s", href, exc)
        return None


# ── Spatial helpers ───────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dLon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample by integer factor (ignores NaN)."""
    rows, cols = arr.shape
    rows2 = (rows // factor) * factor
    cols2 = (cols // factor) * factor
    trimmed = arr[:rows2, :cols2]
    reshaped = trimmed.reshape(rows2 // factor, factor, cols2 // factor, factor)
    return np.nanmean(reshaped, axis=(1, 3))


def _pixel_coords(arr: np.ndarray, bbox: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) arrays matching arr shape."""
    rows, cols = arr.shape
    lats = np.linspace(bbox["north"], bbox["south"], rows)
    lons = np.linspace(bbox["west"],  bbox["east"],  cols)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def _masked_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score computation ignoring NaN and water-land mask."""
    flat = arr[np.isfinite(arr)]
    if flat.size < 100:
        return np.zeros_like(arr)
    mu, sigma = float(np.nanmean(flat)), float(np.nanstd(flat))
    if sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def _find_peak_clusters(
    score_map: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    zscore_threshold: float = 2.5,
    min_separation_px: int = 8,
    max_candidates: int = 20,
) -> list[tuple[float, float, float, float]]:
    """
    Simple non-max suppression peak finder.
    Returns list of (lat, lon, score_raw, zscore).
    """
    from scipy.ndimage import maximum_filter, label

    zs = _masked_zscore(score_map)
    thresh_mask = (zs >= zscore_threshold) & np.isfinite(score_map)
    if not thresh_mask.any():
        return []

    # Local maximum filter
    local_max = (score_map == maximum_filter(score_map, size=min_separation_px))
    peaks_mask = thresh_mask & local_max
    rows, cols = np.where(peaks_mask)

    candidates = []
    for r, c in zip(rows, cols):
        candidates.append((
            float(lat_grid[r, c]),
            float(lon_grid[r, c]),
            float(score_map[r, c]),
            float(zs[r, c]),
        ))

    # Sort by z-score descending, keep top N
    candidates.sort(key=lambda x: -x[3])
    return candidates[:max_candidates]


# ── Loading known wrecks for cross-reference ─────────────────────────────

def _load_known_wrecks() -> list[dict]:
    """Try importing from the inference scorer's correlate module."""
    try:
        from scripts.wh2k_inference_scorer import _load_known_sites
        sites = _load_known_sites()
        return [
            {"lat": float(s.lat), "lon": float(s.lon)}
            for s in sites
            if hasattr(s, "lat") and hasattr(s, "lon")
        ]
    except Exception:
        pass

    # Fallback: look for wreck CSV directly
    candidates_paths = [
        REPO_ROOT / "wreck_hunting_ml" / "wreck_db" / "lake_erie_wrecks.csv",
        REPO_ROOT / "magnetic_data" / "wrecks" / "lake_erie_wrecks.csv",
    ]
    for p in candidates_paths:
        if p.exists():
            import csv
            wrecks = []
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        wrecks.append({"lat": float(row["lat"]), "lon": float(row["lon"])})
                    except Exception:
                        pass
            if wrecks:
                logger.info("Loaded %d known wrecks from %s", len(wrecks), p.name)
                return wrecks

    # Fallback: load from wrecks.db
    try:
        import sqlite3 as _sqlite3
        _db_path = REPO_ROOT / "db" / "wrecks.db"
        _conn = _sqlite3.connect(str(_db_path))
        _rows = _conn.execute(
            "SELECT name, latitude, longitude FROM features "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
            "  AND latitude  BETWEEN 41.2 AND 43.0 "
            "  AND longitude BETWEEN -84.0 AND -78.5"
            "  AND coord_quality = 'dive_verified'"
        ).fetchall()
        _conn.close()
        if _rows:
            logger.info("Loaded %d Lake Erie wrecks from wrecks.db", len(_rows))
            return [{"name": r[0], "lat": float(r[1]), "lon": float(r[2])} for r in _rows]
    except Exception as _exc:
        logger.debug("wrecks.db load failed: %s", _exc)

    logger.warning("No known wreck database found — cross-reference disabled")
    return []


def _cross_reference(
    candidates: list[OpticalCandidate],
    known_wrecks: list[dict],
    nearby_radius_m: float = 2000.0,
) -> list[OpticalCandidate]:
    for c in candidates:
        best = min(
            (_haversine_m(c.lat, c.lon, w["lat"], w["lon"]) for w in known_wrecks),
            default=99999.0,
        )
        c.nearest_known_m = best
        c.known_wreck_nearby = best <= nearby_radius_m
    return candidates


# ── Concept A: Shadow / Surface Roughness ────────────────────────────────

def _concept_shadow_roughness(
    band_arrays: dict[str, np.ndarray],
    bbox: dict,
    scene_date: str,
) -> list[OpticalCandidate]:
    """
    Detect wreck-scale NIR roughness anomalies.

    Strategy:
    1. Compute Sobel gradient magnitude of B08 (NIR, 10m).
    2. Apply a 15-pixel median background subtraction to remove broad
       land/water boundaries — we want small-scale structure.
    3. Z-score the residual roughness map.
    4. Find peaks > 2.5σ; compute elongation from local connected region
       (hull-like = elongated, aspect ratio 2-10).
    5. Score = f(z_score, aspect_ratio).
    """
    from scipy.ndimage import sobel, uniform_filter

    b08 = band_arrays.get("B08")
    if b08 is None or b08.size < 100:
        logger.warning("Concept A: B08 band missing or too small")
        return []

    # Downsample large tiles to ~2000px on longest axis for manageable compute
    # At 10m native, factor=5 gives 50m effective resolution (still resolves large wrecks)
    max_dim = 2000
    factor = max(1, max(b08.shape) // max_dim)
    if factor > 1:
        b08 = _downsample(b08, factor)
        logger.info("Concept A: downsampled B08 by %dx to %s for gradient analysis", factor, b08.shape)

    logger.info("Concept A: computing NIR gradient roughness (%d x %d px)...", *b08.shape)

    # Replace NaN with local median for gradient computation
    b08_clean = np.where(np.isfinite(b08), b08, 0.0)

    # Sobel gradient magnitude
    Gx = sobel(b08_clean, axis=1)
    Gy = sobel(b08_clean, axis=0)
    grad_mag = np.hypot(Gx, Gy)

    # Subtract broad background using uniform_filter (fast; same concept as median for bg)
    bg = uniform_filter(grad_mag, size=max(3, 30 // factor))
    residual = grad_mag - bg
    residual = np.where(np.isfinite(b08), residual, np.nan)

    lat_grid, lon_grid = _pixel_coords(residual, bbox)

    peaks = _find_peak_clusters(
        residual, lat_grid, lon_grid,
        zscore_threshold=2.5,
        min_separation_px=15,
        max_candidates=25,
    )

    candidates = []
    for lat, lon, metric, zsc in peaks:
        # Optical score 0-10 based on z-score (cap at 5σ = score 10)
        score = min(10.0, (zsc / 5.0) * 10.0)
        wreck_score = int(min(10, max(0, round(score * 0.8))))   # slight discount for proxy
        candidates.append(OpticalCandidate(
            lat=lat, lon=lon,
            concept="shadow_roughness",
            score=round(score, 2),
            wreck_score=wreck_score,
            scene_date=scene_date,
            metric=round(metric, 6),
            metric_zscore=round(zsc, 3),
            note="NIR Sobel roughness anomaly",
        ))

    logger.info("Concept A: %d shadow/roughness candidates", len(candidates))
    return candidates


# ── Concept B: Zebra Mussel Clarity ──────────────────────────────────────

def _concept_zebra_clarity(
    band_arrays: dict[str, np.ndarray],
    bbox: dict,
    scene_date: str,
) -> list[OpticalCandidate]:
    """
    Detect spatial outliers in water clarity as a proxy for mussel colonisation.

    Clarity proxy:
      Secchi_proxy = 3.9 * sqrt(B02/B04) + 0.55   (m, valid over open water)

    High local Secchi outlier vs basin mean = anomalously clear patch
    = likely mussel-colonised hard substrate = potential wreck.
    """
    from scipy.ndimage import uniform_filter

    b02 = band_arrays.get("B02")
    b04 = band_arrays.get("B04")
    if b02 is None or b04 is None:
        logger.warning("Concept B: B02 or B04 missing")
        return []

    # Downsample large tiles
    max_dim = 2000
    factor = max(1, max(b02.shape) // max_dim)
    if factor > 1:
        b02 = _downsample(b02, factor)
        b04 = _downsample(b04, factor)

    logger.info("Concept B: computing Secchi clarity proxy (%d x %d px)...", *b02.shape)

    # Safe divide; mask out land (very high reflectance) and clouds (B04>0.3)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (b04 > 0.005) & (b04 < 0.3) & np.isfinite(b02) & np.isfinite(b04),
            b02 / b04,
            np.nan,
        )
        secchi = np.where(np.isfinite(ratio), 3.9 * np.sqrt(ratio) + 0.55, np.nan)

    # Background: uniform filter with 50-pixel (~500m) window
    secchi_clean = np.where(np.isfinite(secchi), secchi, 0.0)
    bg = uniform_filter(secchi_clean, size=50)
    valid = np.isfinite(secchi)
    bg_valid = uniform_filter(valid.astype(float), size=50)
    bg_mean = np.where(bg_valid > 0.1, bg / np.maximum(bg_valid, 1e-6), np.nan)

    # Residual: how much clearer than local 500m average
    residual = np.where(valid, secchi - bg_mean, np.nan)

    lat_grid, lon_grid = _pixel_coords(residual, bbox)

    peaks = _find_peak_clusters(
        residual, lat_grid, lon_grid,
        zscore_threshold=2.5,
        min_separation_px=10,
        max_candidates=25,
    )

    candidates = []
    for lat, lon, metric, zsc in peaks:
        score = min(10.0, (zsc / 5.0) * 10.0)
        wreck_score = int(min(10, max(0, round(score * 0.85))))
        candidates.append(OpticalCandidate(
            lat=lat, lon=lon,
            concept="zebra_clarity",
            score=round(score, 2),
            wreck_score=wreck_score,
            scene_date=scene_date,
            metric=round(metric, 4),
            metric_zscore=round(zsc, 3),
            note="Secchi clarity anomaly — possible zebra mussel colonisation",
        ))

    logger.info("Concept B: %d clarity-anomaly candidates", len(candidates))
    return candidates


# ── Concept C: Post-Storm Sediment Plume ─────────────────────────────────

def _concept_sediment_plume(
    band_arrays: dict[str, np.ndarray],
    bbox: dict,
    scene_date: str,
    baseline_arrays: Optional[dict[str, np.ndarray]] = None,
) -> list[OpticalCandidate]:
    """
    Detect persistent turbidity after storm by NDTI.
    NDTI = (B04 - B03) / (B04 + B03)
    High NDTI = high turbidity / sediment load.

    If a baseline (clear-water) image is provided, compute delta NDTI.
    Without baseline, detect absolute NDTI outliers (less sensitive but works).
    """
    from scipy.ndimage import uniform_filter

    b03 = band_arrays.get("B03")
    b04 = band_arrays.get("B04")
    if b03 is None or b04 is None:
        logger.warning("Concept C: B03 or B04 missing")
        return []

    # Downsample large tiles
    max_dim = 2000
    factor = max(1, max(b03.shape) // max_dim)
    if factor > 1:
        b03 = _downsample(b03, factor)
        b04 = _downsample(b04, factor)

    logger.info("Concept C: computing NDTI sediment proxy (%d x %d px)...", *b03.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = b04 + b03
        ndti = np.where(
            (denom > 0.005) & (b04 < 0.3) & np.isfinite(b03) & np.isfinite(b04),
            (b04 - b03) / denom,
            np.nan,
        )

    if baseline_arrays is not None:
        b03_base = baseline_arrays.get("B03")
        b04_base = baseline_arrays.get("B04")
        if b03_base is not None and b04_base is not None:
            if factor > 1 and b03_base.shape != b03.shape:
                base_factor = max(1, max(b03_base.shape) // max_dim)
                b03_base = _downsample(b03_base, base_factor)
                b04_base = _downsample(b04_base, base_factor)
            if b03_base.shape != b03.shape:
                logger.warning(
                    "Concept C: storm shape %s != baseline shape %s — skipping delta-NDTI",
                    b03.shape, b03_base.shape,
                )
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom_b = b04_base + b03_base
                    ndti_base = np.where(
                        (denom_b > 0.005) & np.isfinite(b03_base) & np.isfinite(b04_base),
                        (b04_base - b03_base) / denom_b,
                        np.nan,
                    )
                # Delta NDTI: positive = MORE turbid than baseline
                ndti = np.where(np.isfinite(ndti) & np.isfinite(ndti_base),
                                ndti - ndti_base, np.nan)
                logger.info("  Using delta-NDTI (post-storm minus baseline)")
        else:
            logger.info("  Using absolute NDTI (no baseline B03/B04)")
    else:
        logger.info("  Using absolute NDTI (no baseline provided)")

    lat_grid, lon_grid = _pixel_coords(ndti, bbox)

    peaks = _find_peak_clusters(
        ndti, lat_grid, lon_grid,
        zscore_threshold=2.5,
        min_separation_px=10,
        max_candidates=25,
    )

    candidates = []
    for lat, lon, metric, zsc in peaks:
        score = min(10.0, (zsc / 5.0) * 10.0)
        wreck_score = int(min(10, max(0, round(score * 0.8))))
        candidates.append(OpticalCandidate(
            lat=lat, lon=lon,
            concept="sediment_plume",
            score=round(score, 2),
            wreck_score=wreck_score,
            scene_date=scene_date,
            metric=round(float(metric), 6),
            metric_zscore=round(zsc, 3),
            note="NDTI turbidity anomaly — possible post-storm sediment plume anchor",
        ))

    logger.info("Concept C: %d sediment-plume candidates", len(candidates))
    return candidates


# ── Scene acquisition ────────────────────────────────────────────────────

def _fetch_best_scene(
    bbox: dict,
    date_start: str,
    date_end: str,
    max_cloud: float,
    band_cache_dir: Path,
    no_fetch: bool,
    label: str = "scene",
) -> tuple[dict[str, np.ndarray], str]:
    """
    Download the lowest-cloud-cover S2 scene for the given date window.
    Returns (band_arrays dict, scene_date ISO string).
    """
    bands_needed = ["B02", "B03", "B04", "B08"]
    band_arrays: dict[str, np.ndarray] = {}
    scene_date = "unknown"

    if no_fetch:
        logger.info("--no-fetch: looking for cached band files for %s", label)
        for band in bands_needed:
            p = band_cache_dir / f"{label}_{band}.npy"
            if p.exists():
                band_arrays[band] = np.load(str(p))
                logger.info("  Loaded %s from cache", p.name)
        meta_p = band_cache_dir / f"{label}_meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            scene_date = meta.get("scene_date", "unknown")
        return band_arrays, scene_date

    items = _stac_search(bbox, date_start, date_end, max_cloud)
    if not items:
        logger.warning("No scenes found for %s in date range %s — %s", label, date_start, date_end)
        return band_arrays, scene_date

    # Pick the least-cloudy item
    item = items[0]
    scene_date = item.get("properties", {}).get("datetime", "")[:10]
    cloud = item.get("properties", {}).get("eo:cloud_cover", "?")
    logger.info("Using scene %s (date=%s, cloud=%.1f%%)", item.get("id", "?"), scene_date, float(cloud or 0))

    for band in bands_needed:
        href = _asset_href(item, band)
        if href is None:
            logger.warning("  No href for band %s in scene %s", band, item.get("id"))
            continue
        cache_p = band_cache_dir / f"{label}_{band}.npy"
        arr = _download_band_window(href, bbox, cache_p)
        if arr is not None:
            band_arrays[band] = arr

    # Save scene metadata
    band_cache_dir.mkdir(parents=True, exist_ok=True)
    meta_p = band_cache_dir / f"{label}_meta.json"
    meta_p.write_text(json.dumps({
        "scene_id":   item.get("id"),
        "scene_date": scene_date,
        "cloud_pct":  cloud,
        "stac_item":  item.get("links", []),
    }, indent=2), encoding="utf-8")

    return band_arrays, scene_date


# ── Main entry ────────────────────────────────────────────────────────────

CONCEPT_CHOICES = ["shadow_roughness", "zebra_clarity", "sediment_plume", "all"]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sentinel-2 optical wreck-proxy POC (3 concepts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concept", default="all", choices=CONCEPT_CHOICES,
                        help="Which concept(s) to run")
    parser.add_argument("--output-dir", default="wreck_hunting_ml/runs/sentinel_optical",
                        help="Directory for KML/JSON output")
    parser.add_argument("--band-cache",
                        default="wreck_hunting_ml/sentinel_bands",
                        help="Local directory for caching downloaded S2 band windows")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip network fetch; use only cached band files")
    parser.add_argument("--max-cloud", type=float, default=MAX_CLOUD_PCT,
                        help="Max cloud cover percent for scene selection")
    parser.add_argument("--date-start", default=DEFAULT_DATE_START,
                        help="Start of search window (YYYY-MM-DD)")
    parser.add_argument("--date-end", default=DEFAULT_DATE_END,
                        help="End of search window (YYYY-MM-DD)")
    parser.add_argument("--storm-date-start", default="2024-01-13",
                        help="Post-storm scene start for sediment plume concept")
    parser.add_argument("--storm-date-end", default="2024-01-20",
                        help="Post-storm scene end for sediment plume concept")
    parser.add_argument("--min-score", type=int, default=5,
                        help="Minimum integer wreck_score to include in output")
    parser.add_argument("--zscore-threshold", type=float, default=2.5,
                        help="Z-score threshold for peak detection")
    parser.add_argument("--bbox-west",  type=float, default=CENTRAL_ERIE_BBOX["west"])
    parser.add_argument("--bbox-south", type=float, default=CENTRAL_ERIE_BBOX["south"])
    parser.add_argument("--bbox-east",  type=float, default=CENTRAL_ERIE_BBOX["east"])
    parser.add_argument("--bbox-north", type=float, default=CENTRAL_ERIE_BBOX["north"])
    args = parser.parse_args(argv)

    bbox = {
        "west":  args.bbox_west,
        "south": args.bbox_south,
        "east":  args.bbox_east,
        "north": args.bbox_north,
    }

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    band_cache = REPO_ROOT / args.band_cache
    band_cache.mkdir(parents=True, exist_ok=True)

    logger.info("==========================================================")
    logger.info("WH2K Sentinel-2 Optical Wreck Proxy POC")
    logger.info("Concept(s): %s", args.concept)
    logger.info("AOI: %.2fW %.2fN — %.2fW %.2fN", bbox["west"], bbox["north"],
                bbox["east"], bbox["south"])
    logger.info("Date window: %s — %s", args.date_start, args.date_end)
    logger.info("==========================================================")

    # ── Acquire clear-water scene ─────────────────────────────────────────
    clear_bands, clear_date = _fetch_best_scene(
        bbox=bbox,
        date_start=args.date_start,
        date_end=args.date_end,
        max_cloud=args.max_cloud,
        band_cache_dir=band_cache,
        no_fetch=args.no_fetch,
        label="clear",
    )

    # ── For sediment plume: also try to grab a post-storm scene ──────────
    storm_bands, storm_date = {}, "unknown"
    if args.concept in ("sediment_plume", "all"):
        storm_bands, storm_date = _fetch_best_scene(
            bbox=bbox,
            date_start=args.storm_date_start,
            date_end=args.storm_date_end,
            max_cloud=40.0,   # relaxed cloud tolerance for post-storm
            band_cache_dir=band_cache,
            no_fetch=args.no_fetch,
            label="storm",
        )

    known_wrecks = _load_known_wrecks()

    all_candidates: list[OpticalCandidate] = []
    concepts_run: list[str] = (
        ["shadow_roughness", "zebra_clarity", "sediment_plume"]
        if args.concept == "all"
        else [args.concept]
    )

    for concept in concepts_run:
        logger.info("----- Running concept: %s -----", concept)

        if concept == "shadow_roughness":
            cands = _concept_shadow_roughness(clear_bands, bbox, clear_date)
        elif concept == "zebra_clarity":
            cands = _concept_zebra_clarity(clear_bands, bbox, clear_date)
        elif concept == "sediment_plume":
            # Use storm_bands as primary + clear_bands as baseline
            primary = storm_bands if storm_bands else clear_bands
            baseline = clear_bands if storm_bands else None
            cands = _concept_sediment_plume(primary, bbox, storm_date, baseline)
        else:
            cands = []

        # Cross-reference with known wreck DB
        if known_wrecks:
            cands = _cross_reference(cands, known_wrecks)

        # Filter by min score
        cands = [c for c in cands if c.wreck_score >= args.min_score]

        logger.info("  -> %d candidates (score >= %d)", len(cands), args.min_score)
        all_candidates.extend(cands)

        # Per-concept KML + JSON
        stem = f"optical_{concept}"
        write_candidates_kml(cands, out_dir / f"{stem}.kml",
                              title=f"WH2K Optical — {concept.replace('_',' ').title()}")
        (out_dir / f"{stem}.json").write_text(
            json.dumps([asdict(c) for c in cands], indent=2),
            encoding="utf-8",
        )

    # ── Combined output ───────────────────────────────────────────────────
    all_candidates.sort(key=lambda c: -c.score)

    write_candidates_kml(
        all_candidates,
        out_dir / "optical_all_concepts.kml",
        title="WH2K Optical — All Concepts",
    )
    (out_dir / "optical_all_concepts.json").write_text(
        json.dumps([asdict(c) for c in all_candidates], indent=2),
        encoding="utf-8",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("==========================================================")
    logger.info("OPTICAL POC SUMMARY")
    logger.info("==========================================================")
    for concept in concepts_run:
        n = sum(1 for c in all_candidates if c.concept == concept)
        logger.info("  %-25s : %d candidates", concept, n)
    logger.info("  %-25s : %d total", "ALL", len(all_candidates))

    new_unknowns = [c for c in all_candidates if not c.known_wreck_nearby]
    logger.info("  %-25s : %d (no known wreck within 2km)", "New unknowns", len(new_unknowns))
    logger.info("Output -> %s", out_dir)
    logger.info("==========================================================")

    if not clear_bands:
        logger.warning(
            "\n  NOTE: No S2 imagery was downloaded or cached.\n"
            "  Results above are empty because no band data was available.\n\n"
            "  To acquire imagery, run WITHOUT --no-fetch and ensure internet access.\n"
            "  Sentinel-2 L2A data is queried from the public Element84 STAC:\n"
            "    %s\n"
            "  Band windows (~10-50 MB each) will be cached to:\n"
            "    %s\n"
            "  You can also manually place Sentinel-2 L2A band GeoTIFF files\n"
            "  for the Erie AOI as:\n"
            "    %s/clear_B02.tif  (band 2 / Blue)\n"
            "    %s/clear_B03.tif  (band 3 / Green)\n"
            "    %s/clear_B04.tif  (band 4 / Red)\n"
            "    %s/clear_B08.tif  (band 8 / NIR)\n"
            "  Then re-run with --no-fetch.",
            STAC_URL, band_cache,
            band_cache, band_cache, band_cache, band_cache,
        )


if __name__ == "__main__":
    main()
