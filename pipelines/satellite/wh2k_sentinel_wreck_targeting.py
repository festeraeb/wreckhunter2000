"""
wh2k_sentinel_wreck_targeting.py
=================================
Season-aware Sentinel-2 wreck signal extractor.

Targets known wreck coordinates from wrecks.db and evaluates three optical
concepts at those exact locations across seasonally-appropriate image archives.

Concepts
--------
shadow_roughness  (SAR proxy via NIR roughness)
  Best window: March–April (early spring), ice-free, some current differential.
  Wrecks create surface roughness / shadow patterns in NIR due to thermal and
  current contrasts directly above the structure.

zebra_clarity     (Mussel clarity proxy via Secchi depth ratio)
  Best window: July–October (peak mussel filtering activity), low current, clear.
  Zebra/quagga mussels colonise hard substrate → anomalously clear patch above
  the wreck relative to surrounding turbid lake water.

sediment_plume    (Post-storm NDTI plume anchor)
  Best window: April–October, targeted at post-storm scenes (any cloud-relaxed
  scene within 1-7 days of elevated turbidity indicators).
  Wrecks act as physical anchors for sediment plumes; turbidity re-suspension
  lingers longer directly above the structure.

Validation logic
----------------
For each wreck × concept × season, the script:
  1. Fetches all STAC scenes in the target season window (multi-year archive).
  2. Extracts a 600 m radius chip centred on the wreck coordinate.
  3. Computes the concept metric on the chip centre (+/- 100 m) vs an annular
     background (300–1200 m radius).
  4. Assigns a z-score: how many σ above/below the local background.
  5. Records hit_rate = fraction of qualifying scenes with |z| > 1.5 in the
     expected direction (dark for SAR, clear for clarity, turbid for plume).

Output
------
  wreck_hunting_ml/runs/wreck_targeting/
    wreck_targets_<concept>.csv      per-concept per-wreck scored table
    wreck_targets_all.csv            combined flat CSV
    wreck_targets_all.kml            KML with all wrecks coloured by best score

Usage
-----
  python scripts/wh2k_sentinel_wreck_targeting.py [--concepts all] \\
         [--max-wrecks 50] [--max-scenes 8] [--min-score 4]

Notes
-----
  Requires: requests, rasterio, numpy, scipy
  Internet access needed for STAC search + COG band downloads.
  Band chips are cached in wreck_hunting_ml/sentinel_bands/chips/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wreck_targeting")

# ── STAC configuration ────────────────────────────────────────────────────────

STAC_URL      = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

ERIE_BBOX = {"west": -83.50, "south": 41.30, "east": -78.85, "north": 42.95}

# Concept-specific season windows.
# All dates combined with the multi-year archive (2017–present).
CONCEPT_SEASONS: dict[str, dict] = {
    "shadow_roughness": {
        "months":      [3, 4],                  # March–April: spring current contrast
        "max_cloud":   15.0,
        "archive_start": "2017-03-01",
        "archive_end":   "2026-04-30",
        "signal_direction": "dark",              # darker / lower NIR above wreck
        "note": "Spring current differential creates NIR roughness / shadow above structure",
    },
    "zebra_clarity": {
        "months":      [7, 8, 9, 10],           # July–October: peak mussel filtering
        "max_cloud":   10.0,
        "archive_start": "2017-07-01",
        "archive_end":   "2026-10-31",
        "signal_direction": "bright",            # clearer water → higher Secchi
        "note": "Zebra/quagga mussels on wreck → anomalously clear patch",
    },
    "sediment_plume": {
        "months":      [4, 5, 6, 7, 8, 9, 10],  # April–Oct: post-storm visible
        "max_cloud":   25.0,
        "archive_start": "2017-04-01",
        "archive_end":   "2026-10-31",
        "signal_direction": "turbid",            # higher NDTI above wreck anchor
        "note": "Wreck traps sediment; plumes re-suspend longer above structure",
    },
}

# Chip geometry
CHIP_SIGNAL_M   = 150    # inner radius for signal measurement (m)
CHIP_BG_INNER_M = 350    # background annulus inner radius (m)
CHIP_BG_OUTER_M = 1200   # background annulus outer radius (m)
DEG_PER_M_LAT   = 1.0 / 111_325.0  # degrees latitude per metre
HIT_ZSCORE_MIN  = 1.5    # z-score threshold to count a scene as a "hit"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WreckTarget:
    wreck_id:     str
    wreck_name:   str
    lat:          float
    lon:          float
    depth_m:      float
    concept:      str
    n_scenes:     int = 0
    n_hits:       int = 0
    hit_rate:     float = 0.0
    mean_zscore:  float = 0.0
    best_zscore:  float = 0.0
    best_date:    str = ""
    score:        float = 0.0   # 0-10 overall
    notes:        str = ""


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _chip_bbox(lat: float, lon: float, radius_m: float) -> dict:
    """Return a bbox dict that encloses a circle of radius_m around (lat, lon)."""
    dlat = radius_m * DEG_PER_M_LAT
    dlon = radius_m * DEG_PER_M_LAT / max(np.cos(np.radians(lat)), 1e-6)
    return {
        "west":  lon - dlon,
        "east":  lon + dlon,
        "south": lat - dlat,
        "north": lat + dlat,
    }


# ── STAC helpers ──────────────────────────────────────────────────────────────

def _stac_search(
    bbox: dict,
    date_start: str,
    date_end: str,
    max_cloud: float,
    months: list[int],
    max_items: int = 100,
) -> list[dict]:
    """
    Search Element84 Earth Search STAC for Sentinel-2 L2A scenes.

    Mirrors the approach in `wh2k_sentinel_optical_poc.py`:
    - Uses stdlib urllib (no requests dependency)
    - CQL2-text cloud filter passed as query param
    - Datetime as RFC-3339 range with T timestamps
    Then post-filters by target months.
    Returns STAC feature dicts sorted by ascending cloud cover.
    """
    import urllib.parse
    import urllib.request

    url = f"{STAC_URL}/collections/{S2_COLLECTION}/items"
    params: dict = {
        "bbox":        f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "datetime":    f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "limit":       max_items,
        "filter":      f"eo:cloud_cover <= {max_cloud}",
        "filter-lang": "cql2-text",
    }
    query = "&".join(
        f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items()
    )
    full_url = f"{url}?{query}"

    try:
        with urllib.request.urlopen(full_url, timeout=25) as resp:
            data = json.loads(resp.read().decode())
        raw = data.get("features", [])
    except Exception as exc:
        log.warning("STAC search failed: %s", exc)
        return []

    items: list[dict] = []
    for item in raw:
        props = item.get("properties", {})
        cloud = props.get("eo:cloud_cover", 100.0)
        try:
            cloud = float(cloud)
        except (TypeError, ValueError):
            cloud = 100.0
        # Month filter
        dt_str = props.get("datetime", "")[:10]
        try:
            month = int(dt_str[5:7])
        except (ValueError, IndexError):
            continue
        if month not in months:
            continue
        item["_cloud"] = cloud
        items.append(item)

    items.sort(key=lambda i: i["_cloud"])
    return items


def _asset_href(item: dict, band: str) -> Optional[str]:
    assets = item.get("assets", {})
    # Try exact band key, then common aliases
    for key in (band, band.lower(), f"B0{band[-1]}", f"b0{band[-1]}"):
        if key in assets:
            href = assets[key].get("href", "")
            if href:
                return href
    return None


def _download_band_chip(
    href: str,
    bbox: dict,
    cache_path: Path,
) -> Optional[np.ndarray]:
    """
    Download a COG window for bbox and return as float32 ndarray.
    Arrays are cached as .npy files.
    """
    if cache_path.exists():
        return np.load(str(cache_path)).astype(np.float32)

    try:
        import rasterio  # type: ignore
        from rasterio.crs import CRS  # type: ignore
        from rasterio.warp import transform_bounds  # type: ignore
    except ImportError:
        log.error("rasterio not installed")
        return None

    try:
        with rasterio.open(href) as src:
            dst_crs = CRS.from_epsg(4326)
            if src.crs and src.crs != dst_crs:
                west, south, east, north = transform_bounds(
                    dst_crs, src.crs,
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                )
            else:
                west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

            from rasterio.windows import from_bounds  # type: ignore
            win = from_bounds(west, south, east, north, src.transform)
            if win.width < 1 or win.height < 1:
                return None
            arr = src.read(1, window=win).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if np.nanmax(arr) > 1e4:          # DN → reflectance
                arr = arr / 10_000.0
            arr[arr <= 0] = np.nan
    except Exception as exc:
        log.debug("Band download failed (%s): %s", href[:60], exc)
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), arr)
    return arr


# ── Known wreck loader ────────────────────────────────────────────────────────

def _load_known_wrecks(max_wrecks: int) -> list[dict]:
    db_path = REPO_ROOT / "db" / "wrecks.db"
    wrecks = []
    if not db_path.exists():
        log.warning("wrecks.db not found at %s", db_path)
        return wrecks
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, latitude AS lat, longitude AS lon,
                   COALESCE(depth, 0) AS depth_m
            FROM features
            WHERE latitude  BETWEEN :s AND :n
              AND longitude BETWEEN :w AND :e
              AND name IS NOT NULL
              AND name != ''
              AND coord_quality = 'dive_verified'
            ORDER BY RANDOM()
            LIMIT :lim
        """, {
            "s": ERIE_BBOX["south"], "n": ERIE_BBOX["north"],
            "w": ERIE_BBOX["west"],  "e": ERIE_BBOX["east"],
            "lim": max_wrecks,
        })
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                depth = float(row["depth_m"] or 0)
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            wrecks.append({
                "id": str(row["id"]),
                "name": str(row["name"]),
                "lat": lat,
                "lon": lon,
                "depth_m": depth,
            })
        log.info("Loaded %d known wrecks from wrecks.db", len(wrecks))
    except Exception as exc:
        log.error("Failed to load wrecks: %s", exc)
    return wrecks


# ── Concept metric extractors ─────────────────────────────────────────────────

def _extract_sar_metric(
    chip_bands: dict[str, np.ndarray],
) -> Optional[float]:
    """
    Concept A: NIR roughness proxy.
    Returns mean Sobel gradient magnitude of B08 in the central chip region.
    Lower value (smoother/flatter = shadow) → dark signal direction.
    """
    from scipy.ndimage import sobel  # type: ignore

    b08 = chip_bands.get("B08")
    if b08 is None or b08.size < 4:
        return None
    b08_c = np.where(np.isfinite(b08), b08, 0.0)
    Gx = sobel(b08_c, axis=1)
    Gy = sobel(b08_c, axis=0)
    grad = np.hypot(Gx, Gy)
    valid = np.isfinite(b08)
    if not valid.any():
        return None
    return float(np.nanmean(grad[valid]))


def _extract_clarity_metric(
    chip_bands: dict[str, np.ndarray],
) -> Optional[float]:
    """
    Concept B: Secchi depth proxy = 3.9 * sqrt(B02/B04) + 0.55 (m).
    Returns mean Secchi within chip; higher = clearer.
    """
    b02 = chip_bands.get("B02")
    b04 = chip_bands.get("B04")
    if b02 is None or b04 is None:
        return None
    valid = (b04 > 0.005) & (b04 < 0.3) & np.isfinite(b02) & np.isfinite(b04)
    if not valid.any():
        return None
    ratio = b02[valid] / b04[valid]
    secchi = 3.9 * np.sqrt(np.maximum(ratio, 0.0)) + 0.55
    return float(np.nanmean(secchi))


def _extract_plume_metric(
    chip_bands: dict[str, np.ndarray],
) -> Optional[float]:
    """
    Concept C: NDTI = (B04 - B03) / (B04 + B03).
    Returns mean NDTI within chip; higher = more turbid.
    """
    b03 = chip_bands.get("B03")
    b04 = chip_bands.get("B04")
    if b03 is None or b04 is None:
        return None
    denom = b04 + b03
    valid = (denom > 0.005) & (b04 < 0.3) & np.isfinite(b03) & np.isfinite(b04)
    if not valid.any():
        return None
    ndti = (b04[valid] - b03[valid]) / denom[valid]
    return float(np.nanmean(ndti))


CONCEPT_METRIC_FN = {
    "shadow_roughness": _extract_sar_metric,
    "zebra_clarity":    _extract_clarity_metric,
    "sediment_plume":   _extract_plume_metric,
}

CONCEPT_BANDS: dict[str, list[str]] = {
    "shadow_roughness": ["B08"],
    "zebra_clarity":    ["B02", "B04"],
    "sediment_plume":   ["B03", "B04"],
}


# ── Per-pixel mask helpers ────────────────────────────────────────────────────

def _radial_masks(
    arr: np.ndarray,
    bbox: dict,
    centre_lat: float,
    centre_lon: float,
    signal_r_m: float,
    bg_inner_m: float,
    bg_outer_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build boolean masks for signal zone and background annulus within arr.
    Returns (signal_mask, bg_mask).
    """
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((h, w), bool), np.zeros((h, w), bool)

    # pixel centres
    lats = np.linspace(bbox["north"], bbox["south"], h)
    lons = np.linspace(bbox["west"],  bbox["east"],  w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Approximate distance (fast; <0.1% error within 100 km)
    dlat_m = (lat_grid - centre_lat) * 111_325.0
    dlon_m = (lon_grid - centre_lon) * 111_325.0 * np.cos(np.radians(centre_lat))
    dist_m = np.hypot(dlat_m, dlon_m)

    signal_mask = (dist_m <= signal_r_m) & np.isfinite(arr)
    bg_mask     = (dist_m >= bg_inner_m) & (dist_m <= bg_outer_m) & np.isfinite(arr)
    return signal_mask, bg_mask


def _score_scene(
    chip_bands: dict[str, np.ndarray],
    bbox: dict,
    wreck_lat: float,
    wreck_lon: float,
    concept: str,
    signal_direction: str,
) -> Optional[float]:
    """
    Score a single scene for one wreck/concept combination.

    Returns a z-score (positive means signal in expected direction,
    negative means anti-signal).  Returns None if data insufficient.
    """
    metric_fn = CONCEPT_METRIC_FN[concept]

    # Need at least one band to build masks
    some_band = next(iter(chip_bands.values()), None)
    if some_band is None or some_band.size < 4:
        return None

    # Build masks on the first available band (geometry is shared across bands)
    sig_mask, bg_mask = _radial_masks(
        some_band, bbox,
        wreck_lat, wreck_lon,
        CHIP_SIGNAL_M, CHIP_BG_INNER_M, CHIP_BG_OUTER_M,
    )
    if sig_mask.sum() == 0 or bg_mask.sum() == 0:
        return None

    # Compute metric in signal zone and background annulus
    sig_bands = {k: v[sig_mask] for k, v in chip_bands.items() if v is not None}
    bg_bands  = {k: v[bg_mask]  for k, v in chip_bands.items() if v is not None}

    # Rebuild 2-D-like arrays from masked flat arrays (metric fns support any shape)
    sig_val = metric_fn({k: v.reshape(1, -1) for k, v in sig_bands.items() if v.size > 0})
    bg_vals = metric_fn({k: v.reshape(1, -1) for k, v in bg_bands.items()  if v.size > 0})

    if sig_val is None or bg_vals is None:
        return None

    # Z-score: sig relative to background
    # Build full background distribution per pixel for std
    bg_all: list[float] = []
    for k, v in bg_bands.items():
        if np.isfinite(v).any():
            bg_all.extend(v[np.isfinite(v)].tolist())
            break  # one band is enough for spread estimate
    if len(bg_all) < 3:
        return None

    bg_std = float(np.std(bg_all))
    if bg_std < 1e-9:
        return None

    raw_z = (sig_val - float(np.mean(bg_all))) / bg_std

    # Flip sign if signal direction is "dark" (we want wreck to be anomalously dark)
    if signal_direction == "dark":
        return -raw_z
    return raw_z


# ── KML writer ────────────────────────────────────────────────────────────────

def _score_to_kml_color(score: float) -> str:
    """Returns KML aabbggrr hex color from score 0–10."""
    t = max(0.0, min(1.0, score / 10.0))
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return f"ff{g:02x}00{r:02x}"


def write_wreck_targets_kml(
    targets: list[WreckTarget],
    out_path: Path,
    title: str = "WH2K Wreck Targets",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        f"<Document><name>{title}</name>",
    ]
    for t in targets:
        color = _score_to_kml_color(t.score)
        lines += [
            "<Placemark>",
            f"  <name>{t.wreck_name} [{t.concept}]</name>",
            f"  <description>score={t.score:.1f} hit_rate={t.hit_rate:.2f} "
            f"n_scenes={t.n_scenes} best_z={t.best_zscore:.2f} "
            f"best_date={t.best_date} depth={t.depth_m}m</description>",
            "  <Style><IconStyle>",
            f"    <color>{color}</color>",
            "    <scale>0.8</scale>",
            "  </IconStyle></Style>",
            f"  <Point><coordinates>{t.lon},{t.lat},0</coordinates></Point>",
            "</Placemark>",
        ]
    lines += ["</Document></kml>"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote KML: %s", out_path)


# ── Main targeting loop ───────────────────────────────────────────────────────

def run_concept_for_wreck(
    wreck: dict,
    concept: str,
    max_scenes: int,
    band_cache_dir: Path,
    no_fetch: bool,
) -> WreckTarget:
    season = CONCEPT_SEASONS[concept]
    target = WreckTarget(
        wreck_id=wreck["id"],
        wreck_name=wreck["name"],
        lat=wreck["lat"],
        lon=wreck["lon"],
        depth_m=wreck["depth_m"],
        concept=concept,
    )

    # Chip bbox: big enough to cover signal + background annulus + download margin
    chip_bbox = _chip_bbox(wreck["lat"], wreck["lon"], CHIP_BG_OUTER_M * 1.2)

    bands_needed = CONCEPT_BANDS[concept]
    scene_label_base = (
        f"wreck{wreck['id'][:8]}_{concept[:5]}"
        .replace("/", "_").replace(" ", "_")
    )

    if no_fetch:
        log.info("  [no-fetch] skipping STAC search for %s / %s", wreck["name"], concept)
        return target

    stac_items = _stac_search(
        bbox=chip_bbox,
        date_start=season["archive_start"],
        date_end=season["archive_end"],
        max_cloud=season["max_cloud"],
        months=season["months"],
        max_items=max(max_scenes * 3, 30),  # over-fetch to select best
    )
    log.info("  %s / %s: %d qualifying scenes (using up to %d)",
             wreck["name"][:30], concept, len(stac_items), max_scenes)

    zscores: list[float] = []
    best_date: str = ""
    best_z: float = -99.0
    signal_direction = season["signal_direction"]

    for item in stac_items[:max_scenes]:
        scene_date = item.get("properties", {}).get("datetime", "")[:10]
        scene_id   = item.get("id", "")[:20].replace(":", "_")
        scene_label = f"{scene_label_base}_{scene_date}"

        # Download required bands for this chip
        chip_bands: dict[str, np.ndarray] = {}
        all_ok = True
        for band in bands_needed:
            href = _asset_href(item, band)
            if href is None:
                all_ok = False
                break
            cache_p = band_cache_dir / f"{scene_label}_{band}.npy"
            arr = _download_band_chip(href, chip_bbox, cache_p)
            if arr is None:
                all_ok = False
                break
            chip_bands[band] = arr

        if not all_ok or not chip_bands:
            continue

        zscore = _score_scene(
            chip_bands=chip_bands,
            bbox=chip_bbox,
            wreck_lat=wreck["lat"],
            wreck_lon=wreck["lon"],
            concept=concept,
            signal_direction=signal_direction,
        )
        if zscore is None:
            continue

        target.n_scenes += 1
        zscores.append(zscore)
        if zscore > HIT_ZSCORE_MIN:
            target.n_hits += 1
        if zscore > best_z:
            best_z = zscore
            best_date = scene_date

    if target.n_scenes > 0:
        target.hit_rate    = round(target.n_hits / target.n_scenes, 3)
        target.mean_zscore = round(float(np.mean(zscores)), 3)
        target.best_zscore = round(best_z, 3)
        target.best_date   = best_date
        # Composite score 0-10:
        #   50% hit_rate, 50% best_z (capped at 4σ)
        hr_score  = target.hit_rate * 10.0
        z_score10 = min(10.0, max(0.0, (target.best_zscore / 4.0) * 10.0))
        target.score = round(0.5 * hr_score + 0.5 * z_score10, 2)

    return target


# ── Entrypoint ────────────────────────────────────────────────────────────────

CONCEPT_CHOICES = ["shadow_roughness", "zebra_clarity", "sediment_plume", "all"]


def _parse_coords_arg(raw: list[str]) -> list[dict]:
    """
    Parse --coords entries of the form  "lat,lon[,name]"
    e.g. "45.78725,-85.6708,TestSite1"
    Degrees-minutes format is NOT supported here; convert before passing.
    Returns list of wreck dicts compatible with run_concept_for_wreck().
    """
    result = []
    for i, entry in enumerate(raw):
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) < 2:
            log.warning("Ignoring malformed --coords entry: %r", entry)
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            log.warning("Cannot parse lat/lon from %r — skipping", entry)
            continue
        name = parts[2] if len(parts) > 2 else f"TestSite{i + 1}"
        result.append({
            "id": f"custom_{i:03d}",
            "name": name,
            "lat": lat,
            "lon": lon,
            "depth_m": 0.0,
        })
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Season-aware Sentinel-2 wreck signal extractor. "
            "Scores known wrecks by optical concept consistency across multi-year archives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concepts", default="all", choices=CONCEPT_CHOICES,
                        help="Which concept(s) to evaluate")
    parser.add_argument("--max-wrecks", type=int, default=50,
                        help="Maximum wrecks to process (random sample from DB)")
    parser.add_argument("--max-scenes", type=int, default=8,
                        help="Maximum Sentinel-2 scenes to fetch per wreck × concept")
    parser.add_argument("--min-score", type=float, default=4.0,
                        help="Minimum score to include in KML/CSV output")
    parser.add_argument("--output-dir",
                        default="wreck_hunting_ml/runs/wreck_targeting",
                        help="Output directory for CSV/KML results")
    parser.add_argument("--band-cache",
                        default="wreck_hunting_ml/sentinel_bands/chips",
                        help="Directory for caching downloaded band chips")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Dry-run: skip STAC queries, use only cached chips")
    parser.add_argument("--all-scores", action="store_true",
                        help="Include all wrecks in output regardless of score")
    parser.add_argument(
        "--coords", nargs="+", metavar="LAT,LON[,NAME]",
        help=(
            "One or more explicit test sites in 'lat,lon[,name]' format. "
            "When provided, skips the wrecks.db lookup entirely. "
            "Works for any lake/location (not restricted to Erie bbox). "
            "Example: --coords '45.787,-85.671,TestA' '43.85,-82.59,TestB'"
        ),
    )
    args = parser.parse_args(argv)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    band_cache = REPO_ROOT / args.band_cache
    band_cache.mkdir(parents=True, exist_ok=True)

    concepts = (
        ["shadow_roughness", "zebra_clarity", "sediment_plume"]
        if args.concepts == "all"
        else [args.concepts]
    )

    # --coords overrides wrecks.db entirely
    if args.coords:
        wrecks = _parse_coords_arg(args.coords)
        if not wrecks:
            log.error("No valid coordinates parsed from --coords. "
                      "Format: lat,lon[,name]  e.g.  45.787,-85.671,TestSite1")
            return
        log.info("Using %d explicit test site(s) from --coords (no DB lookup)", len(wrecks))
        for w in wrecks:
            log.info("  • %s  lat=%.6f  lon=%.6f", w["name"], w["lat"], w["lon"])
    else:
        wrecks = _load_known_wrecks(args.max_wrecks)
        if not wrecks:
            log.error("No wrecks loaded — cannot proceed. Check db/wrecks.db exists.")
            return

    log.info("=" * 60)
    log.info("WH2K Sentinel-2 Season-Aware Wreck Targeting")
    log.info("Wrecks: %d | Concepts: %s | Max scenes/pair: %d",
             len(wrecks), concepts, args.max_scenes)
    log.info("=" * 60)

    all_targets: list[WreckTarget] = []

    for concept in concepts:
        log.info("━━━ Concept: %s ━━━", concept)
        concept_targets: list[WreckTarget] = []

        for i, wreck in enumerate(wrecks):
            log.info("[%d/%d] %s  (%.4f, %.4f)  depth=%.0fm",
                     i + 1, len(wrecks),
                     wreck["name"][:35], wreck["lat"], wreck["lon"], wreck["depth_m"])
            t = run_concept_for_wreck(
                wreck=wreck,
                concept=concept,
                max_scenes=args.max_scenes,
                band_cache_dir=band_cache,
                no_fetch=args.no_fetch,
            )
            concept_targets.append(t)

        concept_targets.sort(key=lambda t: -t.score)

        # Per-concept CSV
        csv_path = out_dir / f"wreck_targets_{concept}.csv"
        _write_csv(concept_targets, csv_path)

        # Per-concept KML (filtered)
        filtered = concept_targets if args.all_scores else [
            t for t in concept_targets if t.score >= args.min_score
        ]
        kml_path = out_dir / f"wreck_targets_{concept}.kml"
        write_wreck_targets_kml(
            filtered, kml_path,
            title=f"WH2K — {concept.replace('_', ' ').title()}",
        )
        all_targets.extend(concept_targets)

        top5 = [f"{t.wreck_name[:20]}={t.score:.1f}" for t in concept_targets[:5]]
        log.info("  Top-5: %s", " | ".join(top5))

    # Combined CSV
    _write_csv(all_targets, out_dir / "wreck_targets_all.csv")

    # Combined KML (best score per wreck across concepts)
    best_per_wreck: dict[str, WreckTarget] = {}
    for t in all_targets:
        prev = best_per_wreck.get(t.wreck_id)
        if prev is None or t.score > prev.score:
            best_per_wreck[t.wreck_id] = t
    combined_kml = [t for t in best_per_wreck.values()
                    if args.all_scores or t.score >= args.min_score]
    combined_kml.sort(key=lambda t: -t.score)
    write_wreck_targets_kml(
        combined_kml,
        out_dir / "wreck_targets_all.kml",
        title="WH2K — All Concepts (best per wreck)",
    )

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    for concept in concepts:
        ct = [t for t in all_targets if t.concept == concept]
        scored = [t for t in ct if t.n_scenes > 0]
        top = sorted(scored, key=lambda t: -t.score)[:3]
        log.info("  %-22s : %d wrecks evaluated, %d with scenes, top=%s",
                 concept, len(ct), len(scored),
                 " | ".join(f"{t.wreck_name[:16]}={t.score:.1f}" for t in top))
    log.info("Output → %s", out_dir)
    log.info("=" * 60)


def _write_csv(targets: list[WreckTarget], path: Path) -> None:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(t) for t in targets]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
