"""
WreckHunter 2000 — Raw Data Fetcher
=====================================
Queries and downloads TRUE raw flight-line / ship-track magnetometer pings
from authoritative sources, bypassing pre-gridded TIF products entirely.

SOURCE PRIORITY ORDER
---------------------
1. NCEI NGDC Trackline (MGD77T / MAG88T)  — marine + aeromagnetic 1-Hz pings
2. NRCan GeoCore XYZ                      — Canadian high-res aeromagnetic ASCII
3. USGS ScienceBase (sciencebasepy)        — US surveys DS-321/DS-411
4. ESA VirES (viresclient)                 — Swarm MAGx_LR satellite baseline
5. GLOS Seagull                            — AUV cruise logs (opportunistic)

HOW NCEI/NGDC TRACKLINE WORKS
-------------------------------
The NGDC hosts a WFS-style REST query against their survey catalog.  Each survey
is uniquely identified by a SURVEY_ID string (e.g. "H09043", "AR0002").
We use the bbox query to find survey IDs, then download the data files.

FORMATS
-------
- MGD77T  : marine geophysics (ship tracks).  Col order: DATE TIME TZ LAT LON
             TWTT DEPTH BATHY MTOTAL MGOBS DIUR MSD MSENS GOBS EOTS FREEAIR
             FAA MTF1 MTF2 MAG_RES_1 MAG_RES_2 MISSD GRAVITY EOTVOS FREEAIR2
- MAG88T  : airborne geophysics (flight lines).  Cols vary by survey; minimum:
             SURVEY_ID LINE_ID LAT LON ALT TMF IGRF RESIDUAL DIURNAL COR_TMF

LORAN-C WARP
-------------
After download, each ping is passed through apply_warp_to_points() which reads
scripts/loran_warp_field.json (produced by wh2k_warp_field_export.py) and
applies the pre-computed IDW correction to re-position the ping to WGS84 centre.

Usage
-----
    # List surveys available — no download
    python -W ignore scripts/wh2k_ncei_fetch.py --list-only

    # Download all Erie raw pings (will be large — start here)
    python -W ignore scripts/wh2k_ncei_fetch.py --lake erie --source ncei

    # Canadian north shore (needed for Ghost-1 constraint)
    python -W ignore scripts/wh2k_ncei_fetch.py --lake erie --source nrcan

    # Swarm satellite baseline (for upward continuation test)
    python -W ignore scripts/wh2k_ncei_fetch.py --lake erie --source swarm

    # All sources, no new network fetch (use whatever is cached locally)
    python -W ignore scripts/wh2k_ncei_fetch.py --lake erie --no-fetch
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Lake bounding boxes ────────────────────────────────────────────────────

LAKE_BBOX = {
    "erie": {
        "lat_min": 41.35, "lat_max": 42.90,
        "lon_min": -83.50, "lon_max": -78.85,
        "label": "Lake Erie",
    },
    "huron": {
        "lat_min": 43.00, "lat_max": 46.50,
        "lon_min": -84.80, "lon_max": -79.50,
        "label": "Lake Huron / Georgian Bay",
    },
    "superior": {
        "lat_min": 46.30, "lat_max": 49.00,
        "lon_min": -92.10, "lon_max": -84.35,
        "label": "Lake Superior",
    },
}

# ── Output directories ─────────────────────────────────────────────────────

RAW_DIR = REPO / "magnetic_data" / "raw" / "ncei_trackline"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SWARM_DIR = REPO / "magnetic_data" / "raw" / "swarm_l2"
SWARM_DIR.mkdir(parents=True, exist_ok=True)

NRCAN_XYZ_DIR = REPO / "magnetic_data" / "raw" / "nrcan_xyz"
NRCAN_XYZ_DIR.mkdir(parents=True, exist_ok=True)

# ── NCEI NGDC API endpoints ────────────────────────────────────────────────

# WFS query for trackline survey catalog
NGDC_SURVEY_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/"
    "trackline_geophysical/FeatureServer/0/query"
)

# Data Access Service for downloading a specific survey
NGDC_DOWNLOAD_URL = "https://www.ngdc.noaa.gov/mgg/dtts/ds.html?id={survey_id}"

# Alternative: NCEI geomagnetic ERDDAP endpoint
NCEI_ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap"

#  NCEI API for trackline geophysical data
NCEI_CATALOG_URL = (
    "https://www.ncei.noaa.gov/access/search/data-search/"
    "trackline-geophysics?keywords={keywords}&startDate={start}&endDate={end}"
)

# NRCan GeoCore (new) / GeoGratis (old) metadata API
NRCAN_GEOCORE_URL = "https://geocore.nrc.gc.ca/api/collections/items"

# USGS ScienceBase API
USGS_SB_SEARCH = "https://www.sciencebase.gov/catalog/items"

# ESA Swarm VirES token endpoint
VIRES_URL = "https://vires.services/ows"


# ── Utility ────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30, retries: int = 3) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wh2k/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            logger.warning("HTTP %d for %s (attempt %d/%d)", e.code, url, attempt + 1, retries)
            if e.code in (429, 503):
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.warning("Request failed: %s (attempt %d/%d)", e, attempt + 1, retries)
            time.sleep(2)
    return None


def _bbox_wkt(bbox: dict) -> str:
    """Return WKT POLYGON for NGDC WFS query."""
    mn_la = bbox["lat_min"]; mx_la = bbox["lat_max"]
    mn_lo = bbox["lon_min"]; mx_lo = bbox["lon_max"]
    return (f"POLYGON (({mn_lo} {mn_la},{mx_lo} {mn_la},"
            f"{mx_lo} {mx_la},{mn_lo} {mx_la},{mn_lo} {mn_la}))")


# ── Step 1: Catalog query — find survey IDs in bbox ───────────────────────

def query_ncei_surveys(bbox: dict) -> list[dict]:
    """
    Query NGDC trackline WFS for aeromagnetic surveys in bbox.
    Returns list of survey metadata dicts with at minimum:
      survey_id, platform, date_start, date_end, has_magnetics
    """
    params = {
        "where": "1=1",
        "geometry": json.dumps({
            "xmin": bbox["lon_min"], "ymin": bbox["lat_min"],
            "xmax": bbox["lon_max"], "ymax": bbox["lat_max"],
            "spatialReference": {"wkid": 4326},
        }),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "SURVEY_ID,PLATFORM,BEGIN_DATE,END_DATE,HAS_MAGNETICS,DATA_TYPES,ENTRY",
        "returnGeometry": "false",
        "f": "json",
        "resultRecordCount": "500",
    }
    url = NGDC_SURVEY_URL + "?" + urllib.parse.urlencode(params)
    logger.info("NCEI survey catalog query: %s …", url[:80])
    raw = _get(url)
    if raw is None:
        logger.warning("NCEI catalog query failed")
        return []
    try:
        data = json.loads(raw)
        features = data.get("features", [])
        surveys = []
        for feat in features:
            attr = feat.get("attributes", {})
            has_mag = str(attr.get("HAS_MAGNETICS", "")).upper()
            if "Y" not in has_mag and "MAG" not in str(attr.get("DATA_TYPES", "")).upper():
                continue  # skip non-magnetic surveys
            surveys.append({
                "survey_id":  attr.get("SURVEY_ID", "?"),
                "platform":   attr.get("PLATFORM", "?"),
                "date_start": attr.get("BEGIN_DATE", "?"),
                "date_end":   attr.get("END_DATE", "?"),
                "entry":      attr.get("ENTRY", "?"),
            })
        logger.info("Found %d magnetic surveys in bbox", len(surveys))
        return surveys
    except Exception as e:
        logger.error("Could not parse NCEI response: %s", e)
        return []


def query_sciencebase_erie() -> list[dict]:
    """
    Query USGS ScienceBase for known Erie/Ohio aeromagnetic datasets.
    Targets Data Series 321 (Ohio) and DS-411 (Michigan).
    """
    items = []
    for term in ["magnetic anomaly Lake Erie", "aeromagnetic Ohio 321",
                 "aeromagnetic Michigan 411"]:
        params = {
            "q": term,
            "max": 20,
            "fields": "id,title,webLinks,files",
            "format": "json",
        }
        url = USGS_SB_SEARCH + "?" + urllib.parse.urlencode(params)
        raw = _get(url, timeout=20)
        if raw is None:
            continue
        try:
            data = json.loads(raw)
            for item in data.get("items", []):
                items.append({
                    "id":    item.get("id"),
                    "title": item.get("title"),
                    "links": [l.get("uri") for l in item.get("webLinks", [])],
                })
        except Exception as e:
            logger.warning("ScienceBase parse error: %s", e)
    logger.info("ScienceBase: found %d items", len(items))
    return items


# ── Step 2: Data download ──────────────────────────────────────────────────

def _download_mgd77t(survey_id: str, out_dir: Path) -> Optional[Path]:
    """
    Attempt download of MGD77T data for a survey.
    Tries the NGDC trackline WFS data export endpoint.
    Output: <out_dir>/<survey_id>.tab (tab-delimited)
    """
    out_file = out_dir / f"{survey_id}.tab"
    if out_file.exists() and out_file.stat().st_size > 1024:
        logger.info("  %s already cached (%d bytes)", out_file.name, out_file.stat().st_size)
        return out_file

    # Try NGDC direct download endpoint
    url = (f"https://www.ngdc.noaa.gov/mgg/trackline/activity.do"
           f"?method=getDataFiles&survey={survey_id}&format=tab")
    raw = _get(url, timeout=60)
    if raw and len(raw) > 200:
        out_file.write_bytes(raw)
        logger.info("  Downloaded %s → %d bytes", survey_id, len(raw))
        return out_file

    # Fallback: ERDDAP if available
    logger.warning("  Could not download %s from NGDC — check manually via "
                   "https://www.ncei.noaa.gov/access/search/data-search/"
                   "trackline-geophysics", survey_id)
    return None


def download_surveys(surveys: list[dict], out_dir: Path) -> list[Path]:
    """Download all available surveys, return list of successfully downloaded files."""
    downloaded = []
    for s in surveys:
        sid = s["survey_id"]
        logger.info("Downloading survey: %s  (%s  %s–%s)",
                    sid, s["platform"], s["date_start"], s["date_end"])
        fp = _download_mgd77t(sid, out_dir)
        if fp:
            downloaded.append(fp)
        time.sleep(0.3)   # be polite to NCEI
    return downloaded


# ── Step 3: Parse MGD77T / MAG88T ─────────────────────────────────────────

# MGD77T column order (NOAA standard, tab-delimited, first 24 fields that matter)
MGD77T_COLS = [
    "SURVEY_ID","TIMEZONE","YEAR","MONTH","DAY","HOUR","MIN","SEC",
    "LAT","LON","PTC","TWT","DEPTH","BBT","CORR_DEPTH","FATHOMS",
    "GOBS","EOTV","FAA","MTF1","MTF2","MAG_ANOM","DIUR","MSF",
]


def parse_mgd77t_file(fp: Path) -> tuple[list[float], list[float], list[float]]:
    """
    Parse a MGD77T tab-delimited file.
    Returns (lons, lats, mag_residuals) with nodata removed.
    """
    lons, lats, mags = [], [], []
    try:
        with open(fp, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("SURVEY"):
                    continue
                parts = line.split("\t")
                if len(parts) < 22:
                    continue
                try:
                    lat = float(parts[8])
                    lon = float(parts[9])
                    mag = float(parts[21])  # MAG_ANOM (residual, IGRF already subtracted)
                    if abs(lat) > 90 or abs(lon) > 180 or abs(mag) > 5000:
                        continue
                    lats.append(lat); lons.append(lon); mags.append(mag)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.error("Parse error for %s: %s", fp.name, e)
    return lons, lats, mags


def parse_mag88t_file(fp: Path) -> tuple[list[float], list[float], list[float]]:
    """
    Parse a MAG88T airborne file (tab or space delimited).
    Column discovery is automatic — looks for LAT, LON, RESIDUAL / MAG_ANOM.
    """
    lons, lats, mags = [], [], []
    try:
        with open(fp, encoding="latin-1") as f:
            lines = f.readlines()

        # Find header line
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            upper = line.upper()
            if "LAT" in upper and "LON" in upper:
                header_line = lines[i]
                data_start = i + 1
                break

        if header_line is None:
            logger.warning("  No header found in %s", fp.name)
            return [], [], []

        # Detect delimiter
        delim = "\t" if "\t" in header_line else None  # None = whitespace
        cols = header_line.strip().split(delim)
        cols_upper = [c.upper().strip() for c in cols]

        def _col(*names):
            for n in names:
                if n.upper() in cols_upper:
                    return cols_upper.index(n.upper())
            return None

        ci_lat = _col("LAT", "LATITUDE", "Y")
        ci_lon = _col("LON", "LONG", "LONGITUDE", "X")
        ci_mag = _col("MAG_ANOM", "RESIDUAL", "RESID", "CORRECTED", "COR_TMF", "MAG_RES")

        if ci_lat is None or ci_lon is None or ci_mag is None:
            logger.warning("  Could not find lat/lon/mag columns in %s. Cols: %s",
                           fp.name, cols_upper)
            return [], [], []

        for line in lines[data_start:]:
            parts = line.strip().split(delim)
            try:
                lat = float(parts[ci_lat])
                lon = float(parts[ci_lon])
                mag = float(parts[ci_mag])
                if abs(lat) > 90 or abs(lon) > 180 or abs(mag) > 5000:
                    continue
                lats.append(lat); lons.append(lon); mags.append(mag)
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.error("Parse error for %s: %s", fp.name, e)
    return lons, lats, mags


# ── Step 4: Apply LORAN-C warp to parsed pings ───────────────────────────

def apply_warp_to_points(
    lons: list[float],
    lats: list[float],
    warp_path: Path,
) -> tuple[list[float], list[float]]:
    """
    Applies the pre-computed LORAN-C warp field (loran_warp_field.json)
    to re-position raw pings to WGS84 corrected coordinates.

    Uses nearest-neighbour lookup in the dense warp grid — O(1) per point
    once the grid is loaded, no IDW re-computation at runtime.
    """
    import numpy as np

    if not warp_path.exists():
        logger.warning("Warp field not found at %s — skipping warp. "
                       "Run wh2k_warp_field_export.py first.", warp_path)
        return lons, lats

    with open(warp_path, encoding="utf-8") as f:
        wf = json.load(f)

    grid_lat = np.array(wf["lat_centers"])
    grid_lon = np.array(wf["lon_centers"])
    dlat_grid = np.array(wf["dlat_deg"])   # shape (n_lat, n_lon)
    dlon_grid = np.array(wf["dlon_deg"])

    lat_min = grid_lat[0];   lat_step = grid_lat[1] - grid_lat[0]
    lon_min = grid_lon[0];   lon_step = grid_lon[1] - grid_lon[0]

    warped_lons, warped_lats = [], []
    for lon, lat in zip(lons, lats):
        # Nearest-neighbour index
        ri = min(max(int(round((lat - lat_min) / lat_step)), 0), dlat_grid.shape[0] - 1)
        ci = min(max(int(round((lon - lon_min) / lon_step)), 0), dlat_grid.shape[1] - 1)
        warped_lats.append(lat + float(dlat_grid[ri, ci]))
        warped_lons.append(lon + float(dlon_grid[ri, ci]))

    return warped_lons, warped_lats


# ── Step 5: Write normalised CSV ──────────────────────────────────────────

def write_normalised_csv(
    lons: list[float],
    lats: list[float],
    mags: list[float],
    source_label: str,
    out_path: Path,
    warped_lons: Optional[list[float]] = None,
    warped_lats: Optional[list[float]] = None,
) -> None:
    """
    Write normalised ping CSV with columns:
    lon_raw, lat_raw, lon_warped, lat_warped, mag_anomaly_nT, source
    """
    use_warp = warped_lons is not None and warped_lats is not None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lon_raw", "lat_raw",
                         "lon_warped", "lat_warped",
                         "mag_anomaly_nT", "source"])
        for i, (lo, la, mag) in enumerate(zip(lons, lats, mags)):
            wlo = warped_lons[i] if use_warp else lo
            wla = warped_lats[i] if use_warp else la
            writer.writerow([round(lo, 7), round(la, 7),
                             round(wlo, 7), round(wla, 7),
                             round(mag, 3), source_label])
    logger.info("  Wrote %d pings → %s", len(lons), out_path.name)


# ── Step 6: ESA Swarm download (viresclient) ──────────────────────────────

def fetch_swarm_baseline(bbox: dict, out_dir: Path, date_range: tuple[str, str] = ("2022-01-01", "2023-12-31")) -> Optional[Path]:
    """
    Download ESA Swarm MAGx_LR (1-Hz) magnetic residuals over the lake bbox.
    Requires viresclient: pip install viresclient

    The Swarm residual field at satellite altitude (~460 km) gives the
    long-wavelength baseline that EMAG2 uses as its reference.  We use it to:
    1. Check upward-continuation amplitude of known wrecks
    2. Verify EMAG2 baseline fill is consistent

    Returns path to downloaded CSV or None if viresclient not available.
    """
    out_file = out_dir / f"swarm_lr_{date_range[0][:7]}_{date_range[1][:7]}.csv"
    if out_file.exists() and out_file.stat().st_size > 10_000:
        logger.info("Swarm data already cached: %s", out_file.name)
        return out_file

    try:
        from viresclient import SwarmRequest  # type: ignore
    except ImportError:
        logger.warning(
            "viresclient not installed. Install with: pip install viresclient\n"
            "Then authenticate: python -c \"from viresclient import ClientConfig; "
            "ClientConfig().set_vires_token('YOUR_TOKEN')\"\n"
            "Token from: https://vires.services/accounts/tokens/\n"
            "Swarm data will be skipped for now."
        )
        _write_swarm_placeholder(out_dir, bbox, date_range)
        return None

    try:
        logger.info("Fetching ESA Swarm MAGx_LR for bbox %s via %s …", bbox["label"], VIRES_URL)
        request = SwarmRequest(url=VIRES_URL)
        request.set_collection("SW_OPER_MAGA_LR_1B")
        request.set_products(
            measurements=["F", "dF_AOCS", "Flags_F"],
            sampling_step="PT60S",   # 1-minute samples (60-sec average of 1-Hz)
        )
        # Filter to bbox — Swarm orbits don't match bbox exactly, we post-filter
        data = request.get_between(
            start_time=date_range[0],
            end_time=date_range[1],
            asynchronous=True,
        ).as_dataframe()

        # Filter to lake bbox
        mask = (
            (data.Latitude  >= bbox["lat_min"]) & (data.Latitude  <= bbox["lat_max"]) &
            (data.Longitude >= bbox["lon_min"]) & (data.Longitude <= bbox["lon_max"])
        )
        data_lake = data[mask]
        data_lake.to_csv(out_file, index=False)
        logger.info("Swarm: %d samples in bbox → %s", len(data_lake), out_file.name)
        return out_file
    except Exception as e:
        logger.error("Swarm download failed: %s", e)
        return None


def _write_swarm_placeholder(out_dir: Path, bbox: dict, date_range: tuple) -> None:
    """Write a placeholder note with exact instructions to get Swarm data."""
    note = {
        "status": "not_yet_downloaded",
        "instructions": {
            "1_install": "pip install viresclient",
            "2_token": "Register free at https://vires.services/ → My tokens → Create token",
            "3_configure": "python -c \"from viresclient import ClientConfig; ClientConfig().set_vires_token('YOUR_TOKEN')\"",
            "4_collection": "SW_OPER_MAGA_LR_1B  (Swarm Alpha, 1-Hz, Level 1B)",
            "5_date_range": f"{date_range[0]} to {date_range[1]}",
            "6_bbox": bbox,
            "7_product": "F (total field), dF_AOCS (residual after CHAOS model subtraction)",
            "note": "The 'dF_AOCS' column is the cleanest residual — equivalent to a 460km-altitude anomaly.",
        },
    }
    note_path = out_dir / "swarm_download_instructions.json"
    note_path.write_text(json.dumps(note, indent=2))
    logger.info("Swarm instructions written → %s", note_path.name)


# ── Step 7: NRCan GeoCore query ────────────────────────────────────────────

def query_nrcan_geogratis(bbox: dict) -> list[dict]:
    """
    Query NRCan GeoCore (replacement for GeoGratis) for aeromagnetic XYZ files.
    Returns list of catalogue items with download links.
    """
    params = {
        "bbox": f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}",
        "type": "FeatureCollection",
        "lang": "en",
        "limit": 50,
    }
    url = NRCAN_GEOCORE_URL + "?" + urllib.parse.urlencode(params)
    logger.info("NRCan GeoCore query: %s …", url[:80])
    raw = _get(url, timeout=30)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
        items = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            title = props.get("title_en", "") or props.get("title", "")
            if not any(k in title.lower() for k in ["aeromagnetic", "magnetic", "magnetics"]):
                continue
            links = props.get("links", {})
            items.append({
                "id":       props.get("id"),
                "title":    title,
                "format":   props.get("format_en", ""),
                "links":    links,
            })
        logger.info("NRCan GeoCore: %d aeromagnetic items in bbox", len(items))
        return items
    except Exception as e:
        logger.warning("NRCan parse error: %s", e)
        return []


# ── Master run ─────────────────────────────────────────────────────────────

def run(
    lake: str,
    sources: list[str],
    no_fetch: bool,
    list_only: bool,
    warp_path: Optional[Path] = None,
) -> dict:
    bbox = LAKE_BBOX.get(lake)
    if bbox is None:
        raise ValueError(f"Unknown lake: {lake}. Choose from {list(LAKE_BBOX)}")

    results = {"lake": lake, "bbox": bbox, "sources_checked": {}}

    # ── NCEI trackline ─────────────────────────────────────────────────────
    if "ncei" in sources or "all" in sources:
        surveys = query_ncei_surveys(bbox)
        results["sources_checked"]["ncei"] = {
            "surveys_found": len(surveys),
            "surveys": surveys,
        }
        if list_only:
            print(f"\nNCEI surveys with magnetics in {bbox['label']} bbox:")
            for s in surveys:
                print(f"  {s['survey_id']:12s}  {s['platform']:20s}  {s['date_start']}–{s['date_end']}")
        elif not no_fetch and surveys:
            downloaded = download_surveys(surveys, RAW_DIR)
            all_lons, all_lats, all_mags = [], [], []
            for fp in downloaded:
                lo, la, mg = (parse_mgd77t_file(fp)
                              if fp.suffix == ".tab" else parse_mag88t_file(fp))
                all_lons.extend(lo);  all_lats.extend(la);  all_mags.extend(mg)

            if all_lons and warp_path and warp_path.exists():
                wlo, wla = apply_warp_to_points(all_lons, all_lats, warp_path)
            else:
                wlo, wla = None, None

            if all_lons:
                out = RAW_DIR / f"erie_ncei_pings_warped.csv"
                write_normalised_csv(all_lons, all_lats, all_mags,
                                     "NCEI_MGD77T", out, wlo, wla)
                results["sources_checked"]["ncei"]["pings_written"] = len(all_lons)

    # ── NRCan GeoCore ──────────────────────────────────────────────────────
    if "nrcan" in sources or "all" in sources:
        nrcan_items = query_nrcan_geogratis(bbox)
        results["sources_checked"]["nrcan"] = {
            "items_found": len(nrcan_items),
            "items": nrcan_items[:10],   # keep summary manageable
        }
        if list_only:
            print(f"\nNRCan GeoCore aeromagnetic items:")
            for it in nrcan_items[:15]:
                print(f"  {it['id']}  {it['title'][:70]}")

    # ── USGS ScienceBase ───────────────────────────────────────────────────
    if "sciencebase" in sources or "all" in sources:
        sb_items = query_sciencebase_erie()
        results["sources_checked"]["sciencebase"] = {
            "items_found": len(sb_items),
            "items": sb_items,
        }
        if list_only:
            print(f"\nUSGS ScienceBase aeromagnetic items:")
            for it in sb_items:
                print(f"  {it['id']}  {it['title'][:70]}")

    # ── Swarm ──────────────────────────────────────────────────────────────
    if "swarm" in sources or "all" in sources:
        if not no_fetch:
            swarm_file = fetch_swarm_baseline(bbox, SWARM_DIR)
            results["sources_checked"]["swarm"] = {
                "file": str(swarm_file) if swarm_file else "not_downloaded",
            }
        else:
            logger.info("--no-fetch: skipping Swarm download")

    # ── Summary ────────────────────────────────────────────────────────────
    cat_path = RAW_DIR / f"{lake}_fetch_catalog.json"
    cat_path.write_text(json.dumps(results, indent=2))
    logger.info("Catalog written → %s", cat_path)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch raw magnetometer pings from authoritative sources")
    parser.add_argument("--lake",      default="erie",
                        choices=list(LAKE_BBOX), help="Target lake (default: erie)")
    parser.add_argument("--source",    dest="sources", action="append",
                        default=[],
                        choices=["ncei", "nrcan", "sciencebase", "swarm", "all"],
                        help="Which source to query (repeat for multiple; default: all)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Catalog/list only — do not download data files")
    parser.add_argument("--list-only", action="store_true",
                        help="Print survey IDs and exit without downloading")
    parser.add_argument("--warp-json", default="scripts/loran_warp_field.json",
                        help="Path to LORAN-C warp field JSON")
    args = parser.parse_args(argv)

    sources = args.sources if args.sources else ["all"]
    warp_path = REPO / args.warp_json

    if args.list_only:
        args.no_fetch = True

    results = run(
        lake=args.lake,
        sources=sources,
        no_fetch=args.no_fetch,
        list_only=args.list_only,
        warp_path=warp_path if warp_path.exists() else None,
    )

    # Print summary
    print("\n" + "=" * 65)
    print(f"DATA FETCH SUMMARY — {LAKE_BBOX[args.lake]['label']}")
    print("=" * 65)
    for src, info in results["sources_checked"].items():
        n = info.get("surveys_found") or info.get("items_found") or info.get("pings_written", 0)
        print(f"  {src:15s}: {n} items")
    if not args.no_fetch:
        out_dir = RAW_DIR / f"{args.lake}_ncei_pings_warped.csv"
        if out_dir.exists():
            print(f"\n  Warped pings written → {out_dir}")
        if not (REPO / args.warp_json).exists():
            print(f"\n  ⚠ Warp field not found at {args.warp_json}")
            print("    Run: python -W ignore scripts/wh2k_warp_field_export.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
