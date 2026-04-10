"""
wh2k_awois_scraper.py
======================
Downloads / scrapes shipwreck coordinates from three public sources and inserts
them into db/wrecks.db as coord_quality='dive_verified'.

Sources
-------
  1. NOAA AWOIS  — ArcGIS REST query (bbox-filtered to Great Lakes)
  2. 3dshipwrecks.org — scrapes wreck listing pages
  3. wrecksite.eu ("clue site") — search results for Great Lakes wrecks
     NOTE: If you meant a different "clue site", update CLUESITE_BASE_URL below.

Usage
-----
  python scripts/wh2k_awois_scraper.py [--dry-run] [--source awois|3d|clue|all]

Options
  --dry-run   Print records without writing to DB
  --source    Which source(s) to ingest (default: all)
  --cache-dir Path to save raw downloaded files (default: data/awois_cache)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # handled at runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH   = REPO_ROOT / "db" / "wrecks.db"

# ── Great Lakes bbox (covers all five lakes + St. Lawrence approach) ─────────
GL_BBOX = {"west": -92.5, "south": 41.2, "east": -75.0, "north": 49.5}

# ── Source URLs ───────────────────────────────────────────────────────────────
# AWOIS via NOAA OCS ArcGIS REST endpoint (layer 0 = wrecks, layer 1 = obstructions)
AWOIS_REST_BASE = (
    "https://gis.charttools.noaa.gov/arcgis/rest/services/MCS/AWOIS/MapServer"
    "/{layer}/query"
)
AWOIS_LAYERS = {0: "wreck", 1: "obstruction"}

# 3D Shipwrecks listing URL  (Great Lakes filter)
SHIPWRECKS3D_BASE = "https://www.3dshipwrecks.org"
SHIPWRECKS3D_LIST = f"{SHIPWRECKS3D_BASE}/shipwrecks/"

# CLUE — Cleveland Lake Underwater Explorers
# https://www.clueshipwrecks.org  (Lake Erie near Cleveland)
# NOTE: CLUE does not publish GPS on public pages; only records where
# coordinates can be parsed from page text will be inserted.
CLUESITE_BASE_URL  = "https://www.clueshipwrecks.org"
CLUESITE_LIST_URL  = f"{CLUESITE_BASE_URL}/shipwrecks.htm"

# Thunder Bay National Marine Sanctuary (NOAA)
# Referenced via michigan.org Michigan Underwater Preserves article
# 82 wrecks with official DDM GPS:  N45°19.396' W83°27.508'
THUNDERBAY_BASE     = "https://thunderbay.noaa.gov"
THUNDERBAY_LIST_URL = f"{THUNDERBAY_BASE}/shipwrecks/"

# Request headers (polite scraping)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; wh2k-wreck-scraper/1.0; "
        "+https://github.com/private/bagrecovery)"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("awois_scraper")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _fetch_url(url: str, params: Optional[dict] = None, retries: int = 3) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("Fetch attempt %d failed (%s) — retrying…", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _in_gl_bbox(lat: float, lon: float) -> bool:
    return (
        GL_BBOX["south"] <= lat <= GL_BBOX["north"]
        and GL_BBOX["west"] <= lon <= GL_BBOX["east"]
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _load_existing(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT name, latitude, longitude FROM features "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    return [{"name": (r[0] or "").strip().upper(), "lat": r[1], "lon": r[2]} for r in rows]


def _is_duplicate(
    name: str,
    lat: float,
    lon: float,
    existing: list[dict],
    radius_m: float = 500.0,
) -> bool:
    name_up = name.strip().upper()
    for ex in existing:
        if ex["name"] == name_up and _haversine_m(lat, lon, ex["lat"], ex["lon"]) <= radius_m:
            return True
    return False


def _insert_record(
    conn: sqlite3.Connection,
    name: str,
    lat: float,
    lon: float,
    depth: Optional[float],
    source_tag: str,
    feature_type: str = "Wreck",
    description: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        log.info("[DRY-RUN] Would insert: %-40s  %.4f  %.4f  src=%s", name, lat, lon, source_tag)
        return True
    conn.execute(
        """
        INSERT INTO features
            (name, latitude, longitude, depth, feature_type, source,
             coord_quality, found_status, description_narrative)
        VALUES (?, ?, ?, ?, ?, ?, 'dive_verified', 'found', ?)
        """,
        (name, lat, lon, depth, feature_type, source_tag, description),
    )
    return True


# ── Source 1: NOAA AWOIS ──────────────────────────────────────────────────────

def _awois_field_lat(props: dict) -> Optional[float]:
    for key in ("LATDEC", "LATD", "LAT", "Y"):
        if key in props and props[key] is not None:
            try:
                return float(props[key])
            except (TypeError, ValueError):
                pass
    return None


def _awois_field_lon(props: dict) -> Optional[float]:
    for key in ("LONDEC", "LOND", "LON", "X"):
        if key in props and props[key] is not None:
            try:
                v = float(props[key])
                # AWOIS longitudes for US coast are negative; some versions omit the sign
                return v if v < 0 else -v
            except (TypeError, ValueError):
                pass
    return None


def _awois_name(props: dict) -> str:
    for key in ("VESSLTERMS", "VESSEL_TERMS", "NAME", "FEATURENAME"):
        if props.get(key):
            return str(props[key]).strip()
    return "UNKNOWN AWOIS FEATURE"


def scrape_awois(cache_dir: Path, dry_run: bool = False) -> list[dict]:
    """Query NOAA AWOIS ArcGIS REST service for features within Great Lakes bbox."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    bbox_str = (
        f"{GL_BBOX['west']},{GL_BBOX['south']},{GL_BBOX['east']},{GL_BBOX['north']}"
    )

    for layer_id, ftype in AWOIS_LAYERS.items():
        cache_file = cache_dir / f"awois_layer{layer_id}.json"
        if cache_file.exists():
            log.info("AWOIS layer %d: using cache %s", layer_id, cache_file)
            raw = cache_file.read_bytes()
        else:
            url = AWOIS_REST_BASE.format(layer=layer_id)
            params = {
                "where": "1=1",
                "geometry": bbox_str,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "json",
                "resultRecordCount": "5000",
            }
            log.info("Fetching AWOIS layer %d from REST API…", layer_id)
            try:
                raw = _fetch_url(url, params)
                cache_file.write_bytes(raw)
            except Exception as exc:
                log.error("AWOIS layer %d fetch failed: %s", layer_id, exc)
                continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("AWOIS layer %d JSON parse error: %s", layer_id, exc)
            continue

        features = data.get("features", [])
        log.info("AWOIS layer %d: %d raw features received", layer_id, len(features))
        for feat in features:
            attrs = feat.get("attributes", {}) or {}
            geom  = feat.get("geometry", {}) or {}

            # Prefer geometry from GeoJSON; fall back to attribute fields
            lat = geom.get("y") or _awois_field_lat(attrs)
            lon = geom.get("x") or _awois_field_lon(attrs)
            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if not _in_gl_bbox(lat, lon):
                continue

            name  = _awois_name(attrs)
            depth_raw = attrs.get("DEPTH") or attrs.get("DEPTH_M")
            try:
                depth = float(depth_raw) * 0.3048 if depth_raw else None  # ft → m
            except (TypeError, ValueError):
                depth = None

            history = str(attrs.get("HISTORY") or attrs.get("REMARKS") or "").strip()
            records.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "depth": depth,
                "source": "NOAA_AWOIS",
                "feature_type": ftype.capitalize(),
                "description": history or None,
            })
        time.sleep(0.5)  # be polite between layer requests

    log.info("AWOIS: %d Great Lakes records collected", len(records))
    return records


# ── Source 2: 3dshipwrecks.org ────────────────────────────────────────────────

_COORD_RE = re.compile(
    r"(\d{1,2})[°\s]+(\d{1,2}(?:\.\d+)?)[′'\s]+([NS])"
    r"[\s,/]+"
    r"(\d{1,3})[°\s]+(\d{1,2}(?:\.\d+)?)[′'\s]+([EW])",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(
    r"(-?\d{1,2}\.\d{4,})\s*,\s*(-?\d{2,3}\.\d{4,})"
)


def _parse_coords_from_text(text: str) -> Optional[tuple[float, float]]:
    """Try to extract lat/lon from arbitrary text (DMS or decimal)."""
    m = _DECIMAL_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _COORD_RE.search(text)
    if m:
        deg, mins, ns, deg2, mins2, ew = m.groups()
        lat = float(deg) + float(mins) / 60
        lon = float(deg2) + float(mins2) / 60
        if ns.upper() == "S":
            lat = -lat
        if ew.upper() == "W":
            lon = -lon
        return lat, lon
    return None


def _parse_depth_ft(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|')", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 0.3048
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def scrape_3dshipwrecks(cache_dir: Path, dry_run: bool = False) -> list[dict]:
    """Scrape shipwreck listing pages from 3dshipwrecks.org."""
    if BeautifulSoup is None:
        log.error("beautifulsoup4 not installed — run: pip install beautifulsoup4 lxml")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # Fetch main listings page
    list_cache = cache_dir / "3dshipwrecks_list.html"
    if list_cache.exists():
        html_bytes = list_cache.read_bytes()
    else:
        log.info("Fetching 3dshipwrecks.org listing page…")
        try:
            html_bytes = _fetch_url(SHIPWRECKS3D_LIST)
            list_cache.write_bytes(html_bytes)
        except Exception as exc:
            log.error("3dshipwrecks.org listing fetch failed: %s", exc)
            return []

    soup = BeautifulSoup(html_bytes, "lxml")

    # Collect wreck detail page links
    wreck_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/shipwreck/" in href or "/wrecks/" in href:
            full = href if href.startswith("http") else SHIPWRECKS3D_BASE + href
            if full not in wreck_links:
                wreck_links.append(full)

    log.info("3dshipwrecks.org: found %d wreck detail links", len(wreck_links))

    for link in wreck_links:
        safe_name = re.sub(r"[^\w]", "_", link.split("/")[-2] or link.split("/")[-1])
        page_cache = cache_dir / f"3dshipwrecks_{safe_name[:80]}.html"
        if page_cache.exists():
            page_bytes = page_cache.read_bytes()
        else:
            try:
                page_bytes = _fetch_url(link)
                page_cache.write_bytes(page_bytes)
                time.sleep(1.5)  # 1.5s between requests
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", link, exc)
                continue

        page_soup = BeautifulSoup(page_bytes, "lxml")
        page_text = page_soup.get_text(" ", strip=True)

        # Name: try <h1> or <title>
        name = ""
        h1 = page_soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)
        if not name:
            title = page_soup.find("title")
            if title:
                name = title.get_text(strip=True).split("|")[0].strip()
        if not name:
            name = safe_name.replace("_", " ").title()

        # Coordinates
        coords = _parse_coords_from_text(page_text)
        if coords is None:
            # Try looking for coord-labelled fields
            for label_text in ["latitude", "longitude", "location", "gps", "coordinates"]:
                tags = page_soup.find_all(string=re.compile(label_text, re.I))
                for tag in tags:
                    parent_text = tag.parent.get_text(" ", strip=True) if tag.parent else ""
                    coords = _parse_coords_from_text(parent_text)
                    if coords:
                        break
                if coords:
                    break

        if coords is None:
            log.debug("No coords found for %s — skipping", name)
            continue

        lat, lon = coords
        if not _in_gl_bbox(lat, lon):
            log.debug("Skipping out-of-bbox wreck: %s  %.4f  %.4f", name, lat, lon)
            continue

        depth = _parse_depth_ft(page_text)
        records.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "source": "3dshipwrecks.org",
            "feature_type": "Wreck",
            "description": link,
        })
        log.info("3dshipwrecks.org: %-40s  %.4f  %.4f", name, lat, lon)

    log.info("3dshipwrecks.org: %d Great Lakes records collected", len(records))
    return records


# ── Source 3: clueshipwrecks.org (CLUE) ──────────────────────────────────────

def scrape_cluesite(cache_dir: Path, dry_run: bool = False) -> list[dict]:
    """
    Scrape CLUE (Cleveland Lake Underwater Explorers) wreck pages.
    URL: https://www.clueshipwrecks.org/shipwrecks.htm

    CLUE does not publish GPS coordinates on their public pages; wreck detail
    pages are scraped for name/depth, and any parseable coordinate text is
    extracted.  Most records will be skipped (no coords); the function still
    runs in case future pages or cached copies include coordinate text.
    """
    if BeautifulSoup is None:
        log.error("beautifulsoup4 not installed — run: pip install beautifulsoup4 lxml")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # Fetch listing page
    list_cache = cache_dir / "clue_shipwrecks_list.html"
    if list_cache.exists():
        listing_bytes = list_cache.read_bytes()
    else:
        log.info("Fetching CLUE shipwrecks listing page…")
        try:
            listing_bytes = _fetch_url(CLUESITE_LIST_URL)
            list_cache.write_bytes(listing_bytes)
        except Exception as exc:
            log.error("CLUE listing fetch failed: %s", exc)
            return []

    soup = BeautifulSoup(listing_bytes, "lxml")

    # Collect all _sw.htm links (individual wreck detail pages)
    detail_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith("_sw.htm") or "_sw.htm" in href:
            name_text = a.get_text(strip=True)
            full_url = href if href.startswith("http") else f"{CLUESITE_BASE_URL}/{href}"
            if full_url not in [lnk for lnk, _ in detail_links]:
                detail_links.append((full_url, name_text))

    log.info("CLUE: found %d wreck detail pages", len(detail_links))

    for page_url, hint_name in detail_links:
        slug = re.sub(r"[^\w]", "_", page_url.split("/")[-1].replace(".htm", ""))
        page_cache = cache_dir / f"clue_{slug}.html"
        if page_cache.exists():
            page_bytes = page_cache.read_bytes()
        else:
            try:
                page_bytes = _fetch_url(page_url)
                page_cache.write_bytes(page_bytes)
                time.sleep(1.5)
            except Exception as exc:
                log.warning("CLUE: failed to fetch %s: %s", page_url, exc)
                continue

        page_soup = BeautifulSoup(page_bytes, "lxml")
        page_text = page_soup.get_text(" ", strip=True)

        # Resolve name: prefer <h1>, fall back to hint from listing
        name = hint_name.strip()
        h1 = page_soup.find("h1")
        if h1 and h1.get_text(strip=True):
            name = h1.get_text(strip=True)

        # Depth
        depth = _parse_depth_ft(page_text)

        # Coordinates — CLUE rarely publishes these publicly
        coords = _parse_coords_from_text(page_text)
        if coords is None:
            log.debug("CLUE: no coords for '%s' — skipping", name)
            continue

        lat, lon = coords
        if not _in_gl_bbox(lat, lon):
            continue

        records.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "source": "clueshipwrecks.org",
            "feature_type": "Wreck",
            "description": page_url,
        })
        log.info("CLUE: %-40s  %.4f  %.4f", name, lat, lon)

    log.info("CLUE: %d records with coordinates (of %d pages)", len(records), len(detail_links))
    return records


# ── Source 4: Thunder Bay NOAA National Marine Sanctuary ─────────────────────

_TB_MIN_SEP = r"[\u2019'\u2032]"  # right single quote U+2019, apostrophe, prime
_THUNDERBAY_GPS_RE = re.compile(
    r"([NS])\s*(\d+)[°\xb0]\s*(\d+\.\d+)" + _TB_MIN_SEP + r"\s+([EW])\s*(\d+)[°\xb0]\s*(\d+\.\d+)" + _TB_MIN_SEP,
    re.IGNORECASE,
)
_THUNDERBAY_GPS_ASCII = re.compile(
    r"GPS\s+Location.{0,100}?([NS])\s*(\d+)[°\xb0]\s*(\d+\.\d+)" + _TB_MIN_SEP + r"\s+([EW])\s*(\d+)[°\xb0]\s*(\d+\.\d+)" + _TB_MIN_SEP,
    re.IGNORECASE,
)
_THUNDERBAY_DEPTH_RE = re.compile(
    r"Depth[:\s]+(\d+(?:\.\d+)?)\s*(Feet|Meters|ft|m)\b",
    re.IGNORECASE,
)


def _parse_ddm(ns: str, deg: str, dec_min: str, ew: str, deg2: str, dec_min2: str) -> tuple[float, float]:
    """Convert Degrees Decimal-Minutes (DDM) strings to decimal degrees."""
    lat = float(deg) + float(dec_min) / 60.0
    lon = float(deg2) + float(dec_min2) / 60.0
    if ns.upper() == "S":
        lat = -lat
    if ew.upper() == "W":
        lon = -lon
    return lat, lon


def scrape_thunderbay(cache_dir: Path, dry_run: bool = False) -> list[dict]:
    """
    Scrape Thunder Bay National Marine Sanctuary (NOAA) wreck pages.
    URL: https://thunderbay.noaa.gov/shipwrecks/

    Each wreck page contains a structured 'GPS Location: N45°19.396' W83°27.508''
    field.  All wrecks are in Lake Huron near Alpena, Michigan.
    Referenced via michigan.org Michigan Underwater Preserves article.
    """
    if BeautifulSoup is None:
        log.error("beautifulsoup4 not installed — run: pip install beautifulsoup4 lxml")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # Fetch listing page
    list_cache = cache_dir / "thunderbay_list.html"
    if list_cache.exists():
        listing_bytes = list_cache.read_bytes()
    else:
        log.info("Fetching Thunder Bay wreck listing…")
        try:
            listing_bytes = _fetch_url(THUNDERBAY_LIST_URL)
            list_cache.write_bytes(listing_bytes)
        except Exception as exc:
            log.error("Thunder Bay listing fetch failed: %s", exc)
            return []

    soup = BeautifulSoup(listing_bytes, "lxml")

    wreck_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/shipwrecks/" in href and href != "/shipwrecks/" and not href.endswith("-list.html"):
            name_text = a.get_text(strip=True)
            full = (
                f"{THUNDERBAY_BASE}{href}"
                if href.startswith("/")
                else href
            )
            # Deduplicate (listing may have duplicate links)
            if full not in [u for u, _ in wreck_links]:
                wreck_links.append((full, name_text))

    log.info("Thunder Bay: found %d wreck detail pages", len(wreck_links))

    for page_url, hint_name in wreck_links:
        slug = re.sub(r"[^\w]", "_", page_url.rstrip("/").split("/")[-1])
        page_cache = cache_dir / f"thunderbay_{slug[:80]}.html"
        if page_cache.exists():
            page_bytes = page_cache.read_bytes()
        else:
            try:
                page_bytes = _fetch_url(page_url)
                page_cache.write_bytes(page_bytes)
                time.sleep(1.0)  # 1 s between requests to NOAA
            except Exception as exc:
                log.warning("Thunder Bay: failed to fetch %s: %s", page_url, exc)
                continue

        page_soup = BeautifulSoup(page_bytes, "lxml")
        page_text = page_soup.get_text(" ", strip=True)

        # ── Name ──────────────────────────────────────────────────────
        name = hint_name.strip()
        h1 = page_soup.find("h1")
        if h1 and h1.get_text(strip=True):
            # e.g. "Albany" or "Albany | Thunder Bay National Marine Sanctuary"
            name = h1.get_text(strip=True).split("|")[0].strip()

        # ── GPS Location ──────────────────────────────────────────────
        # Expect: "GPS Location: N45°19.396' W83°27.508'"
        coords: Optional[tuple[float, float]] = None
        m = _THUNDERBAY_GPS_ASCII.search(page_text)
        if m:
            coords = _parse_ddm(*m.groups())
        else:
            # Fallback: bare DDM pattern anywhere on page
            m2 = _THUNDERBAY_GPS_RE.search(page_text)
            if m2:
                coords = _parse_ddm(*m2.groups())

        if coords is None:
            log.debug("Thunder Bay: no GPS found for '%s' — skipping", name)
            continue

        lat, lon = coords
        if not _in_gl_bbox(lat, lon):
            log.debug("Thunder Bay: out-of-bbox '%s'  %.4f  %.4f", name, lat, lon)
            continue

        # ── Depth ─────────────────────────────────────────────────────
        depth: Optional[float] = None
        dm = _THUNDERBAY_DEPTH_RE.search(page_text)
        if dm:
            val = float(dm.group(1))
            unit = dm.group(2).lower()
            depth = val * 0.3048 if unit.startswith("f") else val

        records.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "source": "ThunderBay_NOAA",
            "feature_type": "Wreck",
            "description": page_url,
        })
        log.info("Thunder Bay: %-40s  %.4f  %.4f  depth=%.1fm",
                 name, lat, lon, depth or 0)

    log.info("Thunder Bay: %d wrecks with GPS collected", len(records))
    return records


# ── Main ingestion flow ───────────────────────────────────────────────────────

def _run_ingest(
    records: list[dict],
    conn: sqlite3.Connection,
    existing: list[dict],
    dry_run: bool,
) -> tuple[int, int]:
    inserted = skipped = 0
    for rec in records:
        name = rec["name"].strip()
        lat  = rec["lat"]
        lon  = rec["lon"]
        if not name or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            skipped += 1
            continue
        if _is_duplicate(name, lat, lon, existing):
            log.debug("Duplicate skipped: %s  %.4f  %.4f", name, lat, lon)
            skipped += 1
            continue
        ok = _insert_record(
            conn,
            name=name,
            lat=lat,
            lon=lon,
            depth=rec.get("depth"),
            source_tag=rec["source"],
            feature_type=rec.get("feature_type", "Wreck"),
            description=rec.get("description"),
            dry_run=dry_run,
        )
        if ok:
            inserted += 1
            # Add to seen set so intra-batch dedup works
            existing.append({"name": name.upper(), "lat": lat, "lon": lon})
    return inserted, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest AWOIS + dive site data into wrecks.db")
    ap.add_argument(
        "--source",
        choices=["awois", "3d", "clue", "thunderbay", "all"],
        default="all",
        help="Which source(s) to run (default: all)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print without writing")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "awois_cache",
        help="Directory for cached raw downloads",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    existing = _load_existing(conn)
    log.info("Loaded %d existing features for dedup check", len(existing))

    total_inserted = total_skipped = 0

    if args.source in ("awois", "all"):
        records = scrape_awois(args.cache_dir)
        ins, skp = _run_ingest(records, conn, existing, args.dry_run)
        log.info("AWOIS: inserted=%d  skipped=%d", ins, skp)
        total_inserted += ins
        total_skipped  += skp

    if args.source in ("3d", "all"):
        records = scrape_3dshipwrecks(args.cache_dir)
        ins, skp = _run_ingest(records, conn, existing, args.dry_run)
        log.info("3dshipwrecks.org: inserted=%d  skipped=%d", ins, skp)
        total_inserted += ins
        total_skipped  += skp

    if args.source in ("clue", "all"):
        records = scrape_cluesite(args.cache_dir)
        ins, skp = _run_ingest(records, conn, existing, args.dry_run)
        log.info("clueshipwrecks.org: inserted=%d  skipped=%d", ins, skp)
        total_inserted += ins
        total_skipped  += skp

    if args.source in ("thunderbay", "all"):
        records = scrape_thunderbay(args.cache_dir)
        ins, skp = _run_ingest(records, conn, existing, args.dry_run)
        log.info("ThunderBay_NOAA: inserted=%d  skipped=%d", ins, skp)
        total_inserted += ins
        total_skipped  += skp

    if not args.dry_run:
        conn.commit()
        log.info("Committed. Total inserted=%d  skipped=%d", total_inserted, total_skipped)
    else:
        log.info("[DRY-RUN] Would insert=%d  would skip=%d", total_inserted, total_skipped)

    conn.close()

    # Re-check count
    conn2 = sqlite3.connect(str(DB_PATH))
    count = conn2.execute(
        "SELECT COUNT(*) FROM features WHERE coord_quality='dive_verified'"
    ).fetchone()[0]
    conn2.close()
    log.info("dive_verified records in DB now: %d", count)


if __name__ == "__main__":
    main()
