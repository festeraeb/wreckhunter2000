#!/usr/bin/env python3
"""
Lake Erie October 2015 — Daily Multi-Sensor Scan
=================================================

Primary missions
----------------
1. HYDROCARBON / OIL LEAK TIMELINE
   Scan the full lake (bbox 41.30–42.50°N, 83.50–78.80°W) for every available
   granule in October 2015.  Identifies the earliest date and source coordinates
   of hydrocarbon / oil anomalies to pinpoint when the leak started.

2. MARQUETTE AND BESSEMER No. 2 WRECK SEARCH  (central basin)
   Marquette-class railroad car ferry, lost 18 November 1909 in a violent storm
   while crossing from Conneaut, OH → Port Stanley, ON.  All 33 crew lost.
   Partially silt-buried in the central basin.  Debris found on both the
   Pennsylvania/Ohio (south) and Ontario (north) shores.

Detection strategy per mission
-------------------------------
HC LEAK — full lake:
  PASS 1  Standard optical / thermal / SAR anomaly (B02, B04, B10, VV/VH)
  PASS 2  Hydrocarbon — B11 SWIR dark (z < −1.8) + B04 Red bright (z > 1.5)
  PASS 3  Stumpf log-ratio bathymetric shallow-anomaly (B02/B03)
  PASS 4  NauticUVs LoG blob scan (B02 + B10)

M&B2 — central basin [41.8, −82.5, 42.5, −80.0]:
  PASS 5  SWIR silt erasure — B11/B12 ratio (sub-silt ferrous hull / rail frames)
  PASS 6  Mussel clear-spot — B02 Blue elevated oval in turbid Erie background
           (zebra/quagga mussels filter water → locally clearer column → higher B02)
  All passes PASs 1-4 also constrained to M&B2 bbox and tagged separately

Output
------
  outputs/erie_oct2015/
    daily/<YYYY-MM-DD>.kmz       one KMZ per granule date
    erie_oct2015_combined.kmz    all detections merged (Google Earth)
    hydrocarbon_timeline.json    first-detection date + source coords
    mb2_candidates.json          M&B2 zone detections sorted by confidence

REQUIRES
--------
  lake_michigan_scan.py  (processing engine, same directory)
  downloads/erie/2015/10/ (HLS granules from download_erie_multiyear.py)
"""

import os
import sys
import json
import math
import re
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict
import simplekml

# UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ── Import the CESAROPS processing engine ────────────────────────────────────
# All band-processing functions live in lake_michigan_scan.py and are
# sensor-agnostic — they work on any raster file regardless of lake.
print("[INIT] Loading CESAROPS processing engine from lake_michigan_scan...", flush=True)
try:
    from lake_michigan_scan import (        # noqa: E402
        process_hydrocarbon_bands,
        process_tiff_with_coords,
        compute_nauticuvs_pass,
        compute_stumpf_pass,
        KNOWN_WRECKS,
        KNOWN_WRECK_RADIUS_DEG,
        _flag_known_wreck,
        _is_linear_wake,
        HAS_GPU,
    )
    print(f"[INIT] Engine loaded  GPU={HAS_GPU}", flush=True)
except ImportError as _ie:
    print(f"[INIT] FATAL: Could not import lake_michigan_scan: {_ie}")
    print("[INIT] Make sure lake_michigan_scan.py is in the same directory.")
    sys.exit(1)

import rasterio
from rasterio.warp import transform as warp_transform

# ── Lake Erie constants ───────────────────────────────────────────────────────
ERIE_BBOX = [41.30, -83.50, 42.50, -78.80]    # [lat_min, lon_min, lat_max, lon_max]

# Central basin — Marquette and Bessemer No. 2 search zone
# Crossing: Conneaut, OH (41.95°N, 80.56°W) → Port Stanley, ON (42.67°N, 81.22°W)
# Storm-drift east + debris on both shores → likely east of midpoint
MB2_SEARCH_BBOX  = [41.80, -82.50, 42.50, -80.00]
MB2_CENTROID_LAT = 42.15
MB2_CENTROID_LON = -81.25
MB2_SEARCH_RADIUS_DEG = 1.0          # broad search — exact position unknown

# Erie calibration anchors (lighthouses / harbour structures)
ERIE_ANCHORS = {
    "buffalo_outer_harbor":  {"name": "Buffalo Outer Harbor Light",   "lat": 42.8639, "lon": -78.8889},
    "erie_land_lighthouse":  {"name": "Erie Land Lighthouse (PA)",    "lat": 42.1334, "lon": -80.0878},
    "cleveland_harbor_west": {"name": "Cleveland Harbor West Pier",   "lat": 41.5136, "lon": -81.7479},
    "lorain_harbor":         {"name": "Lorain Lighthouse",            "lat": 41.4855, "lon": -82.1834},
    "marblehead_lighthouse": {"name": "Marblehead Lighthouse",        "lat": 41.5367, "lon": -82.7264},
    "port_colborne_on":      {"name": "Port Colborne East Pier (ON)", "lat": 42.8789, "lon": -79.2497},
    "point_pelee_on":        {"name": "Point Pelee (ON)",             "lat": 41.9583, "lon": -82.5150},
    "long_point_on":         {"name": "Long Point (ON)",              "lat": 42.5667, "lon": -80.2833},
    "presque_isle_erie_pa":  {"name": "Presque Isle Lighthouse (PA)", "lat": 42.1670, "lon": -80.1000},
}

OUTPUT_DIR = Path(__file__).parent / 'outputs' / 'erie_oct2015'


# ── M&B2 zone flag (analogous to _flag_line5 in lake_michigan_scan) ──────────
def _flag_mb2_zone(lat: float, lon: float) -> bool:
    """Return True if coordinate falls inside the M&B2 central-basin search zone."""
    b = MB2_SEARCH_BBOX  # [lat_min, lon_min, lat_max, lon_max]
    return b[0] <= lat <= b[2] and b[1] <= lon <= b[3]


# ── HLS / Sentinel-2 filename date parser ────────────────────────────────────
def extract_date_from_path(p: Path):
    """
    Extract a datetime.date from an HLS filename or its parent directory hierarchy.

    HLS naming:  HLS.S30.T17TLD.2015274T161600.v2.0.B11.tif
                                     ^^^^^^^ YYYYDOY

    Also handles:  downloads/erie/2015/10/DD/<file>
    and:           various YYYYMMDD patterns in the filename.
    """
    name = p.name

    # HLS pattern: 7-digit block YYYYDOY
    m = re.search(r'\.(\d{4})(\d{3})T\d{6}\.', name)
    if m:
        try:
            year = int(m.group(1))
            doy  = int(m.group(2))
            return date(year, 1, 1) + timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            pass

    # YYYYMMDD pattern in filename
    m = re.search(r'(\d{4})(\d{2})(\d{2})', name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Try to get date from parent-directory structure: .../2015/10/[DD]/
    parts = p.parts
    for i, part in enumerate(parts):
        if part == '2015' and i + 1 < len(parts):
            try:
                month = int(parts[i + 1])
                day = int(parts[i + 2]) if i + 2 < len(parts) and parts[i + 2].isdigit() else 1
                return date(2015, month, day)
            except (ValueError, IndexError):
                pass

    return None


# ── SWIR silt-erasure pass (M&B2 specific) ───────────────────────────────────
def detect_swir_silt_erasure(b11_path: Path, b12_path: Path,
                              scan_bbox=None, top_n: int = 30) -> list:
    """
    B11 / B12 ratio to reveal sub-silt ferrous metal (M&B2 hull + rail-car frames).

    Physics:
      - B11 (SWIR1, 1565nm): moderate SWIR penetration, sensitive to soil moisture and
        iron-oxide mineral content.
      - B12 (SWIR2, 2190nm): deep SWIR, highly sensitive to clay mineral composition.
      - B11/B12 ratio over normal silt: ~0.9–1.1 (dominated by clay/silt mineralogy).
      - B11/B12 over exposed or near-surface ferrous steel: anomalously high (>1.3).
        The welded steel hull and rail-car steel beams differ markedly from surrounding
        lacustrine silt.  Even under a thin silt blanket the ratio is perturbed.
      - A compact elevated-ratio blob in an otherwise uniform silted basin = candidate
        sub-silt metallic structure — tagged MB2_SWIR_SILT_ERASURE.

    Returns list of detection dicts.
    """
    if not b11_path.exists():
        print(f"    [SWE] B11 not found: {b11_path.name}")
        return []
    if not b12_path or not b12_path.exists():
        print(f"    [SWE] B12 not found (needed for silt erasure) — skipping")
        return []

    print(f"  [SWE] SWIR silt-erasure pass: {b11_path.name}")

    with rasterio.open(b11_path) as src11:
        b11 = src11.read(1).astype(np.float32)
        crs = src11.crs

    with rasterio.open(b12_path) as src12:
        b12 = src12.read(1).astype(np.float32)

    # Align shapes — B11 @ 20m, B12 @ 20m in HLS (both 20m native)
    if b12.shape != b11.shape:
        import skimage.transform as skt
        b12 = skt.resize(b12, b11.shape, order=1, anti_aliasing=True,
                         preserve_range=True).astype(np.float32)

    # Compute ratio, masking nodata
    valid = (b11 > 0) & (b12 > 0) & np.isfinite(b11) & np.isfinite(b12)
    ratio = np.full_like(b11, np.nan)
    ratio[valid] = b11[valid] / (b12[valid] + 1e-6)

    valid_ratio = valid & np.isfinite(ratio)
    ratio_vals = ratio[valid_ratio]
    if ratio_vals.size < 100:
        return []

    mean_r = float(np.nanmean(ratio_vals))
    std_r  = float(np.nanstd(ratio_vals))
    if std_r < 1e-6:
        return []

    z_ratio = np.full_like(ratio, np.nan)
    z_ratio[valid_ratio] = (ratio[valid_ratio] - mean_r) / std_r

    # Elevated ratio = anomalous mineral/metal signature
    threshold = 2.5
    anomaly_mask = (z_ratio > threshold) & valid_ratio

    # Apply spatial bbox filter
    rows, cols = np.where(anomaly_mask)
    if scan_bbox is not None and len(rows) > 0:
        with rasterio.open(b11_path) as src:
            lat_min, lon_min, lat_max, lon_max = scan_bbox
            xs_4326 = [lon_min, lon_min, lon_max, lon_max]
            ys_4326 = [lat_min, lat_max, lat_min, lat_max]
            xs_c, ys_c = warp_transform('EPSG:4326', src.crs, xs_4326, ys_4326)
            bb_rows, bb_cols = [], []
            for xc, yc in zip(xs_c, ys_c):
                try:
                    r, c = src.index(xc, yc)
                    bb_rows.append(r); bb_cols.append(c)
                except Exception:
                    pass
        if bb_rows:
            r_min = max(0, min(bb_rows)); r_max = min(b11.shape[0]-1, max(bb_rows))
            c_min = max(0, min(bb_cols)); c_max = min(b11.shape[1]-1, max(bb_cols))
            if r_min > r_max: r_min, r_max = r_max, r_min
            if c_min > c_max: c_min, c_max = c_max, c_min
            in_bbox = ((rows >= r_min) & (rows <= r_max) &
                       (cols >= c_min) & (cols <= c_max))
            rows = rows[in_bbox]; cols = cols[in_bbox]

    print(f"    [SWE] Elevated B11/B12 anomalies (z>{threshold}): {len(rows)}")
    if len(rows) == 0:
        return []

    zvals = z_ratio[rows, cols]
    sort_idx = np.argsort(-zvals)[:top_n]

    detections = []
    with rasterio.open(b11_path) as src:
        for idx in sort_idx:
            r = int(rows[idx]); c = int(cols[idx])
            local_x, local_y = src.xy(r, c)
            lon_pt, lat_pt = warp_transform(src.crs, 'EPSG:4326', [local_x], [local_y])
            lat_v = lat_pt[0]; lon_v = lon_pt[0]
            z_v   = float(z_ratio[r, c])
            raw_r = float(ratio[r, c])
            known = _flag_known_wreck(lat_v, lon_v)
            in_mb2 = _flag_mb2_zone(lat_v, lon_v)
            detections.append({
                "lat":                  lat_v,
                "lon":                  lon_v,
                "zscore":               z_v,
                "type":                 "swir_silt_erasure",
                "source":               b11_path.name,
                "b11_b12_ratio":        round(raw_r, 4),
                "mb2_zone":             in_mb2,
                "mb2_subtype":          "MB2_SWIR_SILT_ERASURE" if in_mb2 else None,
                "known_wreck_hit":      known["id"]   if known else None,
                "known_wreck_name":     known["name"] if known else None,
                "pixel":                {"row": r, "col": c},
            })
    return detections


# ── Mussel clear-spot detection (M&B2 calm-day signature) ────────────────────
def detect_mussel_clearspot(b02_path: Path, scan_bbox=None, top_n: int = 30) -> list:
    """
    Detect mussel clear-spots above the M&B2 wreck.

    Physics:
      Lake Erie central basin has high background turbidity (silt + cyanobacteria).
      Dense zebra/quagga mussel colonies filter the water column above them → locally
      clearer water → elevated B02 Blue reflectance relative to the turbid background.
      The clear-spot appears as a positive B02 z-score blob of 50–300m diameter on
      calm, low-wind days.  On days with wind >15 kt the signature is washed out by
      wave mixing — it is most reliable during calm windows.

    The function computes a standard positive-anomaly (elevated blue) z-score on B02
    within the M&B2 search zone.  Results are tagged MB2_MUSSEL_CLEARSPOT.
    """
    if not b02_path.exists():
        print(f"    [MCS] B02 not found: {b02_path.name}")
        return []

    print(f"  [MCS] Mussel clear-spot scan (B02 Erie turbidity window): {b02_path.name}")

    with rasterio.open(b02_path) as src:
        data = src.read(1).astype(np.float32)
        crs = src.crs

    nodata = (data <= 0) | ~np.isfinite(data)
    data[nodata] = np.nan
    valid = ~nodata & np.isfinite(data)
    valid_vals = data[valid]
    if valid_vals.size < 200:
        return []

    mean_v = float(np.nanmean(valid_vals))
    std_v  = float(np.nanstd(valid_vals))
    if std_v < 1e-6:
        return []

    z = np.full_like(data, np.nan)
    z[valid] = (data[valid] - mean_v) / std_v

    # Positive anomaly (clear / bright relative to turbid Erie background)
    threshold = 2.0
    anomaly_mask = (z > threshold) & valid

    rows, cols = np.where(anomaly_mask)
    if scan_bbox is not None and len(rows) > 0:
        with rasterio.open(b02_path) as src:
            lat_min, lon_min, lat_max, lon_max = scan_bbox
            xs_4326 = [lon_min, lon_min, lon_max, lon_max]
            ys_4326 = [lat_min, lat_max, lat_min, lat_max]
            xs_c, ys_c = warp_transform('EPSG:4326', src.crs, xs_4326, ys_4326)
            bb_rows, bb_cols = [], []
            for xc, yc in zip(xs_c, ys_c):
                try:
                    r, c = src.index(xc, yc)
                    bb_rows.append(r); bb_cols.append(c)
                except Exception:
                    pass
        if bb_rows:
            r_min = max(0, min(bb_rows)); r_max = min(data.shape[0]-1, max(bb_rows))
            c_min = max(0, min(bb_cols)); c_max = min(data.shape[1]-1, max(bb_cols))
            if r_min > r_max: r_min, r_max = r_max, r_min
            if c_min > c_max: c_min, c_max = c_max, c_min
            in_bbox = ((rows >= r_min) & (rows <= r_max) &
                       (cols >= c_min) & (cols <= c_max))
            rows = rows[in_bbox]; cols = cols[in_bbox]

    print(f"    [MCS] Clear-spot B02 anomalies (z>{threshold}, in M&B2 bbox): {len(rows)}")
    if len(rows) == 0:
        return []

    zvals = z[rows, cols]
    sort_idx = np.argsort(-zvals)[:top_n]

    detections = []
    with rasterio.open(b02_path) as src:
        for idx in sort_idx:
            r = int(rows[idx]); c = int(cols[idx])
            local_x, local_y = src.xy(r, c)
            lon_pt, lat_pt = warp_transform(src.crs, 'EPSG:4326', [local_x], [local_y])
            lat_v = lat_pt[0]; lon_v = lon_pt[0]
            z_v   = float(z[r, c])
            known = _flag_known_wreck(lat_v, lon_v)
            in_mb2 = _flag_mb2_zone(lat_v, lon_v)
            detections.append({
                "lat":              lat_v,
                "lon":              lon_v,
                "zscore":           z_v,
                "type":             "mussel_clearspot",
                "source":           b02_path.name,
                "mb2_zone":         in_mb2,
                "mb2_subtype":      "MB2_MUSSEL_CLEARSPOT" if in_mb2 else None,
                "known_wreck_hit":  known["id"]   if known else None,
                "known_wreck_name": known["name"] if known else None,
                "pixel":            {"row": r, "col": c},
            })
    return detections


# ── KMZ output (Erie-specific folders) ───────────────────────────────────────
def create_erie_kmz(detections: list, output_path: Path, title: str = "Lake Erie Scan"):
    """Create KMZ for Google Earth with Erie/M&B2-specific folder structure."""
    kml = simplekml.Kml()

    # Reference anchors
    anchor_folder = kml.newfolder(name="Erie Calibration Anchors")
    for key, a in ERIE_ANCHORS.items():
        pnt = anchor_folder.newpoint(name=a["name"], coords=[(a["lon"], a["lat"])])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"
        pnt.style.iconstyle.scale = 1.2
        pnt.description = f"<b>Calibration Anchor</b><br/>Lat: {a['lat']:.5f}<br/>Lon: {a['lon']:.5f}"

    # M&B2 centroid reference
    mb2_ref = kml.newfolder(name="M&B2 Reference (probable centroid)")
    pnt = mb2_ref.newpoint(
        name="M&B2 Probable Centroid",
        coords=[(MB2_CENTROID_LON, MB2_CENTROID_LAT)]
    )
    pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"
    pnt.style.iconstyle.scale = 1.5
    pnt.description = (
        "<b>Marquette and Bessemer No. 2</b><br/>"
        "Marquette-class car ferry. Lost 18 Nov 1909.<br/>"
        "Conneaut OH → Port Stanley ON crossing.<br/>"
        "Debris: both Ontario (N) and PA/OH (S) shores.<br/>"
        "Approx central basin centroid — location unconfirmed.<br/>"
        f"Lat: {MB2_CENTROID_LAT}  Lon: {MB2_CENTROID_LON}"
    )

    # Known wreck reference positions
    known_ref_folder = kml.newfolder(name="Known Wreck Reference Positions (Erie)")
    for w in KNOWN_WRECKS:
        if w.get("lake") == "erie" or w.get("id") == "marquette_bessemer_2":
            pnt = known_ref_folder.newpoint(
                name=f"{w['name']} ({w.get('year_lost', '?')})",
                coords=[(w["lon"], w["lat"])]
            )
            pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"
            pnt.style.iconstyle.scale = 1.2
            pnt.description = (
                f"<b>{w['name']}</b><br/>Year: {w.get('year_lost', '?')}<br/>"
                f"Depth: {w.get('depth_ft', '?')} ft<br/>"
                f"Type: {w.get('type', '?')}<br/>"
                f"Lat: {w['lat']:.5f}  Lon: {w['lon']:.5f}"
            )

    # Detection folders
    hc_folder       = kml.newfolder(name="Hydrocarbon Detections — Oil Leak Timeline")
    mb2_folder      = kml.newfolder(name="M&B2 Wreck Candidates (central basin)")
    swe_folder      = kml.newfolder(name="M&B2 — SWIR Silt Erasure (sub-silt metal)")
    mcs_folder      = kml.newfolder(name="M&B2 — Mussel Clear-Spot (calm day)")
    known_hit_folder= kml.newfolder(name="Known Wreck HITS (ground-truth calibration)")
    wreck_folder    = kml.newfolder(name="Wreck Candidates (general anomaly)")

    for det in detections:
        lat = det["lat"]; lon = det["lon"]
        z   = det.get("zscore", 0.0)
        det_type = det.get("type", "optical")
        known    = det.get("known_wreck_hit")
        hc_sub   = det.get("hc_subtype", "")
        mb2_sub  = det.get("mb2_subtype", "")
        in_mb2   = det.get("mb2_zone", _flag_mb2_zone(lat, lon))
        scan_date = det.get("scan_date", "")

        desc = (
            f"<b>{det_type.upper()}</b><br/>"
            f"Z-Score: {z:.3f}<br/>"
            f"Date: {scan_date}<br/>"
            f"Source: {det.get('source', '?')}<br/>"
            + (f"<b>KNOWN WRECK: {det.get('known_wreck_name')}</b><br/>" if known else "")
            + (f"HC Subtype: {hc_sub}<br/>" if hc_sub else "")
            + (f"M&B2 Subtype: {mb2_sub}<br/>" if mb2_sub else "")
            + f"<br/>Lat: {lat:.6f}<br/>Lon: {lon:.6f}"
        )

        if known:
            folder = known_hit_folder
            icon   = "http://maps.google.com/mapfiles/kml/paddle/wht-stars.png"
            color  = simplekml.Color.cyan
            label  = f"KNOWN HIT {det.get('known_wreck_name','?')} Z={z:.2f}"
        elif det_type == "hydrocarbon":
            folder = hc_folder
            icon   = "http://maps.google.com/mapfiles/kml/paddle/pink-blank.png"
            color  = simplekml.Color.fuchsia
            label  = f"HC {hc_sub} {scan_date} Z={z:.2f}"
        elif det_type == "swir_silt_erasure" and in_mb2:
            folder = swe_folder
            icon   = "http://maps.google.com/mapfiles/kml/paddle/orange-blank.png"
            color  = simplekml.Color.orange
            label  = f"SWE-SILT {scan_date} Z={z:.2f}"
        elif det_type == "mussel_clearspot" and in_mb2:
            folder = mcs_folder
            icon   = "http://maps.google.com/mapfiles/kml/paddle/ltblu-blank.png"
            color  = simplekml.Color.lightblue
            label  = f"MCS {scan_date} Z={z:.2f}"
        elif in_mb2:
            folder = mb2_folder
            icon   = "http://maps.google.com/mapfiles/kml/paddle/purple-blank.png"
            color  = simplekml.Color.purple
            label  = f"MB2-ZONE {det_type.upper()} Z={z:.2f}"
        else:
            folder = wreck_folder
            icon   = ("http://maps.google.com/mapfiles/kml/paddle/red-stars.png" if abs(z) > 4
                      else "http://maps.google.com/mapfiles/kml/paddle/orange-circle.png" if abs(z) > 3
                      else "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")
            color  = (simplekml.Color.red if abs(z) > 4 else
                      simplekml.Color.orange if abs(z) > 3 else simplekml.Color.yellow)
            label  = f"{det_type.upper()} Z={z:.2f}"

        pnt = folder.newpoint(name=label, coords=[(lon, lat)])
        pnt.style.iconstyle.icon.href = icon
        pnt.style.iconstyle.color = color
        pnt.description = desc

    kml.save(str(output_path))
    print(f"  [KMZ] Saved: {output_path}")


# ── Timeline analysis ─────────────────────────────────────────────────────────
def build_hydrocarbon_timeline(all_detections: list) -> dict:
    """
    From all hydrocarbon detections across all dates, find:
      - First date of any HC anomaly (leak onset candidate)
      - Centroid coordinates of first-detection cluster
      - Day-by-day count to show spread/escalation
    """
    by_date = defaultdict(list)
    for det in all_detections:
        if det.get("type") == "hydrocarbon":
            d = det.get("scan_date")
            if d:
                by_date[d].append(det)

    if not by_date:
        return {"first_detection_date": None, "source_coords": None, "daily_counts": {}}

    sorted_dates = sorted(by_date.keys())
    first_date   = sorted_dates[0]
    first_dets   = by_date[first_date]

    # Centroid of first-day detections (weighted by |z-score|)
    lats = [d["lat"] for d in first_dets]
    lons = [d["lon"] for d in first_dets]
    ws   = [abs(d.get("zscore", 1.0)) for d in first_dets]
    total_w = sum(ws) or 1.0
    cent_lat = sum(la * w for la, w in zip(lats, ws)) / total_w
    cent_lon = sum(lo * w for lo, w in zip(lons, ws)) / total_w

    daily_counts = {d: len(dets) for d, dets in sorted_by_date_items(by_date)}

    return {
        "first_detection_date":     first_date,
        "first_detection_count":    len(first_dets),
        "source_coords":            {"lat": round(cent_lat, 5), "lon": round(cent_lon, 5)},
        "total_hc_detections":      sum(len(v) for v in by_date.values()),
        "daily_counts":             daily_counts,
        "all_dates_with_hc":        sorted_dates,
    }


def sorted_by_date_items(d: dict):
    return sorted(d.items(), key=lambda x: x[0])


# ── Main scan loop ───────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("LAKE ERIE — OCTOBER 2015 DAILY HYDROCARBON + M&B2 WRECK SCAN")
    print("=" * 80)
    print()
    print(f"  Lake Erie bbox:     {ERIE_BBOX}")
    print(f"  M&B2 search bbox:   {MB2_SEARCH_BBOX}")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'daily').mkdir(exist_ok=True)

    # ── Discover TIFFs ────────────────────────────────────────────────────────
    repo_root = Path(__file__).parent
    search_paths = [
        repo_root / 'downloads' / 'erie',
        repo_root / 'downloads' / 'hls',
        Path(os.environ.get('CESAROPS_DATA_DIR', repo_root / 'data')),
    ]
    tiffs = []
    for sp in search_paths:
        if sp.exists():
            tiffs.extend(sp.rglob("*.tif"))
    tiffs = sorted(set(tiffs))

    # Filter to October 2015 only (date extraction from filename / path)
    oct2015_tiffs = []
    for t in tiffs:
        d = extract_date_from_path(t)
        if d and d.year == 2015 and d.month == 10:
            oct2015_tiffs.append((d, t))
    oct2015_tiffs.sort(key=lambda x: x[0])

    print(f"Found {len(tiffs)} total TIFFs")
    print(f"Filtered to October 2015: {len(oct2015_tiffs)} TIFFs")

    # If no Oct-2015-specific TIFFs found via date extraction, fall back to all
    # downloads/erie TIFFs and process them (useful when directory is not dated)
    if not oct2015_tiffs:
        erie_path = repo_root / 'downloads' / 'erie'
        if erie_path.exists():
            fallback = sorted(erie_path.rglob("*.tif"))
            if fallback:
                print(f"  [fallback] Using all {len(fallback)} TIFF(s) in downloads/erie/")
                oct2015_tiffs = [(date(2015, 10, 1), t) for t in fallback]
        if not oct2015_tiffs:
            print("\n  [WARNING] No October 2015 TIFFs found.")
            print("  Run download_erie_multiyear.py first to download the granules.")
            print("  (2015-October is configured for max_results=50 — all available.)\n")

    # Group by date
    by_date: dict[date, list] = defaultdict(list)
    for dt, tf in oct2015_tiffs:
        by_date[dt].append(tf)

    all_detections = []
    daily_summaries = []

    # Skip-band tags shared with lake_michigan_scan Pass 1
    _SKIP_UPPER = {'FMASK', '.B11.', '.SWIR16.', '.SWIR22.',
                   '.SCL.', '.QA_PIXEL.', '.NIR08.', '.NIR.'}

    scan_dates = sorted(by_date.keys())
    print(f"\nScanning {len(scan_dates)} date group(s)...")

    for scan_date in scan_dates:
        tiffs_today = by_date[scan_date]
        date_str = str(scan_date)
        print(f"\n{'─'*70}")
        print(f"  DATE: {date_str}  ({len(tiffs_today)} TIFF(s))")
        print(f"{'─'*70}")

        day_detections = []

        # ── PASS 1: Standard anomaly scan ────────────────────────────────────
        standard_tiffs = [t for t in tiffs_today if
                          not any(tag in t.name.upper() for tag in _SKIP_UPPER)]
        print(f"\n  PASS 1 — Standard anomaly scan ({len(standard_tiffs)} bands)")
        for tiff in standard_tiffs:
            tname = tiff.name.upper()
            is_thermal = 'B10' in tname or 'THERMAL' in tname or 'LWIR' in tname or 'ST_B10' in tname
            is_blue    = '.B02.' in tname or '.BLUE.' in tname
            if is_thermal:
                thresh, cold_sink = 2.0, True
            elif is_blue:
                thresh, cold_sink = 1.2, False
            else:
                thresh, cold_sink = 1.5, False
            try:
                dets = process_tiff_with_coords(
                    tiff, threshold=thresh, scan_bbox=ERIE_BBOX,
                    top_n=200, cold_sink_mode=cold_sink)
                for d in dets:
                    d["scan_date"] = date_str
                    d["mb2_zone"]  = _flag_mb2_zone(d["lat"], d["lon"])
                day_detections.extend(dets)
            except Exception as e:
                print(f"    ERROR: {e}")

        # ── PASS 2: Hydrocarbon B11 SWIR + B04 Red ───────────────────────────
        b11_tiffs = [t for t in tiffs_today if
                     ('.B11.' in t.name.upper() or '.SWIR16.' in t.name.upper()) and
                     'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()]
        print(f"\n  PASS 2 — Hydrocarbon scan ({len(b11_tiffs)} B11 SWIR scene(s))")
        for b11 in b11_tiffs:
            pname = b11.name.lower()
            if '.swir16.tif' in pname:
                b04 = Path(str(b11).replace('.swir16.tif', '.red.tif'))
            else:
                b04 = Path(str(b11).replace('.B11.tif', '.B04.tif'))
            try:
                dets = process_hydrocarbon_bands(b11, b04)
                for d in dets:
                    d["scan_date"] = date_str
                    d["mb2_zone"]  = _flag_mb2_zone(d["lat"], d["lon"])
                    # Reclassify wreck seep near M&B2 as MB2 candidate
                    if d.get("mb2_zone") and d.get("hc_subtype") == "WRECK_SEEP_CANDIDATE":
                        d["hc_subtype"] = "MB2_SEEP_CANDIDATE"
                day_detections.extend(dets)
            except ImportError:
                print("    [HC] scipy not available — skipping (pip install scipy)")
            except Exception as e:
                print(f"    [HC] ERROR: {e}")

        # ── PASS 3: Stumpf bathymetric shallow-anomaly ───────────────────────
        blue_tiffs = [t for t in tiffs_today if
                      ('.B02.' in t.name.upper() or '.BLUE.' in t.name.upper()) and
                      'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()]
        print(f"\n  PASS 3 — Stumpf bathymetric scan ({len(blue_tiffs)} blue band(s))")
        for blue in blue_tiffs:
            pname = blue.name.lower()
            if '.blue.tif' in pname:
                green = Path(str(blue).replace('.blue.tif', '.green.tif'))
            else:
                green = Path(str(blue).replace('.B02.tif', '.B03.tif').replace('B02', 'B03'))
            try:
                dets = compute_stumpf_pass(blue, green, scan_bbox=ERIE_BBOX)
                for d in dets:
                    d["scan_date"] = date_str
                    d["mb2_zone"]  = _flag_mb2_zone(d["lat"], d["lon"])
                day_detections.extend(dets)
            except Exception as e:
                print(f"    [ST] ERROR: {e}")

        # ── PASS 4: NauticUVs LoG blob (B02 + B10) ───────────────────────────
        try:
            from scipy.ndimage import gaussian_laplace  # noqa
            nuv_bands = (
                [t for t in tiffs_today if ('.B02.' in t.name.upper() or '.BLUE.' in t.name.upper())
                 and 'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()] +
                [t for t in tiffs_today if ('B10' in t.name.upper() or 'THERMAL' in t.name.upper())
                 and 'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()]
            )
            print(f"\n  PASS 4 — NauticUVs LoG blob ({len(nuv_bands)} band(s): B02+B10)")
            for nuv_tif in nuv_bands:
                try:
                    # For M&B2 zone: use tight bbox so LoG is constrained
                    dets_erie = compute_nauticuvs_pass(nuv_tif, scan_bbox=ERIE_BBOX, top_n=50)
                    dets_mb2  = compute_nauticuvs_pass(nuv_tif, scan_bbox=MB2_SEARCH_BBOX, top_n=20)
                    for d in dets_erie + dets_mb2:
                        d["scan_date"] = date_str
                        d["mb2_zone"]  = _flag_mb2_zone(d["lat"], d["lon"])
                    day_detections.extend(dets_erie)
                    # M&B2 LoG already included via full-lake (avoid duplicate coords)
                except Exception as e:
                    print(f"    [NUV] ERROR: {e}")
        except ImportError:
            print("  PASS 4 — scipy not available, skipping NauticUVs")

        # ── PASS 5: SWIR silt erasure (M&B2 central basin only) ──────────────
        b12_bands = [t for t in tiffs_today if
                     ('.B12.' in t.name.upper() or '.SWIR22.' in t.name.upper()) and
                     'FMASK' not in t.name.upper()]
        print(f"\n  PASS 5 — SWIR silt erasure B11/B12 (M&B2 zone, {len(b12_bands)} B12 band(s))")
        for b12 in b12_bands:
            pname = b12.name
            b11 = Path(str(b12).replace('.B12.', '.B11.').replace('.swir22.', '.swir16.'))
            try:
                dets = detect_swir_silt_erasure(b11, b12, scan_bbox=MB2_SEARCH_BBOX)
                for d in dets:
                    d["scan_date"] = date_str
                day_detections.extend(dets)
            except Exception as e:
                print(f"    [SWE] ERROR: {e}")

        # ── PASS 6: Mussel clear-spot (M&B2 central basin, B02) ──────────────
        blue_tiffs_mb2 = [t for t in tiffs_today if
                          ('.B02.' in t.name.upper() or '.BLUE.' in t.name.upper()) and
                          'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()]
        print(f"\n  PASS 6 — Mussel clear-spot B02 (M&B2 zone, {len(blue_tiffs_mb2)} blue band(s))")
        for blue in blue_tiffs_mb2:
            try:
                dets = detect_mussel_clearspot(blue, scan_bbox=MB2_SEARCH_BBOX)
                for d in dets:
                    d["scan_date"] = date_str
                day_detections.extend(dets)
            except Exception as e:
                print(f"    [MCS] ERROR: {e}")

        # ── Daily summary ─────────────────────────────────────────────────────
        hc_count  = sum(1 for d in day_detections if d.get("type") == "hydrocarbon")
        mb2_count = sum(1 for d in day_detections if d.get("mb2_zone"))
        print(f"\n  → {len(day_detections)} total detections  "
              f"({hc_count} hydrocarbon,  {mb2_count} in M&B2 zone)")

        # Save daily KMZ
        daily_kmz = OUTPUT_DIR / 'daily' / f'{date_str}.kmz'
        try:
            create_erie_kmz(day_detections, daily_kmz, title=f"Erie {date_str}")
        except Exception as e:
            print(f"    [KMZ] Daily KMZ error: {e}")

        all_detections.extend(day_detections)
        daily_summaries.append({
            "date":              date_str,
            "total_detections":  len(day_detections),
            "hydrocarbon_count": hc_count,
            "mb2_zone_count":    mb2_count,
            "tiff_count":        len(tiffs_today),
        })

    # ── Combined KMZ ─────────────────────────────────────────────────────────
    combined_kmz = OUTPUT_DIR / 'erie_oct2015_combined.kmz'
    try:
        print(f"\n[KMZ] Writing combined KMZ ({len(all_detections)} detections)...")
        create_erie_kmz(all_detections, combined_kmz, title="Erie October 2015 Combined")
    except Exception as e:
        print(f"[KMZ] Combined KMZ error: {e}")

    # ── Hydrocarbon timeline ──────────────────────────────────────────────────
    timeline = build_hydrocarbon_timeline(all_detections)
    timeline["daily_scan_summary"] = daily_summaries
    timeline_path = OUTPUT_DIR / 'hydrocarbon_timeline.json'
    with open(timeline_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Hydrocarbon timeline saved: {timeline_path}")
    if timeline.get("first_detection_date"):
        print(f"    FIRST HC DETECTION:  {timeline['first_detection_date']}")
        sc = timeline.get("source_coords", {})
        print(f"    SOURCE COORDINATES:  {sc.get('lat', '?')}, {sc.get('lon', '?')}")

    # ── M&B2 candidate report ─────────────────────────────────────────────────
    mb2_dets = [d for d in all_detections if (
        d.get("mb2_zone") or
        d.get("type") in ("swir_silt_erasure", "mussel_clearspot") or
        (d.get("hc_subtype", "").startswith("MB2"))
    )]
    # Sort by z-score magnitude descending
    mb2_dets.sort(key=lambda x: abs(x.get("zscore", 0)), reverse=True)
    mb2_path = OUTPUT_DIR / 'mb2_candidates.json'
    with open(mb2_path, 'w', encoding='utf-8') as f:
        json.dump({
            "wreck":        "Marquette and Bessemer No. 2",
            "search_bbox":  MB2_SEARCH_BBOX,
            "centroid_ref": {"lat": MB2_CENTROID_LAT, "lon": MB2_CENTROID_LON},
            "total_candidates": len(mb2_dets),
            "detections":   mb2_dets,
        }, f, indent=2, ensure_ascii=False)
    print(f"[✓] M&B2 candidates saved:    {mb2_path}  ({len(mb2_dets)} detections)")

    # ── Final summary ─────────────────────────────────────────────────────────
    hc_total  = sum(1 for d in all_detections if d.get("type") == "hydrocarbon")
    mb2_total = len(mb2_dets)
    print()
    print("=" * 80)
    print("LAKE ERIE OCTOBER 2015 SCAN COMPLETE")
    print("=" * 80)
    print(f"  Dates scanned:         {len(scan_dates)}")
    print(f"  Total detections:      {len(all_detections)}")
    print(f"  Hydrocarbon:           {hc_total}")
    print(f"  M&B2 zone candidates:  {mb2_total}")
    print(f"  Output directory:      {OUTPUT_DIR}")
    print()
    if timeline.get("first_detection_date"):
        print(f"  OIL LEAK ONSET:        {timeline['first_detection_date']}")
        sc = timeline.get("source_coords", {})
        print(f"  SOURCE COORDINATES:    lat={sc.get('lat','?')}  lon={sc.get('lon','?')}")
    print()


if __name__ == '__main__':
    main()
