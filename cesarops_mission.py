#!/usr/bin/env python3
"""
CESAROPS Mission Runner — Unified Multi-Pass Scan Engine
=========================================================

Single entry point for all sensor passes.  Takes a JSON mission config that
the agent (or Tauri frontend) builds.  All passes can be toggled on/off and
every band threshold can be tuned individually ("intensity mixing").

Usage (CLI / agent)
-------------------
  python cesarops_mission.py --mission-json '<json>'
  python cesarops_mission.py --mission-file mission.json
  python cesarops_mission.py --list-presets

Mission JSON schema
-------------------
{
  "name":       "Triple Lock Lake Erie",        # human label
  "bbox":       [41.30, -83.50, 42.50, -78.80], # [lat_min, lon_min, lat_max, lon_max]
  "output_tag": "erie_oct2015",                 # folder name under outputs/
  "data_dirs":  ["downloads/erie"],             # list of dirs to scan for TIFFs
  "passes": {
    "standard":           { "enabled": true,  "threshold": 1.5 },
    "hydrocarbon":        { "enabled": true,  "swir_thresh": -1.8, "red_thresh": 1.5 },
    "thermal":            { "enabled": true,  "threshold": 2.0 },
    "stumpf":             { "enabled": false, "threshold": 2.0 },
    "nauticuvs":          { "enabled": true,  "energy_threshold": 3.5, "top_n": 50 },
    "swir_silt_erasure":  { "enabled": true,  "threshold": 2.5, "top_n": 30 },
    "mussel_clearspot":   { "enabled": true,  "threshold": 2.0, "top_n": 30 }
  },
  "sub_zones": [
    {
      "name":    "M&B2 Central Basin",
      "bbox":    [41.80, -82.50, 42.50, -80.00],
      "passes":  ["swir_silt_erasure", "mussel_clearspot", "nauticuvs"]
    }
  ]
}

Pass descriptions
-----------------
  standard          Thermal / optical / SAR z-score anomaly scan.
                    threshold: sigma cutoff. thermal auto-uses cold_sink_mode.
  hydrocarbon       B11 SWIR dark + B04 Red bright dual-confirm (oil/fuel slick).
                    swir_thresh: z-score cutoff for B11 (negative). red_thresh: B04.
  thermal           Independent thermal cold-sink pass at higher sensitivity.
                    threshold: cold-sink sigma (positive = hotter, negative = colder).
  stumpf            Log-ratio B02/B03 bathymetric shallow anomaly.
  nauticuvs         Multi-scale LoG blob on B02 + B10 (NauticUVs curvelet energy proxy).
                    energy_threshold: sigma cutoff for LoG peaks.
  swir_silt_erasure B11/B12 ratio — reveals sub-silt ferrous metal (e.g. buried hull).
                    threshold: elevated-ratio sigma cutoff.
  mussel_clearspot  Positive B02 anomaly in turbid Erie background — mussel colony filter.
                    threshold: elevated blue sigma cutoff.
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ── Built-in mission presets ─────────────────────────────────────────────────

MISSION_PRESETS = {
    "triple_lock_erie": {
        "name": "Triple Lock — Lake Erie (full)",
        "bbox": [41.30, -83.50, 42.50, -78.80],
        "output_tag": "triple_lock_erie",
        "data_dirs": ["downloads/erie", "downloads/hls"],
        "passes": {
            "standard":          {"enabled": True,  "threshold": 1.5},
            "hydrocarbon":       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5},
            "thermal":           {"enabled": True,  "threshold": 2.0},
            "stumpf":            {"enabled": False, "threshold": 2.0},
            "nauticuvs":         {"enabled": True,  "energy_threshold": 3.5, "top_n": 50},
            "swir_silt_erasure": {"enabled": False, "threshold": 2.5, "top_n": 30},
            "mussel_clearspot":  {"enabled": False, "threshold": 2.0, "top_n": 30},
        },
        "sub_zones": [],
    },
    "mb2_wreck_hunt": {
        "name": "Marquette and Bessemer No. 2 — Central Basin Hunt",
        "bbox": [41.80, -82.50, 42.50, -80.00],
        "output_tag": "mb2_hunt",
        "data_dirs": ["downloads/erie", "downloads/hls"],
        "passes": {
            "standard":          {"enabled": True,  "threshold": 1.5},
            "hydrocarbon":       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5},
            "thermal":           {"enabled": True,  "threshold": 1.8},
            "stumpf":            {"enabled": True,  "threshold": 2.0},
            "nauticuvs":         {"enabled": True,  "energy_threshold": 3.0, "top_n": 50},
            "swir_silt_erasure": {"enabled": True,  "threshold": 2.5, "top_n": 30},
            "mussel_clearspot":  {"enabled": True,  "threshold": 2.0, "top_n": 30},
        },
        "sub_zones": [],
    },
    "erie_hc_timeline": {
        "name": "Lake Erie Hydrocarbon Timeline (HC + SAR only)",
        "bbox": [41.30, -83.50, 42.50, -78.80],
        "output_tag": "erie_hc_timeline",
        "data_dirs": ["downloads/erie", "downloads/hls"],
        "passes": {
            "standard":          {"enabled": True,  "threshold": 1.5},
            "hydrocarbon":       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5},
            "thermal":           {"enabled": False, "threshold": 2.0},
            "stumpf":            {"enabled": False, "threshold": 2.0},
            "nauticuvs":         {"enabled": False, "energy_threshold": 3.5, "top_n": 50},
            "swir_silt_erasure": {"enabled": False, "threshold": 2.5, "top_n": 30},
            "mussel_clearspot":  {"enabled": False, "threshold": 2.0, "top_n": 30},
        },
        "sub_zones": [],
    },
    "straits_triple_lock": {
        "name": "Triple Lock — Straits of Mackinac",
        "bbox": [45.70, -84.90, 46.05, -84.10],
        "output_tag": "straits_triple_lock",
        "data_dirs": ["downloads/straits", "downloads/michigan", "downloads/hls"],
        "passes": {
            "standard":          {"enabled": True,  "threshold": 1.5},
            "hydrocarbon":       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5},
            "thermal":           {"enabled": True,  "threshold": 2.0},
            "stumpf":            {"enabled": True,  "threshold": 2.0},
            "nauticuvs":         {"enabled": True,  "energy_threshold": 3.5, "top_n": 50},
            "swir_silt_erasure": {"enabled": False, "threshold": 2.5, "top_n": 30},
            "mussel_clearspot":  {"enabled": False, "threshold": 2.0, "top_n": 30},
        },
        "sub_zones": [],
    },
    "andaste_hunt": {
        "name": "Andaste — Lake Michigan South",
        "bbox": [42.30, -88.50, 43.20, -87.40],
        "output_tag": "andaste_hunt",
        "data_dirs": ["downloads/michigan", "downloads/hls"],
        "passes": {
            "standard":          {"enabled": True,  "threshold": 1.2},
            "hydrocarbon":       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5},
            "thermal":           {"enabled": True,  "threshold": 1.8},
            "stumpf":            {"enabled": True,  "threshold": 2.0},
            "nauticuvs":         {"enabled": True,  "energy_threshold": 3.0, "top_n": 50},
            "swir_silt_erasure": {"enabled": False, "threshold": 2.5, "top_n": 30},
            "mussel_clearspot":  {"enabled": False, "threshold": 2.0, "top_n": 30},
        },
        "sub_zones": [],
    },
}


# ── Natural language → preset resolver ───────────────────────────────────────

_KEYWORD_MAP = {
    "triple lock erie":       "triple_lock_erie",
    "triple lock lake erie":  "triple_lock_erie",
    "erie triple lock":       "triple_lock_erie",
    "lake erie full":         "triple_lock_erie",
    "marquette bessemer":     "mb2_wreck_hunt",
    "mb2":                    "mb2_wreck_hunt",
    "m&b2":                   "mb2_wreck_hunt",
    "central basin":          "mb2_wreck_hunt",
    "erie hydrocarbon":       "erie_hc_timeline",
    "erie hc":                "erie_hc_timeline",
    "erie leak":              "erie_hc_timeline",
    "oil leak":               "erie_hc_timeline",
    "straits triple lock":    "straits_triple_lock",
    "straits":                "straits_triple_lock",
    "mackinac":               "straits_triple_lock",
    "andaste":                "andaste_hunt",
    "lake michigan south":    "andaste_hunt",
}

def resolve_preset_by_name(name: str) -> str | None:
    """Return a preset key if the name fuzzy-matches one of the known presets."""
    lower = name.lower()
    for kw, preset in _KEYWORD_MAP.items():
        if kw in lower:
            return preset
    return None


# ── Import CESAROPS processing engine ────────────────────────────────────────

def _load_engine():
    """Import all processing functions from lake_michigan_scan.py."""
    try:
        from lake_michigan_scan import (
            process_hydrocarbon_bands,
            process_tiff_with_coords,
            compute_nauticuvs_pass,
            compute_stumpf_pass,
            KNOWN_WRECKS,
            _flag_known_wreck,
            _is_linear_wake,
        )
        return {
            "process_hydrocarbon_bands":  process_hydrocarbon_bands,
            "process_tiff_with_coords":   process_tiff_with_coords,
            "compute_nauticuvs_pass":     compute_nauticuvs_pass,
            "compute_stumpf_pass":        compute_stumpf_pass,
            "KNOWN_WRECKS":               KNOWN_WRECKS,
            "_flag_known_wreck":          _flag_known_wreck,
            "_is_linear_wake":            _is_linear_wake,
        }
    except ImportError as e:
        print(f"[MISSION] FATAL: lake_michigan_scan.py not found: {e}")
        sys.exit(1)


# ── SWIR silt-erasure (B11/B12 ratio) ────────────────────────────────────────

def _detect_swir_silt_erasure(b11_path: Path, b12_path: Path, scan_bbox, threshold: float, top_n: int, _flag_known_wreck) -> list:
    import rasterio
    from rasterio.warp import transform as warp_transform

    if not b11_path.exists() or not b12_path.exists():
        return []
    print(f"  [SWE] B11/B12 silt erasure: {b11_path.name}")

    with rasterio.open(b11_path) as s11:
        b11 = s11.read(1).astype(np.float32); crs = s11.crs
    with rasterio.open(b12_path) as s12:
        b12 = s12.read(1).astype(np.float32)
    if b12.shape != b11.shape:
        import skimage.transform as skt
        b12 = skt.resize(b12, b11.shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)

    valid = (b11 > 0) & (b12 > 0) & np.isfinite(b11) & np.isfinite(b12)
    ratio = np.full_like(b11, np.nan)
    ratio[valid] = b11[valid] / (b12[valid] + 1e-6)
    vals = ratio[valid & np.isfinite(ratio)]
    if vals.size < 100:
        return []
    mu = float(np.nanmean(vals)); sd = float(np.nanstd(vals))
    if sd < 1e-6:
        return []
    z_ratio = np.full_like(ratio, np.nan)
    z_ratio[valid] = (ratio[valid] - mu) / sd
    anom = (z_ratio > threshold) & valid & np.isfinite(z_ratio)

    rows, cols = np.where(anom)
    rows, cols = _apply_bbox_filter(rows, cols, b11, b11_path, scan_bbox)
    if len(rows) == 0:
        return []
    zv = z_ratio[rows, cols]
    idx = np.argsort(-zv)[:top_n]
    out = []
    with rasterio.open(b11_path) as src:
        for i in idx:
            r, c = int(rows[i]), int(cols[i])
            lx, ly = src.xy(r, c)
            lon_pt, lat_pt = warp_transform(src.crs, 'EPSG:4326', [lx], [ly])
            lat_v, lon_v = lat_pt[0], lon_pt[0]
            known = _flag_known_wreck(lat_v, lon_v)
            out.append({"lat": lat_v, "lon": lon_v, "zscore": float(z_ratio[r, c]),
                        "type": "swir_silt_erasure", "source": b11_path.name,
                        "b11_b12_ratio": round(float(ratio[r, c]), 4),
                        "known_wreck_hit": known["id"] if known else None,
                        "known_wreck_name": known["name"] if known else None,
                        "pixel": {"row": r, "col": c}})
    print(f"    [SWE] {len(out)} detections")
    return out


# ── Mussel clear-spot (positive B02 in turbid background) ────────────────────

def _detect_mussel_clearspot(b02_path: Path, scan_bbox, threshold: float, top_n: int, _flag_known_wreck) -> list:
    import rasterio
    from rasterio.warp import transform as warp_transform

    if not b02_path.exists():
        return []
    print(f"  [MCS] Mussel clearspot B02: {b02_path.name}")
    with rasterio.open(b02_path) as src:
        data = src.read(1).astype(np.float32)
    nodata = (data <= 0) | ~np.isfinite(data)
    data[nodata] = np.nan
    valid = ~nodata
    vals = data[valid]
    if vals.size < 200:
        return []
    mu = float(np.nanmean(vals)); sd = float(np.nanstd(vals))
    if sd < 1e-6:
        return []
    z = np.full_like(data, np.nan)
    z[valid] = (data[valid] - mu) / sd
    anom = (z > threshold) & valid
    rows, cols = np.where(anom)
    rows, cols = _apply_bbox_filter(rows, cols, data, b02_path, scan_bbox)
    if len(rows) == 0:
        return []
    zv = z[rows, cols]
    idx = np.argsort(-zv)[:top_n]
    out = []
    with rasterio.open(b02_path) as src:
        for i in idx:
            r, c = int(rows[i]), int(cols[i])
            lx, ly = src.xy(r, c)
            lon_pt, lat_pt = warp_transform(src.crs, 'EPSG:4326', [lx], [ly])
            lat_v, lon_v = lat_pt[0], lon_pt[0]
            known = _flag_known_wreck(lat_v, lon_v)
            out.append({"lat": lat_v, "lon": lon_v, "zscore": float(z[r, c]),
                        "type": "mussel_clearspot", "source": b02_path.name,
                        "known_wreck_hit": known["id"] if known else None,
                        "known_wreck_name": known["name"] if known else None,
                        "pixel": {"row": r, "col": c}})
    print(f"    [MCS] {len(out)} detections")
    return out


# ── Spatial bbox pixel filter (shared helper) ────────────────────────────────

def _apply_bbox_filter(rows, cols, data, tiff_path: Path, scan_bbox):
    import rasterio
    from rasterio.warp import transform as warp_transform
    if scan_bbox is None or len(rows) == 0:
        return rows, cols
    with rasterio.open(tiff_path) as src:
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
    if not bb_rows:
        return rows, cols
    r_min = max(0, min(bb_rows)); r_max = min(data.shape[0]-1, max(bb_rows))
    c_min = max(0, min(bb_cols)); c_max = min(data.shape[1]-1, max(bb_cols))
    if r_min > r_max: r_min, r_max = r_max, r_min
    if c_min > c_max: c_min, c_max = c_max, c_min
    mask = ((rows >= r_min) & (rows <= r_max) & (cols >= c_min) & (cols <= c_max))
    return rows[mask], cols[mask]


# ── Band name helpers ─────────────────────────────────────────────────────────

_SKIP_UPPER = {'FMASK', '.B11.', '.SWIR16.', '.SWIR22.', '.SCL.', '.QA_PIXEL.', '.NIR08.', '.NIR.'}

def _is_b11(p: Path) -> bool:
    u = p.name.upper()
    return ('.B11.' in u or '.SWIR16.' in u) and 'FMASK' not in u and '.SCL.' not in u

def _is_b12(p: Path) -> bool:
    u = p.name.upper()
    return ('.B12.' in u or '.SWIR22.' in u) and 'FMASK' not in u

def _is_b02(p: Path) -> bool:
    u = p.name.upper()
    return ('.B02.' in u or '.BLUE.' in u) and 'FMASK' not in u and '.SCL.' not in u

def _companion_b04(b11: Path) -> Path:
    n = b11.name.lower()
    if '.swir16.tif' in n:
        return Path(str(b11).replace('.swir16.tif', '.red.tif').replace('.SWIR16.tif', '.red.tif'))
    return Path(str(b11).replace('.B11.tif', '.B04.tif'))

def _companion_b03(b02: Path) -> Path:
    n = b02.name.lower()
    if '.blue.tif' in n:
        return Path(str(b02).replace('.blue.tif', '.green.tif'))
    return Path(str(b02).replace('.B02.tif', '.B03.tif'))

def _companion_b12(b11: Path) -> Path:
    return Path(str(b11).replace('.B11.', '.B12.').replace('.swir16.', '.swir22.'))


# ── KMZ output ────────────────────────────────────────────────────────────────

def _build_kmz(detections: list, output_path: Path, mission_name: str, KNOWN_WRECKS: list):
    import simplekml
    kml = simplekml.Kml()

    ref_folder = kml.newfolder(name="Known Wreck Reference Positions")
    for w in KNOWN_WRECKS:
        if not (isinstance(w.get("lat"), float) and isinstance(w.get("lon"), float)):
            continue
        pnt = ref_folder.newpoint(name=f"{w['name']} ({w.get('year_lost','?')})",
                                  coords=[(w["lon"], w["lat"])])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"
        pnt.description = (f"<b>{w['name']}</b><br/>"
                           f"Type: {w.get('type','?')}<br/>"
                           f"Lat: {w['lat']:.5f}  Lon: {w['lon']:.5f}")

    folders = {
        "known_hit":      kml.newfolder(name="Known Wreck HITS"),
        "hydrocarbon":    kml.newfolder(name="Hydrocarbon — Oil/Fuel"),
        "thermal":        kml.newfolder(name="Thermal — Cold Sink"),
        "swir_silt":      kml.newfolder(name="SWIR Silt Erasure — Sub-silt Metal"),
        "mussel":         kml.newfolder(name="Mussel Clear-Spot"),
        "nauticuvs":      kml.newfolder(name="NauticUVs LoG Blob"),
        "stumpf":         kml.newfolder(name="Stumpf Shallow Anomaly"),
        "general":        kml.newfolder(name="General Optical / SAR Anomaly"),
    }

    _ICONS = {
        "known_hit":   ("http://maps.google.com/mapfiles/kml/paddle/wht-stars.png",  simplekml.Color.cyan),
        "hydrocarbon": ("http://maps.google.com/mapfiles/kml/paddle/pink-blank.png", simplekml.Color.fuchsia),
        "thermal":     ("http://maps.google.com/mapfiles/kml/paddle/ltblu-blank.png", simplekml.Color.aqua),
        "swir_silt":   ("http://maps.google.com/mapfiles/kml/paddle/orange-blank.png", simplekml.Color.orange),
        "mussel":      ("http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png",  simplekml.Color.yellow),
        "nauticuvs":   ("http://maps.google.com/mapfiles/kml/paddle/purple-blank.png", simplekml.Color.purple),
        "stumpf":      ("http://maps.google.com/mapfiles/kml/paddle/grn-blank.png",  simplekml.Color.green),
        "general":     ("http://maps.google.com/mapfiles/kml/paddle/red-circle.png", simplekml.Color.red),
    }

    for det in detections:
        lat_v = det["lat"]; lon_v = det["lon"]
        z_v   = det.get("zscore", 0.0)
        dtype = det.get("type", "optical")
        known = det.get("known_wreck_hit")

        if known:
            fkey = "known_hit"
        elif dtype == "hydrocarbon":
            fkey = "hydrocarbon"
        elif dtype in ("thermal", "optical_thermal"):
            fkey = "thermal"
        elif dtype == "swir_silt_erasure":
            fkey = "swir_silt"
        elif dtype == "mussel_clearspot":
            fkey = "mussel"
        elif dtype == "nauticuvs_candidate":
            fkey = "nauticuvs"
        elif dtype == "stumpf_shallow":
            fkey = "stumpf"
        else:
            fkey = "general"

        icon_url, color = _ICONS[fkey]
        label = f"{dtype.upper()} Z={z_v:.2f}"
        if known:
            label = f"HIT {det.get('known_wreck_name','?')} {label}"

        desc = (f"<b>{dtype.upper()}</b><br/>"
                f"Z: {z_v:.3f}<br/>"
                f"Source: {det.get('source','?')}<br/>"
                f"Date: {det.get('scan_date','?')}<br/>"
                + (f"<b>KNOWN WRECK: {det.get('known_wreck_name')}</b><br/>" if known else "")
                + f"Lat: {lat_v:.6f}<br/>Lon: {lon_v:.6f}")

        pnt = folders[fkey].newpoint(name=label, coords=[(lon_v, lat_v)])
        pnt.style.iconstyle.icon.href = icon_url
        pnt.style.iconstyle.color = color
        pnt.description = desc

    kml.save(str(output_path))
    print(f"  [KMZ] {output_path.name}  ({len(detections)} points)")


# ── Core mission runner ───────────────────────────────────────────────────────

def run_mission(mission: dict) -> dict:
    """
    Execute a mission config dict.  Returns a result summary dict.
    Progress lines are printed to stdout (streamed to Tauri frontend via task_output events).
    """
    engine = _load_engine()
    PHC  = engine["process_hydrocarbon_bands"]
    PTIF = engine["process_tiff_with_coords"]
    PNUV = engine["compute_nauticuvs_pass"]
    PST  = engine["compute_stumpf_pass"]
    KNOWN_WRECKS    = engine["KNOWN_WRECKS"]
    flag_known_wreck = engine["_flag_known_wreck"]

    name       = mission.get("name", "Unnamed Mission")
    bbox       = mission.get("bbox", [41.30, -83.50, 42.50, -78.80])
    output_tag = re.sub(r'[^\w\-]', '_', mission.get("output_tag", "mission"))
    data_dirs  = mission.get("data_dirs", ["downloads/erie", "downloads/hls"])
    passes     = mission.get("passes", {})
    sub_zones  = mission.get("sub_zones", [])

    p_stdrd  = passes.get("standard",          {"enabled": True,  "threshold": 1.5})
    p_hc     = passes.get("hydrocarbon",       {"enabled": True,  "swir_thresh": -1.8, "red_thresh": 1.5})
    p_therm  = passes.get("thermal",           {"enabled": True,  "threshold": 2.0})
    p_stmpf  = passes.get("stumpf",            {"enabled": False, "threshold": 2.0})
    p_nuv    = passes.get("nauticuvs",         {"enabled": True,  "energy_threshold": 3.5, "top_n": 50})
    p_swe    = passes.get("swir_silt_erasure", {"enabled": False, "threshold": 2.5, "top_n": 30})
    p_mcs    = passes.get("mussel_clearspot",  {"enabled": False, "threshold": 2.0, "top_n": 30})

    repo_root = Path(__file__).parent
    out_dir   = repo_root / 'outputs' / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print mission header (streamed to Tauri)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*72}")
    print(f"  CESAROPS MISSION: {name}")
    print(f"  Started:     {ts}")
    print(f"  Bbox:        {bbox}")
    print(f"  Output:      {out_dir}")
    enabled_passes = [k for k, v in passes.items() if v.get("enabled")]
    print(f"  Passes ON:   {', '.join(enabled_passes) or 'none'}")
    print(f"{'='*72}\n")

    # ── Discover TIFFs ────────────────────────────────────────────────────────
    tiffs = []
    for dd in data_dirs:
        dp = repo_root / dd if not Path(dd).is_absolute() else Path(dd)
        if dp.exists():
            tiffs.extend(dp.rglob("*.tif"))
    tiffs = sorted(set(tiffs))
    print(f"[TIFF] Discovered {len(tiffs)} TIFFs across {len(data_dirs)} data dir(s)")
    if not tiffs:
        print("[TIFF] No TIFFs found — run the downloader first.")
        return {"status": "no_data", "name": name, "detections": 0}

    all_detections: list = []

    def run_passes_on_tiffs(tiff_list, active_bbox, zone_label="") -> list:
        """Execute enabled passes on the given tiff list, constrained to active_bbox."""
        zone_dets = []

        # ── Pass 1: Standard anomaly ──────────────────────────────────────────
        if p_stdrd.get("enabled"):
            thresh = float(p_stdrd.get("threshold", 1.5))
            std_tiffs = [t for t in tiff_list if
                         not any(tag in t.name.upper() for tag in _SKIP_UPPER)]
            print(f"\n  {zone_label}PASS 1 — Standard anomaly [{thresh}σ] ({len(std_tiffs)} bands)")
            for tif in std_tiffs:
                tname = tif.name.upper()
                is_thermal = 'B10' in tname or 'THERMAL' in tname or 'LWIR' in tname
                is_blue    = '.B02.' in tname or '.BLUE.' in tname
                if is_thermal:
                    bt, cs = float(p_therm.get("threshold", 2.0)), True
                elif is_blue:
                    bt, cs = 1.2, False
                else:
                    bt, cs = thresh, False
                try:
                    dets = PTIF(tif, threshold=bt, scan_bbox=active_bbox, top_n=200, cold_sink_mode=cs)
                    zone_dets.extend(dets)
                except Exception as e:
                    print(f"    ERR {tif.name}: {e}")

        # ── Pass 2: Hydrocarbon ───────────────────────────────────────────────
        if p_hc.get("enabled"):
            swir_t = float(p_hc.get("swir_thresh", -1.8))
            red_t  = float(p_hc.get("red_thresh",   1.5))
            hc_tiffs = [t for t in tiff_list if _is_b11(t)]
            print(f"\n  {zone_label}PASS 2 — Hydrocarbon [{swir_t}σ / {red_t}σ] ({len(hc_tiffs)} B11 SWIR)")
            for b11 in hc_tiffs:
                b04 = _companion_b04(b11)
                try:
                    dets = PHC(b11, b04, swir_thresh=swir_t, red_thresh=red_t)
                    zone_dets.extend(dets)
                except Exception as e:
                    print(f"    ERR {b11.name}: {e}")

        # ── Pass 3: Stumpf bathymetric ────────────────────────────────────────
        if p_stmpf.get("enabled"):
            blue_tiffs = [t for t in tiff_list if _is_b02(t)]
            print(f"\n  {zone_label}PASS 3 — Stumpf {len(blue_tiffs)} blue band(s)")
            for blue in blue_tiffs:
                green = _companion_b03(blue)
                try:
                    dets = PST(blue, green, scan_bbox=active_bbox)
                    zone_dets.extend(dets)
                except Exception as e:
                    print(f"    ERR {blue.name}: {e}")

        # ── Pass 4: NauticUVs LoG ─────────────────────────────────────────────
        if p_nuv.get("enabled"):
            e_thresh = float(p_nuv.get("energy_threshold", 3.5))
            top_n    = int(p_nuv.get("top_n", 50))
            nuv_bands = (
                [t for t in tiff_list if _is_b02(t)] +
                [t for t in tiff_list if 'B10' in t.name.upper() or 'THERMAL' in t.name.upper()]
            )
            print(f"\n  {zone_label}PASS 4 — NauticUVs LoG [{e_thresh}σ] ({len(nuv_bands)} bands)")
            for nuv_tif in nuv_bands:
                try:
                    dets = PNUV(nuv_tif, scan_bbox=active_bbox,
                                energy_threshold=e_thresh, top_n=top_n)
                    zone_dets.extend(dets)
                except Exception as e:
                    print(f"    ERR {nuv_tif.name}: {e}")

        # ── Pass 5: SWIR silt erasure ─────────────────────────────────────────
        if p_swe.get("enabled"):
            swe_thresh = float(p_swe.get("threshold", 2.5))
            swe_top_n  = int(p_swe.get("top_n", 30))
            b12_tiffs  = [t for t in tiff_list if _is_b12(t)]
            print(f"\n  {zone_label}PASS 5 — SWIR silt erasure [{swe_thresh}σ] ({len(b12_tiffs)} B12)")
            for b12 in b12_tiffs:
                b11 = _companion_b12(b12).with_suffix('.tif')
                b11 = Path(str(b12).replace('.B12.', '.B11.').replace('.swir22.', '.swir16.'))
                dets = _detect_swir_silt_erasure(b11, b12, active_bbox, swe_thresh, swe_top_n, flag_known_wreck)
                zone_dets.extend(dets)

        # ── Pass 6: Mussel clear-spot ─────────────────────────────────────────
        if p_mcs.get("enabled"):
            mcs_thresh = float(p_mcs.get("threshold", 2.0))
            mcs_top_n  = int(p_mcs.get("top_n", 30))
            blue_tiffs = [t for t in tiff_list if _is_b02(t)]
            print(f"\n  {zone_label}PASS 6 — Mussel clearspot [{mcs_thresh}σ] ({len(blue_tiffs)} B02)")
            for blue in blue_tiffs:
                dets = _detect_mussel_clearspot(blue, active_bbox, mcs_thresh, mcs_top_n, flag_known_wreck)
                zone_dets.extend(dets)

        return zone_dets

    # ── Full-bbox scan ────────────────────────────────────────────────────────
    print(f"\n[SCAN] Full bbox pass: {bbox}")
    all_detections.extend(run_passes_on_tiffs(tiffs, bbox, ""))

    # ── Sub-zone scans ────────────────────────────────────────────────────────
    for sz in sub_zones:
        sz_name  = sz.get("name", "Sub-zone")
        sz_bbox  = sz.get("bbox", bbox)
        sz_pass_filter = set(sz.get("passes", []))
        print(f"\n[SCAN] Sub-zone '{sz_name}': {sz_bbox}")

        # Temporarily override pass enable flags for this zone
        orig_flags = {k: v.get("enabled", False) for k, v in passes.items()}
        if sz_pass_filter:
            for k in passes:
                passes[k]["enabled"] = k in sz_pass_filter
        all_detections.extend(run_passes_on_tiffs(tiffs, sz_bbox, f"[{sz_name}] "))
        # Restore
        for k in passes:
            passes[k]["enabled"] = orig_flags[k]

    # ── Output ────────────────────────────────────────────────────────────────
    total = len(all_detections)
    hc    = sum(1 for d in all_detections if d.get("type") == "hydrocarbon")
    known = sum(1 for d in all_detections if d.get("known_wreck_hit"))
    print(f"\n[RESULT] Total: {total}  HC: {hc}  Known-wreck hits: {known}")

    result_json = out_dir / 'mission_results.json'
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump({
            "mission":          name,
            "bbox":             bbox,
            "passes_enabled":   enabled_passes,
            "total_detections": total,
            "hc_count":         hc,
            "known_hits":       known,
            "detections":       all_detections,
            "timestamp":        ts,
        }, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {result_json}")

    try:
        import simplekml as _skml  # noqa
        kmz_path = out_dir / 'mission_results.kmz'
        _build_kmz(all_detections, kmz_path, name, KNOWN_WRECKS)
    except ImportError:
        print("[KMZ]  simplekml not available — skipping KMZ")

    print(f"\n[DONE] Mission '{name}' complete.")
    return {
        "status":           "complete",
        "name":             name,
        "detections":       total,
        "hc_count":         hc,
        "known_hits":       known,
        "output_dir":       str(out_dir),
        "results_json":     str(result_json),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CESAROPS Mission Runner — unified scan engine")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--mission-json",  type=str, help="Mission config as a JSON string")
    grp.add_argument("--mission-file",  type=str, help="Path to a mission config JSON file")
    grp.add_argument("--preset",        type=str, help="Built-in preset name")
    grp.add_argument("--list-presets",  action="store_true", help="List all built-in presets")
    grp.add_argument("--resolve",       type=str, help="Resolve a natural language mission name to a preset key")
    args = parser.parse_args()

    if args.list_presets:
        print("\nBuilt-in mission presets:\n")
        for key, m in MISSION_PRESETS.items():
            en = [k for k, v in m["passes"].items() if v.get("enabled")]
            print(f"  {key:<28} {m['name']}")
            print(f"  {'':28} Bbox:   {m['bbox']}")
            print(f"  {'':28} Passes: {', '.join(en)}\n")
        return

    if args.resolve:
        key = resolve_preset_by_name(args.resolve)
        if key:
            print(json.dumps({"preset_key": key, "name": MISSION_PRESETS[key]["name"]}))
        else:
            print(json.dumps({"preset_key": None, "name": None}))
        return

    if args.mission_json:
        try:
            mission = json.loads(args.mission_json)
        except json.JSONDecodeError as e:
            print(f"[MISSION] Invalid JSON: {e}")
            sys.exit(1)
    elif args.mission_file:
        with open(args.mission_file, encoding='utf-8') as f:
            mission = json.load(f)
    elif args.preset:
        if args.preset not in MISSION_PRESETS:
            # Try natural language resolution
            resolved = resolve_preset_by_name(args.preset)
            if resolved:
                mission = MISSION_PRESETS[resolved].copy()
            else:
                print(f"[MISSION] Unknown preset '{args.preset}'. Use --list-presets.")
                sys.exit(1)
        else:
            mission = MISSION_PRESETS[args.preset].copy()
    else:
        parser.print_help()
        return

    result = run_mission(mission)

    # Print machine-readable summary line (parsed by Tauri for display)
    print(f"\n[MISSION_RESULT] {json.dumps(result)}")


if __name__ == '__main__':
    main()
