#!/usr/bin/env python3
"""
WreckHunter 2000 — KML / KMZ Export
=====================================
Reads the full.json pipeline output and generates a KML (or KMZ) file
suitable for Google Earth, with:

  • Unknown targets — colour-coded by wreck score
  • Known-matched detections — including well matches
  • Nearest known wrecks / wells for EVERY detection (plotted + in balloon)
  • Connection lines from each detection to its nearest match

Usage:
  python wh2k_export_kml.py                           # defaults
  python wh2k_export_kml.py --kmz                     # compressed KMZ
  python wh2k_export_kml.py --input some_other.full.json --output out.kml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = REPO_ROOT / "wreck_hunting_ml" / "outputs" / "v2_crm_targets.full.json"
DEFAULT_DB    = REPO_ROOT / "db" / "wrecks.db"
DEFAULT_WELLS = REPO_ROOT / "eriewelldata" / "wells.csv"

NEAREST_N     = 3       # how many nearest knowns to show per detection
NEARBY_KM     = 25.0    # radius (km) to pull reference wrecks/wells onto map


# ── Haversine ──────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Data loaders ───────────────────────────────────────────────────────────

def load_wrecks(db_path: Path) -> list[dict]:
    if not db_path.exists():
        logger.warning("Wreck DB not found: %s", db_path)
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, latitude, longitude, found_status, feature_type "
        "FROM features "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
        "AND found_status = 'found'"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_wells(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        logger.warning("Wells CSV not found: %s", csv_path)
        return []
    wells = []
    with open(csv_path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("SUR_LAT83") or 0)
                lon = float(row.get("SUR_LONG83") or 0)
                if lat and lon:
                    wells.append({
                        "name": row.get("WELL_NAME", ""),
                        "latitude": lat,
                        "longitude": lon,
                        "td_ft": row.get("TD", ""),
                        "status": row.get("CUR_STATUS", ""),
                    })
            except (ValueError, TypeError):
                continue
    return wells


def nearest_items(lat, lon, items, n=3, key_lat="latitude", key_lon="longitude"):
    """Return list of (dist_m, item) sorted by distance."""
    dists = []
    for item in items:
        d = _haversine_m(lat, lon, item[key_lat], item[key_lon])
        dists.append((d, item))
    dists.sort(key=lambda x: x[0])
    return dists[:n]


# ── KML style definitions ─────────────────────────────────────────────────

# Detection styles keyed by (match_status, score_bracket)
STYLES = {
    # Unknown targets — colour by score
    "unknown_high":     ("ff0000ff", "1.4", "http://maps.google.com/mapfiles/kml/paddle/red-stars.png"),
    "unknown_mid":      ("ff00aaff", "1.2", "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"),
    "unknown_low":      ("ff00ffff", "1.0", "http://maps.google.com/mapfiles/kml/paddle/wht-stars.png"),
    # Known matches
    "known_well":       ("ffff8800", "1.0", "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png"),
    "known_wreck":      ("ff00ff00", "1.0", "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
    # Reference items
    "ref_wreck":        ("ff888888", "0.7", "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
    "ref_well":         ("ffaaaaaa", "0.6", "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"),
    # Connection line
    "conn_line":        ("9900ffff", None, None),
}


def _style_xml(style_id: str, color: str, scale: str, icon_href: str) -> str:
    return (
        f'<Style id="{style_id}">\n'
        f'  <IconStyle>\n'
        f'    <color>{color}</color>\n'
        f'    <scale>{scale}</scale>\n'
        f'    <Icon><href>{icon_href}</href></Icon>\n'
        f'  </IconStyle>\n'
        f'  <LabelStyle><scale>0.8</scale></LabelStyle>\n'
        f'</Style>\n'
    )


def _line_style_xml(style_id: str, color: str, width: int = 2) -> str:
    return (
        f'<Style id="{style_id}">\n'
        f'  <LineStyle><color>{color}</color><width>{width}</width></LineStyle>\n'
        f'</Style>\n'
    )


def _pick_style(det: dict) -> str:
    """Pick KML style id for a detection."""
    match = det.get("known_match", "unknown")
    if match == "wellhead":
        return "known_well"
    if match == "wreck":
        return "known_wreck"
    score = det.get("wreck_score", 0)
    if score >= 8:
        return "unknown_high"
    if score >= 5:
        return "unknown_mid"
    return "unknown_low"


# ── KML builder ────────────────────────────────────────────────────────────

def build_kml(full_json: dict,
              wrecks: list[dict],
              wells: list[dict],
              nearest_n: int = NEAREST_N) -> str:
    """Build complete KML string."""

    all_dets = full_json.get("detections", [])
    knowns   = full_json.get("knowns_subtracted", [])
    unknowns = full_json.get("unknowns_scored", [])
    summary  = full_json.get("summary", {})

    # Merge for easy lookup
    unknowns_by_id = {d["detection_id"]: d for d in unknowns}
    knowns_by_id   = {d["detection_id"]: d for d in knowns}

    lines: list[str] = []
    _a = lines.append

    _a('<?xml version="1.0" encoding="UTF-8"?>')
    _a('<kml xmlns="http://www.opengis.net/kml/2.2">')
    _a('<Document>')
    _a('<name>WreckHunter 2000 — Lake Erie Central Basin</name>')
    _a(f'<description>Total detections: {summary.get("total_detections", len(all_dets))}, '
       f'Known matches: {summary.get("known_matches", len(knowns))}, '
       f'Unknowns: {summary.get("unknowns", len(unknowns))}, '
       f'Score 8+: {summary.get("score_8_plus", 0)}</description>')

    # ── Write styles ──
    for sid, (color, scale, icon) in STYLES.items():
        if icon is not None:
            _a(_style_xml(sid, color, scale, icon))
        else:
            _a(_line_style_xml(sid, color))

    # ── Folder 1: Unknown Targets ──
    _a('<Folder>')
    _a(f'<name>Unknown Targets ({len(unknowns)})</name>')
    _a('<open>1</open>')

    for det in sorted(unknowns, key=lambda d: -d.get("wreck_score", 0)):
        lat, lon = det["lat"], det["lon"]
        sid = _pick_style(det)
        score = det.get("wreck_score", 0)
        cls = det.get("class_name", "?")
        conf = det.get("confidence", 0)
        amp = det.get("peak_amplitude_nt", 0)
        reasons = det.get("score_reasons", [])

        # Find nearest known wrecks & wells
        nw = nearest_items(lat, lon, wrecks, n=nearest_n)
        nwl = nearest_items(lat, lon, wells, n=nearest_n)

        desc_parts = [
            f'<b>Score:</b> {score}/10<br/>',
            f'<b>Class:</b> {cls} ({conf:.1%})<br/>',
            f'<b>Amplitude:</b> {amp:.1f} nT<br/>',
            f'<b>Extent:</b> {det.get("spatial_extent_m", 0):.0f} m<br/>',
            f'<b>Aspect ratio:</b> {det.get("aspect_ratio", 0):.2f}<br/>',
            f'<b>Axis offset:</b> {det.get("axis_offset_from_geology_deg", 0):.1f}°<br/>',
        ]
        if reasons:
            desc_parts.append('<br/><b>Reasons:</b><br/>')
            for r in reasons:
                desc_parts.append(f'&nbsp;&nbsp;• {r}<br/>')

        if nw:
            desc_parts.append('<br/><b>Nearest Known Wrecks:</b><br/>')
            for dist_m, w in nw:
                desc_parts.append(
                    f'&nbsp;&nbsp;{dist_m / 1000:.1f} km — {w["name"]} '
                    f'({w.get("found_status", "?")})<br/>'
                )
        if nwl:
            desc_parts.append('<br/><b>Nearest Wells:</b><br/>')
            for dist_m, w in nwl:
                desc_parts.append(
                    f'&nbsp;&nbsp;{dist_m / 1000:.1f} km — {w["name"]} '
                    f'(TD={w.get("td_ft", "?")} ft)<br/>'
                )

        desc = "".join(desc_parts)
        det_id = det.get("detection_id", "?")

        _a('<Placemark>')
        _a(f'<name>T{det_id} [{cls}] score={score}</name>')
        _a(f'<description><![CDATA[{desc}]]></description>')
        _a(f'<styleUrl>#{sid}</styleUrl>')
        _a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        _a('</Placemark>')

        # Connection line to nearest known wreck
        if nw:
            d0, w0 = nw[0]
            _a('<Placemark>')
            _a(f'<name>→ {w0["name"]} ({d0/1000:.1f} km)</name>')
            _a(f'<styleUrl>#conn_line</styleUrl>')
            _a('<LineString><tessellate>1</tessellate>')
            _a(f'<coordinates>{lon:.6f},{lat:.6f},0 '
               f'{w0["longitude"]:.6f},{w0["latitude"]:.6f},0</coordinates>')
            _a('</LineString></Placemark>')

    _a('</Folder>')

    # ── Folder 2: Known Matches (wells & wrecks) ──
    _a('<Folder>')
    _a(f'<name>Known Matches ({len(knowns)})</name>')
    _a('<open>1</open>')

    for det in knowns:
        lat, lon = det["lat"], det["lon"]
        sid = _pick_style(det)
        match_type = det.get("known_match", "?")
        match_name = det.get("known_match_name", "?")
        match_dist = det.get("known_match_distance_m", -1)
        cls = det.get("class_name", "?")
        conf = det.get("confidence", 0)
        amp = det.get("peak_amplitude_nt", 0)

        # Also find nearest alternatives
        nw = nearest_items(lat, lon, wrecks, n=nearest_n)
        nwl = nearest_items(lat, lon, wells, n=nearest_n)

        desc_parts = [
            f'<b>Match:</b> {match_type.upper()} — {match_name}<br/>',
            f'<b>Match distance:</b> {match_dist:.0f} m<br/>',
            f'<b>Predicted class:</b> {cls} ({conf:.1%})<br/>',
            f'<b>Amplitude:</b> {amp:.1f} nT<br/>',
            f'<b>Extent:</b> {det.get("spatial_extent_m", 0):.0f} m<br/>',
            f'<b>Aspect ratio:</b> {det.get("aspect_ratio", 0):.2f}<br/>',
        ]

        if nw:
            desc_parts.append('<br/><b>Nearest Known Wrecks:</b><br/>')
            for dist_m, w in nw:
                desc_parts.append(
                    f'&nbsp;&nbsp;{dist_m / 1000:.1f} km — {w["name"]} '
                    f'({w.get("found_status", "?")})<br/>'
                )
        if nwl:
            desc_parts.append('<br/><b>Nearest Wells:</b><br/>')
            for dist_m, w in nwl:
                desc_parts.append(
                    f'&nbsp;&nbsp;{dist_m / 1000:.1f} km — {w["name"]} '
                    f'(TD={w.get("td_ft", "?")} ft)<br/>'
                )

        desc = "".join(desc_parts)
        det_id = det.get("detection_id", "?")

        _a('<Placemark>')
        _a(f'<name>M{det_id} [{match_type}] {match_name}</name>')
        _a(f'<description><![CDATA[{desc}]]></description>')
        _a(f'<styleUrl>#{sid}</styleUrl>')
        _a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        _a('</Placemark>')

        # Connection line to matched known site
        if match_type == "wellhead" and nwl:
            d0, w0 = nwl[0]
            _a('<Placemark>')
            _a(f'<name>→ well: {w0["name"]} ({d0/1000:.1f} km)</name>')
            _a(f'<styleUrl>#conn_line</styleUrl>')
            _a('<LineString><tessellate>1</tessellate>')
            _a(f'<coordinates>{lon:.6f},{lat:.6f},0 '
               f'{w0["longitude"]:.6f},{w0["latitude"]:.6f},0</coordinates>')
            _a('</LineString></Placemark>')
        elif match_type == "wreck" and nw:
            d0, w0 = nw[0]
            _a('<Placemark>')
            _a(f'<name>→ wreck: {w0["name"]} ({d0/1000:.1f} km)</name>')
            _a(f'<styleUrl>#conn_line</styleUrl>')
            _a('<LineString><tessellate>1</tessellate>')
            _a(f'<coordinates>{lon:.6f},{lat:.6f},0 '
               f'{w0["longitude"]:.6f},{w0["latitude"]:.6f},0</coordinates>')
            _a('</LineString></Placemark>')

    _a('</Folder>')

    # ── Folder 3: Reference — Nearby Known Wrecks ──
    # Collect all wrecks within NEARBY_KM of any detection
    ref_wreck_ids = set()
    ref_wrecks_to_plot = []
    for det in all_dets:
        for dist_m, w in nearest_items(det["lat"], det["lon"], wrecks, n=5):
            if dist_m <= NEARBY_KM * 1000:
                wid = f'{w["latitude"]:.5f}_{w["longitude"]:.5f}'
                if wid not in ref_wreck_ids:
                    ref_wreck_ids.add(wid)
                    ref_wrecks_to_plot.append(w)

    _a('<Folder>')
    _a(f'<name>Reference: Known Wrecks ({len(ref_wrecks_to_plot)})</name>')
    _a('<visibility>0</visibility>')

    for w in ref_wrecks_to_plot:
        wlat, wlon = w["latitude"], w["longitude"]
        _a('<Placemark>')
        _a(f'<name>{w["name"]}</name>')
        _a(f'<description>Found: {w.get("found_status", "?")}\n'
           f'Type: {w.get("feature_type", "?")}</description>')
        _a('<styleUrl>#ref_wreck</styleUrl>')
        _a(f'<Point><coordinates>{wlon:.6f},{wlat:.6f},0</coordinates></Point>')
        _a('</Placemark>')

    _a('</Folder>')

    # ── Folder 4: Reference — Nearby Wells ──
    ref_well_ids = set()
    ref_wells_to_plot = []
    for det in all_dets:
        for dist_m, w in nearest_items(det["lat"], det["lon"], wells, n=3):
            if dist_m <= NEARBY_KM * 1000:
                wid = f'{w["latitude"]:.5f}_{w["longitude"]:.5f}'
                if wid not in ref_well_ids:
                    ref_well_ids.add(wid)
                    ref_wells_to_plot.append(w)

    _a('<Folder>')
    _a(f'<name>Reference: Nearby Wells ({len(ref_wells_to_plot)})</name>')
    _a('<visibility>0</visibility>')

    for w in ref_wells_to_plot:
        wlat, wlon = w["latitude"], w["longitude"]
        _a('<Placemark>')
        _a(f'<name>{w["name"]}</name>')
        _a(f'<description>TD: {w.get("td_ft", "?")} ft\n'
           f'Status: {w.get("status", "?")}</description>')
        _a('<styleUrl>#ref_well</styleUrl>')
        _a(f'<Point><coordinates>{wlon:.6f},{wlat:.6f},0</coordinates></Point>')
        _a('</Placemark>')

    _a('</Folder>')

    _a('</Document>')
    _a('</kml>')

    return "\n".join(lines)


# ── Write KML / KMZ ───────────────────────────────────────────────────────

def write_kml(kml_text: str, output_path: Path, as_kmz: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if as_kmz:
        kmz_path = output_path.with_suffix(".kmz")
        with zipfile.ZipFile(str(kmz_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml_text)
        logger.info("Wrote KMZ: %s", kmz_path)
        return kmz_path
    else:
        output_path = output_path.with_suffix(".kml")
        output_path.write_text(kml_text, encoding="utf-8")
        logger.info("Wrote KML: %s", output_path)
        return output_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K → KML/KMZ exporter")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to full.json from inference pipeline")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Wrecks database")
    parser.add_argument("--wells-csv", type=str, default=str(DEFAULT_WELLS),
                        help="Wells CSV (Ontario/Erie)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <input_stem>.kml)")
    parser.add_argument("--kmz", action="store_true",
                        help="Output compressed KMZ instead of KML")
    parser.add_argument("--nearest", type=int, default=NEAREST_N,
                        help="How many nearest wrecks/wells per detection (default 3)")
    parser.add_argument("--radius-km", type=float, default=NEARBY_KM,
                        help="Radius (km) to pull reference wrecks/wells (default 25)")
    args = parser.parse_args()

    # Load pipeline output
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return
    full_json = json.loads(input_path.read_text(encoding="utf-8"))

    # Load reference data
    wrecks = load_wrecks(Path(args.db))
    wells  = load_wells(Path(args.wells_csv))
    logger.info("Loaded %d wrecks, %d wells", len(wrecks), len(wells))

    # Set globals for radius
    global NEARBY_KM
    NEARBY_KM = args.radius_km

    # Build KML
    kml_text = build_kml(full_json, wrecks, wells, nearest_n=args.nearest)

    # Output path
    if args.output:
        out = Path(args.output)
    else:
        out = input_path.with_suffix(".kml")

    written = write_kml(kml_text, out, as_kmz=args.kmz)
    print(f"Exported: {written}")
    print(f"  {len(full_json.get('unknowns_scored', []))} unknown targets")
    print(f"  {len(full_json.get('knowns_subtracted', []))} known matches (incl. wells)")
    print(f"  {len(wrecks)} reference wrecks loaded")
    print(f"  {len(wells)} reference wells loaded")


if __name__ == "__main__":
    main()
