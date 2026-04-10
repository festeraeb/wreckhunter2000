"""
_generate_combined_kml.py
=========================
Build a multi-layer KMZ covering BOTH Lake Erie and Lake Huron detections:
  - Orange  : Ontario wells over water (eriewelldata/wells.csv)
  - Red     : Erie high-score unknowns (score >= 7)
  - Yellow  : Erie mid-tier unknowns
  - Purple  : Erie wellhead-excluded
  - Green   : Erie known wreck matches
  - Cyan    : Huron high-score unknowns (score >= 7)
  - Blue    : Huron mid-tier unknowns
  - Magenta : Huron wellhead-excluded
  - Lime    : Huron known wreck matches

Usage:
  python scripts\\_generate_combined_kml.py
"""

import csv
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parents[1]
os.chdir(BASE)

WELLS_CSV       = BASE / "eriewelldata" / "wells.csv"
ERIE_JSON       = BASE / "wreck_hunting_ml" / "output" / "erie_full_lake_targets.full.json"
HURON_JSON      = BASE / "wreck_hunting_ml" / "output" / "huron_targets.full.json"
OUT_KMZ         = BASE / "wreck_hunting_ml" / "output" / "great_lakes_detections.kmz"

# KML colors: aaBBGGRR
COLOR_ORANGE    = "ff0088ff"    # wells
COLOR_RED       = "ff0000ff"    # Erie high
COLOR_YELLOW    = "ff00ffff"    # Erie mid
COLOR_PURPLE    = "ffff00a0"    # Erie wellhead-excl
COLOR_GREEN     = "ff00ff00"    # Erie wreck match
COLOR_CYAN      = "ffffff00"    # Huron high
COLOR_BLUE      = "ffff6600"    # Huron mid
COLOR_MAGENTA   = "ffff00ff"    # Huron wellhead-excl
COLOR_LIME      = "ff00ff80"    # Huron wreck match

LAKE_BOXES = {
    "Lake Erie":     {"lat_min": 41.35, "lat_max": 42.90, "lon_min": -83.60, "lon_max": -78.80},
    "Lake Huron":    {"lat_min": 43.00, "lat_max": 46.30, "lon_min": -84.80, "lon_max": -79.50},
    "Lake Superior": {"lat_min": 46.30, "lat_max": 49.10, "lon_min": -92.20, "lon_max": -84.30},
    "Lake Michigan": {"lat_min": 41.50, "lat_max": 46.10, "lon_min": -88.00, "lon_max": -84.70},
    "Georgian Bay":  {"lat_min": 44.50, "lat_max": 45.80, "lon_min": -81.50, "lon_max": -79.80},
    "Lake St Clair": {"lat_min": 42.20, "lat_max": 42.65, "lon_min": -83.10, "lon_max": -82.35},
}


def is_over_water(lat, lon):
    for lake, box in LAKE_BOXES.items():
        if box["lat_min"] <= lat <= box["lat_max"] and box["lon_min"] <= lon <= box["lon_max"]:
            return lake
    return None


def load_wells():
    wells = []
    skipped = 0
    with open(WELLS_CSV, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["SUR_LAT83"])
                lon = float(row["SUR_LONG83"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if lat == 0 or lon == 0:
                skipped += 1
                continue
            lake = is_over_water(lat, lon)
            if lake:
                wells.append({
                    "lat": lat, "lon": lon,
                    "name": (row.get("WELL_NAME") or row.get("FULL_NAME") or "").strip(),
                    "well_id": (row.get("WELL_ID") or "").strip(),
                    "status": (row.get("CUR_STATUS") or "").strip(),
                    "class": (row.get("CLASS") or "").strip(),
                    "lake": lake,
                })
    print(f"Wells: {len(wells)} over water, {skipped} skipped")
    for lake, cnt in sorted(Counter(w["lake"] for w in wells).items()):
        print(f"  {lake}: {cnt}")
    return wells


def load_detections(json_path, lake_name):
    with open(json_path) as f:
        data = json.load(f)
    dets = data["detections"]
    layers = {"high_unknown": [], "mid_unknown": [], "wellhead": [], "wreck": []}
    for d in dets:
        match_type = d.get("known_match", "unknown")
        score = d.get("wreck_score", 0)
        if match_type == "wreck":
            layers["wreck"].append(d)
        elif match_type == "wellhead":
            layers["wellhead"].append(d)
        elif score >= 7:
            layers["high_unknown"].append(d)
        else:
            layers["mid_unknown"].append(d)
    print(f"\n{lake_name} detections ({len(dets)} total):")
    for k, v in layers.items():
        print(f"  {k}: {len(v)}")
    return layers


def style_block(sid, color, scale=0.8,
                href="http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"):
    return f"""  <Style id="{sid}">
    <IconStyle>
      <color>{color}</color>
      <scale>{scale}</scale>
      <Icon><href>{href}</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0</scale></LabelStyle>
  </Style>"""


def pm(name, desc, lat, lon, surl):
    return f"""    <Placemark>
      <name>{escape(name)}</name>
      <description><![CDATA[{desc}]]></description>
      <styleUrl>#{surl}</styleUrl>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>"""


def det_desc(d, extra=""):
    parts = [
        f"Score: {d['wreck_score']}",
        f"Class: {d['class_name']}",
        f"Confidence: {d['confidence']:.3f}",
        f"Amplitude: {d['peak_amplitude_nt']:.1f} nT",
        f"Extent: {d['spatial_extent_m']:.0f} m",
        f"Aspect Ratio: {d['aspect_ratio']:.2f}",
        f"Detection ID: {d['detection_id']}",
    ]
    if extra:
        parts.insert(0, extra)
    reasons = d.get("score_reasons", [])
    if reasons:
        parts.append(f"Reasons: {'; '.join(reasons)}")
    return "<br/>".join(parts)


def build_kml(wells, erie_layers, huron_layers):
    target_icon = "http://maps.google.com/mapfiles/kml/shapes/target.png"
    forbidden_icon = "http://maps.google.com/mapfiles/kml/shapes/forbidden.png"
    wreck_icon = "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '  <name>Great Lakes ML Detections &amp; Wells</name>',
        '',
        # Shared styles
        style_block("sty_well", COLOR_ORANGE, 0.5),
        # Erie styles
        style_block("sty_erie_high", COLOR_RED, 1.2, target_icon),
        style_block("sty_erie_mid", COLOR_YELLOW, 0.8, target_icon),
        style_block("sty_erie_wh", COLOR_PURPLE, 0.8, forbidden_icon),
        style_block("sty_erie_wreck", COLOR_GREEN, 1.0, wreck_icon),
        # Huron styles
        style_block("sty_huron_high", COLOR_CYAN, 1.2, target_icon),
        style_block("sty_huron_mid", COLOR_BLUE, 0.8, target_icon),
        style_block("sty_huron_wh", COLOR_MAGENTA, 0.8, forbidden_icon),
        style_block("sty_huron_wreck", COLOR_LIME, 1.0, wreck_icon),
        '',
    ]

    # ── Wells ────────────────────────────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>Ontario Wells Over Water (Orange)</name>')
    parts.append('    <visibility>0</visibility>')
    for w in wells:
        desc = (f"Well ID: {w['well_id']}<br/>Status: {w['status']}<br/>"
                f"Class: {w['class']}<br/>Lake: {w['lake']}")
        parts.append(pm(w["name"] or f"Well {w['well_id']}", desc,
                        w["lat"], w["lon"], "sty_well"))
    parts.append('  </Folder>')

    # ── Helper for detection folders ─────────────────────────────────
    def add_det_folder(folder_name, dets, style_id, sort_key=None, desc_fn=None):
        parts.append('  <Folder>')
        parts.append(f'    <name>{folder_name}</name>')
        parts.append('    <visibility>1</visibility>')
        ordered = sorted(dets, key=sort_key or (lambda x: -x.get("wreck_score", 0)))
        for d in ordered:
            desc = desc_fn(d) if desc_fn else det_desc(d)
            parts.append(pm(
                f"Target #{d['detection_id']} (Score {d['wreck_score']})",
                desc, d["lat"], d["lon"], style_id))
        parts.append('  </Folder>')

    def wellhead_desc(d):
        dist = d.get("known_match_distance_m", 0)
        w_name = d.get("known_match_name", "?")
        return det_desc(d, f"Matched Well: {w_name}<br/>Distance: {dist:.0f}m")

    def wreck_desc(d):
        w_name = d.get("known_match_name", "?")
        dist = d.get("known_match_distance_m", 0)
        return det_desc(d, f"Wreck: {w_name}<br/>Distance: {dist:.0f}m")

    # ── ERIE layers ──────────────────────────────────────────────────
    add_det_folder(f"Erie: High-Score Score 7+ (Red) [{len(erie_layers['high_unknown'])}]",
                   erie_layers["high_unknown"], "sty_erie_high")
    add_det_folder(f"Erie: Mid-Tier Score 0-6 (Yellow) [{len(erie_layers['mid_unknown'])}]",
                   erie_layers["mid_unknown"], "sty_erie_mid")
    add_det_folder(f"Erie: Wellhead-Excluded (Purple) [{len(erie_layers['wellhead'])}]",
                   erie_layers["wellhead"], "sty_erie_wh", desc_fn=wellhead_desc)
    add_det_folder(f"Erie: Known Wrecks (Green) [{len(erie_layers['wreck'])}]",
                   erie_layers["wreck"], "sty_erie_wreck", desc_fn=wreck_desc)

    # ── HURON layers ─────────────────────────────────────────────────
    add_det_folder(f"Huron: High-Score Score 7+ (Cyan) [{len(huron_layers['high_unknown'])}]",
                   huron_layers["high_unknown"], "sty_huron_high")
    add_det_folder(f"Huron: Mid-Tier Score 0-6 (Blue) [{len(huron_layers['mid_unknown'])}]",
                   huron_layers["mid_unknown"], "sty_huron_mid")
    add_det_folder(f"Huron: Wellhead-Excluded (Magenta) [{len(huron_layers['wellhead'])}]",
                   huron_layers["wellhead"], "sty_huron_wh", desc_fn=wellhead_desc)
    add_det_folder(f"Huron: Known Wrecks (Lime) [{len(huron_layers['wreck'])}]",
                   huron_layers["wreck"], "sty_huron_wreck", desc_fn=wreck_desc)

    parts.append('</Document>')
    parts.append('</kml>')
    return "\n".join(parts)


def main():
    print("=" * 60)
    print("KMZ Generator — Great Lakes (Erie + Huron)")
    print("=" * 60)

    print("\nLoading wells...")
    wells = load_wells()

    print("\nLoading detections...")
    erie_layers = load_detections(ERIE_JSON, "Erie")
    huron_layers = load_detections(HURON_JSON, "Huron")

    print("\nBuilding KML...")
    kml_text = build_kml(wells, erie_layers, huron_layers)
    print(f"  KML size: {len(kml_text) / 1024:.0f} KB")

    OUT_KMZ.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(OUT_KMZ), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)

    size_mb = OUT_KMZ.stat().st_size / 1024 / 1024
    print(f"\nWrote: {OUT_KMZ}")
    print(f"  Size: {size_mb:.2f} MB")

    erie_total = sum(len(v) for v in erie_layers.values())
    huron_total = sum(len(v) for v in huron_layers.values())
    print(f"\n  Wells: {len(wells)}")
    print(f"  Erie detections: {erie_total}")
    print(f"  Huron detections: {huron_total}")
    print(f"  Combined detections: {erie_total + huron_total}")
    print("\nDone! Open in Google Earth.")


if __name__ == "__main__":
    main()
