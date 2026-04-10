"""
_generate_kml.py
================
Build a multi-layer KMZ with:
  - Orange  : Ontario wells over water (eriewelldata/wells.csv)
  - Red     : High-score unknown detections (score >= 7)
  - Yellow  : Mid-tier unknown detections (score < 7)
  - Purple  : Wellhead-excluded detections
  - Green   : Known wreck matches

Usage:
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  python scripts\\_generate_kml.py
"""

import csv
import json
import os
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parents[1]
os.chdir(BASE)

WELLS_CSV   = BASE / "eriewelldata" / "wells.csv"
RESULTS_JSON = BASE / "wreck_hunting_ml" / "output" / "erie_full_lake_targets.full.json"
OUT_KMZ     = BASE / "wreck_hunting_ml" / "output" / "erie_detections_and_wells.kmz"

# KML colors: aaBBGGRR
COLOR_ORANGE = "ff0088ff"
COLOR_RED    = "ff0000ff"
COLOR_YELLOW = "ff00ffff"
COLOR_PURPLE = "ffff00a0"
COLOR_GREEN  = "ff00ff00"

# Rough Great-Lakes-over-water bounding boxes (conservative)
# For Ontario wells, offshore wells will be in water areas
LAKE_BOXES = {
    "Lake Erie":    {"lat_min": 41.35, "lat_max": 42.90, "lon_min": -83.60, "lon_max": -78.80},
    "Lake Huron":   {"lat_min": 43.00, "lat_max": 46.30, "lon_min": -84.80, "lon_max": -79.50},
    "Lake Superior":{"lat_min": 46.30, "lat_max": 49.10, "lon_min": -92.20, "lon_max": -84.30},
    "Lake Michigan":{"lat_min": 41.50, "lat_max": 46.10, "lon_min": -88.00, "lon_max": -84.70},
    "Georgian Bay": {"lat_min": 44.50, "lat_max": 45.80, "lon_min": -81.50, "lon_max": -79.80},
    "Lake St Clair":{"lat_min": 42.20, "lat_max": 42.65, "lon_min": -83.10, "lon_max": -82.35},
}


def is_over_water(lat: float, lon: float) -> str | None:
    """Return lake name if the point falls in a rough lake bounding box, else None."""
    for lake, box in LAKE_BOXES.items():
        if box["lat_min"] <= lat <= box["lat_max"] and box["lon_min"] <= lon <= box["lon_max"]:
            return lake
    return None


def load_wells() -> list[dict]:
    """Load wells CSV and filter to those over Great Lakes water."""
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
    print(f"Wells: {len(wells)} over water, {skipped} skipped (no coords/not over water)")
    # Breakdown by lake
    from collections import Counter
    lake_counts = Counter(w["lake"] for w in wells)
    for lake, cnt in sorted(lake_counts.items()):
        print(f"  {lake}: {cnt}")
    return wells


def load_detections() -> dict:
    """Load detection JSON and split into layers."""
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
    dets = data["detections"]
    
    layers = {
        "high_unknown": [],   # Red: score >= 7, unknown
        "mid_unknown": [],    # Yellow: score < 7, unknown
        "wellhead": [],       # Purple: wellhead-excluded
        "wreck": [],          # Green: known wreck
    }
    
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
    
    for k, v in layers.items():
        print(f"  {k}: {len(v)} detections")
    return layers


def style_block(style_id: str, color: str, icon_scale: float = 0.8,
                icon_href: str = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png") -> str:
    return f"""  <Style id="{style_id}">
    <IconStyle>
      <color>{color}</color>
      <scale>{icon_scale}</scale>
      <Icon><href>{icon_href}</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0</scale></LabelStyle>
  </Style>"""


def placemark(name: str, desc: str, lat: float, lon: float, style_url: str) -> str:
    return f"""    <Placemark>
      <name>{escape(name)}</name>
      <description><![CDATA[{desc}]]></description>
      <styleUrl>#{style_url}</styleUrl>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>"""


def build_kml(wells: list[dict], layers: dict) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '  <name>Lake Erie Detections &amp; Wells</name>',
        '',
        style_block("sty_well", COLOR_ORANGE, 0.5),
        style_block("sty_high", COLOR_RED, 1.2,
                    "http://maps.google.com/mapfiles/kml/shapes/target.png"),
        style_block("sty_mid", COLOR_YELLOW, 0.8,
                    "http://maps.google.com/mapfiles/kml/shapes/target.png"),
        style_block("sty_wellhead", COLOR_PURPLE, 0.8,
                    "http://maps.google.com/mapfiles/kml/shapes/forbidden.png"),
        style_block("sty_wreck", COLOR_GREEN, 1.0,
                    "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
        '',
    ]

    # ── Wells layer ──────────────────────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>Ontario Wells Over Water (Orange)</name>')
    parts.append('    <visibility>1</visibility>')
    for w in wells:
        desc = (f"Well ID: {w['well_id']}<br/>"
                f"Status: {w['status']}<br/>"
                f"Class: {w['class']}<br/>"
                f"Lake: {w['lake']}")
        parts.append(placemark(
            w["name"] or f"Well {w['well_id']}",
            desc, w["lat"], w["lon"], "sty_well",
        ))
    parts.append('  </Folder>')

    # ── High-score unknowns (Red) ────────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>High-Score Unknowns - Score 7+ (Red)</name>')
    parts.append('    <visibility>1</visibility>')
    for d in sorted(layers["high_unknown"], key=lambda x: -x.get("wreck_score", 0)):
        desc = (f"Score: {d['wreck_score']}<br/>"
                f"Class: {d['class_name']}<br/>"
                f"Confidence: {d['confidence']:.3f}<br/>"
                f"Amplitude: {d['peak_amplitude_nt']:.1f} nT<br/>"
                f"Detection ID: {d['detection_id']}<br/>"
                f"Reasons: {'; '.join(d.get('score_reasons', []))}")
        parts.append(placemark(
            f"Target #{d['detection_id']} (Score {d['wreck_score']})",
            desc, d["lat"], d["lon"], "sty_high",
        ))
    parts.append('  </Folder>')

    # ── Mid-tier unknowns (Yellow) ───────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>Mid-Tier Unknowns - Score 2-6 (Yellow)</name>')
    parts.append('    <visibility>1</visibility>')
    for d in sorted(layers["mid_unknown"], key=lambda x: -x.get("wreck_score", 0)):
        desc = (f"Score: {d['wreck_score']}<br/>"
                f"Class: {d['class_name']}<br/>"
                f"Confidence: {d['confidence']:.3f}<br/>"
                f"Amplitude: {d['peak_amplitude_nt']:.1f} nT<br/>"
                f"Detection ID: {d['detection_id']}<br/>"
                f"Reasons: {'; '.join(d.get('score_reasons', []))}")
        parts.append(placemark(
            f"Target #{d['detection_id']} (Score {d['wreck_score']})",
            desc, d["lat"], d["lon"], "sty_mid",
        ))
    parts.append('  </Folder>')

    # ── Wellhead-excluded (Purple) ───────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>Wellhead-Excluded (Purple)</name>')
    parts.append('    <visibility>1</visibility>')
    for d in sorted(layers["wellhead"], key=lambda x: x.get("known_match_distance_m", 9999)):
        dist = d.get("known_match_distance_m", 0)
        well_name = d.get("known_match_name", "unknown")
        desc = (f"Matched Well: {well_name}<br/>"
                f"Distance: {dist:.0f} m<br/>"
                f"Class: {d['class_name']}<br/>"
                f"Confidence: {d['confidence']:.3f}<br/>"
                f"Amplitude: {d['peak_amplitude_nt']:.1f} nT<br/>"
                f"Detection ID: {d['detection_id']}")
        parts.append(placemark(
            f"Well-Excl #{d['detection_id']} ({dist:.0f}m from {well_name})",
            desc, d["lat"], d["lon"], "sty_wellhead",
        ))
    parts.append('  </Folder>')

    # ── Known wreck matches (Green) ──────────────────────────────────
    parts.append('  <Folder>')
    parts.append('    <name>Known Wreck Matches (Green)</name>')
    parts.append('    <visibility>1</visibility>')
    for d in layers["wreck"]:
        wreck_name = d.get("known_match_name", "unknown")
        dist = d.get("known_match_distance_m", 0)
        desc = (f"Wreck: {wreck_name}<br/>"
                f"Distance: {dist:.0f} m<br/>"
                f"Score: {d['wreck_score']}<br/>"
                f"Class: {d['class_name']}<br/>"
                f"Amplitude: {d['peak_amplitude_nt']:.1f} nT<br/>"
                f"Detection ID: {d['detection_id']}")
        parts.append(placemark(
            f"Wreck: {wreck_name} (#{d['detection_id']})",
            desc, d["lat"], d["lon"], "sty_wreck",
        ))
    parts.append('  </Folder>')

    parts.append('</Document>')
    parts.append('</kml>')
    return "\n".join(parts)


def main():
    print("=" * 60)
    print("KMZ Generator — Lake Erie Detections & Wells")
    print("=" * 60)

    print("\nLoading wells...")
    wells = load_wells()

    print("\nLoading detections...")
    layers = load_detections()

    print("\nBuilding KML...")
    kml_text = build_kml(wells, layers)
    print(f"  KML size: {len(kml_text) / 1024:.0f} KB")

    # Write as KMZ (zipped KML)
    OUT_KMZ.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(OUT_KMZ), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)

    size_mb = OUT_KMZ.stat().st_size / 1024 / 1024
    print(f"\nWrote: {OUT_KMZ}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Wells: {len(wells)}")
    print(f"  High-score (Red): {len(layers['high_unknown'])}")
    print(f"  Mid-tier (Yellow): {len(layers['mid_unknown'])}")
    print(f"  Wellhead-excl (Purple): {len(layers['wellhead'])}")
    print(f"  Wreck matches (Green): {len(layers['wreck'])}")
    print("\nDone! Open in Google Earth.")


if __name__ == "__main__":
    main()
