#!/usr/bin/env python3
"""Export Lake Huron mag anomalies + dipole analysis to KML/KMZ."""
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from dipole_analysis import analyze_candidate, find_best_tif

# ── Load data ────────────────────────────────────────────────────────────────
water = json.loads((REPO / "mag_huron_water_scan.json").read_text())

wd = json.loads((REPO / "webapp" / "data" / "wrecks_full.json").read_text())
wrecks_all = [w for w in wd.values() if isinstance(w, dict)]
huron_wrecks = [
    w for w in wrecks_all
    if isinstance(w.get("latitude"), (int, float))
    and isinstance(w.get("longitude"), (int, float))
    and 43.0 <= w["latitude"] <= 46.5
    and -84.5 <= w["longitude"] <= -79.5
]
steel_wrecks = [
    w for w in huron_wrecks
    if "steel" in str(w.get("hull_material", "")).lower()
    or w.get("is_steel_freighter")
]


def dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.0
    dx = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dy * dy + dx * dx)


def nearest_wrecks(lat, lon, n=3, radius=15):
    hits = []
    for w in huron_wrecks:
        d = dist_km(lat, lon, w["latitude"], w["longitude"])
        if d < radius:
            hits.append((d, w.get("name", "?"), w))
    hits.sort(key=lambda x: x[0])
    return hits[:n]


# ── Dipole analysis for each candidate ───────────────────────────────────────
print("Running dipole analysis on %d candidates..." % len(water))
dipole_results = {}
for c in water:
    key = "%.4f_%.4f" % (c["lat"], c["lon"])
    try:
        tif = find_best_tif(c["lat"], c["lon"])
        r = analyze_candidate(tif, c["lat"], c["lon"])
        if "dipole" in r:
            dipole_results[key] = r
    except Exception:
        pass
print("  %d / %d analyzed successfully" % (len(dipole_results), len(water)))


# ── Build KML ────────────────────────────────────────────────────────────────
kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
doc = ET.SubElement(kml, "Document")
ET.SubElement(doc, "name").text = "Lake Huron Magnetic Anomalies"
ET.SubElement(doc, "description").text = (
    "Water-only magnetic anomaly scan across 6 data sources.\n"
    "Dipole analysis (polarity flip, edge gradient, lobe ratio) by dipole_analysis.py.\n"
    "Generated %s from mag_huron_water_scan.json." % "2026-03-08"
)

# ── Styles ────────────────────────────────────────────────────────────────────
style_map = {
    "likely_manmade": ("ff0000ff", "1.3", "http://maps.google.com/mapfiles/kml/paddle/red-stars.png"),
    "possibly_manmade": ("ff00aaff", "1.1", "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"),
    "ambiguous": ("ff00ffff", "1.0", "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"),
    "geological": ("ff00aa00", "0.9", "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"),
    "unknown": ("ffaaaaaa", "0.9", "http://maps.google.com/mapfiles/kml/paddle/wht-circle.png"),
    "steel_wreck": ("ff888888", "0.7", "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
}
for sid, (color, scale, icon) in style_map.items():
    s = ET.SubElement(doc, "Style", id=sid)
    ist = ET.SubElement(s, "IconStyle")
    ET.SubElement(ist, "color").text = color
    ET.SubElement(ist, "scale").text = scale
    ic = ET.SubElement(ist, "Icon")
    ET.SubElement(ic, "href").text = icon


def pick_style(verdict):
    v = verdict.upper()
    if "LIKELY MAN" in v:
        return "likely_manmade"
    if "POSSIBLY" in v:
        return "possibly_manmade"
    if "AMBIGUOUS" in v:
        return "ambiguous"
    if "GEOL" in v:
        return "geological"
    return "unknown"


# ── Folder: Anomaly Candidates ───────────────────────────────────────────────
folder = ET.SubElement(doc, "Folder")
ET.SubElement(folder, "name").text = "Mag Anomalies — water-only (z > 2)"

for i, c in enumerate(water):
    lat, lon = c["lat"], c["lon"]
    key = "%.4f_%.4f" % (lat, lon)
    dp = dipole_results.get(key, {})

    cls = dp.get("classification", {})
    verdict = cls.get("verdict", "UNKNOWN")
    score = cls.get("score_manmade_pct", 0)

    # Nearby wrecks
    nw = nearest_wrecks(lat, lon)
    if nw:
        wreck_lines = ["Nearest wrecks:"]
        for d, name, w in nw:
            mat = "STEEL" if ("steel" in str(w.get("hull_material", "")).lower() or w.get("is_steel_freighter")) else str(w.get("hull_material", "?"))
            found = w.get("found_status", "?")
            wreck_lines.append("  %.1f km: %s (%s) [%s]" % (d, name, mat, found))
        wreck_text = "\n".join(wreck_lines)
    else:
        wreck_text = "NO KNOWN WRECKS within 15 km — NOVEL ANOMALY"

    # Build description
    lines = [
        "Rank: #%d" % (i + 1),
        "Source: %s" % c["source"],
        "Value: %+.1f nT  (z-score = %.2f)" % (c["value_nT"], c["z_score"]),
        "Type: %s" % c["type"],
        "",
    ]

    dip = dp.get("dipole", {})
    flip = dp.get("flip", {})
    grad = dp.get("gradient", {})
    anom = dp.get("anomaly", {})
    shape = dp.get("shape", {})

    if dp:
        lines += [
            "═══ Dipole Analysis ═══",
            "Peak: %+.1f nT   Trough: %+.1f nT   |Peak|: %.1f nT" % (
                anom.get("peak_above_bg_nT", 0),
                anom.get("trough_below_bg_nT", 0),
                anom.get("peak_abs_nT", 0)),
            "SNR vs background: %.1f" % (anom.get("snr_vs_bg_std", 0) or 0),
            "",
            "Dipolar: %s" % ("YES" if dip.get("is_dipolar") else "NO"),
            "Lobe symmetry ratio: %s" % (
                "%.2f" % dip["lobe_symmetry_ratio"] if dip.get("lobe_symmetry_ratio") else "N/A"),
            "Dipole separation: %s" % (
                "%d m" % dip["separation_m"] if dip.get("separation_m") else "N/A"),
            "Dipole azimuth: %s" % (
                "%.0f°" % dip["azimuth_deg"] if dip.get("azimuth_deg") else "N/A"),
            "",
            "Polarity flip (min): %s" % (
                "%d m" % flip["min_flip_distance_m"] if flip.get("min_flip_distance_m") else "NO FLIP"),
            "Polarity flip (mean): %s" % (
                "%d m" % flip["mean_flip_distance_m"] if flip.get("mean_flip_distance_m") else "N/A"),
            "Flip directions: %d / 8" % flip.get("directions_measured", 0),
            "",
            "Gradient peak: %s" % (
                "%.4f nT/m" % grad["peak_nT_per_m"] if grad.get("peak_nT_per_m") else "N/A"),
            "Gradient contrast: %s" % (
                "%.1fx background" % grad["contrast_ratio"] if grad.get("contrast_ratio") else "N/A"),
            "",
            "Aspect ratio: %s" % (
                "%.1f" % shape["aspect_ratio"] if shape.get("aspect_ratio") else "N/A"),
            "",
            "══════════════════════",
            "VERDICT: %s  (score %d / 100)" % (verdict, score),
        ]
        reasons = cls.get("reasons", [])
        if reasons:
            lines.append("")
            lines.extend(reasons)

    lines += ["", wreck_text]

    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = "#%d  %+.0f nT  z=%.1f  [%s]" % (
        i + 1, c["value_nT"], c["z_score"], c["source"][:15])
    ET.SubElement(pm, "styleUrl").text = "#" + pick_style(verdict)
    ET.SubElement(pm, "description").text = "\n".join(lines)
    pt = ET.SubElement(pm, "Point")
    ET.SubElement(pt, "coordinates").text = "%.6f,%.6f,0" % (lon, lat)


# ── Folder: Known Steel Wrecks (reference) ───────────────────────────────────
wfolder = ET.SubElement(doc, "Folder")
ET.SubElement(wfolder, "name").text = "Known Steel Wrecks — reference (%d)" % len(steel_wrecks)
ET.SubElement(wfolder, "visibility").text = "0"

for w in steel_wrecks:
    wpm = ET.SubElement(wfolder, "Placemark")
    ET.SubElement(wpm, "name").text = w.get("name", "?")
    ET.SubElement(wpm, "styleUrl").text = "#steel_wreck"
    ET.SubElement(wpm, "description").text = (
        "Material: steel\nFound: %s\nDate: %s\nType: %s" % (
            w.get("found_status", "?"),
            w.get("date", "?"),
            w.get("feature_type", "?"),
        )
    )
    wpt = ET.SubElement(wpm, "Point")
    ET.SubElement(wpt, "coordinates").text = "%.6f,%.6f,0" % (w["longitude"], w["latitude"])


# ── Write files ──────────────────────────────────────────────────────────────
tree = ET.ElementTree(kml)
ET.indent(tree, space="  ")

out_kml = REPO / "mag_huron_anomalies.kml"
tree.write(str(out_kml), xml_declaration=True, encoding="UTF-8")
print("Wrote %s  (%d anomalies + %d steel wrecks)" % (out_kml.name, len(water), len(steel_wrecks)))

out_kmz = REPO / "mag_huron_anomalies.kmz"
with zipfile.ZipFile(str(out_kmz), "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(str(out_kml), "doc.kml")
print("Wrote %s  (%.0f KB)" % (out_kmz.name, out_kmz.stat().st_size / 1024))
