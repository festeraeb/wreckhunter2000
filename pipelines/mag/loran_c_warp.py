#!/usr/bin/env python3
"""
_loran_c_warp.py
=================
Compute and apply Loran-C navigation warp correction to ML detections.

Uses known wreck/wellhead matches as ground-truth control points to fit
an affine warp model for each lake (Huron, Erie), then corrects all
detection coordinates.

The GSC aeromagnetic survey grids (1960s-1980s) were positioned using
Loran-C. The systematic position errors (ASFs, propagation effects)
create a smooth spatial warp across each survey grid.  Known wrecks
(diver-confirmed GPS coordinates) and wellheads (modern drill records)
serve as control points to invert the warp.

Outputs:
  wreck_hunting_ml/output/huron_targets.corrected.json
  wreck_hunting_ml/output/erie_full_lake_targets.corrected.json
  wreck_hunting_ml/output/combined_corrected.kml
  wreck_hunting_ml/output/combined_corrected.kmz
"""

import copy
import csv
import json
import math
import sqlite3
import sys
import io
import zipfile
from pathlib import Path

import numpy as np

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"C:\Users\thomf\programming\Bagrecovery")
HURON_JSON = REPO / "wreck_hunting_ml" / "output" / "huron_targets.full.json"
ERIE_JSON  = REPO / "wreck_hunting_ml" / "output" / "erie_full_lake_targets.full.json"
WRECKS_DB  = REPO / "db" / "wrecks.db"
WELLS_CSV  = REPO / "eriewelldata" / "wells.csv"
OUTPUT_DIR = REPO / "wreck_hunting_ml" / "output"

# ── Geometry helpers ─────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon pairs."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


# ── Data loaders ─────────────────────────────────────────────────────────

def load_db_wrecks():
    conn = sqlite3.connect(str(WRECKS_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, latitude, longitude FROM features "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
        "AND found_status = 'found'"
    ).fetchall()
    conn.close()
    return {r["name"]: {"name": r["name"], "lat": r["latitude"], "lon": r["longitude"]}
            for r in rows}


def load_wells():
    wells = []
    with open(WELLS_CSV, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("SUR_LAT83") or 0)
                lon = float(row.get("SUR_LONG83") or 0)
                if lat and lon:
                    name = row.get("WELL_NAME") or row.get("API_NUM") or ""
                    wells.append({"name": name, "lat": lat, "lon": lon})
            except (ValueError, TypeError):
                continue
    return wells


# ── Control point extraction ─────────────────────────────────────────────

def build_wreck_control_points(data, db_wrecks):
    """Extract control points from wreck matches.

    Returns list of (det_lat, det_lon, true_lat, true_lon, label).
    Matches by name AND closest distance (handles duplicate wreck names).
    """
    control = []
    for det in data["detections"]:
        if det.get("known_match") != "wreck":
            continue
        name = det["known_match_name"]
        det_lat, det_lon = det["lat"], det["lon"]
        reported_dist = det.get("known_match_distance_m", 0)

        # Collect all name-matching candidates
        candidates = []
        for dbname, dbw in db_wrecks.items():
            if name.lower() in dbname.lower() or dbname.lower() in name.lower():
                d = haversine(det_lat, det_lon, dbw["lat"], dbw["lon"])
                candidates.append((d, dbw))

        if candidates:
            # Pick the candidate closest to the reported match distance
            candidates.sort(key=lambda c: abs(c[0] - reported_dist))
            best_d, best_w = candidates[0]
            if best_d < 10_000:  # within 10km sanity check
                control.append((det_lat, det_lon, best_w["lat"], best_w["lon"], name))
            else:
                print(f"  Warning: {name} best DB match is {best_d:.0f}m away — skipping")
    return control


def build_well_control_points(data, wells):
    """Extract control points from wellhead matches.

    Matches detection.known_match_name against wells CSV by name,
    then verifies distance is consistent with known_match_distance_m.
    """
    # Index wells by name for fast lookup
    wells_by_name = {}
    for w in wells:
        n = w["name"].strip()
        if n:
            wells_by_name.setdefault(n, []).append(w)

    control = []
    missed = 0
    for det in data["detections"]:
        if det.get("known_match") != "wellhead":
            continue
        name = det["known_match_name"].strip()
        det_lat, det_lon = det["lat"], det["lon"]
        reported_dist = det.get("known_match_distance_m", 0)

        # Exact name match
        candidates = wells_by_name.get(name, [])
        if not candidates:
            # Try partial match
            for wname, wlist in wells_by_name.items():
                if name in wname or wname in name:
                    candidates.extend(wlist)

        # Pick the candidate closest to reported distance
        best = None
        best_diff = float("inf")
        for w in candidates:
            d = haversine(det_lat, det_lon, w["lat"], w["lon"])
            diff = abs(d - reported_dist)
            if diff < best_diff:
                best_diff = diff
                best = w
                best_d = d

        if best and best_diff < 500:  # within 500m of reported distance
            control.append((det_lat, det_lon, best["lat"], best["lon"], name))
        else:
            missed += 1

    if missed:
        print(f"  Warning: {missed} wellhead matches could not be resolved in CSV")
    return control


# ── Affine warp fitting ──────────────────────────────────────────────────

def fit_warp(control_points, label=""):
    """Fit affine warp: true_pos = A @ det_pos + t.

    With n >= 3 non-collinear points, fits full affine (6 params).
    With n < 3, uses constant offset (translation only).

    Returns (params_lat, params_lon) where:
      corrected_lat = params_lat[0]*lat + params_lat[1]*lon + params_lat[2]
      corrected_lon = params_lon[0]*lat + params_lon[1]*lon + params_lon[2]
    """
    n = len(control_points)
    if n == 0:
        print(f"  {label}: NO control points — skipping correction")
        return None, None

    print(f"\n{'='*60}")
    print(f"  {label}: {n} control points")
    print(f"{'='*60}")

    for det_lat, det_lon, true_lat, true_lon, name in control_points:
        d = haversine(det_lat, det_lon, true_lat, true_lon)
        b = bearing(det_lat, det_lon, true_lat, true_lon)
        dlat_m = (true_lat - det_lat) * 111320
        dlon_m = (true_lon - det_lon) * 111320 * math.cos(math.radians(det_lat))
        print(f"    {name:40s}  {d:6.0f}m  {b:5.1f}° {compass(b):4s}  "
              f"dN={dlat_m:+7.0f}m  dE={dlon_m:+7.0f}m")

    if n < 3:
        print(f"  Using constant offset model (n={n} < 3)")
        mean_dlat = sum(cp[2] - cp[0] for cp in control_points) / n
        mean_dlon = sum(cp[3] - cp[1] for cp in control_points) / n
        # Identity + offset:  corrected = 1*lat + 0*lon + mean_dlat
        params_lat = np.array([1.0, 0.0, mean_dlat])
        params_lon = np.array([0.0, 1.0, mean_dlon])
        print(f"  Constant offset:  dLat={mean_dlat:+.6f}°  dLon={mean_dlon:+.6f}°")
        print(f"  (~{mean_dlat * 111320:+.0f}m N/S,  "
              f"~{mean_dlon * 111320 * math.cos(math.radians(control_points[0][0])):+.0f}m E/W)")
        return params_lat, params_lon

    # Full affine: least-squares fit
    A = np.zeros((n, 3))
    b_lat = np.zeros(n)
    b_lon = np.zeros(n)
    for i, (dlat, dlon, tlat, tlon, _) in enumerate(control_points):
        A[i] = [dlat, dlon, 1.0]
        b_lat[i] = tlat
        b_lon[i] = tlon

    params_lat, res_lat, rank_lat, _ = np.linalg.lstsq(A, b_lat, rcond=None)
    params_lon, res_lon, rank_lon, _ = np.linalg.lstsq(A, b_lon, rcond=None)

    # Report residuals
    fitted_lat = A @ params_lat
    fitted_lon = A @ params_lon
    residuals = []
    for i, (dlat, dlon, tlat, tlon, name) in enumerate(control_points):
        rlat = fitted_lat[i] - tlat
        rlon = fitted_lon[i] - tlon
        r_m = math.sqrt((rlat * 111320) ** 2 +
                        (rlon * 111320 * math.cos(math.radians(dlat))) ** 2)
        residuals.append(r_m)

    rms = math.sqrt(sum(r ** 2 for r in residuals) / n)
    print(f"\n  Affine fit (rank lat={rank_lat}, lon={rank_lon}):")
    print(f"    corrected_lat = {params_lat[0]:.8f} * lat + {params_lat[1]:.8f} * lon + {params_lat[2]:.8f}")
    print(f"    corrected_lon = {params_lon[0]:.8f} * lat + {params_lon[1]:.8f} * lon + {params_lon[2]:.8f}")
    print(f"  Residuals per control point:")
    for i, (_, _, _, _, name) in enumerate(control_points):
        print(f"    {name:40s}  {residuals[i]:7.1f}m")
    print(f"  RMS residual: {rms:.1f}m")

    # Show the effective correction at each control point
    print(f"\n  Effective corrections at control points:")
    for dlat, dlon, tlat, tlon, name in control_points:
        clat = params_lat[0] * dlat + params_lat[1] * dlon + params_lat[2]
        clon = params_lon[0] * dlat + params_lon[1] * dlon + params_lon[2]
        shift_m = haversine(dlat, dlon, clat, clon)
        shift_b = bearing(dlat, dlon, clat, clon)
        print(f"    {name:40s}  shift {shift_m:6.0f}m  {shift_b:5.1f}° {compass(shift_b)}")

    return params_lat, params_lon


# ── Apply correction ─────────────────────────────────────────────────────

def apply_correction(lat, lon, params_lat, params_lon):
    new_lat = params_lat[0] * lat + params_lat[1] * lon + params_lat[2]
    new_lon = params_lon[0] * lat + params_lon[1] * lon + params_lon[2]
    return float(new_lat), float(new_lon)


def correct_json(data, params_lat, params_lon, lake_name):
    """Deep-copy data and apply warp correction to all coordinate fields."""
    corrected = copy.deepcopy(data)

    total = 0
    for key in ["detections", "knowns_subtracted", "unknowns_scored"]:
        for det in corrected.get(key, []):
            old_lat, old_lon = det["lat"], det["lon"]
            new_lat, new_lon = apply_correction(old_lat, old_lon, params_lat, params_lon)
            det["lat"] = new_lat
            det["lon"] = new_lon
            det["original_lat"] = old_lat
            det["original_lon"] = old_lon
            total += 1

    corrected.setdefault("summary", {})["loran_c_corrected"] = True
    corrected["summary"]["correction_lake"] = lake_name
    print(f"  Corrected {total} coordinate pairs for {lake_name}")
    return corrected


# ── KML generation ───────────────────────────────────────────────────────

STYLES = {
    "huron_high":   ("ff0000ff", "1.4", "http://maps.google.com/mapfiles/kml/paddle/red-stars.png"),
    "huron_mid":    ("ff00aaff", "1.2", "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"),
    "huron_low":    ("ff00ffff", "1.0", "http://maps.google.com/mapfiles/kml/paddle/wht-stars.png"),
    "erie_high":    ("ff0055ff", "1.4", "http://maps.google.com/mapfiles/kml/paddle/red-diamond.png"),
    "erie_mid":     ("ff55aaff", "1.2", "http://maps.google.com/mapfiles/kml/paddle/ylw-diamond.png"),
    "erie_low":     ("ff55ffff", "1.0", "http://maps.google.com/mapfiles/kml/paddle/wht-diamond.png"),
    "known_wreck":  ("ff00ff00", "1.0", "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
    "known_well":   ("ffff8800", "1.0", "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png"),
    "ref_wreck":    ("ff888888", "0.7", "http://maps.google.com/mapfiles/kml/shapes/shipwreck.png"),
    "control_pt":   ("ff00ff00", "1.2", "http://maps.google.com/mapfiles/kml/paddle/grn-stars.png"),
    "conn_line":    ("9900ffff", None, None),
    "warp_vector":  ("ff00ff00", None, None),
}


def _style_xml(sid, color, scale, icon):
    return (f'<Style id="{sid}">\n'
            f'  <IconStyle><color>{color}</color><scale>{scale}</scale>\n'
            f'    <Icon><href>{icon}</href></Icon></IconStyle>\n'
            f'  <LabelStyle><scale>0.8</scale></LabelStyle>\n'
            f'</Style>\n')


def _line_style_xml(sid, color, width=2):
    return (f'<Style id="{sid}">\n'
            f'  <LineStyle><color>{color}</color><width>{width}</width></LineStyle>\n'
            f'</Style>\n')


def pick_style(det, prefix):
    match = det.get("known_match", "unknown")
    if match == "wellhead":
        return "known_well"
    if match == "wreck":
        return "known_wreck"
    score = det.get("wreck_score", 0)
    if score >= 8:
        return f"{prefix}_high"
    if score >= 5:
        return f"{prefix}_mid"
    return f"{prefix}_low"


def build_combined_kml(huron_data, erie_data, db_wrecks,
                       huron_control, erie_control):
    """Build combined KML with Loran-C-corrected positions."""
    lines = []
    a = lines.append

    a('<?xml version="1.0" encoding="UTF-8"?>')
    a('<kml xmlns="http://www.opengis.net/kml/2.2">')
    a('<Document>')
    a('<name>WreckHunter 2000 — Loran-C Corrected</name>')

    h_sum = huron_data.get("summary", {})
    e_sum = erie_data.get("summary", {})
    a(f'<description>Loran-C warp-corrected detection coordinates.\n'
      f'Huron: {h_sum.get("total_detections", 0)} detections '
      f'({h_sum.get("score_8_plus", 0)} score 8+)\n'
      f'Erie: {e_sum.get("total_detections", 0)} detections '
      f'({e_sum.get("score_8_plus", 0)} score 8+)\n'
      f'Correction uses affine warp from known wreck/well control points.</description>')

    # Styles
    for sid, vals in STYLES.items():
        color, scale, icon = vals
        if icon is not None:
            a(_style_xml(sid, color, scale, icon))
        else:
            a(_line_style_xml(sid, color))

    # ── Huron targets ──
    h_unknowns = huron_data.get("unknowns_scored", [])
    a('<Folder>')
    a(f'<name>Huron Targets — corrected ({len(h_unknowns)})</name>')
    a('<open>1</open>')

    for det in sorted(h_unknowns, key=lambda d: -d.get("wreck_score", 0)):
        lat, lon = det["lat"], det["lon"]
        sid = pick_style(det, "huron")
        score = det.get("wreck_score", 0)
        cls = det.get("class_name", "?")
        conf = det.get("confidence", 0)
        amp = det.get("peak_amplitude_nt", 0)
        reasons = det.get("score_reasons", [])
        olat = det.get("original_lat", lat)
        olon = det.get("original_lon", lon)
        shift_m = haversine(olat, olon, lat, lon)

        desc_parts = [
            f'<b>Score:</b> {score}/10<br/>',
            f'<b>Class:</b> {cls} ({conf:.1%})<br/>',
            f'<b>Amplitude:</b> {amp:.1f} nT<br/>',
            f'<b>Extent:</b> {det.get("spatial_extent_m", 0):.0f} m<br/>',
            f'<b>Aspect ratio:</b> {det.get("aspect_ratio", 0):.2f}<br/>',
            f'<b>Axis offset:</b> {det.get("axis_offset_from_geology_deg", 0):.1f}&deg;<br/>',
            f'<br/><b>Loran-C correction:</b> {shift_m:.0f}m<br/>',
            f'<b>Original:</b> {olat:.6f}, {olon:.6f}<br/>',
            f'<b>Corrected:</b> {lat:.6f}, {lon:.6f}<br/>',
        ]
        if reasons:
            desc_parts.append('<br/><b>Reasons:</b><br/>')
            for r in reasons:
                desc_parts.append(f'&nbsp;&nbsp;&bull; {r}<br/>')

        desc = "".join(desc_parts)
        det_id = det.get("detection_id", "?")

        a('<Placemark>')
        a(f'<name>H{det_id} [{cls}] score={score}</name>')
        a(f'<description><![CDATA[{desc}]]></description>')
        a(f'<styleUrl>#{sid}</styleUrl>')
        a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        a('</Placemark>')

    a('</Folder>')

    # ── Huron known matches ──
    h_knowns = huron_data.get("knowns_subtracted", [])
    a('<Folder>')
    a(f'<name>Huron Known Matches ({len(h_knowns)})</name>')
    for det in h_knowns:
        lat, lon = det["lat"], det["lon"]
        sid = pick_style(det, "huron")
        mtype = det.get("known_match", "?")
        mname = det.get("known_match_name", "?")
        mdist = det.get("known_match_distance_m", -1)
        olat = det.get("original_lat", lat)
        olon = det.get("original_lon", lon)

        desc = (f'<b>Match:</b> {mtype} — {mname}<br/>'
                f'<b>Original dist:</b> {mdist:.0f}m<br/>'
                f'<b>Corrected:</b> {lat:.6f}, {lon:.6f}<br/>'
                f'<b>Original:</b> {olat:.6f}, {olon:.6f}<br/>')

        a('<Placemark>')
        a(f'<name>HM [{mtype}] {mname}</name>')
        a(f'<description><![CDATA[{desc}]]></description>')
        a(f'<styleUrl>#{sid}</styleUrl>')
        a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        a('</Placemark>')
    a('</Folder>')

    # ── Erie targets ──
    e_unknowns = erie_data.get("unknowns_scored", [])
    a('<Folder>')
    a(f'<name>Erie Targets — corrected ({len(e_unknowns)})</name>')
    a('<open>1</open>')

    for det in sorted(e_unknowns, key=lambda d: -d.get("wreck_score", 0)):
        lat, lon = det["lat"], det["lon"]
        sid = pick_style(det, "erie")
        score = det.get("wreck_score", 0)
        cls = det.get("class_name", "?")
        conf = det.get("confidence", 0)
        amp = det.get("peak_amplitude_nt", 0)
        reasons = det.get("score_reasons", [])
        olat = det.get("original_lat", lat)
        olon = det.get("original_lon", lon)
        shift_m = haversine(olat, olon, lat, lon)

        desc_parts = [
            f'<b>Score:</b> {score}/10<br/>',
            f'<b>Class:</b> {cls} ({conf:.1%})<br/>',
            f'<b>Amplitude:</b> {amp:.1f} nT<br/>',
            f'<b>Extent:</b> {det.get("spatial_extent_m", 0):.0f} m<br/>',
            f'<b>Aspect ratio:</b> {det.get("aspect_ratio", 0):.2f}<br/>',
            f'<b>Axis offset:</b> {det.get("axis_offset_from_geology_deg", 0):.1f}&deg;<br/>',
            f'<br/><b>Loran-C correction:</b> {shift_m:.0f}m<br/>',
            f'<b>Original:</b> {olat:.6f}, {olon:.6f}<br/>',
            f'<b>Corrected:</b> {lat:.6f}, {lon:.6f}<br/>',
        ]
        if reasons:
            desc_parts.append('<br/><b>Reasons:</b><br/>')
            for r in reasons:
                desc_parts.append(f'&nbsp;&nbsp;&bull; {r}<br/>')

        desc = "".join(desc_parts)
        det_id = det.get("detection_id", "?")

        a('<Placemark>')
        a(f'<name>E{det_id} [{cls}] score={score}</name>')
        a(f'<description><![CDATA[{desc}]]></description>')
        a(f'<styleUrl>#{sid}</styleUrl>')
        a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        a('</Placemark>')

    a('</Folder>')

    # ── Erie known matches ──
    e_knowns = erie_data.get("knowns_subtracted", [])
    a('<Folder>')
    a(f'<name>Erie Known Matches ({len(e_knowns)})</name>')
    for det in e_knowns:
        lat, lon = det["lat"], det["lon"]
        sid = pick_style(det, "erie")
        mtype = det.get("known_match", "?")
        mname = det.get("known_match_name", "?")
        mdist = det.get("known_match_distance_m", -1)
        olat = det.get("original_lat", lat)
        olon = det.get("original_lon", lon)

        desc = (f'<b>Match:</b> {mtype} — {mname}<br/>'
                f'<b>Original dist:</b> {mdist:.0f}m<br/>'
                f'<b>Corrected:</b> {lat:.6f}, {lon:.6f}<br/>'
                f'<b>Original:</b> {olat:.6f}, {olon:.6f}<br/>')

        a('<Placemark>')
        a(f'<name>EM [{mtype}] {mname}</name>')
        a(f'<description><![CDATA[{desc}]]></description>')
        a(f'<styleUrl>#{sid}</styleUrl>')
        a(f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>')
        a('</Placemark>')
    a('</Folder>')

    # ── Control points + warp vectors ──
    all_control = [(cp, "Huron") for cp in huron_control] + \
                  [(cp, "Erie") for cp in erie_control]
    a('<Folder>')
    a(f'<name>Warp Control Points ({len(all_control)})</name>')
    a('<open>0</open>')

    for (det_lat, det_lon, true_lat, true_lon, name), lake in all_control:
        d = haversine(det_lat, det_lon, true_lat, true_lon)
        b = bearing(det_lat, det_lon, true_lat, true_lon)

        desc = (f'<b>{lake} control point</b><br/>'
                f'Detection: {det_lat:.6f}, {det_lon:.6f}<br/>'
                f'True pos:  {true_lat:.6f}, {true_lon:.6f}<br/>'
                f'Offset: {d:.0f}m at {b:.1f}&deg; ({compass(b)})<br/>')

        a('<Placemark>')
        a(f'<name>CP: {name} ({lake})</name>')
        a(f'<description><![CDATA[{desc}]]></description>')
        a('<styleUrl>#control_pt</styleUrl>')
        a(f'<Point><coordinates>{true_lon:.6f},{true_lat:.6f},0</coordinates></Point>')
        a('</Placemark>')

        # Warp vector line
        a('<Placemark>')
        a(f'<name>warp: {name}</name>')
        a('<styleUrl>#warp_vector</styleUrl>')
        a('<LineString><tessellate>1</tessellate>')
        a(f'<coordinates>{det_lon:.6f},{det_lat:.6f},0 '
          f'{true_lon:.6f},{true_lat:.6f},0</coordinates>')
        a('</LineString></Placemark>')

    a('</Folder>')

    # ── Reference wrecks within range ──
    all_dets = huron_data.get("detections", []) + erie_data.get("detections", [])
    ref_ids = set()
    ref_wrecks = []
    for det in all_dets:
        for dbname, dbw in db_wrecks.items():
            d = haversine(det["lat"], det["lon"], dbw["lat"], dbw["lon"])
            if d < 25_000 and dbname not in ref_ids:
                ref_ids.add(dbname)
                ref_wrecks.append(dbw)

    a('<Folder>')
    a(f'<name>Reference: Known Wrecks ({len(ref_wrecks)})</name>')
    a('<visibility>0</visibility>')
    for w in ref_wrecks:
        a('<Placemark>')
        a(f'<name>{w["name"]}</name>')
        a('<styleUrl>#ref_wreck</styleUrl>')
        a(f'<Point><coordinates>{w["lon"]:.6f},{w["lat"]:.6f},0</coordinates></Point>')
        a('</Placemark>')
    a('</Folder>')

    a('</Document>')
    a('</kml>')
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LORAN-C WARP CORRECTION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    huron_data = json.loads(HURON_JSON.read_text(encoding="utf-8"))
    erie_data = json.loads(ERIE_JSON.read_text(encoding="utf-8"))
    db_wrecks = load_db_wrecks()
    wells = load_wells()
    print(f"  Huron: {len(huron_data['detections'])} detections")
    print(f"  Erie:  {len(erie_data['detections'])} detections")
    print(f"  DB wrecks: {len(db_wrecks)}")
    print(f"  Wells CSV: {len(wells)}")

    # Build control points
    print("\nBuilding control points...")
    huron_wreck_cp = build_wreck_control_points(huron_data, db_wrecks)
    huron_well_cp = build_well_control_points(huron_data, wells)
    huron_control = huron_wreck_cp + huron_well_cp
    print(f"  Huron: {len(huron_wreck_cp)} wreck + {len(huron_well_cp)} wellhead "
          f"= {len(huron_control)} total")

    erie_well_cp = build_well_control_points(erie_data, wells)
    erie_wreck_cp = build_wreck_control_points(erie_data, db_wrecks)
    erie_control = erie_well_cp + erie_wreck_cp
    print(f"  Erie:  {len(erie_well_cp)} wellhead + {len(erie_wreck_cp)} wreck "
          f"= {len(erie_control)} total")

    # Fit warp models
    print("\nFitting warp models...")
    h_plat, h_plon = fit_warp(huron_control, "HURON")
    e_plat, e_plon = fit_warp(erie_control, "ERIE")

    # Apply corrections
    print("\nApplying corrections...")
    if h_plat is not None:
        huron_corrected = correct_json(huron_data, h_plat, h_plon, "Huron")
        out_h = OUTPUT_DIR / "huron_targets.corrected.json"
        out_h.write_text(json.dumps(huron_corrected, indent=2), encoding="utf-8")
        print(f"  Wrote {out_h}")

        # Show how the M&B search zone shifted
        print("\n  Top-10 Huron targets after correction:")
        for det in sorted(huron_corrected["unknowns_scored"],
                          key=lambda d: -d.get("wreck_score", 0))[:10]:
            olat = det.get("original_lat", det["lat"])
            olon = det.get("original_lon", det["lon"])
            shift = haversine(olat, olon, det["lat"], det["lon"])
            brg = bearing(olat, olon, det["lat"], det["lon"])
            print(f"    {det['detection_id']:4d}  score={det['wreck_score']:2d}  "
                  f"{det['class_name']:12s}  "
                  f"({det['lat']:.5f}, {det['lon']:.5f})  "
                  f"shift {shift:.0f}m {compass(brg)}")
    else:
        huron_corrected = huron_data
        print("  Huron: no correction applied")

    if e_plat is not None:
        erie_corrected = correct_json(erie_data, e_plat, e_plon, "Erie")
        out_e = OUTPUT_DIR / "erie_full_lake_targets.corrected.json"
        out_e.write_text(json.dumps(erie_corrected, indent=2), encoding="utf-8")
        print(f"  Wrote {out_e}")

        print("\n  Top-10 Erie targets after correction:")
        for det in sorted(erie_corrected["unknowns_scored"],
                          key=lambda d: -d.get("wreck_score", 0))[:10]:
            olat = det.get("original_lat", det["lat"])
            olon = det.get("original_lon", det["lon"])
            shift = haversine(olat, olon, det["lat"], det["lon"])
            brg = bearing(olat, olon, det["lat"], det["lon"])
            print(f"    {det['detection_id']:4d}  score={det['wreck_score']:2d}  "
                  f"{det['class_name']:12s}  "
                  f"({det['lat']:.5f}, {det['lon']:.5f})  "
                  f"shift {shift:.0f}m {compass(brg)}")
    else:
        erie_corrected = erie_data
        print("  Erie: no correction applied")

    # Generate combined KML
    print("\nGenerating KML/KMZ...")
    kml_text = build_combined_kml(
        huron_corrected, erie_corrected, db_wrecks,
        huron_control, erie_control
    )

    kml_path = OUTPUT_DIR / "combined_corrected.kml"
    kml_path.write_text(kml_text, encoding="utf-8")
    print(f"  Wrote {kml_path}")

    kmz_path = OUTPUT_DIR / "combined_corrected.kmz"
    with zipfile.ZipFile(str(kmz_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)
    print(f"  Wrote {kmz_path}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
