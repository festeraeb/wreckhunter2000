"""
_directional_pull.py
=====================
Analyze directional offset (pull) between known wreck positions and
nearest Huron inference detections.

Tests the user's prediction that detections would show systematic
directional displacement relative to true wreck locations.
"""
import json, math, sqlite3, sys, io
from pathlib import Path

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPO = Path(r"C:\Users\thomf\programming\Bagrecovery")
RESULTS = REPO / "wreck_hunting_ml" / "output" / "huron_targets.full.json"
WRECKS_DB = REPO / "db" / "wrecks.db"

# Known wrecks from training script (ground truth positions — diver-confirmed)
KNOWN_STEEL = [
    {"name": "SS Cedarville",              "lat": 45.9035, "lon": -84.7300},
    {"name": "SS Daniel J. Morrell (stern)","lat": 44.2580, "lon": -82.8348},
    {"name": "SS Daniel J. Morrell (bow)",  "lat": 44.3053, "lon": -82.7527},
    {"name": "SS James Carruthers",         "lat": 44.1736, "lon": -81.6406},
    {"name": "SS Charles S. Price",         "lat": 43.1529, "lon": -82.3529},
    {"name": "SS John McGean",              "lat": 43.9533, "lon": -82.5286},
    {"name": "SS Hydrus",                   "lat": 43.2700, "lon": -82.5300},
    {"name": "SS Argus",                    "lat": 44.1736, "lon": -81.6406},
    {"name": "SS Canisteo",                 "lat": 43.2357, "lon": -82.3049},
    {"name": "SS Glenorchy",                "lat": 43.8097, "lon": -82.5299},
    {"name": "SS North Star",               "lat": 43.3993, "lon": -82.4420},
    {"name": "SS Regina",                   "lat": 43.3411, "lon": -82.4483},
    {"name": "SS Albany",                    "lat": 44.1059, "lon": -82.7003},
]

KNOWN_WOOD = [
    {"name": "Gov. Smith",       "lat": 44.1556, "lon": -82.7000},
    {"name": "Philadelphia",     "lat": 44.0687, "lon": -82.7165},
    {"name": "City of Detroit",  "lat": 44.2079, "lon": -83.0140},
    {"name": "Jacob Bertschy",   "lat": 44.0572, "lon": -82.8846},
    {"name": "Iron Chief",       "lat": 44.0939, "lon": -82.7098},
    {"name": "Troy",             "lat": 44.1442, "lon": -83.0323},
    {"name": "Goliath",          "lat": 43.7835, "lon": -82.5454},
    {"name": "E.P. Dorr",        "lat": 44.1462, "lon": -82.7330},
    {"name": "Waverly",          "lat": 43.7645, "lon": -82.5136},
    {"name": "Fred Lee",         "lat": 44.2071, "lon": -82.7600},
]

ALL_KNOWN = [(w, "steel") for w in KNOWN_STEEL] + [(w, "wood") for w in KNOWN_WOOD]

# Also load "found" wrecks from the features table in wrecks.db
# (this is where inference scorer gets Etruria, W.H. Gilbert, etc.)
DB_WRECKS = []
if WRECKS_DB.exists():
    conn = sqlite3.connect(str(WRECKS_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT name, latitude, longitude FROM features "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
            "AND found_status = 'found'"
        ).fetchall()
        DB_WRECKS = [{"name": r["name"], "lat": r["latitude"], "lon": r["longitude"]} for r in rows]
    except Exception as e:
        # Try unfound wrecks too — any with coordinates
        try:
            rows = conn.execute(
                "SELECT name, latitude, longitude, found_status FROM features "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
            ).fetchall()
            DB_WRECKS = [{"name": r["name"], "lat": r["latitude"], "lon": r["longitude"], 
                          "found": r["found_status"]} for r in rows]
        except Exception as e2:
            print(f"Warning: could not query features: {e2}")
    conn.close()

# ── Helpers ──────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon pairs."""
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to point 2 (0=N, 90=E, 180=S, 270=W)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    x = math.sin(dλ) * math.cos(φ2)
    y = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.atan2(x, y)
    return (math.degrees(θ) + 360) % 360

def compass(deg):
    """Convert degree bearing to compass direction."""
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(deg/22.5) % 16]

# ── Load detections ──────────────────────────────────────────────────────

with open(RESULTS) as f:
    data = json.load(f)
dets = data["detections"]
print(f"Loaded {len(dets)} Huron detections")

# ── DB wrecks loaded above from features table ─────────────────────────

db_wrecks = {w["name"]: w for w in DB_WRECKS}
print(f"Loaded {len(db_wrecks)} found wrecks from wrecks.db features table")

# Show the ones matched in inference
for mname in ["Etruria", "Albany", "W.H. Gilbert"]:
    matches = [n for n in db_wrecks if mname.lower() in n.lower()]
    for n in matches:
        w = db_wrecks[n]
        print(f"  DB wreck: {n} -> {w['lat']:.4f}, {w['lon']:.4f}")

# ── Find the 3 matched wreck detections from the analysis ───────────────

print("\n" + "="*70)
print("MATCHED WRECK DETECTIONS — Directional Offset")
print("="*70)

matched = [d for d in dets if d["known_match"] == "wreck"]
for d in matched:
    det_lat, det_lon = d["lat"], d["lon"]
    name = d["known_match_name"]
    dist = d["known_match_distance_m"]
    
    # Find actual wreck position
    wreck_pos = None
    # Check training lists first (most precise coords)
    for w, wtype in ALL_KNOWN:
        if w["name"] in name or name in w["name"]:
            wreck_pos = (w["lat"], w["lon"], f"training-{wtype}")
            break
    # Check wrecks.db features table
    if wreck_pos is None:
        for dbname, dbw in db_wrecks.items():
            if name.lower() in dbname.lower() or dbname.lower() in name.lower():
                wreck_pos = (dbw["lat"], dbw["lon"], "db-features")
                break
    
    if wreck_pos:
        wlat, wlon, src = wreck_pos
        b = bearing(wlat, wlon, det_lat, det_lon)
        d_calc = haversine(wlat, wlon, det_lat, det_lon)
        # Also compute N-S and E-W components
        dlat_m = (det_lat - wlat) * 111320
        dlon_m = (det_lon - wlon) * 111320 * math.cos(math.radians(wlat))
        print(f"\n  {name} ({src})")
        print(f"    Wreck:     {wlat:.4f}, {wlon:.4f}")
        print(f"    Detection: {det_lat:.4f}, {det_lon:.4f}")
        print(f"    Distance:  {d_calc:.0f}m  (reported: {dist:.0f}m)")
        print(f"    Bearing:   {b:.1f}° ({compass(b)})")
        print(f"    ΔN-S:      {dlat_m:+.0f}m  ΔE-W: {dlon_m:+.0f}m")
        print(f"    Class:     {d['class_name']}  conf={d['confidence']:.3f}  score={d['wreck_score']}")
    else:
        print(f"\n  {name} — NO POSITION FOUND in training lists or wrecks.db")
        print(f"    Detection: {det_lat:.4f}, {det_lon:.4f}")
        print(f"    Dist: {dist:.0f}m  Class: {d['class_name']}  conf={d['confidence']:.3f}")

# ── Broader analysis: nearest detection to each known wreck ─────────────

print("\n" + "="*70)
print("ALL KNOWN WRECKS — Nearest Detection Analysis")
print("="*70)
print(f"{'Wreck':<32} {'Type':6} {'Dist':>6} {'Bearing':>8} {'Dir':4} {'ΔN-S':>8} {'ΔE-W':>8} {'Class':12} {'Conf':>5} {'Scr':>3}")
print("─"*110)

bearings_all = []
bearings_near = []  # only <2km
ns_offsets = []
ew_offsets = []

for w, wtype in ALL_KNOWN:
    wlat, wlon = w["lat"], w["lon"]
    
    # Find nearest detection
    best_dist = float('inf')
    best_det = None
    for d in dets:
        dist = haversine(wlat, wlon, d["lat"], d["lon"])
        if dist < best_dist:
            best_dist = dist
            best_det = d
    
    if best_det and best_dist < 5000:  # within 5km
        b = bearing(wlat, wlon, best_det["lat"], best_det["lon"])
        dlat_m = (best_det["lat"] - wlat) * 111320
        dlon_m = (best_det["lon"] - wlon) * 111320 * math.cos(math.radians(wlat))
        
        bearings_all.append(b)
        ns_offsets.append(dlat_m)
        ew_offsets.append(dlon_m)
        if best_dist < 2000:
            bearings_near.append(b)
        
        print(f"  {w['name']:<30} {wtype:6} {best_dist:6.0f}m {b:7.1f}° {compass(b):4}  {dlat_m:+7.0f}m {dlon_m:+7.0f}m  {best_det['class_name']:12} {best_det['confidence']:.3f} {best_det['wreck_score']:3d}")
    elif best_det:
        print(f"  {w['name']:<30} {wtype:6} {best_dist:6.0f}m  (>5km, no nearby detection)")
    else:
        print(f"  {w['name']:<30} {wtype:6}  NO DETECTIONS")

# ── Summary statistics ──────────────────────────────────────────────────

print("\n" + "="*70)
print("DIRECTIONAL PULL SUMMARY")
print("="*70)

if bearings_all:
    # Circular mean bearing
    sin_sum = sum(math.sin(math.radians(b)) for b in bearings_all)
    cos_sum = sum(math.cos(math.radians(b)) for b in bearings_all)
    mean_bearing = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360
    # Circular variance (0=all aligned, 1=uniform)
    R = math.sqrt(sin_sum**2 + cos_sum**2) / len(bearings_all)
    circ_var = 1 - R
    
    print(f"\n  All wrecks within 5km (n={len(bearings_all)}):")
    print(f"    Mean bearing:       {mean_bearing:.1f}° ({compass(mean_bearing)})")
    print(f"    Resultant length R: {R:.3f}  (1.0=perfect alignment, 0=random)")
    print(f"    Circular variance:  {circ_var:.3f}  (0=aligned, 1=random)")
    print(f"    Mean ΔN-S:          {sum(ns_offsets)/len(ns_offsets):+.0f}m")
    print(f"    Mean ΔE-W:          {sum(ew_offsets)/len(ew_offsets):+.0f}m")
    
    # Bearing distribution by quadrant
    quads = {"N (315-45)": 0, "E (45-135)": 0, "S (135-225)": 0, "W (225-315)": 0}
    for b in bearings_all:
        if b >= 315 or b < 45:     quads["N (315-45)"] += 1
        elif 45 <= b < 135:        quads["E (45-135)"] += 1
        elif 135 <= b < 225:       quads["S (135-225)"] += 1
        else:                      quads["W (225-315)"] += 1
    print(f"\n    Quadrant distribution:")
    for q, n in quads.items():
        bar = "#" * (n * 3)
        print(f"      {q}: {n:2d}  {bar}")

if bearings_near:
    sin_sum = sum(math.sin(math.radians(b)) for b in bearings_near)
    cos_sum = sum(math.cos(math.radians(b)) for b in bearings_near)
    mean_bearing = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360
    R = math.sqrt(sin_sum**2 + cos_sum**2) / len(bearings_near)
    
    print(f"\n  Close matches only (<2km, n={len(bearings_near)}):")
    print(f"    Mean bearing:       {mean_bearing:.1f}° ({compass(mean_bearing)})")
    print(f"    Resultant length R: {R:.3f}")

# ── Flight line vs strike angle analysis ────────────────────────────────

print(f"\n  Reference angles:")
print(f"    N-S Canadian Shield strike: 5°")
print(f"    If pull is systematic, bearings should cluster near one direction")
print(f"    If random (tile quantization), bearings should be roughly uniform")

# ── DB wrecks that were actually matched in inference ───────────────────

print("\n" + "="*70)
print("DB WRECKS (features table) — Nearest Detection")
print("="*70)
print("  These are wrecks from the Swayze/features DB. Coords may be approximate.")

# Check inference-matched wrecks against DB positions
for dw_name, dw in db_wrecks.items():
    wlat, wlon = dw["lat"], dw["lon"]
    # Skip if coords are obviously approximate (whole degrees)
    approx = (wlat == round(wlat)) and (wlon == round(wlon))
    
    best_dist = float('inf')
    best_det = None
    for d in dets:
        dist = haversine(wlat, wlon, d["lat"], d["lon"])
        if dist < best_dist:
            best_dist = dist
            best_det = d
    
    if best_det and best_dist < 5000:
        b = bearing(wlat, wlon, best_det["lat"], best_det["lon"])
        dlat_m = (best_det["lat"] - wlat) * 111320
        dlon_m = (best_det["lon"] - wlon) * 111320 * math.cos(math.radians(wlat))
        flag = " *APPROX*" if approx else ""
        print(f"  {dw_name:<40} {best_dist:6.0f}m {b:7.1f}° {compass(b):4}  ΔN-S:{dlat_m:+7.0f}m ΔE-W:{dlon_m:+7.0f}m{flag}")

# Individual bearing listing
print(f"\n  Individual offset bearings:")
for i, b in enumerate(bearings_all):
    print(f"    {b:6.1f}° ({compass(b)})")
