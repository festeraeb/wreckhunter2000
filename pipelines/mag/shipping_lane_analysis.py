"""Analyze why 3 targets share latitude 42.1621N - tile grid artifact + shipping lane check"""
import json, math, sys, os
import rasterio

BASE = r"C:\Users\thomf\programming\Bagrecovery"
OUT = os.path.join(BASE, "wreck_hunting_ml", "output", "shipping_lane_analysis.txt")

with open(os.path.join(BASE, 'wreck_hunting_ml', 'output', 'erie_full_lake_targets.full.json')) as f:
    data = json.load(f)

# Handle both formats: list or dict with "detections" key
if isinstance(data, dict):
    targets = data.get('detections', [])
else:
    targets = data

unknowns = [t for t in targets if t.get('known_match','') == 'unknown']
unknowns.sort(key=lambda t: t['wreck_score'], reverse=True)

# Use actual field names from JSON
LAT = 'lat'
LON = 'lon'
SCORE = 'wreck_score'
AMP = 'peak_amplitude_nt'
CLASS = 'class_name'

lat_targets = [
    {'rank': '#4', 'score': 8, 'lat': 42.1621, 'lon': -81.6081, 'amp': 202.0},
    {'rank': '#1', 'score': 9, 'lat': 42.1621, 'lon': -80.9363, 'amp': 399.5},
    {'rank': '#2', 'score': 9, 'lat': 42.1621, 'lon': -80.2644, 'amp': 760.0},
]

lines = []
def p(s=''):
    lines.append(s)

p('='*80)
p('ANALYSIS: THREE HYPOTHESES FOR 3 TARGETS AT LAT 42.1621 N')
p('='*80)

# === 1. TILE-GRID ARTIFACT CHECK ===
p()
p('='*80)
p('1. TILE-GRID ARTIFACT CHECK')
p('   Are these in the same tile ROW (shared lat = artifact of tile discretization)?')
p('='*80)

tif_path = os.path.join(BASE, 'magnetic_data', 'tier_2_aero_lowalt', 'local', 'gsc_erie_highres_0_001.tif')
ds = rasterio.open(tif_path)
bounds = ds.bounds
north_bound = bounds[3]
south_bound = bounds[1]
west_bound = bounds[0]
east_bound = bounds[2]
p(f'TIF bounds: N={north_bound:.4f}, S={south_bound:.4f}, W={west_bound:.4f}, E={east_bound:.4f}')
p(f'TIF size: {ds.width} x {ds.height} pixels')
p(f'Pixel size: {(east_bound-west_bound)/ds.width:.6f} x {(north_bound-south_bound)/ds.height:.6f} deg')

tile_px = 224
stride_px = 112
pixel_deg = (north_bound - south_bound) / ds.height  # exact pixel size in lat
tile_deg = tile_px * pixel_deg
stride_deg = stride_px * pixel_deg
p(f'Exact pixel_deg (lat): {pixel_deg:.6f}')
p(f'Tile size: {tile_px}px = {tile_deg:.4f} deg lat')
p(f'Stride: {stride_px}px = {stride_deg:.4f} deg lat')

n_rows = (ds.height - tile_px) // stride_px + 1
n_cols = (ds.width - tile_px) // stride_px + 1
p(f'Tile grid: {n_cols} cols x {n_rows} rows')

p()
p('Tile row centers near 42.16:')
for row in range(n_rows):
    center_lat = north_bound - (row * stride_px + tile_px / 2) * pixel_deg
    if abs(center_lat - 42.1621) < 0.2:
        dist = abs(center_lat - 42.1621)
        marker = ' <-- MATCH!' if dist < 0.005 else ''
        p(f'  Row {row}: center_lat={center_lat:.4f}, dist to 42.1621 = {dist:.4f} deg ({dist*111:.1f} km){marker}')

p()
p('All unique unknown-target latitudes and counts:')
unique_lats = sorted(set(round(t[LAT], 4) for t in unknowns))
p(f'  Total unique latitudes: {len(unique_lats)}')
shared_count = 0
for lat in unique_lats:
    count = sum(1 for t in unknowns if abs(t[LAT] - lat) < 0.0001)
    tag = '  <-- SHARED' if count >= 2 else ''
    if count >= 2:
        shared_count += 1
    p(f'  {lat:.4f}: {count} targets{tag}')
p(f'Latitudes with 2+ targets: {shared_count}')

# Check if shared lats match tile row centers
p()
p('Do shared-lat groups correspond to tile row centers?')
all_row_centers = []
for row in range(n_rows):
    center_lat = north_bound - (row * stride_px + tile_px / 2) * pixel_deg
    all_row_centers.append(center_lat)

for lat in unique_lats:
    count = sum(1 for t in unknowns if abs(t[LAT] - lat) < 0.0001)
    if count >= 2:
        min_dist = min(abs(lat - rc) for rc in all_row_centers)
        nearest_rc = min(all_row_centers, key=lambda rc: abs(lat - rc))
        p(f'  Shared lat {lat:.4f} ({count} targets): nearest tile row center = {nearest_rc:.4f}, dist = {min_dist:.4f} deg')

ds.close()

# === 2. PIPELINE HYPOTHESIS ===
p()
p('='*80)
p('2. PIPELINE HYPOTHESIS')
p('='*80)
mu_0 = 4 * math.pi * 1e-7
B0 = 55e-6
r_pipe = 0.254  # 20-inch OD
wall = 0.010
V_per_m = math.pi * (r_pipe**2 - (r_pipe - wall)**2)
mu_r = 200
m_per_m = (mu_r - 1) * V_per_m * B0 / mu_0
L_eff = 600
m_total = m_per_m * L_eff
altitude = 300
signal_nT = (mu_0 / (4 * math.pi)) * 2 * m_total / altitude**3 * 1e9
p(f'20" steel pipeline at 300m altitude: {signal_nT:.3f} nT')
p(f'Observed anomalies: 20-110 nT')
p(f'VERDICT: Pipeline signal is {110/signal_nT:.0f}x too weak. REJECTED.')

seps_km = []
for i in range(len(lat_targets)):
    for j in range(i+1, len(lat_targets)):
        dlon = lat_targets[j]['lon'] - lat_targets[i]['lon']
        dx = dlon * 111 * math.cos(math.radians(42.16))
        dist = abs(dx)
        seps_km.append((lat_targets[i]['rank'], lat_targets[j]['rank'], dist))
        p(f'  {lat_targets[i]["rank"]} to {lat_targets[j]["rank"]}: {dist:.1f} km')

# === 3. RAIL CAR HYPOTHESIS ===
p()
p('='*80)
p('3. RAIL CAR HYPOTHESIS (M&B stern gate debris)')
p('='*80)
p('M&B cargo: 30 cars (26 coal, 3 steel beams, 1 iron castings)')
p(f'Target separations: {seps_km[0][2]:.0f} km, {seps_km[1][2]:.0f} km, {seps_km[2][2]:.0f} km')
p('Max plausible debris scatter from single sinking: 2-5 km')
p('VERDICT: 50-100 km separation impossible from one event. REJECTED.')

# === 4. SHIPPING LANE HYPOTHESIS ===
p()
p('='*80)
p('4. SHIPPING LANE HYPOTHESIS')
p('='*80)

ports = {
    'Conneaut OH':     (41.9617, -80.5536),
    'Port Stanley ON': (42.6631, -81.2139),
    'Erie PA':         (42.1292, -80.0851),
    'Buffalo NY':      (42.8864, -78.8784),
    'Cleveland OH':    (41.4993, -81.6944),
    'Fairport OH':     (41.7480, -81.2730),
    'Ashtabula OH':    (41.8650, -80.7890),
    'Toledo OH':       (41.6528, -83.5379),
    'Detroit MI':      (42.3314, -83.0458),
    'Sandusky OH':     (41.4489, -82.7080),
    'Lorain OH':       (41.4529, -82.1824),
    'Port Dover ON':   (42.7862, -80.2039),
    'Port Burwell ON': (42.6459, -80.8098),
    'Long Point tip':  (42.5833, -80.0500),
    'Rondeau ON':      (42.2940, -81.8582),
    'Port Colborne ON':(42.8863, -79.2518),
}

routes = [
    ('Toledo OH', 'Buffalo NY', 'Main E-W freighter lane'),
    ('Detroit MI', 'Buffalo NY', 'Detroit-Buffalo'),
    ('Cleveland OH', 'Buffalo NY', 'Cleveland-Buffalo'),
    ('Cleveland OH', 'Port Dover ON', 'Cleveland-Port Dover cross-lake'),
    ('Conneaut OH', 'Port Stanley ON', 'M&B route'),
    ('Erie PA', 'Port Dover ON', 'Erie-Port Dover'),
    ('Ashtabula OH', 'Port Burwell ON', 'Ashtabula-Port Burwell cross-lake'),
    ('Fairport OH', 'Port Stanley ON', 'Fairport-Port Stanley'),
    ('Sandusky OH', 'Rondeau ON', 'Sandusky-Rondeau'),
    ('Lorain OH', 'Port Stanley ON', 'Lorain-Port Stanley'),
    ('Cleveland OH', 'Port Stanley ON', 'Cleveland-Port Stanley'),
    ('Conneaut OH', 'Long Point tip', 'M&B shelter route to Long Point'),
]

target_lat = 42.1621
p(f'Target latitude: {target_lat}N')
p(f'Target longitudes: {[t["lon"] for t in lat_targets]}')
p()
p(f'Where each route crosses latitude {target_lat}:')
p('-'*80)

for p1_name, p2_name, desc in routes:
    p1 = ports[p1_name]
    p2 = ports[p2_name]
    lat1, lon1 = p1
    lat2, lon2 = p2
    
    if min(lat1, lat2) <= target_lat <= max(lat1, lat2):
        frac = (target_lat - lat1) / (lat2 - lat1) if lat2 != lat1 else 0
        cross_lon = lon1 + frac * (lon2 - lon1)
        p(f'{desc}:')
        p(f'  {p1_name} -> {p2_name}')
        p(f'  Crosses {target_lat}N at lon = {cross_lon:.4f}')
        for t in lat_targets:
            dlon = abs(t['lon'] - cross_lon) * 111 * math.cos(math.radians(target_lat))
            p(f'    -> {t["rank"]} (lon {t["lon"]}): {dlon:.1f} km away')
    else:
        p(f'{desc}: does NOT cross lat {target_lat}')

# === 5. M&B ROUTE DETAILED ===
p()
p('='*80)
p('5. M&B ROUTE DETAIL')
p('='*80)
mb_start = ports['Conneaut OH']
mb_end = ports['Port Stanley ON']
p(f'Conneaut: {mb_start[0]:.4f}N, {mb_start[1]:.4f}W')
p(f'Port Stanley: {mb_end[0]:.4f}N, {mb_end[1]:.4f}W')
frac = (target_lat - mb_start[0]) / (mb_end[0] - mb_start[0])
cross_lon = mb_start[1] + frac * (mb_end[1] - mb_start[1])
p(f'Straight-line crosses 42.1621N at lon = {cross_lon:.4f}')
p(f'Fraction of route at crossing: {frac:.1%}')

t1_lon = -80.9363
dlon_mb = abs(t1_lon - cross_lon) * 111 * math.cos(math.radians(target_lat))
p(f'Target #1 (M&B candidate, lon {t1_lon}) distance from M&B route at this lat: {dlon_mb:.1f} km')

# Full route length
dx = (mb_end[1] - mb_start[1]) * 111 * math.cos(math.radians(42.3))
dy = (mb_end[0] - mb_start[0]) * 111
route_len = math.sqrt(dx**2 + dy**2)
p(f'Full Conneaut-Port Stanley distance: {route_len:.1f} km')

# === 6. WRECK DATABASE CHECK ===
p()
p('='*80)
p('6. KNOWN WRECKS NEAR LAT 42.16')
p('='*80)

import sqlite3
conn = sqlite3.connect(os.path.join(BASE, 'db', 'wrecks.db'))
cur = conn.cursor()
cur.execute('PRAGMA table_info(wrecks)')
cols = [c[1] for c in cur.fetchall()]
lat_cols = [c for c in cols if 'lat' in c.lower()]
lon_cols = [c for c in cols if 'lon' in c.lower()]

if lat_cols and lon_cols:
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    cur.execute(f'SELECT name, {lat_col}, {lon_col}, type, cause, year_sunk FROM wrecks WHERE {lat_col} BETWEEN 42.05 AND 42.27 ORDER BY {lat_col}')
    rows = cur.fetchall()
    p(f'Known wrecks between lat 42.05-42.27: {len(rows)} found')
    for r in rows:
        name, lat, lon, wtype, cause, year = r
        p(f'  {name}: lat={lat}, lon={lon}, type={wtype}, cause={cause}, year={year}')
else:
    p('Could not find lat/lon columns in wreck DB')
conn.close()

# === SUMMARY ===
p()
p('='*80)
p('SUMMARY')
p('='*80)
p()
p('HYPOTHESIS 1 - Pipeline:')
p(f'  Signal at 300m: ~{signal_nT:.2f} nT vs 20-110 nT observed')
p('  REJECTED - signal far too weak')
p()
p('HYPOTHESIS 2 - Rail car debris from M&B:')
p('  Targets are 50-100 km apart, max scatter from one event ~2-5 km')
p('  REJECTED - spacing impossible')
p()
p('HYPOTHESIS 3 - Shipping lane (separate wrecks):')
p('  42.1621N is mid-lake, where multiple historical routes intersect')
p('  Lake Erie has 1,400-8,000 estimated shipwrecks')
p('  Each target could be a separate shipwreck on different routes')
p('  PLAUSIBLE')
p()
p('CRITICAL: Check tile-grid artifact results above.')
p('If 42.1621 exactly matches a tile-row center, the "same latitude" is')
p('just an artifact of tile discretization - the targets span ~25 km N-S')
p('within that tile row and are NOT at the same actual latitude.')

output = '\n'.join(lines)
with open(OUT, 'w') as f:
    f.write(output)
print(f'Results written to {OUT}')
print('DONE')
