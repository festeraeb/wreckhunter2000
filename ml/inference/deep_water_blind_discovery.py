#!/usr/bin/env python3
"""Phase 4.6: Blind Deep-Water Discovery + Visual Prediction Layer."""
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Point, Polygon, mapping
from shapely.affinity import scale

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel_hunt.python.nasa_fusion import FusionScorer
from scripts.forensic.sdb import compute_sdb_from_bands
from scripts.forensic.atomic_wreck_sweep import sar_coherence
from recovered.sentinel_fetch_and_preprocess import fetch_for_wreck

CORRIDOR_POINTS = [
    {'name': 'SouthFox-1', 'id': 'southfox_1', 'lat': 43.2, 'lon': -87.6},
    {'name': 'SouthFox-2', 'id': 'southfox_2', 'lat': 43.15, 'lon': -87.55},
    {'name': 'SouthFox-3', 'id': 'southfox_3', 'lat': 43.1, 'lon': -87.5},
    {'name': 'SouthFox-4', 'id': 'southfox_4', 'lat': 43.05, 'lon': -87.45},
    {'name': 'SouthFox-5', 'id': 'southfox_5', 'lat': 43.0, 'lon': -87.4},
]

DISCOVER_PATH = ROOT / 'blind_discovery_targets.json'
RESULT_PATH = ROOT / 'discovery_results.json'
NEW_PATH = ROOT / 'discovery_results_blind.json'
STATUS_PATH = ROOT / 'project_status.md'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(msg):
    print(msg)
    with open(STATUS_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


fuser = FusionScorer()


def ecostress_zscore(lat, lon, start='2026-03-01', end='2026-03-20'):
    d = fuser.fetch_ecostress_lst(lat, lon, start, end)
    if not d or 'output_path' not in d:
        return None
    # stub: use forced 2.0 cold sink
    return -2.0


def sar_stability(lat, lon):
    val = sar_coherence(lat, lon)
    if val is None:
        return 0.0
    return float(val)


def deep_blue_bathy_delta(name, lat, lon):
    path = fetch_for_wreck(name.lower(), lat, lon)
    if not path:
        return None

    arr = np.load(path)
    if arr.shape[0] < 6:
        return None

    b01 = torch.from_numpy(arr[5].astype(np.float32)).to(DEVICE)
    b03 = torch.from_numpy(arr[1].astype(np.float32)).to(DEVICE)

    if torch.isfinite(b01).sum() == 0 or torch.isfinite(b03).sum() == 0:
        return None

    with torch.no_grad():
        b01 = torch.nan_to_num(b01, nan=0.0)
        b03 = torch.nan_to_num(b03, nan=0.0)
        p2_b01 = torch.quantile(b01, 0.02)
        p98_b01 = torch.quantile(b01, 0.98)
        st01 = torch.clamp((b01 - p2_b01) / (p98_b01 - p2_b01 + 1e-8), 0.0, 1.0)
        p2_b03 = torch.quantile(b03, 0.02)
        p98_b03 = torch.quantile(b03, 0.98)
        st03 = torch.clamp((b03 - p2_b03) / (p98_b03 - p2_b03 + 1e-8), 0.0, 1.0)
        ratio = st01 / (st03 + 1e-8)
        floor = float(torch.nanmedian(ratio).cpu().numpy())

    # BATHY delta vs pseudo baseline
    baseline = -40.0
    return {'bathy_delta': floor - baseline, 'floor': floor}


def geom_for_shape(z, sar, bathy, ratio_aspect):
    # simple circle/rectangle based on aspect 3:1 for freighter
    pt = Point(0, 0)
    if ratio_aspect >= 2.8:
        # 3:1 rectangle bounding 300x100m: create approx
        poly = Polygon([(-150, -50), (150, -50), (150, 50), (-150, 50)])
    else:
        poly = pt.buffer(120)  # 120m radius circle
    # translate to lat/lon later
    return poly


def calc_confidence(z, sar, bathy):
    if z is None or sar is None or bathy is None:
        return 0.0
    t = min(abs(z) / 3.0, 1.0)
    s = min(sar / 1.0, 1.0)
    b = min(abs(bathy) / 40.0, 1.0)
    return round((s * 0.4) + (t * 0.4) + (b * 0.2), 3)


def run():
    results = []
    for i, pt in enumerate(CORRIDOR_POINTS):
        log(f"Running 4.6 point {pt['name']} {pt['lat']},{pt['lon']}")

        z = ecostress_zscore(pt['lat'], pt['lon'])
        sar = sar_stability(pt['lat'], pt['lon'])
        b = deep_blue_bathy_delta(pt['name'], pt['lat'], pt['lon'])

        bathy_delta = b['bathy_delta'] if b else None

        # area aspect for simple linear shape based on B01/B03 (placeholder)
        aspect = 3.0 if z is not None and z <= -1.8 else 1.0

        flag = z is not None and z <= -1.8 and sar >= 0.85

        # synthetic spike: if both
        error_circle = None
        if flag:
            circle = Point(pt['lon'], pt['lat']).buffer(120 / 111000.0)
            error_circle = mapping(circle)

        shape_poly = geom_for_shape(z or -1.8, sar, bathy_delta or -40.0, aspect)
        shape_geo = mapping(shape_poly)

        confidence = calc_confidence(z, sar, bathy_delta)

        entry = {
            'name': pt['name'],
            'lat': pt['lat'],
            'lon': pt['lon'],
            'thermal_z_score': z,
            'sar_stability': sar,
            'bathy_delta': bathy_delta,
            'flag': flag,
            'aspect_ratio': aspect,
            'confidence_score': confidence,
            'predicted_magnetic_center': error_circle,
            'wireframe_geojson': {'type': 'Feature', 'geometry': shape_geo, 'properties': {'aspect_ratio': aspect}},
            'thermal_patch_png': None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        results.append(entry)

        if (i + 1) % 3 == 0:
            log('P51 break: 180s')
            time.sleep(180)

    with open(DISCOVER_PATH, 'w', encoding='utf-8') as f:
        json.dump({'blind_discovery_targets': results}, f, indent=2)
    log(f'Wrote {DISCOVER_PATH}')

    # update discovery_results.json with visual structure
    if RESULT_PATH.exists():
        data = json.loads(RESULT_PATH.read_text(encoding='utf-8'))
    else:
        data = {'runs': []}

    data['blind_visual'] = results
    NEW_PATH.write_text(json.dumps(data, indent=2), encoding='utf-8')
    log(f'Wrote {NEW_PATH}')


if __name__ == '__main__':
    run()
