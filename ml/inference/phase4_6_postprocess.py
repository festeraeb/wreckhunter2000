#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
import os, sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recovered.sentinel_fetch_and_preprocess import fetch_for_wreck
from scripts.forensic.sdb import compute_aerosol_squeeze, compute_huron_fog_cutter

# 1) thermal cutoff
corridor_path = ROOT / 'corridor_discovery.json'
blind_path = ROOT / 'blind_discovery_targets.json'
final_path = ROOT / 'final_discovery.json'

with open(corridor_path, 'r', encoding='utf-8') as f:
    corridor = json.load(f)

thermal_cutoff = [p for p in corridor.get('corridor', []) if p.get('ecostress_zscore') is not None and p['ecostress_zscore'] <= -1.5]

# 2) numeric rescue for Fulton, Beaver
rescue_targets = [
    {'name': 'Fulton', 'lat': 42.467, 'lon': -87.09},
    {'name': 'Beaver', 'lat': 42.476, 'lon': -87.02},
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rescue_results = []
for t in rescue_targets:
    out_path = fetch_for_wreck(t['name'].lower(), t['lat'], t['lon'])
    if out_path is None:
        rescue_results.append({'name': t['name'], 'ratio': None})
        continue
    arr = np.load(out_path)
    try:
        b01 = torch.from_numpy(arr[5].astype(np.float32)).to(DEVICE)
        b03 = torch.from_numpy(arr[1].astype(np.float32)).to(DEVICE)
    except Exception:
        rescue_results.append({'name': t['name'], 'ratio': None})
        continue
    b01 = torch.nan_to_num(b01, nan=0.0, posinf=0.0, neginf=0.0)
    b03 = torch.nan_to_num(b03, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        p2_01 = torch.quantile(b01, 0.02)
        p98_01 = torch.quantile(b01, 0.98)
        s01 = torch.clamp((b01 - p2_01) / (p98_01 - p2_01 + 1e-8), 0.0, 1.0)
        p2_03 = torch.quantile(b03, 0.02)
        p98_03 = torch.quantile(b03, 0.98)
        s03 = torch.clamp((b03 - p2_03) / (p98_03 - p2_03 + 1e-8), 0.0, 1.0)
        ratio = (s01 / (s03 + 1e-8)).cpu().numpy()
    rescue_results.append({'name': t['name'], 'ratio_mean': float(np.nanmean(ratio)), 'ratio_fin': float(np.nanpercentile(ratio,98) - np.nanpercentile(ratio,2)), 'nan_count': int(np.isnan(ratio).sum())})

# 3) Symmetry visualizer on top-3 by confidence
with open(blind_path, 'r', encoding='utf-8') as f:
    blind = json.load(f)

cands = blind['blind_discovery_targets']
cands_sorted = sorted(cands, key=lambda x: x.get('confidence_score',0), reverse=True)[:3]

for c in cands_sorted:
    aspect = c.get('aspect_ratio',1.0)
    if aspect >= 2.8:
        c['classification'] = 'High-Confidence Wreck'
    else:
        c['classification'] = 'Infrastructure/Well'

# 4) Visual ghost JSON output
visual = []
for c in cands_sorted:
    circle = None
    if c.get('flag'):
        lat, lon = c['lat'], c['lon']
        circle = {'type':'Feature','geometry':{'type':'Polygon','coordinates':[[]]}, 'properties': {'radius_m':1200,'color':'cyan'}}
    else:
        circle = None
    visual.append({'name': c['name'], 'lat': c['lat'], 'lon': c['lon'], 'classification': c['classification'], 'confidence': c['confidence_score'], 'error_circle': circle})

out = {
    'thermal_cutoff': thermal_cutoff,
    'rescue_results': rescue_results,
    'top_symmetry_candidates': cands_sorted,
    'visual_ghost': visual,
}

with open(final_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print('Written', final_path)
print('Thermal cutoff count:', len(thermal_cutoff))
print('Rescue results:', rescue_results)
print('Top 3 candidates', [x['name'] for x in cands_sorted])
