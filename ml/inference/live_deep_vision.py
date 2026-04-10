#!/usr/bin/env python3
"""Live Deep Vision Execution for Cedarville and Atlantic targets."""
from pathlib import Path
import os
import sys
import json
import numpy as np
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load fetcher explicitly to avoid _msi import issue
fetcher_path = os.path.join(str(ROOT), 'recovered', 'sentinel_fetch_and_preprocess.py')
if not os.path.exists(fetcher_path):
    raise FileNotFoundError(fetcher_path)

spec = importlib.util.spec_from_file_location('recovered_sentinel_fetch_and_preprocess', fetcher_path)
fetcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher_module)
fetch_for_wreck = fetcher_module.fetch_for_wreck

from scripts.forensic.sdb import (
    compute_sdb_from_bands,
    compute_aerosol_squeeze,
    compute_huron_fog_cutter,
    detect_vertical_pulse,
    detect_sdb_deviation,
)
from sentinel_hunt.python.nasa_fusion import FusionScorer


def get_ecostress_zscore(lat, lon, start='2026-03-01', end='2026-03-20'):
    scorer = FusionScorer()  # uses Earthdata token if available
    data = scorer.fetch_ecostress_lst(lat, lon, start, end)
    if not data or 'output_path' not in data:
        return None
    # TODO: actual raster-derived zscore from file
    return 0.0


def run_target(name, lat, lon, baseline_depth=40.0):
    print(f"\n=== Target: {name} ({lat},{lon}) ===")
    out = fetch_for_wreck(name.replace(' ', '_').lower(), lat, lon)
    if out is None:
        return {
            'name': name,
            'lat': lat,
            'lon': lon,
            'status': 'fetch_fail',
        }

    arr = np.load(out)
    # Standard bands: arr[2]=B02, arr[1]=B03, arr[0]=B04, arr[4]=B05, arr[5]=B01
    b03 = arr[1].astype(np.float32)
    b01 = arr[5].astype(np.float32) if arr.shape[0] > 5 else np.zeros_like(b03)
    b04 = arr[0].astype(np.float32)

    sdb = compute_sdb_from_bands(arr[2].astype(np.float32), b03)
    cy, cx = sdb.shape[0] // 2, sdb.shape[1] // 2
    window = sdb[max(0, cy - 40):cy + 40, max(0, cx - 80):cx + 80]
    sdb_median = float(np.nanmedian(window))
    sdb_deviation_flag = detect_sdb_deviation(window, baseline_depth=baseline_depth, threshold_m=3.0)

    aerosol_ratio = compute_aerosol_squeeze(b01, b03)
    aerosol_med = float(np.nanmedian(aerosol_ratio))

    fog = compute_huron_fog_cutter(b01, b04)
    fog_med = float(np.nanmedian(fog))
    pulse = detect_vertical_pulse(fog)

    ecostress_z = get_ecostress_zscore(lat, lon)

    return {
        'name': name,
        'lat': lat,
        'lon': lon,
        'patch': str(out),
        'sdb_median_m': sdb_median,
        'sdb_median_ft': sdb_median * 3.28084,
        'sdb_deviation_flag': sdb_deviation_flag,
        'aerosol_ratio_median': aerosol_med,
        'fog_median': fog_med,
        'pulse': pulse,
        'ecostress_zscore': ecostress_z,
    }


if __name__ == '__main__':
    targets = [
        ('Cedarville', 45.7872, -84.6708, 30.0),
        ('Atlantic', 42.5103, -80.0847, 50.0),
    ]

    results = []
    for name, lat, lon, baseline in targets:
        res = run_target(name, lat, lon, baseline_depth=baseline)
        results.append(res)

    # Postcondition: Deep Blue vs Green spike for Atlantic
    atl = next((r for r in results if r['name'] == 'Atlantic'), None)
    flag = False
    if atl and atl.get('aerosol_ratio_median') is not None:
        # spiking by >2% over deep-water baseline, i.e., ratio >1.02
        if atl['aerosol_ratio_median'] > 0.02:
            flag = True

    # Bathy delta approximation to GEBCO: assume reference from known values
    # (Cedarville baseline 30m, Atlantic baseline 50m deep)
    for r in results:
        if r.get('sdb_median_m') is not None:
            r['bathy_delta_m'] = float(r['sdb_median_m'] - (30.0 if r['name'] == 'Cedarville' else 50.0))
        else:
            r['bathy_delta_m'] = None

    print('\n=== Results ===')
    print(json.dumps(results, indent=2))
    print('\nDeep Blue structural reflection flag:', flag)

    # write status
    status_path = Path(ROOT) / 'project_status.md'
    with open(status_path, 'a', encoding='utf-8') as f:
        f.write('\n## Live Deep Vision results\n')
        f.write(str(results).replace(',', ',\n'))
        f.write('\nDeepBlueReflection='+str(flag)+'\n')

    print('Updated project_status.md')
