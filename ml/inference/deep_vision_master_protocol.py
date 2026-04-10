#!/usr/bin/env python3
"""M2200 Forensic Deep Vision Master Protocol"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recovered.sentinel_fetch_and_preprocess import fetch_for_wreck
from scripts.forensic.sdb import (
    compute_sdb_from_bands,
    compute_aerosol_squeeze,
    compute_huron_fog_cutter,
    detect_vertical_pulse,
    detect_sdb_deviation,
)
from scripts.forensic.atomic_wreck_sweep import ecostress_zscore, sar_coherence

# Master target list
TARGETS = [
    {'name': 'Cedarville', 'id': 'cedarville', 'lat': 45.7872, 'lon': -84.6708, 'baseline': 30.0},
    {'name': 'Acme', 'id': 'acme', 'lat': 42.6100, 'lon': -79.4973, 'baseline': 35.0},
    {'name': 'Sandusky', 'id': 'sandusky', 'lat': 45.7993, 'lon': -84.8375, 'baseline': 25.0},
    {'name': 'Atlantic', 'id': 'atlantic', 'lat': 42.5103, 'lon': -80.0847, 'baseline': 50.0},
    {'name': 'Philadelphia', 'id': 'philadelphia', 'lat': 44.0687, 'lon': -82.7153, 'baseline': 40.0},
    {'name': 'M&B 2', 'id': 'm_and_b_2', 'lat': 43.0000, 'lon': -82.0000, 'baseline': 45.0},
]

DISCOVERY_PATH = ROOT / 'discovery_results.json'
STATUS_PATH = ROOT / 'project_status.md'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Deep Vision device:', DEVICE)


def update_status(msg):
    with open(STATUS_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def safe_array(arr, mask=None):
    a = arr.copy()
    if mask is not None:
        a = np.where(mask, a, np.nan)
    return a


def contrast_squeeze_cuda(arr):
    # arr: 2D float array
    x = torch.from_numpy(arr.astype(np.float32)).to(DEVICE)
    nvalid = torch.isfinite(x)
    if nvalid.sum() == 0:
        return None, None
    valid = x[nvalid]
    p2 = torch.quantile(valid, 0.02)
    p98 = torch.quantile(valid, 0.98)
    stretched = torch.clamp((x - p2) / (p98 - p2), 0.0, 1.0)
    return stretched.cpu().numpy(), {'p2': float(p2.item()), 'p98': float(p98.item()), 'mean': float(stretched[nvalid].mean().item()), 'std': float(stretched[nvalid].std().item())}


def asymmetry_index(arr):
    # center symmetry test; 180-degree rotational asymmetry
    if arr is None:
        return None
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    center = arr.shape[0] // 2
    x = arr
    x_rot = np.rot90(x, 2)
    diff = np.nanmean(np.abs(x - x_rot))
    base = np.nanmean(np.abs(x) + np.abs(x_rot)) / 2.0 + 1e-9
    return float(diff / base)


def wreck_dna_features(name, target, process_mode='masked'):
    lat, lon = target['lat'], target['lon']
    wreck_id = target.get('id', name.replace(' ', '_').lower()).replace('&', 'and').replace(' ', '_')
    out_path = fetch_for_wreck(wreck_id, lat, lon)
    if out_path is None:
        update_status(f"{name} - fetch failed")
        return None

    arr = np.load(out_path)
    if arr.shape[0] < 6:
        update_status(f"{name} - not enough bands")
        return None

    b04 = arr[0].astype(np.float32)
    b03 = arr[1].astype(np.float32)
    b02 = arr[2].astype(np.float32)
    b05 = arr[4].astype(np.float32)
    b01 = arr[5].astype(np.float32)

    if process_mode == 'masked':
        mask = (b01 > 0) & (b03 > 0)
    else:
        mask = None

    b01_u = safe_array(b01, mask)
    b03_u = safe_array(b03, mask)

    cs_b01, st_b01 = contrast_squeeze_cuda(b01_u)
    cs_b03, st_b03 = contrast_squeeze_cuda(b03_u)

    if cs_b01 is None or cs_b03 is None:
        return None

    b01_b03_ratio = np.divide(cs_b01, cs_b03, out=np.full_like(cs_b01, np.nan), where=np.isfinite(cs_b03) & (np.abs(cs_b03) > 1e-6))

    sdb = compute_sdb_from_bands(b02, b03)
    sdb_center = sdb[96:160, 88:168] if sdb.ndim == 2 else sdb
    sdb_median = float(np.nanmedian(sdb_center))
    sdb_delta = sdb_median - target['baseline']

    aerosol = compute_aerosol_squeeze(b01_u, b03_u)
    aerosol_med = float(np.nanmedian(aerosol)) if aerosol is not None else None

    storm = compute_huron_fog_cutter(b01_u, b04)
    pulse = detect_vertical_pulse(storm)

    ec_z = ecostress_zscore(lat, lon)
    sar_p = sar_coherence(lat, lon)

    asym_index = asymmetry_index(cs_b01)

    result = {
        'name': name,
        'mode': process_mode,
        'sdb_b01_b03_ratio': float(np.nanmean(b01_b03_ratio)),
        'thermal_z_score': ec_z,
        'sar_persistence': sar_p,
        'asymmetry_index': asym_index,
        'sdb_median': sdb_median,
        'sdb_delta': sdb_delta,
        'aerosol_med': aerosol_med,
        'pulse': bool(pulse),
    }

    # Save artifacts
    out_dir = ROOT / 'discovery' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{process_mode}_b01_cs.npy", cs_b01)
    np.save(out_dir / f"{process_mode}_b03_cs.npy", cs_b03)
    np.save(out_dir / f"{process_mode}_b01_b03_ratio.npy", b01_b03_ratio)

    update_status(f"{name} {process_mode} complete: asym={asym_index:.4f}")
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available, stop.')

    discovery = {'runs': []}
    target_order = ['Atlantic', 'M&B 2']

    for idx, name in enumerate(target_order):
        print('target loop entry', idx, name)
        target = next((t for t in TARGETS if t['name'] == name), None)
        if target is None:
            print('target missing in list', name)
            continue

        update_status(f"Begin target {name}")

        # Pass A: masked
        ra = wreck_dna_features(name, target, process_mode='masked')

        # Pass B: unmasked
        rb = wreck_dna_features(name, target, process_mode='unmasked')

        if ra and rb:
            censorship_delta = {
                'sdb_delta': rb['sdb_delta'] - ra['sdb_delta'],
                'ratio_diff': rb['sdb_b01_b03_ratio'] - ra['sdb_b01_b03_ratio'],
                'asym_diff': rb['asymmetry_index'] - ra['asymmetry_index'] if ra['asymmetry_index'] is not None and rb['asymmetry_index'] is not None else None,
            }
        else:
            censorship_delta = None

        record = {
            'target': name,
            'pass_a': ra,
            'pass_b': rb,
            'censorship_delta': censorship_delta,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        discovery['runs'].append(record)

        with open(DISCOVERY_PATH, 'w', encoding='utf-8') as f:
            json.dump(discovery, f, indent=2)

        update_status(f"Completed target {name} and saved to discovery results")

        if (idx + 1) % 3 == 0:
            update_status('P51 thermal break start (180s)')
            time.sleep(180)

    # training step (over night)
    update_status('Starting GPU training after sweep')
    print('starting training command')
    train_script = ROOT / 'scripts' / 'forensic' / 'train_wreck_classifier_gpu.py'
    import subprocess
    cmd = [sys.executable, str(train_script)]
    print('training cmd', cmd)
    subprocess.run(cmd, check=True)
    update_status('Training complete; model ready')


if __name__ == '__main__':
    main()
