#!/usr/bin/env python3
"""
i7 CPU-Only Specialized Scan — CESAROPS Pass 2 / 3 / 4

Dedicated script for the i7 node (10.0.0.56):
  - No GPU (Intel HD 4000 only, P1000 moved to Xeon)
  - Edge TPU served locally via tpu_server.py on :5001

Runs ONLY the CPU-native passes:
  PASS 2 — Hydrocarbon / oil-slick  (B11 SWIR dark + B04 Red cross-check)
  PASS 3 — Stumpf bathymetric       (B02 Blue / B03 Green log-ratio)
  PASS 4 — NauticUVs LoG blob       (B02 surface disturbance + B10 cold plume)

Skips Pass 1 (standard CuPy CUDA anomaly scan) — handled by M2200 laptop
and Xeon P1000 nodes.

Usage (local or via scripts/_i7_launch_cpu_passes.py):
    CESAROPS_DATA_DIR=/home/cesarops/downloads python i7_cpu_passes.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ── Import processing functions from main engine ──────────────────────────────
print("[INIT] Loading CESAROPS processing engine...", flush=True)
try:
    from lake_michigan_scan import (
        process_hydrocarbon_bands,
        compute_stumpf_pass,
        compute_nauticuvs_pass,
        create_kmz,
        KNOWN_WRECKS,
        KNOWN_WRECK_RADIUS_DEG,
        _flag_known_wreck,
    )
    print("[INIT] Engine loaded — CPU-only mode (no CUDA required)", flush=True)
except ImportError as e:
    print(f"[INIT] FATAL: Could not import lake_michigan_scan: {e}")
    sys.exit(1)


def main():
    print("=" * 80)
    print("i7 CPU SPECIALIZED SCAN — Pass 2 (HC) / Pass 3 (Stumpf) / Pass 4 (NauticUVs)")
    print("=" * 80)
    print()

    # ── TIF discovery (same logic as lake_michigan_scan.py main()) ────────────
    data_base = Path(os.environ.get('CESAROPS_DATA_DIR', Path(__file__).parent / 'data'))
    repo_root = Path(__file__).parent
    search_paths = [
        data_base,
        data_base / 'michigan',
        data_base / 'superior',
        data_base / 'huron',
        data_base / 'erie',
        data_base / 'ontario',
        data_base / 'straits',
        data_base / 'rossa_forensic_cache',
        data_base / 'sentinel_hunt_cache',
        repo_root / 'downloads' / 'michigan',
        repo_root / 'downloads' / 'superior',
        repo_root / 'downloads' / 'huron',
        repo_root / 'downloads' / 'erie',
        repo_root / 'downloads' / 'straits',
    ]

    tiffs = []
    for sp in search_paths:
        if sp.exists():
            tiffs.extend(sp.rglob("*.tif"))
    tiffs = sorted(set(tiffs))

    print(f"Found {len(tiffs)} TIFs across all lakes")
    print()

    all_detections = []

    # Straits bbox — constrains anomaly selection to water
    STRAITS_BBOX = [45.70, -84.90, 46.05, -84.10]

    # ── PASS 2: Hydrocarbon scan ───────────────────────────────────────────────
    b11_tiffs = [t for t in tiffs if
                 ('.B11.' in t.name.upper() or '.SWIR16.' in t.name.upper()) and
                 'FMASK' not in t.name.upper() and
                 '.SCL.' not in t.name.upper()]
    print(f"PASS 2 — Hydrocarbon / oil-slick scan ({len(b11_tiffs)} B11 SWIR scenes)")
    print("-" * 60)
    for b11_path in b11_tiffs:
        pname = b11_path.name.lower()
        if '.swir16.tif' in pname:
            b04_path = Path(str(b11_path).replace('.swir16.tif', '.red.tif'))
        else:
            b04_path = Path(str(b11_path).replace('.B11.tif', '.B04.tif'))
        try:
            hc_dets = process_hydrocarbon_bands(b11_path, b04_path)
            all_detections.extend(hc_dets)
        except ImportError:
            print("    [HC] scipy not available — pip install scipy")
        except Exception as e:
            print(f"    [HC] ERROR: {e}")

    # ── PASS 3: Stumpf bathymetric ────────────────────────────────────────────
    blue_tiffs = [t for t in tiffs if
                  ('.B02.' in t.name.upper() or '.BLUE.' in t.name.upper()) and
                  'FMASK' not in t.name.upper() and
                  '.SCL.' not in t.name.upper()]
    print()
    print(f"PASS 3 — Stumpf bathymetric shallow-anomaly scan ({len(blue_tiffs)} blue bands)")
    print("-" * 60)
    for blue_path in blue_tiffs:
        pname = blue_path.name.lower()
        if '.blue.tif' in pname:
            green_path = Path(str(blue_path).replace('.blue.tif', '.green.tif'))
        elif '.B02.tif' in blue_path.name:
            green_path = Path(str(blue_path).replace('.B02.tif', '.B03.tif'))
        else:
            green_path = Path(str(blue_path).replace('B02', 'B03').replace('blue', 'green'))
        try:
            st_dets = compute_stumpf_pass(blue_path, green_path, scan_bbox=STRAITS_BBOX)
            all_detections.extend(st_dets)
        except Exception as e:
            print(f"    [ST] ERROR: {e}")

    # ── PASS 4: NauticUVs LoG blob scan ───────────────────────────────────────
    try:
        from scipy.ndimage import gaussian_laplace  # noqa — availability check
        nuv_bands = (
            [t for t in tiffs if ('.B02.' in t.name.upper() or '.BLUE.' in t.name.upper())
             and 'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()] +
            [t for t in tiffs if ('B10' in t.name.upper() or 'THERMAL' in t.name.upper()
             or 'LWIR' in t.name.upper())
             and 'FMASK' not in t.name.upper() and '.SCL.' not in t.name.upper()]
        )
        print()
        print(f"PASS 4 — NauticUVs LoG blob scan ({len(nuv_bands)} bands: B02+B10)")
        print("-" * 60)
        for nuv_tif in nuv_bands:
            print(f"  {nuv_tif.name}")
            try:
                nuv_dets = compute_nauticuvs_pass(nuv_tif, scan_bbox=STRAITS_BBOX)
                all_detections.extend(nuv_dets)
            except Exception as e:
                print(f"    [NUV] ERROR: {e}")
    except ImportError:
        print()
        print("PASS 4 — NauticUVs SKIPPED (pip install scipy)")

    # ── Summary ───────────────────────────────────────────────────────────────
    wreck_hits  = [d for d in all_detections if d.get("known_wreck_hit")]
    hc_hits     = [d for d in all_detections if d.get("type") == "hydrocarbon"]
    st_hits     = [d for d in all_detections if d.get("type") == "stumpf_shallow"]
    nuv_hits    = [d for d in all_detections if d.get("type") == "nauticuvs_candidate"]

    print()
    print("=" * 80)
    print("SCAN SUMMARY (i7 CPU passes)")
    print("=" * 80)
    print(f"  Total detections           : {len(all_detections)}")
    print(f"  Known wreck hits           : {len(wreck_hits)}")
    print(f"  Hydrocarbon anomalies      : {len(hc_hits)}")
    print(f"  Stumpf shallow anomalies   : {len(st_hits)}")
    print(f"  NauticUVs LoG candidates   : {len(nuv_hits)}")

    if wreck_hits:
        print()
        print("  KNOWN WRECK HITS:")
        for d in wreck_hits:
            print(f"    {d['known_wreck_name']}  Z={d['zscore']:.2f}  ({d['lat']:.5f}, {d['lon']:.5f})")

    print("=" * 80)

    # ── Save outputs ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    json_file = output_dir / f"i7_cpu_passes_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "node": "i7",
            "passes": ["hydrocarbon", "stumpf", "nauticuvs"],
            "generated": timestamp,
            "total_detections": len(all_detections),
            "known_wreck_hits": len(wreck_hits),
            "hydrocarbon_anomalies": len(hc_hits),
            "stumpf_shallow": len(st_hits),
            "nauticuvs_candidates": len(nuv_hits),
            "detections": all_detections,
        }, f, indent=2)
    print(f"\n[OK] JSON saved: {json_file}")

    kmz_file = output_dir / f"i7_cpu_passes_{timestamp}.kmz"
    create_kmz(all_detections, kmz_file)
    print(f"[OK] KMZ saved: {kmz_file}")

    print()
    print("=" * 80)
    print("SCAN COMPLETE — Open KMZ in Google Earth Pro")
    print("Layers: Known Wreck Hits | HC | Stumpf | NauticUVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
