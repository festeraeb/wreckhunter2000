#!/usr/bin/env python3
"""
Lake Huron Magnetic Anomaly - Flat Water-Only Scan
===================================================
Reads all available mag GeoTIFFs, masks to water-only pixels over
Lake Huron, and ranks anomalies by z-score.  Quick scan while the
erie_remote pipeline downloads ICESat-2 data.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from geo_filter_candidates import classify_huron

# Lake Huron bbox
BBOX = (-84.5, 43.0, -79.5, 46.5)

GRIDS = [
    ("USGS NAmag",          "magnetic_data/grids/usgs_namag_greatlakes.tif"),
    ("USGS USmag",          "magnetic_data/grids/usgs_usmag_greatlakes.tif"),
    ("Canadian Points",     "magnetic_data/grids/local_canadian_magnetic_points_84_5000_43_0000__79_5000_46_5000.tif"),
    ("Great Lakes Merged",  "magnetic_data/grids/local_great_lakes_magnetic_84_5000_43_0000__79_5000_46_5000.tif"),
    ("EMAG2 Satellite",     "magnetic_data/grids/local_EMAG2_huron_corridor_subset_84_5000_43_0000__79_5000_46_5000.tif"),
    ("NRCan GSC Huron",     "magnetic_data/grids/local_gsc_huron_nrcan_84_5000_43_0000__79_5000_46_5000.tif"),
]


def main():
    print("=" * 70)
    print("LAKE HURON MAGNETIC ANOMALY — FLAT WATER-ONLY SCAN")
    print("=" * 70)
    print(f"  Bbox: {BBOX}")
    print()

    all_candidates = []

    for label, rel_path in GRIDS:
        tif_path = REPO / rel_path
        if not tif_path.exists():
            print(f"SKIP {label}: file not found")
            continue

        with rasterio.open(str(tif_path)) as ds:
            try:
                window = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], ds.transform)
                arr = ds.read(1, window=window)
                transform = ds.window_transform(window)
            except Exception:
                arr = ds.read(1)
                transform = ds.transform
            nodata = ds.nodata

        rows, cols = arr.shape
        ys, xs = np.mgrid[0:rows, 0:cols]
        lons = transform[2] + xs * transform[0] + ys * transform[1]
        lats = transform[5] + xs * transform[3] + ys * transform[4]

        # Valid-data mask
        if nodata is not None:
            valid_mask = (arr != nodata) & np.isfinite(arr)
        else:
            valid_mask = np.isfinite(arr)

        # Water-only filter
        water_mask = np.zeros_like(valid_mask)
        for r in range(rows):
            for c in range(cols):
                if valid_mask[r, c]:
                    if classify_huron(float(lats[r, c]), float(lons[r, c])) == "LAKE":
                        water_mask[r, c] = True

        n_water = int(water_mask.sum())
        if n_water == 0:
            print(f"{label}: no water pixels in bbox\n")
            continue

        water_vals = arr[water_mask]
        water_lats = lats[water_mask]
        water_lons = lons[water_mask]

        # Total-field grids (NRCan GSC): compute residual from mean
        is_total = float(np.mean(water_vals)) > 50000
        if is_total:
            residual = water_vals - np.mean(water_vals)
            atype = "residual-from-mean"
        else:
            residual = water_vals.copy()
            atype = "anomaly-field"

        med = float(np.median(residual))
        std = float(np.std(residual))

        print(f"--- {label} ---")
        ftype = "total field" if is_total else "anomaly field"
        print(f"  Water pixels: {n_water:,}   type: {ftype}")
        print(f"  median={med:.1f} nT   std={std:.1f} nT   range=[{residual.min():.1f}, {residual.max():.1f}]")

        # Anomalies > +2 sigma
        pos_thresh = med + 2.0 * std
        neg_thresh = med - 2.0 * std
        n_pos = int((residual > pos_thresh).sum())
        n_neg = int((residual < neg_thresh).sum())
        print(f"  Anomalies >+2σ ({pos_thresh:.1f} nT): {n_pos}")
        print(f"  Anomalies <-2σ ({neg_thresh:.1f} nT): {n_neg}")

        # Collect top positive anomalies
        sort_idx = np.argsort(residual)[::-1]
        for rank, idx in enumerate(sort_idx[:25]):
            val = float(residual[idx])
            lat_v = float(water_lats[idx])
            lon_v = float(water_lons[idx])
            z = (val - med) / std if std > 0 else 0.0
            all_candidates.append({
                "source": label,
                "lat": round(lat_v, 4),
                "lon": round(lon_v, 4),
                "value_nT": round(val, 1),
                "z_score": round(z, 2),
                "type": atype,
            })
            if rank < 5:
                print(f"    #{rank+1}: ({lat_v:.4f}, {lon_v:.4f})  {val:+.1f} nT  z={z:.2f}")
        print()

    # Deduplicate by proximity
    if not all_candidates:
        print("No anomalies found.")
        return

    all_candidates.sort(key=lambda c: c["z_score"], reverse=True)
    deduped = []
    for c in all_candidates:
        dup = any(abs(c["lat"] - d["lat"]) < 0.05 and abs(c["lon"] - d["lon"]) < 0.05
                  for d in deduped)
        if not dup:
            deduped.append(c)

    print("=" * 70)
    print(f"TOP WATER ANOMALIES — {len(deduped)} unique locations (deduplicated)")
    print("=" * 70)
    for i, c in enumerate(deduped[:40]):
        print(f"  #{i+1:2d}  ({c['lat']:.4f}, {c['lon']:.4f})  {c['value_nT']:+8.1f} nT  "
              f"z={c['z_score']:+5.2f}  [{c['source']}]")

    out = REPO / "mag_huron_water_scan.json"
    out.write_text(json.dumps(deduped, indent=2))
    print(f"\nWrote {len(deduped)} candidates to {out}")


if __name__ == "__main__":
    main()
