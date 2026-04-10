"""Run full mag pipeline + adaptive scan for Lake Huron and Lake Erie.

Usage: python scripts/run_lake_scans.py [--lakes huron,erie]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lake_scan")

LAKES = {
    "huron": {
        "bbox": (-84.5, 43.0, -79.5, 46.5),
        "name": "Lake Huron",
        "pipeline_out": "mag_pipeline_output_huron_v5",
        "adaptive_out": "adaptive_bg_huron_full_1000yd",
        "window_yards": 1000.0,
        "z_thresh": 0.5,
        "edge_z_thresh": 0.5,
        "top_n": 300,
    },
    "erie": {
        "bbox": (-83.6, 41.3, -78.8, 42.9),
        "name": "Lake Erie",
        "pipeline_out": "mag_pipeline_output_erie_v1",
        "adaptive_out": "adaptive_bg_erie_1000yd",
        "window_yards": 1000.0,
        "z_thresh": 0.5,
        "edge_z_thresh": 0.5,
        "top_n": 300,
    },
}


ANOMALY_SOURCES = [
    "local_great_lakes_magnetic_greatlakes.tif",
    "local_canadian_magnetic_points_greatlakes.tif",
    "usgs_namag_greatlakes.tif",
    "usgs_usmag_greatlakes.tif",
]


def clip_tif_to_bbox(src_tif: Path, bbox: tuple, dest: Path) -> Path:
    """Clip a GeoTIFF to bbox, keeping all valid cells."""
    import rasterio
    from rasterio.windows import from_bounds
    lonmin, latmin, lonmax, latmax = bbox
    with rasterio.open(str(src_tif)) as src:
        window = from_bounds(lonmin, latmin, lonmax, latmax, src.transform)
        arr = src.read(1, window=window)
        meta = src.meta.copy()
        meta.update({
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": src.window_transform(window),
        })
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(dest), "w", **meta) as dst:
            dst.write(arr, 1)
    return dest


def run_pipeline_for_lake(lake_cfg: dict) -> list[Path]:
    """Clip greatlakes anomaly TIFs to lake bbox — fast and high-coverage."""
    from pathlib import Path as _Path
    grids_dir = REPO / "magnetic_data/grids"
    bbox = lake_cfg["bbox"]
    lonmin, latmin, lonmax, latmax = bbox
    tag = f"{abs(lonmin):.4f}_{latmin:.4f}__{abs(lonmax):.4f}_{latmax:.4f}".replace(".", "_")

    produced = []
    for src_name in ANOMALY_SOURCES:
        src = grids_dir / src_name
        if not src.exists():
            log.warning("Source TIF not found: %s", src_name)
            continue
        stem = src_name.replace("_greatlakes.tif", "")
        dest = grids_dir / f"{stem}_{tag}.tif"
        if dest.exists() and dest.stat().st_size > 1000:
            log.info("Clip already exists: %s", dest.name)
        else:
            log.info("Clipping %s -> %s", src_name, dest.name)
            clip_tif_to_bbox(src, bbox, dest)
        produced.append(dest)

    log.info("%s: %d clipped TIFs ready", lake_cfg["name"], len(produced))
    return produced


def run_adaptive_scan(lake_cfg: dict, tifs: list[Path]) -> Path:
    """Run adaptive scan on the given TIFs by calling the module directly."""
    if not tifs:
        log.warning("No TIFs to scan for %s", lake_cfg["name"])
        return None

    from scripts.adaptive_background_scan import _extract_candidates, _write_kml
    import csv as _csv
    import json as _json
    from dataclasses import asdict

    out_dir = REPO / lake_cfg["adaptive_out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Adaptive scan: %s  (%d TIFs) ===", lake_cfg["name"], len(tifs))

    candidates = []
    for tif in tifs:
        log.info("  Scanning %s", tif.name)
        cands = _extract_candidates(
            tif,
            window_yards=lake_cfg["window_yards"],
            z_thresh=lake_cfg["z_thresh"],
            edge_z_thresh=lake_cfg["edge_z_thresh"],
            min_pixels=3,
            max_pixels=800,
        )
        log.info("    -> %d candidates", len(cands))
        candidates.extend(cands)

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:lake_cfg["top_n"]]
    log.info("Top %d candidates total for %s", len(candidates), lake_cfg["name"])

    if not candidates:
        log.warning("No candidates found for %s", lake_cfg["name"])
        return None

    json_path = out_dir / "adaptive_candidates.json"
    csv_path = out_dir / "adaptive_candidates.csv"
    kml_path = out_dir / "adaptive_candidates.kml"

    json_path.write_text(_json.dumps([asdict(c) for c in candidates], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(asdict(candidates[0]).keys()))
        w.writeheader()
        for c in candidates:
            w.writerow(asdict(c))
    _write_kml(candidates, kml_path)

    log.info("KML -> %s", kml_path)
    return kml_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lakes", default="huron,erie", help="Comma-separated lake names")
    args = p.parse_args()

    chosen = [k.strip() for k in args.lakes.split(",")]
    results = {}

    for key in chosen:
        cfg = LAKES.get(key)
        if cfg is None:
            log.warning("Unknown lake: %s  (choices: %s)", key, list(LAKES.keys()))
            continue

        t0 = time.time()
        tifs = run_pipeline_for_lake(cfg)
        kml = run_adaptive_scan(cfg, tifs)
        elapsed = time.time() - t0

        results[key] = {
            "name": cfg["name"],
            "tifs_produced": len(tifs),
            "kml": str(kml) if kml else None,
            "elapsed_s": round(elapsed, 1),
        }

    print("\n" + "=" * 60)
    print("LAKE SCAN SUMMARY")
    print("=" * 60)
    for key, r in results.items():
        print(f"\n{r['name']}:")
        print(f"  TIFs: {r['tifs_produced']}")
        print(f"  KML:  {r['kml']}")
        print(f"  Time: {r['elapsed_s']}s")

    # Write combined summary JSON
    out = REPO / "lake_scan_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Summary written to %s", out)


if __name__ == "__main__":
    main()
