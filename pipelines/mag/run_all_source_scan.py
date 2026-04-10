#!/usr/bin/env python3
"""
Run adaptive background scans on ALL available magnetic data sources for Lake Erie.

This goes beyond the original 4-tile scan (run_lake_scans.py) by processing
every unique data source at its best available resolution, then merging
and deduplicating candidates across sources.

Sources processed (in priority order by resolution):
  1. USGS_US (1,270m res) — highest resolution, US aeromagnetic
  2. USGS_NA (2,111m res) — North American aeromagnetic
  3. EMAG2 subsets (3,700m) — satellite-derived
  4. NRCan surveys (3,700m) — Canadian geological surveys
  5. Great Lakes composites (3,700m)

Deduplication: candidates within 2km of each other across sources are merged,
keeping the highest-scoring detection.

Usage:
    python scripts/run_all_source_scan.py
    python scripts/run_all_source_scan.py --output adaptive_bg_erie_all_sources
    python scripts/run_all_source_scan.py --dedup-radius 3000  # meters
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.adaptive_background_scan import _extract_candidates, _write_kml, Candidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("all_source_scan")

# ── Lake Erie config ─────────────────────────────────────────────────────────

ERIE_BBOX = (-83.6, 41.3, -78.8, 42.9)

SCAN_PARAMS = {
    "window_yards": 1000.0,
    "z_thresh": 0.5,
    "edge_z_thresh": 0.5,
    "min_pixels": 3,
    "max_pixels": 800,
}

# Best tile per unique data source, ordered by resolution
# Each is (source_name, filename_pattern) — we find the best Erie subset
ERIE_SOURCES = [
    # Highest resolution first
    ("USGS_US",      "usgs_usmag_83_6000_41_3000__78_8000_42_9000.tif"),
    ("USGS_NA",      "usgs_namag_83_6000_41_3000__78_8000_42_9000.tif"),
    # EMAG2 subsets — focus on Erie extent
    ("EMAG2",        "local_EMAG2_bessemer_erie_subset_83_6000_41_3000__78_8000_42_9000.tif"),
    ("EMAG2_84",     "local_EMAG2_bessemer_erie_subset_84_0000_41_0000__78_0000_43_0000.tif"),
    # NRCan surveys
    ("NRCan_StClair","local_gsc_stclair_nrcan_83_6000_41_3000__78_8000_42_9000.tif"),
    ("NRCan_Huron",  "local_gsc_huron_nrcan_84_0000_41_0000__78_0000_43_0000.tif"),
    ("NRCan_OH",     "local_nrcan_OH_4039B_83_6000_41_3000__78_8000_42_9000.tif"),
    # Composites
    ("GL_mag",       "local_great_lakes_magnetic_83_6000_41_3000__78_8000_42_9000.tif"),
    ("NRCan_pts",    "local_canadian_magnetic_points_83_6000_41_3000__78_8000_42_9000.tif"),
]


# ── Haversine for dedup ──────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    import math
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Spatial dedup ────────────────────────────────────────────────────────────

def deduplicate_candidates(
    all_candidates: list[tuple[str, Candidate]],
    radius_m: float = 2000.0,
) -> list[tuple[str, Candidate]]:
    """Merge candidates within radius_m, keeping highest score."""
    # Sort by score descending — highest-scoring candidates win
    all_candidates.sort(key=lambda x: x[1].score, reverse=True)

    merged: list[tuple[str, Candidate]] = []
    used = set()

    for i, (src_i, cand_i) in enumerate(all_candidates):
        if i in used:
            continue
        # This candidate survives — mark all neighbors as duplicates
        merged.append((src_i, cand_i))
        for j, (src_j, cand_j) in enumerate(all_candidates):
            if j <= i or j in used:
                continue
            d = haversine_m(cand_i.center_lat, cand_i.center_lon,
                           cand_j.center_lat, cand_j.center_lon)
            if d < radius_m:
                used.add(j)
                # Track which sources contributed
                # (we could store this, but for now just count)

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all_source_scan(
    output_dir: str = "adaptive_bg_erie_all_sources",
    dedup_radius_m: float = 2000.0,
    top_n: int = 500,
):
    grids_dir = REPO / "magnetic_data" / "grids"
    out = REPO / output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_candidates: list[tuple[str, Candidate]] = []
    source_stats = {}

    for source_name, filename in ERIE_SOURCES:
        tif_path = grids_dir / filename
        if not tif_path.exists():
            log.warning("MISSING: %s  (%s)", source_name, filename)
            continue

        log.info("Scanning %s: %s", source_name, filename)
        t0 = time.time()
        try:
            candidates = _extract_candidates(
                tif_path,
                **SCAN_PARAMS,
            )
        except Exception as e:
            log.error("FAILED %s: %s", source_name, e)
            source_stats[source_name] = {"candidates": 0, "error": str(e)}
            continue

        elapsed = time.time() - t0
        log.info("  %s: %d candidates in %.1fs", source_name, len(candidates), elapsed)

        for c in candidates:
            # Tag the source in the grid name
            c.source_grid = f"{source_name}:{c.source_grid}"
            all_candidates.append((source_name, c))

        source_stats[source_name] = {
            "candidates": len(candidates),
            "elapsed_s": round(elapsed, 1),
            "file": filename,
        }

    log.info("\nTotal raw candidates across all sources: %d", len(all_candidates))

    # Deduplicate
    merged = deduplicate_candidates(all_candidates, radius_m=dedup_radius_m)
    log.info("After spatial dedup (%dm radius): %d unique candidates", int(dedup_radius_m), len(merged))

    # Cap at top_n
    merged = merged[:top_n]
    log.info("Top %d candidates retained", len(merged))

    # Re-number label IDs
    for i, (src, cand) in enumerate(merged, start=1):
        cand.label_id = i

    # ── Write outputs ──
    candidates_only = [c for _, c in merged]

    # JSON
    json_path = out / "adaptive_candidates.json"
    json_path.write_text(
        json.dumps([asdict(c) for c in candidates_only], indent=2),
        encoding="utf-8",
    )

    # CSV
    csv_path = out / "adaptive_candidates.csv"
    if candidates_only:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(candidates_only[0]).keys()))
            w.writeheader()
            for c in candidates_only:
                w.writerow(asdict(c))

    # KML
    kml_path = out / "adaptive_candidates.kml"
    _write_kml(candidates_only, kml_path)

    # Source provenance
    source_map = {}
    for src, cand in merged:
        if src not in source_map:
            source_map[src] = 0
        source_map[src] += 1

    # Summary
    print("\n" + "=" * 70)
    print("ALL-SOURCE LAKE ERIE SCAN SUMMARY")
    print("=" * 70)

    print(f"\n  Sources scanned:     {len(source_stats)}")
    print(f"  Raw candidates:      {len(all_candidates)}")
    print(f"  After dedup:         {len(merged)}")
    print(f"  Dedup radius:        {int(dedup_radius_m)}m")

    print(f"\n  Per-source results:")
    for src, stats in source_stats.items():
        n = stats["candidates"]
        err = stats.get("error", "")
        surv = source_map.get(src, 0)
        if err:
            print(f"    {src:<20} ERROR: {err}")
        else:
            print(f"    {src:<20} {n:4d} raw -> {surv:4d} survived dedup  ({stats['elapsed_s']:.1f}s)")

    print(f"\n  Outputs:")
    print(f"    JSON: {json_path}")
    print(f"    CSV:  {csv_path}")
    print(f"    KML:  {kml_path}")

    # Save summary JSON
    summary = {
        "total_raw": len(all_candidates),
        "total_deduped": len(merged),
        "dedup_radius_m": dedup_radius_m,
        "source_stats": source_stats,
        "source_survivors": source_map,
    }
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Multi-source Erie aeromagnetic scan")
    parser.add_argument("--output", "-o", default="adaptive_bg_erie_all_sources",
                        help="Output directory name")
    parser.add_argument("--dedup-radius", type=float, default=2000.0,
                        help="Deduplication radius in meters (default: 2000)")
    parser.add_argument("--top-n", type=int, default=500,
                        help="Max candidates to keep (default: 500)")
    args = parser.parse_args()

    run_all_source_scan(
        output_dir=args.output,
        dedup_radius_m=args.dedup_radius,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
