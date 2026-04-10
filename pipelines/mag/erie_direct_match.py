#!/usr/bin/env python3
"""
Direct cross-reference of aero-mag candidates against known wrecks + OGSr wells.
No ML classifier — just haversine distance matching.

Loads ALL 17 raw candidates (13 scored lake-only + 4 dropped).
For each candidate: finds nearest known wreck and nearest well.
For each known wreck: finds nearest candidate.
Outputs: labeled matches table + coverage gap analysis.

Usage:
    python scripts/erie_direct_match.py
    python scripts/erie_direct_match.py --wells path/to/ogsr.csv
    python scripts/erie_direct_match.py --output erie_direct_match_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from erie_wellhead_discriminator import (
    CandidateMatch,
    KnownWreck,
    get_all_known_wrecks,
    haversine_m,
    load_candidates_csv,
    load_ogsr_wells,
    GROUND_TRUTH,
)
from erie_known_wrecks_db import build_database, WreckRecord

# ── Constants ────────────────────────────────────────────────────────────────

SCORED_CSV = "adaptive_bg_erie_1000yd/adaptive_candidates_scored.csv"
RAW_CSV = "adaptive_bg_erie_1000yd/adaptive_candidates.csv"
REPO = Path(__file__).resolve().parents[1]

# Steel-hulled vessel types — these produce the strongest magnetic anomalies
STEEL_TYPES = {"steamer", "propeller", "tug", "sandsucker", "freighter", "whaleback"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_all_candidates() -> list[dict]:
    """Load candidates from all available sources:
      1. Original scored + raw (13 scored + 4 dropped = 17)
      2. All-source scan (multi-source adaptive bg scan)

    Deduplicates by position (within 1km = same candidate).
    """
    scored = {}
    raw = {}

    if Path(SCORED_CSV).exists():
        for row in load_candidates_csv(SCORED_CSV):
            scored[row["label_id"]] = row

    if Path(RAW_CSV).exists():
        for row in load_candidates_csv(RAW_CSV):
            raw[row["label_id"]] = row

    # Merge: scored rows take priority, add dropped rows from raw
    merged = dict(scored)
    for lid, row in raw.items():
        if lid not in merged:
            row["_dropped"] = "true"
            merged[lid] = row

    candidates = list(merged.values())
    base_count = len(candidates)

    # Also load multi-source scan results
    all_src_csv = REPO / "adaptive_bg_erie_all_sources" / "adaptive_candidates.csv"
    if all_src_csv.exists():
        all_src_rows = load_candidates_csv(str(all_src_csv))
        # Deduplicate against existing candidates
        new_count = 0
        for row in all_src_rows:
            try:
                new_lat = float(row.get("center_lat", 0))
                new_lon = float(row.get("center_lon", 0))
            except (ValueError, TypeError):
                continue
            is_dup = False
            for existing in candidates:
                try:
                    ex_lat = float(existing.get("center_lat", 0))
                    ex_lon = float(existing.get("center_lon", 0))
                except (ValueError, TypeError):
                    continue
                if haversine_m(new_lat, new_lon, ex_lat, ex_lon) < 1000:
                    is_dup = True
                    break
            if not is_dup:
                # Re-label to avoid ID collisions
                row["label_id"] = str(base_count + new_count + 1)
                row["_source"] = "all_source_scan"
                candidates.append(row)
                new_count += 1
        if new_count > 0:
            print(f"  + {new_count} new candidates from multi-source scan")

    print(f"Loaded {len(scored)} scored + {len(raw) - len(scored)} dropped + multi-source = {len(candidates)} total candidates")
    return candidates


def basin_label(lon: float) -> str:
    """Return basin name for a longitude."""
    if lon < -82.0:
        return "Western"
    elif lon < -80.0:
        return "Central"
    else:
        return "Eastern"


def is_likely_steel(wreck) -> bool:
    """Heuristic: is this wreck likely steel-hulled (magnetically detectable)?"""
    # Support both KnownWreck and WreckRecord
    hull = getattr(wreck, "hull_material", "") or ""
    if "steel" in hull.lower():
        return True
    vtype = (wreck.vessel_type or "").lower()
    return vtype in STEEL_TYPES


# ── Main analysis ────────────────────────────────────────────────────────────

def run_direct_match(wells_csv: str | None = None, output_csv: str | None = None):
    # ── Load data ──
    candidates = load_all_candidates()

    # Use the expanded 91-wreck database (NDA GPS + Wikipedia + ShipwreckWorld)
    wreck_records = build_database(output_csv=None, compare=False)
    # Convert WreckRecord to a duck-typed object compatible with the rest of this script
    wrecks = wreck_records

    wells = load_ogsr_wells(wells_csv) if wells_csv else []

    if not wells:
        print("WARNING: No OGSr wells CSV provided or found -- skipping wellhead cross-reference.")
        print("  Use --wells path/to/ogsr.csv to include wells.\n")

    n_steel = sum(1 for w in wrecks if is_likely_steel(w))
    print(f"Known wrecks: {len(wrecks)} total ({n_steel} likely steel-hulled)")
    print(f"OGSr wells: {len(wells)}")
    print()

    # ── For each candidate: find nearest wreck and nearest well ──
    print("=" * 100)
    print("CANDIDATE -> NEAREST KNOWN WRECK + NEAREST WELL")
    print("=" * 100)

    results = []
    for cand in sorted(candidates, key=lambda c: float(c.get("_dipole_score", 0) or 0), reverse=True):
        lid = cand["label_id"]
        clat = float(cand["center_lat"])
        clon = float(cand["center_lon"])
        amp = float(cand.get("amplitude_peak_abs", 0) or 0)
        dscore = float(cand.get("_dipole_score", 0) or 0)
        tier = cand.get("_tier", "dropped")
        dropped = cand.get("_dropped", "false") == "true"
        basin = basin_label(clon)

        # Ground truth
        gt_label, gt_name = GROUND_TRUTH.get(int(lid), ("unknown", ""))

        # Find nearest wreck (no radius cutoff — always show nearest)
        best_wreck_dist = float("inf")
        best_wreck = None
        wreck_distances = []
        for w in wrecks:
            d = haversine_m(clat, clon, w.lat, w.lon)
            wreck_distances.append((d, w))
            if d < best_wreck_dist:
                best_wreck_dist = d
                best_wreck = w

        # Top-3 nearest wrecks
        wreck_distances.sort(key=lambda x: x[0])
        top3_wrecks = wreck_distances[:3]

        # Find nearest well
        best_well_dist = float("inf")
        best_well_name = None
        if wells:
            for w in wells:
                d = haversine_m(clat, clon, w.lat, w.lon)
                if d < best_well_dist:
                    best_well_dist = d
                    best_well_name = w.name

        # Classification hint
        if gt_label != "unknown":
            hint = f"CONFIRMED {gt_label.upper()}: {gt_name}"
        elif best_wreck_dist < 2000:
            hint = f"VERY CLOSE to wreck ({best_wreck_dist:.0f}m)"
        elif wells and best_well_dist < 2000:
            hint = f"VERY CLOSE to well ({best_well_dist:.0f}m)"
        elif best_wreck_dist < 5000:
            hint = f"Near wreck ({best_wreck_dist:.0f}m)"
        elif wells and best_well_dist < 5000:
            hint = f"Near well ({best_well_dist:.0f}m)"
        else:
            hint = "No close match"

        # Collect result
        result = {
            "label_id": lid,
            "lat": clat,
            "lon": clon,
            "basin": basin,
            "amplitude_nT": amp,
            "dipole_score": dscore,
            "tier": tier,
            "dropped": dropped,
            "ground_truth": gt_label,
            "nearest_wreck": best_wreck.name if best_wreck else "",
            "wreck_dist_m": best_wreck_dist,
            "wreck_steel": is_likely_steel(best_wreck) if best_wreck else False,
            "nearest_well": best_well_name or "",
            "well_dist_m": best_well_dist if wells else None,
            "hint": hint,
        }
        results.append(result)

        # Print
        tag = " [DROPPED]" if dropped else ""
        steel_tag = " [STEEL]" if best_wreck and is_likely_steel(best_wreck) else ""
        print(f"\n  #{lid}{tag}  ({clat:.4f}, {clon:.4f})  {basin} Basin")
        print(f"    Amplitude: {amp:.1f} nT | Dipole score: {dscore:.0f} | Tier: {tier}")
        if gt_label != "unknown":
            print(f"    * GROUND TRUTH: {gt_label} -- {gt_name}")
        print(f"    Nearest wreck: {best_wreck.name}{steel_tag} at {best_wreck_dist:,.0f}m "
              f"({best_wreck.vessel_type}, {best_wreck.source})")
        for d, w in top3_wrecks[1:]:
            st = " [STEEL]" if is_likely_steel(w) else ""
            print(f"      also: {w.name}{st} at {d:,.0f}m ({w.vessel_type}, {w.coord_quality})")
        if wells:
            print(f"    Nearest well: {best_well_name} at {best_well_dist:,.0f}m")
        print(f"    => {hint}")

    # ── For each known wreck: find nearest candidate ──
    print("\n")
    print("=" * 100)
    print("KNOWN WRECKS -> NEAREST CANDIDATE (coverage analysis)")
    print("=" * 100)

    wreck_coverage = []
    for w in wrecks:
        best_cand_dist = float("inf")
        best_cand_id = None
        for cand in candidates:
            clat = float(cand["center_lat"])
            clon = float(cand["center_lon"])
            d = haversine_m(w.lat, w.lon, clat, clon)
            if d < best_cand_dist:
                best_cand_dist = d
                best_cand_id = cand["label_id"]

        steel_tag = " [STEEL]" if is_likely_steel(w) else ""
        wreck_coverage.append((best_cand_dist, w, best_cand_id))

    # Sort by distance — closest matches first
    wreck_coverage.sort(key=lambda x: x[0])

    print(f"\n  {'Wreck':<25} {'Type':<15} {'Steel?':<7} {'Basin':<10} "
          f"{'Nearest Cand':<13} {'Distance':>10}  Notes")
    print("  " + "-" * 95)

    coverage_within_5km = 0
    coverage_within_10km = 0
    for dist, w, cand_id in wreck_coverage:
        steel = "YES" if is_likely_steel(w) else "no"
        b = basin_label(w.lon)
        note = ""
        if dist < 2000:
            note = "<< STRONG MATCH"
            coverage_within_5km += 1
            coverage_within_10km += 1
        elif dist < 5000:
            note = "<< possible match"
            coverage_within_5km += 1
            coverage_within_10km += 1
        elif dist < 10000:
            note = "(within 10km)"
            coverage_within_10km += 1
        elif dist > 50000:
            note = "NO COVERAGE"

        print(f"  {w.name:<25} {w.vessel_type:<15} {steel:<7} {b:<10} "
              f"#{cand_id:<12} {dist:>9,.0f}m  {note}")

    # ── Summary statistics ──
    print("\n")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_wrecks = len(wrecks)
    print(f"\n  Known wrecks:        {total_wrecks}")
    print(f"  Steel-hulled:        {n_steel}")
    print(f"  Within 5km of cand:  {coverage_within_5km} / {total_wrecks}")
    print(f"  Within 10km of cand: {coverage_within_10km} / {total_wrecks}")
    print(f"  Total candidates:    {len(candidates)} (13 scored + {len(candidates) - 13} dropped)")

    # Basin breakdown
    cand_basins = {"Western": 0, "Central": 0, "Eastern": 0}
    wreck_basins = {"Western": 0, "Central": 0, "Eastern": 0}
    for c in candidates:
        cand_basins[basin_label(float(c["center_lon"]))] += 1
    for w in wrecks:
        wreck_basins[basin_label(w.lon)] += 1

    print(f"\n  Basin distribution:")
    print(f"    {'Basin':<10} {'Candidates':>11} {'Known wrecks':>13}")
    for b in ["Western", "Central", "Eastern"]:
        print(f"    {b:<10} {cand_basins[b]:>11} {wreck_basins[b]:>13}")

    # Geographic overlap assessment
    cand_lons = [float(c["center_lon"]) for c in candidates]
    wreck_lons = [w.lon for w in wrecks]
    print(f"\n  Candidate lon range:  {min(cand_lons):.2f} to {max(cand_lons):.2f}")
    print(f"  Wreck lon range:      {min(wreck_lons):.2f} to {max(wreck_lons):.2f}")
    overlap_min = max(min(cand_lons), min(wreck_lons))
    overlap_max = min(max(cand_lons), max(wreck_lons))
    if overlap_min < overlap_max:
        print(f"  Overlap zone:         {overlap_min:.2f} to {overlap_max:.2f}")
    else:
        print(f"  Overlap zone:         NONE — geographic mismatch!")

    # ── Save results CSV ──
    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {out_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direct cross-reference of mag candidates vs known wrecks + wells"
    )
    parser.add_argument("--wells", type=str, default=None,
                        help="Path to OGSr wells CSV (optional)")
    parser.add_argument("--output", "-o", type=str, default="erie_direct_match_results.csv",
                        help="Output CSV path (default: erie_direct_match_results.csv)")
    args = parser.parse_args()
    run_direct_match(wells_csv=args.wells, output_csv=args.output)


if __name__ == "__main__":
    main()
