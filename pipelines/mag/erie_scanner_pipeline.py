"""
Lake Erie Focused Scanner — Training & Detection Pipeline
==========================================================
Tuned specifically for Lake Erie basin characteristics:
  - Sedimentary basin (low background noise → wellheads stand out)
  - Known gas/oil wells from OGSr Ontario dataset
  - Loran-C warp correction for aero-mag survey targets
  - Satellite data integration (EMAG2, WDMAM, ESA Swarm)
  - Ground truth: #103=Colgate wreck, #63=#85=wellheads

This extends the baseline mag_data_pipeline with Lake Erie–specific:
  1. Well-head discriminator training (RF classifier: wreck vs wellhead vs geological)
  2. Regional background model tuned for Erie basin geology
  3. Satellite mag data gap-filling for areas without aero coverage
  4. Cross-reference against comprehensive known-wreck database
"""

from __future__ import annotations

import json
import logging
import os
import csv
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Imports from sibling modules ─────────────────────────────────────────────
# These are resolved at runtime when called from the FastAPI backend
try:
    from scripts.erie_wellhead_discriminator import (
        load_ogsr_wells, load_candidates_csv, cross_reference_candidates,
        get_all_known_wrecks, extract_discriminator_features, save_results_csv,
        CandidateMatch, Wellhead, KnownWreck, GROUND_TRUTH, haversine_m,
    )
except ImportError:
    from erie_wellhead_discriminator import (
        load_ogsr_wells, load_candidates_csv, cross_reference_candidates,
        get_all_known_wrecks, extract_discriminator_features, save_results_csv,
        CandidateMatch, Wellhead, KnownWreck, GROUND_TRUTH, haversine_m,
    )


# ── Lake Erie bounding box ──────────────────────────────────────────────────

ERIE_BBOX = (-83.50, 41.35, -78.80, 42.90)  # (lon_min, lat_min, lon_max, lat_max)

# Sub-basin definitions for regional tuning
ERIE_BASINS = {
    "western": {"lon_min": -83.50, "lon_max": -82.00, "lat_min": 41.35, "lat_max": 42.00,
                "depth_range_m": (7, 19), "geology": "dolomite_limestone"},
    "central": {"lon_min": -82.00, "lon_max": -80.00, "lat_min": 41.60, "lat_max": 42.60,
                "depth_range_m": (19, 25), "geology": "shale_siltstone"},
    "eastern": {"lon_min": -80.00, "lon_max": -78.80, "lat_min": 42.00, "lat_max": 42.90,
                "depth_range_m": (25, 64), "geology": "shale_mudstone"},
}


# ── Satellite data sources for Lake Erie ─────────────────────────────────────

SATELLITE_SOURCES = {
    "emag2v3": {
        "name": "EMAG2v3 (NOAA)",
        "resolution_arcmin": 2,
        "resolution_m_approx": 3700,
        "type": "global_grid",
        "url_pattern": "https://www.ngdc.noaa.gov/geomag/EMM/data/geomag/EMAG2_V3_20170530.csv",
        "good_for": "gap filling where aero coverage is missing",
    },
    "wdmam": {
        "name": "WDMAM v2 (satellite + aero composite)",
        "resolution_arcmin": 3,
        "resolution_m_approx": 5500,
        "type": "global_grid",
        "good_for": "regional trend removal, basin-scale features",
    },
    "swarm": {
        "name": "ESA Swarm (satellite vector mag)",
        "resolution_arcmin": 0,  # point data along orbits
        "type": "orbit_track",
        "good_for": "temporal variation removal, secular change correction",
    },
    "usgs_usmag": {
        "name": "USGS US Magnetic Anomaly (aero)",
        "resolution_m_approx": 1000,
        "type": "aero_survey",
        "good_for": "primary detection source for US waters",
    },
    "nrcan": {
        "name": "NRCan Canadian Aeromagnetic",
        "resolution_m_approx": 800,
        "type": "aero_survey",
        "good_for": "primary detection source for Canadian waters",
    },
}


# ── Lake Erie Scanner pipeline ──────────────────────────────────────────────

def run_erie_scan(
    candidates_csv: str | Path | None = None,
    wells_csv: str | Path | None = None,
    output_dir: str | Path = "erie_scanner_output",
    wellhead_radius_m: float = 2000.0,
    satellite_sources: list[str] | None = None,
    apply_loran_correction: bool = True,
    retrain: bool = False,
    progress_callback=None,
) -> dict:
    """Run the full Lake Erie focused scanner pipeline.
    
    Steps:
    1. Load existing candidates (from baseline mag pipeline)
    2. Load Ontario well data (OGSr)
    3. Cross-reference candidates against wells + known wrecks
    4. Apply Loran-C warp correction
    5. Train/update discriminator model (wreck vs wellhead vs geological)
    6. Score and rank candidates with discriminator
    7. Export results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg: str, pct: float = -1):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, pct)

    # ── Step 1: Load candidates ──
    progress("Loading mag anomaly candidates...", 0.0)
    if candidates_csv:
        candidates = load_candidates_csv(candidates_csv)
    else:
        # Try default location
        default_csv = Path("adaptive_bg_erie_1000yd/adaptive_candidates_scored.csv")
        if default_csv.exists():
            candidates = load_candidates_csv(default_csv)
        else:
            return {"error": "No candidates CSV provided and default not found"}

    progress(f"Loaded {len(candidates)} candidates", 0.1)

    # ── Step 2: Load wells ──
    progress("Loading Ontario petroleum wells...", 0.15)
    if wells_csv:
        wells = load_ogsr_wells(wells_csv)
    else:
        # Try common locations
        for p in [
            Path("data/wells.csv"),
            Path("reference/wells.csv"),
            Path("erie_scanner_output/wells.csv"),
        ]:
            if p.exists():
                wells = load_ogsr_wells(p)
                break
        else:
            wells = []
            logger.warning("No wells CSV found — running without well discrimination")

    progress(f"Loaded {len(wells)} wells in Lake Erie region", 0.2)

    # ── Step 3: Load known wrecks ──
    known_wrecks = get_all_known_wrecks()
    progress(f"Loaded {len(known_wrecks)} known wrecks", 0.25)

    # ── Step 4: Cross-reference ──
    progress("Cross-referencing candidates against wells and known wrecks...", 0.3)
    matched = cross_reference_candidates(
        candidates=candidates,
        wells=wells,
        known_wrecks=known_wrecks,
        wellhead_radius_m=wellhead_radius_m,
        apply_loran_correction=apply_loran_correction,
    )
    progress(f"Cross-referenced {len(matched)} candidates", 0.5)

    # Count matches
    wellhead_matches = sum(1 for m in matched if m.wellhead_distance_m is not None)
    wreck_matches = sum(1 for m in matched if m.wreck_distance_m is not None)
    gt_wrecks = sum(1 for m in matched if m.ground_truth == "wreck")
    gt_wellheads = sum(1 for m in matched if m.ground_truth == "wellhead")

    progress(f"Found {wellhead_matches} near wellheads, {wreck_matches} near known wrecks", 0.55)
    progress(f"Ground truth: {gt_wrecks} confirmed wrecks, {gt_wellheads} confirmed wellheads", 0.6)

    # ── Step 5: Train discriminator if requested ──
    model_accuracy = None
    if retrain and len(matched) >= 3:
        progress("Training wellhead discriminator model...", 0.65)
        model_accuracy = _train_discriminator(matched, output_dir)
        progress(f"Discriminator trained — accuracy: {model_accuracy:.1%}", 0.8)
    elif retrain:
        progress("Not enough labeled data to train discriminator (need ≥3)", 0.8)

    # ── Step 6: Apply discriminator scoring ──
    progress("Scoring candidates with basin-specific model...", 0.85)
    for m in matched:
        _apply_basin_scoring(m)

    # Re-sort after scoring
    matched.sort(key=lambda x: x.composite_score, reverse=True)

    # ── Step 7: Export ──
    progress("Exporting results...", 0.9)
    save_results_csv(matched, output_dir / "erie_candidates_scored.csv")

    # Save JSON for frontend
    results_json = {
        "candidates": [asdict(m) for m in matched],
        "wellheads_loaded": len(wells),
        "known_wrecks_loaded": len(known_wrecks),
        "candidates_filtered": len(matched),
        "wellhead_matches": wellhead_matches,
        "wreck_matches": wreck_matches,
        "model_accuracy": model_accuracy,
        "ground_truth_summary": {
            "confirmed_wrecks": gt_wrecks,
            "confirmed_wellheads": gt_wellheads,
            "unknown": sum(1 for m in matched if m.ground_truth == "unknown"),
        },
        "basins_covered": list(ERIE_BASINS.keys()),
    }
    with open(output_dir / "erie_scan_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    progress("Lake Erie scan complete", 1.0)
    return results_json


# ── Discriminator training ──────────────────────────────────────────────────

def _train_discriminator(
    matched: list[CandidateMatch],
    output_dir: Path,
) -> float:
    """Train a RandomForest classifier to distinguish wreck vs wellhead vs geological.
    
    Uses ground truth labels + proximity-based pseudo-labels.
    Returns cross-validation accuracy.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        logger.warning("scikit-learn not available — skipping discriminator training")
        return 0.0

    # Build feature matrix
    features = []
    labels = []
    label_map = {"wreck": 1, "wellhead": 0, "geological": 0, "unknown": -1}

    for m in matched:
        feat = extract_discriminator_features(m)
        gt = feat.pop("ground_truth")
        label = label_map.get(gt, -1)
        if label == -1:
            # Use heuristic pseudo-labels for unlabeled candidates
            if m.wellhead_distance_m is not None and m.wellhead_distance_m < 1000:
                label = 0  # likely wellhead
            elif m.wreck_distance_m is not None and m.wreck_distance_m < 3000:
                label = 1  # likely wreck
            else:
                continue  # skip fully unknown for training
        features.append(list(feat.values()))
        labels.append(label)

    if len(features) < 3 or len(set(labels)) < 2:
        logger.warning("Insufficient labeled data for training (%d samples, %d classes)",
                       len(features), len(set(labels)))
        return 0.0

    X = np.array(features, dtype=np.float64)
    y = np.array(labels)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=999999, neginf=-999999)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ensemble
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    # Cross-validate
    n_folds = min(5, len(features))
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
            accuracy = float(np.mean(scores))
        except ValueError:
            accuracy = 0.0
    else:
        accuracy = 0.0

    # Train final model on all data
    clf.fit(X_scaled, y)

    # Save model
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(clf, models_dir / "erie_discriminator.joblib")
    joblib.dump(scaler, models_dir / "erie_scaler.joblib")

    # Save feature importances
    feature_names = list(extract_discriminator_features(matched[0]).keys())
    feature_names.remove("ground_truth")
    importances = dict(zip(feature_names, clf.feature_importances_.tolist()))
    with open(models_dir / "erie_feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    logger.info("Discriminator saved to %s (accuracy=%.1f%%)", models_dir, accuracy * 100)
    return accuracy


# ── Basin-specific scoring ──────────────────────────────────────────────────

def _apply_basin_scoring(match: CandidateMatch):
    """Apply Lake Erie basin-specific scoring adjustments.
    
    Adjustments based on sub-basin characteristics:
    - Western basin: shallow, heavy mineral deposits → raise threshold
    - Central basin: moderate depth, gas wells common → wellhead penalty
    - Eastern basin: deep, fewer wells → lower threshold for detection
    """
    lat, lon = match.center_lat, match.center_lon

    # Determine sub-basin
    basin = None
    for name, bounds in ERIE_BASINS.items():
        if (bounds["lon_min"] <= lon <= bounds["lon_max"]
                and bounds["lat_min"] <= lat <= bounds["lat_max"]):
            basin = name
            break

    if basin is None:
        return  # Outside defined basins

    # ── Wellhead proximity penalty ──
    if match.wellhead_distance_m is not None:
        # Closer to wellhead → bigger penalty
        if match.wellhead_distance_m < 500:
            match.composite_score *= 0.3   # Heavy penalty
            match.all_reasons.append(f"-70% wellhead within {match.wellhead_distance_m:.0f}m")
        elif match.wellhead_distance_m < 1000:
            match.composite_score *= 0.6
            match.all_reasons.append(f"-40% wellhead within {match.wellhead_distance_m:.0f}m")
        elif match.wellhead_distance_m < 2000:
            match.composite_score *= 0.8
            match.all_reasons.append(f"-20% wellhead within {match.wellhead_distance_m:.0f}m")

    # ── Known wreck proximity bonus ──
    if match.wreck_distance_m is not None:
        if match.wreck_distance_m < 1000:
            match.composite_score *= 1.5
            match.all_reasons.append(f"+50% known wreck '{match.nearest_known_wreck}' within {match.wreck_distance_m:.0f}m")
        elif match.wreck_distance_m < 3000:
            match.composite_score *= 1.2
            match.all_reasons.append(f"+20% near wreck '{match.nearest_known_wreck}' within {match.wreck_distance_m:.0f}m")

    # ── Basin-specific adjustments ──
    if basin == "western":
        # Western basin: shallow, lots of mineral deposits and shore infrastructure
        # Raise detection bar
        if match.amplitude_peak_abs < 100:
            match.composite_score *= 0.9
            match.all_reasons.append("-10% western basin low amplitude (< 100 nT)")
    elif basin == "central":
        # Central basin: most gas wells are here
        # Extra penalty for monopolar signatures (wellhead-like)
        if "dipolar" not in " ".join(match.all_reasons).lower():
            match.composite_score *= 0.7
            match.all_reasons.append("-30% central basin monopolar (wellhead characteristic)")
    elif basin == "eastern":
        # Eastern basin: deepest, fewest wells, best wreck hunting
        # Bonus for dipolar signatures
        if "dipolar" in " ".join(match.all_reasons).lower():
            match.composite_score *= 1.1
            match.all_reasons.append("+10% eastern basin dipolar signature bonus")


# ── Satellite data gap analysis ──────────────────────────────────────────────

def identify_coverage_gaps(
    aero_grid_dir: str | Path,
    erie_bbox: tuple = ERIE_BBOX,
) -> dict:
    """Identify areas of Lake Erie with no aero-mag coverage that need satellite gap-filling.
    
    Returns a dict of sub-regions and their coverage status.
    """
    # This checks which tiles exist in the aero grid directory
    aero_dir = Path(aero_grid_dir)
    covered_tiles = set()
    
    if aero_dir.exists():
        for tif in aero_dir.glob("*.tif"):
            covered_tiles.add(tif.stem)

    gaps = {
        "total_tiles_needed": 0,
        "tiles_with_aero": len(covered_tiles),
        "gap_regions": [],
        "satellite_fill_recommended": [],
    }

    # Tile the Erie bbox at ~0.5° resolution
    lon_min, lat_min, lon_max, lat_max = erie_bbox
    tile_step = 0.5
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            gaps["total_tiles_needed"] += 1
            tile_key = f"erie_{lat:.1f}_{lon:.1f}"
            if tile_key not in covered_tiles:
                # Check if any aero tile covers this area
                any_cover = False
                for tk in covered_tiles:
                    if str(lat)[:4] in tk and str(abs(lon))[:4] in tk:
                        any_cover = True
                        break
                if not any_cover:
                    gaps["gap_regions"].append({
                        "lat_center": lat + tile_step / 2,
                        "lon_center": lon + tile_step / 2,
                        "recommended_source": "emag2v3" if lon > -81.0 else "wdmam",
                    })
            lon += tile_step
        lat += tile_step

    gaps["satellite_fill_recommended"] = [
        src for src in ["emag2v3", "wdmam"]
        if any(g["recommended_source"] == src for g in gaps["gap_regions"])
    ]

    return gaps


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Lake Erie Focused Scanner")
    parser.add_argument("--candidates", "-c", help="Path to adaptive_candidates_scored.csv")
    parser.add_argument("--wells", "-w", help="Path to OGSr wells.csv")
    parser.add_argument("--output", "-o", default="erie_scanner_output", help="Output directory")
    parser.add_argument("--wellhead-radius", type=float, default=2000, help="Wellhead match radius (m)")
    parser.add_argument("--no-loran", action="store_true", help="Skip Loran-C warp correction")
    parser.add_argument("--retrain", action="store_true", help="Retrain discriminator model")
    args = parser.parse_args()

    result = run_erie_scan(
        candidates_csv=args.candidates,
        wells_csv=args.wells,
        output_dir=args.output,
        wellhead_radius_m=args.wellhead_radius,
        apply_loran_correction=not args.no_loran,
        retrain=args.retrain,
    )

    print(f"\n{'='*60}")
    print(f"Lake Erie Scan Complete")
    print(f"{'='*60}")
    print(f"Candidates analyzed:  {result.get('candidates_filtered', 0)}")
    print(f"Wellheads loaded:     {result.get('wellheads_loaded', 0)}")
    print(f"Known wrecks loaded:  {result.get('known_wrecks_loaded', 0)}")
    print(f"Near wellheads:       {result.get('wellhead_matches', 0)}")
    print(f"Near known wrecks:    {result.get('wreck_matches', 0)}")
    gt = result.get("ground_truth_summary", {})
    print(f"Confirmed wrecks:     {gt.get('confirmed_wrecks', 0)}")
    print(f"Confirmed wellheads:  {gt.get('confirmed_wellheads', 0)}")
    if result.get("model_accuracy"):
        print(f"Model accuracy:       {result['model_accuracy']:.1%}")
