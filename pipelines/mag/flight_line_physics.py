"""
Flight-Line 1D Profile Physics Validation
==========================================
Extracts raw 1D magnetic profiles from GSC Erie CSV along individual flight lines
near known wellheads, then tests physics formulas on the raw measurements.

This bypasses ALL gridding artifacts (tile centers, interpolation broadening, etc.)
and works directly with the along-track survey measurements.
"""

import json, math, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\thomf\programming\Bagrecovery")
TARGETS = BASE / "wreck_hunting_ml" / "output" / "erie_full_lake_targets.full.json"
GSC_CSV = (BASE / "magnetic_data" / "new data to digest" / "extracted_csv" /
           "Erie__Lake_-_CSV_Point_Data_-_CSV_Donn_es_ponctuelles" / "gsc_erie.csv")
WELLS_CSV = BASE / "eriewelldata" / "wells.csv"
OUTPUT = BASE / "wreck_hunting_ml" / "output" / "flight_line_physics_results.json"

FT_TO_M = 0.3048
DEG_TO_M = 111320.0  # approximate meters per degree latitude


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def perpendicular_distance_m(point_lat, point_lon, line_points):
    """Closest approach distance from a point to a flight line (set of points)."""
    min_dist = float('inf')
    for _, row in line_points.iterrows():
        d = haversine_m(point_lat, point_lon, row["Y"], row["X"])
        if d < min_dist:
            min_dist = d
    return min_dist


def extract_profile(line_df, well_lat, well_lon, radius_m=10000):
    """Extract 1D MAGRAW profile along a flight line within radius_m of a wellhead.
    Returns: along_track_m (array), magraw (array), altitude_m (float), closest_m (float)
    """
    # Compute distance from each point on line to the wellhead
    dists = []
    for _, row in line_df.iterrows():
        d = haversine_m(well_lat, well_lon, row["Y"], row["X"])
        dists.append(d)
    line_df = line_df.copy()
    line_df["dist_to_well"] = dists

    # Closest approach
    closest_idx = line_df["dist_to_well"].idxmin()
    closest_m = line_df.loc[closest_idx, "dist_to_well"]

    # Filter to within radius
    mask = line_df["dist_to_well"] <= radius_m
    subset = line_df[mask].copy()
    if len(subset) < 5:
        return None

    # Compute along-track distance from closest point
    ref_lat = line_df.loc[closest_idx, "Y"]
    ref_lon = line_df.loc[closest_idx, "X"]
    along = []
    for _, row in subset.iterrows():
        d = haversine_m(ref_lat, ref_lon, row["Y"], row["X"])
        # Sign: use longitude difference for E-W lines
        sign = 1.0 if row["X"] >= ref_lon else -1.0
        along.append(sign * d)
    subset["along_track_m"] = along
    subset = subset.sort_values("along_track_m")

    altitude_m = subset["RALT"].median() * FT_TO_M
    return {
        "along_m": subset["along_track_m"].values,
        "magraw": subset["MAGRAW"].values,
        "ralt_m": subset["RALT"].values * FT_TO_M,
        "altitude_m": altitude_m,
        "closest_m": closest_m,
        "n_points": len(subset),
    }


def measure_anomaly(profile):
    """Measure half-width and peak amplitude from a 1D magnetic profile.
    
    Returns dict with: half_width_m, peak_amplitude_nt, peak_pos_m
    """
    along = profile["along_m"]
    mag = profile["magraw"]
    
    if len(mag) < 5:
        return None
    
    # Remove regional trend (linear detrend)
    coeffs = np.polyfit(along, mag, 1)
    trend = np.polyval(coeffs, along)
    residual = mag - trend
    
    # Peak amplitude (max of |residual|)
    peak_idx = np.argmax(np.abs(residual))
    peak_amp = np.abs(residual[peak_idx])
    peak_pos = along[peak_idx]
    
    if peak_amp < 1.0:  # less than 1 nT - no anomaly
        return None
    
    # Half-max level
    half_max = peak_amp / 2.0
    above = np.abs(residual) >= half_max
    
    if not np.any(above):
        return None
    
    # Find half-width: distance between first and last points above half-max
    above_indices = np.where(above)[0]
    left_idx = above_indices[0]
    right_idx = above_indices[-1]
    half_width_m = abs(along[right_idx] - along[left_idx])
    
    # Also try interpolation for better precision
    # Find where |residual| crosses half_max on left side
    left_cross = along[left_idx]
    right_cross = along[right_idx]
    
    for i in range(peak_idx, 0, -1):
        if np.abs(residual[i]) >= half_max and np.abs(residual[i-1]) < half_max:
            # Interpolate
            frac = (half_max - np.abs(residual[i-1])) / (np.abs(residual[i]) - np.abs(residual[i-1]))
            left_cross = along[i-1] + frac * (along[i] - along[i-1])
            break
    
    for i in range(peak_idx, len(residual)-1):
        if np.abs(residual[i]) >= half_max and np.abs(residual[i+1]) < half_max:
            frac = (half_max - np.abs(residual[i+1])) / (np.abs(residual[i]) - np.abs(residual[i+1]))
            right_cross = along[i] + frac * (along[i+1] - along[i])
            break
    
    half_width_interp = abs(right_cross - left_cross)
    
    return {
        "peak_amplitude_nt": float(peak_amp),
        "peak_pos_m": float(peak_pos),
        "half_width_m": float(half_width_m),
        "half_width_interp_m": float(half_width_interp),
        "n_above_half": int(above.sum()),
    }


def main():
    print("=" * 70)
    print("Flight-Line 1D Profile Physics Validation")
    print("=" * 70)
    
    # ── 1. Load known wellhead detections ────────────────────────────
    print("\n[1] Loading known wellhead detections...")
    with open(TARGETS) as f:
        data = json.load(f)
    
    knowns = data["knowns_subtracted"]
    wells = [d for d in knowns if d.get("known_match") == "wellhead"]
    print(f"  Known wellhead detections: {len(wells)}")
    
    # ── 2. Load raw flight-line CSV ─────────────────────────────────
    print("\n[2] Loading GSC Erie flight-line CSV...")
    df = pd.read_csv(GSC_CSV, comment="/", index_col=False)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"  RALT range: {df['RALT'].min():.0f}-{df['RALT'].max():.0f} ft "
          f"= {df['RALT'].min()*FT_TO_M:.0f}-{df['RALT'].max()*FT_TO_M:.0f} m")
    
    # Group by LINE
    lines = df.groupby("LINE")
    line_ids = list(lines.groups.keys())
    print(f"  Flight lines: {len(line_ids)}")
    
    # Pre-compute line mean positions for fast nearest-line lookup
    line_means = []
    for lid, grp in lines:
        line_means.append({
            "line_id": lid,
            "mean_lat": grp["Y"].mean(),
            "mean_lon": grp["X"].mean(),
            "min_lon": grp["X"].min(),
            "max_lon": grp["X"].max(),
            "min_lat": grp["Y"].min(),
            "max_lat": grp["Y"].max(),
            "n_points": len(grp),
        })
    
    # ── 3. For each wellhead, find nearest flight line + extract profile ─
    print("\n[3] Extracting 1D profiles near known wellheads...")
    results = []
    skipped = 0
    
    for i, well in enumerate(wells):
        wlat, wlon = well["lat"], well["lon"]
        
        # Find nearest flight line by closest mean position
        # (rough filter, then refine)
        best_line = None
        best_dist = float('inf')
        
        for lm in line_means:
            # Quick bounding-box check: skip lines far away
            if (wlon < lm["min_lon"] - 0.5 or wlon > lm["max_lon"] + 0.5 or
                wlat < lm["min_lat"] - 0.5 or wlat > lm["max_lat"] + 0.5):
                continue
            d = haversine_m(wlat, wlon, lm["mean_lat"], lm["mean_lon"])
            if d < best_dist:
                best_dist = d
                best_line = lm["line_id"]
        
        if best_line is None or best_dist > 50000:
            skipped += 1
            continue
        
        # Get all points on that line
        line_df = lines.get_group(best_line).copy()
        
        # Actually find TRUE closest approach (not mean)
        dists_to_well = []
        for _, row in line_df.iterrows():
            dists_to_well.append(haversine_m(wlat, wlon, row["Y"], row["X"]))
        true_closest_m = min(dists_to_well)
        
        # Also check the 2 nearest lines (in case mean position is misleading)
        candidate_lines = sorted(line_means, key=lambda lm: haversine_m(wlat, wlon, lm["mean_lat"], lm["mean_lon"]))[:3]
        
        actual_best_line = best_line
        actual_best_dist = true_closest_m
        
        for cand in candidate_lines:
            cand_df = lines.get_group(cand["line_id"])
            for _, row in cand_df.iterrows():
                d = haversine_m(wlat, wlon, row["Y"], row["X"])
                if d < actual_best_dist:
                    actual_best_dist = d
                    actual_best_line = cand["line_id"]
        
        # Extract profile from best line
        line_df = lines.get_group(actual_best_line).copy()
        profile = extract_profile(line_df, wlat, wlon, radius_m=10000)
        
        if profile is None:
            skipped += 1
            continue
        
        # Measure anomaly
        anomaly = measure_anomaly(profile)
        
        result = {
            "well_idx": i,
            "well_name": well.get("known_match_name", ""),
            "well_lat": wlat,
            "well_lon": wlon,
            "line_id": int(actual_best_line) if isinstance(actual_best_line, (int, np.integer)) else actual_best_line,
            "lateral_offset_m": float(actual_best_dist),
            "flight_altitude_m": float(profile["altitude_m"]),
            "n_profile_points": profile["n_points"],
            "gridded_spatial_extent_m": well.get("spatial_extent_m", 0),
            "gridded_peak_amp_nt": well.get("peak_amplitude_nt", 0),
        }
        
        if anomaly:
            result.update({
                "raw_half_width_m": anomaly["half_width_m"],
                "raw_half_width_interp_m": anomaly["half_width_interp_m"],
                "raw_peak_amplitude_nt": anomaly["peak_amplitude_nt"],
                "raw_peak_pos_m": anomaly["peak_pos_m"],
                "raw_n_above_half": anomaly["n_above_half"],
            })
        
        results.append(result)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(wells)} wellheads...")
    
    print(f"  Done: {len(results)} profiles extracted, {skipped} skipped")
    
    # ── 4. Physics tests on raw 1D profiles ─────────────────────────
    print("\n" + "=" * 70)
    print("PHYSICS VALIDATION RESULTS (Raw 1D Profiles)")
    print("=" * 70)
    
    # Filter to those with anomaly measurements
    measured = [r for r in results if "raw_half_width_m" in r]
    print(f"\nProfiles with measurable anomalies: {len(measured)}/{len(results)}")
    
    if not measured:
        print("ERROR: No measurable anomalies found!")
        return
    
    # ── Summary statistics ───────────────────────────────────────────
    hw = np.array([r["raw_half_width_m"] for r in measured])
    hw_interp = np.array([r["raw_half_width_interp_m"] for r in measured])
    amp = np.array([r["raw_peak_amplitude_nt"] for r in measured])
    alt = np.array([r["flight_altitude_m"] for r in measured])
    offset = np.array([r["lateral_offset_m"] for r in measured])
    gridded_ext = np.array([r["gridded_spatial_extent_m"] for r in measured])
    gridded_amp = np.array([r["gridded_peak_amp_nt"] for r in measured])
    
    print(f"\n--- Raw Profile Measurements ---")
    print(f"  half_width_m:     min={hw.min():.0f}  median={np.median(hw):.0f}  "
          f"mean={hw.mean():.0f}  max={hw.max():.0f}  std={hw.std():.0f}")
    print(f"  half_width_interp: min={hw_interp.min():.0f}  median={np.median(hw_interp):.0f}  "
          f"mean={hw_interp.mean():.0f}  max={hw_interp.max():.0f}")
    print(f"  peak_amp_nt:      min={amp.min():.1f}  median={np.median(amp):.1f}  "
          f"mean={amp.mean():.1f}  max={amp.max():.1f}")
    print(f"  flight_alt_m:     min={alt.min():.0f}  median={np.median(alt):.0f}  "
          f"mean={alt.mean():.0f}  max={alt.max():.0f}")
    print(f"  lateral_offset_m: min={offset.min():.0f}  median={np.median(offset):.0f}  "
          f"mean={offset.mean():.0f}  max={offset.max():.0f}")
    
    print(f"\n--- Gridded vs Raw Comparison ---")
    print(f"  Gridded spatial_extent_m: median={np.median(gridded_ext):.0f} mean={gridded_ext.mean():.0f}")
    print(f"  Raw half_width_m:         median={np.median(hw):.0f} mean={hw.mean():.0f}")
    valid_ext = gridded_ext > 0
    if np.any(valid_ext):
        ratio = gridded_ext[valid_ext] / hw[valid_ext]
        print(f"  Ratio (gridded/raw):      median={np.median(ratio):.1f}x  mean={ratio.mean():.1f}x")
    
    # ── TEST 1: Half-width ~ altitude ────────────────────────────────
    print(f"\n--- TEST 1: Half-Width vs Altitude ---")
    print(f"  Theory: half_width ≈ k × altitude (k ≈ 1.0-2.5 for compact dipoles)")
    k_values = hw / alt
    k_interp = hw_interp / alt
    print(f"  k = half_width / altitude:")
    print(f"    min={k_values.min():.2f}  median={np.median(k_values):.2f}  "
          f"mean={k_values.mean():.2f}  max={k_values.max():.2f}")
    print(f"  k (interpolated): median={np.median(k_interp):.2f}  mean={k_interp.mean():.2f}")
    
    # Correlation of half_width vs altitude
    if alt.std() > 0:
        r = np.corrcoef(alt, hw)[0, 1]
        print(f"  Correlation(altitude, half_width): r = {r:.3f}")
    
    in_range = np.sum((k_values >= 0.5) & (k_values <= 5.0))
    print(f"  # with k in [0.5, 5.0]: {in_range}/{len(k_values)} ({100*in_range/len(k_values):.0f}%)")
    
    # ── TEST 2: Amplitude vs Lateral Offset ──────────────────────────
    print(f"\n--- TEST 2: Amplitude vs Lateral Offset ---")
    print(f"  Theory: amplitude decreases with lateral offset (1/r³ when far)")
    
    # Need some spread in offset to test this
    if offset.std() > 10:
        r = np.corrcoef(offset, amp)[0, 1]
        print(f"  Correlation(offset, amplitude): r = {r:.3f}")
        # Bin by offset quartiles
        q25, q50, q75 = np.percentile(offset, [25, 50, 75])
        for lo, hi, label in [(0, q25, "Q1"), (q25, q50, "Q2"), (q50, q75, "Q3"), (q75, offset.max()+1, "Q4")]:
            mask = (offset >= lo) & (offset < hi)
            if mask.sum() > 0:
                print(f"    {label} offset [{lo:.0f}-{hi:.0f}m]: n={mask.sum()}, "
                      f"median amp={np.median(amp[mask]):.1f} nT")
    else:
        print(f"  Offset range too narrow ({offset.std():.0f}m std) for meaningful test")
    
    # ── TEST 3: Offset Estimation ────────────────────────────────────
    print(f"\n--- TEST 3: Offset Estimation ---")
    print(f"  Theory: offset ≈ sqrt(half_width² - altitude²)")
    
    valid = hw > alt  # half_width must exceed altitude for valid sqrt
    print(f"  Profiles where half_width > altitude: {valid.sum()}/{len(hw)}")
    
    if valid.sum() > 3:
        est_offset = np.sqrt(hw[valid]**2 - alt[valid]**2)
        actual_offset = offset[valid]
        residuals = est_offset - actual_offset
        print(f"  Estimated offset: median={np.median(est_offset):.0f}m  mean={est_offset.mean():.0f}m")
        print(f"  Actual offset:    median={np.median(actual_offset):.0f}m  mean={actual_offset.mean():.0f}m")
        print(f"  Residual (est-actual): median={np.median(residuals):.0f}m  MAE={np.mean(np.abs(residuals)):.0f}m")
        if actual_offset.std() > 0:
            r = np.corrcoef(actual_offset, est_offset)[0, 1]
            print(f"  Correlation(actual, estimated): r = {r:.3f}")
    
    # ── TEST 4: Size/Extent Correction ───────────────────────────────
    print(f"\n--- TEST 4: True Size Estimation ---")
    print(f"  Theory: true_size ≈ extent - 2*(altitude + offset)")
    
    true_size = hw - 2 * (alt + offset)
    true_size_interp = hw_interp - 2 * (alt + offset)
    valid_size = true_size > 0
    print(f"  Positive true_size: {valid_size.sum()}/{len(true_size)}")
    if valid_size.sum() > 0:
        print(f"  true_size (positive): median={np.median(true_size[valid_size]):.0f}m  "
              f"mean={true_size[valid_size].mean():.0f}m")
    print(f"  All true_size: median={np.median(true_size):.0f}m  mean={true_size.mean():.0f}m")
    print(f"  (Negative values = anomaly narrower than altitude+offset correction → compact/deep source)")
    
    # ── TEST 5: Slant Distance Check ────────────────────────────────
    print(f"\n--- TEST 5: Slant Distance ---")
    print(f"  Theory: slant_distance = sqrt(altitude² + offset²)")
    slant = np.sqrt(alt**2 + offset**2)
    print(f"  Slant distance: min={slant.min():.0f}m  median={np.median(slant):.0f}m  max={slant.max():.0f}m")
    
    # k_slant = half_width / slant (should be ~1.0-2.0 for compact dipoles)
    k_slant = hw / slant
    print(f"  k_slant = half_width / slant_distance:")
    print(f"    min={k_slant.min():.2f}  median={np.median(k_slant):.2f}  "
          f"mean={k_slant.mean():.2f}  max={k_slant.max():.2f}")
    in_range_slant = np.sum((k_slant >= 0.5) & (k_slant <= 3.0))
    print(f"  # with k_slant in [0.5, 3.0]: {in_range_slant}/{len(k_slant)} ({100*in_range_slant/len(k_slant):.0f}%)")
    
    # ── TEST 6: Amplitude × distance³ consistency ────────────────────
    print(f"\n--- TEST 6: Dipole Moment Consistency ---")
    print(f"  Theory: amplitude × slant³ ≈ constant (magnetic moment) for similar sources")
    moment = amp * slant**3
    cv = moment.std() / moment.mean() if moment.mean() > 0 else float('inf')
    print(f"  amp × slant³: median={np.median(moment):.2e}  mean={moment.mean():.2e}  "
          f"std={moment.std():.2e}  CV={cv:.2f}")
    print(f"  (CV < 0.5 → consistent; CV > 1.0 → variable source strengths)")
    
    # ── Save results ─────────────────────────────────────────────────
    print(f"\n--- Saving results to {OUTPUT.name} ---")
    
    # Convert numpy types for JSON
    output_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            else:
                clean[k] = v
        output_results.append(clean)
    
    summary = {
        "n_profiles": len(results),
        "n_measured": len(measured),
        "raw_half_width_median_m": float(np.median(hw)),
        "raw_peak_amp_median_nt": float(np.median(amp)),
        "k_altitude_median": float(np.median(k_values)),
        "k_slant_median": float(np.median(k_slant)),
        "gridded_to_raw_ratio_median": float(np.median(ratio)) if np.any(valid_ext) else None,
    }
    
    with open(OUTPUT, "w") as f:
        json.dump({"summary": summary, "profiles": output_results}, f, indent=2)
    
    print(f"  Saved {len(output_results)} profile results")
    print("\nDone!")


if __name__ == "__main__":
    main()
