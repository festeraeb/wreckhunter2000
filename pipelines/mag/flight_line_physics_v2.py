"""
Flight-Line 1D Profile Physics Validation v2
=============================================
FIXES from v1:
- Uses ACTUAL well positions from wells.csv (not tile-center detections)
- Narrows search to ±2km along track from closest approach (not ±10km)
- Looks for local peak near wellhead, not global peak
- Vectorized haversine for speed
"""

import json, math, sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\thomf\programming\Bagrecovery")
TARGETS = BASE / "wreck_hunting_ml" / "output" / "erie_full_lake_targets.full.json"
GSC_CSV = (BASE / "magnetic_data" / "new data to digest" / "extracted_csv" /
           "Erie__Lake_-_CSV_Point_Data_-_CSV_Donn_es_ponctuelles" / "gsc_erie.csv")
WELLS_CSV = BASE / "eriewelldata" / "wells.csv"
OUTPUT = BASE / "wreck_hunting_ml" / "output" / "flight_line_physics_v2.json"

FT_TO_M = 0.3048


def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine: single point vs arrays. Returns meters."""
    R = 6_371_000.0
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2_arr)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def find_best_line(well_lat, well_lon, lines_grouped, line_summaries):
    """Find the flight line with closest approach to a well position.
    Uses vectorized distance computation for speed.
    """
    # Quick filter by bounding box (±0.2° ≈ 22km)
    candidates = []
    for ls in line_summaries:
        if (well_lon < ls["min_lon"] - 0.2 or well_lon > ls["max_lon"] + 0.2 or
            well_lat < ls["min_lat"] - 0.2 or well_lat > ls["max_lat"] + 0.2):
            continue
        candidates.append(ls)
    
    if not candidates:
        # Expand search
        candidates = line_summaries
    
    best_line = None
    best_closest = float('inf')
    
    for ls in candidates:
        grp = lines_grouped.get_group(ls["line_id"])
        dists = haversine_vec(well_lat, well_lon, grp["Y"].values, grp["X"].values)
        min_d = dists.min()
        if min_d < best_closest:
            best_closest = min_d
            best_line = ls["line_id"]
    
    return best_line, best_closest


def extract_and_measure(line_df, well_lat, well_lon, radius_m=2000):
    """Extract 1D profile near wellhead and measure anomaly properties.
    
    Returns dict with measurements, or None if no anomaly found.
    """
    lats = line_df["Y"].values
    lons = line_df["X"].values
    mags = line_df["MAGRAW"].values
    ralts = line_df["RALT"].values * FT_TO_M
    
    # Distance from each point to well
    dists = haversine_vec(well_lat, well_lon, lats, lons)
    closest_idx = np.argmin(dists)
    closest_m = dists[closest_idx]
    
    # Along-track distance from closest point
    ref_lat, ref_lon = lats[closest_idx], lons[closest_idx]
    along = haversine_vec(ref_lat, ref_lon, lats, lons)
    # Sign by longitude difference (E-W lines)
    signs = np.where(lons >= ref_lon, 1.0, -1.0)
    along = along * signs
    
    # Sort by along-track
    order = np.argsort(along)
    along = along[order]
    mags = mags[order]
    ralts = ralts[order]
    
    # Reset closest_idx in sorted order
    sorted_closest = np.argmin(np.abs(along))
    
    # Window: ±radius_m from closest approach
    mask = np.abs(along) <= radius_m
    if mask.sum() < 5:
        # Try wider window up to 5km
        mask = np.abs(along) <= 5000
        if mask.sum() < 5:
            return None
    
    a_win = along[mask]
    m_win = mags[mask]
    r_win = ralts[mask]
    altitude_m = np.median(r_win)
    
    # Detrend using EDGES of window (outer 20% on each side)
    n = len(a_win)
    edge = max(2, n // 5)
    edge_x = np.concatenate([a_win[:edge], a_win[-edge:]])
    edge_y = np.concatenate([m_win[:edge], m_win[-edge:]])
    if len(edge_x) >= 2:
        coeffs = np.polyfit(edge_x, edge_y, 1)
        trend = np.polyval(coeffs, a_win)
    else:
        trend = np.full_like(m_win, np.median(m_win))
    residual = m_win - trend
    
    # Find peak near center (within 500m of closest approach)
    near_center = np.abs(a_win) <= 500
    if near_center.sum() < 2:
        near_center = np.abs(a_win) <= 1000
    if near_center.sum() < 2:
        near_center = np.ones(len(a_win), dtype=bool)  # fallback: all
    
    # Peak = max |residual| near center
    local_abs = np.abs(residual) * near_center
    peak_idx = np.argmax(local_abs)
    peak_amp = np.abs(residual[peak_idx])
    peak_pos = a_win[peak_idx]
    
    if peak_amp < 1.0:
        return None
    
    # Half-width from this local peak
    half_max = peak_amp / 2.0
    
    # Scan outward from peak to find half-max crossings
    left_cross = a_win[0]  # default
    right_cross = a_win[-1]
    
    # Left side
    for i in range(peak_idx, 0, -1):
        if np.abs(residual[i]) >= half_max and np.abs(residual[i-1]) < half_max:
            frac = (half_max - np.abs(residual[i-1])) / (np.abs(residual[i]) - np.abs(residual[i-1]) + 1e-12)
            left_cross = a_win[i-1] + frac * (a_win[i] - a_win[i-1])
            break
    
    # Right side
    for i in range(peak_idx, len(residual)-1):
        if np.abs(residual[i]) >= half_max and np.abs(residual[i+1]) < half_max:
            frac = (half_max - np.abs(residual[i+1])) / (np.abs(residual[i]) - np.abs(residual[i+1]) + 1e-12)
            right_cross = a_win[i] + frac * (a_win[i+1] - a_win[i])
            break
    
    half_width = abs(right_cross - left_cross)
    
    # Also count contiguous above-half-max from peak
    above = np.abs(residual) >= half_max
    left_contig = peak_idx
    while left_contig > 0 and above[left_contig - 1]:
        left_contig -= 1
    right_contig = peak_idx
    while right_contig < len(above) - 1 and above[right_contig + 1]:
        right_contig += 1
    contig_width = abs(a_win[right_contig] - a_win[left_contig])
    
    return {
        "lateral_offset_m": float(closest_m),
        "flight_altitude_m": float(altitude_m),
        "n_profile_points": int(mask.sum()),
        "peak_amplitude_nt": float(peak_amp),
        "peak_pos_m": float(peak_pos),
        "half_width_interp_m": float(half_width),
        "half_width_contig_m": float(contig_width),
        "n_above_half": int(above.sum()),
    }


def main():
    print("=" * 70)
    print("Flight-Line 1D Profile Physics Validation v2")
    print("=" * 70)
    
    # ── 1. Load actual well positions ────────────────────────────────
    print("\n[1] Loading well positions from wells.csv...")
    wells_df = pd.read_csv(WELLS_CSV, encoding="latin-1", low_memory=False)
    print(f"  Wells: {len(wells_df)}, columns: {list(wells_df.columns)[:8]}...")
    
    # Ontario MNR wells.csv uses these specific column names
    lat_col = "SUR_LAT83"
    lon_col = "SUR_LONG83"
    name_col = "WELL_NAME"
    fullname_col = "FULL_NAME"
    
    # Convert lat/lon to numeric
    wells_df[lat_col] = pd.to_numeric(wells_df[lat_col], errors="coerce")
    wells_df[lon_col] = pd.to_numeric(wells_df[lon_col], errors="coerce")
    
    valid_wells = wells_df[wells_df[lat_col].notna() & wells_df[lon_col].notna()]
    print(f"  Wells with valid coords: {len(valid_wells)}/{len(wells_df)}")
    print(f"  Lat range: {valid_wells[lat_col].min():.2f} - {valid_wells[lat_col].max():.2f}")
    print(f"  Lon range: {valid_wells[lon_col].min():.2f} - {valid_wells[lon_col].max():.2f}")
    
    # ── 2. Load known wellhead detections for name matching ──────────
    print("\n[2] Loading detection targets for well name matching...")
    with open(TARGETS) as f:
        data = json.load(f)
    knowns = data["knowns_subtracted"]
    det_wells = [d for d in knowns if d.get("known_match") == "wellhead"]
    print(f"  Known wellhead detections: {len(det_wells)}")
    
    # Match detection well names to wells.csv
    matched_wells = []
    unmatched = []
    
    for d in det_wells:
        wname = d.get("known_match_name", "").strip()
        if not wname:
            continue
        
        # Try exact match on WELL_NAME first, then FULL_NAME contains
        match = valid_wells[valid_wells[name_col].astype(str).str.strip() == wname]
        if len(match) == 0:
            # Try partial match on FULL_NAME
            match = valid_wells[valid_wells[fullname_col].astype(str).str.contains(wname, case=False, na=False)]
        if len(match) == 0:
            # Try partial match on WELL_NAME
            match = valid_wells[valid_wells[name_col].astype(str).str.contains(wname, case=False, na=False)]
        
        if len(match) > 0:
            row = match.iloc[0]
            matched_wells.append({
                "name": wname,
                "lat": float(row[lat_col]),
                "lon": float(row[lon_col]),
                "det_lat": d["lat"],
                "det_lon": d["lon"],
                "det_spatial_extent_m": d.get("spatial_extent_m", 0),
                "det_peak_amp_nt": d.get("peak_amplitude_nt", 0),
            })
            continue
        
        # Fallback: use detection position (tile center)
        # but offset by known_match_distance_m in a random direction...
        # Actually, for now just use detection position as approximate
        unmatched.append(wname)
        matched_wells.append({
            "name": wname,
            "lat": d["lat"],  # tile center (approximate)
            "lon": d["lon"],
            "det_lat": d["lat"],
            "det_lon": d["lon"],
            "det_spatial_extent_m": d.get("spatial_extent_m", 0),
            "det_peak_amp_nt": d.get("peak_amplitude_nt", 0),
            "approximate": True,
        })
    
    print(f"  Matched to wells.csv: {len(matched_wells) - len(unmatched)}")
    print(f"  Unmatched (using tile center): {len(unmatched)}")
    
    # ── 3. Load flight-line CSV ──────────────────────────────────────
    print("\n[3] Loading GSC Erie flight-line CSV...")
    df = pd.read_csv(GSC_CSV, comment="/", index_col=False)
    print(f"  Rows: {len(df)}")
    
    lines = df.groupby("LINE")
    line_summaries = []
    for lid, grp in lines:
        line_summaries.append({
            "line_id": lid,
            "mean_lat": grp["Y"].mean(),
            "min_lon": grp["X"].min(), "max_lon": grp["X"].max(),
            "min_lat": grp["Y"].min(), "max_lat": grp["Y"].max(),
        })
    print(f"  Flight lines: {len(line_summaries)}")
    
    # ── 4. Extract profiles ──────────────────────────────────────────
    print("\n[4] Extracting profiles for each wellhead...")
    results = []
    
    for i, well in enumerate(matched_wells):
        wlat, wlon = well["lat"], well["lon"]
        
        # Find best flight line
        best_line, closest_m = find_best_line(wlat, wlon, lines, line_summaries)
        if best_line is None or closest_m > 15000:
            continue
        
        line_df = lines.get_group(best_line)
        
        # Also check 2nd-closest line (might have closer approach)
        # Sort candidates by mean distance
        cand_dists = []
        for ls in line_summaries:
            d = haversine_vec(wlat, wlon, np.array([ls["mean_lat"]]), np.array([ls["min_lon"]]))[0]
            cand_dists.append((ls["line_id"], d))
        cand_dists.sort(key=lambda x: x[1])
        
        # Check top 3
        actual_best_line = best_line
        actual_best_dist = closest_m
        
        for lid, _ in cand_dists[:3]:
            grp = lines.get_group(lid)
            dists = haversine_vec(wlat, wlon, grp["Y"].values, grp["X"].values)
            md = dists.min()
            if md < actual_best_dist:
                actual_best_dist = md
                actual_best_line = lid
        
        line_df = lines.get_group(actual_best_line)
        
        # Extract and measure
        meas = extract_and_measure(line_df, wlat, wlon, radius_m=2000)
        if meas is None:
            continue
        
        result = {
            "well_name": well["name"],
            "well_lat": wlat,
            "well_lon": wlon,
            "line_id": int(actual_best_line) if isinstance(actual_best_line, (int, np.integer)) else actual_best_line,
            "det_spatial_extent_m": well["det_spatial_extent_m"],
            "det_peak_amp_nt": well["det_peak_amp_nt"],
            "approximate_pos": well.get("approximate", False),
        }
        result.update(meas)
        results.append(result)
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(matched_wells)}...")
    
    print(f"  Extracted: {len(results)} profiles")
    
    # ── 5. Physics analysis ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHYSICS VALIDATION v2 (Raw 1D, ±2km window, local peak)")
    print("=" * 70)
    
    if not results:
        print("No results! Check data paths.")
        return
    
    hw = np.array([r["half_width_interp_m"] for r in results])
    hw_c = np.array([r["half_width_contig_m"] for r in results])
    amp = np.array([r["peak_amplitude_nt"] for r in results])
    alt = np.array([r["flight_altitude_m"] for r in results])
    offset = np.array([r["lateral_offset_m"] for r in results])
    det_ext = np.array([r["det_spatial_extent_m"] for r in results])
    det_amp = np.array([r["det_peak_amp_nt"] for r in results])
    
    slant = np.sqrt(alt**2 + offset**2)
    
    print(f"\nProfiles: {len(results)}")
    
    print(f"\n--- Raw Profile Measurements ---")
    print(f"  half_width_interp:  min={hw.min():.0f}  median={np.median(hw):.0f}  "
          f"mean={hw.mean():.0f}  max={hw.max():.0f}  std={hw.std():.0f}")
    print(f"  half_width_contig:  min={hw_c.min():.0f}  median={np.median(hw_c):.0f}  "
          f"mean={hw_c.mean():.0f}  max={hw_c.max():.0f}")
    print(f"  peak_amp_nt:        min={amp.min():.1f}  median={np.median(amp):.1f}  "
          f"mean={amp.mean():.1f}  max={amp.max():.1f}")
    print(f"  flight_alt_m:       min={alt.min():.0f}  median={np.median(alt):.0f}  max={alt.max():.0f}")
    print(f"  lateral_offset_m:   min={offset.min():.0f}  median={np.median(offset):.0f}  max={offset.max():.0f}")
    print(f"  slant_distance_m:   min={slant.min():.0f}  median={np.median(slant):.0f}  max={slant.max():.0f}")
    
    # Gridded vs raw comparison
    valid_ext = det_ext > 0
    if np.any(valid_ext):
        print(f"\n--- Gridded vs Raw ---")
        print(f"  Gridded extent:  median={np.median(det_ext[valid_ext]):.0f}m")
        print(f"  Raw half-width:  median={np.median(hw[valid_ext]):.0f}m")
        ratio = det_ext[valid_ext] / np.maximum(hw[valid_ext], 1)
        print(f"  Ratio (gridded/raw): median={np.median(ratio):.1f}x")
    
    # ── TEST 1: Half-width vs slant distance ─────────────────────────
    print(f"\n--- TEST 1: Half-Width vs Slant Distance ---")
    print(f"  Theory: half_width ≈ k × slant (k ≈ 1.0-2.5 for compact dipoles)")
    k = hw / slant
    print(f"  k = hw / slant:  min={k.min():.2f}  median={np.median(k):.2f}  "
          f"mean={k.mean():.2f}  max={k.max():.2f}  std={k.std():.2f}")
    
    in_range = np.sum((k >= 0.5) & (k <= 5.0))
    print(f"  k in [0.5, 5.0]: {in_range}/{len(k)} ({100*in_range/len(k):.0f}%)")
    
    # Separate by offset magnitude
    close = offset < 500  # close-approach lines
    mid = (offset >= 500) & (offset < 2000)
    far = offset >= 2000
    for label, mask in [("CLOSE <500m", close), ("MID 500-2000m", mid), ("FAR >2000m", far)]:
        if mask.sum() > 0:
            print(f"  {label} (n={mask.sum()}): k_median={np.median(k[mask]):.2f}, "
                  f"hw_median={np.median(hw[mask]):.0f}m, amp_median={np.median(amp[mask]):.1f}nT")
    
    # ── TEST 2: Amplitude vs distance (1/r³) ────────────────────────
    print(f"\n--- TEST 2: Amplitude vs Slant Distance ---")
    print(f"  Theory: amplitude ∝ 1/r³ (for same source)")
    print(f"  NOTE: Different wellheads have different magnetic moments!")
    if slant.std() > 10 and len(slant) > 5:
        r = np.corrcoef(slant, amp)[0, 1]
        print(f"  Correlation(slant, amplitude): r = {r:.3f}")
        # Log-log regression
        valid_amp = amp > 0
        if valid_amp.sum() > 5:
            log_slant = np.log10(slant[valid_amp])
            log_amp = np.log10(amp[valid_amp])
            slope, intercept = np.polyfit(log_slant, log_amp, 1)
            print(f"  Log-log slope: {slope:.2f} (theory: -3.0 for dipole)")
    
    # ── TEST 3: Offset estimation ────────────────────────────────────
    print(f"\n--- TEST 3: Offset Estimation ---")
    print(f"  Theory: offset ≈ sqrt(half_width² - altitude²)")
    valid = hw > alt
    print(f"  Profiles where hw > alt: {valid.sum()}/{len(hw)}")
    if valid.sum() > 3:
        est_offset = np.sqrt(hw[valid]**2 - alt[valid]**2)
        actual = offset[valid]
        residuals = est_offset - actual
        print(f"  Estimated offset:  median={np.median(est_offset):.0f}m")
        print(f"  Actual offset:     median={np.median(actual):.0f}m")
        print(f"  |Residual|:        median={np.median(np.abs(residuals)):.0f}m  MAE={np.mean(np.abs(residuals)):.0f}m")
        if actual.std() > 0:
            r = np.corrcoef(actual, est_offset)[0, 1]
            print(f"  Correlation(actual, estimated): r = {r:.3f}")
    
    # ── TEST 4: Dipole moment consistency ────────────────────────────
    print(f"\n--- TEST 4: Dipole Moment Consistency ---")
    print(f"  Theory: amp × slant³ ≈ const (for similar sources)")
    moment = amp * slant**3
    cv = moment.std() / moment.mean() if moment.mean() > 0 else float('inf')
    print(f"  amp × slant³:  median={np.median(moment):.2e}  CV={cv:.2f}")
    print(f"  (CV < 0.5 → consistent; CV > 1.0 → variable)")
    
    # ── Per-profile detail table ─────────────────────────────────────
    print(f"\n--- Per-Profile Details (sorted by offset) ---")
    sorted_results = sorted(results, key=lambda r: r["lateral_offset_m"])
    print(f"  {'Name':<40s} offset  alt  slant   hw    amp    k")
    print(f"  {'─'*40} {'─'*6} {'─'*4} {'─'*5} {'─'*6} {'─'*6} {'─'*5}")
    for r in sorted_results[:30]:
        s = np.sqrt(r["flight_altitude_m"]**2 + r["lateral_offset_m"]**2)
        k_val = r["half_width_interp_m"] / s if s > 0 else 0
        nm = r["well_name"][:40]
        print(f"  {nm:<40s} {r['lateral_offset_m']:>5.0f}  {r['flight_altitude_m']:>3.0f}  "
              f"{s:>5.0f}  {r['half_width_interp_m']:>5.0f}  {r['peak_amplitude_nt']:>5.1f}  {k_val:>4.2f}")
    
    # ── Save ─────────────────────────────────────────────────────────
    output_data = {
        "summary": {
            "n_profiles": len(results),
            "half_width_interp_median_m": float(np.median(hw)),
            "half_width_contig_median_m": float(np.median(hw_c)),
            "peak_amp_median_nt": float(np.median(amp)),
            "k_slant_median": float(np.median(k)),
            "dipole_moment_cv": float(cv),
        },
        "profiles": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {OUTPUT.name}")


if __name__ == "__main__":
    main()
