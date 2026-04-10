"""Deep dipole analysis on a single candidate point.

Extracts a local patch around the candidate, measures:
- Dipole signature (positive/negative lobe separation and ratio)
- Polarity flip distance (how quickly field reverses = man-made vs geological)
- Floor statistics (background level in the surrounding annulus)
- Gradient sharpness profile
- Aspect ratio (elongated = geological ridge, compact = isolated object)

Usage: python scripts/dipole_analysis.py --lat 42.4250 --lon -80.8130
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
GRIDS = REPO / "magnetic_data" / "grids"

METER_PER_DEG_LAT = 111_320.0

# Window sizes for analysis
INNER_YD = 2000    # core anomaly region (yards)
OUTER_YD = 5000    # background annulus extends to this
YARDS_PER_METER = 1.0936133


def _meters_per_pixel(src):
    res_x_deg = abs(src.transform.a)
    res_y_deg = abs(src.transform.e)
    center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    m_per_deg_lon = METER_PER_DEG_LAT * math.cos(math.radians(center_lat))
    return res_x_deg * m_per_deg_lon, res_y_deg * METER_PER_DEG_LAT


def _px_radius(yards, mx, my):
    m = yards / YARDS_PER_METER
    return max(3, int(round(m / ((mx + my) / 2))))


def analyze_candidate(tif_path: Path, lat: float, lon: float,
                      inner_yd: float = INNER_YD,
                      outer_yd: float = OUTER_YD) -> dict:
    with rasterio.open(str(tif_path)) as src:
        arr = src.read(1).astype("float64")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan

        # Row/col of candidate
        row, col = rowcol(src.transform, lon, lat)
        row, col = int(row), int(col)
        h, w = arr.shape
        if not (0 <= row < h and 0 <= col < w):
            return {"error": f"Point ({lat}, {lon}) outside raster bounds {src.bounds}"}

        mx, my = _meters_per_pixel(src)
        r_inner = _px_radius(inner_yd, mx, my)
        r_outer = _px_radius(outer_yd, mx, my)

        # Build distance grid from candidate center
        rows_g, cols_g = np.ogrid[:h, :w]
        dist_px = np.sqrt(((rows_g - row) ** 2 + (cols_g - col) ** 2))
        dist_m = dist_px * (mx + my) / 2

        # ── Extract analysis zones ────────────────────────────────────────────
        inner_mask = dist_px <= r_inner
        annulus_mask = (dist_px > r_inner) & (dist_px <= r_outer)

        inner_vals = arr[inner_mask & ~np.isnan(arr)]
        annulus_vals = arr[annulus_mask & ~np.isnan(arr)]

        if len(inner_vals) == 0 or len(annulus_vals) == 0:
            return {"error": "Insufficient valid cells in analysis zone"}

        # ── Background stats ──────────────────────────────────────────────────
        bg_mean = float(np.mean(annulus_vals))
        bg_std  = float(np.std(annulus_vals))
        bg_median = float(np.median(annulus_vals))
        bg_min = float(np.min(annulus_vals))
        bg_max = float(np.max(annulus_vals))

        # Anomaly relative to background
        inner_detrended = inner_vals - bg_mean
        peak_pos = float(np.max(inner_detrended))
        peak_neg = float(np.min(inner_detrended))
        peak_abs = float(np.max(np.abs(inner_detrended)))

        # ── Dipole detection ─────────────────────────────────────────────────
        # A dipole has both a + and - lobe. Ratio of |min|/|max| close to 1 => symmetric dipole.
        # Geological features tend to have asymmetric or one-sided signatures.
        detrended_inner_grid = arr.copy()
        detrended_inner_grid -= bg_mean  # rough global detrend

        pos_mask = inner_mask & (detrended_inner_grid > 0.15 * peak_abs)
        neg_mask = inner_mask & (detrended_inner_grid < -0.15 * peak_abs)

        has_pos = bool(pos_mask.any())
        has_neg = bool(neg_mask.any())
        is_dipolar = has_pos and has_neg

        # Separation between positive and negative centroids
        pos_sep_m = None
        neg_sep_m = None
        dipole_separation_m = None
        dipole_azimuth_deg = None
        if is_dipolar:
            pr, pc = np.argwhere(pos_mask).mean(axis=0)
            nr, nc = np.argwhere(neg_mask).mean(axis=0)
            sep_px = math.hypot(pr - nr, pc - nc)
            dipole_separation_m = sep_px * (mx + my) / 2
            # azimuth (degrees from north, clockwise)
            drow = nr - pr  # positive = south
            dcol = nc - pc  # positive = east
            dipole_azimuth_deg = float(math.degrees(math.atan2(dcol * mx, -drow * my)) % 360)

        lobe_ratio = None
        if peak_pos > 0 and abs(peak_neg) > 0:
            lobe_ratio = float(min(peak_pos, abs(peak_neg)) / max(peak_pos, abs(peak_neg)))

        # ── Polarity flip distance ────────────────────────────────────────────
        # How far (in meters) from the peak before the field crosses zero?
        # Fast flip = sharp edge = man-made. Slow flip = geological gradient.
        # Walk radially outward from peak cell in 8 directions.
        peak_row, peak_col = None, None
        peak_search = detrended_inner_grid.copy()
        peak_search[~inner_mask] = 0
        best_abs = 0
        for r2 in range(max(0, row - r_inner), min(h, row + r_inner + 1)):
            for c2 in range(max(0, col - r_inner), min(w, col + r_inner + 1)):
                v = abs(detrended_inner_grid[r2, c2])
                if v > best_abs and not np.isnan(arr[r2, c2]):
                    best_abs = v
                    peak_row, peak_col = r2, c2

        flip_distances_m = []
        if peak_row is not None:
            peak_sign = np.sign(detrended_inner_grid[peak_row, peak_col])
            directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            for dr, dc in directions:
                for step in range(1, r_outer + 1):
                    r2 = peak_row + dr * step
                    c2 = peak_col + dc * step
                    if not (0 <= r2 < h and 0 <= c2 < w):
                        break
                    if np.isnan(arr[r2, c2]):
                        continue
                    v = detrended_inner_grid[r2, c2]
                    if np.sign(v) != peak_sign and abs(v) > 0.05 * peak_abs:
                        dist_step = math.hypot(dr * step, dc * step) * (mx + my) / 2
                        flip_distances_m.append(dist_step)
                        break

        flip_dist_min_m = float(np.min(flip_distances_m)) if flip_distances_m else None
        flip_dist_mean_m = float(np.mean(flip_distances_m)) if flip_distances_m else None

        # ── Spatial gradient sharpness ──────────────────────────────────────
        # Fill NaN with local mean before gradient to avoid NaN propagation
        arr_filled = arr.copy()
        nan_mask = np.isnan(arr_filled)
        if nan_mask.any():
            arr_filled[nan_mask] = float(np.nanmean(arr_filled))
        gy, gx = np.gradient(arr_filled)
        grad_mag = np.hypot(gx / mx, gy / my)  # nT/m
        inner_grad = grad_mag[inner_mask & ~np.isnan(arr)]
        grad_peak = float(np.max(inner_grad)) if len(inner_grad) > 0 else None
        grad_mean = float(np.mean(inner_grad)) if len(inner_grad) > 0 else None
        annulus_grad = grad_mag[annulus_mask & ~np.isnan(arr)]
        bg_grad_mean = float(np.mean(annulus_grad)) if len(annulus_grad) > 0 else None

        # Gradient contrast: how much sharper is the anomaly vs background
        grad_contrast = (grad_peak / bg_grad_mean) if (grad_peak and bg_grad_mean and bg_grad_mean > 0) else None

        # ── Aspect ratio (elongation) ─────────────────────────────────────────
        # Use PCA on the positive lobe pixels
        sig_mask = inner_mask & (np.abs(detrended_inner_grid) > 0.25 * peak_abs) & ~np.isnan(arr)
        aspect_ratio = None
        elongation_azimuth = None
        if sig_mask.sum() >= 4:
            pts = np.argwhere(sig_mask).astype(float)
            pts[:, 0] *= my  # row → meters
            pts[:, 1] *= mx  # col → meters
            pts -= pts.mean(axis=0)
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals.min() > 0:
                aspect_ratio = float(np.sqrt(eigvals.max() / eigvals.min()))
                major_vec = eigvecs[:, np.argmax(eigvals)]
                elongation_azimuth = float(math.degrees(math.atan2(major_vec[1], major_vec[0])) % 180)

        # ── Classification heuristic ─────────────────────────────────────────
        score_manmade = 0.0
        reasons = []

        if is_dipolar:
            score_manmade += 25
            reasons.append("+25 dipolar signature (+ and - lobes present)")
        else:
            reasons.append(" 0 no clear dipole (single-polarity anomaly)")

        if lobe_ratio is not None and lobe_ratio > 0.3:
            score_manmade += 15 * lobe_ratio
            reasons.append(f"+{15*lobe_ratio:.0f} lobe symmetry ratio={lobe_ratio:.2f}")

        if dipole_separation_m is not None and dipole_separation_m < 3000:
            score_manmade += 20
            reasons.append(f"+20 tight dipole separation={dipole_separation_m:.0f}m (<3km)")
        elif dipole_separation_m is not None:
            reasons.append(f" 0 wide dipole sep={dipole_separation_m:.0f}m (geological-scale)")

        if flip_dist_min_m is not None:
            flip_km = flip_dist_min_m / 1000
            if flip_km < 2.0:
                pts_f = 20 * (1 - flip_km / 2.0)
                score_manmade += pts_f
                reasons.append(f"+{pts_f:.0f} fast polarity flip at {flip_dist_min_m:.0f}m (<2km)")
            else:
                reasons.append(f" 0 slow flip at {flip_dist_min_m/1000:.1f}km (geological)")
        else:
            reasons.append(" ? flip distance: no zero-crossing found (one-sided anomaly)")

        if grad_contrast is not None:
            if grad_contrast > 3:
                score_manmade += 20
                reasons.append(f"+20 gradient contrast={grad_contrast:.1f}x (very sharp edges)")
            elif grad_contrast > 1.5:
                score_manmade += 10
                reasons.append(f"+10 gradient contrast={grad_contrast:.1f}x (moderately sharp)")
            else:
                reasons.append(f" 0 gradient contrast={grad_contrast:.1f}x (smooth, geological-like)")

        if aspect_ratio is not None:
            if aspect_ratio < 3:
                score_manmade += 10
                reasons.append(f"+10 compact shape (aspect ratio={aspect_ratio:.1f}x)")
            elif aspect_ratio < 6:
                reasons.append(f" 0 moderately elongated (aspect={aspect_ratio:.1f}x)")
            else:
                reasons.append(f"-10 very elongated (aspect={aspect_ratio:.1f}x, likely ridge/fault)")
                score_manmade -= 10

        if score_manmade >= 60:
            classification = "LIKELY MAN-MADE (strong)"
        elif score_manmade >= 40:
            classification = "POSSIBLY MAN-MADE (moderate)"
        elif score_manmade >= 20:
            classification = "AMBIGUOUS"
        else:
            classification = "LIKELY GEOLOGICAL"

        return {
            "input": {"lat": lat, "lon": lon, "tif": tif_path.name},
            "background": {
                "mean_nT": round(bg_mean, 2),
                "std_nT": round(bg_std, 2),
                "median_nT": round(bg_median, 2),
                "min_nT": round(bg_min, 2),
                "max_nT": round(bg_max, 2),
                "annulus_inner_yd": inner_yd,
                "annulus_outer_yd": outer_yd,
            },
            "anomaly": {
                "peak_above_bg_nT": round(peak_pos, 2),
                "trough_below_bg_nT": round(peak_neg, 2),
                "peak_abs_nT": round(peak_abs, 2),
                "peak_total_nT": round(peak_pos + bg_mean, 2),
                "snr_vs_bg_std": round(peak_abs / bg_std, 2) if bg_std > 0 else None,
            },
            "dipole": {
                "is_dipolar": is_dipolar,
                "has_positive_lobe": has_pos,
                "has_negative_lobe": has_neg,
                "lobe_symmetry_ratio": round(lobe_ratio, 3) if lobe_ratio else None,
                "separation_m": round(dipole_separation_m) if dipole_separation_m else None,
                "azimuth_deg": round(dipole_azimuth_deg, 1) if dipole_azimuth_deg else None,
            },
            "flip": {
                "min_flip_distance_m": round(flip_dist_min_m) if flip_dist_min_m else None,
                "mean_flip_distance_m": round(flip_dist_mean_m) if flip_dist_mean_m else None,
                "directions_measured": len(flip_distances_m),
            },
            "gradient": {
                "peak_nT_per_m": round(grad_peak, 4) if grad_peak else None,
                "mean_nT_per_m": round(grad_mean, 4) if grad_mean else None,
                "bg_nT_per_m": round(bg_grad_mean, 4) if bg_grad_mean else None,
                "contrast_ratio": round(grad_contrast, 2) if grad_contrast else None,
            },
            "shape": {
                "aspect_ratio": round(aspect_ratio, 2) if aspect_ratio else None,
                "elongation_azimuth_deg": round(elongation_azimuth, 1) if elongation_azimuth else None,
                "size_reported_m": "3781x1274",
            },
            "classification": {
                "score_manmade_pct": round(score_manmade, 1),
                "verdict": classification,
                "reasons": reasons,
            },
        }


def find_best_tif(lat: float, lon: float) -> Path:
    """Find the tightest-bbox TIF that contains the point, preferring USGS USmag."""
    candidates = []
    for tif in sorted(GRIDS.glob("*.tif")):
        try:
            with rasterio.open(str(tif)) as src:
                b = src.bounds
                if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                    area = (b.right - b.left) * (b.top - b.bottom)
                    # prefer usgs_usmag for resolution
                    pref = 0 if "usmag" in tif.name else 1
                    candidates.append((pref, area, tif))
        except Exception:
            pass
    if not candidates:
        raise RuntimeError(f"No TIF covers ({lat}, {lon})")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def main():
    p = argparse.ArgumentParser(description="Deep dipole analysis on a candidate point")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tif", type=str, default=None, help="Specific TIF filename (auto-detected if omitted)")
    p.add_argument("--inner-yd", type=float, default=INNER_YD)
    p.add_argument("--outer-yd", type=float, default=OUTER_YD)
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    args = p.parse_args()

    if args.tif:
        tif = GRIDS / args.tif
    else:
        tif = find_best_tif(args.lat, args.lon)
        print(f"Auto-selected TIF: {tif.name}\n")

    result = analyze_candidate(tif, args.lat, args.lon,
                               inner_yd=args.inner_yd, outer_yd=args.outer_yd)

    if args.json or "error" in result:
        print(json.dumps(result, indent=2))
        return

    # Human-readable report
    print("=" * 60)
    print(f"  DIPOLE ANALYSIS  ({result['input']['lat']}, {result['input']['lon']})")
    print(f"  Source: {result['input']['tif']}")
    print("=" * 60)

    bg = result["background"]
    an = result["anomaly"]
    di = result["dipole"]
    fl = result["flip"]
    gr = result["gradient"]
    sh = result["shape"]
    cl = result["classification"]

    print(f"\n── BACKGROUND (floor) ──────────────────────────────────")
    print(f"  Mean (annulus {bg['annulus_inner_yd']}–{bg['annulus_outer_yd']}yd): {bg['mean_nT']:+.1f} nT")
    print(f"  Median: {bg['median_nT']:+.1f} nT   Std: {bg['std_nT']:.1f} nT")
    print(f"  Range:  {bg['min_nT']:+.1f} to {bg['max_nT']:+.1f} nT")

    print(f"\n── ANOMALY AMPLITUDE ───────────────────────────────────")
    print(f"  Peak above background:  {an['peak_above_bg_nT']:+.1f} nT")
    print(f"  Trough below background:{an['trough_below_bg_nT']:+.1f} nT")
    print(f"  peak_abs (detrended):    {an['peak_abs_nT']:.1f} nT")
    print(f"  Total field at peak:     {an['peak_total_nT']:.1f} nT")
    print(f"  SNR vs bg std:           {an['snr_vs_bg_std']:.1f}x")

    print(f"\n── DIPOLE CHARACTER ────────────────────────────────────")
    print(f"  Dipolar?:          {'YES ✓' if di['is_dipolar'] else 'NO — single-polarity'}")
    print(f"  Positive lobe:     {'present' if di['has_positive_lobe'] else 'absent'}")
    print(f"  Negative lobe:     {'present' if di['has_negative_lobe'] else 'absent'}")
    if di['lobe_symmetry_ratio'] is not None:
        sym = di['lobe_symmetry_ratio']
        sym_label = "symmetric (wreck-like)" if sym > 0.5 else ("asymmetric" if sym > 0.2 else "very asymmetric (geological)")
        print(f"  Lobe symmetry:     {sym:.2f}  → {sym_label}")
    if di['separation_m']:
        sep_label = "compact (man-made scale)" if di['separation_m'] < 3000 else "wide (geological scale)"
        print(f"  Lobe separation:   {di['separation_m']:,} m  → {sep_label}")
    if di['azimuth_deg'] is not None:
        print(f"  Dipole azimuth:    {di['azimuth_deg']:.0f}°  (negative lobe direction from positive)")

    print(f"\n── POLARITY FLIP ───────────────────────────────────────")
    if fl['min_flip_distance_m']:
        print(f"  Min flip distance: {fl['min_flip_distance_m']:,} m  ({fl['min_flip_distance_m']/1000:.2f} km)")
        print(f"  Mean flip dist:    {fl['mean_flip_distance_m']:,} m")
        print(f"  Measured in:       {fl['directions_measured']} directions")
        quick = fl['min_flip_distance_m'] < 2000
        print(f"  Assessment:        {'QUICK FLIP — sharp, wreck-like edge' if quick else 'SLOW FLIP — geological gradient'}")
    else:
        print(f"  No zero-crossing found (one-sided anomaly — no flip)")

    print(f"\n── GRADIENT SHARPNESS ──────────────────────────────────")
    if gr['peak_nT_per_m'] and not math.isnan(gr['peak_nT_per_m']):
        print(f"  Peak gradient:  {gr['peak_nT_per_m']:.4f} nT/m")
        print(f"  Mean gradient:  {gr['mean_nT_per_m']:.4f} nT/m")
        print(f"  BG gradient:    {gr['bg_nT_per_m']:.4f} nT/m")
        crat = gr['contrast_ratio']
        print(f"  Contrast ratio: {crat:.1f}x vs background" if crat else "  Contrast ratio: n/a")
    else:
        print(f"  Gradient: n/a (NaN in source data — raster resolution limited)")

    print(f"\n── SHAPE ────────────────────────────────────────────────")
    if sh['aspect_ratio']:
        ar = sh['aspect_ratio']
        shape_label = ("compact/round" if ar < 2 else ("elongated ~2x" if ar < 4 else f"strongly elongated {ar:.1f}x (ridge/fault?)"))
        print(f"  Aspect ratio:   {ar:.1f}x  → {shape_label}")
    if sh['elongation_azimuth_deg'] is not None:
        print(f"  Long axis:      {sh['elongation_azimuth_deg']:.0f}°")
    print(f"  Reported size:  {sh['size_reported_m']}")

    print(f"\n── VERDICT ──────────────────────────────────────────────")
    print(f"  Man-made score: {cl['score_manmade_pct']:.0f}/100")
    print(f"  >>> {cl['verdict']} <<<")
    print(f"\n  Scoring breakdown:")
    for r in cl["reasons"]:
        print(f"    {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
