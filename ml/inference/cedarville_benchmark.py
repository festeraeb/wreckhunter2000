"""Cedarville benchmark harness.

Attempts to fetch Sentinel-2 data for Cedarville (45.6586, -84.3486).
If data access is unavailable, falls back to a synthetic test to exercise the
SDB pipeline and compare against NOAA chart depth (~30 m).
"""
from typing import Tuple
import sys
import os
import numpy as np
from datetime import datetime

# Ensure workspace root is on sys.path so `scripts` package imports work when
# running this file as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.forensic.get_temporal_strategy import get_temporal_strategy
from scripts.forensic.sdb import compute_sdb_from_bands, detect_floor_jump


CEDARVILLE_COORD = (45.6586, -84.3486)
NOAA_CHART_DEPTH_M = 30.0


def try_download_sentinel2_tile(lat: float, lon: float, date: str = None):
    """Placeholder: attempt to download Sentinel-2 bands (blue & green).

    Returns (blue_array, green_array, profile) on success, or None on fail.
    """
    try:
        import importlib.util
        fetcher_path = os.path.abspath(os.path.join(ROOT, 'recovered', 'sentinel_fetch_and_preprocess.py'))
        if not os.path.exists(fetcher_path):
            print('Fetcher module missing at', fetcher_path)
            return None
        spec = importlib.util.spec_from_file_location('recovered_sentinel_fetch_and_preprocess', fetcher_path)
        fetcher_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetcher_module)
        fetch_for_wreck = fetcher_module.fetch_for_wreck

        out_file = fetch_for_wreck('cedarville_benchmark', lat, lon)
        if out_file is None:
            return None

        import numpy as _np
        arr = _np.load(out_file)
        blue = arr[2].astype(_np.float32)
        green = arr[1].astype(_np.float32)
        red_edge = arr[4].astype(_np.float32)  # deep ratio proxy channel
        coastal_aerosol = arr[5].astype(_np.float32)  # mussel index proxy channel
        profile = None
        return blue, green, red_edge, coastal_aerosol, profile
    except Exception as e:
        print('Error in try_download_sentinel2_tile:', e)
        import traceback; traceback.print_exc()
        return None


def synthetic_sdb_test(shape: Tuple[int, int] = (200, 200)):
    """Generate synthetic blue/green arrays representing ~30m depth with a
    localized 'wreck' anomaly (floor jump).
    """
    ny, nx = shape
    # Base reflectances: deeper water -> lower blue/green ratio
    base_blue = 0.03
    base_green = 0.025
    blue = np.full((ny, nx), base_blue)
    green = np.full((ny, nx), base_green)

    # Inject a wreck-sized rectangle (approx 60x120 pixels) in center with
    # altered reflectances simulating shallower water (higher blue/green)
    cy, cx = ny // 2, nx // 2
    hy, hx = 30, 60
    blue[cy - hy:cy + hy, cx - hx:cx + hx] = base_blue * 1.5
    green[cy - hy:cy + hy, cx - hx:cx + hx] = base_green * 1.1
    return blue, green


def run_cedarville_benchmark():
    print("Cedarville benchmark start")
    ts = get_temporal_strategy(None)
    print("Temporal strategy:", ts)

    lat, lon = CEDARVILLE_COORD
    download_result = try_download_sentinel2_tile(lat, lon)

    if download_result is None:
        print("Sentinel-2 access unavailable or not implemented; using synthetic fallback.")
        blue, green = synthetic_sdb_test()
        red_edge, coastal_aerosol = None, None
    else:
        blue, green, red_edge, coastal_aerosol, profile = download_result

    depth = compute_sdb_from_bands(blue, green)

    deep_water_depth = None
    mussel_index = None
    if red_edge is not None and coastal_aerosol is not None and not (np.isnan(red_edge).all() or np.isnan(coastal_aerosol).all()):
        from scripts.forensic.sdb import compute_deep_water_spectral_shift, compute_mussel_index
        deep_water_depth = compute_deep_water_spectral_shift(red_edge, coastal_aerosol)
        mussel_index = compute_mussel_index(coastal_aerosol, red_edge)

    # Simple summary statistics near center
    cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
    center_window = depth[cy - 40:cy + 40, cx - 80:cx + 80]
    median_depth = float(np.nanmedian(center_window))
    min_depth = float(np.nanmin(center_window))
    max_depth = float(np.nanmax(center_window))

    print(f"Cedarville window median depth (m): {median_depth:.2f}")
    print(f"Cedarville window min/max depth (m): {min_depth:.2f} / {max_depth:.2f}")

    flagged = detect_floor_jump(center_window, threshold_m=1.5)
    if flagged:
        print("Result: Cedarville flagged as Structural Deviation Anomaly (floor jump detected).")
    else:
        print("Result: No significant floor jump detected at threshold.")

    # Compare to NOAA chart depth
    diff = median_depth - NOAA_CHART_DEPTH_M
    print(f"NOAA chart depth: {NOAA_CHART_DEPTH_M} m; SDB median difference: {diff:.2f} m")

    if deep_water_depth is not None:
        deep_med = float(np.nanmedian(deep_water_depth))
        print(f"Deep spectral shift depth proxy median: {deep_med:.2f} m ({deep_med*3.28084:.2f} ft)")
        mussel_med = float(np.nanmedian(mussel_index))
        print(f"Mussel index median: {mussel_med:.4f}")

    return {
        "median_depth": median_depth,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "flagged": bool(flagged),
        "noaa_depth": NOAA_CHART_DEPTH_M,
        "difference": diff,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    result = run_cedarville_benchmark()
    print("Benchmark result object:", result)
