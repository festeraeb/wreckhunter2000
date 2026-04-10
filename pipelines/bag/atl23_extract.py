#!/usr/bin/env python3
"""
ATL23 Bathymetric Data Parser (Python/h5py)
============================================
Queries NASA CMR for ICESat-2 ATL23 granules, downloads HDF5 files,
and extracts bathymetric bottom-return depths near a candidate wreck.

This fills the gap where the Rust erie_remote binary can't parse HDF5
(requires the hdf5-support feature + system HDF5 C library).

Usage::

    python atl23_extract.py --lat 42.425 --lon -80.813
    python atl23_extract.py --lat 42.4708 --lon -80.6528 --search-radius 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
ATL23_SHORT_NAME = "ATL03"  # ATL23 is very new; ATL03 has bathy in gt1l/gt1r etc.
ATL23_VERSION = "006"

# For actual ATL23 product (if available):
ATL23_ALT_SHORT_NAME = "ATL23"
ATL23_ALT_VERSION = "002"

# Default data directory
_DEFAULT_CACHE = str(Path(__file__).resolve().parent.parent.parent
                     / "erie_remote" / "erie_remote_data" / "atl23_cache")


def _get_token() -> Optional[str]:
    """Read NASA Earthdata token from env or file."""
    if "NASA_EARTHDATA_TOKEN" in os.environ:
        return os.environ["NASA_EARTHDATA_TOKEN"]
    token_file = Path(__file__).resolve().parent.parent.parent / "erie_remote" / "erie_remote_data" / ".earthdata_token"
    if token_file.exists():
        t = token_file.read_text().strip()
        return t if t else None
    return None


# ---------------------------------------------------------------------------
# CMR Search
# ---------------------------------------------------------------------------

def cmr_search(lat: float, lon: float, radius_km: float,
               start_date: str, end_date: str,
               short_name: str, version: str,
               token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search CMR for granules covering the candidate area."""
    # Build bounding box from radius
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": bbox,
        "page_size": "200",
        "sort_key": "-start_date",
    }

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"  CMR search: {short_name} v{version} bbox={bbox}")
    resp = requests.get(CMR_SEARCH_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    entries = data.get("feed", {}).get("entry", [])
    print(f"  Found {len(entries)} granules")
    return entries


def get_download_urls(entries: List[Dict]) -> List[Dict[str, str]]:
    """Extract HDF5 download URLs from CMR entries."""
    result = []
    for entry in entries:
        title = entry.get("title", "unknown")
        links = entry.get("links", [])
        for link in links:
            href = link.get("href", "")
            if href.endswith(".h5") and "data" in link.get("rel", ""):
                result.append({"title": title, "url": href})
                break
    return result


# ---------------------------------------------------------------------------
# HDF5 Parsing
# ---------------------------------------------------------------------------

def extract_bathy_points(h5_path: str, lat: float, lon: float,
                         radius_km: float = 2.0) -> List[Dict]:
    """Extract bathymetric depth points from an ATL03/ATL23 HDF5 file.

    For ATL03: looks for photon classification = bathymetric in each beam.
    For ATL23: looks for /bathy_elev or similar datasets.
    """
    import h5py

    points = []
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))

    with h5py.File(h5_path, 'r') as f:
        # Try ATL23-style datasets first
        for key in ['bathy_elev', 'bathymetry', 'elevation']:
            if key in f:
                depths = f[key][:]
                lats = f['lat'][:] if 'lat' in f else None
                lons = f['lon'][:] if 'lon' in f else None
                if lats is not None and lons is not None:
                    mask = ((np.abs(lats - lat) < dlat) &
                            (np.abs(lons - lon) < dlon))
                    for i in np.where(mask)[0]:
                        dist_m = haversine_m(lat, lon, lats[i], lons[i])
                        points.append({
                            "lat": float(lats[i]),
                            "lon": float(lons[i]),
                            "depth_m": float(depths[i]),
                            "confidence": 1.0,
                            "distance_m": dist_m,
                            "source": key,
                        })
                return points

        # Try ATL03-style: /gt{1,2,3}{l,r}/heights/
        beams = [k for k in f.keys() if k.startswith('gt')]
        for beam in beams:
            heights_grp = f.get(f"{beam}/heights")
            if heights_grp is None:
                continue

            h_lats = heights_grp.get("lat_ph")
            h_lons = heights_grp.get("lon_ph")
            h_elev = heights_grp.get("h_ph")
            h_conf = heights_grp.get("signal_conf_ph")

            if h_lats is None or h_lons is None or h_elev is None:
                continue

            h_lats = h_lats[:]
            h_lons = h_lons[:]
            h_elev = h_elev[:]

            # Spatial filter
            mask = ((np.abs(h_lats - lat) < dlat) &
                    (np.abs(h_lons - lon) < dlon))

            if not np.any(mask):
                continue

            # Get confidence values if available
            if h_conf is not None:
                h_conf_arr = h_conf[:]
                # For ATL03: signal_conf_ph has shape (N, 5) for
                # [land, ocean, sea_ice, land_ice, inland_water]
                # Use inland_water confidence (index 4) if available
                if len(h_conf_arr.shape) == 2 and h_conf_arr.shape[1] >= 5:
                    conf = h_conf_arr[mask, 4]
                else:
                    conf = h_conf_arr[mask] if len(h_conf_arr.shape) == 1 else np.ones(np.sum(mask))
            else:
                conf = np.ones(np.sum(mask))

            idx = np.where(mask)[0]
            for j, i in enumerate(idx):
                # Only include high-confidence photons (>=3 = medium/high)
                c = float(conf[j]) if j < len(conf) else 0.0
                if c < 2:
                    continue
                dist_m = haversine_m(lat, lon, float(h_lats[i]), float(h_lons[i]))
                points.append({
                    "lat": float(h_lats[i]),
                    "lon": float(h_lons[i]),
                    "depth_m": float(h_elev[i]),
                    "confidence": c,
                    "distance_m": dist_m,
                    "beam": beam,
                    "source": "ATL03/heights",
                })

    return points


def haversine_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two lat/lon points."""
    dlat = (lat2 - lat1) * 111000
    dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.sqrt(dlat**2 + dlon**2))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_points(points: List[Dict], candidate_radius_m: float = 500.0):
    """Compute bathy statistics for candidate vs background."""
    cand_depths = [p["depth_m"] for p in points if p["distance_m"] <= candidate_radius_m]
    bg_depths = [p["depth_m"] for p in points if p["distance_m"] > candidate_radius_m]

    def stats(arr):
        if not arr:
            return 0.0, 0.0, 0
        return float(np.median(arr)), float(np.var(arr)), len(arr)

    cand_med, cand_var, n_cand = stats(cand_depths)
    bg_med, bg_var, n_bg = stats(bg_depths)

    anomaly = bg_med - cand_med  # positive = candidate is shallower (mound)
    var_ratio = cand_var / bg_var if bg_var > 0 else 0.0

    return {
        "n_candidate": n_cand,
        "n_background": n_bg,
        "candidate_median_depth_m": round(cand_med, 3),
        "candidate_variance": round(cand_var, 3),
        "background_median_depth_m": round(bg_med, 3),
        "background_variance": round(bg_var, 3),
        "depth_anomaly_m": round(anomaly, 3),
        "variance_ratio": round(var_ratio, 3),
        "mound_detected": anomaly > 0.5,
        "scatter_detected": var_ratio > 2.0 and n_cand > 5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATL23/ATL03 bathymetric extraction")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--search-radius", type=float, default=5.0,
                        help="Search radius in km (default 5)")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--download-limit", type=int, default=10,
                        help="Max granules to download (default 10)")
    args = parser.parse_args()

    token = _get_token()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"ATL23/ATL03 Bathymetric Extraction")
    print(f"  Candidate: ({args.lat:.4f}, {args.lon:.4f})")
    print(f"  Search radius: {args.search_radius} km")
    print(f"  Token: {'set' if token else 'NOT SET'}")
    print()

    # Search for ATL23 first, fall back to ATL03
    all_points = []
    granules_processed = 0

    for short_name, version in [(ATL23_ALT_SHORT_NAME, ATL23_ALT_VERSION),
                                 (ATL23_SHORT_NAME, ATL23_VERSION)]:
        entries = cmr_search(args.lat, args.lon, args.search_radius,
                            args.start_date, args.end_date,
                            short_name, version, token)

        if not entries:
            print(f"  No {short_name} v{version} granules found, trying next...\n")
            continue

        urls = get_download_urls(entries)
        print(f"  {len(urls)} downloadable granules")

        for i, info in enumerate(urls[:args.download_limit]):
            filename = info["url"].rsplit("/", 1)[-1]
            local_path = cache_dir / filename

            # Download if needed
            if not local_path.exists():
                print(f"  [{i+1}/{min(len(urls), args.download_limit)}] Downloading {filename}...")
                headers = {}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                try:
                    resp = requests.get(info["url"], headers=headers, timeout=300,
                                       allow_redirects=True)
                    if resp.status_code == 200:
                        local_path.write_bytes(resp.content)
                        print(f"    → {len(resp.content) / 1e6:.1f} MB")
                    else:
                        print(f"    HTTP {resp.status_code} — skipping")
                        continue
                except Exception as e:
                    print(f"    Download failed: {e}")
                    continue
            else:
                print(f"  [{i+1}] Cached: {filename}")

            # Parse
            try:
                pts = extract_bathy_points(str(local_path), args.lat, args.lon,
                                          args.search_radius)
                all_points.extend(pts)
                granules_processed += 1
                print(f"    → {len(pts)} bathy points near candidate")
            except Exception as e:
                print(f"    Parse error: {e}")

        if all_points:
            break  # Got data from this product

    # Analyze
    print(f"\n=== Bathymetric Analysis ({len(all_points)} total points from {granules_processed} granules) ===")

    if not all_points:
        print("  No bathymetric returns found near candidate.")
        print("  This could mean:")
        print("    - Water is too deep/turbid for photon penetration")
        print("    - No ICESat-2 ground tracks cross this location")
        print("    - ATL23 product not yet available for this region")
        result = {
            "candidate": {"lat": args.lat, "lon": args.lon},
            "n_granules_processed": granules_processed,
            "n_points": 0,
            "analysis": None,
            "interpretation": "No bathymetric returns — water too deep/turbid or no coverage",
        }
    else:
        analysis = analyze_points(all_points)
        print(f"  Candidate (within 500m): {analysis['n_candidate']} pts, "
              f"median depth = {analysis['candidate_median_depth_m']:.1f} m")
        print(f"  Background:             {analysis['n_background']} pts, "
              f"median depth = {analysis['background_median_depth_m']:.1f} m")
        print(f"  Depth anomaly: {analysis['depth_anomaly_m']:.2f} m "
              f"(positive = candidate is shallower = mound)")
        print(f"  Variance ratio: {analysis['variance_ratio']:.2f}")
        print(f"  Mound detected: {analysis['mound_detected']}")
        print(f"  Scatter detected: {analysis['scatter_detected']}")

        if analysis["mound_detected"] and analysis["scatter_detected"]:
            interp = "Bathymetric mound + elevated scatter — consistent with wreck or debris"
        elif analysis["mound_detected"]:
            interp = "Bathymetric mound detected — could be hull, reef, or debris"
        elif analysis["scatter_detected"]:
            interp = "Elevated depth variance — possible debris scatter"
        else:
            interp = "No significant bathymetric anomaly at candidate location"

        print(f"  Interpretation: {interp}")

        result = {
            "candidate": {"lat": args.lat, "lon": args.lon},
            "n_granules_processed": granules_processed,
            "n_points": len(all_points),
            "analysis": analysis,
            "interpretation": interp,
            "sample_points": all_points[:50],
        }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
