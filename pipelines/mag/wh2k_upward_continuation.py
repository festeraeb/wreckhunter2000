"""
WreckHunter 2000 — Upward Continuation & Satellite Proof Module
=================================================================
Mathematically simulates what an aeromagnetic anomaly looks like at satellite
altitude (400 km) and compares the result to real ESA Swarm / EMAG2 data.

THE TEST:
  1. Take a known large steel freighter from Tier 1 (Aero) data.
  2. Run upward continuation to 400 km.
  3. Load real Tier 4 (Swarm/EMAG2) at same coordinates.
  4. Compare: does the real satellite data show a bump where the simulation says?

If real satellite data shows even 0.5 nT where the simulation predicts →
satellite detection is viable for 700ft+ steel.

Also provides the upward continuation filter used by the synthetic tile
generator to tag targets as SAT_VISIBLE.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Upward Continuation (Fourier Domain) ───────────────────────────────────

def upward_continue(
    grid: np.ndarray,
    continuation_height_m: float,
    cell_size_m: float,
) -> np.ndarray:
    """Upward continue a 2D potential field grid.

    The classic Fourier-domain upward continuation:
      F_continued(k) = F_original(k) · exp(-2π|k|·Δz)

    This attenuates high-frequency (shallow/small) sources exponentially
    while preserving low-frequency (deep/large) sources.

    Parameters:
        grid: 2D array of magnetic anomaly values (nT)
        continuation_height_m: height to continue upward (metres).
            For satellite: 400,000 m.  For high-altitude aero: 1,000-5,000 m.
        cell_size_m: grid cell dimension in metres.
    
    Returns:
        Upward-continued grid (same shape, nT).
    """
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny, d=cell_size_m).reshape(-1, 1)
    kx = np.fft.fftfreq(nx, d=cell_size_m).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)

    # Upward continuation operator
    uc_filter = np.exp(-2 * np.pi * k_mag * continuation_height_m)

    return np.real(np.fft.ifft2(fft * uc_filter))


def downward_continue(
    grid: np.ndarray,
    continuation_depth_m: float,
    cell_size_m: float,
    max_amplification: float = 100.0,
) -> np.ndarray:
    """Downward continuation (inverse of upward) — enhances shallow sources.

    WARNING: Downward continuation amplifies noise. Use with caution.
    A max_amplification cap prevents numerical explosion.
    """
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny, d=cell_size_m).reshape(-1, 1)
    kx = np.fft.fftfreq(nx, d=cell_size_m).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)

    # Downward = inverse of upward = exp(+2π|k|Δz), capped
    dc_filter = np.exp(2 * np.pi * k_mag * continuation_depth_m)
    dc_filter = np.minimum(dc_filter, max_amplification)

    return np.real(np.fft.ifft2(fft * dc_filter))


# ── Satellite Proof Test ───────────────────────────────────────────────────

@dataclass
class SatelliteProofResult:
    """Result of comparing simulated satellite signal to real satellite data."""
    wreck_name: str
    lat: float
    lon: float
    # Simulation results
    aero_peak_nt: float              # Peak from real Tier 1 aero data
    simulated_sat_peak_nt: float     # After upward continuation to 400km
    # Real satellite data
    real_sat_value_nt: float         # Value in Tier 4 at same location
    real_sat_background_nt: float    # Background level (median in annulus)
    real_sat_anomaly_nt: float       # real_sat_value - background
    # Verdict
    sim_predicts_visible: bool       # Simulated peak > threshold
    real_shows_bump: bool            # Real anomaly > threshold
    correlation_confirmed: bool      # Both agree
    sat_detection_viable: bool       # Real data confirms the simulation
    # Details
    continuation_height_m: float
    threshold_nt: float


def run_satellite_proof(
    aero_grid: np.ndarray,
    aero_cell_size_m: float,
    satellite_grid: np.ndarray,
    satellite_cell_size_m: float,
    target_row_col_aero: tuple[int, int],
    target_row_col_sat: tuple[int, int],
    wreck_name: str = "Unknown",
    lat: float = 0.0,
    lon: float = 0.0,
    continuation_height_m: float = 400_000.0,
    threshold_nt: float = 0.5,
    annulus_radius_px: int = 10,
) -> SatelliteProofResult:
    """Run the full satellite proof test for a single known wreck.

    1. Extract the aero peak at the wreck location.
    2. Upward-continue the entire aero grid to satellite altitude.
    3. Read the simulated satellite value at the wreck location.
    4. Read the real satellite value at the same (projected) location.
    5. Compare: does real satellite data show the predicted bump?

    Parameters:
        aero_grid: Tier 1 aeromagnetic grid (nT)
        aero_cell_size_m: cell size of aero grid (metres)
        satellite_grid: Tier 4 satellite grid (nT) — EMAG2/Swarm/WDMAM
        satellite_cell_size_m: cell size of satellite grid (metres)
        target_row_col_aero: (row, col) of wreck in aero_grid
        target_row_col_sat: (row, col) of wreck in satellite_grid
        wreck_name: name for reporting
        lat, lon: coordinates for reporting
        continuation_height_m: altitude to continue to (default 400 km)
        threshold_nt: minimum anomaly to count as "visible" (default 0.5 nT)
        annulus_radius_px: radius for background estimation in satellite grid
    """
    r_aero, c_aero = target_row_col_aero
    r_sat, c_sat = target_row_col_sat
    h, w = satellite_grid.shape

    # 1. Aero peak at wreck location (use a small window)
    win = 5
    aero_patch = aero_grid[
        max(0, r_aero - win): r_aero + win + 1,
        max(0, c_aero - win): c_aero + win + 1,
    ]
    aero_peak = float(np.max(np.abs(aero_patch)))

    # 2. Upward continue
    continued = upward_continue(aero_grid, continuation_height_m, aero_cell_size_m)
    sim_sat_peak = float(np.max(np.abs(
        continued[
            max(0, r_aero - win): r_aero + win + 1,
            max(0, c_aero - win): c_aero + win + 1,
        ]
    )))

    # 3. Real satellite value
    real_sat_value = float(satellite_grid[
        np.clip(r_sat, 0, h - 1),
        np.clip(c_sat, 0, w - 1),
    ])

    # 4. Background from annulus around the target in satellite grid
    rows_g, cols_g = np.ogrid[:h, :w]
    dist = np.sqrt((rows_g - r_sat) ** 2 + (cols_g - c_sat) ** 2)
    annulus = (dist > annulus_radius_px) & (dist < annulus_radius_px * 3)
    valid_annulus = satellite_grid[annulus]
    valid_annulus = valid_annulus[~np.isnan(valid_annulus)]
    background = float(np.median(valid_annulus)) if len(valid_annulus) > 0 else 0.0

    real_anomaly = real_sat_value - background

    # 5. Verdicts
    sim_predicts = sim_sat_peak >= threshold_nt
    real_bump = abs(real_anomaly) >= threshold_nt
    correlation = sim_predicts and real_bump
    viable = correlation  # Both simulation and reality agree

    result = SatelliteProofResult(
        wreck_name=wreck_name,
        lat=lat,
        lon=lon,
        aero_peak_nt=aero_peak,
        simulated_sat_peak_nt=sim_sat_peak,
        real_sat_value_nt=real_sat_value,
        real_sat_background_nt=background,
        real_sat_anomaly_nt=real_anomaly,
        sim_predicts_visible=sim_predicts,
        real_shows_bump=real_bump,
        correlation_confirmed=correlation,
        sat_detection_viable=viable,
        continuation_height_m=continuation_height_m,
        threshold_nt=threshold_nt,
    )

    logger.info(
        "Satellite Proof [%s]: aero=%.1f nT → sim_sat=%.4f nT | "
        "real_sat=%.4f nT (bg=%.4f, anomaly=%.4f) | viable=%s",
        wreck_name, aero_peak, sim_sat_peak,
        real_sat_value, background, real_anomaly,
        "YES" if viable else "NO",
    )

    return result


# ── Batch Satellite Proof (All Known Large Steel Wrecks) ──────────────────

def batch_satellite_proof(
    aero_tif_path: str | Path,
    satellite_tif_path: str | Path,
    wreck_locations: list[dict],
    continuation_height_m: float = 400_000.0,
    threshold_nt: float = 0.5,
) -> list[dict]:
    """Run satellite proof on multiple known wrecks.

    Requires rasterio for GeoTIFF I/O.

    wreck_locations: list of dicts with keys:
        name, lat, lon (and optionally length_ft, material)
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        logger.error("rasterio is required for batch_satellite_proof")
        return []

    results = []

    with rasterio.open(str(aero_tif_path)) as aero_src, \
         rasterio.open(str(satellite_tif_path)) as sat_src:

        aero_grid = aero_src.read(1).astype(np.float64)
        sat_grid = sat_src.read(1).astype(np.float64)

        # Handle nodata
        if aero_src.nodata is not None:
            aero_grid[aero_grid == aero_src.nodata] = np.nan
        if sat_src.nodata is not None:
            sat_grid[sat_grid == sat_src.nodata] = np.nan

        # Cell sizes in metres (approximate)
        aero_res_deg = abs(aero_src.transform.a)
        sat_res_deg = abs(sat_src.transform.a)
        center_lat = (aero_src.bounds.top + aero_src.bounds.bottom) / 2
        m_per_deg = 111_320 * math.cos(math.radians(center_lat))
        aero_cell_m = aero_res_deg * m_per_deg
        sat_cell_m = sat_res_deg * m_per_deg

        for wreck in wreck_locations:
            name = wreck.get("name", "Unknown")
            lat = wreck["lat"]
            lon = wreck["lon"]

            try:
                r_aero, c_aero = rowcol(aero_src.transform, lon, lat)
                r_sat, c_sat = rowcol(sat_src.transform, lon, lat)
            except Exception:
                logger.warning("Wreck %s at (%.4f, %.4f) outside grid bounds", name, lat, lon)
                continue

            # Bounds check
            if not (0 <= r_aero < aero_grid.shape[0] and 0 <= c_aero < aero_grid.shape[1]):
                continue
            if not (0 <= r_sat < sat_grid.shape[0] and 0 <= c_sat < sat_grid.shape[1]):
                continue

            proof = run_satellite_proof(
                aero_grid, aero_cell_m,
                sat_grid, sat_cell_m,
                (int(r_aero), int(c_aero)),
                (int(r_sat), int(c_sat)),
                wreck_name=name,
                lat=lat,
                lon=lon,
                continuation_height_m=continuation_height_m,
                threshold_nt=threshold_nt,
            )
            results.append(asdict(proof))

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K Satellite Proof Test")
    parser.add_argument("--aero-tif", required=True,
                        help="Path to Tier 1 aeromagnetic GeoTIFF")
    parser.add_argument("--sat-tif", required=True,
                        help="Path to Tier 4 satellite GeoTIFF (EMAG2/Swarm)")
    parser.add_argument("--wrecks-json", required=True,
                        help="JSON file with [{name, lat, lon, ...}, ...]")
    parser.add_argument("--height-m", type=float, default=400_000.0,
                        help="Continuation height (metres). Default: 400000 (satellite)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum anomaly (nT) to count as visible")
    parser.add_argument("--output", type=str, default="satellite_proof_results.json")
    args = parser.parse_args()

    with open(args.wrecks_json) as f:
        wrecks = json.load(f)

    results = batch_satellite_proof(
        aero_tif_path=args.aero_tif,
        satellite_tif_path=args.sat_tif,
        wreck_locations=wrecks,
        continuation_height_m=args.height_m,
        threshold_nt=args.threshold,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_viable = sum(1 for r in results if r["sat_detection_viable"])
    logger.info("Satellite Proof Summary: %d / %d wrecks show viable detection",
                n_viable, len(results))


if __name__ == "__main__":
    main()
