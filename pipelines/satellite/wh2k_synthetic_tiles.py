"""
WreckHunter 2000 — Physics-Based Synthetic Tile Generator
============================================================
Generates 3-channel image tiles (NSS, VDR, Tilt-Angle) for ResNet-18 training.

Three archetypes with physics constraints:
  1. STEEL_HULL   ("Iron Giant")  — 700ft+ prolate spheroid, massive moment
  2. WOOD_CARGO   ("Cargo Ghost") — 150-300ft point-source, high but tight moment
  3. WELLHEAD     ("Pin-Prick")   — Vertical cylinder / monopole, sharp footprint

Each tile also embeds a synthetic Regional Strike (long-wavelength linear geology)
at a random angle.  Wreck dipoles are forced to a DIFFERENT random angle,
teaching the model that wreck-axis ⊥ geology-axis (correlation ≈ 0).

NautiCurvs integration:
  After generation, tiles can be fed through the NautiCurvs curvelet filter
  to verify that geology is removed while off-axis dipoles survive.

Satellite visibility:
  Every target gets an Upward Continuation tag (SAT_VISIBLE) based on
  whether the signal persists when continued to 400 km altitude.

Output:  NPZ tiles (NSS + VDR + Tilt) sized for ResNet-18 (224×224 or 256×256).

NEVER store synthetic data in the real wreck database.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ── Physical Constants ──────────────────────────────────────────────────────

MU_0 = 4 * math.pi * 1e-7            # T·m/A
EARTH_FIELD_NT = 55_000.0             # nT at Lake Erie
EARTH_INCL_DEG = 68.0
EARTH_DECL_DEG = -9.0

# NE-SW geological strike for Lake Erie
NE_SW_STRIKE_DEG = 45.0

# Basin noise (nT std dev)
BASIN_NOISE = {"western": 12.0, "central": 6.0, "eastern": 3.5}

# ── Archetype Definitions ───────────────────────────────────────────────────

@dataclass
class Archetype:
    """Physics-based target archetype for synthetic generation."""
    label: str
    label_id: int
    length_ft_range: tuple[float, float]
    moment_range: tuple[float, float]    # A·m²
    geometry: str                         # "prolate_spheroid", "point_cluster", "cylinder"
    burial_depth_range: tuple[float, float]  # metres below lake bed
    sat_visible_expected: bool
    description: str


ARCHETYPES = {
    "STEEL_HULL": Archetype(
        label="STEEL_HULL",
        label_id=1,
        length_ft_range=(700, 1000),
        moment_range=(5e6, 2e7),        # Massive — taconite carriers, car ferries
        geometry="prolate_spheroid",
        burial_depth_range=(0, 5),
        sat_visible_expected=True,       # Should persist at satellite altitude
        description="700ft+ steel freighter — pill-shaped, high-amplitude, spatially broad",
    ),
    "WOOD_CARGO": Archetype(
        label="WOOD_CARGO",
        label_id=2,
        length_ft_range=(150, 300),
        moment_range=(1e6, 5e6),        # High but spatially tight (iron ore cargo)
        geometry="point_cluster",
        burial_depth_range=(0, 3),
        sat_visible_expected=False,      # Disappears at satellite altitude
        description="150-300ft wooden ship w/ ore cargo — sharp spike, compact",
    ),
    "WELLHEAD": Archetype(
        label="WELLHEAD",
        label_id=3,
        length_ft_range=(5, 20),
        moment_range=(0.02e6, 0.12e6),  # Small metal, vertical casing
        geometry="cylinder",
        burial_depth_range=(0, 0),       # Surface feature
        sat_visible_expected=False,
        description="Vertical cylinder monopole — very sharp, tiny footprint",
    ),
    "GEOLOGY_ONLY": Archetype(
        label="GEOLOGY_ONLY",
        label_id=0,
        length_ft_range=(0, 0),
        moment_range=(0, 0),
        geometry="none",
        burial_depth_range=(0, 0),
        sat_visible_expected=False,
        description="Background — regional geology + instrument noise only",
    ),
}


# ── Dipole Field Computation (Total-Field Anomaly) ──────────────────────────

def _compute_dipole_field(
    moment: float,
    orientation_deg: float,
    total_depth_m: float,
    grid_extent_m: float,
    n_pixels: int,
    flight_line_offset_m: float = 0.0,
) -> np.ndarray:
    """Compute ΔT anomaly from a magnetic dipole on a 2D grid.

    Returns a (n_pixels, n_pixels) array in nanoTesla.
    """
    x = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    y = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    X, Y = np.meshgrid(x, y)
    X = X + flight_line_offset_m

    R = np.sqrt(X**2 + Y**2 + total_depth_m**2)
    R = np.maximum(R, 1.0)

    incl = math.radians(EARTH_INCL_DEG)
    theta = math.radians(orientation_deg)

    mx = moment * math.cos(incl) * math.sin(theta)
    my = moment * math.cos(incl) * math.cos(theta)
    mz = moment * math.sin(incl)

    m_dot_r = mx * X + my * Y + mz * total_depth_m
    factor = MU_0 / (4 * math.pi) * 1e9  # → nT

    Bx = factor * (3 * m_dot_r * X / R**5 - mx / R**3)
    By = factor * (3 * m_dot_r * Y / R**5 - my / R**3)
    Bz = factor * (3 * m_dot_r * total_depth_m / R**5 - mz / R**3)

    decl = math.radians(EARTH_DECL_DEG)
    Tx = math.cos(incl) * math.sin(decl)
    Ty = math.cos(incl) * math.cos(decl)
    Tz = math.sin(incl)

    delta_T = Bx * Tx + By * Ty + Bz * Tz
    return delta_T


def _compute_prolate_spheroid_field(
    moment: float,
    orientation_deg: float,
    length_m: float,
    total_depth_m: float,
    grid_extent_m: float,
    n_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Model a long steel hull as a chain of dipoles along its keel axis.

    Distributes the total moment across sub-dipoles spaced along the hull.
    Produces a spatially broader, pill-shaped anomaly compared to a point dipole.
    """
    n_sub = max(5, int(length_m / 30))  # ~1 sub-dipole per 30 m
    sub_moment = moment / n_sub

    x = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    y = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    X, Y = np.meshgrid(x, y)

    total_field = np.zeros((n_pixels, n_pixels), dtype=np.float64)

    theta = math.radians(orientation_deg)
    incl = math.radians(EARTH_INCL_DEG)
    decl = math.radians(EARTH_DECL_DEG)

    half_len = length_m / 2
    offsets = np.linspace(-half_len, half_len, n_sub)

    for offset in offsets:
        cx = offset * math.sin(theta)
        cy = offset * math.cos(theta)

        Xp = X - cx
        Yp = Y - cy

        R = np.sqrt(Xp**2 + Yp**2 + total_depth_m**2)
        R = np.maximum(R, 1.0)

        # Sub-dipole has same orientation as whole hull
        mx = sub_moment * math.cos(incl) * math.sin(theta)
        my = sub_moment * math.cos(incl) * math.cos(theta)
        mz = sub_moment * math.sin(incl)

        m_dot_r = mx * Xp + my * Yp + mz * total_depth_m
        factor = MU_0 / (4 * math.pi) * 1e9

        Bx = factor * (3 * m_dot_r * Xp / R**5 - mx / R**3)
        By = factor * (3 * m_dot_r * Yp / R**5 - my / R**3)
        Bz = factor * (3 * m_dot_r * total_depth_m / R**5 - mz / R**3)

        Tx = math.cos(incl) * math.sin(decl)
        Ty = math.cos(incl) * math.cos(decl)
        Tz = math.sin(incl)

        total_field += Bx * Tx + By * Ty + Bz * Tz

    return total_field


def _compute_cylinder_field(
    moment: float,
    total_depth_m: float,
    casing_length_m: float,
    grid_extent_m: float,
    n_pixels: int,
) -> np.ndarray:
    """Model a vertical wellhead casing as a monopole-like source.

    A vertical steel casing acts like a vertically oriented dipole that's
    elongated — producing a sharp, nearly circular anomaly with weak negative
    ring.  We approximate this as a single vertical dipole (orientation = 0°,
    all moment in Z).
    """
    x = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    y = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2 + total_depth_m**2)
    R = np.maximum(R, 1.0)

    # Purely vertical moment
    mx, my = 0.0, 0.0
    mz = moment

    m_dot_r = mz * total_depth_m
    factor = MU_0 / (4 * math.pi) * 1e9

    Bx = factor * (3 * m_dot_r * X / R**5 - mx / R**3)
    By = factor * (3 * m_dot_r * Y / R**5 - my / R**3)
    Bz = factor * (3 * m_dot_r * total_depth_m / R**5 - mz / R**3)

    incl = math.radians(EARTH_INCL_DEG)
    decl = math.radians(EARTH_DECL_DEG)
    Tx = math.cos(incl) * math.sin(decl)
    Ty = math.cos(incl) * math.cos(decl)
    Tz = math.sin(incl)

    return Bx * Tx + By * Ty + Bz * Tz


# ── Derived Grid Layers (NSS, VDR, Tilt-Angle) ─────────────────────────────

def _compute_nss(grid: np.ndarray) -> np.ndarray:
    """Normalised Source Strength (analytic signal amplitude).
    
    NSS = sqrt(dT/dx² + dT/dy² + dT/dz²)
    We approximate dT/dz via Hilbert-like vertical derivative (Laplacian proxy).
    """
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    # Approximate vertical derivative as Laplacian (standard in potential field work)
    dz = ndimage.laplace(grid)
    nss = np.sqrt(dx**2 + dy**2 + dz**2)
    return nss


def _compute_vdr(grid: np.ndarray) -> np.ndarray:
    """Vertical Derivative (first vertical derivative).
    
    Approximated in the Fourier domain: multiply spectrum by |k|.
    """
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = 1e-10  # avoid DC amplification
    vdr = np.real(np.fft.ifft2(fft * k_mag * 2 * np.pi))
    return vdr


def _compute_tilt_angle(grid: np.ndarray) -> np.ndarray:
    """Tilt Angle (TDR) = atan2(VDR, THDR).
    
    THDR = Total Horizontal Derivative = sqrt(dT/dx² + dT/dy²)
    Tilt angle normalises amplitude, making deep and shallow sources comparable.
    """
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    thdr = np.sqrt(dx**2 + dy**2)
    vdr = _compute_vdr(grid)
    tilt = np.arctan2(vdr, thdr + 1e-12)  # radians, range [-π/2, π/2]
    return tilt


# ── Regional Strike (Geological Noise) Injection ───────────────────────────

def _generate_regional_strike(
    n_pixels: int,
    strike_angle_deg: float,
    amplitude_nt: float,
    wavelength_pixels: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a long-wavelength linear magnetic ridge (geological feature).

    This teaches the model what a geological 'dyke' or regional trend looks like.
    The key: it's LINEAR and at a consistent angle across the tile.
    """
    x = np.linspace(-1, 1, n_pixels)
    y = np.linspace(-1, 1, n_pixels)
    X, Y = np.meshgrid(x, y)

    theta = math.radians(strike_angle_deg)
    # Project grid onto strike-perpendicular direction
    perp = X * math.cos(theta) - Y * math.sin(theta)

    freq = n_pixels / max(wavelength_pixels, 1)
    ridge = amplitude_nt * np.sin(2 * np.pi * freq * perp)

    # Add some natural variation (not perfectly sinusoidal)
    ridge += rng.normal(0, amplitude_nt * 0.1, ridge.shape)

    return ridge


# ── Satellite Visibility Check (Upward Continuation) ───────────────────────

def _upward_continue(grid: np.ndarray, dz_m: float, dx_m: float) -> np.ndarray:
    """Upward continuation in Fourier domain.

    Attenuates high-frequency (shallow) sources and preserves deep/large sources.
    Used to test if a target is visible at satellite altitude (400 km).

    grid:  2D anomaly in nT
    dz_m:  continuation height in metres (e.g. 400_000 for satellite)
    dx_m:  grid cell size in metres
    """
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny, d=dx_m).reshape(-1, 1)
    kx = np.fft.fftfreq(nx, d=dx_m).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    # Upward continuation filter: exp(-2π·|k|·Δz)
    uc_filter = np.exp(-2 * np.pi * k_mag * dz_m)
    return np.real(np.fft.ifft2(fft * uc_filter))


def check_satellite_visibility(
    grid: np.ndarray,
    dx_m: float = 100.0,
    satellite_height_m: float = 400_000.0,
    threshold_nt: float = 0.5,
) -> dict:
    """Test whether a magnetic target is visible from satellite altitude.

    Returns dict with:
      sat_visible: bool
      peak_at_surface_nt: float
      peak_at_satellite_nt: float
      attenuation_ratio: float
    """
    continued = _upward_continue(grid, satellite_height_m, dx_m)
    peak_surface = float(np.max(np.abs(grid)))
    peak_sat = float(np.max(np.abs(continued)))
    ratio = peak_sat / (peak_surface + 1e-12)

    return {
        "sat_visible": peak_sat >= threshold_nt,
        "peak_at_surface_nt": peak_surface,
        "peak_at_satellite_nt": peak_sat,
        "attenuation_ratio": ratio,
        "satellite_height_m": satellite_height_m,
    }


# ── Single Tile Generator ──────────────────────────────────────────────────

def generate_tile(
    archetype_key: str,
    basin: str = "central",
    n_pixels: int = 224,
    grid_extent_m: float = 2000.0,
    rng: np.random.Generator | None = None,
    inject_geology: bool = True,
) -> dict:
    """Generate a single 3-channel training tile (NSS, VDR, Tilt-Angle).

    Returns dict with:
      tile: np.ndarray shape (3, n_pixels, n_pixels)  [CHW]
      label: str (archetype label)
      label_id: int
      metadata: dict (physics params, satellite visibility, etc.)
    """
    if rng is None:
        rng = np.random.default_rng()

    arch = ARCHETYPES[archetype_key]
    noise_std = BASIN_NOISE.get(basin, 6.0)
    dx_m = grid_extent_m / n_pixels

    # ── Step 1: Generate the target anomaly ─────────────────────────────
    if archetype_key == "GEOLOGY_ONLY":
        # Pure background — no target
        anomaly = np.zeros((n_pixels, n_pixels), dtype=np.float64)
        target_orientation = 0.0
        moment = 0.0
        burial = 0.0
        water_depth = rng.uniform(10, 60)
        length_m = 0.0
    else:
        moment = rng.uniform(*arch.moment_range)
        burial = rng.uniform(*arch.burial_depth_range)
        water_depth_base = {"western": 12.0, "central": 22.0, "eastern": 45.0}
        water_depth = water_depth_base.get(basin, 22.0) * rng.uniform(0.7, 1.3)
        total_depth = 300.0 + water_depth + burial  # Aero survey at 300 m

        # Random target orientation (forced off regional strike)
        target_orientation = rng.uniform(0, 360)
        # 40% chance: force perpendicular to geology
        if rng.random() < 0.4:
            target_orientation = NE_SW_STRIKE_DEG + 90 + rng.normal(0, 15)

        length_ft = rng.uniform(*arch.length_ft_range)
        length_m = length_ft * 0.3048

        if arch.geometry == "prolate_spheroid":
            anomaly = _compute_prolate_spheroid_field(
                moment, target_orientation, length_m, total_depth,
                grid_extent_m, n_pixels, rng,
            )
        elif arch.geometry == "cylinder":
            anomaly = _compute_cylinder_field(
                moment, total_depth, rng.uniform(10, 50),
                grid_extent_m, n_pixels,
            )
        else:  # point_cluster
            anomaly = _compute_dipole_field(
                moment, target_orientation, total_depth,
                grid_extent_m, n_pixels,
            )

        # Silt burial attenuation
        if burial > 0:
            skin_depth = 50.0
            anomaly *= math.exp(-burial / skin_depth)

    # ── Step 2: Inject geological regional strike ───────────────────────
    if inject_geology:
        strike_angle = NE_SW_STRIKE_DEG + rng.normal(0, 10)
        strike_amplitude = rng.uniform(5, 40)  # nT
        wavelength_px = rng.uniform(n_pixels * 0.3, n_pixels * 0.8)
        geology = _generate_regional_strike(
            n_pixels, strike_angle, strike_amplitude, wavelength_px, rng,
        )
        anomaly = anomaly + geology

    # ── Step 3: Add basin-specific instrument noise ─────────────────────
    anomaly += rng.normal(0, noise_std, anomaly.shape)

    # ── Step 4: Compute 3 derived layers ────────────────────────────────
    nss = _compute_nss(anomaly)
    vdr = _compute_vdr(anomaly)
    tilt = _compute_tilt_angle(anomaly)

    tile = np.stack([nss, vdr, tilt], axis=0).astype(np.float32)  # (3, H, W)

    # ── Step 5: Satellite visibility check ──────────────────────────────
    sat_check = check_satellite_visibility(anomaly, dx_m)

    # ── Step 6: Axis correlation (wreck vs geology) ─────────────────────
    strike_angle_used = strike_angle if inject_geology else NE_SW_STRIKE_DEG
    angle_diff = abs(target_orientation - strike_angle_used)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    metadata = {
        "archetype": archetype_key,
        "basin": basin,
        "moment": moment,
        "orientation_deg": target_orientation,
        "burial_depth_m": burial,
        "water_depth_m": water_depth,
        "length_m": length_m,
        "geology_strike_deg": strike_angle_used,
        "axis_offset_from_geology_deg": angle_diff,
        "noise_std_nt": noise_std,
        "grid_extent_m": grid_extent_m,
        "dx_m": dx_m,
        "sat_visible": sat_check["sat_visible"],
        "peak_at_surface_nt": sat_check["peak_at_surface_nt"],
        "peak_at_satellite_nt": sat_check["peak_at_satellite_nt"],
        "sat_attenuation_ratio": sat_check["attenuation_ratio"],
        "sat_visible_tag": "SAT_VISIBLE" if sat_check["sat_visible"] else "SAT_INVISIBLE",
    }

    return {
        "tile": tile,
        "label": arch.label,
        "label_id": arch.label_id,
        "metadata": metadata,
    }


# ── Batch Tile Generator ───────────────────────────────────────────────────

def generate_training_tiles(
    n_steel: int = 500,
    n_wood: int = 500,
    n_wellhead: int = 500,
    n_geology: int = 5000,
    n_pixels: int = 224,
    grid_extent_m: float = 2000.0,
    basins: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Generate a complete training tile dataset.

    Returns dict with:
      tiles: np.ndarray (N, 3, H, W) float32
      labels: np.ndarray (N,) int — 0=GEOLOGY, 1=STEEL_HULL, 2=WOOD_CARGO, 3=WELLHEAD
      metadata: list[dict]
    """
    if basins is None:
        basins = ["western", "central", "eastern"]

    rng = np.random.default_rng(seed)

    plan = [
        ("STEEL_HULL", n_steel),
        ("WOOD_CARGO", n_wood),
        ("WELLHEAD", n_wellhead),
        ("GEOLOGY_ONLY", n_geology),
    ]

    all_tiles = []
    all_labels = []
    all_meta = []

    total = sum(count for _, count in plan)
    generated = 0

    for archetype_key, count in plan:
        logger.info("Generating %d %s tiles...", count, archetype_key)
        for i in range(count):
            basin = basins[rng.integers(0, len(basins))]
            try:
                result = generate_tile(
                    archetype_key, basin, n_pixels, grid_extent_m, rng,
                )
                all_tiles.append(result["tile"])
                all_labels.append(result["label_id"])
                all_meta.append(result["metadata"])
            except Exception as e:
                logger.warning("Tile generation failed (%s #%d): %s", archetype_key, i, e)
                continue

            generated += 1
            if generated % 500 == 0:
                logger.info("  Progress: %d / %d tiles", generated, total)

    tiles_arr = np.array(all_tiles, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    logger.info("Generated %d total tiles: %s", len(tiles_arr), {
        "GEOLOGY_ONLY": int(np.sum(labels_arr == 0)),
        "STEEL_HULL": int(np.sum(labels_arr == 1)),
        "WOOD_CARGO": int(np.sum(labels_arr == 2)),
        "WELLHEAD": int(np.sum(labels_arr == 3)),
    })

    return {
        "tiles": tiles_arr,
        "labels": labels_arr,
        "metadata": all_meta,
    }


# ── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K Synthetic Tile Generator")
    parser.add_argument("--n-steel", type=int, default=500)
    parser.add_argument("--n-wood", type=int, default=500)
    parser.add_argument("--n-wellhead", type=int, default=500)
    parser.add_argument("--n-geology", type=int, default=5000)
    parser.add_argument("--pixels", type=int, default=224,
                        help="Tile size (pixels). 224 for ResNet-18.")
    parser.add_argument("--extent-m", type=float, default=2000.0,
                        help="Grid extent in metres (2000 = 2km × 2km chip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="wreck_hunting_ml/data/synthetic")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = generate_training_tiles(
        n_steel=args.n_steel,
        n_wood=args.n_wood,
        n_wellhead=args.n_wellhead,
        n_geology=args.n_geology,
        n_pixels=args.pixels,
        grid_extent_m=args.extent_m,
        seed=args.seed,
    )

    # Save tiles
    np.savez_compressed(
        out_dir / "synthetic_tiles.npz",
        tiles=result["tiles"],
        labels=result["labels"],
    )

    # Save metadata
    with open(out_dir / "synthetic_metadata.json", "w") as f:
        json.dump(result["metadata"], f, indent=2, default=str)

    logger.info("Saved to %s/synthetic_tiles.npz (%d tiles)", out_dir, len(result["labels"]))


if __name__ == "__main__":
    main()
