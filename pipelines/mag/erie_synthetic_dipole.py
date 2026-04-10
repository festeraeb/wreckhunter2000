"""
Lake Erie Off-Axis Detector — Synthetic Dipole Generator
==========================================================
Generates synthetic magnetic anomaly training data using the classic 
magnetic dipole model B(r). Creates realistic wreck-like and wellhead-like
signatures at varying distances from flight lines, burial depths, and
orientations — tuned per Lake Erie basin.

Physics:
  B(r) = μ₀/(4π) · [3(m·r̂)r̂ - m] / r³
  Total-field anomaly ΔT ≈ B(r) · T̂  (projection onto Earth's field)

Parameters varied:
  - Distance from flight line: 0–2000m (step ~200m)
  - Burial depth in silt: 0–8m (attenuates amplitude)
  - Orientation: random + forced perpendicular to NE-SW geology
  - Size: scaled to 300-ft steel hull (Colgate-class moment)
  - Noise: real Lake Erie background from clean grid patches

NEVER store synthetic data in the real wreck database.
"""

from __future__ import annotations

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Physical Constants ──────────────────────────────────────────────────────

MU_0 = 4 * math.pi * 1e-7          # Permeability of free space (T·m/A)
EARTH_FIELD_NT = 55_000.0            # Approximate Earth's field at Lake Erie (nT)
EARTH_INCL_DEG = 68.0                # Magnetic inclination at Lake Erie
EARTH_DECL_DEG = -9.0                # Magnetic declination at Lake Erie

# Magnetic moment estimates for known wreck types
# moment = (mass_kg * susceptibility * earth_field) / mu_0
WRECK_MOMENTS = {
    "whaleback_300ft": 2.5e6,        # Colgate-class (308 ft, ~2500 tons steel)
    "steel_freighter_250ft": 2.0e6,  # Typical Great Lakes steel freighter
    "steel_freighter_200ft": 1.2e6,  # Smaller steel hull
    "iron_schooner_150ft": 0.5e6,    # Iron-hulled schooner
    "steel_tug_80ft": 0.15e6,        # Steel tug
    "iron_ore_cargo": 3.0e6,         # Ship with iron ore cargo (very strong)
    "car_ferry": 3.5e6,              # Car ferry with vehicles (massive moment)
}

WELLHEAD_MOMENTS = {
    "gas_well_active": 0.08e6,       # Active gas wellhead + casing
    "gas_well_abandoned": 0.05e6,    # Abandoned wellhead (less metal)
    "oil_well": 0.12e6,              # Oil well with pump jack remnants
    "test_hole": 0.02e6,             # Stratigraphic test hole
}

# Basin-specific noise characteristics (nT standard deviation)
BASIN_NOISE = {
    "western": 12.0,   # Shallow, mineral-rich, noisy
    "central": 6.0,    # Moderate depth, quieter
    "eastern": 3.5,    # Deep, very quiet sedimentary
}

# NE-SW geological strike angle for Lake Erie region
NE_SW_STRIKE_DEG = 45.0


# ── Core dipole field calculation ───────────────────────────────────────────

@dataclass
class DipoleSource:
    """A magnetic dipole source (wreck or wellhead)."""
    lat: float = 42.0
    lon: float = -81.0
    moment: float = 2.5e6           # Magnetic moment (A·m²)
    orientation_deg: float = 0.0     # Dipole axis orientation (CW from N)
    burial_depth_m: float = 0.0      # Depth below lake bed
    water_depth_m: float = 20.0      # Water depth
    label: int = 1                   # 1=wreck, 0=wellhead
    source_type: str = "whaleback_300ft"
    basin: str = "central"


def compute_dipole_field_2d(
    source: DipoleSource,
    survey_height_m: float = 300.0,   # Typical aero-mag survey altitude
    grid_size_m: float = 5000.0,      # Grid extent
    resolution_m: float = 100.0,      # Grid resolution
    flight_line_offset_m: float = 0.0,  # Lateral offset from flight line
) -> dict:
    """Compute the total-field anomaly from a magnetic dipole on a 2D grid.
    
    Returns a dict with the grid and extracted features matching the
    candidate CSV format.
    """
    # Sensor distance = survey height + water depth + burial depth
    total_depth = survey_height_m + source.water_depth_m + source.burial_depth_m

    # Create observation grid
    n_pts = int(grid_size_m / resolution_m) + 1
    x = np.linspace(-grid_size_m / 2, grid_size_m / 2, n_pts)
    y = np.linspace(-grid_size_m / 2, grid_size_m / 2, n_pts)
    X, Y = np.meshgrid(x, y)

    # Shift grid to simulate flight line offset
    X = X + flight_line_offset_m

    # Distance from dipole in 3D
    R = np.sqrt(X**2 + Y**2 + total_depth**2)
    R = np.maximum(R, 1.0)  # Avoid singularity

    # Dipole orientation (unit vector)
    theta_rad = math.radians(source.orientation_deg)
    incl_rad = math.radians(EARTH_INCL_DEG)
    
    # Magnetic moment vector (tilted by orientation + inclination)
    mx = source.moment * math.cos(incl_rad) * math.sin(theta_rad)
    my = source.moment * math.cos(incl_rad) * math.cos(theta_rad)
    mz = source.moment * math.sin(incl_rad)

    # Dot product m·r for each grid point
    m_dot_r = mx * X + my * Y + mz * total_depth

    # Dipole field components B(r) = μ₀/(4π) * [3(m·r̂)r̂ - m] / r³
    factor = MU_0 / (4 * math.pi) * 1e9  # Convert to nT

    Bx = factor * (3 * m_dot_r * X / R**5 - mx / R**3)
    By = factor * (3 * m_dot_r * Y / R**5 - my / R**3)
    Bz = factor * (3 * m_dot_r * total_depth / R**5 - mz / R**3)

    # Total field anomaly ΔT ≈ projection onto Earth's field direction
    decl_rad = math.radians(EARTH_DECL_DEG)
    Tx = math.cos(incl_rad) * math.sin(decl_rad)
    Ty = math.cos(incl_rad) * math.cos(decl_rad)
    Tz = math.sin(incl_rad)

    delta_T = Bx * Tx + By * Ty + Bz * Tz

    # Amplitude attenuation from silt burial (exponential decay)
    if source.burial_depth_m > 0:
        # Silt attenuates high-frequency components; approximate as
        # amplitude reduction proportional to (depth_factor)
        skin_depth = 50.0  # Effective skin depth in Lake Erie silt (m)
        atten = math.exp(-source.burial_depth_m / skin_depth)
        delta_T *= atten

    return {
        "grid": delta_T,
        "x": x,
        "y": y,
        "peak_amplitude_nt": float(np.max(np.abs(delta_T))),
        "mean_amplitude_nt": float(np.mean(np.abs(delta_T[delta_T != 0]))),
        "source": source,
        "flight_line_offset_m": flight_line_offset_m,
        "total_depth_m": total_depth,
    }


# ── Feature extraction from synthetic grid ──────────────────────────────────

def extract_features_from_grid(grid_result: dict, basin: str = "central") -> dict:
    """Extract the same feature set as real candidates from a synthetic grid.
    
    Features match the expanded set required by the XGBoost classifier:
    amplitude_peak_abs, gradient_contrast, dipole_separation_m, lobe_symmetry_ratio,
    axis_offset_deg, aspect_ratio, distance_to_nearest_flight_line_m,
    basin_id (one-hot), local_snr_vs_basin_median, curvelet_edge_score
    """
    grid = grid_result["grid"]
    x = grid_result["x"]
    resolution = x[1] - x[0] if len(x) > 1 else 100.0
    source: DipoleSource = grid_result["source"]

    # Peak amplitudes
    amp_peak = float(np.max(np.abs(grid)))
    amp_mean = float(np.mean(np.abs(grid)))

    # Gradient (spatial derivatives)
    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_contrast = float(np.max(grad_mag) / (np.mean(grad_mag) + 1e-10))

    # Dipole separation: distance between positive and negative peaks
    pos_idx = np.unravel_index(np.argmax(grid), grid.shape)
    neg_idx = np.unravel_index(np.argmin(grid), grid.shape)
    dipole_sep_px = math.sqrt((pos_idx[0] - neg_idx[0])**2 + (pos_idx[1] - neg_idx[1])**2)
    dipole_separation_m = dipole_sep_px * resolution

    # Lobe symmetry: ratio of negative peak to positive peak
    pos_max = float(np.max(grid))
    neg_min = float(np.min(grid))
    lobe_symmetry = abs(neg_min) / (pos_max + 1e-10) if pos_max > 0 else 0

    # Axis offset from NE-SW geological strike
    dy = pos_idx[0] - neg_idx[0]
    dx = pos_idx[1] - neg_idx[1]
    dipole_axis_deg = math.degrees(math.atan2(dx, dy)) % 360
    axis_offset_deg = abs(dipole_axis_deg - NE_SW_STRIKE_DEG)
    if axis_offset_deg > 180:
        axis_offset_deg = 360 - axis_offset_deg
    if axis_offset_deg > 90:
        axis_offset_deg = 180 - axis_offset_deg

    # Aspect ratio: extent of anomaly at half-max
    threshold = amp_peak * 0.5
    above_thresh = np.abs(grid) > threshold
    if np.any(above_thresh):
        rows = np.any(above_thresh, axis=1)
        cols = np.any(above_thresh, axis=0)
        height_px = np.sum(rows)
        width_px = np.sum(cols)
    else:
        height_px = width_px = 1
    aspect_ratio = max(width_px, height_px) / max(min(width_px, height_px), 1)

    # Pixel count (number of cells above threshold)
    pixel_count = int(np.sum(above_thresh))

    # Basin-specific SNR 
    basin_noise = BASIN_NOISE.get(basin, 6.0)
    local_snr = amp_peak / (basin_noise + 1e-10)

    # Curvelet edge score (approximation — real one comes from Rust)
    # Use Laplacian-based edge detection as proxy
    from scipy import ndimage
    laplacian = ndimage.laplace(grid)
    curvelet_edge_score = float(np.max(np.abs(laplacian)) / (amp_peak + 1e-10))

    # Flip distance (distance from positive peak to zero crossing along dipole axis)
    center_row = grid.shape[0] // 2
    center_col = grid.shape[1] // 2
    profile = grid[center_row, :] if abs(dx) > abs(dy) else grid[:, center_col]
    zero_crossings = np.where(np.diff(np.sign(profile)))[0]
    if len(zero_crossings) > 0:
        flip_distance_km = float(np.min(np.abs(zero_crossings - len(profile) // 2)) * resolution / 1000.0)
    else:
        flip_distance_km = 5.0  # Default: far = geological-like

    # Basin one-hot
    basin_id_map = {"western": 0, "central": 1, "eastern": 2}
    basin_id = basin_id_map.get(basin, 1)

    return {
        "amplitude_peak_abs": amp_peak,
        "amplitude_mean_abs": amp_mean,
        "gradient_contrast": gradient_contrast,
        "dipole_separation_m": dipole_separation_m,
        "lobe_symmetry_ratio": min(lobe_symmetry, 2.0),
        "axis_offset_deg": axis_offset_deg,
        "aspect_ratio": aspect_ratio,
        "flip_distance_km": flip_distance_km,
        "pixel_count": pixel_count,
        "distance_to_nearest_flight_line_m": grid_result["flight_line_offset_m"],
        "basin_western": 1 if basin_id == 0 else 0,
        "basin_central": 1 if basin_id == 1 else 0,
        "basin_eastern": 1 if basin_id == 2 else 0,
        "local_snr_vs_basin_median": local_snr,
        "curvelet_edge_score": curvelet_edge_score,
        "width_m": float(width_px * resolution),
        "height_m": float(height_px * resolution),
    }


# ── Batch synthetic generation ──────────────────────────────────────────────

def generate_wreck_synthetics(
    n_per_basin: int = 10_000,
    basins: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict]]:
    """Generate synthetic wreck anomalies for all 3 basins.
    
    Varies:
    - Flight line offset: 0–2000m (push off-axis detection)
    - Burial depth: 0–8m silt
    - Orientation: random + forced perpendicular to NE-SW
    - Wreck size: sampled from known types
    - Basin-specific water depth and noise
    
    Returns dict[basin_name] → list of feature dicts (label=1).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if basins is None:
        basins = ["western", "central", "eastern"]

    basin_water_depth = {"western": 12.0, "central": 22.0, "eastern": 45.0}
    wreck_types = list(WRECK_MOMENTS.keys())

    all_synthetics: dict[str, list[dict]] = {}

    for basin in basins:
        logger.info("Generating %d wreck synthetics for %s basin...", n_per_basin, basin)
        synthetics = []
        noise_std = BASIN_NOISE[basin]
        base_water_depth = basin_water_depth.get(basin, 22.0)

        for i in range(n_per_basin):
            # Random parameters
            wreck_type = wreck_types[rng.integers(0, len(wreck_types))]
            moment = WRECK_MOMENTS[wreck_type] * rng.uniform(0.5, 1.5)  # ±50% variation
            offset_m = rng.uniform(0, 2000)           # Off-axis push
            burial_m = rng.uniform(0, 8)               # Silt burial
            orientation = rng.uniform(0, 360)           # Random orientation
            water_depth = base_water_depth * rng.uniform(0.7, 1.3)

            # 30% of samples: force perpendicular to NE-SW (anomalous orientation)
            if rng.random() < 0.3:
                orientation = NE_SW_STRIKE_DEG + 90 + rng.normal(0, 10)

            source = DipoleSource(
                moment=moment,
                orientation_deg=orientation % 360,
                burial_depth_m=burial_m,
                water_depth_m=water_depth,
                label=1,
                source_type=wreck_type,
                basin=basin,
            )

            try:
                result = compute_dipole_field_2d(
                    source,
                    survey_height_m=300.0,
                    grid_size_m=5000.0,
                    resolution_m=100.0,
                    flight_line_offset_m=offset_m,
                )

                # Add realistic noise
                result["grid"] += rng.normal(0, noise_std, result["grid"].shape)

                features = extract_features_from_grid(result, basin)
                features["label"] = 1
                features["source_type"] = wreck_type
                features["synthetic"] = True
                features["burial_depth_m"] = burial_m
                features["flight_line_offset_m"] = offset_m
                synthetics.append(features)
            except Exception as e:
                logger.debug("Synthetic generation failed for sample %d: %s", i, e)
                continue

            if (i + 1) % 1000 == 0:
                logger.info("  %s basin: %d/%d generated", basin, i + 1, n_per_basin)

        all_synthetics[basin] = synthetics
        logger.info("Generated %d wreck synthetics for %s basin", len(synthetics), basin)

    return all_synthetics


def generate_wellhead_synthetics(
    n_per_basin: int = 3_000,
    basins: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict]]:
    """Generate synthetic wellhead anomalies.
    
    Wellheads differ from wrecks:
    - Monopolar (single positive spike, no negative lobe)
    - Small spatial extent
    - High amplitude relative to size
    - Aligned with or random to geology (no forced perpendicular)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    if basins is None:
        basins = ["western", "central", "eastern"]

    basin_water_depth = {"western": 12.0, "central": 22.0, "eastern": 45.0}
    well_types = list(WELLHEAD_MOMENTS.keys())

    all_synthetics: dict[str, list[dict]] = {}

    for basin in basins:
        logger.info("Generating %d wellhead synthetics for %s basin...", n_per_basin, basin)
        synthetics = []
        noise_std = BASIN_NOISE[basin]

        for i in range(n_per_basin):
            well_type = well_types[rng.integers(0, len(well_types))]
            moment = WELLHEAD_MOMENTS[well_type] * rng.uniform(0.5, 2.0)
            # Wellheads are at known positions, no flight line offset effect on detectability
            # but we vary to teach the model that wellheads appear everywhere
            offset_m = rng.uniform(0, 1500)
            # Wellheads: mostly vertical orientation (casing string)
            orientation = rng.uniform(0, 360)
            water_depth = basin_water_depth.get(basin, 22.0) * rng.uniform(0.5, 1.5)

            source = DipoleSource(
                moment=moment,
                orientation_deg=orientation,
                burial_depth_m=0,  # Wellheads are at surface
                water_depth_m=water_depth,
                label=0,
                source_type=well_type,
                basin=basin,
            )

            try:
                result = compute_dipole_field_2d(
                    source,
                    survey_height_m=300.0,
                    grid_size_m=3000.0,  # Smaller grid for wellheads
                    resolution_m=100.0,
                    flight_line_offset_m=offset_m,
                )
                result["grid"] += rng.normal(0, noise_std, result["grid"].shape)

                features = extract_features_from_grid(result, basin)
                features["label"] = 0
                features["source_type"] = well_type
                features["synthetic"] = True
                features["burial_depth_m"] = 0
                features["flight_line_offset_m"] = offset_m
                synthetics.append(features)
            except Exception as e:
                logger.debug("Wellhead synthetic failed for sample %d: %s", i, e)
                continue

        all_synthetics[basin] = synthetics
        logger.info("Generated %d wellhead synthetics for %s basin", len(synthetics), basin)

    return all_synthetics


def generate_geological_synthetics(
    n_per_basin: int = 2_000,
    basins: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict]]:
    """Generate geological false positive synthetics.
    
    Geological anomalies are:
    - Broad, smooth, large-scale
    - Aligned with NE-SW regional strike
    - Low gradient contrast
    """
    if rng is None:
        rng = np.random.default_rng(456)
    if basins is None:
        basins = ["western", "central", "eastern"]

    all_synthetics: dict[str, list[dict]] = {}

    for basin in basins:
        logger.info("Generating %d geological synthetics for %s basin...", n_per_basin, basin)
        synthetics = []
        noise_std = BASIN_NOISE[basin]

        for i in range(n_per_basin):
            # Geological: large moment, aligned with strike, far from flight line
            moment = rng.uniform(5e6, 50e6)  # Much larger than wrecks
            orientation = NE_SW_STRIKE_DEG + rng.normal(0, 15)  # Aligned with geology
            offset_m = rng.uniform(0, 2000)

            source = DipoleSource(
                moment=moment,
                orientation_deg=orientation % 360,
                burial_depth_m=rng.uniform(10, 100),  # Deep geological sources
                water_depth_m=rng.uniform(10, 60),
                label=-1,  # Geological (also negative class)
                source_type="geological",
                basin=basin,
            )

            try:
                result = compute_dipole_field_2d(
                    source,
                    survey_height_m=300.0,
                    grid_size_m=10000.0,  # Larger grid for geological features
                    resolution_m=200.0,
                    flight_line_offset_m=offset_m,
                )
                result["grid"] += rng.normal(0, noise_std, result["grid"].shape)

                features = extract_features_from_grid(result, basin)
                features["label"] = 0  # Negative class (not wreck)
                features["source_type"] = "geological"
                features["synthetic"] = True
                features["burial_depth_m"] = float(source.burial_depth_m)
                features["flight_line_offset_m"] = offset_m
                synthetics.append(features)
            except Exception as e:
                logger.debug("Geological synthetic failed: %s", e)
                continue

        all_synthetics[basin] = synthetics
        logger.info("Generated %d geological synthetics for %s basin", len(synthetics), basin)

    return all_synthetics
