"""
WreckHunter 2000 — CRM-Physics Synthetic Tile Generator (v2)
=============================================================
Encodes Construction Remanent Magnetization (CRM) into wreck dipoles.

Key CRM physics insight (from user):
  "A ship takes on the magnetic properties of where it was built...
   the M and B Bessemer steel riveted laid up in an east-west orientation
   will have a different angle than the earth."

  - Hot rivets cool on the building ways → TRM aligned with construction heading
  - Ship sinks at different heading → CRM points "wrong way"
  - Total wreck anomaly = Induced (aligned with wreck heading) + Remanent (construction heading)
  - Wells have only Induced (vertical casing, no CRM) → symmetrical anomaly
  - This asymmetry difference is the SOLE discriminator at sub-pixel resolution

Ground truth calibration (March 2026 discovery report):
  - Ghost #1: James H. Reed (steel freighter, 1944) — amp=709.9 nT, asym=0.15
  - Ghost #2: Well T007348 (Ontario well)           — amp=721.3 nT, asym=0.11
  - SAME amplitude, DIFFERENT asymmetry → CRM is the discriminator

Changes from v1:
  - STEEL_HULL: Two-component dipole (Induced + CRM) with Koenigsberger ratio Q
  - WOOD_CARGO: Weak CRM from iron fittings (low Q)
  - WELLHEAD: Unchanged (pure vertical monopole — no CRM)
  - Wreck heading ≠ construction heading → azimuth deviation → asymmetric anomaly

Output:  NPZ tiles (NSS + VDR + Tilt) sized for ResNet-18 (224×224).
         Format identical to v1 — drop-in replacement for training.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import time
import gc

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ── Physical Constants ──────────────────────────────────────────────────────

MU_0 = 4 * math.pi * 1e-7            # T·m/A
EARTH_FIELD_NT = 55_000.0             # nT at Lake Erie (modern)
EARTH_INCL_DEG = 68.0                 # Modern inclination at Lake Erie
EARTH_DECL_DEG = -9.0                 # Modern declination at Lake Erie

# NE-SW geological strike for Lake Erie
NE_SW_STRIKE_DEG = 45.0

# Basin noise (nT std dev)
BASIN_NOISE = {"western": 12.0, "central": 6.0, "eastern": 3.5}

# ── CRM Constants (Great Lakes Shipbuilding) ────────────────────────────────

# Construction sites with approximate IGRF values circa 1880-1920.
# Ships built on the Great Lakes were constructed primarily in these cities.
# The inclination/declination difference from modern wreck-site values
# adds an additional (small) angular offset on top of heading mismatch.
CONSTRUCTION_SITES = {
    "cleveland":  {"lat": 41.5, "incl_deg": 72.0, "decl_deg": -5.0},
    "detroit":    {"lat": 42.3, "incl_deg": 71.5, "decl_deg": -4.5},
    "lorain":     {"lat": 41.5, "incl_deg": 72.0, "decl_deg": -5.5},
    "bay_city":   {"lat": 43.6, "incl_deg": 73.0, "decl_deg": -3.5},
    "toledo":     {"lat": 41.7, "incl_deg": 72.0, "decl_deg": -5.0},
    "buffalo":    {"lat": 42.9, "incl_deg": 72.5, "decl_deg": -6.0},
    "ashtabula":  {"lat": 41.9, "incl_deg": 72.0, "decl_deg": -5.0},
}
SITE_NAMES = list(CONSTRUCTION_SITES.keys())


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
    # CRM parameters (v2)
    Q_range: tuple[float, float] = (0.0, 0.0)  # Koenigsberger ratio range


ARCHETYPES = {
    "STEEL_HULL": Archetype(
        label="STEEL_HULL",
        label_id=1,
        length_ft_range=(700, 1000),
        moment_range=(5e6, 2e7),
        geometry="prolate_spheroid",
        burial_depth_range=(0, 5),
        sat_visible_expected=True,
        description="700ft+ steel freighter — CRM from riveted Bessemer steel",
        # Riveted ships (pre-1940): strong CRM. Q = 0.5 – 1.5
        # Every rivet is a separate permanent magnet aligned with construction heading.
        Q_range=(0.5, 1.5),
    ),
    "WOOD_CARGO": Archetype(
        label="WOOD_CARGO",
        label_id=2,
        length_ft_range=(150, 300),
        moment_range=(1e6, 5e6),
        geometry="point_cluster",
        burial_depth_range=(0, 3),
        sat_visible_expected=False,
        description="150-300ft wooden ship w/ ore cargo — weak CRM from iron fittings",
        # Wooden hulls with iron hardware (knees, bolts, chains): weak CRM.
        Q_range=(0.05, 0.25),
    ),
    "WELLHEAD": Archetype(
        label="WELLHEAD",
        label_id=3,
        length_ft_range=(5, 20),
        moment_range=(0.02e6, 0.12e6),
        geometry="cylinder",
        burial_depth_range=(0, 0),
        sat_visible_expected=False,
        description="Vertical cylinder monopole — no CRM, purely induced",
        # Wells: installed in-situ, no construction remanence.
        Q_range=(0.0, 0.0),
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
        Q_range=(0.0, 0.0),
    ),
}


# ── CRM Moment Vector Computation ──────────────────────────────────────────

def _compute_crm_moment_vector(
    total_moment: float,
    wreck_heading_deg: float,
    construction_heading_deg: float,
    Q: float,
    construction_site: str = "cleveland",
) -> tuple[float, float, float]:
    """Compute total moment vector (mx, my, mz) as sum of Induced + Remanent.

    Physics model:
      - Ship hull = prolate spheroid → easy to magnetize along long axis
      - INDUCED: horizontal component aligned with WRECK resting heading
        (shape anisotropy concentrates induced moment along hull axis)
      - REMANENT (CRM): horizontal component aligned with CONSTRUCTION heading
        (rivets cooled with hull on ways → TRM locked to construction heading)
      - Vertical components: both follow Earth's inclination (wreck site vs
        construction site — slightly different, adding extra angular offset)

    The heading MISMATCH (wreck_heading ≠ construction_heading) creates the
    azimuth deviation that distinguishes wrecks from wells.

    Parameters
    ----------
    total_moment : float
        Total dipole moment magnitude in A·m².
    wreck_heading_deg : float
        Heading (0-360°, CW from N) of the hull as it rests on the lake bed.
    construction_heading_deg : float
        Heading of the hull on the building ways during construction.
    Q : float
        Koenigsberger ratio: |M_remanent| / |M_induced|.
    construction_site : str
        Key into CONSTRUCTION_SITES for historical IGRF at shipyard.

    Returns
    -------
    (mx, my, mz) : tuple of float
        Total moment vector components (A·m²).
    """
    # Partition total moment magnitude between induced and remanent.
    # |M_total|² ≠ |M_ind|² + |M_rem|² in general (vector sum),
    # but we set magnitudes such that |M_ind| + |M_rem| ≈ total_moment
    # for the 1D (parallel) case, scaling appropriately.
    m_ind_mag = total_moment / (1.0 + Q)
    m_rem_mag = Q * m_ind_mag

    # ── Induced component: current Earth field × hull shape anisotropy ──
    # Horizontal direction → along wreck heading (shape aniso concentrates it)
    # Vertical → Earth's inclination
    wreck_incl = math.radians(EARTH_INCL_DEG)
    wreck_rad = math.radians(wreck_heading_deg)

    mind_x = m_ind_mag * math.cos(wreck_incl) * math.sin(wreck_rad)
    mind_y = m_ind_mag * math.cos(wreck_incl) * math.cos(wreck_rad)
    mind_z = m_ind_mag * math.sin(wreck_incl)

    # ── Remanent component: construction-site field × hull heading on ways ──
    site = CONSTRUCTION_SITES.get(construction_site, CONSTRUCTION_SITES["cleveland"])
    c_incl = math.radians(site["incl_deg"])
    c_rad = math.radians(construction_heading_deg)

    mrem_x = m_rem_mag * math.cos(c_incl) * math.sin(c_rad)
    mrem_y = m_rem_mag * math.cos(c_incl) * math.cos(c_rad)
    mrem_z = m_rem_mag * math.sin(c_incl)

    return (mind_x + mrem_x, mind_y + mrem_y, mind_z + mrem_z)


# ── Dipole Field Computation (Total-Field Anomaly) ──────────────────────────

def _compute_dipole_field_vec(
    mx: float,
    my: float,
    mz: float,
    total_depth_m: float,
    grid_extent_m: float,
    n_pixels: int,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> np.ndarray:
    """Compute ΔT anomaly from a dipole with explicit moment vector (mx,my,mz).

    Returns a (n_pixels, n_pixels) array in nanoTesla.
    """
    x = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    y = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    X, Y = np.meshgrid(x, y)
    X = X - center_offset_x
    Y = Y - center_offset_y

    R = np.sqrt(X**2 + Y**2 + total_depth_m**2)
    R = np.maximum(R, 1.0)

    m_dot_r = mx * X + my * Y + mz * total_depth_m
    factor = MU_0 / (4 * math.pi) * 1e9  # → nT

    Bx = factor * (3 * m_dot_r * X / R**5 - mx / R**3)
    By = factor * (3 * m_dot_r * Y / R**5 - my / R**3)
    Bz = factor * (3 * m_dot_r * total_depth_m / R**5 - mz / R**3)

    # Earth field unit vector (for total-field projection)
    incl = math.radians(EARTH_INCL_DEG)
    decl = math.radians(EARTH_DECL_DEG)
    Tx = math.cos(incl) * math.sin(decl)
    Ty = math.cos(incl) * math.cos(decl)
    Tz = math.sin(incl)

    return Bx * Tx + By * Ty + Bz * Tz


def _compute_prolate_spheroid_field_crm(
    mx: float,
    my: float,
    mz: float,
    wreck_heading_deg: float,
    length_m: float,
    total_depth_m: float,
    grid_extent_m: float,
    n_pixels: int,
) -> np.ndarray:
    """Model a steel hull as a chain of sub-dipoles along its keel axis.

    Each sub-dipole carries 1/n_sub of the TOTAL CRM moment vector (mx,my,mz).
    Sub-dipoles are distributed along the hull axis defined by wreck_heading_deg.
    """
    n_sub = max(5, int(length_m / 30))
    sub_mx = mx / n_sub
    sub_my = my / n_sub
    sub_mz = mz / n_sub

    x = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    y = np.linspace(-grid_extent_m / 2, grid_extent_m / 2, n_pixels)
    X, Y = np.meshgrid(x, y)

    total_field = np.zeros((n_pixels, n_pixels), dtype=np.float64)

    theta = math.radians(wreck_heading_deg)
    half_len = length_m / 2
    offsets = np.linspace(-half_len, half_len, n_sub)

    # Earth field unit vector (constant across tile)
    incl = math.radians(EARTH_INCL_DEG)
    decl = math.radians(EARTH_DECL_DEG)
    Tx = math.cos(incl) * math.sin(decl)
    Ty = math.cos(incl) * math.cos(decl)
    Tz = math.sin(incl)
    factor = MU_0 / (4 * math.pi) * 1e9

    for offset in offsets:
        cx = offset * math.sin(theta)
        cy = offset * math.cos(theta)

        Xp = X - cx
        Yp = Y - cy

        R = np.sqrt(Xp**2 + Yp**2 + total_depth_m**2)
        R = np.maximum(R, 1.0)

        m_dot_r = sub_mx * Xp + sub_my * Yp + sub_mz * total_depth_m

        Bx = factor * (3 * m_dot_r * Xp / R**5 - sub_mx / R**3)
        By = factor * (3 * m_dot_r * Yp / R**5 - sub_my / R**3)
        Bz = factor * (3 * m_dot_r * total_depth_m / R**5 - sub_mz / R**3)

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

    Pure vertical dipole — no CRM. A vertical steel casing in Earth's field
    acquires induced magnetization overwhelmingly in the vertical direction
    due to shape anisotropy (demagnetisation factor N_along ≈ 0 for long rod).
    """
    # All moment vertical — identical to v1
    return _compute_dipole_field_vec(
        mx=0.0, my=0.0, mz=moment,
        total_depth_m=total_depth_m,
        grid_extent_m=grid_extent_m,
        n_pixels=n_pixels,
    )


# ── Derived Grid Layers (NSS, VDR, Tilt-Angle) ─────────────────────────────

def _compute_nss(grid: np.ndarray) -> np.ndarray:
    """Normalised Source Strength (analytic signal amplitude).

    NSS = sqrt(dT/dx² + dT/dy² + dT/dz²)
    """
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    dz = ndimage.laplace(grid)
    nss = np.sqrt(dx**2 + dy**2 + dz**2)
    return nss


def _compute_vdr(grid: np.ndarray) -> np.ndarray:
    """Vertical Derivative via Fourier: multiply spectrum by |k|."""
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = 1e-10
    vdr = np.real(np.fft.ifft2(fft * k_mag * 2 * np.pi))
    return vdr


def _compute_tilt_angle(grid: np.ndarray) -> np.ndarray:
    """Tilt Angle (TDR) = atan2(VDR, THDR).

    Amplitude-independent: normalises deep and shallow sources.
    KEY channel for CRM detection — asymmetric CRM dipoles produce
    asymmetric tilt-angle patterns distinguishable from symmetric wells.
    """
    dx = np.gradient(grid, axis=1)
    dy = np.gradient(grid, axis=0)
    thdr = np.sqrt(dx**2 + dy**2)
    vdr = _compute_vdr(grid)
    tilt = np.arctan2(vdr, thdr + 1e-12)
    return tilt


# ── Regional Strike (Geological Noise) Injection ───────────────────────────

def _generate_regional_strike(
    n_pixels: int,
    strike_angle_deg: float,
    amplitude_nt: float,
    wavelength_pixels: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a long-wavelength linear magnetic ridge (geological feature)."""
    x = np.linspace(-1, 1, n_pixels)
    y = np.linspace(-1, 1, n_pixels)
    X, Y = np.meshgrid(x, y)

    theta = math.radians(strike_angle_deg)
    perp = X * math.cos(theta) - Y * math.sin(theta)

    freq = n_pixels / max(wavelength_pixels, 1)
    ridge = amplitude_nt * np.sin(2 * np.pi * freq * perp)
    ridge += rng.normal(0, amplitude_nt * 0.1, ridge.shape)

    return ridge


# ── Satellite Visibility Check (Upward Continuation) ───────────────────────

def _upward_continue(grid: np.ndarray, dz_m: float, dx_m: float) -> np.ndarray:
    """Upward continuation in Fourier domain."""
    fft = np.fft.fft2(grid)
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny, d=dx_m).reshape(-1, 1)
    kx = np.fft.fftfreq(nx, d=dx_m).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    uc_filter = np.exp(-2 * np.pi * k_mag * dz_m)
    return np.real(np.fft.ifft2(fft * uc_filter))


def check_satellite_visibility(
    grid: np.ndarray,
    dx_m: float = 100.0,
    satellite_height_m: float = 400_000.0,
    threshold_nt: float = 0.5,
) -> dict:
    """Test whether a magnetic target is visible from satellite altitude."""
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


# ── Meyer Curvelet Denoising (Python port of nauticuvs FDCT) ────────────────

def _meyer_nu(t: np.ndarray) -> np.ndarray:
    """Meyer smooth partition function ν(t). C³ on [0,1], ν(0)=0, ν(1)=1.

    Identical to nauticuvs::windows::meyer_nu — the polynomial
    t⁴(35 − 84t + 70t² − 20t³) ensures smooth window transitions
    with no spectral ringing.
    """
    t = np.clip(t, 0.0, 1.0)
    return t**4 * (35.0 - 84.0 * t + 70.0 * t**2 - 20.0 * t**3)


def _curvelet_denoise(
    grid: np.ndarray,
    num_scales: int = 5,
    threshold_sigma: float = 2.0,
    hard: bool = True,
) -> np.ndarray:
    """Meyer-window radial bandpass denoising (simplified curvelet).

    Port of the nauticuvs FDCT radial windowing with Meyer ν(t) transitions.
    Omits angular subdivision (not needed for 224×224 synthetic tiles —
    the signal is smooth dipole fields, not sharp edges).

    Tight-frame property: forward → inverse recovers input exactly
    when no thresholding is applied (POU normalization guarantees this).

    Steps:
        1. Zero-pad to power of 2, 2D FFT
        2. Build Meyer radial windows at octave boundaries
        3. Partition-of-unity (POU) normalization
        4. Decompose into scale subbands
        5. Estimate noise from finest scale (MAD estimator)
        6. Threshold detail subbands (hard or soft)
        7. Reconstruct via tight-frame inverse
    """
    ny, nx = grid.shape

    # Zero-pad to next power of 2 (matches nauticuvs forward.rs)
    ny2 = 1 << int(np.ceil(np.log2(max(ny, 2))))
    nx2 = 1 << int(np.ceil(np.log2(max(nx, 2))))
    padded = np.zeros((ny2, nx2), dtype=np.float64)
    padded[:ny, :nx] = grid

    spectrum = np.fft.fft2(padded)

    # Radial frequency grid
    fy = np.fft.fftfreq(ny2).reshape(-1, 1)
    fx = np.fft.fftfreq(nx2).reshape(1, -1)
    r = np.sqrt(fx**2 + fy**2)

    # Scale boundaries: octave spacing from Nyquist (0.5) down
    # Matches nauticuvs scale_boundaries() in windows.rs
    bounds = [0.5 / (2.0 ** j) for j in range(num_scales)]
    bounds.reverse()  # ascending: [smallest_freq, ..., 0.5]

    # Build radial windows using Meyer transitions
    windows = []

    # Coarse window: everything below bounds[1]
    if len(bounds) >= 2:
        t = np.clip((r - bounds[0]) / (bounds[1] - bounds[0] + 1e-15), 0, 1)
        w_coarse = 1.0 - _meyer_nu(t)
    else:
        w_coarse = np.ones_like(r)
    windows.append(w_coarse)

    # Detail windows: intermediate scales
    for j in range(1, len(bounds) - 1):
        lo = bounds[j - 1]
        mid = bounds[j]
        hi = bounds[j + 1] if j + 1 < len(bounds) else 0.5
        rising = _meyer_nu(np.clip((r - lo) / (mid - lo + 1e-15), 0, 1))
        falling = 1.0 - _meyer_nu(np.clip((r - mid) / (hi - mid + 1e-15), 0, 1))
        windows.append(rising * falling)

    # Fine window: everything above bounds[-2]
    if len(bounds) >= 2:
        t = np.clip((r - bounds[-2]) / (bounds[-1] - bounds[-2] + 1e-15), 0, 1)
        w_fine = _meyer_nu(t)
    else:
        w_fine = np.ones_like(r)
    windows.append(w_fine)

    # Partition of unity
    pou = sum(w**2 for w in windows)
    pou = np.maximum(pou, 1e-12)
    pou_inv_sqrt = 1.0 / np.sqrt(pou)

    # Decompose into subbands
    subbands = []
    for w in windows:
        coeffs = np.fft.ifft2(spectrum * w * pou_inv_sqrt)
        subbands.append(coeffs)

    # Estimate noise sigma from finest scale (MAD estimator)
    finest_abs = np.abs(subbands[-1])
    sigma = float(np.median(finest_abs)) / 0.6745
    threshold = threshold_sigma * sigma * np.sqrt(2.0 * np.log(ny2 * nx2))

    # Threshold detail subbands (skip coarse[0] and fine[-1])
    for i in range(1, len(subbands) - 1):
        c = subbands[i]
        if hard:
            subbands[i] = np.where(np.abs(c) >= threshold, c, 0)
        else:
            mag = np.abs(c)
            subbands[i] = np.where(
                mag > 0, c / (mag + 1e-15), 0
            ) * np.maximum(mag - threshold, 0)

    # Reconstruct via tight-frame inverse
    result_spectrum = np.zeros_like(spectrum)
    for w, coeffs in zip(windows, subbands):
        result_spectrum += np.fft.fft2(coeffs) * w * pou_inv_sqrt

    result = np.real(np.fft.ifft2(result_spectrum))
    return result[:ny, :nx]


# ── Track-Line Aliasing (Amplitude-Driven PSF) ─────────────────────────────

def _apply_trackline_aliasing(
    anomaly: np.ndarray,
    survey_line_spacing_m: float = 1000.0,
    grid_extent_m: float = 2000.0,
    n_pixels: int = 224,
    survey_direction_deg: float = 0.0,
) -> np.ndarray:
    """Simulate anisotropic PSF from aeromagnetic survey line gridding.

    Physics: Along flight lines, sampling is dense (~10m). Cross-track,
    sampling is sparse (~1000m line spacing). Gridding (min-curvature/kriging)
    interpolates cross-track, creating an anisotropic point spread function.

    The cross-track blur scales with source strength via the dipole 1/r³
    falloff: stronger anomalies are detected on more adjacent flight lines,
    yielding better cross-track constraint and LESS blur.

    Applies identically to ALL source types (wrecks AND wells). Per user:
    "i dont know that wells dont hit multiple lines if there is enough
    steel under the earth it might. the data is likely in there though"
    """
    dx_m = grid_extent_m / n_pixels
    peak_nt = float(np.max(np.abs(anomaly)))

    # Sub-threshold anomalies (geology-only): no aliasing artefact
    detection_threshold_nt = 3.0
    if peak_nt < detection_threshold_nt:
        return anomaly

    # Number of survey lines detecting this source above threshold.
    # Dipole field falls off as 1/r³ → detection range ∝ (peak/thresh)^(1/3)
    n_lines = (peak_nt / detection_threshold_nt) ** (1.0 / 3.0)
    n_lines = np.clip(n_lines, 1.0, 20.0)

    # Cross-track sigma: inversely proportional to line count.
    # 1 line → sigma ≈ spacing/3 (poorly constrained); many lines → small sigma
    cross_sigma_m = survey_line_spacing_m / (3.0 * n_lines)
    cross_sigma_px = float(np.clip(cross_sigma_m / dx_m, 1.0, 30.0))

    # Along-track: well-sampled (~10m), minimal sigma
    along_sigma_px = 0.5

    # Rotate grid so flight-line direction aligns with x-axis,
    # apply anisotropic Gaussian, rotate back.
    angle = -survey_direction_deg
    rotated = ndimage.rotate(anomaly, angle, reshape=False, order=3, mode='reflect')
    blurred = ndimage.gaussian_filter(rotated, sigma=[cross_sigma_px, along_sigma_px])
    result = ndimage.rotate(blurred, -angle, reshape=False, order=3, mode='reflect')

    return result


# ── Single Tile Generator ──────────────────────────────────────────────────

def generate_tile(
    archetype_key: str,
    basin: str = "central",
    n_pixels: int = 224,
    grid_extent_m: float = 2000.0,
    rng: np.random.Generator | None = None,
    inject_geology: bool = True,
    survey_line_spacing_m: float = 1000.0,
    curvelet_sharpen: bool = True,
    curvelet_scales: int = 5,
) -> dict:
    """Generate a single 3-channel training tile (NSS, VDR, Tilt-Angle).

    v2: Steel and wood wrecks now encode CRM physics —
        Induced (wreck heading) + Remanent (construction heading).
        The heading mismatch creates asymmetric anomalies the model
        can learn to distinguish from symmetric well signatures.

    Returns dict with:
      tile: np.ndarray shape (3, n_pixels, n_pixels)  [CHW]
      label: str (archetype label)
      label_id: int
      metadata: dict (physics params, CRM params, satellite visibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    arch = ARCHETYPES[archetype_key]
    noise_std = BASIN_NOISE.get(basin, 6.0)
    dx_m = grid_extent_m / n_pixels

    # ── Step 1: Generate the target anomaly ─────────────────────────────
    if archetype_key == "GEOLOGY_ONLY":
        anomaly = np.zeros((n_pixels, n_pixels), dtype=np.float64)
        wreck_heading = 0.0
        construction_heading = 0.0
        Q = 0.0
        moment = 0.0
        burial = 0.0
        water_depth = rng.uniform(10, 60)
        length_m = 0.0
        construction_site = "cleveland"
    else:
        moment = rng.uniform(*arch.moment_range)
        burial = rng.uniform(*arch.burial_depth_range)
        water_depth_base = {"western": 12.0, "central": 22.0, "eastern": 45.0}
        water_depth = water_depth_base.get(basin, 22.0) * rng.uniform(0.7, 1.3)
        total_depth = 300.0 + water_depth + burial  # Aero survey at 300 m

        # Random wreck heading (how it rests on the bottom)
        wreck_heading = rng.uniform(0, 360)
        # 40% chance: force perpendicular to geology (decorrelation teaching)
        if rng.random() < 0.4:
            wreck_heading = NE_SW_STRIKE_DEG + 90 + rng.normal(0, 15)

        length_ft = rng.uniform(*arch.length_ft_range)
        length_m = length_ft * 0.3048

        # ── CRM parameters ──────────────────────────────────────────
        Q = rng.uniform(*arch.Q_range)
        # Random construction heading (independent of wreck heading)
        construction_heading = rng.uniform(0, 360)
        # Random construction site (weighted toward Cleveland/Detroit)
        site_weights = np.array([0.30, 0.25, 0.10, 0.10, 0.10, 0.08, 0.07])
        construction_site = rng.choice(SITE_NAMES, p=site_weights)

        if arch.geometry == "prolate_spheroid":
            # ── STEEL_HULL with CRM ──
            mx, my, mz = _compute_crm_moment_vector(
                moment, wreck_heading, construction_heading, Q, construction_site,
            )
            anomaly = _compute_prolate_spheroid_field_crm(
                mx, my, mz,
                wreck_heading, length_m, total_depth,
                grid_extent_m, n_pixels,
            )

        elif arch.geometry == "cylinder":
            # ── WELLHEAD: pure vertical, no CRM ──
            anomaly = _compute_cylinder_field(
                moment, total_depth, rng.uniform(10, 50),
                grid_extent_m, n_pixels,
            )

        else:
            # ── WOOD_CARGO with weak CRM ──
            mx, my, mz = _compute_crm_moment_vector(
                moment, wreck_heading, construction_heading, Q, construction_site,
            )
            anomaly = _compute_dipole_field_vec(
                mx, my, mz,
                total_depth_m=total_depth,
                grid_extent_m=grid_extent_m,
                n_pixels=n_pixels,
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

    # ── Step 3b: Track-line aliasing (amplitude-driven anisotropic PSF) ─
    survey_direction_deg = float(rng.choice([0.0, 45.0, 90.0, 135.0])) + rng.normal(0, 5)
    anomaly = _apply_trackline_aliasing(
        anomaly, survey_line_spacing_m, grid_extent_m, n_pixels,
        survey_direction_deg,
    )

    # ── Step 3c: Curvelet edge-preserving denoising ─────────────────────
    if curvelet_sharpen:
        anomaly = _curvelet_denoise(anomaly, num_scales=curvelet_scales)

    # ── Step 4: Compute 3 derived layers ────────────────────────────────
    nss = _compute_nss(anomaly)
    vdr = _compute_vdr(anomaly)
    tilt = _compute_tilt_angle(anomaly)

    tile = np.stack([nss, vdr, tilt], axis=0).astype(np.float32)  # (3, H, W)

    # ── Step 5: Satellite visibility check ──────────────────────────────
    sat_check = check_satellite_visibility(anomaly, dx_m)

    # ── Step 6: Axis & CRM metrics ──────────────────────────────────────
    strike_angle_used = strike_angle if inject_geology else NE_SW_STRIKE_DEG
    angle_diff = abs(wreck_heading - strike_angle_used)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    # CRM heading mismatch: how far construction heading is from wreck heading
    heading_mismatch = abs(construction_heading - wreck_heading)
    if heading_mismatch > 180:
        heading_mismatch = 360 - heading_mismatch

    metadata = {
        "archetype": archetype_key,
        "basin": basin,
        "moment": moment,
        "wreck_heading_deg": wreck_heading,
        "construction_heading_deg": construction_heading,
        "heading_mismatch_deg": heading_mismatch,
        "Q_koenigsberger": Q,
        "construction_site": construction_site,
        "burial_depth_m": burial,
        "water_depth_m": water_depth,
        "length_m": length_m,
        "geology_strike_deg": strike_angle_used,
        "axis_offset_from_geology_deg": angle_diff,
        "noise_std_nt": noise_std,
        "survey_direction_deg": survey_direction_deg,
        "survey_line_spacing_m": survey_line_spacing_m,
        "curvelet_sharpened": curvelet_sharpen,
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
    survey_line_spacing_m: float = 1000.0,
    curvelet_sharpen: bool = True,
    curvelet_scales: int = 5,
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
        logger.info("Generating %d %s tiles (v2 CRM-physics)...", count, archetype_key)
        for i in range(count):
            basin = basins[rng.integers(0, len(basins))]
            try:
                result = generate_tile(
                    archetype_key, basin, n_pixels, grid_extent_m, rng,
                    survey_line_spacing_m=survey_line_spacing_m,
                    curvelet_sharpen=curvelet_sharpen,
                    curvelet_scales=curvelet_scales,
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
            # Thermal break: free memory + cool CPU every 100 tiles
            if generated % 100 == 0:
                gc.collect()
                time.sleep(0.3)

    tiles_arr = np.array(all_tiles, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    logger.info("Generated %d total tiles (v2 CRM-physics): %s", len(tiles_arr), {
        "GEOLOGY_ONLY": int(np.sum(labels_arr == 0)),
        "STEEL_HULL": int(np.sum(labels_arr == 1)),
        "WOOD_CARGO": int(np.sum(labels_arr == 2)),
        "WELLHEAD": int(np.sum(labels_arr == 3)),
    })

    # CRM statistics
    steel_meta = [m for m in all_meta if m["archetype"] == "STEEL_HULL"]
    if steel_meta:
        mismatches = [m["heading_mismatch_deg"] for m in steel_meta]
        q_vals = [m["Q_koenigsberger"] for m in steel_meta]
        logger.info("STEEL_HULL CRM stats: heading_mismatch=%.1f±%.1f°, Q=%.2f±%.2f",
                    np.mean(mismatches), np.std(mismatches),
                    np.mean(q_vals), np.std(q_vals))

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

    parser = argparse.ArgumentParser(
        description="WH2K Synthetic Tile Generator v2 (CRM-Physics)")
    parser.add_argument("--n-steel", type=int, default=500)
    parser.add_argument("--n-wood", type=int, default=500)
    parser.add_argument("--n-wellhead", type=int, default=500)
    parser.add_argument("--n-geology", type=int, default=5000)
    parser.add_argument("--survey-line-spacing", type=float, default=1000.0,
                        help="Survey line spacing in metres (default: 1000)")
    parser.add_argument("--curvelet-scales", type=int, default=5,
                        help="Number of curvelet decomposition scales")
    parser.add_argument("--no-curvelet", action="store_true",
                        help="Disable curvelet edge-preserving sharpening")
    parser.add_argument("--pixels", type=int, default=224,
                        help="Tile size (pixels). 224 for ResNet-18.")
    parser.add_argument("--extent-m", type=float, default=2000.0,
                        help="Grid extent in metres (2000 = 2km × 2km chip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="wreck_hunting_ml/data/synthetic")
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
        survey_line_spacing_m=args.survey_line_spacing,
        curvelet_sharpen=not args.no_curvelet,
        curvelet_scales=args.curvelet_scales,
    )

    # Save tiles (same format as v1 — drop-in replacement)
    np.savez_compressed(
        out_dir / "synthetic_tiles.npz",
        tiles=result["tiles"],
        labels=result["labels"],
    )

    # Save metadata (includes CRM params for diagnostics)
    with open(out_dir / "synthetic_metadata.json", "w") as f:
        json.dump(result["metadata"], f, indent=2, default=str)

    logger.info("Saved to %s/synthetic_tiles.npz (%d tiles, v2 CRM-physics)",
                out_dir, len(result["labels"]))

    # Print CRM summary
    steel_meta = [m for m in result["metadata"] if m["archetype"] == "STEEL_HULL"]
    if steel_meta:
        mismatches = [m["heading_mismatch_deg"] for m in steel_meta]
        logger.info("CRM heading mismatch distribution:")
        logger.info("  min=%.1f°  mean=%.1f°  max=%.1f°",
                    min(mismatches), np.mean(mismatches), max(mismatches))


if __name__ == "__main__":
    main()
