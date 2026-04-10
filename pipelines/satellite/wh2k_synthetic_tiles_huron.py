"""
WreckHunter 2000 — Lake Huron Synthetic Tile Generator
========================================================
Wraps the CRM-physics v2 engine with Huron-specific parameters.

Key differences from Lake Erie (v2):
  - Magnetic inclination: 72° (vs Erie 68°) — more northerly latitude
  - Magnetic declination: -10° (vs Erie -9°)
  - Geological strike: N-S ~5° (Canadian Shield edge) vs NE-SW 45° (Erie)
  - Basin noise: 10-30 nT (vs Erie 3.5-12 nT) — Precambrian basement
  - Construction sites: added Collingwood, Midland, Owen Sound, Sarnia
  - Basins: southern / central / northern / georgian (vs western / central / eastern)

Output: NPZ tiles (NSS + VDR + Tilt) sized 224×224 — drop-in for training.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_and_patch_v2():
    """Load the v2 CRM-physics generator and patch in Huron constants."""
    import sys
    v2_path = Path(__file__).parent / "wh2k_synthetic_tiles_v2.py"
    spec = importlib.util.spec_from_file_location("wh2k_synthetic_tiles_v2", v2_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wh2k_synthetic_tiles_v2"] = mod
    spec.loader.exec_module(mod)

    # ── Lake Huron magnetic field parameters ──
    mod.EARTH_INCL_DEG = 72.0       # Higher than Erie (68°) — more northerly
    mod.EARTH_DECL_DEG = -10.0      # Slightly more westerly than Erie (-9°)

    # ── N-S Canadian Shield geological strike ──
    # Erie has NE-SW (45°) from Appalachian structures.
    # Huron's eastern and northern shores are Canadian Shield edge:
    # predominantly N-S to NNW-SSE strike (~0-10°).
    mod.NE_SW_STRIKE_DEG = 5.0

    # ── Basin noise — Canadian Shield = strong geological magnetic signal ──
    # Southern Huron (sedimentary, like Erie): moderate.
    # Northern Huron + Georgian Bay (Precambrian outcrop): very high.
    mod.BASIN_NOISE = {
        "southern":  10.0,      # Sedimentary — similar to Erie eastern basin
        "central":   18.0,      # Transitional — Shield influence begins
        "northern":  25.0,      # Strong Shield geology below flight level
        "georgian":  30.0,      # Georgian Bay — dense Precambrian outcrop
    }

    # ── Shipyards relevant to Lake Huron vessels ──
    # IGRF values circa 1880-1920 at each shipyard.
    # Collingwood, Midland, Owen Sound were major Huron builders.
    # Cleveland kept as hidden fallback — v2 line 206 has a hardcoded
    # CONSTRUCTION_SITES["cleveland"] fallback in _compute_crm_moment_vector,
    # and line 657 uses "cleveland" as default for GEOLOGY_ONLY tiles.
    # SITE_NAMES (used by rng.choice) must be exactly 7 to match site_weights.
    mod.CONSTRUCTION_SITES = {
        "collingwood": {"lat": 44.50, "incl_deg": 73.5, "decl_deg": -6.0},
        "midland":     {"lat": 44.75, "incl_deg": 73.5, "decl_deg": -6.0},
        "owen_sound":  {"lat": 44.57, "incl_deg": 73.5, "decl_deg": -6.0},
        "port_huron":  {"lat": 43.00, "incl_deg": 72.5, "decl_deg": -5.0},
        "sarnia":      {"lat": 42.97, "incl_deg": 72.5, "decl_deg": -5.5},
        "detroit":     {"lat": 42.33, "incl_deg": 71.5, "decl_deg": -4.5},
        "bay_city":    {"lat": 43.59, "incl_deg": 73.0, "decl_deg": -3.5},
        # Fallback entry — not in SITE_NAMES, never selected by rng.choice,
        # but needed because v2 hardcodes CONSTRUCTION_SITES["cleveland"]
        # as default in _compute_crm_moment_vector and GEOLOGY_ONLY branch.
        "cleveland":   {"lat": 41.50, "incl_deg": 71.0, "decl_deg": -4.0},
    }
    # Only the first 7 go into SITE_NAMES (matches hardcoded site_weights len=7)
    mod.SITE_NAMES = list(mod.CONSTRUCTION_SITES.keys())[:7]

    # ── WOOD_CARGO archetype override — boiler TRM physics ──
    # Huron has many wooden steamers/tugs with cast-iron boilers + engines.
    # Boilers acquire thermoremanent magnetization (TRM) from heating/cooling
    # cycles — much stronger than CRM from iron fittings alone.
    # A 20-50 ton Scotch boiler + 10-30 ton engine complex creates anomalies
    # that partially overlap with small steel hulls at the lower end.
    wood = mod.ARCHETYPES["WOOD_CARGO"]
    mod.ARCHETYPES["WOOD_CARGO"] = mod.Archetype(
        label=wood.label,
        label_id=wood.label_id,
        length_ft_range=(70, 300),          # include small tugs (70ft) to large steamers
        moment_range=(1e6, 1e7),            # upper end: steamer w/ boiler+engine TRM
        geometry=wood.geometry,
        burial_depth_range=wood.burial_depth_range,
        sat_visible_expected=wood.sat_visible_expected,
        description="70-300ft wooden vessel — iron fittings (sailing) to boiler+engine TRM (steamer)",
        Q_range=(0.05, 0.8),               # upper end: TRM from cast-iron boilers
    )

    return mod


def generate_huron_tiles(
    n_steel: int = 500,
    n_wood: int = 500,
    n_wellhead: int = 200,
    n_geology: int = 5000,
    n_pixels: int = 224,
    grid_extent_m: float = 2000.0,
    seed: int = 42,
    survey_line_spacing_m: float = 1000.0,
    curvelet_sharpen: bool = True,
    curvelet_scales: int = 5,
) -> dict:
    """Generate Huron-physics synthetic training tiles.

    Returns dict with tiles (N,3,224,224), labels (N,), metadata (list[dict]).
    """
    gen = _load_and_patch_v2()

    basins = ["southern", "central", "northern", "georgian"]

    return gen.generate_training_tiles(
        n_steel=n_steel,
        n_wood=n_wood,
        n_wellhead=n_wellhead,
        n_geology=n_geology,
        n_pixels=n_pixels,
        grid_extent_m=grid_extent_m,
        basins=basins,
        seed=seed,
        survey_line_spacing_m=survey_line_spacing_m,
        curvelet_sharpen=curvelet_sharpen,
        curvelet_scales=curvelet_scales,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Generate Lake Huron CRM-physics synthetic tiles")
    p.add_argument("--n-steel",    type=int, default=500)
    p.add_argument("--n-wood",     type=int, default=500)
    p.add_argument("--n-wellhead", type=int, default=200)
    p.add_argument("--n-geology",  type=int, default=5000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     type=str,
                   default=str(Path(__file__).resolve().parents[1]
                               / "wreck_hunting_ml" / "data" / "synthetic"
                               / "huron_synthetic_tiles.npz"))
    args = p.parse_args()

    logger.info("Generating Huron synthetic tiles with CRM physics...")
    logger.info("  EARTH_INCL=72°, EARTH_DECL=-10°, STRIKE=5° (N-S Shield)")
    logger.info("  Basins: southern(10nT), central(18nT), northern(25nT), georgian(30nT)")

    data = generate_huron_tiles(
        n_steel=args.n_steel,
        n_wood=args.n_wood,
        n_wellhead=args.n_wellhead,
        n_geology=args.n_geology,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        tiles=data["tiles"],
        labels=data["labels"],
    )

    size_mb = out.stat().st_size / 1e6
    logger.info("Saved %d Huron tiles → %s (%.1f MB)", len(data["tiles"]), out, size_mb)
    logger.info("Label distribution: GEOLOGY=%d, STEEL=%d, WOOD=%d, WELL=%d",
                int(np.sum(data["labels"] == 0)),
                int(np.sum(data["labels"] == 1)),
                int(np.sum(data["labels"] == 2)),
                int(np.sum(data["labels"] == 3)))


if __name__ == "__main__":
    main()
