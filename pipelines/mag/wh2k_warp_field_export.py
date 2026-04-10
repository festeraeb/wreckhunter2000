"""
WreckHunter 2000 — LORAN-C Warp Field Exporter
================================================
Converts the IDW rubber-sheet correction (computed from datum_anchors.json)
into a dense, regular lat/lon grid saved as loran_warp_field.json.

WHY JSON INSTEAD OF RECOMPUTING IDW PER-PING
---------------------------------------------
The harvester applies this correction to every raw ping (potentially millions).
IDW is O(n_anchors) per query which is fine for small counts but adds up.
A pre-computed grid of ~1000×2000 cells (2km spacing) covers all of Erie in
~2 MB JSON and makes warp application O(1) per ping via nearest-neighbour lookup.

Grid cells are spaced at GRID_SPACING_KM (default 2 km).  Sufficient precision
for LORAN-C uncertainty (~300 m); can be tightened to 0.5 km if anchor density
grows.

OUTPUT: scripts/loran_warp_field.json
    {
        "generated":   "2026-03-16T12:00:00",
        "anchor_count": 6,
        "grid_spacing_km": 2.0,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "lat_centers": [...],   // 1-D array, length n_lat
        "lon_centers": [...],   // 1-D array, length n_lon
        "dlat_deg":    [[...]],  // 2-D array (n_lat, n_lon) — warp in latitude
        "dlon_deg":    [[...]],  // 2-D array (n_lat, n_lon) — warp in longitude
        "anchors_used": [...]   // summary of source anchors
    }

Usage:
    python -W ignore scripts/wh2k_warp_field_export.py
    python -W ignore scripts/wh2k_warp_field_export.py --spacing-km 0.5  # finer
    python -W ignore scripts/wh2k_warp_field_export.py --lake superior    # other lake
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Lake bounding boxes ────────────────────────────────────────────────────

LAKE_BBOX = {
    "erie":     (40.80, -83.60, 42.95, -78.80),
    "huron":    (43.00, -84.80, 46.50, -79.50),
    "superior": (46.30, -92.10, 49.00, -84.35),
    "michigan": (41.60, -88.00, 46.10, -84.80),
    "ontario":  (43.10, -79.90, 44.30, -76.00),
}

ANCHOR_FILE  = REPO / "scripts" / "datum_anchors.json"
WARP_OUT     = REPO / "scripts" / "loran_warp_field.json"

# ── IDW implementation ─────────────────────────────────────────────────────

IDW_POWER = 2.0
MIN_ANCHOR_DIST_M = 10.0   # avoid division by zero at anchor centre


def _haversine_m(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _idw_shift(
    query_lat: float,
    query_lon: float,
    anchor_lats: np.ndarray,
    anchor_lons: np.ndarray,
    shift_n_m: np.ndarray,   # northward correction in metres
    shift_e_m: np.ndarray,   # eastward correction in metres
    power: float = IDW_POWER,
) -> tuple[float, float]:
    """
    Inverse-distance-weighted interpolation of shift vectors.
    Returns (shift_north_m, shift_east_m) at the query point.
    """
    dists = np.array([
        max(_haversine_m(query_lat, query_lon, la, lo), MIN_ANCHOR_DIST_M)
        for la, lo in zip(anchor_lats, anchor_lons)
    ])
    weights = 1.0 / (dists ** power)
    w_sum = weights.sum()
    return float((weights * shift_n_m).sum() / w_sum), float((weights * shift_e_m).sum() / w_sum)


# ── Main ───────────────────────────────────────────────────────────────────

def build_warp_field(
    lake: str = "erie",
    spacing_km: float = 2.0,
    anchor_file: Path = ANCHOR_FILE,
    out_file: Path = WARP_OUT,
) -> Path:

    # ── Load anchors ───────────────────────────────────────────────────────
    if not anchor_file.exists():
        logger.error("datum_anchors.json not found at %s", anchor_file)
        logger.error("Run wh2k_wreck_anchor_verifier.py first to build anchor file.")
        sys.exit(1)

    all_anchors = json.loads(anchor_file.read_text(encoding="utf-8"))
    verified = [a for a in all_anchors if a.get("verified") is True]

    if not verified:
        logger.error("No verified anchors found in %s", anchor_file)
        logger.error("Anchors must have 'verified': true  and 'shift_m' dict.")
        sys.exit(1)

    logger.info("Loaded %d / %d verified anchors", len(verified), len(all_anchors))

    # Build numpy arrays for IDW
    a_lats  = np.array([a["verified_gps"][0] for a in verified])
    a_lons  = np.array([a["verified_gps"][1] for a in verified])
    # shift_m = GPS - survey_pos  →  adds this to raw positions to correct them
    a_sn    = np.array([a["shift_m"]["north"] for a in verified])
    a_se    = np.array([a["shift_m"]["east"]  for a in verified])

    for i, a in enumerate(verified):
        logger.info("  %s  GPS=(%.4f,%.4f)  shift N%+.0f E%+.0f m",
                    a["name"], a_lats[i], a_lons[i], a_sn[i], a_se[i])

    # ── Build grid ─────────────────────────────────────────────────────────
    bbox = LAKE_BBOX.get(lake, LAKE_BBOX["erie"])
    lat_min, lon_min, lat_max, lon_max = bbox

    spacing_deg_lat = spacing_km / 111.32
    cos_mid = math.cos(math.radians((lat_min + lat_max) / 2))
    spacing_deg_lon = spacing_km / (111.32 * cos_mid)

    lat_centers = np.arange(lat_min, lat_max + spacing_deg_lat * 0.5, spacing_deg_lat)
    lon_centers = np.arange(lon_min, lon_max + spacing_deg_lon * 0.5, spacing_deg_lon)

    n_lat = len(lat_centers)
    n_lon = len(lon_centers)
    logger.info("Grid: %d lat × %d lon = %d cells  (%.1f km spacing)",
                n_lat, n_lon, n_lat * n_lon, spacing_km)

    dlat_deg = np.zeros((n_lat, n_lon), dtype=np.float32)
    dlon_deg = np.zeros((n_lat, n_lon), dtype=np.float32)

    # Vectorised IDW — compute for all grid cells
    logger.info("Computing IDW warp field …")
    for ri, lat in enumerate(lat_centers):
        if ri % 50 == 0:
            logger.info("  Row %d / %d …", ri, n_lat)
        cos_lat = math.cos(math.radians(lat))
        for ci, lon in enumerate(lon_centers):
            sn, se = _idw_shift(lat, lon, a_lats, a_lons, a_sn, a_se)
            # Convert metres → degrees
            dlat_deg[ri, ci] = sn / 111_320.0
            dlon_deg[ri, ci] = se / (111_320.0 * cos_lat)

    # ── Serialise ──────────────────────────────────────────────────────────
    field = {
        "generated":      datetime.now(timezone.utc).isoformat(),
        "lake":           lake,
        "anchor_count":   len(verified),
        "grid_spacing_km": spacing_km,
        "bbox":           list(bbox),
        "lat_centers":    lat_centers.tolist(),
        "lon_centers":    lon_centers.tolist(),
        "dlat_deg":       dlat_deg.tolist(),
        "dlon_deg":       dlon_deg.tolist(),
        "anchors_used": [
            {
                "id":   a["id"],
                "name": a["name"],
                "type": a.get("type", "unknown"),
                "gps":  a["verified_gps"],
                "shift_m": a["shift_m"],
            }
            for a in verified
        ],
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(field, separators=(",", ":")), encoding="utf-8")

    size_kb = out_file.stat().st_size // 1024
    logger.info("Warp field exported → %s  (%d KB, %d cells)", out_file.name, size_kb, n_lat * n_lon)

    # Print corner-point spot-check
    for lat, lon, label in [
        (41.5, -82.5, "Central Erie"),
        (42.4, -81.0, "Off Canadian shore"),
        (42.0, -80.5, "East basin"),
    ]:
        sn, se = _idw_shift(lat, lon, a_lats, a_lons, a_sn, a_se)
        logger.info("  Spot-check %-24s  N%+.1f m  E%+.1f m", label, sn, se)

    return out_file


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export LORAN-C IDW warp field to dense JSON grid"
    )
    parser.add_argument("--lake",       default="erie", choices=list(LAKE_BBOX))
    parser.add_argument("--spacing-km", type=float, default=2.0,
                        help="Grid spacing in km (default 2.0; use 0.5 for finer)")
    parser.add_argument("--anchor-file", default=str(ANCHOR_FILE))
    parser.add_argument("--out",         default=str(WARP_OUT))
    args = parser.parse_args(argv)

    out = build_warp_field(
        lake=args.lake,
        spacing_km=args.spacing_km,
        anchor_file=Path(args.anchor_file),
        out_file=Path(args.out),
    )
    print(f"\nWarp field ready: {out}")
    print("Pass to harvester with:  --warp-json scripts/loran_warp_field.json")


if __name__ == "__main__":
    main()
