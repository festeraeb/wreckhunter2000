"""
WreckHunter 2000 — Wreck-Derived Anchor Verifier
=================================================
Uses known wrecks with confirmed dive-GPS positions as LORAN-C
rubber-sheet anchor points.

The Logic
---------
Each known wreck gives us THREE positions:

    A  verified_gps  — dive-confirmed GPS (AWOIS ROV, Shipwreck World, NOAA)
    B  aeromag_peak  — peak of the raw flight-line magnetic anomaly (computed here)
    C  loran_reported — original LORAN-C coords from historical logs (optional)

    shift_vector = A - B  (or A - C if we have original LORAN coords)

This is BETTER than a harbor lighthouse anchor because:
    1. The wreck is distributed MID-LAKE — fills the triangulation gap
    2. The anomaly peak is the same physical object the LORAN-C was measuring
    3. No need to assume the shore-structure correction transfers to deep water

Promotion Criteria (anchor becomes verified=True)
--------------------------------------------------
    • aeromag offset from GPS < MAX_AEROMAG_OFFSET_M  (default 500 m)
      — normal dipole shift for hull sizes 240–444 ft at this mag latitude
    • raw ping count within search radius ≥ MIN_PINGS  (default 20)
    • absolute nT range ≥ MIN_NT_RANGE  (default 15 nT for a steel hull)

Usage
-----
    # Dry-run: report only, do not modify datum_anchors.json
    python -W ignore scripts/wh2k_wreck_anchor_verifier.py \\
        --raw-csv magnetic_data/raw/local_mage_csv/nrcan_OH_4039B.csv \\
        --dry-run

    # Promote and persist new anchors
    python -W ignore scripts/wh2k_wreck_anchor_verifier.py \\
        --raw-csv magnetic_data/raw/local_mage_csv/nrcan_OH_4039B.csv

    # Also test with Huron gsc data (for Carruthers area prep)
    python -W ignore scripts/wh2k_wreck_anchor_verifier.py \\
        --raw-csv magnetic_data/raw/local_mage_csv/gsc_huron_nrcan.csv \\
        --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Import provenance helpers — standardise to anomaly space before any comparison
try:
    from wh2k_data_provenance import check_source, standardise_to_anomaly, igrf_approx_nT
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

SEARCH_RADIUS_KM      = 3.0    # raw ping extraction radius around each wreck
GRID_SPACING_M        = 50.0   # tighter grid than ghost zoom — want peak precision
MAX_AEROMAG_OFFSET_M  = 500.0  # max GPS-to-peak offset to accept as valid anchor
MIN_PINGS             = 20     # minimum raw pings needed to estimate a peak
MIN_NT_RANGE          = 15.0   # nT range threshold — below this is probably noise

ANCHOR_FILE = REPO / "scripts" / "datum_anchors.json"

# ── Known wreck registry ───────────────────────────────────────────────────
# verified_gps = modern dive-confirmed GPS
# loran_reported = original LORAN-C/NAD27 coordinates from historical report (if known)
# source = primary source for GPS verification
KNOWN_WRECKS = [
    {
        "id":           "wreck_merida",
        "name":         "SS Merida",
        "verified_gps": [42.014, -80.851],
        "length_ft":    408,
        "depth_ft":     64,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA chart-confirmed",
        "loran_reported": None,   # fill when historical record available
        "notes": "Sank 1916, central-east Erie. Strong mid-lake anchor.",
    },
    {
        "id":           "wreck_lr_doty",
        "name":         "SS L.R. Doty",
        "verified_gps": [41.983, -81.631],
        "length_ft":    285,
        "depth_ft":     68,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "Central Basin west. 68 ft depth — anomaly should be clean.",
    },
    {
        "id":           "wreck_admiral",
        "name":         "SS Admiral",
        "verified_gps": [42.025, -81.150],
        "length_ft":    296,
        "depth_ft":     58,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "Central Basin. Shallow (58ft) — good survey signal.",
    },
    {
        "id":           "wreck_craftsman",
        "name":         "SS Craftsman",
        "verified_gps": [42.118, -81.441],
        "length_ft":    444,
        "depth_ft":     72,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "Longest hull in validation set (444 ft). Strong signal expected.",
    },
    {
        "id":           "wreck_whaleback_consort",
        "name":         "Whaleback Consort",
        "verified_gps": [42.427, -80.813],
        "length_ft":    308,
        "depth_ft":     59,
        "type":         "steel_whaleback",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "NORTH SHORE side — critical geometry. Only wreck near Canadian waters. "
                 "If verified, fills the triangulation gap for Ghost-1.",
    },
    {
        "id":           "wreck_minnedosa",
        "name":         "SS Minnedosa",
        "verified_gps": [42.051, -81.249],
        "length_ft":    240,
        "depth_ft":     55,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "Smallest hull in set (240 ft). Validation of minimum detection threshold.",
    },
    {
        "id":           "wreck_clarion",
        "name":         "SS Clarion",
        "verified_gps": [41.980, -81.520],
        "length_ft":    265,
        "depth_ft":     61,
        "type":         "steel_freighter",
        "source":       "AWOIS NOAA",
        "loran_reported": None,
        "notes": "Western central basin.",
    },
]

# ── Utility ────────────────────────────────────────────────────────────────

def _haversine_m(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _metres_to_deg(metres: float, lat_ref: float) -> tuple[float, float]:
    dlat = metres / 111_320.0
    dlon = metres / (111_320.0 * math.cos(math.radians(lat_ref)))
    return dlat, dlon


# ── Raw CSV loading ────────────────────────────────────────────────────────

def _detect_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    fl = [f.upper() for f in fieldnames]
    def _find(*candidates) -> str:
        for c in candidates:
            if c in fl:
                return fieldnames[fl.index(c)]
        raise KeyError(f"None of {candidates} found in header: {fieldnames}")
    lon = _find("LON", "LONG", "LONGITUDE", "SUR_LONG83", "X")
    lat = _find("LAT", "LATITUDE",  "SUR_LAT83",  "Y")
    mag = _find("TMF", "MAG", "RESIDMAG", "RESIDUAL", "TMAGF",
                "CORR_MAG", "MFIELD", "MGNT", "MAGFIELD", "MAG_FIELD")
    return lon, lat, mag


def load_raw_pings(
    csv_path: Path,
    centre_lat: float,
    centre_lon: float,
    radius_km: float = SEARCH_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pings from bbox and return them in ANOMALY space.

    Standardisation is automatic:
    - OH_4039B (mag_anomaly centred near 0)  → pass-through
    - gsc_huron (absolute TMF ~57,875 nT)    → subtract per-point IGRF approximation
    - Any other absolute source detected      → same IGRF subtraction

    This guarantees that nT_range and anomaly peak comparisons are apples-to-apples
    across Erie and Huron regardless of which CSV is passed in.
    """
    dlat, dlon = _metres_to_deg(radius_km * 1000, centre_lat)
    lat_min = centre_lat - dlat;  lat_max = centre_lat + dlat
    lon_min = centre_lon - dlon;  lon_max = centre_lon + dlon

    lons, lats, mags = [], [], []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        col_lon, col_lat, col_mag = _detect_columns(reader.fieldnames or [])
        for row in reader:
            try:
                lat = float(row[col_lat]);  lon = float(row[col_lon])
                mag = float(row[col_mag])
                if mag <= -9990:   # nodata sentinel
                    continue
            except (ValueError, KeyError):
                continue
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                lons.append(lon); lats.append(lat); mags.append(mag)

    lons_a = np.array(lons)
    lats_a = np.array(lats)
    mags_a = np.array(mags)

    if len(mags_a) == 0:
        return lons_a, lats_a, mags_a

    # ── Auto-detect processing level and standardise to anomaly ───────────
    mean_val = float(mags_a.mean())
    if abs(mean_val) > 5000:
        # Absolute TMF — subtract per-point IGRF approximation
        if _PROVENANCE_AVAILABLE:
            mags_a = standardise_to_anomaly(lons_a, lats_a, mags_a, "absolute_tmf")
        else:
            # Fallback: subtract dataset mean (removes regional field approximately)
            mags_a = mags_a - mean_val
        logger.debug(
            "  Auto-standardised %s from absolute TMF (mean=%.0f nT) to anomaly space",
            csv_path.name, mean_val
        )
    # else: already anomaly, pass through unchanged

    return lons_a, lats_a, mags_a


# ── Aeromag peak finder ────────────────────────────────────────────────────

def find_aeromag_peak(
    lons: np.ndarray,
    lats: np.ndarray,
    mags: np.ndarray,
    centre_lat: float,
    centre_lon: float,
    spacing_m: float = GRID_SPACING_M,
    radius_km: float = SEARCH_RADIUS_KM,
) -> tuple[float, float, float, np.ndarray]:
    """
    Grid raw pings, find the absolute-maximum anomaly peak.

    Returns
    -------
    peak_lat, peak_lon  — geographic position of peak
    peak_nT             — absolute magnitude at peak
    grid_mag            — 2-D gridded array (for diagnostics)
    """
    n_cells = int(2 * radius_km * 1000 / spacing_m)
    dlat, dlon = _metres_to_deg(radius_km * 1000, centre_lat)

    lon_vec = np.linspace(centre_lon - dlon, centre_lon + dlon, n_cells)
    lat_vec = np.linspace(centre_lat - dlat, centre_lat + dlat, n_cells)
    glon, glat = np.meshgrid(lon_vec, lat_vec)

    cos_ref = math.cos(math.radians(centre_lat))
    x_pts = (lons - centre_lon) * 111_320.0 * cos_ref
    y_pts = (lats - centre_lat) * 111_320.0
    x_grid = (glon - centre_lon) * 111_320.0 * cos_ref
    y_grid = (glat - centre_lat) * 111_320.0

    # Regional (mean) field removal — subtract median to expose local anomaly
    mag_residual = mags - np.median(mags)

    grid_mag = griddata(
        np.column_stack([x_pts, y_pts]),
        mag_residual,
        np.column_stack([x_grid.ravel(), y_grid.ravel()]),
        method="linear",
    ).reshape(glon.shape).astype(np.float32)

    nan_mask = np.isnan(grid_mag)
    if nan_mask.any():
        grid_nn = griddata(
            np.column_stack([x_pts, y_pts]),
            mag_residual,
            np.column_stack([x_grid.ravel(), y_grid.ravel()]),
            method="nearest",
        ).reshape(glon.shape).astype(np.float32)
        grid_mag[nan_mask] = grid_nn[nan_mask]

    # Find absolute maximum
    abs_grid = np.abs(grid_mag)
    r, c = divmod(int(abs_grid.argmax()), abs_grid.shape[1])
    peak_lat = float(glat[r, c])
    peak_lon = float(glon[r, c])
    peak_nT  = float(grid_mag[r, c])

    return peak_lat, peak_lon, peak_nT, grid_mag


# ── Dipole midpoint (optional refinement) ─────────────────────────────────

def find_dipole_midpoint(grid_mag: np.ndarray, glat: np.ndarray, glon: np.ndarray) -> tuple[float, float]:
    """
    For a steel hull at mid-magnetic-latitude, the anomaly is a dipole:
    a positive lobe and a negative lobe.  The TRUE hull position is the
    midpoint between them, not the positive peak.

    Returns (mid_lat, mid_lon) — may be NaN if dipole pattern not clear.
    """
    pos_r, pos_c = divmod(int(grid_mag.argmax()), grid_mag.shape[1])
    neg_r, neg_c = divmod(int(grid_mag.argmin()), grid_mag.shape[1])

    # Only trust midpoint if the negative lobe is ≥ 30% of positive amplitude
    pos_amp = float(grid_mag[pos_r, pos_c])
    neg_amp = float(grid_mag[neg_r, neg_c])
    if abs(neg_amp) < 0.30 * abs(pos_amp):
        return float("nan"), float("nan")

    mid_lat = (float(glat[pos_r, pos_c]) + float(glat[neg_r, neg_c])) / 2
    mid_lon = (float(glon[pos_r, pos_c]) + float(glon[neg_r, neg_c])) / 2
    return mid_lat, mid_lon


# ── Expected dipole offset calculator ─────────────────────────────────────

def expected_dipole_offset_m(depth_ft: float, length_ft: float) -> float:
    """
    Rule-of-thumb dipole offset for Lake Erie magnetic inclination (~69°).
    At 69° inclination, dipole half-separation ≈ 0.7 × depth.
    For a long hull the anomaly also spreads along the hull axis.

    Returns expected offset (m) between GPS centre and positive anomaly peak.
    """
    depth_m  = depth_ft * 0.3048
    length_m = length_ft * 0.3048
    return 0.70 * depth_m + 0.10 * (length_m / 2)


# ── Load / save datum anchors ──────────────────────────────────────────────

def load_anchors() -> list[dict]:
    if ANCHOR_FILE.exists():
        return json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    return []


def save_anchors(anchors: list[dict]) -> None:
    ANCHOR_FILE.write_text(json.dumps(anchors, indent=2), encoding="utf-8")


# ── Main verifier ──────────────────────────────────────────────────────────

def verify_wreck_as_anchor(
    wreck: dict,
    csv_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    For a single known wreck:
      1. Load raw pings within SEARCH_RADIUS_KM
      2. Grid to GRID_SPACING_M, find residual anomaly peak
      3. Also try dipole midpoint
      4. Compute GPS-to-peak offset
      5. Judge viability as rubber-sheet anchor
      6. If viable and not dry_run: upsert into datum_anchors.json

    Returns a result dict with full diagnostics.
    """
    gps_lat = wreck["verified_gps"][0]
    gps_lon = wreck["verified_gps"][1]
    name    = wreck["name"]

    logger.info("─" * 60)
    logger.info("Verifying: %s  GPS=(%.4f, %.4f)", name, gps_lat, gps_lon)

    # ── Step 1: Load raw pings ─────────────────────────────────────────────
    lons, lats, mags = load_raw_pings(csv_path, gps_lat, gps_lon)
    n_pings = len(mags)
    logger.info("  Raw pings loaded: %d", n_pings)

    if n_pings < MIN_PINGS:
        logger.warning("  → SKIP: too few raw pings (%d < %d)", n_pings, MIN_PINGS)
        return {
            "wreck": name,
            "viable": False,
            "reason": f"too_few_pings ({n_pings})",
            "n_pings": n_pings,
        }

    nT_range = float(mags.max() - mags.min())
    logger.info("  nT range: %.1f nT  (min=%.1f max=%.1f)", nT_range, mags.min(), mags.max())

    if nT_range < MIN_NT_RANGE:
        logger.warning("  → SKIP: nT range too low (%.1f < %.1f)", nT_range, MIN_NT_RANGE)
        return {
            "wreck": name,
            "viable": False,
            "reason": f"low_nT ({nT_range:.1f} nT)",
            "n_pings": n_pings,
            "nT_range": round(nT_range, 1),
        }

    # ── Step 2: Grid and find peak ─────────────────────────────────────────
    n_cells = int(2 * SEARCH_RADIUS_KM * 1000 / GRID_SPACING_M)
    dlat, dlon = _metres_to_deg(SEARCH_RADIUS_KM * 1000, gps_lat)
    lon_vec = np.linspace(gps_lon - dlon, gps_lon + dlon, n_cells)
    lat_vec = np.linspace(gps_lat - dlat, gps_lat + dlat, n_cells)
    glon_grid, glat_grid = np.meshgrid(lon_vec, lat_vec)

    peak_lat, peak_lon, peak_nT, grid_mag = find_aeromag_peak(
        lons, lats, mags, gps_lat, gps_lon
    )

    peak_offset_m    = _haversine_m(gps_lat, gps_lon, peak_lat, peak_lon)
    expected_offset  = expected_dipole_offset_m(wreck["depth_ft"], wreck["length_ft"])

    logger.info("  Aeromag peak: (%.5f, %.5f)  nT=%.1f", peak_lat, peak_lon, peak_nT)
    logger.info("  GPS→peak offset: %.0f m   (expected dipole offset: ~%.0f m)",
                peak_offset_m, expected_offset)

    # ── Step 3: Dipole midpoint (better hull-centre estimate) ──────────────
    mid_lat, mid_lon = find_dipole_midpoint(grid_mag, glat_grid, glon_grid)
    if not math.isnan(mid_lat):
        mid_offset_m = _haversine_m(gps_lat, gps_lon, mid_lat, mid_lon)
        logger.info("  Dipole midpoint: (%.5f, %.5f)  offset=%.0f m", mid_lat, mid_lon, mid_offset_m)
    else:
        mid_offset_m = float("nan")
        logger.info("  Dipole midpoint: not clear (single lobe or noise)")

    # ── Step 4: Choose best survey_pos ─────────────────────────────────────
    # Use midpoint if it's closer to GPS, else use peak
    if not math.isnan(mid_offset_m) and mid_offset_m < peak_offset_m:
        survey_lat, survey_lon = mid_lat, mid_lon
        survey_method = "dipole_midpoint"
        survey_offset_m = mid_offset_m
    else:
        survey_lat, survey_lon = peak_lat, peak_lon
        survey_method = "anomaly_peak"
        survey_offset_m = peak_offset_m

    # ── Step 5: Viability decision ─────────────────────────────────────────
    viable = survey_offset_m <= MAX_AEROMAG_OFFSET_M
    reason = ("ok" if viable
              else f"offset_too_large ({survey_offset_m:.0f} m > {MAX_AEROMAG_OFFSET_M:.0f} m)")

    # Compute shift vector (GPS - survey_pos) ← what datum_correction uses
    shift_lat_m = (gps_lat - survey_lat) * 111_320.0
    shift_lon_m = ((gps_lon - survey_lon) * 111_320.0
                   * math.cos(math.radians(gps_lat)))
    shift_total_m = math.sqrt(shift_lat_m ** 2 + shift_lon_m ** 2)

    logger.info("  Survey method: %s   offset_from_GPS=%.0f m   VIABLE=%s",
                survey_method, survey_offset_m, viable)
    logger.info("  Shift vector: N%+.1f m  E%+.1f m  (total=%.0f m)",
                shift_lat_m, shift_lon_m, shift_total_m)

    result = {
        "wreck":           name,
        "id":              wreck["id"],
        "viable":          viable,
        "reason":          reason,
        "n_pings":         n_pings,
        "nT_range":        round(nT_range, 1),
        "peak_nT":         round(peak_nT, 2),
        "aeromag_peak":    [round(peak_lat, 6), round(peak_lon, 6)],
        "dipole_midpoint": ([round(mid_lat, 6), round(mid_lon, 6)]
                            if not math.isnan(mid_lat) else None),
        "survey_pos":      [round(survey_lat, 6), round(survey_lon, 6)],
        "survey_method":   survey_method,
        "offset_m":        round(survey_offset_m, 1),
        "expected_dipole_offset_m": round(expected_offset, 1),
        "shift_N_m":       round(shift_lat_m, 1),
        "shift_E_m":       round(shift_lon_m, 1),
        "shift_total_m":   round(shift_total_m, 1),
        "GPS":             [gps_lat, gps_lon],
        "length_ft":       wreck["length_ft"],
        "depth_ft":        wreck["depth_ft"],
        "source":          wreck.get("source", "AWOIS"),
        "notes":           wreck.get("notes", ""),
    }

    # ── Step 6: Upsert into datum_anchors.json ─────────────────────────────
    if viable and not dry_run:
        anchors = load_anchors()
        existing_ids = {a["id"] for a in anchors}

        anchor_entry = {
            "id":           wreck["id"],
            "name":         wreck["name"],
            "type":         "wreck",
            "verified":     True,
            "verified_gps": [gps_lat, gps_lon],
            "survey_pos":   [round(survey_lat, 6), round(survey_lon, 6)],
            "survey_method": survey_method,
            "shift_m": {
                "north": round(shift_lat_m, 1),
                "east":  round(shift_lon_m, 1),
                "total": round(shift_total_m, 1),
            },
            "source":       wreck.get("source", "AWOIS NOAA"),
            "notes":        wreck.get("notes", ""),
            "length_ft":    wreck["length_ft"],
            "depth_ft":     wreck["depth_ft"],
            "offset_from_gps_m": round(survey_offset_m, 1),
        }

        if wreck["id"] in existing_ids:
            # Update existing — don't duplicate
            for i, a in enumerate(anchors):
                if a["id"] == wreck["id"]:
                    anchors[i] = anchor_entry
                    break
            logger.info("  Updated anchor in datum_anchors.json: %s", wreck["id"])
        else:
            anchors.append(anchor_entry)
            logger.info("  Added new anchor to datum_anchors.json: %s", wreck["id"])

        save_anchors(anchors)

    return result


# ── Coverage analysis ──────────────────────────────────────────────────────

def analyse_coverage_improvement(results: list[dict]) -> dict:
    """
    Show how the triangulation coverage changes after adding wreck anchors.
    Computes the Voronoi-like coverage of the lake for before/after.
    """
    anchors_before = load_anchors()
    shore_anchors = [a for a in anchors_before if a.get("type") != "wreck"]
    wreck_anchors = [r for r in results if r["viable"]]

    # Test points: a 10×20 grid covering Lake Erie
    test_lats = np.linspace(41.4, 42.9, 15)
    test_lons = np.linspace(-83.5, -78.5, 25)
    grid_la, grid_lo = np.meshgrid(test_lats, test_lons)

    # For each test point: how many verified anchors within 100 km?
    def count_within(test_lat, test_lon, anchor_list) -> int:
        return sum(
            1 for a in anchor_list
            if a.get("verified", False) and
               _haversine_m(test_lat, test_lon,
                            a["verified_gps"][0], a["verified_gps"][1]) < 100_000
        )

    # Before (shore only)
    coverage_before = [count_within(la, lo, shore_anchors)
                       for la, lo in zip(grid_la.ravel(), grid_lo.ravel())]
    # After (shore + wrecks)
    all_anchors = shore_anchors + [
        {"id": r["id"], "verified": True, "verified_gps": r["GPS"], "type": "wreck"}
        for r in wreck_anchors
    ]
    coverage_after = [count_within(la, lo, all_anchors)
                      for la, lo in zip(grid_la.ravel(), grid_lo.ravel())]

    n_points     = len(coverage_before)
    n_zero_before = sum(1 for c in coverage_before if c < 3)
    n_zero_after  = sum(1 for c in coverage_after  if c < 3)

    return {
        "test_points":       n_points,
        "triangulation_eligible_before": n_points - n_zero_before,
        "triangulation_eligible_after":  n_points - n_zero_after,
        "poorly_covered_before_pct": round(100 * n_zero_before / n_points, 1),
        "poorly_covered_after_pct":  round(100 * n_zero_after  / n_points, 1),
        "new_wreck_anchors": len(wreck_anchors),
        "wreck_anchor_names": [r["wreck"] for r in wreck_anchors],
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Use known dive-verified wrecks as LORAN-C rubber-sheet anchors"
    )
    parser.add_argument("--raw-csv",  required=True,
                        help="Raw flight-line CSV (nrcan_OH_4039B.csv or gsc_huron_nrcan.csv)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Report only — do NOT modify datum_anchors.json")
    parser.add_argument("--output",   default="wreck_hunting_ml/models/wreck_anchor_report.json")
    args = parser.parse_args(argv)

    csv_path = Path(args.raw_csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("WRECK ANCHOR VERIFIER")
    logger.info("  Source: %s", csv_path.name)
    logger.info("  Mode  : %s", "DRY-RUN (no changes)" if args.dry_run else "LIVE (will update anchors)")
    logger.info("  Wrecks: %d", len(KNOWN_WRECKS))
    logger.info("=" * 65)

    # ── Step 0: Provenance check BEFORE any wreck processing ──────────────
    # This detects the processing level (anomaly vs absolute TMF) and warns of
    # cross-line DC bias that would corrupt peak-position estimates.
    if _PROVENANCE_AVAILABLE:
        # Map the csv path to a known source ID for the check
        csv_name = csv_path.name.lower()
        source_id = None
        if "4039b" in csv_name or "erie" in csv_name:
            source_id = "OH_4039B"
        elif "huron" in csv_name:
            source_id = "gsc_huron"

        if source_id:
            logger.info("Running provenance check for source: %s", source_id)
            prov = check_source(source_id, sample_max=50_000)
            print()
            print("PROVENANCE CHECK")
            print(f"  Detected level  : {prov.detected_level}")
            print(f"  Needs IGRF strip: {'YES (auto-applied in load_raw_pings)' if prov.needs_igrf_removal else 'No'}")
            print(f"  Cross-line DC   : {prov.crossline_dc_bias:.1f} nT roughness"
                  f"  ({'OK' if prov.crossline_ok else 'WARNING — may shift peak positions ±100-300 m'})")
            print(f"  Same-day metadata: NOT AVAILABLE (pre-processed GSC grid)")
            print(f"  Explanation     : Diurnal+IGRF correction applied by GSC at processing time.")
            print(f"                    Flight line/date info discarded in final grid product.")
            if prov.warnings:
                for w in prov.warnings:
                    print(f"  ⚠  {w}")
            print()
            if not prov.crossline_ok:
                logger.warning(
                    "Cross-line DC bias (%.1f nT) detected. Anomaly peaks may be offset "
                    "by 50-300 m from true positions. Consider this uncertainty in anchor assessment.",
                    prov.crossline_dc_bias
                )
    else:
        logger.info("wh2k_data_provenance not available — skipping provenance check")

    results = []
    for wreck in KNOWN_WRECKS:
        r = verify_wreck_as_anchor(wreck, csv_path, dry_run=args.dry_run)
        results.append(r)

    # Coverage analysis
    cov = analyse_coverage_improvement(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("WRECK ANCHOR VERIFICATION REPORT")
    print("=" * 80)
    print(f"  {'Wreck':<28} {'Pings':>6}  {'nT_range':>8}  {'Offset_m':>8}  "
          f"{'Shift_N':>8}  {'Shift_E':>8}  {'Viable':>7}")
    print("  " + "-" * 78)
    for r in results:
        if "error" in r:
            print(f"  {r['wreck']:<28}  ERROR")
            continue
        offset = r.get("offset_m", "—")
        sn     = r.get("shift_N_m", "—")
        se     = r.get("shift_E_m", "—")
        viable = "✓ YES" if r["viable"] else f"✗ NO ({r['reason']})"
        print(f"  {r['wreck']:<28} {r['n_pings']:6d}  {r.get('nT_range',0):8.1f}  "
              f"{offset:8.0f}  {sn:+8.1f}  {se:+8.1f}  {viable}")

    print()
    print("TRIANGULATION COVERAGE IMPACT")
    print(f"  Shore anchors only : {cov['poorly_covered_before_pct']}% of lake poorly covered (<3 anchors)")
    print(f"  + Wreck anchors    : {cov['poorly_covered_after_pct']}% of lake poorly covered (<3 anchors)")
    print(f"  New verified anchors added: {cov['new_wreck_anchors']}")
    for n in cov["wreck_anchor_names"]:
        print(f"    → {n}")

    print()
    viable = [r for r in results if r.get("viable")]
    print(f"VERDICT: {len(viable)}/{len(results)} wrecks promoted as rubber-sheet anchors.")
    if not args.dry_run and viable:
        print(f"  datum_anchors.json updated → {ANCHOR_FILE}")
    print("=" * 80)

    # Save full report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_csv":     str(csv_path),
        "dry_run":        args.dry_run,
        "coverage":       cov,
        "wreck_results":  results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Full report → %s", out_path)


if __name__ == "__main__":
    main()
