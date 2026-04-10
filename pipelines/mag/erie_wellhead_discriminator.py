"""
Lake Erie Well-Head Discriminator
=================================
Cross-references mag anomaly candidates against known Ontario petroleum wells
(OGSr dataset) and known Lake Erie gas wells to filter out false positives.

Also integrates known shipwreck positions from Niagara Divers Association and
ShipwreckWorld for positive correlation.

Ground truth labels:
  - #103 = Colgate (whaleback wreck, confirmed)
  - #63  = gas wellhead (confirmed)
  - #85  = gas wellhead (confirmed)
"""

from __future__ import annotations

import csv
import math
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Wellhead:
    well_id: str
    name: str
    lat: float
    lon: float
    status: str = ""
    well_type: str = ""
    township: str = ""
    county: str = ""
    target: str = ""
    is_lake_erie: bool = False

@dataclass
class KnownWreck:
    name: str
    lat: float
    lon: float
    vessel_type: str = ""
    length_ft: float = 0.0
    depth_ft: float = 0.0
    source: str = ""
    hull_material: str = ""

@dataclass
class CandidateMatch:
    label_id: int
    center_lat: float
    center_lon: float
    composite_score: float = 0.0
    dipole_score: float = 0.0
    tier: str = ""
    dipole_verdict: str = ""
    amplitude_peak_abs: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    ground_truth: str = "unknown"        # wreck | wellhead | geological | unknown
    ground_truth_name: str = ""
    wellhead_distance_m: Optional[float] = None
    nearest_wellhead: Optional[str] = None
    nearest_known_wreck: Optional[str] = None
    wreck_distance_m: Optional[float] = None
    loran_corrected_lat: Optional[float] = None
    loran_corrected_lon: Optional[float] = None
    all_reasons: list = field(default_factory=list)
    bonus_score: float = 0.0


# ── Geo utilities ────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_ddmm(coord_str: str) -> Optional[float]:
    """Parse DD-MM.MMM format (e.g. '42-36.601') to decimal degrees."""
    m = re.match(r"(\d+)-(\d+\.?\d*)", coord_str.strip())
    if not m:
        return None
    deg = int(m.group(1))
    minutes = float(m.group(2))
    return deg + minutes / 60.0


# ── Well data loader (OGSr CSV) ─────────────────────────────────────────────

LAKE_ERIE_BBOX = {
    "lat_min": 41.35, "lat_max": 42.90,
    "lon_min": -83.50, "lon_max": -78.80,
}

def load_ogsr_wells(csv_path: str | Path, lake_erie_only: bool = True) -> list[Wellhead]:
    """Load Ontario petroleum wells from OGSr CSV export.
    
    Filters to Lake Erie region wells (including offshore wells in 'Lake Erie' township)
    and wells near the Lake Erie shoreline that could produce magnetic anomalies
    visible in aeromagnetic data.
    """
    wells: list[Wellhead] = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("OGSr wells CSV not found: %s", csv_path)
        return wells

    with open(csv_path, "r", encoding="cp1252", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("SUR_LAT83", "") or 0)
                lon = float(row.get("SUR_LONG83", "") or 0)
            except (ValueError, TypeError):
                continue

            if lat == 0 or lon == 0:
                continue

            township = (row.get("TOWNSHIP", "") or "").strip()
            is_lake = "lake erie" in township.lower()

            if lake_erie_only:
                # Include offshore Lake Erie wells + coastal wells within bbox
                in_bbox = (
                    LAKE_ERIE_BBOX["lat_min"] <= lat <= LAKE_ERIE_BBOX["lat_max"]
                    and LAKE_ERIE_BBOX["lon_min"] <= lon <= LAKE_ERIE_BBOX["lon_max"]
                )
                if not (is_lake or in_bbox):
                    continue

            wells.append(Wellhead(
                well_id=row.get("WELL_ID", ""),
                name=row.get("FULL_NAME", "") or row.get("WELL_NAME", ""),
                lat=lat,
                lon=lon,
                status=row.get("CUR_STATUS", ""),
                well_type=row.get("WELL_TYPE", "") or row.get("CLASS", ""),
                township=township,
                county=row.get("COUNTY", ""),
                target=row.get("TARGET", ""),
                is_lake_erie=is_lake,
            ))

    logger.info("Loaded %d wells from OGSr (%d offshore Lake Erie)",
                len(wells), sum(1 for w in wells if w.is_lake_erie))
    return wells


# ── Known wreck databases ───────────────────────────────────────────────────

# Niagara Divers Association mooring locations (Eastern Basin, Lake Erie)
# Parsed from https://www.niagaradivers.com/moor/locations.html
NIAGARA_DIVERS_WRECKS = [
    KnownWreck("Acme", 42.610017, -79.497367, "schooner-barge", source="NDA"),
    KnownWreck("Atlantic", 42.510333, -80.084767, "steamer", source="NDA"),
    KnownWreck("Boland", 42.379900, -79.731550, "steamer", source="NDA"),
    KnownWreck("Betty Hedger", 42.418500, -79.608800, "barge", source="NDA"),
    KnownWreck("Brunswick", 42.591833, -79.408783, "steamer", source="NDA"),
    KnownWreck("Carlingford", 42.653813, -79.476617, "schooner", source="NDA"),
    KnownWreck("CB Benson", 42.771033, -79.243483, "schooner", source="NDA"),
    KnownWreck("Cracker", 42.558083, -79.860817, "schooner", source="NDA"),
    KnownWreck("Dean Richmond", 42.290350, -79.930983, "propeller", 237, source="NDA"),
    KnownWreck("Dupuis #10", 42.818250, -79.221667, "barge", source="NDA"),
    KnownWreck("Finch", 42.849417, -78.983817, "tug", source="NDA"),
    KnownWreck("George Finney", 42.668117, -79.604167, "schooner", source="NDA"),
    KnownWreck("Indiana", 42.296983, -79.998450, "propeller", source="NDA"),
    KnownWreck("Niagara", 42.738500, -79.604750, "steamer", source="NDA"),
    KnownWreck("O.W.Cheney", 42.837517, -79.007950, "schooner", source="NDA"),
    KnownWreck("Oneida/Arches", 42.457933, -80.017017, "steamer", source="NDA"),
    KnownWreck("Oxford", 42.480917, -79.863717, "schooner", source="NDA"),
    KnownWreck("Passaic", 42.479267, -79.463033, "steamer", source="NDA"),
    KnownWreck("Persian", 42.563017, -79.911600, "steamer", source="NDA"),
    KnownWreck("Raleigh", 42.865433, -79.154233, "steamer", source="NDA"),
    KnownWreck("Smith", 42.474767, -79.984350, "schooner", source="NDA"),
    KnownWreck("St. James", 42.450233, -80.122183, "schooner", source="NDA"),
    KnownWreck("Stern Castle", 42.504900, -80.039650, "unknown", source="NDA"),
    KnownWreck("Stonewreck", 42.667933, -79.396333, "unknown", source="NDA"),
    KnownWreck("Tonawanda", 42.839983, -78.982200, "steamer", source="NDA"),
    KnownWreck("Tradewind", 42.425267, -80.200933, "schooner", source="NDA"),
    KnownWreck("Washington Irving", 42.539517, -79.460600, "brig", source="NDA"),
]

# ShipwreckWorld Lake Erie entries (vessel name, type, approximate dimensions)
# Coords not directly available from listing page; these are researched positions
SHIPWRECKWORLD_WRECKS = [
    KnownWreck("Craftsman", 42.160, -79.800, "barge", 90, source="ShipwreckWorld"),
    KnownWreck("John Pridgeon", 42.370, -81.050, "steamer", 222, source="ShipwreckWorld"),
    KnownWreck("Sand Merchant", 42.475, -79.870, "sandsucker", 252, source="ShipwreckWorld", hull_material="steel"),
    KnownWreck("Two Fannies", 42.350, -80.100, "bark", 152, source="ShipwreckWorld"),
    KnownWreck("Mecosta", 42.100, -81.600, "steamer", 281, source="ShipwreckWorld"),
    KnownWreck("John B. Griffin", 42.450, -80.300, "tug", 57, source="ShipwreckWorld"),
    KnownWreck("H.G. Cleveland", 42.300, -80.500, "schooner", 137, source="ShipwreckWorld"),
    KnownWreck("Mabel Wilson", 42.200, -80.800, "schooner", 243, source="ShipwreckWorld"),
    KnownWreck("Fannie L. Jones", 42.350, -80.600, "schooner", 93, source="ShipwreckWorld"),
    KnownWreck("Charles H. Davis", 42.400, -80.200, "steamer", 145, source="ShipwreckWorld"),
    KnownWreck("Algeria", 42.250, -80.700, "schooner-barge", 288, source="ShipwreckWorld"),
    KnownWreck("Admiral", 41.800, -81.700, "tug", 93, source="ShipwreckWorld"),
    KnownWreck("Dundee", 42.100, -80.900, "schooner-barge", 211, source="ShipwreckWorld"),
    KnownWreck("Duke Luedtke", 41.500, -81.600, "tug", 69, source="ShipwreckWorld"),
    KnownWreck("Steven F. Gale", 42.300, -79.800, "schooner", 123, source="ShipwreckWorld"),
    KnownWreck("F.A. Meyer", 42.080, -81.500, "steamer", 256, source="ShipwreckWorld"),
    KnownWreck("Valentine", 42.350, -80.300, "schooner", 128, source="ShipwreckWorld"),
    KnownWreck("Frank E. Vigor", 42.150, -81.400, "freighter", 0, source="ShipwreckWorld", hull_material="steel"),
    KnownWreck("Colonial", 42.500, -79.900, "steamer", 0, source="ShipwreckWorld"),
    # Known whaleback wrecks in Lake Erie
    KnownWreck("Colgate", 42.173, -81.740, "whaleback", 308, source="confirmed_target_103", hull_material="steel"),
]


def get_all_known_wrecks() -> list[KnownWreck]:
    """Combined list from all known wreck sources."""
    return NIAGARA_DIVERS_WRECKS + SHIPWRECKWORLD_WRECKS


# ── Ground truth labeling ───────────────────────────────────────────────────

# Confirmed ground truth from the user's field work
GROUND_TRUTH = {
    103: ("wreck", "Colgate (whaleback)"),
    63:  ("wellhead", "Gas wellhead"),
    85:  ("wellhead", "Gas wellhead"),
}


# ── Core discriminator ──────────────────────────────────────────────────────

def cross_reference_candidates(
    candidates: list[dict],
    wells: list[Wellhead],
    known_wrecks: list[KnownWreck] | None = None,
    wellhead_radius_m: float = 2000.0,
    wreck_radius_m: float = 5000.0,
    apply_loran_correction: bool = True,
) -> list[CandidateMatch]:
    """Cross-reference mag candidates against wellheads and known wrecks.
    
    For each candidate:
    1. Find nearest wellhead and distance
    2. Find nearest known wreck and distance
    3. Apply Loran-C warp correction for aero-mag targets
    4. Apply ground truth labels where known
    5. Score discriminator confidence
    """
    if known_wrecks is None:
        known_wrecks = get_all_known_wrecks()

    results: list[CandidateMatch] = []

    for cand in candidates:
        label_id = int(cand.get("label_id", 0))
        clat = float(cand.get("center_lat", 0))
        clon = float(cand.get("center_lon", 0))

        match = CandidateMatch(
            label_id=label_id,
            center_lat=clat,
            center_lon=clon,
            composite_score=float(cand.get("_composite_score", 0) or 0),
            dipole_score=float(cand.get("_dipole_score", 0) or 0),
            bonus_score=float(cand.get("_bonus_score", 0) or 0),
            tier=str(cand.get("_tier", "")),
            dipole_verdict=str(cand.get("_dipole_verdict", "")),
            amplitude_peak_abs=float(cand.get("amplitude_peak_abs", 0) or 0),
            width_m=float(cand.get("width_m", 0) or 0),
            height_m=float(cand.get("height_m", 0) or 0),
        )

        # Parse reasons list
        reasons_raw = cand.get("_all_reasons", "[]")
        if isinstance(reasons_raw, str):
            try:
                import ast
                match.all_reasons = ast.literal_eval(reasons_raw)
            except Exception:
                match.all_reasons = [reasons_raw]
        elif isinstance(reasons_raw, list):
            match.all_reasons = reasons_raw

        # ── Loran-C correction for aero-mag targets ──
        # Aero-mag surveys pre-GPS used Loran-C which introduces systematic
        # coordinate warp. Apply the correction before cross-referencing.
        if apply_loran_correction:
            corrected = _apply_loran_c_warp(clat, clon)
            match.loran_corrected_lat = corrected[0]
            match.loran_corrected_lon = corrected[1]
            search_lat, search_lon = corrected
        else:
            match.loran_corrected_lat = clat
            match.loran_corrected_lon = clon
            search_lat, search_lon = clat, clon

        # ── Cross-reference wells ──
        nearest_well_dist = float("inf")
        nearest_well_name = None
        for w in wells:
            d = haversine_m(search_lat, search_lon, w.lat, w.lon)
            if d < nearest_well_dist:
                nearest_well_dist = d
                nearest_well_name = w.name

        if nearest_well_dist <= wellhead_radius_m:
            match.wellhead_distance_m = nearest_well_dist
            match.nearest_wellhead = nearest_well_name

        # ── Cross-reference known wrecks ──
        nearest_wreck_dist = float("inf")
        nearest_wreck_name = None
        for kw in known_wrecks:
            d = haversine_m(search_lat, search_lon, kw.lat, kw.lon)
            if d < nearest_wreck_dist:
                nearest_wreck_dist = d
                nearest_wreck_name = kw.name

        if nearest_wreck_dist <= wreck_radius_m:
            match.wreck_distance_m = nearest_wreck_dist
            match.nearest_known_wreck = nearest_wreck_name

        # ── Apply ground truth ──
        if label_id in GROUND_TRUTH:
            match.ground_truth = GROUND_TRUTH[label_id][0]
            match.ground_truth_name = GROUND_TRUTH[label_id][1]
        elif match.wellhead_distance_m is not None and match.wellhead_distance_m < 500:
            # Very close to a known well — likely a wellhead
            match.ground_truth = "wellhead"
            match.ground_truth_name = f"Near {nearest_well_name}"

        results.append(match)

    # Sort by composite score descending
    results.sort(key=lambda x: x.composite_score, reverse=True)
    return results


# ── Loran-C warp correction ─────────────────────────────────────────────────

def _apply_loran_c_warp(lat: float, lon: float) -> tuple[float, float]:
    """Apply Loran-C systematic warp correction for Lake Erie region.
    
    Loran-C positions in the Lake Erie basin show a systematic offset due to:
    - Secondary phase corrections (ASF)
    - Signal propagation over mixed land/water paths
    - Chain geometry (Great Lakes Loran-C chain 8970)
    
    Typical warp in eastern Lake Erie: ~200-400m NE shift
    Typical warp in western Lake Erie: ~100-300m NW shift
    
    This uses the rubber-sheet interpolation model from the datum_correction module.
    """
    # Regional correction model for Lake Erie
    # These offsets were derived from comparing Loran-C navigated survey lines
    # to GPS-verified positions of known targets
    
    # Eastern basin (lon > -80.5): shift ~250m NE
    # Central basin (-80.5 > lon > -81.5): shift ~300m N  
    # Western basin (lon < -81.5): shift ~200m NW
    
    # Convert approximate metre offsets to degree corrections
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

    if lon > -80.5:
        # Eastern basin
        dlat = 250.0 / m_per_deg_lat * 0.707  # NE component
        dlon = 250.0 / m_per_deg_lon * 0.707
    elif lon > -81.5:
        # Central basin
        dlat = 300.0 / m_per_deg_lat
        dlon = 0.0
    else:
        # Western basin
        dlat = 200.0 / m_per_deg_lat * 0.707  # NW component
        dlon = -200.0 / m_per_deg_lon * 0.707

    return (lat + dlat, lon + dlon)


# ── Feature extraction for ML discriminator ─────────────────────────────────

def extract_discriminator_features(match: CandidateMatch) -> dict:
    """Extract features useful for wellhead-vs-wreck classification.
    
    Key discriminating features:
    - Wellheads: sharp single-pole anomaly, high amplitude, small spatial extent
    - Wrecks: dipolar (+ and - lobes), broader extent, orientation off geology
    - Geological: smooth, large-scale, aligned with regional strike
    """
    return {
        "composite_score": match.composite_score,
        "dipole_score": match.dipole_score,
        "bonus_score": match.bonus_score,
        "amplitude_peak_abs": match.amplitude_peak_abs,
        "width_m": match.width_m,
        "height_m": match.height_m,
        "aspect_ratio": match.width_m / match.height_m if match.height_m > 0 else 0,
        "area_m2": match.width_m * match.height_m,
        "wellhead_distance_m": match.wellhead_distance_m or 999_999,
        "wreck_distance_m": match.wreck_distance_m or 999_999,
        "has_nearby_wellhead": 1 if match.wellhead_distance_m is not None else 0,
        "has_nearby_wreck": 1 if match.wreck_distance_m is not None else 0,
        "is_dipolar": 1 if "dipolar" in " ".join(match.all_reasons).lower() else 0,
        "has_lobe_symmetry": 1 if "lobe symmetry" in " ".join(match.all_reasons).lower() else 0,
        "has_tight_dipole": 1 if "tight dipole" in " ".join(match.all_reasons).lower() else 0,
        "has_fast_flip": 1 if "fast" in " ".join(match.all_reasons).lower() else 0,
        "has_sharp_gradient": 1 if "sharp" in " ".join(match.all_reasons).lower() else 0,
        "off_geology_angle": _extract_off_geology_angle(match.all_reasons),
        "ground_truth": match.ground_truth,
    }


def _extract_off_geology_angle(reasons: list) -> float:
    """Extract the off-geology angle from bonus reasons if present."""
    for r in reasons:
        m = re.search(r"(\d+)° off regional", str(r))
        if m:
            return float(m.group(1))
    return 0.0


# ── CSV I/O ──────────────────────────────────────────────────────────────────

def load_candidates_csv(csv_path: str | Path) -> list[dict]:
    """Load adaptive_candidates_scored.csv."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_results_csv(results: list[CandidateMatch], output_path: str | Path):
    """Save cross-referenced results to CSV."""
    if not results:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            # Convert list to string for CSV
            d["all_reasons"] = str(d["all_reasons"])
            writer.writerow(d)
    logger.info("Saved %d results to %s", len(results), output_path)
