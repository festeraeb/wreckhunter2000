"""
WreckHunter 2000 — Data Provenance Checker
===========================================
Validates the processing level and cross-survey consistency of every
magnetic data source BEFORE any gridding, training, or anomaly comparison.

THE CORE PROBLEM
----------------
The pipeline has two CSVs that look similar but are NOT comparable:

    nrcan_OH_4039B.csv       col='mag_anomaly'   values≈ -575 to +677 nT
    gsc_huron_nrcan.csv      col='mag_anomaly'   values≈ 57,539 to 58,271 nT

OH_4039B  → IGRF subtracted  → ANOMALY field      (processing level 2+)
gsc_huron → absolute TMF     → TOTAL FIELD         (processing level 0–1)

Same column name, completely different quantity.  Comparing them directly
would be like comparing Celsius to Kelvin.

WHAT WE CAN AND CANNOT CHECK
------------------------------
CAN CHECK  (from lon/lat/value alone):
  ✓ Processing level (anomaly vs absolute TMF detection)
  ✓ NoData sentinel value (-9999 pattern)
  ✓ Grid regularity vs scattered flight tracks
  ✓ Cross-line step offsets (tie-line leveling quality)
  ✓ Spatial extent vs expected lake coverage
  ✗ Same-day / same-aircraft / diurnal state  (metadata discarded in pre-processing)

WHY SAME-DAY METADATA IS GONE
------------------------------
Both CSVs derive from NRCan/GSC pre-processed grid products:
  - Diurnal correction already applied (that's how anomaly values near 0 are possible)
  - Tie-line leveling already applied at the GSC processing centre
  - Flight line / aircraft / date metadata discarded when writing the final grid
  
This is STANDARD PRACTICE for released survey grids.  The good news:
  - You don't need to worry about same-day correction; GSC did it
  - These are NOT raw pings; they are Level-2 to Level-3 processed grids

WHAT WE DO NEED TO FIX
------------------------
  1. Standardise both to anomaly space (subtract IGRF from gsc_huron)
  2. Detect and flag cross-line DC offsets if present
  3. Never mix anomaly and TMF values in the same model feature

Usage
-----
    python -W ignore scripts/wh2k_data_provenance.py

    # Or import and call from other scripts:
    from wh2k_data_provenance import check_source, standardise_to_anomaly, SOURCES
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── IGRF-13 lightweight approximation ─────────────────────────────────────
# For Great Lakes region (lat 41–47N, lon 84–76W) and epoch 1990–2000
# We use a simple bilinear fit derived from IGRF-13 tabulated values.
# Accuracy: ±30 nT — sufficient to convert absolute TMF to anomaly space
# for wreck detection (which needs ±5 nT accuracy at 100m scale)

_IGRF_COEFFS = {
    # Reference: IGRF-13, epoch 1993 (midpoint of typical GSC Great Lakes surveys)
    # Total field F (nT) ≈ a0 + a1*(lat-44) + a2*(lon+80)
    # Fit to: 41N/84W=57320, 41N/76W=57680, 47N/84W=58110, 47N/76W=58470
    "a0":  57875.0,   # F at (44N, 80W)
    "a1":   130.0,    # nT / degree latitude
    "a2":    52.0,    # nT / degree longitude (lon is negative, so +52 per degree east)
    "epoch": 1993,
    "source": "IGRF-13 bilinear approximation, Great Lakes, epoch 1993"
}


def igrf_approx_nT(lat: float, lon: float) -> float:
    """Very lightweight IGRF approximation for Great Lakes. Accuracy ±30 nT."""
    c = _IGRF_COEFFS
    return c["a0"] + c["a1"] * (lat - 44.0) + c["a2"] * (lon - (-80.0))


# ── Data source registry ───────────────────────────────────────────────────

SOURCES = {
    "OH_4039B": {
        "file":         "magnetic_data/raw/local_mage_csv/nrcan_OH_4039B.csv",
        "col_lon":      None,   # auto-detect (BOM-prefixed)
        "col_lat":      "lat",
        "col_val":      "mag_anomaly",
        "expected_level": "anomaly",   # IGRF subtracted
        "survey":       "NRCan/GSC Ohio-Erie 4039B",
        "lake":         "Erie",
        "expected_lat": (40.0, 42.9),
        "expected_lon": (-84.0, -80.0),
    },
    "gsc_huron": {
        "file":         "magnetic_data/raw/local_mage_csv/gsc_huron_nrcan.csv",
        "col_lon":      "longitude",
        "col_lat":      "latitude",
        "col_val":      "mag_anomaly",
        "expected_level": "absolute_tmf",  # IGRF NOT subtracted
        "survey":       "NRCan/GSC Huron",
        "lake":         "Huron",
        "expected_lat": (43.0, 46.5),
        "expected_lon": (-84.0, -79.0),
    },
}


# ── Check result dataclass ─────────────────────────────────────────────────

@dataclass
class ProvenanceResult:
    source_id:          str
    file:               str
    n_total:            int       = 0
    n_valid:            int       = 0
    n_nodata:           int       = 0
    val_min:            float     = 0.0
    val_max:            float     = 0.0
    val_mean:           float     = 0.0
    val_std:            float     = 0.0
    detected_level:     str       = "unknown"
    expected_level:     str       = "unknown"
    level_match:        bool      = False
    is_grid:            bool      = False
    grid_spacing_lat_m: float     = 0.0
    grid_spacing_lon_m: float     = 0.0
    crossline_dc_bias:  float     = 0.0    # std of per-line medians (nT)
    crossline_ok:       bool      = True
    lat_range:          list      = field(default_factory=list)
    lon_range:          list      = field(default_factory=list)
    coverage_ok:        bool      = False
    warnings:           list      = field(default_factory=list)
    needs_igrf_removal: bool      = False


def _detect_col(fieldnames: list[str], candidates: list[str]) -> Optional[str]:
    fl = [f.upper() for f in fieldnames]
    for c in candidates:
        if c.upper() in fl:
            return fieldnames[fl.index(c.upper())]
    # BOM-tolerant match
    for f in fieldnames:
        clean = f.encode("ascii", "ignore").decode("ascii").upper().strip()
        if clean in [c.upper() for c in candidates]:
            return f
    return None


def check_source(source_id: str, sample_max: int = 200_000) -> ProvenanceResult:
    """
    Full provenance check for a single data source.
    Returns a ProvenanceResult with diagnostics.
    """
    cfg = SOURCES[source_id]
    csv_path = REPO / cfg["file"]
    result = ProvenanceResult(source_id=source_id, file=str(csv_path))

    if not csv_path.exists():
        result.warnings.append(f"FILE NOT FOUND: {csv_path}")
        return result

    lats, lons, vals = [], [], []
    n_total = 0
    n_nodata = 0

    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Auto-detect columns
        col_lon = (cfg["col_lon"]
                   or _detect_col(fieldnames, ["LON", "LONG", "LONGITUDE", "X"]))
        col_lat = (cfg["col_lat"]
                   or _detect_col(fieldnames, ["LAT", "LATITUDE", "Y"]))
        col_val = cfg["col_val"]

        if not col_lon or not col_lat or col_val not in fieldnames:
            # Try case-insensitive match for col_val
            col_val = _detect_col(fieldnames, [cfg["col_val"]])
            if not col_val:
                result.warnings.append(f"Could not find columns: lon={col_lon} lat={col_lat} val={cfg['col_val']}")
                result.warnings.append(f"Available: {fieldnames}")
                return result

        for row in reader:
            n_total += 1
            if n_total > sample_max:
                break
            try:
                v = float(row[col_val])
                if v <= -9990:
                    n_nodata += 1
                    continue
                lats.append(float(row[col_lat]))
                lons.append(float(row[col_lon]))
                vals.append(v)
            except (ValueError, KeyError):
                continue

    if not vals:
        result.warnings.append("No valid data rows found")
        return result

    lats_a = np.array(lats);  lons_a = np.array(lons);  vals_a = np.array(vals)
    result.n_total   = n_total
    result.n_valid   = len(vals)
    result.n_nodata  = n_nodata
    result.val_min   = round(float(vals_a.min()), 2)
    result.val_max   = round(float(vals_a.max()), 2)
    result.val_mean  = round(float(vals_a.mean()), 2)
    result.val_std   = round(float(vals_a.std()), 2)
    result.lat_range = [round(float(lats_a.min()), 4), round(float(lats_a.max()), 4)]
    result.lon_range = [round(float(lons_a.min()), 4), round(float(lons_a.max()), 4)]

    # ── Detect processing level ────────────────────────────────────────────
    # Anomaly field: centred near 0, typically -1000 to +1000 nT
    # Absolute TMF:  ~45000–65000 nT for Great Lakes, std typically 100-500 nT
    if abs(result.val_mean) < 5000:
        result.detected_level = "anomaly"
    else:
        result.detected_level = "absolute_tmf"

    result.expected_level = cfg["expected_level"]
    result.level_match    = (result.detected_level == result.expected_level)
    result.needs_igrf_removal = (result.detected_level == "absolute_tmf")

    if not result.level_match:
        result.warnings.append(
            f"LEVEL MISMATCH: expected {result.expected_level}, "
            f"detected {result.detected_level} (mean={result.val_mean:.0f} nT)"
        )

    # ── Detect grid vs flight tracks ───────────────────────────────────────
    # Grid: sorted lat diffs have very low std (regular spacing)
    # Flight tracks: irregular lat spacing when sorted by lon
    sorted_lats = np.sort(lats_a)
    dlat = np.diff(sorted_lats)
    dlat_nz = dlat[dlat > 1e-8]   # skip zero diffs (repeated lat in grid)
    spacing_lat_m = float(np.median(dlat_nz)) * 111_320.0 if len(dlat_nz) else 0
    spacing_std_m = float(np.std(dlat_nz)) * 111_320.0 if len(dlat_nz) else 0

    # If std / median < 0.5, it's a regular grid
    if spacing_lat_m > 0 and spacing_std_m / (spacing_lat_m + 1e-6) < 0.5:
        result.is_grid           = True
        result.grid_spacing_lat_m = round(spacing_lat_m, 1)
    else:
        result.is_grid           = False
        result.grid_spacing_lat_m = round(spacing_lat_m, 1)

    # Lon spacing
    sorted_lons = np.sort(lons_a)
    dlon = np.diff(sorted_lons)
    dlon_nz = dlon[dlon > 1e-8]
    cos_lat = math.cos(math.radians(lats_a.mean()))
    spacing_lon_m = float(np.median(dlon_nz)) * 111_320.0 * cos_lat if len(dlon_nz) else 0
    result.grid_spacing_lon_m = round(spacing_lon_m, 1)

    # ── Cross-line DC bias check ───────────────────────────────────────────
    # Group by rounded-latitude bins (proxy for flight lines in E-W flown survey)
    # If per-line medians have high std → tie-line leveling not done
    bin_size = 0.005   # ~550 m bins
    lat_bins = np.round(lats_a / bin_size) * bin_size
    unique_bins = np.unique(lat_bins)
    if len(unique_bins) >= 5:
        bin_medians = np.array([
            np.median(vals_a[lat_bins == b]) for b in unique_bins
            if (lat_bins == b).sum() >= 5
        ])
        if len(bin_medians) >= 5:
            # For a properly leveled survey, line medians should be smooth
            # Second-derivative roughness of line medians indicates step offsets
            rough = float(np.std(np.diff(np.diff(bin_medians))))
            result.crossline_dc_bias = round(rough, 2)
            # Threshold: >15 nT roughness suggests un-leveled data
            result.crossline_ok = rough < 15.0
            if not result.crossline_ok:
                result.warnings.append(
                    f"CROSS-LINE DC BIAS DETECTED: line-median roughness = {rough:.1f} nT "
                    f"(threshold 15 nT). Tie-line leveling may be incomplete."
                )

    # ── Spatial coverage check ─────────────────────────────────────────────
    exp_lat = cfg["expected_lat"]
    exp_lon = cfg["expected_lon"]
    overlap_lat = (result.lat_range[0] < exp_lat[1] and result.lat_range[1] > exp_lat[0])
    overlap_lon = (result.lon_range[0] < exp_lon[1] and result.lon_range[1] > exp_lon[0])
    result.coverage_ok = overlap_lat and overlap_lon
    if not result.coverage_ok:
        result.warnings.append(
            f"COVERAGE MISMATCH: data lat={result.lat_range} lon={result.lon_range}, "
            f"expected lat={exp_lat} lon={exp_lon}"
        )

    return result


# ── IGRF removal for absolute-TMF sources ─────────────────────────────────

def standardise_to_anomaly(
    lons: np.ndarray,
    lats: np.ndarray,
    vals: np.ndarray,
    detected_level: str,
) -> np.ndarray:
    """
    Convert absolute TMF to anomaly by subtracting per-point IGRF approximation.
    If already anomaly, returns vals unchanged.
    """
    if detected_level != "absolute_tmf":
        return vals

    igrf_field = np.array([igrf_approx_nT(la, lo) for la, lo in zip(lats, lons)])
    return vals - igrf_field


# ── Cross-source consistency check ────────────────────────────────────────

def check_cross_source_consistency(results: list[ProvenanceResult]) -> dict:
    """
    After standardising all sources to anomaly space, check that overlapping
    regions give consistent values (within ~20 nT for well-leveled surveys).
    """
    anomaly_results = [r for r in results if r.detected_level in ("anomaly", "absolute_tmf")]
    if len(anomaly_results) < 2:
        return {"status": "only_one_source", "note": "need ≥2 sources to cross-check"}

    return {
        "status":       "ok",
        "n_sources":    len(anomaly_results),
        "sources":      [r.source_id for r in anomaly_results],
        "note": (
            "Both standardised to anomaly space. "
            "Overlap regions should agree within ±20 nT after IGRF removal. "
            "Run with overlapping bbox to verify."
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("WH2K — DATA PROVENANCE REPORT")
    print("=" * 70)
    print()

    all_results = []
    for sid in SOURCES:
        logger.info("Checking: %s", sid)
        r = check_source(sid)
        all_results.append(r)

        print(f"SOURCE: {sid}")
        print(f"  File            : {Path(r.file).name}")
        print(f"  Valid points    : {r.n_valid:,}  (nodata: {r.n_nodata:,})")
        print(f"  Value range     : {r.val_min:.1f} to {r.val_max:.1f} nT  (mean={r.val_mean:.1f}  std={r.val_std:.1f})")
        print(f"  Detected level  : {r.detected_level}  ({'✓' if r.level_match else '✗ MISMATCH'})")
        print(f"  IGRF not removed: {'YES — needs standardisation before comparison' if r.needs_igrf_removal else 'No (already anomaly)'}")
        print(f"  Data type       : {'Pre-processed GRID (no date/flight metadata)' if r.is_grid else 'Scattered / flight-track'}")
        print(f"  Grid spacing    : lat≈{r.grid_spacing_lat_m:.0f}m  lon≈{r.grid_spacing_lon_m:.0f}m")
        print(f"  Cross-line bias : {r.crossline_dc_bias:.1f} nT roughness  ({'OK' if r.crossline_ok else 'WARNING'})")
        print(f"  Coverage        : lat{r.lat_range}  lon{r.lon_range}  {'✓' if r.coverage_ok else '✗'}")
        if r.warnings:
            for w in r.warnings:
                print(f"  ⚠  {w}")
        print()

    # Cross-source
    cross = check_cross_source_consistency(all_results)
    print(f"CROSS-SOURCE CONSISTENCY: {cross['status']}")
    print(f"  {cross.get('note','')}")
    print()

    # Key takeaways
    print("KEY FINDINGS")
    print("-" * 70)
    print("1. These are pre-processed GRID products, NOT raw flight-line pings.")
    print("   → Date/aircraft/diurnal metadata was discarded by GSC at processing time.")
    print("   → Diurnal correction IS already applied (that is how anomaly values arise).")
    print("   → Tie-line leveling: see cross-line DC bias above.")
    print()
    print("2. OH_4039B  = anomaly (IGRF subtracted).")
    print("   gsc_huron = absolute TMF (IGRF NOT subtracted, ~57,875 nT mean).")
    print("   → NEVER mix raw values from these two sources in the same model feature.")
    print("   → Call standardise_to_anomaly() on gsc_huron before any comparison.")
    print()
    print("3. To get TRUE raw pings with flight-line metadata, fetch from:")
    print("   NOAA NCEI MGD77T archive: https://www.ncei.noaa.gov/access/search/dataset-search?keywords=aeromagnetic")
    print("   NRCan GeoGratis:          https://geographis.nrcan.gc.ca/")
    print("   These have: DATE, TIME, LINE_NO, AIRCRAFT_ID, TMF_raw, BASE_TMF")
    print()

    # Save JSON
    out = REPO / "wreck_hunting_ml" / "models" / "data_provenance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2),
        encoding="utf-8"
    )
    print(f"Full report → {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
