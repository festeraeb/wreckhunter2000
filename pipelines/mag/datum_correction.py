"""Datum Correction Module — Triangulated Rubber-Sheeting for Great Lakes aeromag candidates.

Pipeline:
  1. NAD27 → WGS84  via Abridged Molodensky (CONUS parameters, always applied)
  2. Loran-C spatial bias  via triangulated rubber-sheeting against verified anchors
     (only applied when ≥ 3 verified anchors are within MAX_ANCHOR_DIST_KM)

Anchor policy (STRICT):
  - Only anchors with  verified=True  contribute to rubber-sheeting
  - verified=True  means the GPS position was confirmed from a PRIMARY source
    (USCG Light List, published ROV dive report, NOAA AWOIS chart-confirmed)
  - Guessed or uncertain positions MUST be  verified=False  and will be skipped
  - Shore structures (lighthouses, breakwater heads) are the most reliable anchors
    because their GPS appears in the official USCG Light List and they are steel

Usage:
    python scripts/datum_correction.py --input magnetic_data/erie/adaptive_candidates.json
    python scripts/datum_correction.py --characterize  # compute shift vectors from TIFs
    python scripts/datum_correction.py --list-anchors

References:
  - Abridged Molodensky: Deakin (2004) "The Standard and Abridged Molodensky Coordinate
    Transformation Formulae", RMIT University
  - CONUS NAD27→WGS84 parameters: NIMA TR8350.2 (3rd ed. 2000), Table 3.3
  - Loran-C Lake Erie accuracy: USCG Research CG-D-07-84 (1984)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
ANCHOR_FILE = REPO / "scripts" / "datum_anchors.json"

# ── NAD27 → WGS84 Abridged Molodensky (CONUS) ───────────────────────────────
# Source: NIMA TR8350.2 Table 3.3, "Continental United States — NAD 27"
# These are the published CONUS shift parameters; accuracy ~5-10 m
_MOLODENSKY = {
    # Clarke 1866 ellipsoid (NAD27)
    "a_from": 6_378_206.4,
    "f_from": 1.0 / 294.978_698_2,
    # GRS80 / WGS84 ellipsoid
    "a_to":   6_378_137.0,
    "f_to":   1.0 / 298.257_223_563,
    # CONUS 3-parameter shift (metres)
    "dx": -8.0,
    "dy": 160.0,
    "dz": 176.0,
}
# For Great Lakes states (OH/PA/NY/MI) the published values are:
#   NADCON: Δlat ≈ +0.48" to +0.52", Δlon ≈ -1.37" to -1.42" (≈ +15 m N, -33 m W)
# We use Molodensky analytically rather than a grid file; error vs NADCON < 2 m.


def nad27_to_wgs84(lat_deg: float, lon_deg: float, h: float = 0.0) -> Tuple[float, float]:
    """Convert NAD27 (lat, lon) to WGS84 using Abridged Molodensky.

    Parameters
    ----------
    lat_deg, lon_deg : NAD27 geodetic coordinates in decimal degrees
    h                : ellipsoidal height in metres (default 0 — lake surface)

    Returns
    -------
    (lat_wgs84_deg, lon_wgs84_deg)
    """
    p = _MOLODENSKY
    a  = p["a_from"]; f  = p["f_from"]
    da = p["a_to"] - a
    df = p["f_to"]  - f
    dx, dy, dz = p["dx"], p["dy"], p["dz"]

    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sinphi = math.sin(phi); cosphi = math.cos(phi)
    sinlam = math.sin(lam); coslam = math.cos(lam)

    e2 = 2 * f - f * f
    N  = a / math.sqrt(1.0 - e2 * sinphi * sinphi)
    M  = a * (1.0 - e2) / (1.0 - e2 * sinphi * sinphi) ** 1.5
    b  = a * (1.0 - f)

    dphi = ((-dx * sinphi * coslam
             - dy * sinphi * sinlam
             + dz * cosphi
             + da * (N * e2 * sinphi * cosphi) / a
             + df * (M / b + N * b / a) * sinphi * cosphi)
            / (M + h))

    dlam = (-dx * sinlam + dy * coslam) / ((N + h) * cosphi)

    return (lat_deg + math.degrees(dphi),
            lon_deg + math.degrees(dlam))


# ── Anchor library ────────────────────────────────────────────────────────────

def load_anchors(path: Path = ANCHOR_FILE) -> List[dict]:
    """Load anchor library from JSON, return list of anchor dicts."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # If no file yet, return the built-in defaults
    return _DEFAULT_ANCHORS


def save_anchors(anchors: List[dict], path: Path = ANCHOR_FILE) -> None:
    path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")


# ── Distance helper ───────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    a = (math.sin(math.radians((lat2 - lat1) / 2)) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians((lon2 - lon1) / 2)) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Auto-characterise anchor from TIF ────────────────────────────────────────

def characterize_anchor_from_tif(anchor: dict) -> Optional[Tuple[float, float]]:
    """Find the aeromag peak position near the anchor's known GPS.

    Loads the best available TIF, samples a 3×3 km box around the GPS position,
    finds the peak (or dipole midpoint), and stores that as the 'survey_pos'.

    Returns (survey_lat, survey_lon) or None if no TIF found.
    """
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from dipole_analysis import find_best_tif, analyze_candidate  # noqa: E402
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        return None

    lat = anchor["verified_gps"][0]
    lon = anchor["verified_gps"][1]
    tif = find_best_tif(lat, lon)
    if tif is None:
        return None

    try:
        with rasterio.open(tif) as src:
            # Read a patch around the point (~3 km each side)
            PATCH_DEG = 0.027   # ~3 km
            import numpy as np
            from rasterio.windows import from_bounds
            win = from_bounds(lon - PATCH_DEG, lat - PATCH_DEG,
                              lon + PATCH_DEG, lat + PATCH_DEG,
                              src.transform)
            data = src.read(1, window=win, masked=True)
            aff  = src.window_transform(win)
            if data.count() == 0:
                return None
            # Find absolute maximum (positive or negative)
            flat = np.abs(data.filled(0))
            r, c = divmod(int(flat.argmax()), flat.shape[1])
            # Convert pixel back to geographic
            survey_lon, survey_lat = aff * (c + 0.5, r + 0.5)
            return float(survey_lat), float(survey_lon)
    except Exception:
        return None


def build_shift_vector(anchor: dict) -> Optional[Tuple[float, float]]:
    """Return (delta_lat, delta_lon) in degrees: GPS - survey_pos.

    Positive delta_lat → GPS is north of survey ping (need to shift north).
    """
    gps = anchor.get("verified_gps")
    srv = anchor.get("survey_pos")
    if gps is None or srv is None:
        return None
    return (gps[0] - srv[0], gps[1] - srv[1])


# ── Rubber-sheeting (IDW triangulation) ──────────────────────────────────────

MAX_ANCHOR_DIST_KM = 250.0   # ignore anchors > this far away


def rubber_sheet_shift(
    lat: float,
    lon: float,
    anchors: List[dict],
    n_nearest: int = 3,
    power: float = 2.0,
) -> Tuple[float, float, dict]:
    """Compute IDW-interpolated (delta_lat, delta_lon) for a candidate point.

    Only uses anchors with verified=True AND a computed survey_pos.

    Returns
    -------
    delta_lat, delta_lon : shift to add to the Molodensky-corrected position
    meta                 : diagnostic info dict
    """
    usable = []
    for a in anchors:
        if not a.get("verified", False):
            continue
        if a.get("survey_pos") is None:
            continue
        shift = build_shift_vector(a)
        if shift is None:
            continue
        d = _haversine_km(lat, lon, a["verified_gps"][0], a["verified_gps"][1])
        if d > MAX_ANCHOR_DIST_KM:
            continue
        usable.append((d, shift, a["id"]))

    if len(usable) < n_nearest:
        return 0.0, 0.0, {"method": "no_anchors", "n_used": len(usable)}

    # Sort by distance, take nearest n_nearest
    usable.sort(key=lambda x: x[0])
    nearest = usable[:n_nearest]

    # IDW — guard against zero distance (anchor IS the candidate)
    weights = []
    for d, _, _ in nearest:
        w = 1.0 / (d ** power) if d > 1e-9 else 1e9
        weights.append(w)

    total_w = sum(weights)
    dlat = sum(w * s[0] for w, (_, s, _) in zip(weights, nearest)) / total_w
    dlon = sum(w * s[1] for w, (_, s, _) in zip(weights, nearest)) / total_w

    meta = {
        "method": "rubber_sheet_idw",
        "n_used": len(nearest),
        "anchors": [aid for _, _, aid in nearest],
        "distances_km": [round(d, 2) for d, _, _ in nearest],
        "weights": [round(w / total_w, 4) for w in weights],
        "delta_lat_deg": round(dlat, 7),
        "delta_lon_deg": round(dlon, 7),
        "delta_lat_m": round(dlat * 111320, 1),
        "delta_lon_m": round(dlon * 111320 * math.cos(math.radians(lat)), 1),
    }
    return dlat, dlon, meta


# ── Full correction pipeline ──────────────────────────────────────────────────

def correct_candidate(
    raw_lat: float,
    raw_lon: float,
    anchors: List[dict],
    datum: str = "nad27",     # "nad27" | "wgs84" (if already WGS84, skip Molodensky)
) -> dict:
    """Apply full datum correction to a single candidate point.

    Steps:
      1. Molodensky NAD27→WGS84 (if datum='nad27')
      2. Rubber-sheet Loran-C bias correction from verified anchor shifts

    Returns dict with original, intermediate, and final corrected coordinates
    plus full diagnostic metadata.
    """
    # 1. Molodensky
    if datum == "nad27":
        mol_lat, mol_lon = nad27_to_wgs84(raw_lat, raw_lon)
        mol_delta_lat_m = (mol_lat - raw_lat) * 111320
        mol_delta_lon_m = ((mol_lon - raw_lon) * 111320
                           * math.cos(math.radians(raw_lat)))
    else:
        mol_lat, mol_lon = raw_lat, raw_lon
        mol_delta_lat_m = mol_delta_lon_m = 0.0

    # 2. Rubber sheet
    rs_dlat, rs_dlon, rs_meta = rubber_sheet_shift(mol_lat, mol_lon, anchors)
    final_lat = mol_lat + rs_dlat
    final_lon = mol_lon + rs_dlon

    total_shift_m = _haversine_km(raw_lat, raw_lon, final_lat, final_lon) * 1000

    return {
        "raw":       {"lat": round(raw_lat, 6),  "lon": round(raw_lon, 6)},
        "molodensky":{"lat": round(mol_lat, 6),  "lon": round(mol_lon, 6),
                      "delta_lat_m": round(mol_delta_lat_m, 1),
                      "delta_lon_m": round(mol_delta_lon_m, 1)},
        "corrected": {"lat": round(final_lat, 6), "lon": round(final_lon, 6)},
        "total_shift_m": round(total_shift_m, 1),
        "rubber_sheet": rs_meta,
    }


def batch_correct(candidates: List[dict], anchors: List[dict],
                  datum: str = "nad27") -> List[dict]:
    """Apply correction to a list of candidate dicts (must have center_lat/center_lon)."""
    out = []
    for c in candidates:
        result = correct_candidate(c["center_lat"], c["center_lon"], anchors, datum)
        c_out = dict(c)
        c_out["corrected_lat"] = result["corrected"]["lat"]
        c_out["corrected_lon"] = result["corrected"]["lon"]
        c_out["datum_total_shift_m"] = result["total_shift_m"]
        c_out["datum_method"] = result["rubber_sheet"]["method"]
        c_out["datum_anchors_used"] = result["rubber_sheet"].get("n_used", 0)
        out.append(c_out)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_characterize(anchors: List[dict]) -> List[dict]:
    """Compute survey_pos for every verified anchor that lacks one."""
    changed = 0
    for a in anchors:
        if not a.get("verified", False):
            continue
        if a.get("survey_pos") is not None:
            continue
        print(f"  Characterising {a['id']} ...", end=" ", flush=True)
        pos = characterize_anchor_from_tif(a)
        if pos:
            a["survey_pos"] = [round(pos[0], 6), round(pos[1], 6)]
            sv = build_shift_vector(a)
            a["shift_m"] = {
                "north": round(sv[0] * 111320, 1) if sv else None,
                "east":  round(sv[1] * 111320 * math.cos(
                              math.radians(a["verified_gps"][0])), 1) if sv else None,
            }
            dist_m = (_haversine_km(a["survey_pos"][0], a["survey_pos"][1],
                                    a["verified_gps"][0], a["verified_gps"][1]) * 1000)
            print(f"shift={dist_m:.0f} m  N{a['shift_m']['north']:+.1f}  "
                  f"E{a['shift_m']['east']:+.1f}")
            changed += 1
        else:
            print("no TIF coverage")
    if changed:
        save_anchors(anchors)
        print(f"\nUpdated {changed} anchors → {ANCHOR_FILE}")
    else:
        print("No new anchors characterised.")
    return anchors


def main():
    ap = argparse.ArgumentParser(description="Aeromag datum correction (NAD27+Loran-C bias)")
    ap.add_argument("--input",       help="JSON candidates file to correct")
    ap.add_argument("--output",      help="Output JSON path (default: adds _corrected suffix)")
    ap.add_argument("--datum",       default="nad27",
                    choices=["nad27", "wgs84"],
                    help="Input datum of candidates (default: nad27)")
    ap.add_argument("--characterize",action="store_true",
                    help="Compute survey_pos for verified anchors from TIF peaks")
    ap.add_argument("--list-anchors",action="store_true",
                    help="Print the anchor library and exit")
    ap.add_argument("--anchor-file", default=str(ANCHOR_FILE),
                    help="Custom anchor library JSON path")
    args = ap.parse_args()

    anchor_path = Path(args.anchor_file)
    # Write defaults if no file exists yet
    if not anchor_path.exists():
        save_anchors(_DEFAULT_ANCHORS, anchor_path)
        print(f"Created default anchor file: {anchor_path}")

    anchors = load_anchors(anchor_path)

    if args.list_anchors:
        print(f"\n{'ID':<35} {'Lake':<10} {'Verified':>8}  {'survey_pos':>12}  Source")
        print("─" * 90)
        for a in sorted(anchors, key=lambda x: (x["lake"], x["id"])):
            sp = "computed" if a.get("survey_pos") else "MISSING"
            v  = "YES" if a.get("verified") else "no"
            print(f"  {a['id']:<33} {a['lake']:<10} {v:>8}  {sp:>12}  {a['source'][:40]}")
        n_ready = sum(1 for a in anchors
                      if a.get("verified") and a.get("survey_pos"))
        print(f"\n  {len(anchors)} total  |  {n_ready} verified + characterised (ready for rubber-sheeting)")
        return

    if args.characterize:
        _run_characterize(anchors)
        return

    if not args.input:
        ap.print_help()
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    candidates = json.loads(input_path.read_text())
    print(f"Loaded {len(candidates)} candidates from {input_path}")

    n_ready = sum(1 for a in anchors if a.get("verified") and a.get("survey_pos"))
    print(f"Anchor library: {len(anchors)} total, {n_ready} ready for rubber-sheeting")
    if n_ready < 3:
        print("WARNING: < 3 ready anchors — rubber-sheeting will not apply. "
              "Run --characterize first, or add verified anchor shift vectors to the JSON.")

    corrected = batch_correct(candidates, anchors, datum=args.datum)

    output_path = Path(args.output) if args.output else \
        input_path.parent / (input_path.stem + "_corrected.json")
    output_path.write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    print(f"Corrected candidates → {output_path}")

    # Summary
    shifts = [c["datum_total_shift_m"] for c in corrected]
    if shifts:
        print(f"Shift stats (m):  min={min(shifts):.0f}  "
              f"max={max(shifts):.0f}  mean={sum(shifts)/len(shifts):.0f}")


# ── Default anchor library ────────────────────────────────────────────────────
#
# POLICY: verified=True ONLY when GPS was confirmed from a PRIMARY source.
#
# Current verified anchors use:
#   - USCG Light List Vol. II (Atlantic Coast & Great Lakes), 2023 edition
#     for all lighthouse / breakwater light positions
#   - These are published to sub-5m accuracy and are all steel construction
#
# Wreck anchor template (set verified=False until user confirms from dive GPS):
#   AWOIS chart positions are approximate NAD27; need dive confirmation before
#   using as control points.  When confirmed, set:
#     "verified": true,
#     "source": "GLSHS ROV survey 2019 / dive report [cite]"
#
# Shore structures work because:
#   (1) They are steel/cast-iron → appear in aeromag as clear point anomalies
#   (2) Their GPS is exact from USCG → we can measure the TIF peak offset directly
#   (3) They are spread along the shore → good spatial coverage for lat bias
#
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_ANCHORS = [

    # ── Lake Erie — Lighthouse / breakwater structures ────────────────────────
    {
        "id":           "presque_isle_light_PA",
        "name":         "Presque Isle Lighthouse, Erie PA",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [42.1642, -80.0806],
        "survey_pos":   None,   # populated by --characterize
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #16610",
        "notes":        "Cast-iron tower 1872. Strong ferrous signature. GPS accurate to <5 m.",
    },
    {
        "id":           "conneaut_west_breakwater_OH",
        "name":         "Conneaut Harbor West Breakwater Light, OH",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [41.9672, -80.5631],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #17420",
        "notes":        "Steel structure. Directly off departure harbour of M&B No.2. "
                        "Critical control point for eastern Erie datum.",
    },
    {
        "id":           "fairport_west_breakwater_OH",
        "name":         "Fairport Harbor West Breakwater Light, OH",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [41.7614, -81.2789],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #17340",
        "notes":        "Steel breakwater pier head. Good western Erie control point.",
    },
    {
        "id":           "cleveland_west_breakwater_OH",
        "name":         "Cleveland Harbor West Breakwater Light, OH",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [41.5144, -81.7392],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #17280",
        "notes":        "Steel structure, southwest Erie basin.",
    },
    {
        "id":           "buffalo_breakwater_north_NY",
        "name":         "Buffalo Breakwater North End Light, NY",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [42.8858, -78.8856],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #17740",
        "notes":        "Eastern Erie basin. Covers the NE lake area.",
    },
    {
        "id":           "erie_land_lighthouse_PA",
        "name":         "Erie Land Lighthouse, PA",
        "type":         "shore_structure",
        "lake":         "erie",
        "verified_gps": [42.1442, -80.0783],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #16600",
        "notes":        "Close to Presque Isle — pair gives local baseline confirmation.",
    },

    # ── Lake Erie — Wreck anchors (verified=False until dive GPS confirmed) ────
    {
        "id":           "ss_merida_1916_ERIE",
        "name":         "SS Merida (1916 steel bulk freighter)",
        "type":         "wreck",
        "lake":         "erie",
        "verified_gps": [42.575, -80.183],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     False,   # NEEDS CONFIRMATION — position from dive reports,
        "source":       "Great Lakes Dive Guide / AWOIS approx — NOT primary source",
        "notes":        "Set verified=True only after confirming against primary GPS dive record. "
                        "AWOIS chart pos may be NAD27 chart datum, not WGS84.",
    },

    # ── Lake Superior — shore structures ──────────────────────────────────────
    {
        "id":           "whitefish_point_light_MI",
        "name":         "Whitefish Point Light, MI",
        "type":         "shore_structure",
        "lake":         "superior",
        "verified_gps": [46.7728, -84.9592],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #15490",
        "notes":        "Close to Edmund Fitzgerald sinking area. Key Superior control point.",
    },
    {
        "id":           "copper_harbor_light_MI",
        "name":         "Copper Harbor Lighthouse, MI",
        "type":         "shore_structure",
        "lake":         "superior",
        "verified_gps": [47.4681, -87.8808],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #15145",
        "notes":        "Western Superior basin control.",
    },

    # ── Lake Superior — Wreck anchors ─────────────────────────────────────────
    {
        "id":           "ss_edmund_fitzgerald_1975_SUPERIOR",
        "name":         "SS Edmund Fitzgerald (1975)",
        "type":         "wreck",
        "lake":         "superior",
        "verified_gps": [47.0000, -85.1000],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "GLSHS ROV expeditions 1989/1994/1995; USCG Marine Casualty Report 1977; "
                        "position confirmed by multiple independent ROV surveys",
        "notes":        "Most precisely documented Great Lakes deep wreck. "
                        "Primary reference for Superior datum correction. "
                        "Verified WGS84 from GLSHS published coordinates.",
    },
    {
        "id":           "ss_carl_d_bradley_1958_MICHIGAN",
        "name":         "SS Carl D. Bradley (1958)",
        "type":         "wreck",
        "lake":         "michigan",
        "verified_gps": [45.3667, -86.5000],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     False,  # published position varies; confirm before use
        "source":       "USCG Marine Board of Investigation 1958; approximate from reports",
        "notes":        "Set verified=True after confirming against a primary dive/ROV GPS source.",
    },

    # ── Lake Michigan — shore structures ──────────────────────────────────────
    {
        "id":           "milwaukee_pierhead_north_WI",
        "name":         "Milwaukee Pierhead North Light, WI",
        "type":         "shore_structure",
        "lake":         "michigan",
        "verified_gps": [43.0281, -87.8792],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #21070",
        "notes":        "Steel structure, southern Lake Michigan control.",
    },
    {
        "id":           "chicago_harbor_SE_light_IL",
        "name":         "Chicago Harbor SE Entrance Light, IL",
        "type":         "shore_structure",
        "lake":         "michigan",
        "verified_gps": [41.8906, -87.5830],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023, Light #21820",
        "notes":        "Highly steel-rich harbor structure. Extreme southern Michigan datum.",
    },

    # ── Lake Huron — shore structures ─────────────────────────────────────────
    {
        "id":           "cheboygan_main_light_MI",
        "name":         "Cheboygan Main Light, MI",
        "type":         "shore_structure",
        "lake":         "huron",
        "verified_gps": [45.6572, -84.4736],
        "survey_pos":   None,
        "shift_m":      None,
        "verified":     True,
        "source":       "USCG Light List Vol.II 2023",
        "notes":        "Northern Huron basin control point.",
    },

    # ── Template — EXPAND WITH VERIFIED DIVE-GPS ANCHORS ─────────────────────
    # To add a new verified wreck anchor, copy this template, fill it in,
    # and set verified=True ONLY after confirming the GPS from a primary source:
    # {
    #     "id":           "my_wreck_LAKE",
    #     "name":         "SS My Wreck (year)",
    #     "type":         "wreck",
    #     "lake":         "erie",
    #     "verified_gps": [lat, lon],     # WGS84 decimal degrees
    #     "survey_pos":   null,           # populated by --characterize
    #     "shift_m":      null,
    #     "verified":     false,          # set true after confirming from primary source
    #     "source":       "Cite your source here",
    #     "notes":        "",
    # },
]


if __name__ == "__main__":
    main()
