"""Geo-filter + deep scoring pipeline for adaptive scan candidates.

For every candidate:
  1. Geo-classify (LAKE / SHORE / LAND) using shoreline polygon tables
  2. Deduplicate hits within 5 km (multi-source corroboration is a scoring bonus)
  3. Run full dipole analysis (symmetry, flip distance, gradient sharpness, shape)
  4. Apply extra discriminators:
       - Multi-source corroboration bonus
       - SNR vs local background
       - Depth proxy from anomaly half-width
       - Regional geology context (quiet sedimentary basin = anomaly more suspicious)
       - Dipole axis alignment vs regional geological strike
  5. Assign composite tier:  TIER1 / TIER2 / TIER3 / GEOLOGICAL / DISCARD
  6. Write color-coded KML (green=TIER1, teal=TIER2, cyan=TIER3, gray=GEOLOGICAL, red=LAND)

Usage:
  python scripts/geo_filter_candidates.py --lakes huron,erie
  python scripts/geo_filter_candidates.py --lakes erie --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.dipole_analysis import analyze_candidate, find_best_tif  # noqa: E402

# Datum correction is optional; gracefully skip if module not yet characterised
try:
    from scripts.datum_correction import load_anchors, batch_correct, ANCHOR_FILE
    _DATUM_AVAILABLE = True
except ImportError:
    _DATUM_AVAILABLE = False

METER_PER_DEG_LAT = 111_320.0

# ── Approximate western shoreline of Lake Huron (MI side), longitude at each latitude ──
# These are the MINIMUM longitudes at which water exists (east = larger values).
# Points WEST of this line are over Michigan land.
HURON_WEST_SHORE = [
    # (lat_deg, min_lon_for_water)
    (43.0, -82.45),   # Port Huron
    (43.3, -82.50),
    (43.6, -82.60),   # start of Saginaw Bay region
    (43.8, -83.80),   # Saginaw Bay extends west to ~-83.9
    (44.0, -83.90),
    (44.2, -83.70),   # north end of Saginaw Bay
    (44.5, -83.40),   # main lakeshore resumes
    (44.8, -83.50),
    (45.0, -84.00),
    (45.3, -84.20),
    (45.5, -84.50),   # Straits of Mackinac/Cheboygan area
    (45.8, -84.70),
    (46.1, -84.00),   # UP Michigan, east of I-75
    (46.5, -83.50),
]

# Georgian Bay / North Channel are east of ~-80°W
# Everything east of -79.5 in our scan is Ontario / Georgian Bay = legit water
EAST_ONTARIO_LON = -79.5


def _west_shore_lon_at_lat(lat: float) -> float:
    """Interpolate the Lake Huron western shoreline longitude at a given latitude."""
    pts = HURON_WEST_SHORE
    if lat <= pts[0][0]:
        return pts[0][1]
    if lat >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        lat0, lon0 = pts[i]
        lat1, lon1 = pts[i + 1]
        if lat0 <= lat <= lat1:
            t = (lat - lat0) / (lat1 - lat0)
            return lon0 + t * (lon1 - lon0)
    return pts[-1][1]


def classify_huron(lat: float, lon: float) -> str:
    min_lon = _west_shore_lon_at_lat(lat)
    if lon < min_lon - 0.30:  # clearly over land
        return "LAND"
    if lon < min_lon:         # within 0.3° (~25 km) of shore — shore zone
        return "SHORE"
    return "LAKE"


# Lake Erie: simple bbox — Erie is fully enclosed east-west between -83.6 and -78.8
# Western shore (OH/MI) approx -83.5; eastern shore (NY/PA) approx -78.8
# Northern shore (ON) approx 43.0; southern shore (OH) approx 41.3
def classify_erie(lat: float, lon: float) -> str:
    # main lake water is roughly north of 41.4N and east of -83.5W
    if lon < -83.5 or lat < 41.35:
        return "LAND"
    if lon < -83.3:
        return "SHORE"
    return "LAKE"


CLASSIFIERS = {
    "huron": classify_huron,
    "erie": classify_erie,
}

SCAN_DIRS = {
    "huron": REPO / "adaptive_bg_huron_full_1000yd",
    "erie":  REPO / "adaptive_bg_erie_1000yd",
}

KML_COLORS = {
    # tier → (outline_hex_abgr, fill_hex_abgr)
    "TIER1":      ("ff00ff00", "aa00ff00"),  # bright green  — strong man-made
    "TIER2":      ("ff00dd88", "8800dd88"),  # green-teal    — moderate man-made
    "TIER3":      ("ff00ccff", "6600ccff"),  # yellow-cyan   — weak man-made
    "GEOLOGICAL": ("ff888888", "44888888"),  # gray          — likely geological
    "SHORE":      ("ff0088ff", "440088ff"),  # orange        — shore-zone, check
    "LAND":       ("ff0000ff", "440000ff"),  # red           — discard
    "DISCARD":    ("ff0000ff", "440000ff"),  # red
    "UNKNOWN":    ("ffaaaaaa", "44aaaaaa"),  # light gray
}

# ── Regional geology context table ───────────────────────────────────────────
# Lake Erie sits on flat-lying Paleozoic sedimentary rock — magnetically very quiet.
# Any anomaly >150nT above floor there is geologically unusual.
# Lake Huron north shore is on the Precambrian Canadian Shield — more magnetic.
LAKE_GEO_QUIET = {
    "erie":  True,   # sedimentary basin, quiet
    "huron": False,  # mixed geology, noisier
}

# Regional geological strike in the Great Lakes (Precambrian basement trends NE-SW)
# A dipole axis close to 45° or 225° is consistent with regional geology.
# Axes near 0°/90°/180° are anomalous and add to man-made score.
REGIONAL_STRIKE_DEG = 45.0  # NE-SW


def _strike_deviation(azimuth_deg: float) -> float:
    """Angular deviation (0-90) of azimuth from regional geological strike."""
    diff = abs((azimuth_deg % 180) - (REGIONAL_STRIKE_DEG % 180))
    return min(diff, 180 - diff)


# ── Extra discriminator scoring ───────────────────────────────────────────────
def _extra_score(c: dict, analysis: dict, lake: str) -> tuple[float, list[str]]:
    """Compute bonus/penalty score on top of dipole_analysis base score."""
    bonus = 0.0
    reasons = []

    # 1. Multi-source corroboration ─────────────────────────────────────────
    n_hits = c.get("_n_hits", 1)
    if n_hits >= 3:
        bonus += 30
        reasons.append(f"+30 corroborated by {n_hits} independent surveys")
    elif n_hits == 2:
        bonus += 15
        reasons.append(f"+15 corroborated by 2 independent surveys")

    # 2. SNR vs local floor ─────────────────────────────────────────────────
    snr = analysis.get("anomaly", {}).get("snr_vs_bg_std", 0) or 0
    if snr > 4.0:
        bonus += 15
        reasons.append(f"+15 exceptional SNR={snr:.1f}x above local floor")
    elif snr > 2.5:
        bonus += 8
        reasons.append(f"+8 strong SNR={snr:.1f}x above local floor")

    # 3. Regional geology context ───────────────────────────────────────────
    if LAKE_GEO_QUIET.get(lake, False):
        amp_above = analysis.get("anomaly", {}).get("peak_above_bg_nT", 0) or 0
        if amp_above > 200:
            bonus += 15
            reasons.append(f"+15 {amp_above:.0f}nT in geologically QUIET sedimentary basin (Erie)")
        elif amp_above > 100:
            bonus += 8
            reasons.append(f"+8 {amp_above:.0f}nT in quiet sedimentary basin — unusual")

    # 4. Depth proxy from half-width ────────────────────────────────────────
    # Peter's half-slope rule: source depth ≈ half-width at half-amplitude.
    # Adaptive scan width_m is the labeled region extent, not strictly half-amp width,
    # but it's the best proxy we have.  For coarse-resolution data, this gives a
    # rough upper bound; very narrow anomalies suggest shallow, compact sources.
    width_m = c.get("width_m", 3700)
    # At 2 arcmin resolution, minimum reliable width is 1 pixel ≈ 3700m.
    # If the scored width is smaller than one pixel it suggests a sub-pixel source.
    depth_proxy_m = width_m / 2.0
    if depth_proxy_m < 500:
        bonus += 12
        reasons.append(f"+12 depth proxy {depth_proxy_m:.0f}m — matches ship-depth range")
    elif depth_proxy_m < 1500:
        bonus += 5
        reasons.append(f"+5 depth proxy {depth_proxy_m:.0f}m — plausible for shallow wreck")

    # 5. Dipole axis vs regional geological strike ──────────────────────────
    az = analysis.get("dipole", {}).get("azimuth_deg")
    if az is not None:
        dev = _strike_deviation(az)
        if dev > 60:
            bonus += 10
            reasons.append(f"+10 dipole axis {az:.0f}° is {dev:.0f}° off regional NE-SW geology (anomalous orientation)")
        elif dev < 20:
            bonus -= 8
            reasons.append(f"-8 dipole axis {az:.0f}° aligns with regional NE-SW strike (geological-consistent)")

    # 6. Gradient contrast vs background ───────────────────────────────────
    # Already in dipole base score; no double-count — skip.

    # 7. Asymmetric dipole penalty mitigation at coarse resolution ──────────
    # At 2 arcmin the negative lobe of a shallow wreck often falls in an adjacent cell.
    # If the resolution is coarse (width_m > 3000) reduce the geological asymmetry penalty.
    lobe_ratio = analysis.get("dipole", {}).get("lobe_symmetry_ratio")
    if lobe_ratio is not None and lobe_ratio < 0.25 and width_m > 3000:
        bonus += 8
        reasons.append(f"+8 asymmetric dipole ({lobe_ratio:.2f}) at coarse res — negative lobe likely in adjacent pixel")

    return bonus, reasons


def score_candidate(c: dict, lake: str, verbose: bool = False) -> dict:
    """Run full dipole analysis + extra discriminators on a single candidate."""
    if c.get("_geo_tag") == "LAND":
        c["_tier"] = "DISCARD"
        c["_composite_score"] = 0
        c["_dipole_score"] = 0
        c["_dipole_verdict"] = "LAND — NOT ANALYZED"
        return c

    lat, lon = c["center_lat"], c["center_lon"]
    try:
        tif = find_best_tif(lat, lon)
        analysis = analyze_candidate(tif, lat, lon)

        if "error" in analysis:
            c["_dipole_error"] = analysis["error"]
            c["_dipole_score"] = 0
            c["_dipole_verdict"] = f"ERROR: {analysis['error']}"
            c["_composite_score"] = 0
            c["_tier"] = "UNKNOWN"
            return c

        base_score = analysis["classification"]["score_manmade_pct"]
        base_reasons = analysis["classification"]["reasons"]
        bonus, bonus_reasons = _extra_score(c, analysis, lake)

        composite = base_score + bonus
        c["_dipole"] = analysis
        c["_dipole_score"] = round(base_score, 1)
        c["_dipole_verdict"] = analysis["classification"]["verdict"]
        c["_bonus_score"] = round(bonus, 1)
        c["_bonus_reasons"] = bonus_reasons
        c["_composite_score"] = round(composite, 1)
        c["_all_reasons"] = base_reasons + bonus_reasons
        c["_tif_used"] = tif.name

        geo_tag = c.get("_geo_tag", "LAKE")
        if geo_tag in ("LAND",):
            c["_tier"] = "DISCARD"
        elif composite >= 75:
            c["_tier"] = "TIER1"
        elif composite >= 50:
            c["_tier"] = "TIER2"
        elif composite >= 25:
            c["_tier"] = "TIER3"
        else:
            c["_tier"] = "GEOLOGICAL"

        if verbose:
            _print_candidate_report(c)

    except Exception as e:
        c["_dipole_error"] = str(e)
        c["_dipole_score"] = 0
        c["_dipole_verdict"] = f"FAILED: {e}"
        c["_composite_score"] = 0
        c["_tier"] = "UNKNOWN"

    return c


def _print_candidate_report(c: dict) -> None:
    lat, lon = c["center_lat"], c["center_lon"]
    tier = c.get("_tier", "?")
    comp = c.get("_composite_score", 0)
    verdict = c.get("_dipole_verdict", "")
    print(f"\n  ── ({lat:.4f}, {lon:.4f})  tier={tier}  composite={comp:.0f}/100 ──")
    print(f"     dipole_score={c.get('_dipole_score',0):.0f}  bonus={c.get('_bonus_score',0):.0f}  verdict={verdict}")
    di = (c.get("_dipole") or {}).get("dipole", {})
    an = (c.get("_dipole") or {}).get("anomaly", {})
    bg = (c.get("_dipole") or {}).get("background", {})
    fl = (c.get("_dipole") or {}).get("flip", {})
    print(f"     bg_mean={bg.get('mean_nT','?'):+.1f}nT  snr={an.get('snr_vs_bg_std','?'):.1f}x  "
          f"peak_above={an.get('peak_above_bg_nT','?'):+.1f}nT")
    print(f"     dipolar={di.get('is_dipolar','?')}  lobe_sym={di.get('lobe_symmetry_ratio','?')}  "
          f"sep={di.get('separation_m','?')}m  az={di.get('azimuth_deg','?')}°")
    print(f"     flip_min={fl.get('min_flip_distance_m','?')}m")
    for r in c.get("_all_reasons", []):
        print(f"       {r}")

# ── Dedup ─────────────────────────────────────────────────────────────────────
DEDUP_DIST_M = 5_000  # merge candidates within 5 km


def _dist_m(la1, lo1, la2, lo2) -> float:
    dlat = (la2 - la1) * METER_PER_DEG_LAT
    dlon = (lo2 - lo1) * (METER_PER_DEG_LAT * math.cos(math.radians((la1 + la2) / 2)))
    return math.hypot(dlat, dlon)


def dedup(cands: list[dict], dist_m: float = DEDUP_DIST_M) -> list[dict]:
    """Merge candidates within dist_m of each other, keeping highest score."""
    out: list[dict] = []
    used = [False] * len(cands)
    for i, c in enumerate(cands):
        if used[i]:
            continue
        group = [c]
        for j in range(i + 1, len(cands)):
            if not used[j]:
                d = _dist_m(c["center_lat"], c["center_lon"],
                            cands[j]["center_lat"], cands[j]["center_lon"])
                if d <= dist_m:
                    group.append(cands[j])
                    used[j] = True
        best = max(group, key=lambda x: x["score"])
        best["_sources"] = sorted({g["source_grid"] for g in group})
        best["_n_hits"] = len(group)
        out.append(best)
        used[i] = True
    return out


def _write_filtered_kml(cands: list[dict], classifier, out_kml: Path) -> None:
    placemarks = []
    for c in cands:
        geo_tag = c.get("_geo_tag", "LAKE")
        tier = c.get("_tier", geo_tag)
        color_key = tier if tier in KML_COLORS else geo_tag
        outline_hex, fill_hex = KML_COLORS.get(color_key, KML_COLORS["UNKNOWN"])
        lat, lon = c["center_lat"], c["center_lon"]
        half_w_m = c["width_m"] / 2.0
        half_h_m = c["height_m"] / 2.0
        dlat = half_h_m / METER_PER_DEG_LAT
        dlon = half_w_m / (METER_PER_DEG_LAT * math.cos(math.radians(lat)) + 1e-9)
        ring = [
            (lon - dlon, lat + dlat), (lon + dlon, lat + dlat),
            (lon + dlon, lat - dlat), (lon - dlon, lat - dlat),
            (lon - dlon, lat + dlat),
        ]
        coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in ring)
        sources = ", ".join(c.get("_sources", [c["source_grid"]]))
        n_hits = c.get("_n_hits", 1)
        comp = c.get("_composite_score", 0)
        verdict = c.get("_dipole_verdict", "")
        reasons = "\n".join(c.get("_all_reasons", []))
        an = (c.get("_dipole") or {}).get("anomaly", {})
        bg = (c.get("_dipole") or {}).get("background", {})
        di = (c.get("_dipole") or {}).get("dipole", {})
        fl = (c.get("_dipole") or {}).get("flip", {})
        gr = (c.get("_dipole") or {}).get("gradient", {})
        snr = an.get("snr_vs_bg_std", "?")
        bg_mean = bg.get("mean_nT", "?")
        peak_above = an.get("peak_above_bg_nT", "?")
        is_dipolar = di.get("is_dipolar", "?")
        lobe_sym = di.get("lobe_symmetry_ratio", "?")
        sep_m = di.get("separation_m", "?")
        az = di.get("azimuth_deg", "?")
        flip = fl.get("min_flip_distance_m", "?")
        grad_c = gr.get("contrast_ratio", "?")

        desc = (
            f"lat={lat:.4f}  lon={lon:.4f}\n"
            f"TIER={tier}  composite={comp:.0f}/100\n"
            f"verdict: {verdict}\n"
            f"\n── AMPLITUDE ──\n"
            f"bg_floor={bg_mean}nT  peak_above={peak_above}nT  SNR={snr}x\n"
            f"amp_peak_reported={c['amplitude_peak_abs']:.1f}nT\n"
            f"\n── DIPOLE ──\n"
            f"dipolar={is_dipolar}  lobe_sym={lobe_sym}  sep={sep_m}m  az={az}deg\n"
            f"polarity_flip_min={flip}m  gradient_contrast={grad_c}x\n"
            f"\n── META ──\n"
            f"hits={n_hits}  sources={sources}\n"
            f"tif={c.get('_tif_used','?')}\n"
            f"\n── SCORING ──\n{reasons}"
        )

        placemarks.append(f"""
<Placemark>
  <name>[{tier}] {comp:.0f}/100 amp={c['amplitude_peak_abs']:.0f}nT hits={n_hits}</name>
  <description><![CDATA[{desc}]]></description>
  <Style>
    <IconStyle><color>{outline_hex}</color><scale>1.0</scale></IconStyle>
  </Style>
  <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
</Placemark>
<Placemark>
  <name>Box {tier} {lat:.3f},{lon:.3f}</name>
  <Style>
    <LineStyle><color>{outline_hex}</color><width>2</width></LineStyle>
    <PolyStyle><color>{fill_hex}</color></PolyStyle>
  </Style>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
""")

    legend = (
        "TIER1 bright green = strong man-made (75+/100)\n"
        "TIER2 green-teal   = moderate man-made (50-74)\n"
        "TIER3 yellow-cyan  = weak man-made (25-49)\n"
        "GEOLOGICAL gray    = likely geological\n"
        "LAND red           = land/road noise DISCARD\n"
    )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        '    <name>Scored Candidates — Geo+Dipole+Context</name>\n'
        f'    <description><![CDATA[{legend}]]></description>\n'
        + "".join(placemarks)
        + "  </Document>\n</kml>"
    )
    out_kml.write_text(kml, encoding="utf-8")


def process_lake(lake: str, verbose: bool = False) -> dict:
    scan_dir = SCAN_DIRS[lake]
    json_path = scan_dir / "adaptive_candidates.json"
    if not json_path.exists():
        print(f"No candidates found for {lake}: {json_path}")
        return {}

    raw = json.loads(json_path.read_text())
    classifier = CLASSIFIERS[lake]

    # Dedup
    deduped = dedup(raw)

    # Geo-classify
    for c in deduped:
        c["_geo_tag"] = classifier(c["center_lat"], c["center_lon"])

    # Run dipole analysis + extra scoring on every candidate
    print(f"\n{'='*60}")
    print(f"  {lake.upper()}  raw={len(raw)}  deduped={len(deduped)}")
    print(f"  Running dipole analysis on {len(deduped)} candidates...")
    for i, c in enumerate(deduped):
        print(f"  [{i+1}/{len(deduped)}] ({c['center_lat']:.4f},{c['center_lon']:.4f}) ...", end=" ", flush=True)
        score_candidate(c, lake, verbose=verbose)
        print(f"tier={c.get('_tier','?')}  composite={c.get('_composite_score',0):.0f}")

    # Sort by composite score
    deduped.sort(key=lambda x: x.get("_composite_score", 0), reverse=True)

    # Tier counts
    from collections import Counter
    tier_counts = Counter(c.get("_tier", "UNKNOWN") for c in deduped)

    print(f"\n{'='*60}")
    print(f"  {lake.upper()} — FINAL SCORED TABLE")
    print(f"{'='*60}")
    print(f"{'#':>3}  {'lat':>7}  {'lon':>8}  {'comp':>5}  {'dipole':>5}  {'bonus':>5}  {'tier':12}  verdict")
    for i, c in enumerate(deduped):
        print(
            f"{i+1:3d}  {c['center_lat']:7.4f}  {c['center_lon']:8.4f}"
            f"  {c.get('_composite_score',0):5.0f}"
            f"  {c.get('_dipole_score',0):5.0f}"
            f"  {c.get('_bonus_score',0):5.0f}"
            f"  {c.get('_tier','?'):12s}  {c.get('_dipole_verdict','')[:45]}"
        )
    print(f"\n  Tier summary: " + "  ".join(f"{t}={n}" for t, n in sorted(tier_counts.items())))

    land = [c for c in deduped if c.get("_geo_tag") == "LAND"]
    if land:
        print(f"\n  *** {len(land)} LAND/ROAD DISCARDS: ***")
        for c in land:
            print(f"      ({c['center_lat']:.4f},{c['center_lon']:.4f}) amp={c['amplitude_peak_abs']:.0f}nT")

    # Write outputs
    kml_out = scan_dir / "adaptive_candidates_scored.kml"
    _write_filtered_kml(deduped, classifier, kml_out)

    json_out = scan_dir / "adaptive_candidates_scored.json"
    json_out.write_text(json.dumps(deduped, indent=2, default=str), encoding="utf-8")

    lake_only = [c for c in deduped if c.get("_geo_tag") == "LAKE"]

    # Optional datum correction (NAD27→WGS84 + Loran-C rubber-sheeting)
    if lake_only and _DATUM_AVAILABLE and ANCHOR_FILE.exists():
        try:
            anchors = load_anchors(ANCHOR_FILE)
            n_ready = sum(1 for a in anchors
                          if a.get("verified") and a.get("survey_pos") and a.get("lake") == lake)
            if n_ready >= 3:
                lake_only = batch_correct(lake_only, anchors, datum="nad27")
                print(f"\n  Datum correction applied ({n_ready} anchors for {lake})")
            else:
                # Always apply Molodensky even without rubber-sheeting anchors
                lake_only = batch_correct(lake_only, anchors, datum="nad27")
                print(f"\n  Molodensky only applied (only {n_ready} characterised anchors for {lake}; "
                      f"run: python scripts/datum_correction.py --characterize)")
        except Exception as _de:
            print(f"  Datum correction skipped: {_de}")

    csv_out = scan_dir / "adaptive_candidates_scored.csv"
    if lake_only:
        safe_fields = [k for k in lake_only[0].keys() if not k.startswith("_dipole")]
        extra = ["_geo_tag", "_tier", "_composite_score", "_dipole_score",
                 "_bonus_score", "_dipole_verdict", "_n_hits", "_tif_used"]
        # Add datum columns if correction ran
        datum_cols = [k for k in ["corrected_lat", "corrected_lon",
                                  "datum_total_shift_m", "datum_method",
                                  "datum_anchors_used"]
                      if k in lake_only[0]]
        with csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=safe_fields + extra + datum_cols)
            w.writeheader()
            for c in lake_only:
                w.writerow({k: c.get(k, "") for k in w.fieldnames})

    return {
        "raw": len(raw), "deduped": len(deduped),
        "tier_counts": dict(tier_counts),
        "kml": str(kml_out),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lakes", default="huron,erie")
    p.add_argument("--verbose", action="store_true", help="Print per-candidate scoring breakdown")
    p.add_argument("--no-datum", action="store_true",
                   help="Skip datum correction (useful when anchor file not characterised)")
    args = p.parse_args()
    if args.no_datum:
        global _DATUM_AVAILABLE
        _DATUM_AVAILABLE = False
    summary = {}
    for lake in args.lakes.split(","):
        summary[lake] = process_lake(lake.strip(), verbose=args.verbose)
    print("\n\nSUMMARY:")
    for lake, r in summary.items():
        if r:
            tiers = r.get("tier_counts", {})
            print(f"  {lake}: {tiers}")
            print(f"    KML: {r['kml']}")


if __name__ == "__main__":
    main()
