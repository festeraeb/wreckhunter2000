"""Historical Drift Analysis Engine for Bagrecovery.

Uses CESAROPS physics layer (no OpenDrift required) to run:
  - Forward seeding: from last known position + estimated storm forcing
  - Backward drift: from debris/body recovery to estimate sinking origin
  - Consistency check: forward from magnetic candidate → does it reach known debris?
  - Analog storm scoring: find modern Dec storms similar to the 1909 event

The engine is self-contained with CESAROPS as an OPTIONAL dependency.
If CESAROPS is not installed, it falls back to internal Haversine physics.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── CESAROPS physics (optional but preferred) ────────────────────────────────
_CES_PHYSICS = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cesarops" / "src"))
    from cesarops.core.physics import (
        haversine as _ces_haversine,
        destination_point as _ces_dest,
        basic_drift_prediction as _ces_drift,
        wind_to_uv as _ces_wind_uv,
    )
    _CES_PHYSICS = True
except ImportError:
    _CES_PHYSICS = False

REPO = Path(__file__).resolve().parents[1]
METER_PER_DEG_LAT = 111_320.0


# ── Pure-Python fallbacks (mirror CESAROPS physics.py exactly) ───────────────
def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in nautical miles."""
    if _CES_PHYSICS:
        return _ces_haversine(lat1, lon1, lat2, lon2)
    EARTH_R_KM = 6371.0; KM_TO_NM = 0.539957
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return EARTH_R_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * KM_TO_NM


def _destination(lat, lon, bearing_deg, dist_nm) -> Tuple[float, float]:
    """New point from start + bearing + nautical miles."""
    if _CES_PHYSICS:
        return _ces_dest(lat, lon, bearing_deg, dist_nm)
    NM_TO_M = 1852.0; R = 6_371_000.0
    d = dist_nm * NM_TO_M / R
    b = math.radians(bearing_deg); lr = math.radians(lat); lor = math.radians(lon)
    lat2 = math.asin(math.sin(lr)*math.cos(d) + math.cos(lr)*math.sin(d)*math.cos(b))
    lon2 = lor + math.atan2(math.sin(b)*math.sin(d)*math.cos(lr),
                            math.cos(d)-math.sin(lr)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def _basic_drift(wind_speed_ms, wind_dir_deg, current_u=0.0, current_v=0.0,
                 wave_ht=0.0, windage=0.03) -> Tuple[float, float]:
    """Drift velocity (u, v) in m/s."""
    if _CES_PHYSICS:
        return _ces_drift(wind_speed_ms, wind_dir_deg, current_u, current_v, wave_ht, windage)
    rad = math.radians(wind_dir_deg)
    wu = wind_speed_ms * math.sin(rad); wv = wind_speed_ms * math.cos(rad)
    return current_u + windage*wu, current_v + windage*wv


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class StormPhase:
    """A single time phase of a storm (constant wind speed/dir over dt_hours)."""
    label: str
    wind_speed_mph: float
    wind_dir_deg: float        # meteorological (FROM direction, 0=N)
    dt_hours: float
    wave_ht_ft: float = 0.0
    current_speed_kt: float = 0.0
    current_dir_deg: float = 0.0


@dataclass
class Anchor:
    """A known positional or temporal constraint on a case."""
    name: str
    lat: float
    lon: float
    timestamp: Optional[str] = None    # ISO8601
    anchor_type: str = "debris"        # departure | sinking_est | lifeboat | body | debris | candidate
    confidence: float = 0.5            # 0-1
    notes: str = ""
    # If this anchor is a DRIFT ENDPOINT (body/debris), these describe the object:
    object_type: str = "debris"        # ship | lifeboat | body | life_ring | cargo
    windage_factor: float = 0.03       # fraction of wind speed that drives drift


@dataclass
class DriftParticle:
    lat: float; lon: float
    path: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DriftResult:
    """Output of a Monte Carlo drift run."""
    mode: str                          # "forward" | "backward"
    n_particles: int
    duration_hours: float
    final_positions: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    spread_nm: float                   # 1-sigma radius around centroid
    bbox: Tuple[float, float, float, float]  # minlat, minlon, maxlat, maxlon
    path_samples: List[List[Tuple[float, float]]]  # up to 10 sampled paths
    storm_phases: List[dict] = field(default_factory=list)
    notes: str = ""


# ── Pre-populated M&B No.2 case ──────────────────────────────────────────────

MB2_CASE = {
    "name": "Marquette & Bessemer No. 2",
    "vessel_type": "Steel railroad car ferry",
    "length_ft": 338,
    "tonnage_grt": 3160,
    "ferrous_mass_tons_est": 3000,   # hull + 26 railroad cars of freight
    "crew": 33,

    # ── chronology ──────────────────────────────────────────────────────────
    "departure": Anchor(
        name="Departure — Conneaut, OH",
        lat=41.953, lon=-80.554,
        timestamp="1909-12-07T21:00:00",
        anchor_type="departure",
        confidence=0.95,
        notes="Left Conneaut approximately 9 PM Dec 7, 1909. Exact time disputed; "
              "departure was confirmed by dock workers.",
        object_type="ship",
        windage_factor=0.03,
    ),
    "destination": Anchor(
        name="Intended Destination — Port Stanley, ON",
        lat=42.668, lon=-81.546,
        anchor_type="departure",
        confidence=1.0,
        notes="Confirmed route Conneaut → Port Stanley.",
        object_type="ship",
        windage_factor=0.0,
    ),

    # ── stopped watch anchor (best sinking time estimate) ────────────────────
    "stopped_watch": Anchor(
        name="Stopped Watch — First Engineer body",
        lat=42.14, lon=-80.09,    # Body found near Presque Isle / Erie PA shore
        timestamp="1910-01-26T00:00:00",
        anchor_type="body",
        confidence=0.7,
        notes="Body of chief engineer found near Erie, PA shore Jan 26, 1910. "
              "Watch stopped at approximately 12:17 AM — believed Dec 8, 1909, "
              "~3h17m after 9 PM departure (exact date and AM/PM debated). "
              "Recovery location is approximate — confirm from historical records.",
        object_type="body",
        windage_factor=0.02,
    ),

    # ── estimated sinking time from watch ───────────────────────────────────
    "sinking_timestamp": "1909-12-08T00:17:00",   # 12:17 AM Dec 8
    "sinking_hours_after_departure": 3.3,

    # ── known debris / recovery points (fill in from your records) ───────────
    "anchors": [
        Anchor(
            name="Life Ring / Debris — south shore",
            lat=41.97, lon=-80.45,
            anchor_type="debris",
            confidence=0.5,
            notes="Approximate — update with specific recovery location from records.",
            object_type="life_ring",
            windage_factor=0.08,
        ),
        Anchor(
            name="Magnetic Candidate (Tier1 ERIE)",
            lat=42.4250, lon=-80.8130,
            anchor_type="candidate",
            confidence=0.75,
            notes="305 nT anomaly, composite score 99/100, dipolar, compact. "
                  "30.6 nm from Conneaut, consistent with 9.3 kt for 3.3h.",
            object_type="ship",
            windage_factor=0.0,
        ),
    ],

    # ── Dec 7-8 1909 storm reconstruction ───────────────────────────────────
    # Source: Monthly Weather Review Dec 1909, analog composite, historian accounts
    # The storm moved NE across Ontario; winds were NW-W at peak
    "storm_phases": [
        StormPhase("Pre-storm (dep. to ~11 PM Dec 7)",
                   wind_speed_mph=30, wind_dir_deg=270,  # W
                   dt_hours=2.0, wave_ht_ft=4, current_speed_kt=0.3, current_dir_deg=90),
        StormPhase("Storm build (11 PM Dec 7 - 1 AM Dec 8)",
                   wind_speed_mph=55, wind_dir_deg=300,  # NW
                   dt_hours=2.0, wave_ht_ft=8, current_speed_kt=0.5, current_dir_deg=120),
        StormPhase("Storm peak (1 AM - 6 AM Dec 8)",
                   wind_speed_mph=68, wind_dir_deg=315,  # NNW
                   dt_hours=5.0, wave_ht_ft=14, current_speed_kt=0.6, current_dir_deg=135),
        StormPhase("Post-sink debris drift (Dec 8 - Jan 26 1910)",
                   wind_speed_mph=18, wind_dir_deg=285,  # avg W winds for 49 days
                   dt_hours=49*24, wave_ht_ft=3, current_speed_kt=0.2, current_dir_deg=100),
    ],
}


# ── Monte Carlo drift engine ─────────────────────────────────────────────────

MPH_TO_MS = 0.44704


def _run_particles(
    start_lat: float,
    start_lon: float,
    phases: List[StormPhase],
    n: int,
    windage_factor: float,
    direction: int = 1,         # +1 forward, -1 backward
    wind_spread_deg: float = 20.0,   # ±1σ wind direction uncertainty
    wind_speed_cv: float = 0.15,     # ±1σ coefficient of variation on wind speed
    rng_seed: Optional[int] = None,
) -> DriftResult:
    """Core Monte Carlo drift — runs N particles through the storm phases."""
    rng = random.Random(rng_seed)
    particles = [DriftParticle(start_lat, start_lon) for _ in range(n)]

    total_hours = sum(p.dt_hours for p in phases)
    DT_STEP = 0.25  # 15-minute time steps

    for phase in phases:
        steps = int(round(phase.dt_hours / DT_STEP))
        ws_ms = phase.wind_speed_mph * MPH_TO_MS
        cs_ms = phase.current_speed_kt * 0.514444
        wave_ft = phase.wave_ht_ft

        for p in particles:
            for _ in range(steps):
                # Sample from parameter uncertainty
                eff_ws = ws_ms * max(0.1, rng.gauss(1.0, wind_speed_cv))
                eff_wd = phase.wind_dir_deg + rng.gauss(0, wind_spread_deg)
                cur_dir = math.radians(phase.current_dir_deg)
                cur_u = cs_ms * math.sin(cur_dir)
                cur_v = cs_ms * math.cos(cur_dir)

                du, dv = _basic_drift(eff_ws, eff_wd, cur_u, cur_v,
                                      wave_ft * 0.3048, windage_factor)
                # Convert m/s → nm in DT_STEP hours, apply direction
                speed_ms = math.hypot(du, dv)
                if speed_ms < 1e-9:
                    continue
                drift_dir_deg = (math.degrees(math.atan2(du, dv)) + 360) % 360
                dist_nm = direction * speed_ms * DT_STEP * 3600 / 1852.0
                p.lat, p.lon = _destination(p.lat, p.lon, drift_dir_deg, dist_nm)
                p.path.append((p.lat, p.lon))

    finals = [(p.lat, p.lon) for p in particles]
    lats = [f[0] for f in finals]; lons = [f[1] for f in finals]
    clat = sum(lats)/n; clon = sum(lons)/n
    spread = sum(_haversine(clat, clon, la, lo) for la, lo in finals) / n

    # Sample up to 10 paths for KML/display
    samples = [p.path[::max(1, len(p.path)//50)] for p in
               rng.sample(particles, min(10, n))]

    return DriftResult(
        mode="forward" if direction == 1 else "backward",
        n_particles=n,
        duration_hours=total_hours,
        final_positions=finals,
        centroid=(clat, clon),
        spread_nm=spread,
        bbox=(min(lats), min(lons), max(lats), max(lons)),
        path_samples=samples,
        storm_phases=[asdict(ph) if hasattr(ph, '__dataclass_fields__') else ph
                      for ph in phases],
    )


def forward_seed(
    dep_lat: float, dep_lon: float,
    phases: List[StormPhase],
    object_type: str = "ship",
    n_particles: int = 500,
) -> DriftResult:
    """Forward seeding from departure point through storm phases."""
    windage = _windage_for_type(object_type)
    # Only use phases up to sinking time for forward seeding of the ship
    return _run_particles(dep_lat, dep_lon, phases, n_particles, windage,
                          direction=1, rng_seed=42)


def backward_drift(
    recovery_lat: float, recovery_lon: float,
    phases: List[StormPhase],
    object_type: str = "body",
    n_particles: int = 500,
) -> DriftResult:
    """Backward drift from a recovery point to estimate origin."""
    windage = _windage_for_type(object_type)
    return _run_particles(recovery_lat, recovery_lon, phases, n_particles, windage,
                          direction=-1, rng_seed=42)


def consistency_check(
    candidate_lat: float, candidate_lon: float,
    debris_anchor: Anchor,
    phases: List[StormPhase],
    n_particles: int = 200,
) -> dict:
    """Forward drift from candidate through post-sink phases, compare to debris location."""
    post_sink_phases = [p for p in phases if "debris" in p.label.lower() or
                        p.dt_hours > 24]
    if not post_sink_phases:
        post_sink_phases = phases[-1:]

    result = _run_particles(candidate_lat, candidate_lon, post_sink_phases,
                            n_particles, debris_anchor.windage_factor,
                            direction=1, rng_seed=123)

    dist_to_debris = _haversine(result.centroid[0], result.centroid[1],
                                debris_anchor.lat, debris_anchor.lon)
    frac_within_1sigma = sum(
        1 for lat, lon in result.final_positions
        if _haversine(lat, lon, debris_anchor.lat, debris_anchor.lon) <= result.spread_nm
    ) / result.n_particles

    return {
        "candidate": (candidate_lat, candidate_lon),
        "debris_target": (debris_anchor.lat, debris_anchor.lon),
        "predicted_centroid": result.centroid,
        "dist_centroid_to_debris_nm": round(dist_to_debris, 2),
        "spread_1sigma_nm": round(result.spread_nm, 2),
        "fraction_within_spread": round(frac_within_1sigma, 3),
        "consistent": dist_to_debris <= 2 * result.spread_nm,
        "drift_result": result,
    }


def _windage_for_type(obj: str) -> float:
    return {"ship": 0.03, "lifeboat": 0.10, "life_ring": 0.08,
            "body": 0.02, "debris": 0.04, "cargo": 0.03}.get(obj.lower(), 0.03)


# ── Ensemble runner (analog storms) ──────────────────────────────────────────

def ensemble_forward_seed(
    dep_lat: float,
    dep_lon: float,
    analog_phase_sets: List[List[StormPhase]],
    analog_labels: List[str],
    object_type: str = "ship",
    n_particles: int = 300,
) -> dict:
    """Run forward seeding for multiple analog storm phase sets.

    Returns a summary with per-analog centroids + an ensemble centroid
    (mean of all analog centroids) and inter-analog spread.

    Parameters
    ----------
    analog_phase_sets : list of StormPhase lists
        One list per analog storm.  Each list covers the sinking window only
        (pre-storm + build + peak).
    analog_labels : list of str
        Human-readable name for each analog.
    """
    results = []
    for phases, label in zip(analog_phase_sets, analog_labels):
        res = forward_seed(dep_lat, dep_lon, phases, object_type, n_particles)
        res.notes = label
        results.append(res)

    if not results:
        return {"error": "No analog results"}

    # Ensemble centroid = mean of all analog centroids
    e_lat = sum(r.centroid[0] for r in results) / len(results)
    e_lon = sum(r.centroid[1] for r in results) / len(results)

    # Inter-analog spread (how consistent are the analogs with each other?)
    inter_spread = sum(_haversine(e_lat, e_lon, r.centroid[0], r.centroid[1])
                       for r in results) / len(results)

    # Weighted ensemble: weight by analog similarity score if available
    return {
        "ensemble_centroid": (round(e_lat, 5), round(e_lon, 5)),
        "ensemble_inter_analog_spread_nm": round(inter_spread, 2),
        "n_analogs": len(results),
        "per_analog": [
            {
                "label": r.notes,
                "centroid": r.centroid,
                "spread_nm": round(r.spread_nm, 2),
                "dist_to_ensemble_nm": round(
                    _haversine(r.centroid[0], r.centroid[1], e_lat, e_lon), 2),
            }
            for r in results
        ],
        "drift_results": results,
    }


def build_analog_phases_from_1909(
    scale_factor: float = 1.0,
    direction_offset: float = 0.0,
) -> List[StormPhase]:
    """Return a modified copy of the 1909 storm phases with optional scaling.

    scale_factor > 1 = stronger storm; < 1 = weaker.
    direction_offset (°) = rotate all wind directions.
    Useful for sensitivity analysis.
    """
    phases = []
    for p in MB2_CASE["storm_phases"][:3]:   # sinking window only
        phases.append(StormPhase(
            label=p.label + f" (scale={scale_factor:.2f} off={direction_offset:.0f}°)",
            wind_speed_mph=round(p.wind_speed_mph * scale_factor, 1),
            wind_dir_deg=(p.wind_dir_deg + direction_offset) % 360,
            dt_hours=p.dt_hours,
            wave_ht_ft=round(p.wave_ht_ft * scale_factor, 1),
            current_speed_kt=p.current_speed_kt,
            current_dir_deg=p.current_dir_deg,
        ))
    return phases


def sensitivity_sweep(
    dep_lat: float, dep_lon: float,
    n_particles: int = 200,
) -> List[dict]:
    """Sweep wind speed ±20% and direction ±20° to bound sinking zone."""
    results = []
    for speed_scale in (0.8, 1.0, 1.2):
        for dir_offset in (-20, 0, 20):
            phases = build_analog_phases_from_1909(speed_scale, dir_offset)
            res = forward_seed(dep_lat, dep_lon, phases, "ship", n_particles)
            results.append({
                "speed_scale": speed_scale,
                "dir_offset_deg": dir_offset,
                "centroid": res.centroid,
                "spread_nm": round(res.spread_nm, 2),
            })
    return results


# ── CESAROPS case export ─────────────────────────────────────────────────────

def export_cesarops_case(case: dict, drift_results: dict,
                         out_path: Path) -> Path:
    """Write a CESAROPS-compatible case_study JSON file."""
    payload = {
        "case_name": case["name"],
        "vessel_type": case.get("vessel_type", ""),
        "last_seen": asdict(case["departure"]),
        "object_type": "Large Steel Vessel",
        "windage": 0.03,
        "sinking_timestamp": case.get("sinking_timestamp"),
        "anchors": [asdict(a) for a in case.get("anchors", [])],
        "storm_phases": [asdict(p) if hasattr(p, '__dataclass_fields__') else p
                         for p in case.get("storm_phases", [])],
        "drift_results": {
            k: {
                "mode": v.mode,
                "centroid": v.centroid,
                "spread_nm": v.spread_nm,
                "n_particles": v.n_particles,
                "bbox": v.bbox,
            } for k, v in drift_results.items() if isinstance(v, DriftResult)
        },
        "notes": (
            "Exported from Bagrecovery historical drift analysis. "
            "Storm phases are reconstructed from historical accounts and analog storm composites. "
            "Import into CESAROPS for full OpenDrift Lagrangian refinement."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


# ── KML export ───────────────────────────────────────────────────────────────

def export_kml(
    results: dict[str, DriftResult],
    anchors: List[Anchor],
    out_path: Path,
    title: str = "Historical Drift Analysis",
) -> Path:
    """Export drift cones, centroids and anchors to KML."""
    METER_PER_DEG = 111320.0

    def _circle_poly(lat, lon, radius_nm, n=32):
        pts = []
        radius_m = radius_nm * 1852.0
        for i in range(n + 1):
            ang = math.radians(360 * i / n)
            dlat = (radius_m * math.cos(ang)) / METER_PER_DEG
            dlon = (radius_m * math.sin(ang)) / (METER_PER_DEG * math.cos(math.radians(lat)) + 1e-9)
            pts.append(f"{lon+dlon:.6f},{lat+dlat:.6f},0")
        return " ".join(pts)

    COLORS = {
        "forward":  ("ff00ff00", "4400ff00"),
        "backward": ("ff0088ff", "4400aaff"),
    }
    ANCHOR_ICONS = {
        "departure": "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png",
        "body":      "http://maps.google.com/mapfiles/kml/paddle/red-circle.png",
        "debris":    "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png",
        "candidate": "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png",
    }

    placemarks = []

    # Drift cones
    for key, res in results.items():
        if not isinstance(res, DriftResult):
            continue
        clat, clon = res.centroid
        oc, fc = COLORS.get(res.mode, ("ffaaaaaa", "44aaaaaa"))

        # Centroid marker
        placemarks.append(f"""
<Placemark>
  <name>{key} centroid</name>
  <description>mode={res.mode}  spread={res.spread_nm:.1f}nm  n={res.n_particles}</description>
  <Point><coordinates>{clon:.6f},{clat:.6f},0</coordinates></Point>
</Placemark>""")

        # 1-sigma spread circle
        placemarks.append(f"""
<Placemark>
  <name>{key} 1σ cone</name>
  <Style>
    <LineStyle><color>{oc}</color><width>2</width></LineStyle>
    <PolyStyle><color>{fc}</color></PolyStyle>
  </Style>
  <Polygon><outerBoundaryIs><LinearRing>
    <coordinates>{_circle_poly(clat, clon, res.spread_nm)}</coordinates>
  </LinearRing></outerBoundaryIs></Polygon>
</Placemark>""")

        # Sample paths
        for i, path in enumerate(res.path_samples[:5]):
            if len(path) < 2:
                continue
            coords = " ".join(f"{lo:.6f},{la:.6f},0" for la, lo in path)
            placemarks.append(f"""
<Placemark>
  <name>{key} path {i+1}</name>
  <Style><LineStyle><color>{oc}</color><width>1</width></LineStyle></Style>
  <LineString><coordinates>{coords}</coordinates></LineString>
</Placemark>""")

    # Anchors
    for a in anchors:
        icon = ANCHOR_ICONS.get(a.anchor_type, ANCHOR_ICONS["debris"])
        placemarks.append(f"""
<Placemark>
  <name>{a.name}</name>
  <description><![CDATA[type={a.anchor_type}  conf={a.confidence}
{a.notes}]]></description>
  <Style><IconStyle><Icon><href>{icon}</href></Icon></IconStyle></Style>
  <Point><coordinates>{a.lon:.6f},{a.lat:.6f},0</coordinates></Point>
</Placemark>""")

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        f'  <Document><name>{title}</name>\n'
        + "".join(placemarks)
        + "\n  </Document>\n</kml>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")
    return out_path


# ── Quick CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    case = MB2_CASE
    dep = case["departure"]
    phases = case["storm_phases"]
    sinking_phases = phases[:2]   # just pre-storm + build
    post_phases = phases[3:]      # long debris drift phase

    print("Running M&B No.2 forward seed (ship, 3.3h storm)...")
    fwd = forward_seed(dep.lat, dep.lon, sinking_phases, "ship", n_particles=500)
    print(f"  centroid = {fwd.centroid[0]:.4f}°N, {fwd.centroid[1]:.4f}°W  spread={fwd.spread_nm:.1f}nm")

    print("Running backward drift from stopped-watch body recovery...")
    bdy = case["stopped_watch"]
    bwd = backward_drift(bdy.lat, bdy.lon, [phases[3]], "body", n_particles=500)
    print(f"  centroid = {bwd.centroid[0]:.4f}°N, {bwd.centroid[1]:.4f}°W  spread={bwd.spread_nm:.1f}nm")

    print("Checking consistency with magnetic candidate...")
    cand = case["anchors"][1]  # candidate
    check = consistency_check(cand.lat, cand.lon, case["anchors"][0], post_phases)
    print(f"  dist centroid→debris = {check['dist_centroid_to_debris_nm']:.1f}nm  "
          f"consistent={check['consistent']}")

    out = REPO / "outputs" / "mb2_historic_drift.kml"
    export_kml({"forward": fwd, "backward": bwd}, [dep, bdy] + case["anchors"], out)
    print(f"KML: {out}")

    out_j = REPO / "outputs" / "mb2_cesarops_case.json"
    export_cesarops_case(case, {"forward": fwd, "backward": bwd}, out_j)
    print(f"CESAROPS JSON: {out_j}")
