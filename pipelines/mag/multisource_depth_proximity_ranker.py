#!/usr/bin/env python3
"""Rank targets using multisource anomaly evidence + depth/proximity detectability.

Goals:
1) Prioritize likely targets to go scan.
2) Quantify where a steel wreck of expected size/depth is likely *not there*.

Inputs:
- Aero candidate JSON from adaptive scan.
- Satellite candidate JSON from adaptive scan.
- Track coverage CSV (for proximity to flown/surveyed tracks).

Outputs:
- ranked_targets.csv            (overall view)
- go_scan_targets.csv           (high-priority outbound scans)
- likely_not_there.csv          (high-confidence negative areas)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sigmoid(x: float) -> float:
    # Clamp to avoid overflow on very large distances.
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SiteScore:
    lat: float
    lon: float
    aero_score: float
    sat_score: float
    fused_anomaly_score: float
    nearest_cov_point_km: float
    track_distance_m_proxy: float
    mbes_distance_m_proxy: float
    coverage_confidence: float
    assumed_depth_m: float
    expected_wreck_length_m: float
    detectability_prior: float
    go_scan_priority: float
    likely_not_there_confidence: float
    note: str


def _load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _to_sites(cands: list[dict]) -> dict[tuple[float, float], float]:
    sites: dict[tuple[float, float], float] = {}
    for c in cands:
        lat = float(c.get("center_lat", c.get("lat", 0.0)))
        lon = float(c.get("center_lon", c.get("lon", 0.0)))
        score = float(c.get("score", c.get("anomaly_score", 0.0)))
        key = (round(lat, 5), round(lon, 5))
        sites[key] = max(score, sites.get(key, 0.0))
    return sites


def _load_coverage_points(path: Path) -> list[dict]:
    points = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                lat = float(row.get("latitude", ""))
                lon = float(row.get("longitude", ""))
            except ValueError:
                continue

            def _float_or_nan(v: str) -> float:
                try:
                    return float(v)
                except Exception:
                    return float("nan")

            points.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "track_distance_m": _float_or_nan(row.get("track_distance_m", "nan")),
                    "mbes_distance_m": _float_or_nan(row.get("mbes_distance_m", "nan")),
                }
            )
    return points


def _nearest_coverage(lat: float, lon: float, points: list[dict]) -> tuple[float, float, float]:
    if not points:
        return float("nan"), float("nan"), float("nan")
    best = None
    best_km = float("inf")
    for p in points:
        d = haversine_km(lat, lon, p["lat"], p["lon"])
        if d < best_km:
            best_km = d
            best = p
    return best_km, float(best.get("track_distance_m", float("nan"))), float(best.get("mbes_distance_m", float("nan")))


def _coverage_conf(track_m: float, mbes_m: float) -> float:
    # Smaller track/MBES distance implies better local coverage confidence.
    # Tuned to be conservative when proxies are far from the site.
    t = 0.0 if math.isnan(track_m) else sigmoid((2500.0 - track_m) / 600.0)
    m = 0.0 if math.isnan(mbes_m) else sigmoid((1500.0 - mbes_m) / 500.0)
    return 0.55 * t + 0.45 * m


def _detectability_prior(expected_len_m: float, depth_m: float) -> float:
    # Size-to-depth ratio: larger shallower targets are easier to detect.
    depth_m = max(depth_m, 1.0)
    ratio = expected_len_m / depth_m
    return sigmoid((ratio - 0.35) / 0.10)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multisource depth/proximity target ranker")
    ap.add_argument("--aero-json", required=True)
    ap.add_argument("--sat-json", required=True)
    ap.add_argument("--coverage-csv", required=True)
    ap.add_argument("--out-dir", default="multisource_rank_output")
    ap.add_argument("--assumed-depth-m", type=float, default=80.0)
    ap.add_argument("--expected-wreck-length-m", type=float, default=45.0)
    ap.add_argument("--top-n", type=int, default=200)
    args = ap.parse_args()

    aero = _to_sites(_load_candidates(Path(args.aero_json)))
    sat = _to_sites(_load_candidates(Path(args.sat_json)))
    cov = _load_coverage_points(Path(args.coverage_csv))

    keys = sorted(set(aero.keys()) | set(sat.keys()))
    rows: list[SiteScore] = []

    det_prior = _detectability_prior(args.expected_wreck_length_m, args.assumed_depth_m)
    for lat, lon in keys:
        a = float(aero.get((lat, lon), 0.0))
        s = float(sat.get((lat, lon), 0.0))

        # Fuse evidence with slight preference to aero for edge structure.
        fused = 0.6 * a + 0.4 * s

        nearest_km, track_m, mbes_m = _nearest_coverage(lat, lon, cov)
        cov_conf = _coverage_conf(track_m, mbes_m)

        # High priority to scan: anomaly present but coverage is weak.
        go_scan = fused * (1.0 - cov_conf) * (0.6 + 0.4 * det_prior)

        # Likely-not-there: strong coverage + low anomaly + decent detectability prior.
        not_there = cov_conf * (1.0 - min(fused, 1.0)) * det_prior

        note = ""
        if cov_conf > 0.75 and fused < 0.25:
            note = "high_coverage_low_anomaly"
        elif go_scan > 0.45:
            note = "candidate_for_outbound_scan"

        rows.append(
            SiteScore(
                lat=lat,
                lon=lon,
                aero_score=a,
                sat_score=s,
                fused_anomaly_score=fused,
                nearest_cov_point_km=nearest_km,
                track_distance_m_proxy=track_m,
                mbes_distance_m_proxy=mbes_m,
                coverage_confidence=cov_conf,
                assumed_depth_m=args.assumed_depth_m,
                expected_wreck_length_m=args.expected_wreck_length_m,
                detectability_prior=det_prior,
                go_scan_priority=go_scan,
                likely_not_there_confidence=not_there,
                note=note,
            )
        )

    rows.sort(key=lambda r: r.go_scan_priority, reverse=True)
    rows = rows[: args.top_n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_csv = out_dir / "ranked_targets.csv"
    go_csv = out_dir / "go_scan_targets.csv"
    not_csv = out_dir / "likely_not_there.csv"
    ranked_json = out_dir / "ranked_targets.json"

    fields = list(asdict(rows[0]).keys()) if rows else [
        "lat", "lon", "aero_score", "sat_score", "fused_anomaly_score",
        "nearest_cov_point_km", "track_distance_m_proxy", "mbes_distance_m_proxy",
        "coverage_confidence", "assumed_depth_m", "expected_wreck_length_m",
        "detectability_prior", "go_scan_priority", "likely_not_there_confidence", "note"
    ]

    def write_csv(path: Path, data: list[SiteScore]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow(asdict(r))

    write_csv(ranked_csv, rows)
    go_rows = [r for r in rows if r.go_scan_priority >= 0.25]
    go_rows.sort(key=lambda r: r.go_scan_priority, reverse=True)
    write_csv(go_csv, go_rows)

    not_rows = [r for r in rows if r.likely_not_there_confidence >= 0.55]
    not_rows.sort(key=lambda r: r.likely_not_there_confidence, reverse=True)
    write_csv(not_csv, not_rows)

    ranked_json.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    print(f"sites={len(rows)}")
    print(f"go_scan={len(go_rows)}")
    print(f"likely_not_there={len(not_rows)}")
    print(f"ranked_csv={ranked_csv}")
    print(f"go_scan_csv={go_csv}")
    print(f"not_there_csv={not_csv}")
    print(f"ranked_json={ranked_json}")


if __name__ == "__main__":
    main()
