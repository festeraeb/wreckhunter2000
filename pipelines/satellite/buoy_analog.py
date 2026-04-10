"""NDBC Buoy Data Fetcher and Analog Storm Extractor.

Downloads standard meteorological data from NOAA NDBC for the three key
Lake Erie buoys used to characterize modern analog storms:

  45132  Port Stanley (Central-East basin, Canadian side)
  45142  Central Lake Erie (deepest water, best mid-lake signal)
  45005  Western Erie Basin (seiche / drawdown indicator)

Known analog storms used for 1909 M&B No.2 model calibration:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Elliott     Dec 23-24 2022  Bomb cyclone — SW→NW wind shift,
                              15-22 ft waves, 7 ft seiche, best match
  Jan2024     Jan 13-14 2024  65+ mph SW, lake-bed exposed Put-in-Bay
  Sandy       Oct 30-31 2012  NE winds (counter-example, debris goes SW)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
  python scripts/buoy_analog.py --storm elliott --fetch
  python scripts/buoy_analog.py --storm all --fetch --compare
  python scripts/buoy_analog.py --list-cached
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import sys
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Force UTF-8 output so Unicode arrows/symbols don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Import StormPhase from engine ────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from historical_drift import StormPhase  # noqa: E402

CACHE_DIR = REPO / "magnetic_data" / "buoy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Buoy definitions ─────────────────────────────────────────────────────────
# Source column: "ndbc" = NOAA NDBC (confirmed working)
#                "eccc" = Environment Canada MEDS (requires separate fetch)
BUOYS = {
    "45005": {"name": "Western Erie Basin (seiche/drawdown)",
              "lat": 41.67, "lon": -82.40, "source": "ndbc"},
    "45132": {"name": "Port Stanley (Central-East Erie, Canadian)",
              "lat": 42.49, "lon": -81.72, "source": "eccc",
              "eccc_note": "Canadian ECCC buoy. Historical data via MSC/MEDS portal."},
    "45142": {"name": "Central Lake Erie (Canadian)",
              "lat": 42.16, "lon": -81.28, "source": "eccc",
              "eccc_note": "Canadian ECCC buoy. Not available on NDBC."},
}

# Primary NDBC buoy for all analogs (45005 is the reliably archived one)
NDBC_PRIMARY = "45005"

# ── NDBC URL patterns ────────────────────────────────────────────────────────
# Standard met: annual .txt.gz  (historical, years ≤ last full year)
_NDBC_HIST = "https://www.ndbc.noaa.gov/data/historical/stdmet/{buoy}h{year}.txt.gz"
# Continuous wind (cwind) — 10-min data, available historically as well
_NDBC_CWIND = "https://www.ndbc.noaa.gov/data/historical/cwind/{buoy}c{year}.txt.gz"
# Real-time (last 45 days)
_NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2/{buoy}.txt"

# NOAA CO-OPS water level — Great Lakes use IGLD datum (no tidal datum)
_COOPS_WL = (
    "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    "?begin_date={start}&end_date={end}&station={station}"
    "&product=hourly_height&datum=IGLD&time_zone=gmt&units=english&format=json"
)
COOPS_STATIONS = {
    "toledo":  "9063053",
    "buffalo": "9014070",
}

# ── Pre-defined analog storm windows ─────────────────────────────────────────
ANALOG_STORMS = {
    "elliott": {
        "label": "Elliott (Bomb Cyclone) Dec 23-24 2022",
        "best_match_reason": (
            "Near-perfect 1909 pattern match. SW winds sustained at 50+ mph, "
            "then violent shift to W/NW (hurricane force). 15-22 ft waves central "
            "basin. 7 ft seiche Toledo drop / 11 ft Buffalo surge = confirms NW "
            "surface-current push toward PA/NY shore identical to 1909 debris field."
        ),
        "similarity_score": 0.93,
        "buoys": ["45005"],           # NDBC-available; 45132/45142 are ECCC
        "eccc_buoys": ["45132", "45142"],
        "windows": [(2022, 12, 23, 24)],
        "target_buoy": "45005",
        "seiche_notes": "Toledo -7.0 ft  /  Buffalo +10.8 ft  (NOAA CO-OPS)",
        "coops_window": {"start": "20221221", "end": "20221226"},  # 6-day window captures full seiche
        "buoy_note": "45005 pulled ~Dec 6 2022 (winter lay-up). Use historical wind reports + CO-OPS seiche as primary data.",
    },
    "jan2024": {
        "label": "January 13-14 2024 Winter Storm",
        "best_match_reason": (
            "65+ mph sustained SW winds. Lake-bed exposed Put-in-Bay western basin "
            "(matching Dec 1909 'low water' reports). Significant seiche. "
            "Good second-reference storm for SW-only forcing calibration."
        ),
        "similarity_score": 0.78,
        "buoys": ["45005"],
        "eccc_buoys": ["45132", "45142"],
        "windows": [(2024, 1, 13, 14)],
        "target_buoy": "45005",
        "seiche_notes": "Put-in-Bay lake-bed exposed; Buffalo surge reported ~6 ft",
        "coops_window": {"start": "20240111", "end": "20240116"},
        "buoy_note": "Check if 45005 deployed in Jan 2024; if not, use CO-OPS + wind reports.",
    },
    "sandy": {
        "label": "Superstorm Sandy Oct 30-31 2012",
        "best_match_reason": (
            "NE/N winds (counter-example). Debris drifted SOUTHWEST — opposite of 1909. "
            "Use to validate backward-drift direction-sensitivity: if Sandy conditions "
            "were used, candidate should be SOUTHWEST of Conneaut, NOT our NNW candidate. "
            "This CONFIRMS 1909 forcing was NW not NE."
        ),
        "similarity_score": -0.40,
        "buoys": ["45005"],
        "eccc_buoys": ["45132", "45142"],
        "windows": [(2012, 10, 30, 31)],
        "target_buoy": "45005",
        "seiche_notes": "Reverse seiche — Buffalo drop / Toledo rise",
        "coops_window": {"start": "20121028", "end": "20121102"},
        "buoy_note": "Sandy: 45005 likely still deployed in Oct 2012.",
    },
}

# ── NDBC data parser ──────────────────────────────────────────────────────────
# Standard met columns (2007+ format):
# YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE
_FILL = {999, 9999, 99.0, 999.0, 9999.0}

def _is_fill(v):
    try:
        f = float(v)
        return f in _FILL or f >= 999
    except Exception:
        return True


def parse_ndbc_stdmet(raw_text: str, year: int, month: int,
                      day_start: int, day_end: int) -> List[dict]:
    """Parse NDBC standard meteorological text into a list of hourly records
    for the specified UTC day range (inclusive)."""
    rows = []
    lines = raw_text.splitlines()
    header = None
    for line in lines:
        if line.startswith("#YY") or line.startswith("YY"):
            header = line.lstrip("#").split()
            continue
        if line.startswith("#"):
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            yy = int(parts[0])
            mm = int(parts[1])
            dd = int(parts[2])
            hh = int(parts[3])
            # Accept both 2-digit (19xx) and 4-digit years
            if yy < 100:
                yy += 1900 if yy > 50 else 2000
        except Exception:
            continue
        if yy != year or mm != month:
            continue
        if not (day_start <= dd <= day_end):
            continue

        rec = {"timestamp": f"{yy:04d}-{mm:02d}-{dd:02d}T{hh:02d}:00:00Z"}
        for col, val in zip(header[5:], parts[5:]):
            if not _is_fill(val):
                rec[col] = float(val)
        rows.append(rec)
    return rows


def _fetch_ndbc_gz(url: str) -> Optional[str]:
    """Fetch a .txt.gz from NDBC, return decompressed text (or None on error)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BagrecoveryResearch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return gzip.decompress(raw).decode("ascii", errors="replace")
    except Exception as e:
        return None


def _fetch_ndbc_plain(url: str) -> Optional[str]:
    """Fetch a plain-text NDBC file (real-time)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BagrecoveryResearch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("ascii", errors="replace")
    except Exception as e:
        return None


def fetch_buoy_storm(buoy_id: str, year: int, month: int,
                     day_start: int, day_end: int,
                     force: bool = False) -> List[dict]:
    """Fetch and cache NDBC hourly records for a buoy/date window."""
    key = f"{buoy_id}_{year}{month:02d}{day_start:02d}-{day_end:02d}.json"
    cache_path = CACHE_DIR / key
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    # Try historical gz
    url = _NDBC_HIST.format(buoy=buoy_id.lower(), year=year)
    text = _fetch_ndbc_gz(url)

    if text is None:
        # Fallback to current-year real-time if fetching recent data
        url_rt = _NDBC_REALTIME.format(buoy=buoy_id.upper())
        text = _fetch_ndbc_plain(url_rt)

    if text is None:
        print(f"  WARNING: Could not fetch buoy {buoy_id} for {year}")
        return []

    records = parse_ndbc_stdmet(text, year, month, day_start, day_end)
    cache_path.write_text(json.dumps(records, indent=2))
    print(f"  Cached {len(records)} records → {cache_path.name}")
    return records


def fetch_coops_water_level(station: str, start_yyyymmdd: str,
                             end_yyyymmdd: str, force: bool = False) -> List[dict]:
    """Fetch NOAA CO-OPS 6-minute water level for seiche analysis."""
    key = f"coops_{station}_{start_yyyymmdd}_{end_yyyymmdd}.json"
    cache_path = CACHE_DIR / key
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    url = _COOPS_WL.format(
        start=start_yyyymmdd, end=end_yyyymmdd, station=station
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BagrecoveryResearch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        if "data" not in data:
            print(f"  CO-OPS no data for {station}: {data.get('error', '')}")
            return []
        records = [{"t": r["t"], "v": float(r["v"])} for r in data["data"]]
        cache_path.write_text(json.dumps(records, indent=2))
        print(f"  CO-OPS {station}: {len(records)} records cached")
        return records
    except Exception as e:
        print(f"  CO-OPS fetch error {station}: {e}")
        return []


# ── Convert buoy records → StormPhase objects ─────────────────────────────────

MS_TO_MPH = 2.23694
KT_TO_MPH  = 1.15078
M_TO_FT    = 3.28084


def records_to_phases(records: List[dict], storm_label: str,
                       phase_width_hours: int = 3) -> List[StormPhase]:
    """Bin buoy records into consecutive StormPhase objects.

    Creates one phase per `phase_width_hours` block, computing median
    wind speed / direction and peak wave height for each block.
    Direction averaging uses circular statistics to handle 0/360 wrap.
    """
    if not records:
        return []

    # Sort by timestamp
    records = sorted(records, key=lambda r: r["timestamp"])

    def _parse_ts(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    t0 = _parse_ts(records[0]["timestamp"])
    phases = []
    block: List[dict] = []

    def _flush(blk: List[dict], idx: int) -> Optional[StormPhase]:
        if not blk:
            return None
        wspds = [r["WSPD"] * MS_TO_MPH for r in blk if "WSPD" in r]
        wdirs = [r["WDIR"] for r in blk if "WDIR" in r]
        wvhts = [r["WVHT"] * M_TO_FT for r in blk if "WVHT" in r]
        gsts  = [r["GST"] * MS_TO_MPH for r in blk if "GST" in r]

        if not wspds:
            return None

        ws = sorted(wspds)[len(wspds)//2]   # median
        wg = max(gsts) if gsts else ws * 1.3
        wh = max(wvhts) if wvhts else 0.0

        # Circular mean for wind direction
        if wdirs:
            sins = [math.sin(math.radians(d)) for d in wdirs]
            coss = [math.cos(math.radians(d)) for d in wdirs]
            wd = (math.degrees(math.atan2(sum(sins)/len(sins),
                                          sum(coss)/len(coss))) + 360) % 360
        else:
            wd = 270.0

        ts_start = blk[0]["timestamp"][:10]
        ts_end   = blk[-1]["timestamp"][11:16]
        label = f"{storm_label} ph{idx+1} [{ts_start} {ts_end}]"

        return StormPhase(
            label=label,
            wind_speed_mph=round(ws, 1),
            wind_dir_deg=round(wd, 1),
            dt_hours=float(phase_width_hours),
            wave_ht_ft=round(wh, 1),
            current_speed_kt=0.0,  # buoy doesn't have currents; set via ADCP later
            current_dir_deg=0.0,
        )

    for rec in records:
        t = _parse_ts(rec["timestamp"])
        elapsed_h = (t - t0).total_seconds() / 3600
        expected_block = int(elapsed_h // phase_width_hours)
        current_block  = len(phases)
        if expected_block > current_block:
            ph = _flush(block, current_block)
            if ph:
                phases.append(ph)
            block = []
        block.append(rec)
    if block:
        ph = _flush(block, len(phases))
        if ph:
            phases.append(ph)

    return phases


# ── Analog similarity scorer ──────────────────────────────────────────────────

def score_analog_similarity(analog_phases: List[StormPhase],
                              target_phases: List[StormPhase]) -> dict:
    """Score how well an analog storm matches the 1909 target pattern.

    Checks:
      - Peak wind speed match (±30%)
      - Wind direction shift from SW/W to NW/N (critical pattern)
      - Storm duration match
      - Wave height match
    """
    if not analog_phases or not target_phases:
        return {"score": 0.0, "details": "No data"}

    # Peak wind score
    a_peak = max(p.wind_speed_mph for p in analog_phases)
    t_peak = max(p.wind_speed_mph for p in target_phases)
    speed_ratio = min(a_peak, t_peak) / max(a_peak, t_peak)

    # Direction shift: does the analog go from SW/W to NW?
    a_dirs = [p.wind_dir_deg for p in analog_phases]
    first_dir, peak_dir = a_dirs[0], a_dirs[a_dirs.index(max(
        a_dirs, key=lambda d: analog_phases[a_dirs.index(d)].wind_speed_mph))]
    # SW/W = 200-280; NW = 280-340
    starts_sw = 200 <= first_dir <= 310
    shifts_nw = any(280 <= d <= 340 for d in a_dirs)
    direction_match = 1.0 if (starts_sw and shifts_nw) else (0.5 if shifts_nw else 0.1)

    # Wave height score
    a_wave = max(p.wave_ht_ft for p in analog_phases)
    t_wave = max(p.wave_ht_ft for p in target_phases)
    wave_ratio = min(a_wave, t_wave) / max(a_wave, t_wave) if max(a_wave, t_wave) > 0 else 0.5

    composite = 0.4 * speed_ratio + 0.4 * direction_match + 0.2 * wave_ratio

    return {
        "score": round(composite, 3),
        "peak_wind_mph": round(a_peak, 1),
        "peak_wave_ft": round(a_wave, 1),
        "starts_sw": starts_sw,
        "shifts_nw": shifts_nw,
        "speed_ratio": round(speed_ratio, 3),
        "direction_match": round(direction_match, 3),
        "wave_ratio": round(wave_ratio, 3),
    }


# ── Seiche analysis ───────────────────────────────────────────────────────────

def analyze_seiche(toledo_records: List[dict],
                   buffalo_records: List[dict]) -> dict:
    """Compute seiche amplitude and timing from CO-OPS water level data."""
    if not toledo_records or not buffalo_records:
        return {"error": "No water level data"}

    t_vals = [r["v"] for r in toledo_records]
    b_vals = [r["v"] for r in buffalo_records]

    t_min, t_max = min(t_vals), max(t_vals)
    b_min, b_max = min(b_vals), max(b_vals)

    toledo_drop = t_max - t_min    # range during storm
    buffalo_surge = b_max - b_min

    # Timing of extremes
    t_min_time = toledo_records[t_vals.index(t_min)]["t"]
    b_max_time = buffalo_records[b_vals.index(b_max)]["t"]

    return {
        "toledo_drop_ft": round(toledo_drop, 2),
        "buffalo_surge_ft": round(buffalo_surge, 2),
        "toledo_min_time_utc": t_min_time,
        "buffalo_max_time_utc": b_max_time,
        "implied_current_direction": "E→W surface flow" if toledo_drop > buffalo_surge else "W→E surface flow",
        "seiche_consistent_1909": toledo_drop >= 4.0,
    }


# ── Master fetch + analysis pipeline ─────────────────────────────────────────

def run_analog(storm_key: str, force: bool = False) -> dict:
    """Fetch buoy data, build storm phases, score similarity, run seiche."""
    storm = ANALOG_STORMS.get(storm_key)
    if not storm:
        raise ValueError(f"Unknown storm: {storm_key}. Options: {list(ANALOG_STORMS)}")

    print(f"\n{'='*60}")
    print(f"  {storm['label']}")
    print(f"  {storm['best_match_reason'][:80]}...")
    print(f"{'='*60}")

    all_phases = {}
    for buoy_id in storm["buoys"]:
        print(f"\n  Fetching buoy {buoy_id} ({BUOYS[buoy_id]['name']})...")
        all_records = []
        for year, month, d0, d1 in storm["windows"]:
            recs = fetch_buoy_storm(buoy_id, year, month, d0, d1, force=force)
            all_records.extend(recs)
        all_phases[buoy_id] = records_to_phases(
            all_records, f"{storm_key}_{buoy_id}"
        )
        print(f"    → {len(all_phases[buoy_id])} phases built")

    # Seiche analysis
    print(f"\n  Fetching CO-OPS water levels...")
    cw = storm.get("coops_window", {})
    toledo_wl  = fetch_coops_water_level(
        COOPS_STATIONS["toledo"], cw["start"], cw["end"], force=force)
    buffalo_wl = fetch_coops_water_level(
        COOPS_STATIONS["buffalo"], cw["start"], cw["end"], force=force)
    seiche = analyze_seiche(toledo_wl, buffalo_wl)

    # Score against 1909 target phases
    from historical_drift import MB2_CASE
    target_phases = MB2_CASE["storm_phases"][:3]   # exclude 49-day debris phase
    primary_phases = all_phases.get(storm["target_buoy"], [])
    similarity = score_analog_similarity(primary_phases, target_phases)

    result = {
        "storm": storm_key,
        "label": storm["label"],
        "known_similarity": storm["similarity_score"],
        "computed_similarity": similarity,
        "seiche": seiche,
        "phases_by_buoy": {k: [asdict(p) for p in v] for k, v in all_phases.items()},
        "primary_phases": [asdict(p) for p in primary_phases],
    }

    out_path = CACHE_DIR / f"{storm_key}_analysis.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Similarity score: {similarity['score']:.3f}")
    print(f"  Peak wind: {similarity.get('peak_wind_mph', '?')} mph")
    print(f"  Peak wave: {similarity.get('peak_wave_ft', '?')} ft")
    print(f"  Starts SW: {similarity.get('starts_sw')}  Shifts NW: {similarity.get('shifts_nw')}")
    print(f"  Seiche — Toledo drop: {seiche.get('toledo_drop_ft', '?')} ft  "
          f"Buffalo surge: {seiche.get('buffalo_surge_ft', '?')} ft")
    print(f"  Analysis saved: {out_path.name}")
    return result


def load_cached_phases(storm_key: str,
                        buoy_id: str = None) -> List[StormPhase]:
    """Load previously fetched analog phases from cache as StormPhase objects."""
    key = storm_key if storm_key in ANALOG_STORMS else None
    if not key:
        return []
    storm = ANALOG_STORMS[key]
    target_buoy = buoy_id or storm["target_buoy"]
    cache = CACHE_DIR / f"{storm_key}_analysis.json"
    if not cache.exists():
        return []
    data = json.loads(cache.read_text())
    raw = data.get("phases_by_buoy", {}).get(target_buoy, [])
    return [StormPhase(**p) for p in raw]


def list_cached() -> List[str]:
    return [p.stem for p in CACHE_DIR.glob("*_analysis.json")]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NDBC buoy analog storm fetcher")
    ap.add_argument("--storm", default="elliott",
                    help=f"Storm key(s), comma-sep or 'all'. Options: {list(ANALOG_STORMS)}")
    ap.add_argument("--fetch", action="store_true", help="Download from NDBC (else use cache)")
    ap.add_argument("--compare", action="store_true", help="Print side-by-side comparison")
    ap.add_argument("--list-cached", action="store_true")
    args = ap.parse_args()

    if args.list_cached:
        print("Cached analyses:", list_cached())
        sys.exit(0)

    keys = list(ANALOG_STORMS) if args.storm == "all" else [k.strip() for k in args.storm.split(",")]
    results = {}
    for k in keys:
        results[k] = run_analog(k, force=args.fetch)

    if args.compare:
        print(f"\n{'─'*70}")
        print("  ANALOG COMPARISON vs 1909 M&B No.2 Storm")
        print(f"{'─'*70}")
        print(f"  {'Storm':<30}  {'Score':>6}  {'Peak wind':>10}  {'Peak wave':>10}  SW→NW?")
        for k, r in results.items():
            s = r["computed_similarity"]
            print(f"  {r['label'][:30]:<30}  {s['score']:>6.3f}  "
                  f"{s.get('peak_wind_mph', '?'):>10}  "
                  f"{s.get('peak_wave_ft', '?'):>10}  "
                  f"{'✅' if s.get('starts_sw') and s.get('shifts_nw') else '❌'}")
