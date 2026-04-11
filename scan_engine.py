#!/usr/bin/env python3
"""
CESAROPS Unified Scan Engine
=============================
One parameterized engine for all lakes, all dates, all detection passes.
No hardcoded lakes, dates, or paths.

Usage (as library):
    from scan_engine import ScanEngine
    engine = ScanEngine(bbox=[41.3,-83.5,42.5,-78.8], date_start="2015-10-01",
                        date_end="2015-10-31", output_dir="outputs/my_scan")
    report = engine.run()

Usage (CLI — see scan_cli.py for the human interface):
    engine = ScanEngine(...)
    engine.run()

Every detection dict has at minimum:
    lat, lon, zscore, type, source, scan_date, known_wreck_hit, pass_id
"""

import json
import math
import os
import re
import sys
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ── Import the processing functions from lake_michigan_scan (the real engine) ─
try:
    from lake_michigan_scan import (
        process_hydrocarbon_bands,
        process_tiff_with_coords,
        compute_nauticuvs_pass,
        compute_stumpf_pass,
        _flag_known_wreck,
        _is_linear_wake,
        KNOWN_WRECKS,
        KNOWN_WRECK_RADIUS_DEG,
        HAS_GPU,
    )
    _ENGINE_OK = True
except ImportError as e:
    print(f"[scan_engine] WARNING: lake_michigan_scan not importable: {e}")
    _ENGINE_OK = False
    HAS_GPU = False

try:
    from lake_erie_scan import (
        detect_swir_silt_erasure,
        detect_mussel_clearspot,
        build_hydrocarbon_timeline,
    )
    _ERIE_PASSES_OK = True
except ImportError:
    _ERIE_PASSES_OK = False

try:
    from triple_lock_fusion import (
        process_thermal_for_coldsink,
        process_sar_for_steel,
        process_optical_for_aluminum,
        fuse_triple_lock,
    )
    _TRIPLE_LOCK_OK = True
except ImportError:
    _TRIPLE_LOCK_OK = False

try:
    import simplekml
    _KML_OK = True
except ImportError:
    _KML_OK = False

try:
    import rasterio
    from rasterio.warp import transform as warp_transform
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False


# ═════════════════════════════════════════════════════════════════════════════
#  LAKE / REGION PRESETS  (convenience — everything can be overridden via bbox)
# ═════════════════════════════════════════════════════════════════════════════

LAKE_PRESETS = {
    "superior":  {"bbox": [46.5, -92.0, 48.0, -84.5],   "label": "Lake Superior"},
    "michigan":  {"bbox": [41.5, -88.0, 46.0, -85.5],   "label": "Lake Michigan"},
    "straits":   {"bbox": [45.65, -85.0, 46.10, -84.10], "label": "Straits of Mackinac"},
    "huron":     {"bbox": [42.5, -84.0, 46.0, -81.0],   "label": "Lake Huron"},
    "erie":      {"bbox": [41.3, -83.5, 42.5, -78.8],   "label": "Lake Erie"},
    "ontario":   {"bbox": [43.2, -79.5, 44.2, -76.0],   "label": "Lake Ontario"},
}

# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION PASS REGISTRY — tuneable thresholds
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PASS_CONFIG = {
    "standard_anomaly": {
        "enabled": True,
        "desc": "Z-score anomaly (optical, thermal, SAR bands)",
        "thermal_thresh": 2.0,
        "blue_thresh": 1.2,
        "default_thresh": 1.5,
        "cold_sink_thermal": True,
        "top_n": 200,
    },
    "hydrocarbon": {
        "enabled": True,
        "desc": "B11 SWIR dark + B04 Red bright = oil/fuel",
        "swir_thresh": -1.8,
        "red_thresh": 1.5,
    },
    "stumpf_bathy": {
        "enabled": True,
        "desc": "B02/B03 log-ratio bathymetric shallow anomaly",
        "top_n": 100,
    },
    "nauticuvs": {
        "enabled": True,
        "desc": "Multi-scale LoG blob detection (B02 + B10)",
        "top_n": 50,
    },
    "swir_silt_erasure": {
        "enabled": False,
        "desc": "B11/B12 ratio for sub-silt ferrous metal (Erie-specific)",
        "top_n": 30,
    },
    "mussel_clearspot": {
        "enabled": False,
        "desc": "B02 elevated oval in turbid background (Erie-specific)",
        "top_n": 30,
    },
    "triple_lock": {
        "enabled": False,
        "desc": "Multi-sensor fusion (thermal+SAR+optical must agree)",
        "tolerance_m": 50,
    },
}

# Band-tag filters for PASS 1 (skip non-imaging bands)
_SKIP_TAGS = {'FMASK', '.B11.', '.SWIR16.', '.SWIR22.',
              '.SCL.', '.QA_PIXEL.', '.NIR08.', '.NIR.'}


# ═════════════════════════════════════════════════════════════════════════════
#  DATE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_date_from_path(p: Path, fallback_year: int = None) -> Optional[date]:
    """Extract date from HLS filename, YYYYMMDD patterns, or directory hierarchy."""
    name = p.name
    # HLS: YYYYDOY pattern
    m = re.search(r'\.(\d{4})(\d{3})T\d{6}\.', name)
    if m:
        try:
            year, doy = int(m.group(1)), int(m.group(2))
            return date(year, 1, 1) + timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            pass
    # YYYYMMDD in filename
    m = re.search(r'(\d{4})(\d{2})(\d{2})', name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # Directory hierarchy: .../YYYY/MM/DD/ or .../YYYY/MM/
    parts = p.parts
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 4:
            yr = int(part)
            if 2000 <= yr <= 2030 and i + 1 < len(parts):
                try:
                    month = int(parts[i + 1])
                    day = int(parts[i + 2]) if i + 2 < len(parts) and parts[i + 2].isdigit() else 1
                    return date(yr, month, day)
                except (ValueError, IndexError):
                    pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  KMZ OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def create_scan_kmz(detections: list, output_path: Path, title: str = "CESAROPS Scan"):
    """Write a multi-folder KMZ from detection dicts."""
    if not _KML_OK or not detections:
        return
    kml = simplekml.Kml(name=title)

    # Group by type
    by_type = defaultdict(list)
    for d in detections:
        by_type[d.get("type", "unknown")].append(d)

    # Color map
    colors = {
        "hydrocarbon":       simplekml.Color.red,
        "thermal":           simplekml.Color.blue,
        "optical_blue":      simplekml.Color.cyan,
        "stumpf_shallow":    simplekml.Color.green,
        "nauticuvs_candidate": simplekml.Color.yellow,
        "swir_silt_erasure": simplekml.Color.orange,
        "mussel_clearspot":  simplekml.Color.purple,
        "triple_lock":       simplekml.Color.white,
    }

    for det_type, dets in sorted(by_type.items()):
        folder = kml.newfolder(name=f"{det_type} ({len(dets)})")
        color = colors.get(det_type, simplekml.Color.grey)
        for d in dets:
            pt = folder.newpoint(name=f"{det_type} z={d.get('zscore', 0):.1f}")
            pt.coords = [(d["lon"], d["lat"])]
            pt.style.iconstyle.color = color
            pt.style.iconstyle.scale = 0.6
            desc_lines = [
                f"Type: {det_type}",
                f"Z-score: {d.get('zscore', 'N/A')}",
                f"Date: {d.get('scan_date', '?')}",
                f"Source: {d.get('source', '?')}",
            ]
            if d.get("known_wreck_hit"):
                desc_lines.append(f"KNOWN WRECK: {d.get('known_wreck_name', d['known_wreck_hit'])}")
            if d.get("hc_subtype"):
                desc_lines.append(f"HC subtype: {d['hc_subtype']}")
            pt.description = "\n".join(desc_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kml.savekmz(str(output_path))
    print(f"  [KMZ] Saved: {output_path} ({len(detections)} points)")


# ═════════════════════════════════════════════════════════════════════════════
#  SCAN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ScanEngine:
    """
    Parameterized satellite scan engine.

    Parameters
    ----------
    bbox : list[float]
        [lat_min, lon_min, lat_max, lon_max] — area to scan.
    date_start : str
        Start date "YYYY-MM-DD".
    date_end : str
        End date "YYYY-MM-DD".
    output_dir : str or Path
        Where to write results (JSON, KMZ).
    data_dirs : list[str or Path], optional
        Where to find TIFF files.  Defaults to downloads/<lake>/ + downloads/hls/.
    passes : dict, optional
        Override DEFAULT_PASS_CONFIG (merge — missing keys use defaults).
    lake : str, optional
        Lake name for preset bbox + auto-enabling Erie-specific passes.
    label : str, optional
        Human label for the scan (used in output filenames).
    """

    def __init__(self, bbox=None, date_start=None, date_end=None,
                 output_dir=None, data_dirs=None, passes=None,
                 lake=None, label=None):

        # Resolve lake preset
        if lake and lake.lower() in LAKE_PRESETS:
            preset = LAKE_PRESETS[lake.lower()]
            if bbox is None:
                bbox = preset["bbox"]
            if label is None:
                label = preset["label"]

        if bbox is None:
            raise ValueError("bbox is required (or provide lake= name)")
        if date_start is None or date_end is None:
            raise ValueError("date_start and date_end are required")

        self.bbox = bbox
        self.date_start = date.fromisoformat(date_start)
        self.date_end = date.fromisoformat(date_end)
        self.label = label or f"scan_{date_start}_{date_end}"

        repo_root = Path(__file__).resolve().parent
        self.output_dir = Path(output_dir) if output_dir else (
            repo_root / "outputs" / self.label.lower().replace(" ", "_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "daily").mkdir(exist_ok=True)

        # Data search paths
        if data_dirs:
            self.data_dirs = [Path(d) for d in data_dirs]
        else:
            self.data_dirs = [
                repo_root / "downloads",
                Path(os.environ.get("CESAROPS_DATA_DIR", repo_root / "data")),
            ]

        # Pass config — start from defaults, merge user overrides
        self.passes = {}
        for k, v in DEFAULT_PASS_CONFIG.items():
            self.passes[k] = dict(v)
        if passes:
            for k, v in passes.items():
                if k in self.passes:
                    self.passes[k].update(v)
                else:
                    self.passes[k] = v

        # Auto-enable Erie-specific passes if lake is erie
        if lake and lake.lower() == "erie" and _ERIE_PASSES_OK:
            self.passes["swir_silt_erasure"]["enabled"] = True
            self.passes["mussel_clearspot"]["enabled"] = True

    # ── TIFF Discovery ──────────────────────────────────────────────────────

    def discover_tiffs(self) -> Dict[date, List[Path]]:
        """Find all TIFFs in data_dirs, filter by date range, group by date."""
        all_tiffs = []
        for d in self.data_dirs:
            if d.exists():
                all_tiffs.extend(d.rglob("*.tif"))
        all_tiffs = sorted(set(all_tiffs))

        by_date = defaultdict(list)
        for t in all_tiffs:
            dt = extract_date_from_path(t)
            if dt and self.date_start <= dt <= self.date_end:
                by_date[dt].append(t)

        return dict(sorted(by_date.items()))

    # ── Band Selection Helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_thermal(name: str) -> bool:
        n = name.upper()
        return 'B10' in n or 'THERMAL' in n or 'LWIR' in n or 'ST_B10' in n

    @staticmethod
    def _is_blue(name: str) -> bool:
        n = name.upper()
        return '.B02.' in n or '.BLUE.' in n

    @staticmethod
    def _is_b11(name: str) -> bool:
        n = name.upper()
        return ('.B11.' in n or '.SWIR16.' in n) and 'FMASK' not in n and '.SCL.' not in n

    @staticmethod
    def _is_b12(name: str) -> bool:
        n = name.upper()
        return ('.B12.' in n or '.SWIR22.' in n) and 'FMASK' not in n

    @staticmethod
    def _is_standard(name: str) -> bool:
        n = name.upper()
        return not any(tag in n for tag in _SKIP_TAGS)

    @staticmethod
    def _companion(tiff: Path, from_band: str, to_band: str) -> Path:
        """Derive companion band path (e.g. B11 -> B04, B02 -> B03)."""
        name = tiff.name
        pairs = [
            (f'.{from_band}.tif', f'.{to_band}.tif'),
            (f'.{from_band.lower()}.tif', f'.{to_band.lower()}.tif'),
        ]
        for old, new in pairs:
            if old.lower() in name.lower():
                return Path(str(tiff).replace(old, new))
        # HLS naming: case-sensitive band tag
        return Path(re.sub(rf'\.{from_band}\.', f'.{to_band}.', str(tiff), flags=re.IGNORECASE))

    # ── Pass Runners ────────────────────────────────────────────────────────

    def _run_standard_anomaly(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 1: Standard z-score anomaly on all imaging bands."""
        cfg = self.passes["standard_anomaly"]
        if not cfg["enabled"] or not _ENGINE_OK:
            return []

        standard = [t for t in tiffs if self._is_standard(t.name)]
        print(f"\n  PASS 1 -- Standard anomaly ({len(standard)} bands)")
        dets = []
        for tiff in standard:
            is_thermal = self._is_thermal(tiff.name)
            is_blue = self._is_blue(tiff.name)
            if is_thermal:
                thresh, cold = cfg["thermal_thresh"], cfg["cold_sink_thermal"]
            elif is_blue:
                thresh, cold = cfg["blue_thresh"], False
            else:
                thresh, cold = cfg["default_thresh"], False
            try:
                hits = process_tiff_with_coords(
                    tiff, threshold=thresh, scan_bbox=self.bbox,
                    top_n=cfg["top_n"], cold_sink_mode=cold)
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "standard_anomaly"
                dets.extend(hits)
            except Exception as e:
                print(f"    ERROR: {e}")
        return dets

    def _run_hydrocarbon(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 2: Hydrocarbon (B11 SWIR dark + B04 Red bright)."""
        cfg = self.passes["hydrocarbon"]
        if not cfg["enabled"] or not _ENGINE_OK:
            return []

        b11s = [t for t in tiffs if self._is_b11(t.name)]
        print(f"\n  PASS 2 -- Hydrocarbon ({len(b11s)} B11 scene(s))")
        dets = []
        for b11 in b11s:
            b04 = self._companion(b11, "B11", "B04")
            try:
                hits = process_hydrocarbon_bands(
                    b11, b04,
                    swir_thresh=cfg.get("swir_thresh", -1.8),
                    red_thresh=cfg.get("red_thresh", 1.5))
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "hydrocarbon"
                dets.extend(hits)
            except ImportError:
                print("    [HC] scipy not available")
            except Exception as e:
                print(f"    [HC] ERROR: {e}")
        return dets

    def _run_stumpf(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 3: Stumpf bathymetric B02/B03 log-ratio."""
        cfg = self.passes["stumpf_bathy"]
        if not cfg["enabled"] or not _ENGINE_OK:
            return []

        blues = [t for t in tiffs if self._is_blue(t.name)]
        print(f"\n  PASS 3 -- Stumpf bathymetric ({len(blues)} blue band(s))")
        dets = []
        for blue in blues:
            green = self._companion(blue, "B02", "B03")
            try:
                hits = compute_stumpf_pass(blue, green, scan_bbox=self.bbox,
                                           top_n=cfg.get("top_n", 100))
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "stumpf_bathy"
                dets.extend(hits)
            except Exception as e:
                print(f"    [ST] ERROR: {e}")
        return dets

    def _run_nauticuvs(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 4: NauticUVs LoG blob (B02 + B10)."""
        cfg = self.passes["nauticuvs"]
        if not cfg["enabled"] or not _ENGINE_OK:
            return []
        try:
            from scipy.ndimage import gaussian_laplace  # noqa
        except ImportError:
            print("  PASS 4 -- scipy not available, skipping NauticUVs")
            return []

        nuv_bands = ([t for t in tiffs if self._is_blue(t.name)] +
                     [t for t in tiffs if self._is_thermal(t.name)])
        print(f"\n  PASS 4 -- NauticUVs LoG blob ({len(nuv_bands)} band(s))")
        dets = []
        for tif in nuv_bands:
            try:
                hits = compute_nauticuvs_pass(tif, scan_bbox=self.bbox,
                                              top_n=cfg.get("top_n", 50))
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "nauticuvs"
                dets.extend(hits)
            except Exception as e:
                print(f"    [NUV] ERROR: {e}")
        return dets

    def _run_swir_silt_erasure(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 5: SWIR silt erasure (B11/B12 ratio for sub-silt metal)."""
        cfg = self.passes["swir_silt_erasure"]
        if not cfg["enabled"] or not _ERIE_PASSES_OK:
            return []

        b12s = [t for t in tiffs if self._is_b12(t.name)]
        print(f"\n  PASS 5 -- SWIR silt erasure ({len(b12s)} B12 band(s))")
        dets = []
        for b12 in b12s:
            b11 = self._companion(b12, "B12", "B11")
            try:
                hits = detect_swir_silt_erasure(b11, b12, scan_bbox=self.bbox,
                                                top_n=cfg.get("top_n", 30))
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "swir_silt_erasure"
                dets.extend(hits)
            except Exception as e:
                print(f"    [SWE] ERROR: {e}")
        return dets

    def _run_mussel_clearspot(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 6: Mussel clear-spot (elevated B02 in turbid background)."""
        cfg = self.passes["mussel_clearspot"]
        if not cfg["enabled"] or not _ERIE_PASSES_OK:
            return []

        blues = [t for t in tiffs if self._is_blue(t.name)]
        print(f"\n  PASS 6 -- Mussel clear-spot ({len(blues)} blue band(s))")
        dets = []
        for blue in blues:
            try:
                hits = detect_mussel_clearspot(blue, scan_bbox=self.bbox,
                                               top_n=cfg.get("top_n", 30))
                for h in hits:
                    h["scan_date"] = scan_date
                    h["pass_id"] = "mussel_clearspot"
                dets.extend(hits)
            except Exception as e:
                print(f"    [MCS] ERROR: {e}")
        return dets

    def _run_triple_lock(self, tiffs: List[Path], scan_date: str) -> list:
        """PASS 7: Triple-lock fusion (thermal + SAR + optical must agree)."""
        cfg = self.passes["triple_lock"]
        if not cfg["enabled"] or not _TRIPLE_LOCK_OK:
            return []

        thermal_tifs = [t for t in tiffs if self._is_thermal(t.name)]
        sar_tifs = [t for t in tiffs if any(tag in t.name.upper() for tag in ['_VV', '_VH', 'SAR'])]
        optical_tifs = [t for t in tiffs if self._is_standard(t.name) and not self._is_thermal(t.name)]

        print(f"\n  PASS 7 -- Triple-lock fusion (T:{len(thermal_tifs)} S:{len(sar_tifs)} O:{len(optical_tifs)})")
        thermal_dets, sar_dets, optical_dets = [], [], []
        for t in thermal_tifs:
            try:
                thermal_dets.extend(process_thermal_for_coldsink(t))
            except Exception as e:
                print(f"    [TL-T] ERROR: {e}")
        for s in sar_tifs:
            try:
                sar_dets.extend(process_sar_for_steel(s))
            except Exception as e:
                print(f"    [TL-S] ERROR: {e}")
        for o in optical_tifs[:5]:  # limit optical to avoid explosion
            try:
                optical_dets.extend(process_optical_for_aluminum(o))
            except Exception as e:
                print(f"    [TL-O] ERROR: {e}")

        fused = fuse_triple_lock(thermal_dets, sar_dets, optical_dets,
                                 tolerance_m=cfg.get("tolerance_m", 50))
        for f in fused:
            f["scan_date"] = scan_date
            f["pass_id"] = "triple_lock"
            f["type"] = "triple_lock"
        return fused

    # ── Main Scan Loop ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full scan. Returns a report dict."""
        print("=" * 80)
        print(f"CESAROPS UNIFIED SCAN: {self.label}")
        print("=" * 80)
        print(f"  BBOX:       {self.bbox}")
        print(f"  Dates:      {self.date_start} to {self.date_end}")
        print(f"  Output:     {self.output_dir}")
        print(f"  GPU:        {HAS_GPU}")
        enabled = [k for k, v in self.passes.items() if v.get("enabled")]
        print(f"  Passes:     {', '.join(enabled)}")
        print()

        if not _ENGINE_OK:
            print("FATAL: Processing engine not available (lake_michigan_scan.py)")
            return {"error": "engine_not_available"}

        # Discover data
        by_date = self.discover_tiffs()
        print(f"Found data for {len(by_date)} date(s) in range")
        if not by_date:
            print("\n  [WARNING] No TIFFs found in date range.")
            print("  Check data_dirs or run the downloader first.")
            print(f"  Searched: {[str(d) for d in self.data_dirs]}")
            return {"error": "no_data", "data_dirs": [str(d) for d in self.data_dirs]}

        all_detections = []
        daily_summaries = []

        for scan_date_obj, tiffs_today in sorted(by_date.items()):
            date_str = str(scan_date_obj)
            print(f"\n{'='*70}")
            print(f"  DATE: {date_str}  ({len(tiffs_today)} TIFF(s))")
            print(f"{'='*70}")

            day_dets = []

            # Run each enabled pass
            day_dets.extend(self._run_standard_anomaly(tiffs_today, date_str))
            day_dets.extend(self._run_hydrocarbon(tiffs_today, date_str))
            day_dets.extend(self._run_stumpf(tiffs_today, date_str))
            day_dets.extend(self._run_nauticuvs(tiffs_today, date_str))
            day_dets.extend(self._run_swir_silt_erasure(tiffs_today, date_str))
            day_dets.extend(self._run_mussel_clearspot(tiffs_today, date_str))
            day_dets.extend(self._run_triple_lock(tiffs_today, date_str))

            # Daily stats
            hc_count = sum(1 for d in day_dets if d.get("type") == "hydrocarbon")
            wreck_hits = sum(1 for d in day_dets if d.get("known_wreck_hit"))
            print(f"\n  -> {len(day_dets)} detections  "
                  f"({hc_count} hydrocarbon, {wreck_hits} known-wreck hits)")

            # Daily KMZ
            try:
                create_scan_kmz(day_dets,
                                self.output_dir / "daily" / f"{date_str}.kmz",
                                title=f"{self.label} {date_str}")
            except Exception as e:
                print(f"    [KMZ] Daily error: {e}")

            all_detections.extend(day_dets)
            daily_summaries.append({
                "date": date_str,
                "total": len(day_dets),
                "hydrocarbon": hc_count,
                "known_wreck_hits": wreck_hits,
                "tiff_count": len(tiffs_today),
                "passes_run": [k for k in self.passes if self.passes[k]["enabled"]],
            })

        # ── Combined outputs ────────────────────────────────────────────────

        # Combined KMZ
        try:
            create_scan_kmz(all_detections,
                            self.output_dir / "combined.kmz",
                            title=f"{self.label} Combined")
        except Exception as e:
            print(f"[KMZ] Combined error: {e}")

        # Hydrocarbon timeline
        timeline = build_hydrocarbon_timeline(all_detections) if _ERIE_PASSES_OK else \
            self._simple_hc_timeline(all_detections)
        timeline["daily_scan_summary"] = daily_summaries

        timeline_path = self.output_dir / "hydrocarbon_timeline.json"
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        # All detections JSON
        dets_path = self.output_dir / "all_detections.json"
        with open(dets_path, 'w', encoding='utf-8') as f:
            json.dump({
                "scan_label": self.label,
                "bbox": self.bbox,
                "date_range": [str(self.date_start), str(self.date_end)],
                "total_detections": len(all_detections),
                "passes": {k: v for k, v in self.passes.items() if v.get("enabled")},
                "detections": all_detections,
            }, f, indent=2, ensure_ascii=False, default=str)

        # Report
        report = {
            "label": self.label,
            "bbox": self.bbox,
            "date_range": [str(self.date_start), str(self.date_end)],
            "dates_scanned": len(by_date),
            "total_detections": len(all_detections),
            "hydrocarbon_total": sum(1 for d in all_detections if d.get("type") == "hydrocarbon"),
            "known_wreck_hits": sum(1 for d in all_detections if d.get("known_wreck_hit")),
            "output_dir": str(self.output_dir),
            "files": {
                "timeline": str(timeline_path),
                "detections": str(dets_path),
                "combined_kmz": str(self.output_dir / "combined.kmz"),
            },
            "hydrocarbon_timeline": timeline,
            "daily_summary": daily_summaries,
        }

        # Print summary
        print()
        print("=" * 80)
        print(f"SCAN COMPLETE: {self.label}")
        print("=" * 80)
        print(f"  Dates scanned:     {len(by_date)}")
        print(f"  Total detections:  {len(all_detections)}")
        print(f"  Hydrocarbon:       {report['hydrocarbon_total']}")
        print(f"  Known wreck hits:  {report['known_wreck_hits']}")
        print(f"  Output:            {self.output_dir}")
        if timeline.get("first_detection_date"):
            print(f"\n  HC LEAK ONSET:     {timeline['first_detection_date']}")
            sc = timeline.get("source_coords", {})
            print(f"  SOURCE COORDS:     lat={sc.get('lat', '?')}  lon={sc.get('lon', '?')}")
        print()

        return report

    @staticmethod
    def _simple_hc_timeline(detections: list) -> dict:
        """Fallback HC timeline when lake_erie_scan is unavailable."""
        by_date = defaultdict(list)
        for d in detections:
            if d.get("type") == "hydrocarbon" and d.get("scan_date"):
                by_date[d["scan_date"]].append(d)
        if not by_date:
            return {"first_detection_date": None, "source_coords": None, "daily_counts": {}}
        sorted_dates = sorted(by_date.keys())
        first = by_date[sorted_dates[0]]
        lats = [d["lat"] for d in first]
        lons = [d["lon"] for d in first]
        ws = [abs(d.get("zscore", 1.0)) for d in first]
        tw = sum(ws) or 1.0
        return {
            "first_detection_date": sorted_dates[0],
            "first_detection_count": len(first),
            "source_coords": {"lat": round(sum(la*w for la, w in zip(lats, ws))/tw, 5),
                              "lon": round(sum(lo*w for lo, w in zip(lons, ws))/tw, 5)},
            "total_hc_detections": sum(len(v) for v in by_date.values()),
            "daily_counts": {d: len(v) for d, v in sorted(by_date.items())},
            "all_dates_with_hc": sorted_dates,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE ENTRY (for quick testing)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("scan_engine.py is a library. Use scan_cli.py for manual scans,")
    print("or agent_scan_tools.py for agent-driven scans.")
    print()
    print("Available lakes:", ", ".join(LAKE_PRESETS.keys()))
    print("Available passes:", ", ".join(DEFAULT_PASS_CONFIG.keys()))
