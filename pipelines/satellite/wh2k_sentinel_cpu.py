"""
WreckHunter 2000 — Sentinel CPU Background Scanner
====================================================
Runs alongside GPU Phase 2 training on the CPU.

Sentinel duties:
  1. Check every --poll-interval seconds if best_resnet18.pt has improved
     (compares val_acc stored in checkpoint vs last known val_acc).
  2. On improvement (or on first run), kick off a full grid scan using CPU.
  3. Produce Sentinel Report JSON + KML of all Score 8-10 unknowns.
  4. Track candidates across sweeps — flag any newcomer not seen before.
  5. Write incremental delta report so you can watch discovery in real-time.

Why CPU? The M2200 VRAM is fully occupied by batch training. Grid scan
is not latency-critical — CPU inference on 224×224 tiles is ~2ms each.

Usage (background, alongside GPU training):
  cd C:\\Users\\thomf\\programming\\Bagrecovery
  Start-Process `
    -FilePath "C:\\Users\\thomf\\miniconda3\\envs\\wh2k\\python.exe" `
    -ArgumentList "scripts\\wh2k_sentinel_cpu.py",
      "--checkpoint", "wreck_hunting_ml\\models\\best_resnet18.pt",
      "--grid-tif", "magnetic_data\\grids\\usgs_namag_83_6000_41_3000__78_8000_42_9000.tif",
      "--output-dir", "wreck_hunting_ml\\sentinel",
      "--poll-interval", "300" `
    -RedirectStandardOutput "wreck_hunting_ml\\sentinel_stdout.log" `
    -RedirectStandardError  "wreck_hunting_ml\\sentinel_stderr.log" `
    -NoNewWindow

One-shot (no poll loop):
  ... --sweeps 1
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# ── Score threshold for "interesting" targets in reports ──────────────────
MIN_SCORE = 8
SCAN_CONFIDENCE_THRESHOLD = 0.45


# ── KML Helpers ───────────────────────────────────────────────────────────

def _kml_color(score: int) -> str:
    """AABBGGRR KML color by score."""
    if score == 10:
        return "ff0000ff"   # red
    elif score >= 8:
        return "ff0088ff"   # orange
    else:
        return "ff00ffff"   # yellow


def write_kml(targets: list[dict], out_path: Path, title: str = "Sentinel Scan") -> None:
    """Write a simple KML placemark file for the target list."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>{title}</name>",
        "<Style id='s10'><IconStyle><color>ff0000ff</color><scale>1.4</scale>"
        "<Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>"
        "</IconStyle></Style>",
        "<Style id='s8'><IconStyle><color>ff0088ff</color><scale>1.2</scale>"
        "<Icon><href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon>"
        "</IconStyle></Style>",
    ]

    for t in targets:
        score = t.get("wreck_score", 0)
        style = "s10" if score == 10 else "s8"
        name = t.get("name", f"Target-{t.get('rank','?')}")
        desc = (
            f"Score: {score}/10 | Conf: {t.get('confidence', 0):.2f} | "
            f"Extent: {t.get('extent_m', 0):.0f}m | AR: {t.get('aspect_ratio', 0):.1f} | "
            f"Amp: {t.get('amplitude_nt', 0):.1f}nT"
        )
        lines += [
            "<Placemark>",
            f"<name>{name}</name>",
            f"<description>{desc}</description>",
            f"<styleUrl>#{style}</styleUrl>",
            "<Point>",
            f"<coordinates>{t['lon']},{t['lat']},0</coordinates>",
            "</Point>",
            "</Placemark>",
        ]

    lines += ["</Document>", "</kml>"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("KML written → %s (%d placemarks)", out_path, len(targets))


# ── Sentinel State ────────────────────────────────────────────────────────

class SentinelState:
    """Persists last-known checkpoint accuracy and candidate history."""

    def __init__(self, state_path: Path):
        self.path = state_path
        self.last_val_acc: float = -1.0
        self.last_epoch: int = -1
        self.sweep_count: int = 0
        self.all_candidates: dict[str, dict] = {}   # key = "lat,lon" rounded to 4 dp
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                s = json.loads(self.path.read_text())
                self.last_val_acc  = s.get("last_val_acc", -1.0)
                self.last_epoch    = s.get("last_epoch", -1)
                self.sweep_count   = s.get("sweep_count", 0)
                self.all_candidates = s.get("all_candidates", {})
                logger.info("Sentinel state loaded (sweep %d, last_acc=%.4f)",
                            self.sweep_count, self.last_val_acc)
            except Exception as e:
                logger.warning("State load failed: %s — starting fresh", e)

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "last_val_acc":   self.last_val_acc,
            "last_epoch":     self.last_epoch,
            "sweep_count":    self.sweep_count,
            "all_candidates": self.all_candidates,
        }, indent=2), encoding="utf-8")

    def is_model_improved(self, checkpoint_path: Path) -> tuple[bool, float, int]:
        """Returns (improved, new_val_acc, new_epoch)."""
        try:
            import torch
            ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            acc = float(ck.get("val_acc", 0.0))
            epoch = int(ck.get("epoch", 0))
            improved = (acc > self.last_val_acc + 1e-4) or (self.last_epoch < 0)
            return improved, acc, epoch
        except Exception as e:
            logger.warning("Could not read checkpoint: %s", e)
            return False, self.last_val_acc, self.last_epoch

    def register_candidates(self, new_targets: list[dict]) -> list[dict]:
        """Add new targets to history. Returns only the NEW ones (not previously seen)."""
        fresh = []
        for t in new_targets:
            key = f"{t['lat']:.4f},{t['lon']:.4f}"
            if key not in self.all_candidates:
                t["first_seen_sweep"] = self.sweep_count
                fresh.append(t)
            self.all_candidates[key] = {**t, "last_seen_sweep": self.sweep_count}
        return fresh


# ── Grid Scan (CPU) ───────────────────────────────────────────────────────

def run_cpu_scan(
    checkpoint_path: Path,
    grid_tif_path: Path,
    min_score: int = MIN_SCORE,
    confidence_threshold: float = SCAN_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Full grid scan using CPU inference.
    Returns list of target dicts with score >= min_score.
    """
    try:
        from scripts.wh2k_inference_scorer import scan_grid, correlate_and_subtract, score_unknowns
    except ImportError:
        logger.error("Cannot import wh2k_inference_scorer — is it in scripts/?")
        return []

    logger.info("Scanning grid: %s (CPU, conf>=%.2f)", grid_tif_path, confidence_threshold)
    t0 = time.time()

    detections = scan_grid(
        model_path=checkpoint_path,
        grid_tif_path=grid_tif_path,
        confidence_threshold=confidence_threshold,
        device="cpu",
    )
    logger.info("Scan complete: %d raw detections in %.1fs", len(detections), time.time() - t0)

    unknowns, known_count = correlate_and_subtract(detections)
    if isinstance(known_count, (list, tuple, set, dict)):
        known_removed = len(known_count)
    else:
        try:
            known_removed = int(known_count)
        except Exception:
            known_removed = 0
    logger.info("After subtract: %d unknowns (removed %d known)", len(unknowns), known_removed)

    scored = score_unknowns(unknowns)
    targets = [d for d in scored if d.wreck_score >= min_score]
    logger.info("Targets score>=%d: %d", min_score, len(targets))

    # Convert Detection objects to plain dicts
    return [
        {
            "lat":          d.lat,
            "lon":          d.lon,
            "wreck_score":  d.wreck_score,
            "confidence":   d.confidence,
            "class_name":   d.class_name,
            "extent_m":     d.spatial_extent_m,
            "aspect_ratio": d.aspect_ratio,
            "amplitude_nt": d.peak_amplitude_nt,
            "score_reasons": d.score_reasons,
            "axis_offset_deg": d.axis_offset_from_geology_deg,
        }
        for d in targets
    ]


# ── Sentinel Main Loop ────────────────────────────────────────────────────

def run_sentinel(
    checkpoint_path: Path,
    grid_tif_path: Path,
    output_dir: Path,
    poll_interval: float = 300.0,
    max_sweeps: int = 0,           # 0 = run forever
    force_first_sweep: bool = True,
    min_score: int = MIN_SCORE,
) -> None:
    """
    Main sentinel loop.  Polls checkpoint for improvements, scans on change.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state = SentinelState(output_dir / "sentinel_state.json")
    sweep = 0

    logger.info("Sentinel started — polling %s every %.0fs",
                checkpoint_path.name, poll_interval)

    while True:
        improved, new_acc, new_epoch = state.is_model_improved(checkpoint_path)

        if improved or (force_first_sweep and state.sweep_count == 0):
            sweep += 1
            state.sweep_count = sweep
            state.last_val_acc = new_acc
            state.last_epoch   = new_epoch

            logger.info(
                "━━ Sweep #%d ━━  model epoch=%d val_acc=%.4f",
                sweep, new_epoch, new_acc,
            )

            targets = run_cpu_scan(checkpoint_path, grid_tif_path, min_score=min_score)
            fresh   = state.register_candidates(targets)

            # Rank by score then confidence
            targets.sort(key=lambda t: (t["wreck_score"], t["confidence"]), reverse=True)
            for i, t in enumerate(targets, 1):
                t["rank"] = i
                t["name"] = f"Sentinel-{sweep}-#{i}"
                is_new = any(
                    abs(t["lat"] - f["lat"]) < 0.001 and abs(t["lon"] - f["lon"]) < 0.001
                    for f in fresh
                )
                t["new_this_sweep"] = is_new

            # Write sweep report
            report = {
                "sweep":           sweep,
                "model_epoch":     new_epoch,
                "model_val_acc":   round(new_acc, 4),
                "total_targets":   len(targets),
                "new_targets":     len(fresh),
                "min_score":       min_score,
                "targets":         targets,
            }
            report_path = output_dir / f"sentinel_sweep_{sweep:03d}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Report → %s", report_path)

            # Write KML
            kml_path = output_dir / f"sentinel_sweep_{sweep:03d}.kml"
            write_kml(targets, kml_path, title=f"Sentinel Sweep {sweep} (epoch {new_epoch})")

            # Write running "latest" files (overwritten each sweep)
            (output_dir / "sentinel_latest.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            write_kml(targets, output_dir / "sentinel_latest.kml",
                      title=f"Sentinel Latest (sweep {sweep})")

            # Delta report — NEW candidates only
            if fresh:
                fresh_sorted = sorted(fresh,
                                      key=lambda t: (t["wreck_score"], t["confidence"]),
                                      reverse=True)
                for i, t in enumerate(fresh_sorted, 1):
                    logger.info(
                        "  NEW #%-2d  lat=%.5f lon=%.5f  score=%d  conf=%.2f"
                        "  extent=%.0fm  AR=%.1f",
                        i, t["lat"], t["lon"], t["wreck_score"], t["confidence"],
                        t["extent_m"], t["aspect_ratio"],
                    )
                delta_path = output_dir / f"sentinel_delta_{sweep:03d}.json"
                delta_path.write_text(json.dumps(fresh_sorted, indent=2), encoding="utf-8")
                logger.info("Delta (new candidates) → %s", delta_path)
            else:
                logger.info("No new candidates this sweep")

            state.save()

            if max_sweeps > 0 and sweep >= max_sweeps:
                logger.info("Reached max sweeps (%d) — stopping", max_sweeps)
                break

        else:
            logger.debug("No model improvement (acc=%.4f <= %.4f) — sleeping %.0fs",
                         new_acc, state.last_val_acc, poll_interval)

        time.sleep(poll_interval)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(REPO_ROOT / "wreck_hunting_ml" / "sentinel.log",
                                mode="a", encoding="utf-8"),
        ],
    )

    p = argparse.ArgumentParser(description="WH2K Sentinel — CPU background scanner")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to best_resnet18.pt (monitored for improvements)")
    p.add_argument("--grid-tif",       required=True,
                   help="GeoTIFF magnetic grid to scan")
    p.add_argument("--output-dir",     default="wreck_hunting_ml/sentinel")
    p.add_argument("--poll-interval",  type=float, default=300.0,
                   help="Seconds between checkpoint polls (default 300 = 5 min)")
    p.add_argument("--sweeps",         type=int,   default=0,
                   help="Max number of scan sweeps (0 = infinite loop)")
    p.add_argument("--min-score",      type=int,   default=MIN_SCORE,
                   help=f"Minimum wreck score to include (default {MIN_SCORE})")
    p.add_argument("--no-first-sweep", action="store_true",
                   help="Don't scan on startup, wait for model improvement")
    args = p.parse_args()

    if not Path(args.grid_tif).exists():
        logger.warning("Grid TIF not found: %s — Sentinel will scan when it appears", args.grid_tif)

    run_sentinel(
        checkpoint_path=Path(args.checkpoint),
        grid_tif_path=Path(args.grid_tif),
        output_dir=Path(args.output_dir),
        poll_interval=args.poll_interval,
        max_sweeps=args.sweeps,
        force_first_sweep=not args.no_first_sweep,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
