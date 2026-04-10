"""
wh2k_run_huron_inference.py
============================
Run the full WH2K inference pipeline on Lake Huron aeromagnetic GeoTIFF.

Uses the same trained ResNet-18 model (trained on Erie synthetic + real data)
to scan the Huron raster for steel hull / wood cargo / wellhead signatures.

Usage
-----
  python scripts/wh2k_run_huron_inference.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add scripts dir to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from wh2k_inference_scorer import run_full_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("huron_inference")

# ── Lake Huron bounds (from rasterizer output extent) ─────────────────────

LAKE_HURON = {
    "lon_min": -85.10,
    "lat_min": 42.90,
    "lon_max": -81.18,
    "lat_max": 46.30,
}

# ── Paths ─────────────────────────────────────────────────────────────────

MODEL_PATH = REPO_ROOT / "wreck_hunting_ml" / "models" / "best_resnet18.pt"
HURON_TIF = REPO_ROOT / "magnetic_data" / "tier_2_aero_lowalt" / "local" / "gsc_huron_highres_0_001.tif"
WRECKS_DB = REPO_ROOT / "db" / "wrecks.db"
WELLS_CSV = REPO_ROOT / "eriewelldata" / "wells.csv"
OUTPUT_DIR = REPO_ROOT / "wreck_hunting_ml" / "output"

OUTPUT_GEOJSON = OUTPUT_DIR / "huron_targets.geojson"


def main() -> None:
    # Validate paths
    for label, p in [("Model", MODEL_PATH), ("Huron TIF", HURON_TIF), ("Wrecks DB", WRECKS_DB)]:
        if not p.exists():
            log.error("%s not found: %s", label, p)
            sys.exit(1)

    wells_arg = str(WELLS_CSV) if WELLS_CSV.exists() else None
    if wells_arg is None:
        log.warning("Wells CSV not found — running without well correlation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("WH2K Lake Huron Inference")
    log.info("=" * 60)
    log.info("Model    : %s", MODEL_PATH)
    log.info("Grid TIF : %s", HURON_TIF)
    log.info("Wrecks DB: %s", WRECKS_DB)
    log.info("Wells CSV: %s", wells_arg or "(none)")
    log.info("Bbox     : lon [%.2f, %.2f]  lat [%.2f, %.2f]",
             LAKE_HURON["lon_min"], LAKE_HURON["lon_max"],
             LAKE_HURON["lat_min"], LAKE_HURON["lat_max"])
    log.info("Output   : %s", OUTPUT_GEOJSON)

    summary = run_full_inference(
        model_path=str(MODEL_PATH),
        grid_tif_path=str(HURON_TIF),
        db_path=str(WRECKS_DB),
        wells_csv=wells_arg,
        output_geojson=str(OUTPUT_GEOJSON),
        confidence_threshold=0.5,
        min_score=8,
        match_radius_m=2000.0,
        device="auto",
        bbox=LAKE_HURON,
    )

    log.info("=" * 60)
    log.info("HURON INFERENCE COMPLETE")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
