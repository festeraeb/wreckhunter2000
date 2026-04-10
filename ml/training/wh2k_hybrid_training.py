"""
WreckHunter 2000 — Hybrid Training Orchestrator
==================================================
Single entry point that runs the complete "Anchor & Supplement" pipeline:

  1. EXTRACT REAL CHIPS  — Pull 2km×2km chips from catalog GeoTIFFs at all
                            known AWOIS wreck and wellhead locations.
  2. SATELLITE PROOF     — Take a known 700ft+ steel wreck, run upward
                            continuation to 400km, cross-check real Swarm data.
  3. SYNTHETIC BOOST     — Generate synthetic tiles (Iron Giant, Cargo Ghost,
                            Pin-Prick, Geology Only) to fill sparse target classes.
  4. TRAINING            — Train ResNet-18 on merged real+synthetic dataset.
  5. INFERENCE           — Run trained model on Lake Erie Central Basin.
                            Output GeoJSON of Score 8-10 unknowns.

Usage:
  # Full pipeline
  python wh2k_hybrid_training.py --all

  # Just generate data (no training)
  python wh2k_hybrid_training.py --data-only

  # Just train (data already generated)
  python wh2k_hybrid_training.py --train-only

  # Just run inference (model already trained)
  python wh2k_hybrid_training.py --infer-only --grid-tif magnetic_data/grids/erie.tif
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "wreck_hunting_ml" / "data"
MODEL_DIR = REPO_ROOT / "wreck_hunting_ml" / "models"


def step_1_extract_real_chips(wells_csv=None, n_background=5000):
    """Step 1: Extract Real Chips from catalog grids at known positions."""
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting Real Chips (AWOIS + GLSC + Background)")
    logger.info("=" * 60)

    from scripts.wh2k_chip_extractor import extract_all_chips

    result = extract_all_chips(
        chip_px=224,
        chip_extent_m=2000.0,
        n_background=n_background,
        wells_csv=wells_csv,
        output_dir=DATA_DIR / "real_chips",
    )

    logger.info("Real chips: %d extracted", len(result["labels"]))
    return result


def step_2_satellite_proof(aero_tif=None, sat_tif=None, wrecks_json=None):
    """Step 2: Satellite Proof — upward continuation test."""
    logger.info("=" * 60)
    logger.info("STEP 2: Satellite Proof (Upward Continuation to 400km)")
    logger.info("=" * 60)

    if not aero_tif or not sat_tif:
        logger.warning(
            "Satellite proof requires --aero-tif and --sat-tif. "
            "Skipping (run separately with wh2k_upward_continuation.py)."
        )
        return None

    from scripts.wh2k_upward_continuation import batch_satellite_proof

    if wrecks_json and Path(wrecks_json).exists():
        with open(wrecks_json) as f:
            wrecks = json.load(f)
    else:
        # Use default known large steel wrecks
        wrecks = [
            {"name": "Colgate (Whaleback)", "lat": 42.425, "lon": -80.813},
            {"name": "SS Atlantic", "lat": 42.520, "lon": -81.300},
        ]
        logger.info("Using %d default known wrecks for satellite proof", len(wrecks))

    results = batch_satellite_proof(
        aero_tif_path=aero_tif,
        satellite_tif_path=sat_tif,
        wreck_locations=wrecks,
        continuation_height_m=400_000.0,
        threshold_nt=0.5,
    )

    out_path = DATA_DIR / "satellite_proof_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_viable = sum(1 for r in results if r.get("sat_detection_viable"))
    logger.info("Satellite proof: %d / %d wrecks show viable detection", n_viable, len(results))
    return results


def step_3_synthetic_boost(n_steel=500, n_wood=500, n_wellhead=500, n_geology=5000, seed=42):
    """Step 3: Generate synthetic tiles to boost sparse classes."""
    logger.info("=" * 60)
    logger.info("STEP 3: Synthetic Boost (3 Archetypes + Geology Background)")
    logger.info("=" * 60)

    from scripts.wh2k_synthetic_tiles import generate_training_tiles
    import numpy as np

    result = generate_training_tiles(
        n_steel=n_steel,
        n_wood=n_wood,
        n_wellhead=n_wellhead,
        n_geology=n_geology,
        n_pixels=224,
        grid_extent_m=2000.0,
        seed=seed,
    )

    out_dir = DATA_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "synthetic_tiles.npz",
        tiles=result["tiles"],
        labels=result["labels"],
    )
    with open(out_dir / "synthetic_metadata.json", "w") as f:
        json.dump(result["metadata"], f, indent=2, default=str)

    logger.info("Synthetic tiles: %d generated", len(result["labels"]))
    return result


def step_4_train(epochs=50, batch_size=32, lr=1e-4, device="auto"):
    """Step 4: Train ResNet-18 on merged real + synthetic data."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training ResNet-18 (Hybrid: Real + Synthetic)")
    logger.info("=" * 60)

    from scripts.wh2k_resnet_training import train_resnet18

    real_path = DATA_DIR / "real_chips" / "real_chips.npz"
    synth_path = DATA_DIR / "synthetic" / "synthetic_tiles.npz"

    if not real_path.exists() and not synth_path.exists():
        logger.error("No training data found. Run steps 1 and/or 3 first.")
        return None

    results = train_resnet18(
        real_data_path=str(real_path) if real_path.exists() else None,
        synthetic_data_path=str(synth_path) if synth_path.exists() else None,
        output_dir=str(MODEL_DIR),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
    )

    logger.info("Training complete. Test accuracy: %.4f", results.get("test_accuracy", 0))
    return results


def step_5_inference(grid_tif, wells_csv=None, min_score=8, device="auto"):
    """Step 5: Run inference on Lake Erie Central Basin."""
    logger.info("=" * 60)
    logger.info("STEP 5: Inference — Scan → Correlate → Subtract → Score → Export")
    logger.info("=" * 60)

    model_path = MODEL_DIR / "best_resnet18.pt"
    if not model_path.exists():
        logger.error("No trained model found at %s. Run step 4 first.", model_path)
        return None

    from scripts.wh2k_inference_scorer import run_full_inference

    output_path = REPO_ROOT / "wreck_hunting_ml" / "output" / "wh2k_targets.geojson"

    summary = run_full_inference(
        model_path=str(model_path),
        grid_tif_path=grid_tif,
        wells_csv=wells_csv,
        output_geojson=str(output_path),
        min_score=min_score,
        device=device,
    )

    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="WreckHunter 2000 — Hybrid Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (generate data + train + infer)
  python wh2k_hybrid_training.py --all --grid-tif magnetic_data/grids/erie.tif

  # Generate all training data only
  python wh2k_hybrid_training.py --data-only

  # Train on existing data
  python wh2k_hybrid_training.py --train-only --epochs 100

  # Run inference with existing model
  python wh2k_hybrid_training.py --infer-only --grid-tif magnetic_data/grids/erie.tif
        """,
    )

    # Pipeline control
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run full pipeline (steps 1-5)")
    group.add_argument("--data-only", action="store_true", help="Steps 1-3 only (extract + synthetic)")
    group.add_argument("--train-only", action="store_true", help="Step 4 only (ResNet training)")
    group.add_argument("--infer-only", action="store_true", help="Step 5 only (inference + scoring)")

    # Data options
    parser.add_argument("--wells-csv", type=str, default=None)
    parser.add_argument("--n-background", type=int, default=5000)
    parser.add_argument("--n-steel", type=int, default=500)
    parser.add_argument("--n-wood", type=int, default=500)
    parser.add_argument("--n-wellhead", type=int, default=500)
    parser.add_argument("--n-geology", type=int, default=5000)

    # Satellite proof
    parser.add_argument("--aero-tif", type=str, default=None,
                        help="Tier 1 aero GeoTIFF for satellite proof")
    parser.add_argument("--sat-tif", type=str, default=None,
                        help="Tier 4 satellite GeoTIFF for satellite proof")
    parser.add_argument("--wrecks-json", type=str, default=None,
                        help="Known large steel wrecks JSON for satellite proof")

    # Training options
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Inference options
    parser.add_argument("--grid-tif", type=str, default=None,
                        help="GeoTIFF grid for Central Basin inference")
    parser.add_argument("--min-score", type=int, default=8)

    args = parser.parse_args()

    if args.all or args.data_only:
        step_1_extract_real_chips(wells_csv=args.wells_csv, n_background=args.n_background)
        step_2_satellite_proof(aero_tif=args.aero_tif, sat_tif=args.sat_tif, wrecks_json=args.wrecks_json)
        step_3_synthetic_boost(
            n_steel=args.n_steel, n_wood=args.n_wood,
            n_wellhead=args.n_wellhead, n_geology=args.n_geology,
            seed=args.seed,
        )

    if args.all or args.train_only:
        step_4_train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)

    if args.all or args.infer_only:
        if not args.grid_tif:
            logger.error("--grid-tif is required for inference. Provide path to Central Basin GeoTIFF.")
        else:
            step_5_inference(
                grid_tif=args.grid_tif,
                wells_csv=args.wells_csv,
                min_score=args.min_score,
                device=args.device,
            )

    logger.info("WreckHunter 2000 pipeline finished.")


if __name__ == "__main__":
    main()
