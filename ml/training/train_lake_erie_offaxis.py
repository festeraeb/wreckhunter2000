"""
Lake Erie Off-Axis Detector — XGBoost Training Pipeline
========================================================
Trains 4 models to push aeromagnetic detection to 1-2km+ off flight line 
while hard-rejecting wellheads, tuned per Lake Erie basin.

Models:
  1. Western Basin model  (west of 82.0°W)
  2. Central Basin model  (82.0°W to 80.0°W — sweet spot + Iron Mountain zone)
  3. Eastern Basin model  (east of 80.0°W)
  4. Erie-wide model      (all basins combined)

Training data:
  - Positives: Colgate (#103), SS Atlantic, SS Merida + NDA/ShipwreckWorld known wrecks
  - Negatives: OGSr Ontario petroleum wells (#63, #85, etc.)
  - Synthetics: Dipole-model generated (10k wreck + 3k wellhead + 2k geological per basin)
  - NEVER mix synthetic data into real DB

Feature set (expanded):
  amplitude_peak_abs, amplitude_mean_abs, gradient_contrast, dipole_separation_m,
  lobe_symmetry_ratio, axis_offset_deg (from NE-SW strike), aspect_ratio,
  flip_distance_km, pixel_count, distance_to_nearest_flight_line_m,
  basin_id (one-hot), local_snr_vs_basin_median, curvelet_edge_score

Usage:
  python train_lake_erie_offaxis.py --candidates path/to/candidates.csv \\
                                     --wells path/to/ogsr_wells.csv \\
                                     --output models/erie/
  
  Or call from API: POST /tools/erie-scanner/train
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature names (must match erie_synthetic_dipole.extract_features_from_grid) ──

FEATURE_NAMES = [
    "amplitude_peak_abs",
    "amplitude_mean_abs",
    "gradient_contrast",
    "dipole_separation_m",
    "lobe_symmetry_ratio",
    "axis_offset_deg",
    "aspect_ratio",
    "flip_distance_km",
    "pixel_count",
    "distance_to_nearest_flight_line_m",
    "basin_western",
    "basin_central",
    "basin_eastern",
    "local_snr_vs_basin_median",
    "curvelet_edge_score",
    "width_m",
    "height_m",
]

# Basin boundaries (updated per friend's suggestion: hard boundaries)
BASIN_EDGES = {
    "western":  (-83.50, -82.00),
    "central":  (-82.00, -80.00),
    "eastern":  (-80.00, -78.80),
}

# Basin-specific noise levels (nT std dev)
BASIN_NOISE = {"western": 12.0, "central": 6.0, "eastern": 3.5}


# ── Real data loading ───────────────────────────────────────────────────────

def load_real_candidates(candidates_csv: str | Path) -> list[dict]:
    """Load adaptive_candidates_scored.csv and extract features for training."""
    rows = []
    with open(candidates_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("center_lat", 0))
                lon = float(row.get("center_lon", 0))
                amp_peak = float(row.get("amplitude_peak_abs", 0) or 0)
                amp_mean = float(row.get("amplitude_mean_abs", 0) or 0)
                width = float(row.get("width_m", 0) or 0)
                height = float(row.get("height_m", 0) or 0)
                pixel_count = int(float(row.get("pixel_count", 0) or 0))
                composite = float(row.get("_composite_score", 0) or 0)
                dipole_score = float(row.get("_dipole_score", 0) or 0)
                label_id = int(float(row.get("label_id", 0) or 0))
            except (ValueError, TypeError):
                continue

            # Determine basin from longitude
            basin = _get_basin(lon)
            basin_id = {"western": 0, "central": 1, "eastern": 2}.get(basin, 1)

            # Extract dipole separation from reasons if available
            all_reasons = str(row.get("_all_reasons", ""))
            dipole_sep = _parse_dipole_sep(all_reasons)
            lobe_sym = _parse_lobe_symmetry(all_reasons)
            axis_offset = _parse_axis_offset(all_reasons)
            flip_dist = _parse_flip_distance(all_reasons)
            gradient = _parse_gradient_contrast(all_reasons, amp_peak)

            # SNR relative to basin median
            local_snr = amp_peak / (BASIN_NOISE.get(basin, 6.0) + 1e-10)

            # Curvelet edge score (placeholder — will come from Rust when available)
            curvelet = 0.0

            feat = {
                "amplitude_peak_abs": amp_peak,
                "amplitude_mean_abs": amp_mean,
                "gradient_contrast": gradient,
                "dipole_separation_m": dipole_sep,
                "lobe_symmetry_ratio": lobe_sym,
                "axis_offset_deg": axis_offset,
                "aspect_ratio": max(width, height) / max(min(width, height), 1),
                "flip_distance_km": flip_dist,
                "pixel_count": pixel_count,
                "distance_to_nearest_flight_line_m": 0.0,  # Unknown for real data
                "basin_western": 1 if basin_id == 0 else 0,
                "basin_central": 1 if basin_id == 1 else 0,
                "basin_eastern": 1 if basin_id == 2 else 0,
                "local_snr_vs_basin_median": local_snr,
                "curvelet_edge_score": curvelet,
                "width_m": width,
                "height_m": height,
                # Metadata (not features)
                "_label_id": label_id,
                "_lat": lat,
                "_lon": lon,
                "_basin": basin,
                "_composite_score": composite,
                "_dipole_score": dipole_score,
            }
            rows.append(feat)

    logger.info("Loaded %d real candidates from %s", len(rows), candidates_csv)
    return rows


def _get_basin(lon: float) -> str:
    for name, (lo, hi) in BASIN_EDGES.items():
        if lo <= lon <= hi:
            return name
    return "central"


def _parse_dipole_sep(reasons: str) -> float:
    import re
    m = re.search(r"sep[:\s]*(\d+\.?\d*)\s*m", reasons, re.I)
    return float(m.group(1)) if m else 0.0


def _parse_lobe_symmetry(reasons: str) -> float:
    import re
    m = re.search(r"symmetry[:\s]*(\d+\.?\d*)", reasons, re.I)
    return float(m.group(1)) if m else 0.0


def _parse_axis_offset(reasons: str) -> float:
    import re
    m = re.search(r"(\d+)° off regional", reasons, re.I)
    return float(m.group(1)) if m else 0.0


def _parse_flip_distance(reasons: str) -> float:
    import re
    m = re.search(r"flip[:\s]*(\d+\.?\d*)\s*(?:m|km)", reasons, re.I)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if "m" in reasons[m.start():m.end()].lower() and "km" not in reasons[m.start():m.end()].lower() else val
    return 0.0


def _parse_gradient_contrast(reasons: str, amp_peak: float) -> float:
    import re
    m = re.search(r"gradient[:\s]*(\d+\.?\d*)", reasons, re.I)
    return float(m.group(1)) if m else amp_peak * 0.1  # Default estimate


# ── Ground truth labeling for real data ──────────────────────────────────────

# Confirmed ground truth
CONFIRMED_POSITIVES = {103}  # Colgate whaleback
CONFIRMED_NEGATIVES = {63, 85}  # Gas wellheads

# Additional known positives from cross-reference with wreck databases
KNOWN_WRECK_LABEL_IDS: set[int] = set()  # Add any that are identified

def label_real_data(
    candidates: list[dict],
    wells: list[dict] | None = None,
    wellhead_radius_m: float = 1000.0,
) -> tuple[list[dict], list[int]]:
    """Assign labels to real candidates based on ground truth + well proximity.
    
    Returns (features, labels) where:
      label = 1 → confirmed or likely wreck
      label = 0 → confirmed or likely wellhead/geological
      label = -1 → unlabeled (excluded from supervised training)
    """
    try:
        from scripts.erie_wellhead_discriminator import haversine_m
    except ImportError:
        from erie_wellhead_discriminator import haversine_m

    features_out = []
    labels_out = []

    for cand in candidates:
        lid = cand["_label_id"]

        if lid in CONFIRMED_POSITIVES or lid in KNOWN_WRECK_LABEL_IDS:
            features_out.append(cand)
            labels_out.append(1)
        elif lid in CONFIRMED_NEGATIVES:
            features_out.append(cand)
            labels_out.append(0)
        elif wells:
            # Check proximity to any known well
            lat, lon = cand["_lat"], cand["_lon"]
            near_well = False
            for w in wells:
                d = haversine_m(lat, lon, float(w.get("lat", 0)), float(w.get("lon", 0)))
                if d < wellhead_radius_m:
                    near_well = True
                    break
            if near_well:
                features_out.append(cand)
                labels_out.append(0)  # Pseudo-label: near wellhead
            else:
                features_out.append(cand)
                labels_out.append(-1)  # Unlabeled
        else:
            features_out.append(cand)
            labels_out.append(-1)

    n_pos = sum(1 for l in labels_out if l == 1)
    n_neg = sum(1 for l in labels_out if l == 0)
    n_unk = sum(1 for l in labels_out if l == -1)
    logger.info("Labeled real data: %d positive, %d negative, %d unlabeled", n_pos, n_neg, n_unk)
    return features_out, labels_out


# ── Training pipeline ───────────────────────────────────────────────────────

def train_all_models(
    candidates_csv: str | Path | None = None,
    wells_csv: str | Path | None = None,
    output_dir: str | Path = "models/erie",
    n_synth_wreck: int = 10_000,
    n_synth_wellhead: int = 3_000,
    n_synth_geological: int = 2_000,
    progress_callback=None,
) -> dict:
    """Train all 4 XGBoost models (3 basin + 1 Erie-wide).
    
    Returns a report dict with model paths, metrics, and feature importances.
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, precision_score, recall_score
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install xgboost scikit-learn joblib"}

    try:
        from scripts.erie_synthetic_dipole import (
            generate_wreck_synthetics, generate_wellhead_synthetics,
            generate_geological_synthetics,
        )
    except ImportError:
        from erie_synthetic_dipole import (
            generate_wreck_synthetics, generate_wellhead_synthetics,
            generate_geological_synthetics,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg, pct=-1):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, pct)

    report = {
        "models": {},
        "training_start": time.time(),
        "feature_names": FEATURE_NAMES,
    }

    # ── Step 1: Load real data ──
    progress("Loading real candidate data...", 0.0)
    real_candidates = []
    if candidates_csv:
        real_candidates = load_real_candidates(candidates_csv)

    # Load wells for pseudo-labeling
    wells_list = []
    if wells_csv:
        try:
            try:
                from scripts.erie_wellhead_discriminator import load_ogsr_wells
            except ImportError:
                from erie_wellhead_discriminator import load_ogsr_wells
            from dataclasses import asdict
            wells = load_ogsr_wells(wells_csv)
            wells_list = [asdict(w) for w in wells]
        except Exception as e:
            logger.warning("Could not load wells: %s", e)

    # Label real data
    labeled_real, real_labels = label_real_data(real_candidates, wells_list)
    progress(f"Real data: {len(labeled_real)} candidates", 0.05)

    # ── Step 2: Generate synthetics ──
    progress("Generating synthetic training data...", 0.1)
    rng = np.random.default_rng(42)

    progress(f"Generating {n_synth_wreck} wreck synthetics per basin...", 0.12)
    wreck_synth = generate_wreck_synthetics(n_synth_wreck, rng=rng)

    progress(f"Generating {n_synth_wellhead} wellhead synthetics per basin...", 0.25)
    wellhead_synth = generate_wellhead_synthetics(n_synth_wellhead, rng=rng)

    progress(f"Generating {n_synth_geological} geological synthetics per basin...", 0.35)
    geo_synth = generate_geological_synthetics(n_synth_geological, rng=rng)

    # ── Step 3: Build training datasets per basin ──
    progress("Building training datasets...", 0.45)
    basins = ["western", "central", "eastern"]
    basin_datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_X, all_y = [], []

    for basin in basins:
        X_basin, y_basin = [], []

        # Add real labeled data for this basin
        for cand, label in zip(labeled_real, real_labels):
            if label == -1:
                continue  # Skip unlabeled
            if cand["_basin"] == basin:
                feat_vec = [cand.get(f, 0.0) for f in FEATURE_NAMES]
                X_basin.append(feat_vec)
                y_basin.append(label)

        # Add synthetics
        for s in wreck_synth.get(basin, []):
            feat_vec = [s.get(f, 0.0) for f in FEATURE_NAMES]
            X_basin.append(feat_vec)
            y_basin.append(1)

        for s in wellhead_synth.get(basin, []):
            feat_vec = [s.get(f, 0.0) for f in FEATURE_NAMES]
            X_basin.append(feat_vec)
            y_basin.append(0)

        for s in geo_synth.get(basin, []):
            feat_vec = [s.get(f, 0.0) for f in FEATURE_NAMES]
            X_basin.append(feat_vec)
            y_basin.append(0)

        X_arr = np.array(X_basin, dtype=np.float64)
        y_arr = np.array(y_basin, dtype=np.int32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

        basin_datasets[basin] = (X_arr, y_arr)
        all_X.extend(X_basin)
        all_y.extend(y_basin)

        logger.info("%s basin: %d samples (%d pos, %d neg)",
                    basin, len(y_basin), sum(y_basin), len(y_basin) - sum(y_basin))

    # Erie-wide dataset
    X_all = np.array(all_X, dtype=np.float64)
    y_all = np.array(all_y, dtype=np.int32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    basin_datasets["erie_wide"] = (X_all, y_all)

    # ── Step 4: Train models ──
    model_names = basins + ["erie_wide"]
    total_models = len(model_names)

    for i, model_name in enumerate(model_names):
        pct = 0.5 + 0.4 * (i / total_models)
        progress(f"Training {model_name} model...", pct)

        X, y = basin_datasets[model_name]
        if len(X) < 10 or len(np.unique(y)) < 2:
            logger.warning("Skipping %s: insufficient data (%d samples)", model_name, len(X))
            report["models"][model_name] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        # Class weights: heavy penalty for false-positive wells 
        # (wellhead precision > 99% is the target)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            tree_method="hist",     # Fast histogram-based
            use_label_encoder=False,
        )

        # Train/validation split for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Cross-validation
        cv_model = xgb.XGBClassifier(
            n_estimators=model.best_iteration + 1 if hasattr(model, 'best_iteration') else 200,
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42,
            tree_method="hist", use_label_encoder=False,
        )

        n_folds = min(5, min(np.sum(y == 0), np.sum(y == 1)))
        if n_folds >= 2:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
            cv_accuracy = float(np.mean(cv_scores))
        else:
            cv_accuracy = 0.0

        # Full predictions for metrics
        y_pred = model.predict(X)
        precision = float(precision_score(y, y_pred, zero_division=0))
        recall = float(recall_score(y, y_pred, zero_division=0))

        # Feature importances
        importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

        # Save model
        model_path = output_dir / f"erie_{model_name}_xgb.json"
        model.save_model(str(model_path))

        # Also export ONNX for Rust integration
        onnx_path = _export_onnx(model, output_dir / f"erie_{model_name}.onnx")

        report["models"][model_name] = {
            "status": "trained",
            "samples": int(len(y)),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == 0)),
            "cv_accuracy": cv_accuracy,
            "precision": precision,
            "recall": recall,
            "best_iteration": int(model.best_iteration) if hasattr(model, 'best_iteration') else 0,
            "feature_importances": importances,
            "model_path": str(model_path),
            "onnx_path": str(onnx_path) if onnx_path else None,
        }

        progress(f"  {model_name}: accuracy={cv_accuracy:.3f} precision={precision:.3f} recall={recall:.3f}", pct + 0.08)

    # ── Step 5: Save report ──
    report["training_end"] = time.time()
    report["training_duration_s"] = report["training_end"] - report["training_start"]

    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    progress("Training complete!", 1.0)
    return report


# ── ONNX export for Rust integration ────────────────────────────────────────

def _export_onnx(model, output_path: Path) -> Optional[Path]:
    """Export XGBoost model to ONNX format for Rust inference."""
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, len(FEATURE_NAMES)]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        onnxmltools.utils.save_model(onnx_model, str(output_path))
        logger.info("ONNX model saved: %s", output_path)
        return output_path
    except ImportError:
        logger.info("onnxmltools not installed — ONNX export skipped. Install with: pip install onnxmltools")
        return None
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)
        return None


# ── Prediction / inference ──────────────────────────────────────────────────

def predict_candidates(
    features: list[dict],
    model_dir: str | Path = "models/erie",
    model_name: str = "erie_wide",
) -> list[dict]:
    """Run trained model on new candidates and return wreck probabilities."""
    import xgboost as xgb

    model_dir = Path(model_dir)
    model_path = model_dir / f"erie_{model_name}_xgb.json"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return features

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    X = np.array([
        [f.get(name, 0.0) for name in FEATURE_NAMES]
        for f in features
    ], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    probs = model.predict_proba(X)[:, 1]  # P(wreck)

    for feat, prob in zip(features, probs):
        feat["wreck_prob"] = float(prob)
        feat["off_axis_distance_estimate"] = feat.get("distance_to_nearest_flight_line_m", 0.0)

    return features


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Lake Erie Off-Axis Detector Training")
    parser.add_argument("--candidates", "-c",
                        default="adaptive_bg_erie_1000yd/adaptive_candidates_scored.csv",
                        help="Path to adaptive_candidates_scored.csv")
    parser.add_argument("--wells", "-w", help="Path to OGSr wells.csv")
    parser.add_argument("--output", "-o", default="models/erie", help="Output directory for models")
    parser.add_argument("--n-wreck", type=int, default=10000, help="Wreck synthetics per basin")
    parser.add_argument("--n-wellhead", type=int, default=3000, help="Wellhead synthetics per basin")
    parser.add_argument("--n-geological", type=int, default=2000, help="Geological synthetics per basin")
    args = parser.parse_args()

    print("=" * 70)
    print("Lake Erie Off-Axis Detector — XGBoost Training")
    print("=" * 70)
    print(f"Candidates: {args.candidates}")
    print(f"Wells:      {args.wells or 'none'}")
    print(f"Output:     {args.output}")
    print(f"Synthetics: {args.n_wreck} wreck + {args.n_wellhead} wellhead + {args.n_geological} geo per basin")
    print("=" * 70)

    result = train_all_models(
        candidates_csv=args.candidates if Path(args.candidates).exists() else None,
        wells_csv=args.wells,
        output_dir=args.output,
        n_synth_wreck=args.n_wreck,
        n_synth_wellhead=args.n_wellhead,
        n_synth_geological=args.n_geological,
    )

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"\nTraining completed in {result.get('training_duration_s', 0):.1f}s")
    for name, info in result.get("models", {}).items():
        if info.get("status") == "trained":
            print(f"  {name}: accuracy={info['cv_accuracy']:.3f} "
                  f"precision={info['precision']:.3f} recall={info['recall']:.3f} "
                  f"({info['samples']} samples)")
        else:
            print(f"  {name}: {info.get('status')} ({info.get('reason', '')})")
