"""
Lake Erie Feedback Loop — Continuous Model Improvement
=======================================================
Automatically labels new candidates by cross-referencing against the
Wreck Hunter 2000 database and OGSr well data, then performs incremental
XGBoost model updates without full retraining.

Flow:
  new candidate → check wreck DB → check OGSRL wells → auto-label
    → accumulate labeled samples → incremental XGBoost warm-start retrain

This module NEVER modifies the real wreck database. Labels are stored in
a separate feedback_labels.json file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scripts.erie_wellhead_discriminator import (
        haversine_m, load_ogsr_wells, get_all_known_wrecks, Wellhead, KnownWreck,
    )
    from scripts.train_lake_erie_offaxis import FEATURE_NAMES, predict_candidates
except ImportError:
    from erie_wellhead_discriminator import (
        haversine_m, load_ogsr_wells, get_all_known_wrecks, Wellhead, KnownWreck,
    )
    from train_lake_erie_offaxis import FEATURE_NAMES, predict_candidates


@dataclass
class FeedbackLabel:
    """A labeled candidate from the feedback loop."""
    lat: float
    lon: float
    label: int              # 1=wreck, 0=not-wreck
    source: str             # "wreck_db", "ogsrl_well", "user", "model_confident"
    confidence: float       # 0.0-1.0
    match_name: str = ""
    match_distance_m: float = 0.0
    features: dict = None
    timestamp: float = 0.0


# ── Core feedback functions ─────────────────────────────────────────────────

def new_candidate(
    lat: float,
    lon: float,
    features: dict,
    wells_csv: str | Path | None = None,
    wreck_db_path: str | Path | None = None,
    wellhead_radius_m: float = 1000.0,
    wreck_radius_m: float = 5000.0,
    feedback_dir: str | Path = "models/erie/feedback",
) -> FeedbackLabel:
    """Process a new candidate through the feedback pipeline.
    
    1. Check against known wreck list → auto-label as wreck (label=1)
    2. Check against OGSr wells → auto-label as wellhead (label=0)
    3. If no match, run model prediction → label if confident (>0.9 or <0.1)
    4. Store label in feedback_labels.json
    """
    feedback_dir = Path(feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Check known wrecks
    known_wrecks = get_all_known_wrecks()
    for kw in known_wrecks:
        d = haversine_m(lat, lon, kw.lat, kw.lon)
        if d < wreck_radius_m:
            label = FeedbackLabel(
                lat=lat, lon=lon, label=1,
                source="wreck_db", confidence=max(0.5, 1.0 - d / wreck_radius_m),
                match_name=kw.name, match_distance_m=d,
                features=features, timestamp=time.time(),
            )
            _save_feedback(label, feedback_dir)
            return label

    # Check OGSr wells
    if wells_csv:
        wells = load_ogsr_wells(wells_csv)
    else:
        wells = []
        for p in [Path("data/wells.csv"), Path("reference/wells.csv")]:
            if p.exists():
                wells = load_ogsr_wells(p)
                break

    for w in wells:
        d = haversine_m(lat, lon, w.lat, w.lon)
        if d < wellhead_radius_m:
            label = FeedbackLabel(
                lat=lat, lon=lon, label=0,
                source="ogsrl_well", confidence=max(0.5, 1.0 - d / wellhead_radius_m),
                match_name=w.name, match_distance_m=d,
                features=features, timestamp=time.time(),
            )
            _save_feedback(label, feedback_dir)
            return label

    # No DB match — use model prediction if available
    try:
        predicted = predict_candidates([features])
        prob = predicted[0].get("wreck_prob", 0.5)
        if prob > 0.9:
            label = FeedbackLabel(
                lat=lat, lon=lon, label=1,
                source="model_confident", confidence=prob,
                features=features, timestamp=time.time(),
            )
        elif prob < 0.1:
            label = FeedbackLabel(
                lat=lat, lon=lon, label=0,
                source="model_confident", confidence=1.0 - prob,
                features=features, timestamp=time.time(),
            )
        else:
            label = FeedbackLabel(
                lat=lat, lon=lon, label=-1,
                source="model_uncertain", confidence=prob,
                features=features, timestamp=time.time(),
            )
    except Exception:
        label = FeedbackLabel(
            lat=lat, lon=lon, label=-1,
            source="no_model", confidence=0.0,
            features=features, timestamp=time.time(),
        )

    _save_feedback(label, feedback_dir)
    return label


def _save_feedback(label: FeedbackLabel, feedback_dir: Path):
    """Append a feedback label to the persistent store."""
    store_path = feedback_dir / "feedback_labels.json"
    existing = []
    if store_path.exists():
        with open(store_path) as f:
            existing = json.load(f)
    existing.append(asdict(label))
    with open(store_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def load_feedback_labels(feedback_dir: str | Path = "models/erie/feedback") -> list[FeedbackLabel]:
    """Load all accumulated feedback labels."""
    store_path = Path(feedback_dir) / "feedback_labels.json"
    if not store_path.exists():
        return []
    with open(store_path) as f:
        data = json.load(f)
    return [FeedbackLabel(**d) for d in data]


# ── Incremental retraining ──────────────────────────────────────────────────

def incremental_retrain(
    model_dir: str | Path = "models/erie",
    feedback_dir: str | Path = "models/erie/feedback",
    min_new_samples: int = 5,
) -> dict:
    """Perform incremental XGBoost retrain with accumulated feedback labels.
    
    Uses XGBoost's warm-start capability (xgb_model parameter) to update
    existing models without full retraining. Only retrains if enough new
    labeled samples have accumulated.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return {"error": "xgboost not installed"}

    model_dir = Path(model_dir)
    feedback_dir = Path(feedback_dir)

    # Load feedback labels
    labels = load_feedback_labels(feedback_dir)
    labeled = [l for l in labels if l.label in (0, 1)]

    if len(labeled) < min_new_samples:
        return {
            "status": "skipped",
            "reason": f"Only {len(labeled)} labeled samples (need {min_new_samples})",
        }

    # Build feature matrix from feedback
    X_new = []
    y_new = []
    for lbl in labeled:
        if lbl.features:
            feat_vec = [lbl.features.get(f, 0.0) for f in FEATURE_NAMES]
            X_new.append(feat_vec)
            y_new.append(lbl.label)

    if len(X_new) < min_new_samples:
        return {"status": "skipped", "reason": "Insufficient feature data in feedback"}

    X_new = np.array(X_new, dtype=np.float64)
    y_new = np.array(y_new, dtype=np.int32)
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)

    results = {}

    # Update each model
    for model_name in ["western", "central", "eastern", "erie_wide"]:
        model_path = model_dir / f"erie_{model_name}_xgb.json"
        if not model_path.exists():
            results[model_name] = {"status": "skipped", "reason": "model not found"}
            continue

        # Load existing model
        existing = xgb.XGBClassifier()
        existing.load_model(str(model_path))

        # Incremental training (warm start)
        try:
            # XGBoost supports continuing training with xgb_model parameter
            dtrain = xgb.DMatrix(X_new, label=y_new)
            params = existing.get_xgb_params()
            params["process_type"] = "update"
            params["updater"] = "refresh"
            params["refresh_leaf"] = True

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=10,  # Small update
                xgb_model=str(model_path),
            )

            # Save updated model
            backup_path = model_dir / f"erie_{model_name}_xgb.backup.json"
            model_path.rename(backup_path)
            bst.save_model(str(model_path))

            results[model_name] = {
                "status": "updated",
                "new_samples": len(X_new),
                "backup": str(backup_path),
            }
        except Exception as e:
            results[model_name] = {"status": "failed", "error": str(e)}

    # Record retrain event
    retrain_log = feedback_dir / "retrain_log.json"
    log_entries = []
    if retrain_log.exists():
        with open(retrain_log) as f:
            log_entries = json.load(f)
    log_entries.append({
        "timestamp": time.time(),
        "new_samples": len(X_new),
        "results": results,
    })
    with open(retrain_log, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    return {"status": "completed", "models_updated": results}


# ── Feedback report ─────────────────────────────────────────────────────────

def feedback_report(
    model_dir: str | Path = "models/erie",
    feedback_dir: str | Path = "models/erie/feedback",
) -> dict:
    """Generate a summary of the feedback loop state."""
    feedback_dir = Path(feedback_dir)
    model_dir = Path(model_dir)

    labels = load_feedback_labels(feedback_dir)
    n_wreck = sum(1 for l in labels if l.label == 1)
    n_notwreck = sum(1 for l in labels if l.label == 0)
    n_uncertain = sum(1 for l in labels if l.label == -1)

    sources = {}
    for l in labels:
        sources[l.source] = sources.get(l.source, 0) + 1

    # Check retrain history
    retrain_log = feedback_dir / "retrain_log.json"
    retrains = []
    if retrain_log.exists():
        with open(retrain_log) as f:
            retrains = json.load(f)

    # Check model existence
    models_available = {}
    for name in ["western", "central", "eastern", "erie_wide"]:
        path = model_dir / f"erie_{name}_xgb.json"
        models_available[name] = path.exists()

    return {
        "total_feedback_labels": len(labels),
        "wreck_labels": n_wreck,
        "not_wreck_labels": n_notwreck,
        "uncertain_labels": n_uncertain,
        "label_sources": sources,
        "total_retrains": len(retrains),
        "last_retrain": retrains[-1]["timestamp"] if retrains else None,
        "models_available": models_available,
        "ready_for_retrain": n_wreck + n_notwreck >= 5,
    }
