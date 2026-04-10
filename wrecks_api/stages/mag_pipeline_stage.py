from __future__ import annotations

import sys
from pathlib import Path

# Ensure pipeline source directories are importable
_repo_root = Path(__file__).resolve().parents[2]
for _sub in ("pipelines/mag", "ml/training", "scripts"):
    _p = str(_repo_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_mag_pipeline_stage(_paths: list, output_dir: str, cfg: dict) -> dict:
    """Orchestrate the magnetic anomaly pipeline.

    Supports two modes controlled by cfg["mode"]:
      - "full"       (default) Data acquisition + detection via mag_data_pipeline
      - "validate"   Original model-only validation (leave-one-out, saved models)
    """
    if not cfg.get("run_mag_pipeline", True):
        return {"enabled": False, "status": "skipped", "reason": "disabled_by_config"}

    mode = cfg.get("mode", "full")

    if mode == "validate":
        return _run_validation_mode(output_dir, cfg)

    return _run_full_pipeline(output_dir, cfg)


def _run_full_pipeline(output_dir: str, cfg: dict) -> dict:
    """Full data acquisition + anomaly detection pipeline."""
    from mag_data_pipeline import run_pipeline

    source_keys = cfg.get("sources", ["usgs_namag", "usgs_usmag"])
    if isinstance(source_keys, str):
        source_keys = [s.strip() for s in source_keys.split(",")]
    bbox = tuple(cfg.get("bbox", [-92.5, 41.0, -75.0, 49.0]))
    stages = cfg.get("stages", "all")
    models_dir = str(cfg.get("mag_models_dir", "bagfilework/training/models"))
    db_path = str(cfg.get("db_path", "db/wrecks.db"))
    threshold = float(cfg.get("threshold", 0.3))

    result = run_pipeline(
        output_dir=output_dir,
        source_keys=source_keys,
        bbox=bbox,
        models_dir=models_dir,
        db_path=db_path,
        stages=stages,
        threshold=threshold,
    )

    return {
        "enabled": True,
        "mode": "full",
        "status": result.get("status", "completed"),
        "stages": result.get("stages", []),
        "grids_produced": result.get("grids_produced", []),
        "patches_produced": result.get("patches_produced", 0),
        "detections": result.get("detections", 0),
        "candidates_count": result.get("candidates_count", 0),
        "output_dir": output_dir,
    }


def _run_validation_mode(output_dir: str, cfg: dict) -> dict:
    """Original model-validation-only mode (training refinement)."""
    try:
        from mag_pipeline_validator import (
            leave_one_out_validation,
            validate_with_saved_models,
            write_csv,
            write_json,
        )
    except ImportError as e:
        return {
            "enabled": True,
            "mode": "validate",
            "status": "failed",
            "error": f"mag_pipeline_validator not importable: {e}",
        }

    models_dir = str(cfg.get("mag_models_dir", "bagfilework/training/models"))
    stage_dir = Path(output_dir) / "pipeline" / "mag_validation"
    stage_dir.mkdir(parents=True, exist_ok=True)

    saved_results = validate_with_saved_models(models_dir)
    loo_results = leave_one_out_validation()
    output = {
        "saved_model_validation": saved_results,
        "leave_one_out": loo_results,
    }

    json_path = stage_dir / "mag_validation_results.json"
    write_json(output, str(json_path))
    write_csv(saved_results.get("wreck_predictions", []), str(stage_dir / "mag_wreck_predictions.csv"))
    if isinstance(loo_results, dict):
        write_csv(loo_results.get("per_wreck", []), str(stage_dir / "mag_loo_results.csv"))

    return {
        "enabled": True,
        "mode": "validate",
        "status": "completed",
        "models_dir": models_dir,
        "saved_model_error": saved_results.get("error"),
        "loo_error": loo_results.get("error") if isinstance(loo_results, dict) else None,
        "output_json": str(json_path),
    }
