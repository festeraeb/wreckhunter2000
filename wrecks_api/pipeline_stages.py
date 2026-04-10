"""Pipeline orchestrator for modular post-scan stages."""

from __future__ import annotations

import traceback
from typing import Callable

from wrecks_api.stages.mag_pipeline_stage import run_mag_pipeline_stage

StageResult = dict
StageRunner = Callable[[list, str, dict], StageResult]


STAGE_REGISTRY: dict[str, StageRunner] = {
    "mag_pipeline": run_mag_pipeline_stage,
}


def initial_pipeline_state(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    requested = cfg.get("pipeline_stages")
    if isinstance(requested, list) and requested:
        stages = [s for s in requested if isinstance(s, str) and s in STAGE_REGISTRY]
    else:
        stages = list(STAGE_REGISTRY.keys())
    return {name: {"status": "pending"} for name in stages}


def run_pipeline_stages(paths: list, output_dir: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    state = initial_pipeline_state(cfg)

    for stage_name in list(state.keys()):
        runner = STAGE_REGISTRY[stage_name]
        try:
            state[stage_name] = runner(paths, output_dir, cfg)
        except Exception as e:
            state[stage_name] = {
                "enabled": True,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(limit=3),
            }

    return state
