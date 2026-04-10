# Bagrecovery Archive Manifest
Generated after full triage and extraction of `incomming/Bagrecovery/`.

## ✅ EXTRACTED — Safe to archive/delete from Bagrecovery

All valuable code has been copied to `wreckhunter2000-1/` proper:

| Destination | Source in Bagrecovery | Count |
|---|---|---|
| `pipelines/mag/` | `scripts/*.py` (mag/Erie/Huron pipeline) | 47 files |
| `pipelines/satellite/` | `scripts/*.py` (ECOSTRESS/SWOT/Sentinel/SAR) | 23 files |
| `pipelines/bag/` | `scripts/*.py` (BAG HDF5 scanner/viz) | 21 files |
| `ml/training/` | `scripts/*.py` + `scripts/forensic/*.py` | 12 files |
| `ml/inference/` | `scripts/*.py` + `scripts/forensic/*.py` | 16 files |
| `news_search/` | `scripts/chronicling_america_search.py` etc. | 4 files |
| `wrecks_api/` | `wrecks_api/` (app.py, stages/, etc.) | 7 files |
| `cesarops_core/` | `research optimization*.zip` (extracted) | 64 files |
| `cesarops_core/foundry_agent/` | `foundry_agent_scaffold/` | 5 files |
| `sentinel_hunt_src/` | `sentinel_hunt/src/*.rs` + `Cargo.toml` | 11 rs files |
| `models/erie/` | `models/erie/*.json` | 5 files |

⭐ **cesarops.com copies**: `pipelines/satellite/` and `cesarops_core/` are also designated for the public cesarops.com offering.

---

## 🗑️ SAFE TO DELETE from Bagrecovery

These are confirmed superseded, compiled artifacts, or pure noise:

- `frontendgpt/` — **OLDER** frontend, missing `AgentPanel.tsx`. The `tauri/src/components/` version is ahead.
- `legacy_spaghetti.zip` — 113MB, 26,416 .py files of old history. Leave zipped or delete.
- `bag_processor_rust.zip` (x2) — ~99MB each, compiled Rust artifacts.
- `venv/`, `venv_wh2k/`, `.venv/` — Python virtual environments.
- `node_modules/` — npm artifacts.
- `__pycache__/` directories — everywhere.
- Root-level junk scripts: `q.py`, `sleep.py`, `debug_*.py`, `replace_args*.py`, `patch_lib*.py`
- Ghost branch directories: `bfscanner branch/`, `Magwork Branch/`, `Master Branch/`

---

## ⚠️ DO NOT COMMIT / SECURITY

- `sentinel_hunt/earthdata_token.json` — **LIVE NASA Earthdata JWT** (uid: cesarops.com, exp: ~May 2026). Token added to `.gitignore`. Consider rotating at https://urs.earthdata.nasa.gov
- `incomming/` as a whole is now gitignored.

---

## 📦 LEAVE IN PLACE (data, not code)

These are output/data folders, not code. Leave them where they are:

- `adaptive_bg_*/` — adaptive background output tiles
- `mag_pipeline_output_*/` — mag pipeline run outputs
- `multisource_rank_*/` — multi-source ranking output
- `dist/WreckHunter2000_Clean/` — distribution archive

---

## 🔍 PARTIALLY INSPECTED — Review Before Archiving

- `recovered/` — 40+ XGBoost training_report JSON files. May contain unique basin-specific model configs. Scan before deleting.
- `scripts/_audit_*.py`, `_check_*.py` — DB audit scripts, may have unfinished enrichment logic.
- `bagfilework/` subdirs (nested deep) — Some appear to be old iteration history.
- `research optimization*.zip` / `rec-reorg branch-mostly cesarops.zip` — The main zip was extracted. The rec-reorg branch zip was NOT yet extracted. Check if it has unique content.

---

## 🏗️ NEW STRUCTURE IN wreckhunter2000-1

```
pipelines/
  mag/          ← Erie, Huron scanners, mag pipeline, KML, data ingestion
  satellite/    ← ⭐ ECOSTRESS, SWOT, Sentinel, SAR temporal (also cesarops.com)
  bag/          ← BAG HDF5 scanner, visualization, alignment
ml/
  training/     ← XGBoost/LightGBM/ResNet trainers, GPU training
  inference/    ← Inference scorer, deep-water detection, phase post-processing
news_search/    ← Chronicling America search, API probes
wrecks_api/     ← FastAPI app (app.py, stages/, envdb.py)
cesarops_core/  ← ⭐ Full SonarSniffer + CESARops SAR engine (also cesarops.com)
  src/cesarops/    ← bathymetry_mapper, sensor_fusion, sonar modules
  src/sonarsniffer/ ← engine, parser, target detection, ML pipeline
  analytics/       ← sar_metrics.py
  reports/         ← incident_report.py, pdf_export.py
  rosa_case/       ← ROSA case analysis scripts (18 files)
  foundry_agent/   ← Azure Foundry deployment scaffold
sentinel_hunt_src/ ← Rust: Sentinel satellite STAC search crate (src only)
models/
  wreck_classifier.pkl + meta.json  ← existing
  erie/                              ← Erie XGBoost models (4 basin models)
```
