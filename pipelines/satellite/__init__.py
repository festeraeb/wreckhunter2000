"""
pipelines.satellite — WreckHunter2000 Satellite Hunting Submodule
=================================================================
Self-contained satellite imagery acquisition and wreck-targeting pipeline
for Great Lakes shipwreck detection.

This package is the source of truth for satellite hunting code.
It can be copied intact to:
  - CESARops SAR build (as a submodule / pipelines/satellite/)
  - Any standalone satellite scanner tool

Acquisition tools (call from CLI or import as functions):
  cmr_search              NASA CMR granule catalog query (HLS, SAR, SWOT, ICESat-2)
  universal_downloader    Multi-source downloader (ASF HyP3, Copernicus, PO.DAAC, USGS, HLS)
  batch_download_manager  Multi-node / swarm downloader — distributes lake×year tasks across nodes
  nasa_earthdata_client   Low-level NASA Earthdata auth helper

Wreck-targeting pipeline (wh2k_* scripts):
  wh2k_sentinel_wreck_targeting   Season-aware optical wreck signal extractor (z-score vs background)
  wh2k_chip_extractor             600m radius chip cutter centred on wreck coordinates
  wh2k_sentinel_cpu               CPU-optimised Sentinel-2 band processor
  wh2k_sentinel_optical_poc       Optical proof-of-concept runner
  wh2k_ab_attenuation             Anomalous bottom attenuation detector
  wh2k_raw_ghost_zoom             Raw ghost / shadow zoom detector
  wh2k_synthetic_tiles            Synthetic tile generator for ML training (Erie)
  wh2k_synthetic_tiles_huron      Synthetic tile generator (Huron)
  wh2k_synthetic_tiles_v2         Synthetic tile generator v2 (improved augmentation)
  wh2k_build_satellite_bundle_manifest  Build ingest manifests for bundle packs

SAR / temporal:
  sar_temporal_persistence        Multi-pass SAR anomaly persistence scorer
  apply_opera_dswx                Apply OPERA DSWx surface water mask to scenes

Environmental / context:
  fetch_ecostress_data    ECOSTRESS land surface temperature fetch
  fetch_swot_data         SWOT SSH fetch via PO.DAAC
  buoy_analog             Buoy-based scene analog selector
  historical_drift        Historical drift / current context for scene windows

Utilities:
  probe_sources           Health-check all configured satellite data sources
  run_ingest_new          Run new-data ingest cycle
  summarize_ingest        Print summary of last ingest run
  check_sentinel_deps     Verify Sentinel/sentinelsat dependencies are installed
  install_sentinel_deps   Auto-install Sentinel processing dependencies
  nasa_fusion_test        Multi-modal NASA data fusion smoke test
"""

# Acquisition layer — import-safe (stdlib + requests only)
from pathlib import Path as _Path

# Resolve acquisition scripts that may live at repo root
_REPO = _Path(__file__).resolve().parents[2]
_SCANNER = _Path(__file__).resolve().parent

__version__ = "0.1.0"
__all__ = [
    "cmr_search",
    "universal_downloader",
    "batch_download_manager",
    "nasa_earthdata_client",
    "wh2k_sentinel_wreck_targeting",
    "wh2k_chip_extractor",
    "sar_temporal_persistence",
]
