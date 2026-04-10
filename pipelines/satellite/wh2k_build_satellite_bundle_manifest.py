"""
Build consolidated Erie+Huron satellite data manifest.

Outputs:
  magnetic_data/tier_4_satellite/erie_huron_bundle/satellite_bundle_manifest.json

Includes:
- EMAG2 Erie/Huron subset files already present in repo
- Swarm fetch catalogs/status from wh2k_ncei_fetch
- Sentinel band file inventory (if any local files exist)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EMAG2_DIR = REPO / "magnetic_data" / "tier_3_aero_regional" / "emag2"
TIER4_DIR = REPO / "magnetic_data" / "tier_4_satellite"
RAW_TRACKLINE = REPO / "magnetic_data" / "raw" / "ncei_trackline"
OUT_DIR = TIER4_DIR / "erie_huron_bundle"
OUT_FILE = OUT_DIR / "satellite_bundle_manifest.json"


def _file_entry(p: Path) -> dict:
    return {
        "path": str(p.relative_to(REPO)).replace("\\", "/"),
        "size_bytes": p.stat().st_size,
    }


def collect_emag2() -> list[dict]:
    entries: list[dict] = []
    if not EMAG2_DIR.exists():
        return entries

    # Erie + Huron-specific subsets and supporting csvs.
    patterns = [
        "local_EMAG2_bessemer_erie_subset_*.tif",
        "local_EMAG2_huron_corridor_subset_*.tif",
        "EMAG2_bessemer_erie_subset.csv",
        "EMAG2_huron_corridor_subset.csv",
    ]
    for pat in patterns:
        for p in sorted(EMAG2_DIR.glob(pat)):
            entries.append(_file_entry(p))

    return entries


def collect_tier4_local() -> list[dict]:
    entries: list[dict] = []
    local_dir = TIER4_DIR / "local"
    if not local_dir.exists():
        return entries
    for p in sorted(local_dir.glob("*")):
        if p.is_file():
            entries.append(_file_entry(p))
    return entries


def collect_swarm_status() -> dict:
    status: dict = {
        "erie_catalog": None,
        "huron_catalog": None,
        "swarm_instructions": None,
        "note": "Swarm requires valid VirES token configured in viresclient.",
    }

    erie_cat = RAW_TRACKLINE / "erie_fetch_catalog.json"
    huron_cat = RAW_TRACKLINE / "huron_fetch_catalog.json"
    swarm_note = REPO / "magnetic_data" / "raw" / "swarm_l2" / "swarm_download_instructions.json"

    if erie_cat.exists():
        status["erie_catalog"] = json.loads(erie_cat.read_text(encoding="utf-8"))
    if huron_cat.exists():
        status["huron_catalog"] = json.loads(huron_cat.read_text(encoding="utf-8"))
    if swarm_note.exists():
        status["swarm_instructions"] = json.loads(swarm_note.read_text(encoding="utf-8"))

    return status


def collect_sentinel_band_files() -> list[dict]:
    entries: list[dict] = []

    # Typical Sentinel-2 band naming fragments.
    band_fragments = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    exts = {".tif", ".tiff", ".jp2"}

    # Limit to likely data areas to avoid scanning build artifacts.
    roots = [
        REPO / "magnetic_data",
        REPO / "bagfilework" / "data",
    ]

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            name_up = p.name.upper()
            if any(frag in name_up for frag in band_fragments) and ("S2" in name_up or "SENTINEL" in name_up):
                entries.append(_file_entry(p))

    entries.sort(key=lambda x: x["path"])
    return entries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emag2 = collect_emag2()
    tier4_local = collect_tier4_local()
    swarm = collect_swarm_status()
    sentinel_bands = collect_sentinel_band_files()

    manifest = {
        "bundle": "erie_huron_satellite",
        "generated_by": "scripts/wh2k_build_satellite_bundle_manifest.py",
        "emag2_files": emag2,
        "tier4_local_files": tier4_local,
        "swarm_status": swarm,
        "sentinel_band_files": sentinel_bands,
        "sentinel_note": (
            "No Sentinel band rasters found locally if sentinel_band_files is empty. "
            "Acquire S2 L2A bands for Erie/Huron AOIs and place under magnetic_data/tier_4_satellite/sentinel_bands/."
        ),
    }

    OUT_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_FILE}")
    print(f"EMAG2 files: {len(emag2)}")
    print(f"Tier4 local files: {len(tier4_local)}")
    print(f"Sentinel band files: {len(sentinel_bands)}")


if __name__ == "__main__":
    main()
