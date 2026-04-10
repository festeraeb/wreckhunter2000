"""Run stage_ingest for the Great Lakes bbox on all local CSV sources.

Usage: python scripts/run_ingest_new.py [--bbox lonmin latmin lonmax latmax]
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.mag_data_pipeline import stage_ingest, MAG_DATA_DIR, MAG_DATA_GRIDS, DEFAULT_BBOX

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bbox", nargs=4, type=float, metavar=("lonmin","latmin","lonmax","latmax"),
                   default=list(DEFAULT_BBOX))
    args=p.parse_args()
    bbox=tuple(args.bbox)
    print(f"Ingesting with bbox={bbox}")
    sp=stage_ingest(
        data_dir=MAG_DATA_DIR,
        grids_dir=MAG_DATA_GRIDS,
        downloaded={},  # empty — local CSVs picked up automatically
        bbox=bbox,
        progress_callback=lambda m: print(" ",m),
    )
    print(f"\nStatus: {sp.status}")
    print(f"Message: {sp.message}")
    print(json.dumps(sp.details, indent=2))

if __name__=="__main__":
    main()
