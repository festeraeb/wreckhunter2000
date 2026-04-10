"""Normalize NRCan HXYZ-format CSV files for the mag pipeline.

NRCan CSVs have `/`-prefixed comment lines, then a data header like:
  X,Y,TIME,MAGLEV,SRVMGLEV,MAGRAW,RALT,...

This script skips comment lines, renames X→longitude, Y→latitude, MAGRAW→mag_anomaly,
drops masked rows (MAGLEV==500000), then copies to magnetic_data/raw/local_mage_csv/
so the pipeline's `_list_local_mage_csvs()` picks them up automatically.

Usage: python scripts/normalize_nrcan_csvs.py
"""
from __future__ import annotations
import csv
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "magnetic_data" / "raw"
OUT_DIR = RAW / "local_mage_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG_DIRS = {
    "local_mage_csv","nrcan_ca_1km_rtf","nrcan_ca_1km_vd",
    "nrcan_ca_200m_rtf","nrcan_ca_200m_vd","usgs_namag","usgs_usmag"
}

def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalize_hxyz(src: pathlib.Path, dest: pathlib.Path) -> tuple[int, int]:
    """Normalize one HXYZ CSV. Returns (rows_written, rows_skipped)."""
    written = 0
    skipped = 0
    with src.open("r", encoding="utf-8", errors="ignore") as fin:
        # skip comment lines
        lines = [l for l in fin if not l.startswith("/")]
    if not lines:
        return 0, 0
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    has_magraw = "MAGRAW" in fieldnames
    has_xy = "X" in fieldnames and "Y" in fieldnames

    if not has_xy:
        print(f"  SKIP {src.name}: no X/Y columns (found: {fieldnames[:8]})")
        return 0, 0

    mag_field = "MAGRAW" if has_magraw else (
        next((f for f in fieldnames if any(t in f.upper() for t in ("MAGRAW","F_MTF","R_MTF","TMI","_MAG","ANOM"))), None)
    )
    if not mag_field:
        print(f"  SKIP {src.name}: no magnetic value column")
        return 0, 0

    out_header = ["longitude","latitude","mag_anomaly"]
    with dest.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(out_header)
        for row in reader:
            try:
                lon = float(row["X"])
                lat = float(row["Y"])
                mag = float(row[mag_field])
            except (ValueError, KeyError):
                skipped += 1
                continue
            # filter sentinel masked/invalid values
            if abs(mag) > 200000 or abs(lon) > 180 or abs(lat) > 90:
                skipped += 1
                continue
            writer.writerow([lon, lat, mag])
            written += 1
    return written, skipped


def main():
    total_written = 0
    for src_dir in sorted(RAW.iterdir()):
        if not src_dir.is_dir() or src_dir.name in ORIG_DIRS:
            continue
        for csv_path in src_dir.rglob("*.csv"):
            out_name = slug(csv_path.stem) + "_nrcan.csv"
            dest = OUT_DIR / out_name
            if dest.exists():
                print(f"  Already exists: {dest.name}")
                continue
            print(f"Normalizing {csv_path.parent.name[:35]}/{csv_path.name} -> {out_name}")
            w, s = normalize_hxyz(csv_path, dest)
            if w > 0:
                print(f"  {w:,} rows written, {s:,} skipped")
                total_written += w
            else:
                dest.unlink(missing_ok=True)
    print(f"\nDone. Total rows written: {total_written:,}")


if __name__ == "__main__":
    main()
