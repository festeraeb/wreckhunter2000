from __future__ import annotations

import csv
import io
import math
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DROP_DIR = REPO / "magnetic_data" / "new data to digest"
EXTRACTED = DROP_DIR / "extracted_csv"
OUT_DIR = DROP_DIR / "digested_normalized"

LON_KEYS = ["longitude", "lon", "long", "x"]
LAT_KEYS = ["latitude", "lat", "y"]
MAG_KEYS = [
    "mag_anomaly", "mreslvl", "mreslc", "f_mtf", "srvmglev", "magraw", "maglev", "totmag"
]


def _to_float(v: str) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _pick_key(keys: list[str], cols: list[str]) -> str | None:
    col_l = [c.lower().strip() for c in cols]
    for k in keys:
        if k in col_l:
            return cols[col_l.index(k)]
    return None


def digest_csv_file(path: Path) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("/")]
    if not lines:
        return 0, "no_non_comment_rows"

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    if not reader.fieldnames:
        return 0, "no_header"

    lon_k = _pick_key(LON_KEYS, reader.fieldnames)
    lat_k = _pick_key(LAT_KEYS, reader.fieldnames)
    mag_k = _pick_key(MAG_KEYS, reader.fieldnames)
    if not lon_k or not lat_k or not mag_k:
        return 0, f"missing_keys lon={lon_k} lat={lat_k} mag={mag_k}"

    out_rows: list[tuple[float, float, float]] = []
    mag_vals: list[float] = []

    for row in reader:
        lon = _to_float(row.get(lon_k, ""))
        lat = _to_float(row.get(lat_k, ""))
        mag = _to_float(row.get(mag_k, ""))
        if lon is None or lat is None or mag is None:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        out_rows.append((lon, lat, mag))
        mag_vals.append(mag)

    # Heuristic for placeholder fields like MAGLEV/SRVMGLEV=500000.
    if out_rows and mag_k.lower() in {"maglev", "srvmglev"}:
        mean_mag = sum(mag_vals) / max(len(mag_vals), 1)
        if abs(mean_mag) > 300000:
            fallback = None
            for k in ["f_mtf", "mreslvl", "mreslc", "magraw", "totmag"]:
                if k in [c.lower() for c in reader.fieldnames]:
                    fallback = reader.fieldnames[[c.lower() for c in reader.fieldnames].index(k)]
                    break
            if fallback:
                out_rows = []
                reader2 = csv.DictReader(io.StringIO("\n".join(lines)))
                for row in reader2:
                    lon = _to_float(row.get(lon_k, ""))
                    lat = _to_float(row.get(lat_k, ""))
                    mag = _to_float(row.get(fallback, ""))
                    if lon is None or lat is None or mag is None:
                        continue
                    if abs(lat) > 90 or abs(lon) > 180:
                        continue
                    out_rows.append((lon, lat, mag))
                mag_k = fallback

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_digested.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["longitude", "latitude", "mag_anomaly"])
        for lon, lat, mag in out_rows:
            w.writerow([f"{lon:.7f}", f"{lat:.7f}", f"{mag:.6f}"])

    return len(out_rows), f"ok mag_field={mag_k}"


def digest_erie_xyz() -> tuple[int, str]:
    zpath = DROP_DIR / "ERIE.zip"
    if not zpath.exists():
        return 0, "missing_ERIE.zip"

    with zipfile.ZipFile(zpath, "r") as zf:
        xyz_name = next((n for n in zf.namelist() if n.lower().endswith(".xyz")), None)
        if not xyz_name:
            return 0, "no_xyz_in_zip"

        rows: list[tuple[float, float, float]] = []
        with zf.open(xyz_name, "r") as raw:
            for bline in raw:
                line = bline.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                # Observed layout includes lat at idx 5, lon idx 6, anomaly near last col.
                lat = _to_float(parts[5])
                lon = _to_float(parts[6])
                mag = _to_float(parts[-1])
                if lon is None or lat is None or mag is None:
                    continue
                if abs(lat) > 90 or abs(lon) > 180:
                    continue
                rows.append((lon, lat, mag))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "erie_xyz_digested.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["longitude", "latitude", "mag_anomaly"])
        for lon, lat, mag in rows:
            w.writerow([f"{lon:.7f}", f"{lat:.7f}", f"{mag:.6f}"])

    return len(rows), "ok mag_field=last_column"


def main() -> None:
    print("Digesting drop folder:", DROP_DIR)
    total = 0

    if EXTRACTED.exists():
        for p in sorted(EXTRACTED.rglob("*.csv")):
            n, status = digest_csv_file(p)
            total += n
            print(f"- {p.name}: {n} rows ({status})")
    else:
        print("- extracted_csv not found; run extraction first")

    n_xyz, status_xyz = digest_erie_xyz()
    total += n_xyz
    print(f"- ERIE.zip xyz: {n_xyz} rows ({status_xyz})")

    print("Total digested rows:", total)
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
