"""
WreckHunter 2000 — Raw Ghost Zoom  (MGD77T / FVD / STEEL_HULL Zoom)
=====================================================================
Extracts raw magnetometer flight-line pings for a tight 2-km block
around each Ghost target, grids to ~100 m, applies FVD sharpening,
runs the ResNet-18 model, and writes per-Ghost diagnostic JSON + KML.

Ghost #1: 42.2619, -80.8133   (Central Basin — priority)
Ghost #2: 42.5857, -80.0334   (Central Basin — NOTE: 362 m from abandoned well)

Usage:
  cd C:/Users/thomf/programming/Bagrecovery
  python -W ignore scripts/wh2k_raw_ghost_zoom.py \\
      --checkpoint wreck_hunting_ml/models/best_resnet18.pt \\
      --raw-csv    magnetic_data/raw/local_mage_csv/nrcan_OH_4039B.csv \\
      --output-dir wreck_hunting_ml/models \\
      --device     cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Ghost Target Registry ──────────────────────────────────────────────────

GHOSTS = [
    {
        "name": "Ghost-1-NE",
        "lat":  42.2619,
        "lon": -80.8133,
        "note": "Central Basin — priority target",
        "gas_well_flag": False,
    },
    {
        "name": "Ghost-2-SW",
        "lat":  42.5857,
        "lon": -80.0334,
        "note": "Probable gas well (362 m from Pembina abandoned dry hole, 30 wells <3 km)",
        "gas_well_flag": True,
    },
]

ZOOM_RADIUS_KM = 2.0         # ±2 km box around ghost centre
GRID_SPACING_M  = 100.0      # target grid cell size
CLASS_NAMES     = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}

# ── Utility ────────────────────────────────────────────────────────────────

def haversine(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def metres_to_deg(metres: float, lat_ref: float) -> tuple[float, float]:
    """Returns (dlat_deg, dlon_deg) for a given metre offset."""
    dlat = metres / 111_320.0
    dlon = metres / (111_320.0 * math.cos(math.radians(lat_ref)))
    return dlat, dlon


# ── Raw CSV loader ─────────────────────────────────────────────────────────

def _detect_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    """
    Auto-detect lon / lat / mag column names from header.
    Handles NRCan MGD77T, GSC, and common NOAA MGD77 variants.
    """
    fl = [f.upper() for f in fieldnames]

    def _find(*candidates) -> str:
        for c in candidates:
            if c in fl:
                return fieldnames[fl.index(c)]
        raise KeyError(f"None of {candidates} found in CSV header: {fieldnames}")

    lon = _find("LON", "LONG", "LONGITUDE", "SUR_LONG83", "X", "EASTING")
    lat = _find("LAT", "LATITUDE",  "SUR_LAT83",  "Y", "NORTHING")
    mag = _find("TMF", "MAG", "RESIDMAG", "RESIDUAL", "TMAGF", "CORR_MAG",
                "MFIELD", "MGNT", "MAGFIELD", "MAG_FIELD")
    return lon, lat, mag


def load_raw_pings(
    csv_path: Path,
    centre_lat: float,
    centre_lon: float,
    radius_km: float = ZOOM_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads raw CSV and returns (lons, lats, mags) for points within radius_km.
    """
    dlat, dlon = metres_to_deg(radius_km * 1000, centre_lat)
    lat_min = centre_lat - dlat;  lat_max = centre_lat + dlat
    lon_min = centre_lon - dlon;  lon_max = centre_lon + dlon

    lons, lats, mags = [], [], []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        col_lon, col_lat, col_mag = _detect_columns(reader.fieldnames or [])
        for row in reader:
            try:
                lat = float(row[col_lat])
                lon = float(row[col_lon])
                mag = float(row[col_mag])
            except (ValueError, KeyError):
                continue
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                lons.append(lon)
                lats.append(lat)
                mags.append(mag)

    return np.array(lons), np.array(lats), np.array(mags)


# ── Gridding ───────────────────────────────────────────────────────────────

def grid_pings(
    lons: np.ndarray,
    lats: np.ndarray,
    mags: np.ndarray,
    centre_lat: float,
    centre_lon: float,
    radius_km: float = ZOOM_RADIUS_KM,
    spacing_m: float = GRID_SPACING_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates scattered pings onto a regular grid using linear method.
    Returns (grid_lon, grid_lat, grid_mag) — all 2-D arrays.
    """
    n_cells = int(2 * radius_km * 1000 / spacing_m)
    dlat, dlon = metres_to_deg(radius_km * 1000, centre_lat)

    lon_vec = np.linspace(centre_lon - dlon, centre_lon + dlon, n_cells)
    lat_vec = np.linspace(centre_lat - dlat, centre_lat + dlat, n_cells)
    glon, glat = np.meshgrid(lon_vec, lat_vec)

    # Convert to local metres for better interpolation conditioning
    ref_lat = centre_lat
    x_pts = (lons - centre_lon) * 111_320.0 * math.cos(math.radians(ref_lat))
    y_pts = (lats - centre_lat) * 111_320.0
    x_grid = (glon - centre_lon) * 111_320.0 * math.cos(math.radians(ref_lat))
    y_grid = (glat - centre_lat) * 111_320.0

    grid_mag = griddata(
        np.column_stack([x_pts, y_pts]),
        mags,
        np.column_stack([x_grid.ravel(), y_grid.ravel()]),
        method="linear",
    ).reshape(glon.shape).astype(np.float32)

    # Fill NaN edges with nearest-neighbour
    nan_mask = np.isnan(grid_mag)
    if nan_mask.any():
        grid_mag_nn = griddata(
            np.column_stack([x_pts, y_pts]),
            mags,
            np.column_stack([x_grid.ravel(), y_grid.ravel()]),
            method="nearest",
        ).reshape(glon.shape).astype(np.float32)
        grid_mag[nan_mask] = grid_mag_nn[nan_mask]

    return glon, glat, grid_mag


# ── FVD sharpening (reuses exact logic from discovery_report_v2) ──────────

def _fvd(g: np.ndarray) -> np.ndarray:
    """First Vertical Derivative via Fourier |k| filter."""
    f = np.fft.fft2(g.astype(np.float64))
    ny, nx = g.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k  = np.sqrt(kx ** 2 + ky ** 2) * 2 * math.pi
    return np.real(np.fft.ifft2(f * k)).astype(np.float32)


def _tilt(g: np.ndarray) -> np.ndarray:
    """Tilt Derivative."""
    dx = np.gradient(g.astype(np.float64), axis=1)
    dy = np.gradient(g.astype(np.float64), axis=0)
    thdr = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    return np.arctan2(_fvd(g), thdr).astype(np.float32)


def asymmetry_score(nss_block: np.ndarray) -> float:
    h, w = nss_block.shape
    gx = np.gradient(nss_block.astype(np.float64), axis=1)
    gy = np.gradient(nss_block.astype(np.float64), axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
    yy, xx = np.mgrid[0:h, 0:w]
    cy = np.sum(mag * yy) / np.sum(mag)
    cx = np.sum(mag * xx) / np.sum(mag)
    dy = abs(cy - (h - 1) / 2)
    dx = abs(cx - (w - 1) / 2)
    offset = math.sqrt(dx ** 2 + dy ** 2)
    max_offset = math.sqrt(((h - 1) / 2) ** 2 + ((w - 1) / 2) ** 2) + 1e-10
    return float(min(offset / max_offset, 1.0))


# ── Model inference ────────────────────────────────────────────────────────

def run_model_on_grid(
    grid_mag: np.ndarray,
    checkpoint_path: Path,
    device: str = "cpu",
    radius: int = 2,
) -> list[dict]:
    """
    Slides a 5×5 (2*radius+1 each side) neighbourhood across the 100-m grid,
    bicubic-upsamples to 224×224, runs model, records STEEL_HULL hits.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models import resnet18
    except ImportError as e:
        logger.error("torch not available: %s", e)
        return []

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ns   = ckpt.get("norm_stats", {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]})
    ch_mean = torch.tensor(ns["mean"], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.tensor(ns["std"],  dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.where(ch_std < 1e-8, torch.ones_like(ch_std), ch_std)

    model = resnet18(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Build 3-channel grid: [NSS, FVD, Tilt]
    nss  = grid_mag.copy()
    vdr  = _fvd(nss)
    tilt = _tilt(nss)

    # Normalise each channel to [0,1] locally
    def norm01(a: np.ndarray) -> np.ndarray:
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-10)

    grid3 = np.stack([norm01(nss), norm01(vdr), norm01(tilt)], axis=0)  # (3, H, W)

    H, W = grid_mag.shape
    ws   = 2 * radius + 1
    hits = []

    with torch.no_grad():
        for r in range(radius, H - radius):
            for c in range(radius, W - radius):
                block = grid3[:, r - radius:r + radius + 1, c - radius:c + radius + 1]
                # 5×5 patch → (1, 3, 5, 5) → bicubic upsample → (1, 3, 224, 224)
                t = torch.tensor(block[None], dtype=torch.float32, device=device)
                t = F.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
                t = (t - ch_mean) / ch_std
                logits = model(t)
                probs  = torch.softmax(logits, dim=1).squeeze()
                pred   = int(probs.argmax().item())
                conf   = float(probs.max().item())
                asym   = asymmetry_score(block[0])  # use NSS channel

                hits.append({
                    "row": r, "col": c,
                    "pred_class": pred,
                    "pred_label": CLASS_NAMES[pred],
                    "conf": round(conf, 4),
                    "asym": round(asym, 4),
                    "fvd_peak": round(float(vdr[r - radius:r + radius + 1, c - radius:c + radius + 1].max()), 2),
                    "nss_range": round(float(nss[r - radius:r + radius + 1, c - radius:c + radius + 1].max()
                                             - nss[r - radius:r + radius + 1, c - radius:c + radius + 1].min()), 2),
                })

    return hits


# ── KML writer ─────────────────────────────────────────────────────────────

def write_kml(results: list[dict], out_path: Path) -> None:
    placemarks = []
    for r in results:
        if r["pred_class"] != 1:
            continue
        color = "ff0000ff" if r["asym"] >= 0.20 else "ff00ff00"
        desc = (f"Conf={r['conf']:.3f}  Asym={r['asym']:.3f}\n"
                f"FVD_peak={r['fvd_peak']} nT\n"
                f"NSS_range={r['nss_range']} nT")
        placemarks.append(f"""    <Placemark>
      <name>STEEL_HULL ({r['conf']:.2f})</name>
      <description>{desc}</description>
      <Style><IconStyle><color>{color}</color><scale>0.8</scale></IconStyle></Style>
      <Point><coordinates>{r['lon']},{r['lat']},0</coordinates></Point>
    </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Raw Ghost Zoom — STEEL_HULL Hits</name>
{"".join(placemarks)}
  </Document>
</kml>"""
    out_path.write_text(kml, encoding="utf-8")
    logger.info("KML → %s  (%d STEEL_HULL placemarks)", out_path, len(placemarks))


# ── Main ───────────────────────────────────────────────────────────────────

def zoom_ghost(
    ghost: dict,
    csv_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    name = ghost["name"]
    lat  = ghost["lat"]
    lon  = ghost["lon"]

    logger.info("=" * 65)
    logger.info("GHOST ZOOM: %s  lat=%.4f  lon=%.4f", name, lat, lon)
    if ghost.get("gas_well_flag"):
        logger.warning("  ⚠ GAS WELL FLAG: %s", ghost["note"])
    logger.info("  Extracting raw pings ±%.1f km …", ZOOM_RADIUS_KM)

    # 1. Extract raw pings
    lons, lats, mags = load_raw_pings(csv_path, lat, lon, ZOOM_RADIUS_KM)
    if len(mags) < 10:
        logger.warning("  Only %d raw pings in bbox — skipping (no survey coverage)", len(mags))
        return {"ghost": name, "error": "insufficient_raw_data", "raw_pings": len(mags)}

    logger.info("  Raw pings: %d   mag range: %.1f – %.1f nT   (span=%.1f nT)",
                len(mags), mags.min(), mags.max(), mags.max() - mags.min())

    # 2. Grid to 100 m
    logger.info("  Gridding to %.0f m resolution …", GRID_SPACING_M)
    glon, glat, grid_mag = grid_pings(lons, lats, mags, lat, lon)
    logger.info("  Grid shape: %s", grid_mag.shape)

    # 3. Run FVD statistics
    fvd_grid = _fvd(grid_mag)
    logger.info("  FVD range: %.2f – %.2f nT/m   (95th pct = %.2f)",
                fvd_grid.min(), fvd_grid.max(), float(np.percentile(np.abs(fvd_grid), 95)))

    # 4. Run model
    logger.info("  Running ResNet-18 on 100-m grid …")
    cell_hits = run_model_on_grid(grid_mag, checkpoint_path, device)

    # Attach lat/lon to each hit
    H, W = grid_mag.shape
    for h in cell_hits:
        r, c = h["row"], h["col"]
        h["lat"] = round(float(glat[r, c]), 6)
        h["lon"] = round(float(glon[r, c]), 6)

    # Summary
    steel_hits = [h for h in cell_hits if h["pred_class"] == 1 and h["conf"] >= 0.35]
    top_steel  = sorted(steel_hits, key=lambda x: -x["conf"])[:10]

    logger.info("  STEEL_HULL hits (conf≥0.35): %d   top-1 conf=%.3f  asym=%.3f",
                len(steel_hits),
                top_steel[0]["conf"]   if top_steel else 0,
                top_steel[0]["asym"]   if top_steel else 0)

    result = {
        "ghost": name,
        "lat": lat,
        "lon": lon,
        "gas_well_flag": ghost["gas_well_flag"],
        "note": ghost["note"],
        "raw_pings": int(len(mags)),
        "raw_nT_range": round(float(mags.max() - mags.min()), 1),
        "raw_nT_max": round(float(mags.max()), 1),
        "raw_nT_min": round(float(mags.min()), 1),
        "grid_shape": list(grid_mag.shape),
        "grid_spacing_m": GRID_SPACING_M,
        "fvd_95pct": round(float(np.percentile(np.abs(fvd_grid), 95)), 3),
        "steel_hull_hits": len(steel_hits),
        "top_hits": top_steel[:5],
    }

    # Save JSON
    jout = output_dir / f"ghost_zoom_{name}.json"
    jout.write_text(json.dumps(result, indent=2))
    logger.info("  JSON → %s", jout)

    # Save KML
    kout = output_dir / f"ghost_zoom_{name}.kml"
    write_kml(cell_hits, kout)

    return result


def main(argv: Optional[list[str]] = None) -> None:
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    parser = argparse.ArgumentParser(description="Ghost raw zoom + FVD scan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-csv",    required=True,
                        help="nrcan_OH_4039B.csv (raw flight-line pings)")
    parser.add_argument("--output-dir", default="wreck_hunting_ml/models")
    parser.add_argument("--device",     default=_default_device(), choices=["cpu", "cuda"])
    args = parser.parse_args(argv)

    csv_path  = Path(args.raw_csv)
    ckpt_path = Path(args.checkpoint)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ghost in GHOSTS:
        r = zoom_ghost(ghost, csv_path, ckpt_path, out_dir, args.device)
        all_results.append(r)

    summary_path = out_dir / "ghost_zoom_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Summary → %s", summary_path)

    # Print headline
    print("\n" + "=" * 65)
    print("GHOST RAW ZOOM — SUMMARY")
    for r in all_results:
        if "error" in r:
            print(f"  {r['ghost']:20s}  ERROR: {r['error']}")
            continue
        hits = r["steel_hull_hits"]
        rng  = r["raw_nT_range"]
        flag = " ⚠ GAS WELL" if r["gas_well_flag"] else ""
        print(f"  {r['ghost']:20s}  raw={r['raw_pings']:5d}pts  nT_range={rng:7.1f}  STEEL_HULL_hits={hits}{flag}")
    print("=" * 65)


if __name__ == "__main__":
    main()
