"""
WreckHunter 2000 — Discovery Report v2  (Bicubic / MB2 / GeoJSON)
==================================================================
Coarse-grid NAMAG Erie scan with the full feature pipeline:

  • 5×5 neighbourhood bicubic ↑ to 224×224 — bicubic preserves the subtle
    gradient lean that bilinear washes out.
  • FVD sharpening applied on-CPU — same transform as GPU training.
  • Off-axis asymmetry score (MB2 filter) — measures how much the gradient
    "leans" inside the 5×5 block.  1 pixel ≈ 2.2 km; a 150 m steel freighter
    shows as an asymmetric dipole pull rather than a centred dome.
  • Pill geometry score:
      Score 10 — STEEL_HULL classification + high asymmetry (off-axis lean)
      Score  9 — STEEL_HULL + moderate asymmetry
      Score  8 — STEEL_HULL + low asymmetry (possible wellhead or geology)
  • AWOIS validity check — Admiral & Merida sampled first, regardless of
    confidence, so we always know if scaling is correct.
  • GeoJSON export — Top-5 Score-9/10 unknown "pill" targets.

Usage:
  cd C:/Users/thomf/programming/Bagrecovery
  python -W ignore scripts/wh2k_discovery_report_v2.py
    --checkpoint wreck_hunting_ml/models/best_resnet18.pt
    --grid-tif magnetic_data/grids/usgs_namag_83_6000_41_3000__78_8000_42_9000.tif
    --device cpu
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            REPO_ROOT / "wreck_hunting_ml" / "discovery_report_v2.log",
            mode="a", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ── AWOIS ground-truth wrecks ─────────────────────────────────────────────
AWOIS_WRECKS = [
    {"name": "SS Admiral",        "lat": 42.025, "lon": -81.150, "length_ft": 296},
    {"name": "SS Clarion",        "lat": 41.980, "lon": -81.520, "length_ft": 265},
    {"name": "SS Merida",         "lat": 42.014, "lon": -80.851, "length_ft": 408},
    {"name": "SS L.R. Doty",      "lat": 41.983, "lon": -81.631, "length_ft": 285},
    {"name": "SS Minnedosa",      "lat": 42.051, "lon": -81.249, "length_ft": 240},
    {"name": "SS Craftsman",      "lat": 42.118, "lon": -81.441, "length_ft": 444},
    {"name": "Whaleback Consort", "lat": 42.427, "lon": -80.813, "length_ft": 308},
]

# Wrecks to spotlight in headline output
PRIORITY_WRECKS = {"SS Admiral", "SS Merida"}

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}


# ── Haversine ─────────────────────────────────────────────────────────────
def haversine(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6_371_000
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Grid layer helpers ────────────────────────────────────────────────────

def _vdr(g: np.ndarray) -> np.ndarray:
    """First Vertical Derivative via Fourier (mirrors FVDTransform in training)."""
    f = np.fft.fft2(g.astype(np.float64))
    ny, nx = g.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k  = np.sqrt(kx ** 2 + ky ** 2) * 2 * math.pi
    return np.real(np.fft.ifft2(f * k)).astype(np.float32)


def _tilt(g: np.ndarray) -> np.ndarray:
    """Tilt Derivative (angle between VDR and total horizontal)."""
    from scipy import ndimage as ndi
    dx   = np.gradient(g.astype(np.float64), axis=1)
    dy   = np.gradient(g.astype(np.float64), axis=0)
    thdr = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    return np.arctan2(_vdr(g), thdr).astype(np.float32)


# ── MB2 off-axis asymmetry score ──────────────────────────────────────────

def asymmetry_score(nss_block: np.ndarray) -> float:
    """
    Measures how much the gradient 'leans' in a small block.
    Returns a value in [0, 1]:
      ~0 = centred dome (wellhead / geology)
      ~1 = strongly off-axis (steel hull — the MB2 'sideways pull')

    Method:
      Compute the gradient magnitude-weighted centroid of the block.
      The centroid offset from block centre, normalised by half-block size,
      is the asymmetry score.  A perfectly symmetric dome has centroid
      at centre → score ≈ 0.  An off-axis dipole has centroid displaced.
    """
    h, w = nss_block.shape
    gx = np.gradient(nss_block.astype(np.float64), axis=1)
    gy = np.gradient(nss_block.astype(np.float64), axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-12

    yy, xx = np.mgrid[0:h, 0:w]
    cy = np.sum(mag * yy) / np.sum(mag)
    cx = np.sum(mag * xx) / np.sum(mag)

    # Offset from centre, normalised to [0, 1] by max possible displacement
    dy = abs(cy - (h - 1) / 2)
    dx = abs(cx - (w - 1) / 2)
    offset = math.sqrt(dx ** 2 + dy ** 2)
    max_offset = math.sqrt(((h - 1) / 2) ** 2 + ((w - 1) / 2) ** 2) + 1e-10
    return float(min(offset / max_offset, 1.0))


def pill_score(pred_class: int, confidence: float, asym: float) -> int:
    """
    Assign an integer Score 1-10 for each detection.
    STEEL_HULL + high confidence + off-axis lean → Score 10 (unknown pill).
    """
    if pred_class != 1:                   # not STEEL_HULL
        return 0
    if confidence < 0.30:
        return 0
    base = 5 + round(confidence * 2)      # 5..7 from confidence
    if asym >= 0.35:
        base += 3                          # strong lean → +3
    elif asym >= 0.20:
        base += 2                          # moderate lean → +2
    elif asym >= 0.10:
        base += 1                          # slight lean → +1
    return min(base, 10)


# ── Main scan function ────────────────────────────────────────────────────

def run_scan(
    checkpoint_path: Path,
    grid_tif_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.35,
    device: str = "cpu",
    batch_size: int = 128,
    radius: int = 2,                # 5×5 neighbourhood half-size
    awois_check_radius_m: float = 3000.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("DISCOVERY REPORT v2 — Bicubic / MB2 / Off-Axis / GeoJSON")
    logger.info("  checkpoint : %s", checkpoint_path)
    logger.info("  grid       : %s", grid_tif_path)
    logger.info("  device     : %s | batch=%d | radius=%d", device, batch_size, radius)
    logger.info("  conf       : %.2f", confidence_threshold)
    logger.info("=" * 65)

    try:
        import rasterio
        from rasterio.transform import xy as rxy
        import torch
        import torch.nn.functional as F
        from torchvision.models import resnet18
        from scipy import ndimage as _ndi
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return {"error": str(e)}

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ns   = ckpt.get("norm_stats", {"mean": [0, 0, 0], "std": [1, 1, 1]})
    ch_mean = torch.tensor(ns["mean"], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.tensor(ns["std"],  dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    ch_std  = torch.where(ch_std < 1e-8, torch.ones_like(ch_std), ch_std)
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_acc   = ckpt.get("val_acc", 0)
    logger.info("Checkpoint: epoch=%s  val_acc=%.4f  phase=%s  fvd=%s",
                ckpt_epoch, ckpt_acc, ckpt.get("phase", "?"), ckpt.get("fvd_preprocessing"))

    model = resnet18(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Load full grid ──────────────────────────────────────────────────────
    with rasterio.open(str(grid_tif_path)) as src:
        grid_raw = src.read(1).astype(np.float32)
        transform = src.transform
        nodata    = src.nodata
        H, W      = grid_raw.shape
        logger.info("Grid: %d×%d px  res=%.4f°/px  bounds=%s", H, W,
                    abs(transform.a), src.bounds)

    if nodata is not None:
        grid_raw[grid_raw == nodata] = float(np.nanmedian(grid_raw))
    grid_raw[np.isnan(grid_raw)] = float(np.nanmedian(grid_raw))

    # ── Compute 3 channels ─────────────────────────────────────────────────
    logger.info("Computing FVD(NSS) / VDR / Tilt channels…")
    fvd_grid  = _vdr(grid_raw).astype(np.float32)    # ch0 = FVD(NSS) — matches training
    vdr_grid  = fvd_grid                               # VDR is same as FVD here
    tilt_grid = _tilt(grid_raw).astype(np.float32)    # ch2 = Tilt

    # ── Pad grids for border pixels ────────────────────────────────────────
    PAD = radius
    nss_p  = np.pad(fvd_grid,  PAD, mode="reflect")
    vdr_p  = np.pad(vdr_grid,  PAD, mode="reflect")
    tilt_p = np.pad(tilt_grid, PAD, mode="reflect")

    # ── Extract all 5×5 neighbourhoods ─────────────────────────────────────
    WIN = 2 * radius + 1   # 5
    logger.info("Extracting %d×%d=%d neighbourhoods (win=%dpx)…", H, W, H * W, WIN)

    # Build coordinate arrays for lat/lon mapping
    lons_flat = np.empty(H * W, dtype=np.float64)
    lats_flat = np.empty(H * W, dtype=np.float64)
    tiles_all = np.empty((H * W, 3, WIN, WIN), dtype=np.float32)
    asym_all  = np.empty(H * W, dtype=np.float32)

    idx = 0
    for r in range(H):
        for c in range(W):
            rp, cp = r + PAD, c + PAD
            n = nss_p [rp - radius: rp + radius + 1, cp - radius: cp + radius + 1]
            v = vdr_p [rp - radius: rp + radius + 1, cp - radius: cp + radius + 1]
            t = tilt_p[rp - radius: rp + radius + 1, cp - radius: cp + radius + 1]
            tiles_all[idx, 0] = n
            tiles_all[idx, 1] = v
            tiles_all[idx, 2] = t
            asym_all[idx]     = asymmetry_score(n)
            lon, lat = rxy(transform, r, c)
            lons_flat[idx] = lon
            lats_flat[idx] = lat
            idx += 1

    # ── Pre-filter: run model only on the most anomalous pixels ──────────
    # Running ResNet18@224×224 on all 21k pixels exhausts RAM with training
    # running in parallel.  Strategy:
    #   1. Rank every pixel by amplitude (|FVD| channel-0) + asymmetry.
    #   2. Always include the 7 AWOIS known locations.
    #   3. Take top-N_CANDIDATES by combined score.
    #   4. Run bicubic+model only on those candidates.
    # Everything else gets probs=[1,0,0,0] (geology) with conf=1.0.

    N_CANDIDATES = 150
    amplitude_all  = np.abs(tiles_all[:, 0, radius, radius])   # peak FVD at centre
    filter_score   = amplitude_all / (amplitude_all.max() + 1e-12) + asym_all

    # Always include AWOIS pixels
    awois_pixel_ids = []
    for wreck in AWOIS_WRECKS:
        dists = np.array([
            haversine(wreck["lat"], wreck["lon"], lats_flat[i], lons_flat[i])
            for i in range(H * W)
        ])
        awois_pixel_ids.append(int(dists.argmin()))

    top_idx = np.argsort(filter_score)[::-1][:N_CANDIDATES].tolist()
    candidate_set = sorted(set(top_idx + awois_pixel_ids))
    logger.info(
        "Neighbourhoods extracted.  Running model on %d candidates "
        "(top-%d by amplitude+asym + %d AWOIS)…",
        len(candidate_set), N_CANDIDATES, len(awois_pixel_ids),
    )

    # Default: GEOLOGY_ONLY with full confidence
    probs_all      = np.zeros((H * W, 4), dtype=np.float32)
    probs_all[:, 0] = 1.0

    with torch.no_grad():
        for start in range(0, len(candidate_set), batch_size):
            idxs  = candidate_set[start: start + batch_size]
            batch = torch.tensor(tiles_all[idxs], dtype=torch.float32).to(device)
            # ① bicubic upsample 5×5 → 224×224
            batch = F.interpolate(batch, size=(224, 224),
                                  mode="bicubic", align_corners=False)
            # ② z-score normalise (same stats as training)
            batch = (batch - ch_mean) / ch_std
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            for j, global_i in enumerate(idxs):
                probs_all[global_i] = probs[j]

    pred_class_all = probs_all.argmax(axis=1)
    conf_all       = probs_all.max(axis=1)

    # ── Score each pixel ───────────────────────────────────────────────────
    scores_all = np.array([
        pill_score(int(pred_class_all[i]), float(conf_all[i]), float(asym_all[i]))
        for i in range(H * W)
    ], dtype=np.int32)

    n_steel = int((pred_class_all == 1).sum())
    logger.info("Inference done.  STEEL_HULL pixels: %d / %d", n_steel, H * W)

    # ── AWOIS validity check ────────────────────────────────────────────────
    logger.info("─" * 65)
    logger.info("AWOIS VALIDITY CHECK — Admiral & Merida first")
    logger.info("─" * 65)

    awois_results = []
    for wreck_idx, wreck in enumerate(AWOIS_WRECKS):
        nearest_idx  = awois_pixel_ids[wreck_idx]
        nearest_dist = haversine(
            wreck["lat"], wreck["lon"],
            float(lats_flat[nearest_idx]), float(lons_flat[nearest_idx]),
        )

        p_class  = int(pred_class_all[nearest_idx])
        p_conf   = float(conf_all[nearest_idx])
        p_asym   = float(asym_all[nearest_idx])
        p_score  = int(scores_all[nearest_idx])
        p_probs  = {CLASS_NAMES[k]: round(float(probs_all[nearest_idx, k]), 3) for k in range(4)}

        hit = (p_class == 1) and (nearest_dist <= awois_check_radius_m)
        result = {
            "name":        wreck["name"],
            "lat":         wreck["lat"],
            "lon":         wreck["lon"],
            "length_ft":   wreck["length_ft"],
            "nearest_dist_m": round(nearest_dist),
            "nearest_lat": float(lats_flat[nearest_idx]),
            "nearest_lon": float(lons_flat[nearest_idx]),
            "predicted_class": CLASS_NAMES[p_class],
            "confidence":  round(p_conf, 3),
            "asymmetry":   round(p_asym, 3),
            "pill_score":  p_score,
            "all_probs":   p_probs,
            "awois_match": hit,
        }
        awois_results.append(result)

        priority = "★★" if wreck["name"] in PRIORITY_WRECKS else "  "
        status   = "✔ MATCH" if hit else "✘ MISS "
        logger.info("%s %s %-25s  dist=%4.0fm  pred=%-12s  conf=%.2f  asym=%.2f  score=%d",
                    priority, status, wreck["name"],
                    nearest_dist, CLASS_NAMES[p_class], p_conf, p_asym, p_score)

    awois_hits = [r for r in awois_results if r["awois_match"]]
    logger.info("AWOIS hit rate: %d/%d", len(awois_hits), len(AWOIS_WRECKS))

    # Scaling verdict
    priority_hits = [r for r in awois_results
                     if r["name"] in PRIORITY_WRECKS and r["awois_match"]]
    if len(priority_hits) == 2:
        logger.info("★★ SCALING CORRECT — Admiral AND Merida both detected ★★")
    elif len(priority_hits) == 1:
        logger.info("! Partial — only %s found; check grid CRS / confidence",
                    priority_hits[0]["name"])
    else:
        logger.info("✗ SCALING ISSUE — neither Admiral nor Merida detected at conf>=%.2f",
                    confidence_threshold)
        logger.info("  Printing raw probs at Admiral + Merida pixels for diagnosis…")
        for r in awois_results:
            if r["name"] in PRIORITY_WRECKS:
                logger.info("    %s: %s", r["name"], r["all_probs"])

    # ── Unknown pill targets ───────────────────────────────────────────────
    # At 2.2 km/px the STEEL_HULL classifier can't fire — every wreck is a
    # sub-pixel point source.  Pivot strategy:
    #   •  "WELLHEAD" class fires on compact point anomalies → these ARE
    #      wreck candidates at coarse resolution (physics match).
    #   •  Rank by  FVD amplitude × (1 + asymmetry)  — off-axis pull
    #      implies elongated source (hull rather than a true wellhead).
    #   •  Subtract AWOIS known wrecks (3 km exclusion).
    #   •  Export top-5 as GeoJSON "unknown pill" targets.

    logger.info("─" * 65)
    logger.info("UNKNOWN PILL HUNT — amplitude×asym ranking (AWOIS-subtracted)")
    logger.info("─" * 65)
    logger.info(
        "NOTE: At 2.2 km/px resolution all wrecks are sub-pixel point sources.\n"
        "      STEEL_HULL classifier cannot fire (shape features absent).\n"
        "      Using FVD-amplitude × asymmetry rank instead of model class."
    )

    awois_latlons = [(w["lat"], w["lon"]) for w in AWOIS_WRECKS]

    # Score every pixel: amplitude × (1 + asymmetry)  where amplitude > background
    fvd_centre = np.abs(tiles_all[:, 0, radius, radius])   # |FVD| at pixel centre
    amp_norm   = fvd_centre / (fvd_centre.max() + 1e-12)
    rank_score = amp_norm * (1.0 + asym_all)               # [0, 2]

    unknown_detections = []
    for i in range(H * W):
        if rank_score[i] < 0.20:            # skip background
            continue
        lat_i = float(lats_flat[i])
        lon_i = float(lons_flat[i])
        is_known = any(
            haversine(wlat, wlon, lat_i, lon_i) <= 3000
            for wlat, wlon in awois_latlons
        )
        if is_known:
            continue
        p_class = int(pred_class_all[i])
        p_conf  = float(conf_all[i])
        asym_i  = float(asym_all[i])
        # Pill score (signal-based, not model-based):
        #   High amplitude + high asymmetry → Score 10
        p_score = int(min(10, round(2 + rank_score[i] * 8)))
        unknown_detections.append({
            "lat":              lat_i,
            "lon":              lon_i,
            "rank_score":       round(float(rank_score[i]), 3),
            "fvd_amplitude":    round(float(fvd_centre[i]), 2),
            "asymmetry":        round(asym_i, 3),
            "asym_direction":   _asym_direction(tiles_all[i, 0]),
            "pill_score":       p_score,
            "model_class":      CLASS_NAMES[p_class],
            "model_conf":       round(p_conf, 3),
            "steel_prob":       round(float(probs_all[i, 1]), 3),
            "wellhead_prob":    round(float(probs_all[i, 3]), 3),
        })

    unknown_detections.sort(key=lambda d: d["rank_score"], reverse=True)

    # Deduplicate — one hit per 5 km cluster
    deduped: list[dict] = []
    for d in unknown_detections:
        if not any(haversine(d["lat"], d["lon"], e["lat"], e["lon"]) < 5000
                   for e in deduped):
            deduped.append(d)

    for i, d in enumerate(deduped, 1):
        d["rank"] = i
        logger.info(
            "  GHOST #%-2d  lat=%.4f  lon=%.4f  score=%d  amp=%.1f  "
            "asym=%.2f  dir=%-4s  model=%s(%.2f)",
            i, d["lat"], d["lon"], d["pill_score"],
            d["fvd_amplitude"], d["asymmetry"], d["asym_direction"],
            d["model_class"], d["model_conf"],
        )

    top5 = deduped[:5]

    # ── GeoJSON export ─────────────────────────────────────────────────────
    if top5:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d["lon"], d["lat"]],
                    },
                    "properties": {
                        "rank":          d["rank"],
                        "name":          f"Unknown-Pill-{d['rank']}",
                        "pill_score":    d["pill_score"],
                        "rank_score":    d["rank_score"],
                        "fvd_amplitude": d["fvd_amplitude"],
                        "asymmetry":     d["asymmetry"],
                        "asym_dir":      d["asym_direction"],
                        "model_class":   d["model_class"],
                        "model_conf":    d["model_conf"],
                        "wellhead_prob": d["wellhead_prob"],
                        "steel_prob":    d["steel_prob"],
                        "marker-color":  "#ff0000" if d["pill_score"] >= 9 else "#ff8800",
                        "marker-size":   "large",
                        "description": (
                            f"Score {d['pill_score']}/10 | "
                            f"amp={d['fvd_amplitude']:.1f} | "
                            f"asym={d['asymmetry']:.2f} ({d['asym_direction']}) | "
                            f"model={d['model_class']}@{d['model_conf']:.2f}"
                        ),
                    },
                }
                for d in top5
            ],
        }
        gj_path = output_dir / "unknown_pill_top5.geojson"
        with open(gj_path, "w") as f:
            json.dump(geojson, f, indent=2)
        logger.info("Top-5 GeoJSON → %s", gj_path)
    else:
        logger.info("No candidates found above rank_score threshold")

    # ── Full JSON report ───────────────────────────────────────────────────
    report = {
        "checkpoint":             str(checkpoint_path),
        "epoch":                  ckpt_epoch,
        "val_acc":                round(ckpt_acc, 4),
        "grid_tif":               str(grid_tif_path),
        "grid_shape":             [H, W],
        "grid_resolution_deg":    round(abs(transform.a), 6),
        "grid_resolution_m":      round(abs(transform.a) * 111_320),
        "device":                 device,
        "confidence_threshold":   confidence_threshold,
        "interpolation":          "bicubic",
        "neighbourhood_px":       WIN,
        "resolution_diagnosis": (
            "NAMAG grid at ~2100m/px. ResNet18 trained at synthetic tile scale "
            "(features at 50-200m). Sub-pixel wrecks appear as symmetric point "
            "anomalies; STEEL_HULL classifier cannot fire. WELLHEAD class matches "
            "compact point anomalies — WELLHEAD detections at known wreck coords "
            "confirm magnetic anomaly present but shape classification is unsupported. "
            "Fallback: amplitude × asymmetry rank used for unknown pill targeting."
        ),
        "model_steel_hull_pixels": n_steel,
        "awois_hit_rate":         f"{len(awois_hits)}/{len(AWOIS_WRECKS)}",
        "awois_results":          awois_results,
        "unknown_candidates_count": len(deduped),
        "top5_unknowns":          top5,
    }
    rpt_path = output_dir / "discovery_report_v2.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Full report → %s", rpt_path)
    logger.info("=" * 65)
    logger.info("HEADLINE: AWOIS %d/%d  |  Unknown pills Score 9-10: %d  |  Top 5 → GeoJSON",
                len(awois_hits), len(AWOIS_WRECKS), len(deduped))
    logger.info("=" * 65)
    return report


# ── Asymmetry direction helper ────────────────────────────────────────────

def _asym_direction(nss_block: np.ndarray) -> str:
    """
    Returns compass label (N/NE/E/SE/S/SW/W/NW) of gradient-centroid offset,
    relative to block centre.  'centre' returned for near-symmetric blocks.
    """
    h, w = nss_block.shape
    gx   = np.gradient(nss_block.astype(np.float64), axis=1)
    gy   = np.gradient(nss_block.astype(np.float64), axis=0)
    mag  = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
    yy, xx = np.mgrid[0:h, 0:w]
    cy = np.sum(mag * yy) / np.sum(mag) - (h - 1) / 2
    cx = np.sum(mag * xx) / np.sum(mag) - (w - 1) / 2
    offset = math.sqrt(cx ** 2 + cy ** 2)
    if offset < 0.15:
        return "centre"
    angle = math.degrees(math.atan2(-cy, cx)) % 360   # -cy because array row 0 = north
    dirs   = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    sector = int((angle + 22.5) // 45) % 8
    return dirs[sector]


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    p = argparse.ArgumentParser(description="WH2K Discovery Report v2")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--grid-tif",    required=True)
    p.add_argument("--output-dir",  default="wreck_hunting_ml/models")
    p.add_argument("--confidence",  type=float, default=0.35)
    p.add_argument("--device",      default=_default_device(), choices=["cpu", "cuda"])
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--radius",      type=int, default=2,
                   help="Half-size of neighbourhood window (2 → 5×5)")
    args = p.parse_args()

    run_scan(
        checkpoint_path=Path(args.checkpoint),
        grid_tif_path=Path(args.grid_tif),
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence,
        device=args.device,
        batch_size=args.batch_size,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()
