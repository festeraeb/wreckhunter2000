"""
WreckHunter 2000 — Inference & Scoring Pipeline
==================================================
Runs a trained ResNet-18 across Lake Erie (full lake or per-basin),
cross-references detections against known wells + wrecks, subtracts knowns,
and produces a scored GeoJSON of unknown targets.

Workflow:
  1. SCAN   — Tile the selected Erie region into overlapping 2km×2km chips.
               Run each chip through ResNet-18.  Flag every anomaly.
  2. CORRELATE — Cross-reference every detection against the wellhead DB
                  and known shipwreck DB (Swayze + AWOIS + online AWOIS query).
  3. SUBTRACT — Remove detections that match known wells/wrecks.
  4. SCORE    — Rate every remaining "unknown" on a 1-10 scale:
                  10: High-intensity, off-axis, pill shape       → Steel Wreck
                   7: Moderate-intensity, point-source, off-axis → Wooden/Cargo Wreck
                   3: Linear, low-intensity                      → Geological dyke
  5. EXPORT  — Output a GeoJSON of all Score 8-10 targets.

Usage:
  python wh2k_inference_scorer.py \\
      --model wreck_hunting_ml/models/best_resnet18.pt \\
      --grid-tif magnetic_data/grids/erie_nss.tif \\
      --output scored_targets.geojson

  Per-basin:
  python wh2k_inference_scorer.py \\
      --model wreck_hunting_ml/models/best_resnet18.pt \\
      --grid-tif magnetic_data/tier_2_aero_lowalt/local/gsc_erie_highres_0_001.tif \\
      --basin eastern --output erie_eastern_targets.geojson

  Or for the full pipeline:
  python wh2k_inference_scorer.py --full-pipeline \\
      --model wreck_hunting_ml/models/best_resnet18.pt \\
      --grids-dir magnetic_data/grids \\
      --db db/wrecks.db \\
      --wells-csv data/ogsr_wells.csv
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Lake Erie bounds (derived from GSC Erie TIF actual extent)
LAKE_ERIE = {
    "lon_min": -83.57,
    "lat_min": 41.28,
    "lon_max": -78.78,
    "lat_max": 42.95,
}

ERIE_BASINS = {
    "western": {"lon_min": -83.57, "lat_min": 41.28, "lon_max": -82.50, "lat_max": 42.95},
    "central": {"lon_min": -82.50, "lat_min": 41.28, "lon_max": -80.50, "lat_max": 42.95},
    "eastern": {"lon_min": -80.50, "lat_min": 41.28, "lon_max": -78.78, "lat_max": 42.95},
}

# Backward compat alias
CENTRAL_BASIN = ERIE_BASINS["central"]

CLASS_NAMES = {0: "GEOLOGY_ONLY", 1: "STEEL_HULL", 2: "WOOD_CARGO", 3: "WELLHEAD"}
NE_SW_STRIKE_DEG = 45.0
TILE_DX_M = 8.928571428571429   # metres per pixel at 224px / 2000m


# ── Detection Data Class ──────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detection from the inference pass."""
    detection_id: int
    lat: float
    lon: float
    predicted_class: int
    class_name: str
    confidence: float                 # Max softmax probability
    probabilities: dict               # All class probabilities
    # Grid-derived features
    peak_amplitude_nt: float = 0.0
    axis_offset_from_geology_deg: float = 0.0
    spatial_extent_m: float = 0.0
    aspect_ratio: float = 1.0
    # Cross-reference
    known_match: str = "unknown"       # "wellhead", "wreck", "unknown"
    known_match_name: str = ""
    known_match_distance_m: float = -1.0
    # Final scoring
    wreck_score: int = 0              # 1-10 scale
    score_reasons: list[str] = field(default_factory=list)
    sat_visible: bool = False


# ── Tiling & Inference ─────────────────────────────────────────────────────

def _chip_coords(grid_shape, chip_px, overlap_frac, transform):
    """Generate (row_start, col_start, lat_center, lon_center) for tiling."""
    from rasterio.transform import xy

    h, w = grid_shape
    step = int(chip_px * (1 - overlap_frac))
    step = max(step, 1)

    for r in range(0, h - chip_px + 1, step):
        for c in range(0, w - chip_px + 1, step):
            center_r = r + chip_px // 2
            center_c = c + chip_px // 2
            lon, lat = xy(transform, center_r, center_c)
            yield r, c, lat, lon


def scan_grid(
    model_path: str | Path,
    grid_tif_path: str | Path,
    chip_px: int = 224,
    overlap_frac: float = 0.25,
    confidence_threshold: float = 0.5,
    bbox: dict | None = None,
    device: str = "auto",
) -> list[Detection]:
    """Tile a GeoTIFF, run ResNet-18 inference, collect detections.

    Only returns tiles where the model predicts a non-geology class
    with confidence above threshold.
    """
    try:
        import torch
        from torchvision.models import resnet18
        import rasterio
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return []

    from scipy import ndimage as _nd

    if bbox is None:
        bbox = LAKE_ERIE

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    # Per-tile robust normalization: amplitude-invariant (shape-focused)
    # Each tile's channels are independently scaled to [0,1] via 2nd-98th percentile,
    # so the CNN sees the same dipole morphology regardless of survey altitude.
    use_pertile_norm = True
    norm_stats = checkpoint.get("norm_stats", {"mean": [0, 0, 0], "std": [1, 1, 1]})
    ch_mean = np.array(norm_stats["mean"], dtype=np.float32).reshape(3, 1, 1)
    ch_std = np.array(norm_stats["std"], dtype=np.float32).reshape(3, 1, 1)

    model = resnet18(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # FVD preprocessing — must match training when checkpoint was trained with it
    use_fvd = checkpoint.get("fvd_preprocessing", False)
    _k_cache: dict[tuple[int, int], "torch.Tensor"] = {}

    def _apply_fvd(x: "torch.Tensor") -> "torch.Tensor":
        """Replace ch0 (NSS) with FVD(NSS) — matches training FVDTransform."""
        B, C, H, W = x.shape
        key = (H, W)
        if key not in _k_cache:
            kx = torch.fft.fftfreq(W, d=TILE_DX_M) * (2 * math.pi)
            ky = torch.fft.fftfreq(H, d=TILE_DX_M) * (2 * math.pi)
            KY, KX = torch.meshgrid(ky, kx, indexing="ij")
            _k_cache[key] = torch.sqrt(KX**2 + KY**2)
        K = _k_cache[key].to(x.device)
        nss = x[:, 0]
        spec = torch.fft.fft2(nss)
        fvd = torch.real(torch.fft.ifft2(spec * K))
        out = x.clone()
        out[:, 0] = fvd
        return out

    if use_fvd:
        logger.info("FVD preprocessing enabled (matches training checkpoint)")

    # Open grid
    with rasterio.open(str(grid_tif_path)) as src:
        full_grid = src.read(1).astype(np.float64)
        transform = src.transform
        if src.nodata is not None:
            full_grid[full_grid == src.nodata] = np.nan

        # Clip to bbox
        from rasterio.transform import rowcol
        r_min, c_min = rowcol(transform, bbox["lon_min"], bbox["lat_max"])
        r_max, c_max = rowcol(transform, bbox["lon_max"], bbox["lat_min"])
        r_min, r_max = max(0, int(r_min)), min(full_grid.shape[0], int(r_max))
        c_min, c_max = max(0, int(c_min)), min(full_grid.shape[1], int(c_max))
        grid = full_grid[r_min:r_max, c_min:c_max]

        # Update transform for clipped grid
        from rasterio.transform import Affine
        clipped_transform = transform * Affine.translation(c_min, r_min)

    # Impute NaN
    if np.any(np.isnan(grid)):
        grid[np.isnan(grid)] = np.nanmedian(grid)

    # Compute 3 layers
    def _nss(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        dz = _nd.laplace(g)
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _vdr(g):
        fft = np.fft.fft2(g)
        ny, nx = g.shape
        ky = np.fft.fftfreq(ny).reshape(-1, 1)
        kx = np.fft.fftfreq(nx).reshape(1, -1)
        k = np.sqrt(kx**2 + ky**2)
        k[0, 0] = 1e-10
        return np.real(np.fft.ifft2(fft * k * 2 * np.pi))

    def _tilt(g):
        dx = np.gradient(g, axis=1)
        dy = np.gradient(g, axis=0)
        thdr = np.sqrt(dx**2 + dy**2)
        vdr = _vdr(g)
        return np.arctan2(vdr, thdr + 1e-12)

    logger.info("Computing NSS, VDR, Tilt layers for grid (%d×%d)...", grid.shape[0], grid.shape[1])
    nss_full = _nss(grid).astype(np.float32)
    vdr_full = _vdr(grid).astype(np.float32)
    tilt_full = _tilt(grid).astype(np.float32)

    # Scan
    detections = []
    det_id = 0
    n_tiles = 0

    logger.info("Scanning bbox [%.2f,%.2f]-[%.2f,%.2f] with %dpx chips, %.0f%% overlap...",
                 bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"],
                 chip_px, overlap_frac * 100)

    with torch.no_grad():
        for r, c, lat, lon in _chip_coords(grid.shape, chip_px, overlap_frac, clipped_transform):
            if not (bbox["lat_min"] <= lat <= bbox["lat_max"] and
                    bbox["lon_min"] <= lon <= bbox["lon_max"]):
                continue

            # Extract chip
            nss_chip = nss_full[r:r + chip_px, c:c + chip_px]
            vdr_chip = vdr_full[r:r + chip_px, c:c + chip_px]
            tilt_chip = tilt_full[r:r + chip_px, c:c + chip_px]

            if nss_chip.shape != (chip_px, chip_px):
                continue

            tile = np.stack([nss_chip, vdr_chip, tilt_chip], axis=0)
            if use_pertile_norm:
                # Per-tile robust percentile normalization — amplitude-invariant.
                # A 50 nT wreck at 300m and a 400 nT wreck at 100m both map to [0,1]
                # so the CNN only sees the dipole SHAPE, not the raw magnitude.
                for ch_i in range(tile.shape[0]):
                    lo, hi = np.percentile(tile[ch_i], 2), np.percentile(tile[ch_i], 98)
                    if hi - lo < 1e-9:
                        tile[ch_i] = 0.0
                    else:
                        tile[ch_i] = np.clip((tile[ch_i] - lo) / (hi - lo), 0.0, 1.0)
            else:
                tile = (tile - ch_mean) / (ch_std + 1e-8)

            tensor = torch.tensor(tile[np.newaxis], dtype=torch.float32).to(device)
            if use_fvd:
                tensor = _apply_fvd(tensor)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            n_tiles += 1

            # Only keep non-geology detections above threshold
            if pred_class != 0 and confidence >= confidence_threshold:
                # Extract features from the raw chip for scoring
                raw_chip = grid[r:r + chip_px, c:c + chip_px]
                peak_amp = float(np.max(np.abs(raw_chip - np.median(raw_chip))))

                # Axis analysis
                grad_x = np.gradient(raw_chip, axis=1)
                grad_y = np.gradient(raw_chip, axis=0)
                peak_idx = np.unravel_index(np.argmax(np.abs(raw_chip)), raw_chip.shape)
                trough_idx = np.unravel_index(np.argmin(raw_chip), raw_chip.shape)
                dy = peak_idx[0] - trough_idx[0]
                dx = peak_idx[1] - trough_idx[1]
                anomaly_axis = math.degrees(math.atan2(dx, dy)) % 360
                axis_offset = abs(anomaly_axis - NE_SW_STRIKE_DEG)
                if axis_offset > 180:
                    axis_offset = 360 - axis_offset
                if axis_offset > 90:
                    axis_offset = 180 - axis_offset

                # Spatial extent (half-max contour)
                threshold = peak_amp * 0.5
                above = np.abs(raw_chip - np.median(raw_chip)) > threshold
                extent_px = float(np.sqrt(np.sum(above)))
                # Estimate metres per pixel
                res_deg = abs(clipped_transform.a)
                m_per_px = res_deg * 111_320 * math.cos(math.radians(lat))
                spatial_extent_m = extent_px * m_per_px

                # Aspect ratio
                if np.any(above):
                    rows = np.any(above, axis=1)
                    cols = np.any(above, axis=0)
                    h_px = max(np.sum(rows), 1)
                    w_px = max(np.sum(cols), 1)
                    aspect = max(h_px, w_px) / min(h_px, w_px)
                else:
                    aspect = 1.0

                det = Detection(
                    detection_id=det_id,
                    lat=lat,
                    lon=lon,
                    predicted_class=pred_class,
                    class_name=CLASS_NAMES.get(pred_class, "UNKNOWN"),
                    confidence=round(confidence, 4),
                    probabilities={CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
                    peak_amplitude_nt=round(peak_amp, 2),
                    axis_offset_from_geology_deg=round(axis_offset, 1),
                    spatial_extent_m=round(spatial_extent_m, 1),
                    aspect_ratio=round(aspect, 2),
                )
                detections.append(det)
                det_id += 1

    logger.info("Scanned %d tiles, found %d detections above %.2f confidence",
                n_tiles, len(detections), confidence_threshold)
    return detections


# ── Cross-Reference (Correlate & Subtract) ─────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def correlate_and_subtract(
    detections: list[Detection],
    db_path: str | Path = REPO_ROOT / "db" / "wrecks.db",
    wells_csv: Optional[str | Path] = None,
    match_radius_m: float = 2000.0,
) -> tuple[list[Detection], list[Detection]]:
    """Cross-reference detections against known wells and wrecks.

    Returns (unknowns, knowns) where:
      unknowns = detections that DON'T match any known site
      knowns = detections that DO match (subtracted)
    """
    # Load known wrecks
    known_wrecks = []
    db_path = Path(db_path)
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT name, latitude, longitude FROM features "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
                "AND found_status = 'found'"
            ).fetchall()
            known_wrecks = [dict(r) for r in rows]
        finally:
            conn.close()

    # Load wells
    known_wells = []
    if wells_csv and Path(wells_csv).exists():
        import csv
        with open(wells_csv, "r", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get("SUR_LAT83") or row.get("latitude") or row.get("lat") or 0)
                    lon = float(row.get("SUR_LONG83") or row.get("longitude") or row.get("lon") or 0)
                    if lat and lon:
                        well_name = row.get("WELL_NAME") or row.get("API_NUM") or row.get("name", "")
                        known_wells.append({"name": well_name, "latitude": lat, "longitude": lon})
                except (ValueError, TypeError):
                    continue

    logger.info("Cross-referencing %d detections against %d wrecks + %d wells (radius %.0fm)",
                len(detections), len(known_wrecks), len(known_wells), match_radius_m)

    unknowns = []
    knowns = []

    for det in detections:
        matched = False

        # Check wells
        for well in known_wells:
            dist = _haversine_m(det.lat, det.lon, well["latitude"], well["longitude"])
            if dist <= match_radius_m:
                det.known_match = "wellhead"
                det.known_match_name = well["name"]
                det.known_match_distance_m = round(dist, 1)
                matched = True
                break

        # Check wrecks
        if not matched:
            for wreck in known_wrecks:
                dist = _haversine_m(det.lat, det.lon, wreck["latitude"], wreck["longitude"])
                if dist <= match_radius_m:
                    det.known_match = "wreck"
                    det.known_match_name = wreck["name"]
                    det.known_match_distance_m = round(dist, 1)
                    matched = True
                    break

        if matched:
            knowns.append(det)
        else:
            unknowns.append(det)

    logger.info("Correlation: %d matched knowns (subtracted), %d unknowns remaining",
                len(knowns), len(unknowns))
    return unknowns, knowns


# ── Scoring Engine ─────────────────────────────────────────────────────────

def score_unknowns(detections: list[Detection]) -> list[Detection]:
    """Score each unknown detection on a 1-10 wreck likelihood scale.

    Scoring criteria:
      10: High-intensity, off-axis (>45°), pill shape, model says STEEL_HULL with high confidence
       7: Moderate intensity, point-source, off-axis, WOOD_CARGO class
       3: Linear (high aspect ratio), low-intensity, aligned with geology → geological dyke

    Amplitude thresholds are percentile-relative (P25/P75 of the detection
    population) so scoring works the same regardless of survey altitude.
    """
    # Compute amplitude percentiles from the detection population
    amps = [d.peak_amplitude_nt for d in detections if d.peak_amplitude_nt > 0]
    if len(amps) >= 4:
        amp_p25 = float(np.percentile(amps, 25))
        amp_p75 = float(np.percentile(amps, 75))
        logger.info("Amplitude percentiles: P25=%.1f nT, P75=%.1f nT (n=%d)", amp_p25, amp_p75, len(amps))
    else:
        amp_p25 = amp_p75 = None

    for det in detections:
        score = 1
        reasons = []

        # Class-based base score
        if det.predicted_class == 1:  # STEEL_HULL
            score = 7
            reasons.append("Model: STEEL_HULL")
        elif det.predicted_class == 2:  # WOOD_CARGO
            score = 5
            reasons.append("Model: WOOD_CARGO")
        elif det.predicted_class == 3:  # WELLHEAD (shouldn't be here after subtract, but edge cases)
            score = 2
            reasons.append("Model: WELLHEAD (uncorrelated)")

        # Confidence bonus
        if det.confidence >= 0.85:
            score += 1
            reasons.append(f"High confidence ({det.confidence:.2f})")
        elif det.confidence < 0.6:
            score -= 1
            reasons.append(f"Low confidence ({det.confidence:.2f})")

        # Amplitude bonus — percentile-relative, not absolute nT.
        # This makes scoring altitude-invariant: closer surveys produce
        # stronger pings across the board, so only the relative rank matters.
        if amp_p75 is not None and amp_p25 is not None:
            if det.peak_amplitude_nt >= amp_p75:
                score += 1
                reasons.append(f"Strong amplitude ({det.peak_amplitude_nt:.1f} nT, ≥P75={amp_p75:.1f})")
            elif det.peak_amplitude_nt <= amp_p25:
                score -= 1
                reasons.append(f"Weak amplitude ({det.peak_amplitude_nt:.1f} nT, ≤P25={amp_p25:.1f})")
        else:
            # Fallback for single-detection edge case
            if det.peak_amplitude_nt >= 50:
                score += 1
                reasons.append(f"Strong amplitude ({det.peak_amplitude_nt:.1f} nT)")
            elif det.peak_amplitude_nt < 10:
                score -= 1
                reasons.append(f"Weak amplitude ({det.peak_amplitude_nt:.1f} nT)")

        # Off-axis bonus (wreck-like: oriented differently from geology)
        if det.axis_offset_from_geology_deg >= 45:
            score += 1
            reasons.append(f"Off-axis ({det.axis_offset_from_geology_deg:.0f}° from geology)")
        elif det.axis_offset_from_geology_deg < 15:
            score -= 1
            reasons.append(f"Aligned with geology ({det.axis_offset_from_geology_deg:.0f}°)")

        # Aspect ratio (compact = wreck-like, very elongated = geology)
        if det.aspect_ratio < 2.0:
            reasons.append("Compact shape (wreck-like)")
        elif det.aspect_ratio > 5.0:
            score -= 2
            reasons.append(f"Very elongated (AR={det.aspect_ratio:.1f}, likely geological)")

        # Spatial extent
        if 100 < det.spatial_extent_m < 500:
            reasons.append(f"Wreck-scale extent ({det.spatial_extent_m:.0f}m)")
        elif det.spatial_extent_m > 2000:
            score -= 1
            reasons.append(f"Very large extent ({det.spatial_extent_m:.0f}m, likely regional)")

        # Clamp score
        det.wreck_score = max(1, min(10, score))
        det.score_reasons = reasons

    return detections


# ── GeoJSON Export ─────────────────────────────────────────────────────────

def export_geojson(
    detections: list[Detection],
    output_path: str | Path,
    min_score: int = 8,
) -> Path:
    """Export scored detections as GeoJSON (FeatureCollection).

    Only includes detections with wreck_score >= min_score.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for det in detections:
        if det.wreck_score < min_score:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [det.lon, det.lat],
            },
            "properties": {
                "detection_id": det.detection_id,
                "wreck_score": det.wreck_score,
                "predicted_class": det.class_name,
                "confidence": det.confidence,
                "peak_amplitude_nt": det.peak_amplitude_nt,
                "axis_offset_deg": det.axis_offset_from_geology_deg,
                "spatial_extent_m": det.spatial_extent_m,
                "aspect_ratio": det.aspect_ratio,
                "score_reasons": "; ".join(det.score_reasons),
                "sat_visible": det.sat_visible,
                "known_match": det.known_match,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "pipeline": "WreckHunter 2000 — Inference Pipeline",
            "total_detections": len(detections),
            "exported_count": len(features),
            "min_score": min_score,
            "region": "Lake Erie Central Basin",
        },
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info("Exported %d targets (score >= %d) to %s", len(features), min_score, output_path)
    return output_path


# ── Full Pipeline Orchestrator ─────────────────────────────────────────────

def run_full_inference(
    model_path: str | Path,
    grid_tif_path: str | Path,
    db_path: str | Path = REPO_ROOT / "db" / "wrecks.db",
    wells_csv: Optional[str | Path] = None,
    output_geojson: str | Path = "wh2k_targets.geojson",
    confidence_threshold: float = 0.5,
    min_score: int = 8,
    match_radius_m: float = 2000.0,
    device: str = "auto",
    bbox: dict | None = None,
) -> dict:
    """Run the complete inference pipeline: Scan → Correlate → Subtract → Score → Export.

    Returns summary dict.
    """
    logger.info("=" * 60)
    logger.info("WreckHunter 2000 — Full Inference Pipeline")
    logger.info("=" * 60)

    # Step 1: Scan
    logger.info("STEP 1: Scanning Lake Erie...")
    detections = scan_grid(
        model_path, grid_tif_path,
        confidence_threshold=confidence_threshold,
        device=device,
        bbox=bbox,
    )

    if not detections:
        logger.warning("No detections found. Check grid coverage and model.")
        return {"total_detections": 0, "unknowns": 0, "exported": 0}

    # Step 2+3: Correlate & Subtract
    logger.info("STEP 2+3: Cross-referencing & subtracting knowns...")
    unknowns, knowns = correlate_and_subtract(
        detections, db_path, wells_csv, match_radius_m,
    )

    # Step 4: Score
    logger.info("STEP 4: Scoring unknowns...")
    scored = score_unknowns(unknowns)

    # Step 5: Export
    logger.info("STEP 5: Exporting GeoJSON (score >= %d)...", min_score)
    out_path = export_geojson(scored, output_geojson, min_score)

    # Also export full results as JSON
    full_path = Path(output_geojson).with_suffix(".full.json")
    all_results = {
        "detections": [asdict(d) for d in detections],
        "knowns_subtracted": [asdict(d) for d in knowns],
        "unknowns_scored": [asdict(d) for d in scored],
        "summary": {
            "total_detections": len(detections),
            "known_matches": len(knowns),
            "unknowns": len(unknowns),
            "score_8_plus": sum(1 for d in scored if d.wreck_score >= 8),
            "score_10": sum(1 for d in scored if d.wreck_score == 10),
        },
    }
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary = all_results["summary"]
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total detections: %d", summary["total_detections"])
    logger.info("  Known matches (subtracted): %d", summary["known_matches"])
    logger.info("  Unknowns scored: %d", summary["unknowns"])
    logger.info("  Score 8+ targets: %d", summary["score_8_plus"])
    logger.info("  Score 10 targets: %d", summary["score_10"])
    logger.info("  GeoJSON: %s", out_path)
    logger.info("  Full JSON: %s", full_path)
    logger.info("=" * 60)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="WH2K Inference & Scoring Pipeline")
    parser.add_argument("--model", required=True, help="Path to best_resnet18.pt")
    parser.add_argument("--grid-tif", required=True, help="Path to aeromagnetic GeoTIFF")
    parser.add_argument("--db", type=str, default=str(REPO_ROOT / "db" / "wrecks.db"))
    parser.add_argument("--wells-csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="wh2k_targets.geojson")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--min-score", type=int, default=8)
    parser.add_argument("--match-radius", type=float, default=2000.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--basin", type=str, default="all",
                        choices=["all", "western", "central", "eastern"],
                        help="Run on full lake or a specific Erie basin")
    args = parser.parse_args()

    basin_bbox = LAKE_ERIE if args.basin == "all" else ERIE_BASINS[args.basin]

    run_full_inference(
        model_path=args.model,
        grid_tif_path=args.grid_tif,
        db_path=args.db,
        wells_csv=args.wells_csv,
        output_geojson=args.output,
        confidence_threshold=args.confidence,
        min_score=args.min_score,
        match_radius_m=args.match_radius,
        device=args.device,
        bbox=basin_bbox,
    )


if __name__ == "__main__":
    main()
