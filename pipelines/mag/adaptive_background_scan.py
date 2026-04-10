#!/usr/bin/env python3
"""Adaptive background magnetic scan for subtle, sharp-edged anomalies.

This script models local background per moving window (default 1000 yards), then
searches for nominal-amplitude spikes that still have strong edge behavior.

It is intended for deep-wreck style screening where broad geological gradients can
hide weak targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy import ndimage


YARDS_PER_METER = 1.0936133
METER_PER_DEG_LAT = 111_320.0


@dataclass
class Candidate:
    source_grid: str
    label_id: int
    center_lat: float
    center_lon: float
    pixel_count: int
    width_m: float
    height_m: float
    amplitude_peak_abs: float
    amplitude_mean_abs: float
    local_z_max: float
    local_z_mean: float
    edge_z_max: float
    edge_z_mean: float
    score: float


def _meters_per_pixel(src: rasterio.io.DatasetReader) -> tuple[float, float]:
    """Approximate meters per pixel from raster geotransform at center latitude."""
    res_x_deg = abs(src.transform.a)
    res_y_deg = abs(src.transform.e)
    center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    m_per_deg_lon = METER_PER_DEG_LAT * math.cos(math.radians(center_lat))
    mx = max(res_x_deg * m_per_deg_lon, 0.1)
    my = max(res_y_deg * METER_PER_DEG_LAT, 0.1)
    return mx, my


def _window_px(window_yards: float, mx: float, my: float) -> tuple[int, int]:
    window_m = window_yards / YARDS_PER_METER
    wx = max(3, int(round(window_m / mx)))
    wy = max(3, int(round(window_m / my)))
    # Keep odd windows for symmetric local statistics.
    if wx % 2 == 0:
        wx += 1
    if wy % 2 == 0:
        wy += 1
    return wy, wx


def _local_zscore(arr: np.ndarray, wy: int, wx: int, eps: float = 1e-6) -> np.ndarray:
    mean = ndimage.uniform_filter(arr, size=(wy, wx), mode="nearest")
    mean_sq = ndimage.uniform_filter(arr * arr, size=(wy, wx), mode="nearest")
    var = np.maximum(mean_sq - (mean * mean), 0.0)
    std = np.sqrt(var)
    return (arr - mean) / (std + eps)


def _extract_candidates(
    src_path: Path,
    window_yards: float,
    z_thresh: float,
    edge_z_thresh: float,
    min_pixels: int,
    max_pixels: int,
) -> list[Candidate]:
    out: list[Candidate] = []
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr = np.nan_to_num(arr, nan=0.0)

        mx, my = _meters_per_pixel(src)
        wy, wx = _window_px(window_yards, mx, my)

        abs_arr = np.abs(arr)
        local_z = _local_zscore(abs_arr, wy, wx)

        gy, gx = np.gradient(arr)
        grad_mag = np.hypot(gx, gy)
        edge_z = _local_zscore(grad_mag, wy, wx)

        # Candidate mask: nominal spikes above local background with sharp edges.
        mask = (local_z >= z_thresh) & (edge_z >= edge_z_thresh)
        labels, nlab = ndimage.label(mask)
        if nlab == 0:
            return out

        slices = ndimage.find_objects(labels)
        for i, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            comp = labels[slc] == i
            pix = int(comp.sum())
            if pix < min_pixels or pix > max_pixels:
                continue

            rr, cc = np.where(comp)
            r0, c0 = slc[0].start, slc[1].start
            gr = rr + r0
            gc = cc + c0

            comp_abs = abs_arr[gr, gc]
            comp_z = local_z[gr, gc]
            comp_ez = edge_z[gr, gc]

            # Center on strongest absolute amplitude in the component.
            k = int(np.argmax(comp_abs))
            center_r = int(gr[k])
            center_c = int(gc[k])
            center_lon, center_lat = xy(src.transform, center_r, center_c)

            rmin, rmax = int(gr.min()), int(gr.max())
            cmin, cmax = int(gc.min()), int(gc.max())
            width_m = (cmax - cmin + 1) * mx
            height_m = (rmax - rmin + 1) * my

            # Weighted score emphasizes edge sharpness slightly more for wreck-like shapes.
            score = 0.45 * float(comp_z.max()) + 0.55 * float(comp_ez.max())

            out.append(
                Candidate(
                    source_grid=src_path.name,
                    label_id=i,
                    center_lat=float(center_lat),
                    center_lon=float(center_lon),
                    pixel_count=pix,
                    width_m=float(width_m),
                    height_m=float(height_m),
                    amplitude_peak_abs=float(comp_abs.max()),
                    amplitude_mean_abs=float(comp_abs.mean()),
                    local_z_max=float(comp_z.max()),
                    local_z_mean=float(comp_z.mean()),
                    edge_z_max=float(comp_ez.max()),
                    edge_z_mean=float(comp_ez.mean()),
                    score=float(score),
                )
            )

    return out


def _write_kml(cands: list[Candidate], out_kml: Path) -> None:
    placemarks = []
    for c in cands:
        half_w_m = c.width_m / 2.0
        half_h_m = c.height_m / 2.0
        dlat = half_h_m / METER_PER_DEG_LAT
        dlon = half_w_m / (METER_PER_DEG_LAT * math.cos(math.radians(c.center_lat)) + 1e-9)

        ring = [
            (c.center_lon - dlon, c.center_lat + dlat),
            (c.center_lon + dlon, c.center_lat + dlat),
            (c.center_lon + dlon, c.center_lat - dlat),
            (c.center_lon - dlon, c.center_lat - dlat),
            (c.center_lon - dlon, c.center_lat + dlat),
        ]
        coords = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in ring)
        placemarks.append(
            f"""
<Placemark>
  <name>{c.source_grid} #{c.label_id} score={c.score:.2f}</name>
  <description>local_z_max={c.local_z_max:.2f}, edge_z_max={c.edge_z_max:.2f}, size_m={c.width_m:.1f}x{c.height_m:.1f}</description>
  <Point><coordinates>{c.center_lon:.6f},{c.center_lat:.6f},0</coordinates></Point>
</Placemark>
<Placemark>
  <name>Footprint {c.source_grid} #{c.label_id}</name>
  <Style><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><color>2200ffff</color></PolyStyle></Style>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
"""
        )

    kml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<kml xmlns=\"http://www.opengis.net/kml/2.2\">
  <Document>
    <name>Adaptive Background Scan Candidates</name>
    {''.join(placemarks)}
  </Document>
</kml>
"""
    out_kml.write_text(kml, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive background + edge anomaly detector")
    p.add_argument("--input-glob", required=True, help="Glob for input GeoTIFF files")
    p.add_argument("--output-dir", default="adaptive_bg_output")
    p.add_argument("--window-yards", type=float, default=1000.0, help="Background window in yards")
    p.add_argument("--z-thresh", type=float, default=1.0, help="Minimum local z-score for amplitude")
    p.add_argument("--edge-z-thresh", type=float, default=1.0, help="Minimum local z-score for edge strength")
    p.add_argument("--min-pixels", type=int, default=3)
    p.add_argument("--max-pixels", type=int, default=800)
    p.add_argument("--top-n", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    for tif in files:
        candidates.extend(
            _extract_candidates(
                tif,
                window_yards=args.window_yards,
                z_thresh=args.z_thresh,
                edge_z_thresh=args.edge_z_thresh,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[: args.top_n]

    json_path = out_dir / "adaptive_candidates.json"
    csv_path = out_dir / "adaptive_candidates.csv"
    kml_path = out_dir / "adaptive_candidates.kml"

    json_path.write_text(json.dumps([asdict(c) for c in candidates], indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(candidates[0]).keys()) if candidates else [
            "source_grid", "label_id", "center_lat", "center_lon", "pixel_count", "width_m", "height_m",
            "amplitude_peak_abs", "amplitude_mean_abs", "local_z_max", "local_z_mean", "edge_z_max", "edge_z_mean", "score"
        ])
        w.writeheader()
        for c in candidates:
            w.writerow(asdict(c))

    _write_kml(candidates, kml_path)

    print(f"inputs={len(files)}")
    print(f"candidates={len(candidates)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"kml={kml_path}")


if __name__ == "__main__":
    main()
