"""
WreckHunter 2000 — A/B Spectral Attenuation Report
===================================================
Compares two magnetic grids over the same footprint and quantifies attenuation
of short-wavelength content from A to B.

Intended use:
- A = higher-detail source (for example low-altitude aeromag)
- B = smoothed/downsampled source (for example regional GeoTIFF)

Outputs:
- JSON report with aggregate metrics and radial-spectrum table
- CSV radial spectrum for plotting

Usage:
  python scripts/wh2k_ab_attenuation.py \
    --a path/to/high_detail.tif \
    --b path/to/regional.tif
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SpectrumSummary:
    low_band_mean_a: float
    low_band_mean_b: float
    high_band_mean_a: float
    high_band_mean_b: float
    high_to_low_ratio_a: float
    high_to_low_ratio_b: float
    attenuation_factor: float
    attenuation_db: float
    corr_spatial: float
    corr_spectral: float


def _read_tif(path: Path) -> tuple[np.ndarray, float, float]:
    import rasterio

    with rasterio.open(path) as src:
        grid = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            grid[grid == nodata] = np.nan
        transform = src.transform
        dx = float(abs(transform.a))
        dy = float(abs(transform.e))
    return grid, dx, dy


def _fill_nans(g: np.ndarray) -> np.ndarray:
    out = g.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        raise ValueError("Grid has no finite cells")
    med = float(np.nanmedian(out))
    out[~finite] = med
    return out


def _crop_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _radial_psd(grid: np.ndarray, cell_size_m: float, n_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    y = np.hanning(h)
    x = np.hanning(w)
    win = np.outer(y, x)

    z = grid - np.mean(grid)
    z = z * win

    spec = np.fft.fft2(z)
    power = np.abs(spec) ** 2

    ky = np.fft.fftfreq(h, d=cell_size_m).reshape(-1, 1)
    kx = np.fft.fftfreq(w, d=cell_size_m).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)

    k_flat = k.ravel()
    p_flat = power.ravel()

    k_max = np.max(k_flat)
    if k_max <= 0:
        raise ValueError("Invalid frequency grid")

    edges = np.linspace(0.0, k_max, n_bins + 1)
    idx = np.digitize(k_flat, edges) - 1

    k_centers = []
    p_means = []
    for i in range(n_bins):
        mask = idx == i
        if not np.any(mask):
            continue
        k_centers.append(float(0.5 * (edges[i] + edges[i + 1])))
        p_means.append(float(np.mean(p_flat[mask])))

    return np.asarray(k_centers), np.asarray(p_means)


def _band_split_indices(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if freq.size < 10:
        raise ValueError("Not enough spectral bins for band split")
    q1 = np.quantile(freq, 0.25)
    q3 = np.quantile(freq, 0.75)
    low = freq <= q1
    high = freq >= q3
    return low, high


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compare_grids(a_path: Path, b_path: Path, output_dir: Path, tag: str) -> dict:
    a_raw, ax, ay = _read_tif(a_path)
    b_raw, bx, by = _read_tif(b_path)

    a_raw, b_raw = _crop_to_common(a_raw, b_raw)
    a = _fill_nans(a_raw)
    b = _fill_nans(b_raw)

    cell_a = 0.5 * (ax + ay)
    cell_b = 0.5 * (bx + by)
    cell = max(cell_a, cell_b)

    k_a, p_a = _radial_psd(a, cell)
    k_b, p_b = _radial_psd(b, cell)

    n = min(len(k_a), len(k_b))
    k = k_a[:n]
    p_a = p_a[:n]
    p_b = p_b[:n]

    low_idx, high_idx = _band_split_indices(k)

    low_a = float(np.mean(p_a[low_idx]))
    low_b = float(np.mean(p_b[low_idx]))
    high_a = float(np.mean(p_a[high_idx]))
    high_b = float(np.mean(p_b[high_idx]))

    h2l_a = _safe_ratio(high_a, low_a)
    h2l_b = _safe_ratio(high_b, low_b)
    attenuation_factor = _safe_ratio(h2l_b, h2l_a)
    attenuation_db = 10.0 * np.log10(max(attenuation_factor, 1e-12))

    corr_spatial = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    pa_log = np.log10(np.maximum(p_a, 1e-12))
    pb_log = np.log10(np.maximum(p_b, 1e-12))
    corr_spectral = float(np.corrcoef(pa_log, pb_log)[0, 1])

    summary = SpectrumSummary(
        low_band_mean_a=low_a,
        low_band_mean_b=low_b,
        high_band_mean_a=high_a,
        high_band_mean_b=high_b,
        high_to_low_ratio_a=h2l_a,
        high_to_low_ratio_b=h2l_b,
        attenuation_factor=float(attenuation_factor),
        attenuation_db=float(attenuation_db),
        corr_spatial=corr_spatial,
        corr_spectral=corr_spectral,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ab_attenuation_{tag}.csv"
    json_path = output_dir / f"ab_attenuation_{tag}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["radial_frequency_cycles_per_m", "power_a", "power_b", "ratio_b_over_a"])
        for ki, ai, bi in zip(k, p_a, p_b):
            w.writerow([ki, ai, bi, _safe_ratio(bi, ai)])

    report = {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "shape_used": [int(a.shape[0]), int(a.shape[1])],
        "cell_size_m_used": cell,
        "summary": summary.__dict__,
        "notes": {
            "interpretation": "attenuation_factor < 1 means B has weaker high-frequency content than A",
            "bands": "low <= 25th percentile of radial frequency, high >= 75th percentile",
        },
        "radial_spectrum_csv": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("A/B attenuation report saved: %s", json_path)
    logger.info("Radial spectrum CSV saved: %s", csv_path)

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="WH2K A/B Spectral Attenuation Report")
    p.add_argument("--a", required=True, help="Path to grid A (higher-detail reference)")
    p.add_argument("--b", required=True, help="Path to grid B (comparison grid)")
    p.add_argument("--output-dir", default="wreck_hunting_ml/output", help="Output folder")
    p.add_argument("--tag", default="run", help="Output filename tag")
    args = p.parse_args()

    a = Path(args.a)
    b = Path(args.b)

    report = compare_grids(
        a_path=a,
        b_path=b,
        output_dir=Path(args.output_dir),
        tag=args.tag,
    )

    s = report["summary"]
    print("\n" + "=" * 66)
    print("A/B ATTENUATION SUMMARY")
    print(f"A: {a}")
    print(f"B: {b}")
    print(f"Shape used: {report['shape_used'][0]} x {report['shape_used'][1]}")
    print(f"Spatial corr:  {s['corr_spatial']:.4f}")
    print(f"Spectral corr: {s['corr_spectral']:.4f}")
    print(f"High/Low ratio A: {s['high_to_low_ratio_a']:.6f}")
    print(f"High/Low ratio B: {s['high_to_low_ratio_b']:.6f}")
    print(f"Attenuation factor (B/A): {s['attenuation_factor']:.6f}")
    print(f"Attenuation dB: {s['attenuation_db']:.2f} dB")
    print("=" * 66)


if __name__ == "__main__":
    main()
