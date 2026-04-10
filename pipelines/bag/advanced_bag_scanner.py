#!/usr/bin/env python3
"""
Advanced BAG File Scanner with Redaction Signature Detection
Enhanced for speed, accuracy, and signature analysis
"""

import os
import sys
import numpy as np
import rasterio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import tempfile
import hashlib
from pyproj import Transformer
from scipy import ndimage
import math

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RedactionSignature:
    """Represents a detected redaction signature"""
    signature_type: str  # 'smoothing', 'removal', 'alteration', 'pattern'
    confidence: float
    location: Tuple[float, float]  # lat, lon
    bounding_box: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    size_pixels: int
    size_meters_sq: float
    redactor_id: Optional[str] = None  # Attempt to identify redactor
    technique_used: str = "unknown"
    evidence: Dict = None

@dataclass
class WreckCandidate:
    """Enhanced wreck candidate with redaction analysis"""
    latitude: float
    longitude: float
    size_sq_meters: float
    size_sq_feet: float
    width_meters: float
    height_meters: float
    width_feet: float
    height_feet: float
    confidence: float
    method: str
    processing_time_ms: int
    bounding_box: Tuple[float, float, float, float]
    redaction_signatures: List[RedactionSignature] = None
    elevation_stats: Dict = None
    uncertainty_stats: Dict = None
    anomaly_score: float = 0.0
    shape_complexity: float = 0.0
    depth_gradient: float = 0.0

class AdvancedBagScanner:
    """Advanced BAG scanner with redaction signature detection"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.db_path = self.config.get('database_path', 'wrecks.db')
        self._init_database()

        # Redaction signature patterns
        self.redaction_patterns = {
            'smoothing': {
                'description': 'Artificial smoothing of elevation data',
                'indicators': ['low_variance_regions', 'gradient_anomalies', 'edge_sharpness']
            },
            'removal': {
                'description': 'Complete removal of elevation features',
                'indicators': ['flat_regions', 'data_gaps', 'boundary_artifacts']
            },
            'alteration': {
                'description': 'Modification of existing elevation data',
                'indicators': ['statistical_outliers', 'pattern_breaks', 'metadata_inconsistencies']
            },
            'pattern_overlay': {
                'description': 'Artificial pattern overlay to hide features',
                'indicators': ['repeating_patterns', 'unnatural_symmetry', 'frequency_anomalies']
            }
        }

    def _default_config(self) -> Dict:
        return {
            'min_wreck_size_sq_ft': 25.0,
            'max_wreck_size_sq_ft': 50000.0,
            'min_confidence': 0.3,
            'anomaly_threshold': 2.5,  # Z-score threshold
            'skip_pattern': [5, 10, 15],  # Multi-resolution scanning
            'redaction_sensitivity': 0.7,
            # Redaction signature detection is computationally heavy on large BAG rasters.
            # Keep off by default for interactive scans; can be explicitly enabled.
            'enable_redaction_signatures': False,
            'max_workers': 4,
            'database_path': 'wrecks.db',
            'output_kml': True,
            'output_kmz': True,
            'include_elevation_stats': True,
            'include_uncertainty_analysis': True,
            # Windowed / chunked processing
            'use_windowed_reads': True,
            'window_concurrency': 4,  # Number of windows processed in parallel
            'min_window_area_pixels': 32*32  # Minimum area to consider processing a tile
        }

    def _init_database(self):
        """Initialize database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create enhanced tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bag_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                total_candidates INTEGER DEFAULT 0,
                redaction_signatures INTEGER DEFAULT 0,
                processing_time_ms INTEGER,
                scanner_version TEXT DEFAULT '2.0'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wreck_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                size_sq_meters REAL,
                size_sq_feet REAL,
                width_meters REAL,
                height_meters REAL,
                confidence REAL,
                method TEXT,
                anomaly_score REAL,
                shape_complexity REAL,
                depth_gradient REAL,
                FOREIGN KEY (scan_id) REFERENCES bag_scans(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS redaction_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                signature_type TEXT,
                confidence REAL,
                redactor_id TEXT,
                technique_used TEXT,
                evidence TEXT,  -- JSON
                FOREIGN KEY (candidate_id) REFERENCES wreck_candidates(id)
            )
        ''')

        conn.commit()
        conn.close()

    def scan_bag_file(self, bag_path: str) -> Dict:
        """Scan a single BAG file with enhanced detection"""
        start_time = datetime.now()
        logger.info(f"🔬 Starting advanced scan of {Path(bag_path).name}")

        try:
            with rasterio.open(bag_path) as src:
                # Read all bands
                elevation = src.read(1)
                uncertainty = src.read(2) if src.count >= 2 else None

                # Basic validation
                if elevation.size == 0:
                    raise ValueError("Empty elevation data")

                # Multi-resolution scanning (windowed when enabled)
                if self.config.get('use_windowed_reads'):
                    logger.info("Using windowed reads for chunked processing")
                    candidates = []
                    windows = list(src.block_windows(1))
                    max_workers = min(self.config.get('window_concurrency', 4), len(windows))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for _, win in windows:
                            # Read windowed arrays
                            elev_win = src.read(1, window=win)
                            unc_win = src.read(2, window=win) if src.count >= 2 else None
                            win_transform = rasterio.windows.transform(win, src.transform)

                            # Skip tiny windows
                            if elev_win.size < self.config.get('min_window_area_pixels', 1024):
                                continue

                            futures.append(executor.submit(self._multi_resolution_scan, elev_win, unc_win, win_transform, src.crs))

                        for f in futures:
                            try:
                                res = f.result()
                                candidates.extend(res)
                            except Exception as e:
                                logger.error(f"Window processing failed: {e}")

                    # Merge overlapping candidates across windows
                    candidates = self._merge_overlapping_candidates(candidates)

                    # Redaction signature analysis is optional (expensive on large rasters).
                    signatures = (
                        self._analyze_redaction_signatures(elevation, uncertainty, src.transform)
                        if self.config.get('enable_redaction_signatures', False)
                        else []
                    )
                else:
                    candidates = self._multi_resolution_scan(elevation, uncertainty, src.transform, src.crs)

                    # Redaction signature analysis is optional (expensive on large rasters).
                    signatures = (
                        self._analyze_redaction_signatures(elevation, uncertainty, src.transform)
                        if self.config.get('enable_redaction_signatures', False)
                        else []
                    )

                # Correlate candidates with signatures
                enhanced_candidates = self._correlate_candidates_signatures(candidates, signatures)

                # Filter and rank candidates
                final_candidates = self._filter_and_rank_candidates(enhanced_candidates)

                # Calculate processing time
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

                # Store results in database
                scan_id = self._store_scan_results(bag_path, final_candidates, signatures, processing_time)

                # Generate outputs (pass raw grids for 3D viewer generation)
                outputs = self._generate_outputs(bag_path, final_candidates, signatures,
                                                  uncertainty=uncertainty, elevation=elevation)

                result = {
                    'file': Path(bag_path).name,
                    'scan_id': scan_id,
                    'candidates': [asdict(c) for c in final_candidates],
                    'redaction_signatures': [asdict(s) for s in signatures],
                    'total_candidates': len(final_candidates),
                    'total_signatures': len(signatures),
                    'processing_time_ms': processing_time,
                    'outputs': outputs,
                    'success': True
                }

                logger.info(f"✅ Completed scan: {len(final_candidates)} candidates, {len(signatures)} signatures")
                return result

        except Exception as e:
            logger.error(f"❌ Scan failed for {bag_path}: {e}")
            return {
                'file': Path(bag_path).name,
                'candidates': [],
                'redaction_signatures': [],
                'total_candidates': 0,
                'total_signatures': 0,
                'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
                'error': str(e),
                'success': False
            }

    def _multi_resolution_scan(self, elevation: np.ndarray, uncertainty: Optional[np.ndarray],
                             transform, crs) -> List[WreckCandidate]:
        """Multi-resolution scanning for better detection"""
        candidates = []

        for skip_step in self.config['skip_pattern']:
            logger.debug(f"Scanning with skip pattern: {skip_step}")

            # Detect anomalies at this resolution
            resolution_candidates = self._detect_anomalies_at_resolution(
                elevation, uncertainty, transform, skip_step, crs
            )

            candidates.extend(resolution_candidates)

        # Remove duplicates and merge overlapping detections
        return self._merge_overlapping_candidates(candidates)

    def _detect_anomalies_at_resolution(self, elevation: np.ndarray, uncertainty: Optional[np.ndarray],
                                       transform, skip_step: int, crs) -> List[WreckCandidate]:
        """Detect anomalies at a specific resolution"""
        candidates = []

        # Calculate statistics
        valid_mask = ~np.ma.masked_invalid(elevation).mask
        if elevation.dtype == np.float32 and np.any(np.abs(elevation) > 1e10):
            # Handle BAG nodata values
            valid_mask = valid_mask & (np.abs(elevation) < 1e10)

        if not valid_mask.any():
            return candidates

        valid_elevation = elevation[valid_mask]
        mean_val = np.mean(valid_elevation)
        std_val = np.std(valid_elevation)

        # Scan with skip pattern
        rows, cols = elevation.shape

        for i in range(0, rows, skip_step):
            for j in range(0, cols, skip_step):
                if not valid_mask[i, j]:
                    continue

                # Calculate local anomaly score
                local_window = elevation[max(0, i-5):min(rows, i+6), max(0, j-5):min(cols, j+6)]
                local_valid = local_window[valid_mask[max(0, i-5):min(rows, i+6), max(0, j-5):min(cols, j+6)]]

                if len(local_valid) < 9:  # Need minimum local samples
                    continue

                local_mean = np.mean(local_valid)
                local_std = np.std(local_valid)
                local_z = abs(elevation[i, j] - local_mean) / (local_std + 1e-6)

                # Enhanced anomaly detection
                if local_z > self.config['anomaly_threshold']:
                    # Grow region around anomaly
                    region = self._grow_anomaly_region(elevation, valid_mask, i, j, mean_val, std_val)

                    if region['size_pixels'] >= 10:  # Minimum size
                        candidate = self._create_candidate_from_region(region, transform, elevation, uncertainty, crs)
                        if candidate:
                            candidates.append(candidate)

        return candidates

    def _grow_anomaly_region(self, elevation: np.ndarray, valid_mask: np.ndarray,
                           start_i: int, start_j: int, global_mean: float, global_std: float) -> Dict:
        """Grow a connected region of anomalies"""
        rows, cols = elevation.shape
        visited = np.zeros_like(valid_mask, dtype=bool)
        region_pixels = []

        to_visit = [(start_i, start_j)]
        visited[start_i, start_j] = True

        while to_visit:
            i, j = to_visit.pop()

            if not valid_mask[i, j]:
                continue

            z_score = abs(elevation[i, j] - global_mean) / global_std
            if z_score > 1.5:  # Connected anomaly threshold
                region_pixels.append((i, j))

                # Add neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and
                            not visited[ni, nj] and valid_mask[ni, nj]):
                            visited[ni, nj] = True
                            to_visit.append((ni, nj))

        if not region_pixels:
            return {'size_pixels': 0}

        # Calculate region properties
        pixel_coords = np.array(region_pixels)
        min_i, min_j = pixel_coords.min(axis=0)
        max_i, max_j = pixel_coords.max(axis=0)

        return {
            'size_pixels': len(region_pixels),
            'bounding_box_pixels': (min_i, min_j, max_i, max_j),
            'centroid_pixels': pixel_coords.mean(axis=0),
            'pixel_coords': region_pixels
        }

    def _create_candidate_from_region(self, region: Dict, transform, elevation: np.ndarray,
                                    uncertainty: Optional[np.ndarray], crs=None) -> Optional[WreckCandidate]:
        """Create a wreck candidate from a detected region"""
        if region['size_pixels'] < 10:
            return None

        centroid_i, centroid_j = region['centroid_pixels']
        min_i, min_j, max_i, max_j = region['bounding_box_pixels']

        # Map pixel coordinates to spatial coordinates (handle projected and geographic CRS)
        try:
            # Map pixels to coordinate space using affine transform
            center_x = transform.c + centroid_j * transform.a + centroid_i * transform.b
            center_y = transform.f + centroid_j * transform.d + centroid_i * transform.e

            # Convert bounding box corners
            corners = [
                (min_j, min_i), (max_j, min_i), (min_j, max_i), (max_j, max_i)
            ]
            corner_coords = []
            for cj, ci in corners:
                x = transform.c + cj * transform.a + ci * transform.b
                y = transform.f + cj * transform.d + ci * transform.e
                corner_coords.append((x, y))

        except Exception:
            # Fallback to pixel coordinates
            center_x, center_y = centroid_j, centroid_i
            corner_coords = [(min_j, min_i), (max_j, min_i), (min_j, max_i), (max_j, max_i)]

        # If CRS is geographic (degrees), transform to a metric CRS for accurate size computation
        if crs and getattr(crs, 'is_geographic', False):
            try:
                src_crs = str(crs)
                transformer = Transformer.from_crs(src_crs, "EPSG:3857", always_xy=True)
                easting, northing = transformer.transform(center_x, center_y)
                corner_m = [transformer.transform(x, y) for x, y in corner_coords]
                min_e = min(c[0] for c in corner_m)
                max_e = max(c[0] for c in corner_m)
                min_n = min(c[1] for c in corner_m)
                max_n = max(c[1] for c in corner_m)
            except Exception:
                # Approximate conversion if transformer fails
                lat_deg = center_y
                meters_per_deg_lon = 111320 * math.cos(math.radians(lat_deg))
                meters_per_deg_lat = 110574
                corner_m = [(x * meters_per_deg_lon, y * meters_per_deg_lat) for x, y in corner_coords]
                min_e = min(c[0] for c in corner_m)
                max_e = max(c[0] for c in corner_m)
                min_n = min(c[1] for c in corner_m)
                max_n = max(c[1] for c in corner_m)
                easting, northing = (center_x * meters_per_deg_lon, center_y * meters_per_deg_lat)
        else:
            # treat as projected (meters)
            try:
                easting, northing = center_x, center_y
                min_e = min(c[0] for c in corner_coords)
                max_e = max(c[0] for c in corner_coords)
                min_n = min(c[1] for c in corner_coords)
                max_n = max(c[1] for c in corner_coords)
            except Exception:
                # fallback to pixel coords
                easting, northing = centroid_j, centroid_i
                min_e, max_e = min_j, max_j
                min_n, max_n = min_i, max_i

        # Calculate dimensions
        width_m = abs(max_e - min_e)
        height_m = abs(max_n - min_n)
        area_m2 = width_m * height_m

        # Convert units
        area_ft2 = area_m2 * 10.7639
        width_ft = width_m * 3.28084
        height_ft = height_m * 3.28084

        # Size validation
        if not (self.config['min_wreck_size_sq_ft'] <= area_ft2 <= self.config['max_wreck_size_sq_ft']):
            return None

        # Calculate confidence based on size and shape
        size_confidence = min(1.0, area_ft2 / 1000.0)  # Larger features more confident
        shape_confidence = min(1.0, region['size_pixels'] / 200.0)  # More pixels more confident
        confidence = (size_confidence + shape_confidence) / 2.0

        # Calculate additional metrics
        elevation_stats = self._calculate_elevation_stats(elevation, region['pixel_coords'])
        uncertainty_stats = self._calculate_uncertainty_stats(uncertainty, region['pixel_coords']) if uncertainty is not None else {}

        # Calculate shape complexity (perimeter/area ratio)
        shape_complexity = self._calculate_shape_complexity(region['pixel_coords'])

        # Calculate depth gradient
        depth_gradient = self._calculate_depth_gradient(elevation, region['pixel_coords'])

        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(elevation, region['pixel_coords'])

        return WreckCandidate(
            latitude=northing,
            longitude=easting,
            size_sq_meters=area_m2,
            size_sq_feet=area_ft2,
            width_meters=width_m,
            height_meters=height_m,
            width_feet=width_ft,
            height_feet=height_ft,
            confidence=confidence,
            method="advanced_multi_resolution",
            processing_time_ms=0,  # Will be set later
            bounding_box=(min_e, min_n, max_e, max_n),
            elevation_stats=elevation_stats,
            uncertainty_stats=uncertainty_stats,
            anomaly_score=anomaly_score,
            shape_complexity=shape_complexity,
            depth_gradient=depth_gradient
        )

    def _calculate_elevation_stats(self, elevation: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> Dict:
        """Calculate elevation statistics for a region"""
        elevations = [elevation[i, j] for i, j in pixel_coords if 0 <= i < elevation.shape[0] and 0 <= j < elevation.shape[1]]

        if not elevations:
            return {}

        elevations = np.array(elevations)
        return {
            'mean': float(np.mean(elevations)),
            'std': float(np.std(elevations)),
            'min': float(np.min(elevations)),
            'max': float(np.max(elevations)),
            'range': float(np.max(elevations) - np.min(elevations)),
            'median': float(np.median(elevations))
        }

    def _calculate_uncertainty_stats(self, uncertainty: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> Dict:
        """Calculate uncertainty statistics for a region"""
        if uncertainty is None:
            return {}

        uncertainties = [uncertainty[i, j] for i, j in pixel_coords if 0 <= i < uncertainty.shape[0] and 0 <= j < uncertainty.shape[1]]

        if not uncertainties:
            return {}

        uncertainties = np.array(uncertainties)
        return {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'min': float(np.min(uncertainties)),
            'max': float(np.max(uncertainties)),
            'median': float(np.median(uncertainties))
        }

    def _calculate_shape_complexity(self, pixel_coords: List[Tuple[int, int]]) -> float:
        """Calculate shape complexity (perimeter to area ratio)"""
        if len(pixel_coords) < 4:
            return 0.0

        # Simple perimeter calculation
        perimeter = 0
        coords_set = set(pixel_coords)

        for i, j in pixel_coords:
            # Check 4-connected neighbors
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ni, nj in neighbors:
                if (ni, nj) not in coords_set:
                    perimeter += 1

        area = len(pixel_coords)
        return perimeter / area if area > 0 else 0.0

    def _calculate_depth_gradient(self, elevation: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> float:
        """Calculate average depth gradient in the region"""
        if len(pixel_coords) < 4:
            return 0.0

        gradients = []
        for i, j in pixel_coords:
            # Calculate local gradient
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < elevation.shape[0] and 0 <= nj < elevation.shape[1]:
                        neighbors.append(elevation[ni, nj])

            if neighbors:
                local_gradient = np.std([elevation[i, j]] + neighbors)
                gradients.append(local_gradient)

        return float(np.mean(gradients)) if gradients else 0.0

    def _calculate_anomaly_score(self, elevation: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> float:
        """Calculate anomaly score for the region"""
        if not pixel_coords:
            return 0.0

        # Compare region stats to global stats
        region_elevations = np.array([elevation[i, j] for i, j in pixel_coords])

        # Simple global stats (in practice, you'd want more sophisticated background modeling)
        all_valid = elevation[~np.ma.masked_invalid(elevation).mask]
        if len(all_valid) == 0:
            return 0.0

        global_mean = np.mean(all_valid)
        global_std = np.std(all_valid)

        region_mean = np.mean(region_elevations)
        region_std = np.std(region_elevations)

        # Anomaly based on deviation from global statistics
        mean_anomaly = abs(region_mean - global_mean) / (global_std + 1e-6)
        std_anomaly = region_std / (global_std + 1e-6)

        return float((mean_anomaly + std_anomaly) / 2.0)

    def _merge_overlapping_candidates(self, candidates: List[WreckCandidate]) -> List[WreckCandidate]:
        """Merge overlapping or nearby candidates"""
        if len(candidates) <= 1:
            return candidates

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        merged = []

        for candidate in candidates:
            # Check if this candidate overlaps significantly with any merged candidate
            overlaps = False
            for merged_candidate in merged:
                if self._candidates_overlap(candidate, merged_candidate):
                    overlaps = True
                    # Merge: keep the higher confidence one, but combine stats
                    if candidate.confidence > merged_candidate.confidence:
                        # Replace with higher confidence candidate
                        merged.remove(merged_candidate)
                        merged.append(candidate)
                    break

            if not overlaps:
                merged.append(candidate)

        return merged

    def _candidates_overlap(self, c1: WreckCandidate, c2: WreckCandidate, threshold: float = 0.3) -> bool:
        """Check if two candidates overlap significantly"""
        # Simple bounding box overlap check
        min_lon1, min_lat1, max_lon1, max_lat1 = c1.bounding_box
        min_lon2, min_lat2, max_lon2, max_lat2 = c2.bounding_box

        # Calculate overlap area
        overlap_lon = max(0, min(max_lon1, max_lon2) - max(min_lon1, min_lon2))
        overlap_lat = max(0, min(max_lat1, max_lat2) - max(min_lat1, min_lat2))
        overlap_area = overlap_lon * overlap_lat

        # Calculate union area
        area1 = (max_lon1 - min_lon1) * (max_lat1 - min_lat1)
        area2 = (max_lon2 - min_lon2) * (max_lat2 - min_lat2)
        union_area = area1 + area2 - overlap_area

        if union_area == 0:
            return False

        iou = overlap_area / union_area
        return iou > threshold

    def _analyze_redaction_signatures(self, elevation: np.ndarray, uncertainty: Optional[np.ndarray],
                                    transform) -> List[RedactionSignature]:
        """Analyze the BAG file for redaction signatures"""
        signatures = []

        # Analyze for different types of redaction signatures
        signatures.extend(self._detect_smoothing_signatures(elevation, transform))
        signatures.extend(self._detect_removal_signatures(elevation, transform))
        signatures.extend(self._detect_alteration_signatures(elevation, uncertainty, transform))
        signatures.extend(self._detect_pattern_signatures(elevation, transform))

        # Attempt to identify redactors based on signature patterns
        signatures = self._identify_redactors(signatures)

        return signatures

    def _detect_smoothing_signatures(self, elevation: np.ndarray, transform) -> List[RedactionSignature]:
        """Detect artificial smoothing signatures"""
        signatures = []

        # Look for regions with unnaturally low variance
        kernel_size = 20
        rows, cols = elevation.shape

        for i in range(0, rows - kernel_size, kernel_size // 2):
            for j in range(0, cols - kernel_size, kernel_size // 2):
                window = elevation[i:i+kernel_size, j:j+kernel_size]
                valid_data = window[~np.ma.masked_invalid(window).mask]

                if len(valid_data) < kernel_size * kernel_size * 0.8:  # 80% valid data
                    continue

                local_std = np.std(valid_data)
                local_mean = np.mean(valid_data)

                # Compare to expected variance based on depth
                expected_std = max(0.1, abs(local_mean) * 0.01)  # Rough estimate

                if local_std < expected_std * 0.5:  # Unnaturally smooth
                    # Convert window center to coordinates
                    center_i, center_j = i + kernel_size // 2, j + kernel_size // 2

                    try:
                        lon = transform.c + center_j * transform.a + center_i * transform.b
                        lat = transform.f + center_j * transform.d + center_i * transform.e
                    except:
                        lat, lon = center_i, center_j  # Fallback

                    bbox_size = kernel_size * abs(transform.a)  # Approximate meters
                    bbox = (lon - bbox_size/2, lat - bbox_size/2, lon + bbox_size/2, lat + bbox_size/2)

                    confidence = min(1.0, (expected_std / (local_std + 1e-6) - 1) / 10.0)

                    if confidence > self.config['redaction_sensitivity']:
                        signatures.append(RedactionSignature(
                            signature_type='smoothing',
                            confidence=confidence,
                            location=(lat, lon),
                            bounding_box=bbox,
                            size_pixels=kernel_size * kernel_size,
                            size_meters_sq=bbox_size * bbox_size,
                            technique_used='artificial_smoothing',
                            evidence={
                                'local_std': float(local_std),
                                'expected_std': float(expected_std),
                                'variance_ratio': float(local_std / expected_std)
                            }
                        ))

        return signatures

    def _detect_removal_signatures(self, elevation: np.ndarray, transform) -> List[RedactionSignature]:
        """Detect feature removal signatures"""
        signatures = []

        # Look for flat regions that interrupt natural depth variation
        rows, cols = elevation.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if np.ma.is_masked(elevation[i, j]):
                    continue

                # Check if this pixel is unnaturally flat compared to neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.ma.is_masked(elevation[ni, nj]):
                            neighbors.append(elevation[ni, nj])

                if len(neighbors) < 4:
                    continue

                neighbor_std = np.std(neighbors)
                center_val = elevation[i, j]

                # Look for pixels that are too similar to neighbors (potential fill/flat)
                if neighbor_std > 0.1:  # Only check in variable areas
                    similarity = 1.0 - abs(center_val - np.mean(neighbors)) / (neighbor_std * 2)
                    if similarity > 0.95:  # Very similar to neighbors
                        try:
                            lon = transform.c + j * transform.a + i * transform.b
                            lat = transform.f + j * transform.d + i * transform.e
                        except:
                            lat, lon = i, j

                        confidence = similarity

                        if confidence > self.config['redaction_sensitivity']:
                            signatures.append(RedactionSignature(
                                signature_type='removal',
                                confidence=confidence,
                                location=(lat, lon),
                                bounding_box=(lon-5, lat-5, lon+5, lat+5),  # Small region
                                size_pixels=1,
                                size_meters_sq=25,  # 5x5 meters
                                technique_used='feature_flattening',
                                evidence={
                                    'center_value': float(center_val),
                                    'neighbor_mean': float(np.mean(neighbors)),
                                    'neighbor_std': float(neighbor_std),
                                    'similarity': float(similarity)
                                }
                            ))

        return signatures

    def _detect_alteration_signatures(self, elevation: np.ndarray, uncertainty: Optional[np.ndarray],
                                    transform) -> List[RedactionSignature]:
        """Detect data alteration signatures"""
        signatures = []

        if uncertainty is None:
            return signatures

        # Look for inconsistencies between elevation and uncertainty
        rows, cols = elevation.shape

        for i in range(0, rows, 10):  # Sample every 10 pixels
            for j in range(0, cols, 10):
                if (np.ma.is_masked(elevation[i, j]) or
                    np.ma.is_masked(uncertainty[i, j])):
                    continue

                elev_val = elevation[i, j]
                uncert_val = uncertainty[i, j]

                # Check for unnatural uncertainty patterns
                # High uncertainty in areas that should be well-surveyed
                if abs(elev_val) < 100:  # Shallow water
                    expected_uncertainty = max(0.1, abs(elev_val) * 0.01)
                    if uncert_val > expected_uncertainty * 3:  # Much higher than expected
                        try:
                            lon = transform.c + j * transform.a + i * transform.b
                            lat = transform.f + j * transform.d + i * transform.e
                        except:
                            lat, lon = i, j

                        confidence = min(1.0, uncert_val / (expected_uncertainty + 1e-6) / 5.0)

                        if confidence > self.config['redaction_sensitivity']:
                            signatures.append(RedactionSignature(
                                signature_type='alteration',
                                confidence=confidence,
                                location=(lat, lon),
                                bounding_box=(lon-10, lat-10, lon+10, lat+10),
                                size_pixels=100,  # 10x10 pixel region
                                size_meters_sq=100,  # Approximate
                                technique_used='uncertainty_manipulation',
                                evidence={
                                    'elevation': float(elev_val),
                                    'uncertainty': float(uncert_val),
                                    'expected_uncertainty': float(expected_uncertainty),
                                    'ratio': float(uncert_val / expected_uncertainty)
                                }
                            ))

        return signatures

    def _detect_pattern_signatures(self, elevation: np.ndarray, transform) -> List[RedactionSignature]:
        """Detect artificial pattern signatures"""
        signatures = []

        # Look for repeating patterns that indicate artificial overlay
        rows, cols = elevation.shape

        # Use 2D autocorrelation to detect repeating patterns
        try:
            from scipy import signal

            # Sample a region
            sample_size = min(100, rows, cols)
            sample = elevation[:sample_size, :sample_size]
            sample = sample[~np.ma.masked_invalid(sample).mask]

            if sample.size < sample_size * sample_size * 0.5:
                return signatures

            # Calculate autocorrelation
            corr = signal.correlate2d(sample, sample, mode='full')
            corr = corr[corr.shape[0]//2:, corr.shape[1]//2:]  # Second half

            # Look for strong secondary peaks indicating periodicity
            max_corr = np.max(corr)
            mean_corr = np.mean(corr)

            if max_corr > mean_corr * 3:  # Strong periodicity
                # Find peak locations
                peaks = np.where(corr > max_corr * 0.8)

                if len(peaks[0]) > 1:  # Multiple peaks
                    period_y = np.mean(np.diff(np.sort(peaks[0])))
                    period_x = np.mean(np.diff(np.sort(peaks[1])))

                    if 5 < period_y < 50 and 5 < period_x < 50:  # Reasonable period
                        confidence = min(1.0, (max_corr / mean_corr - 1) / 10.0)

                        if confidence > self.config['redaction_sensitivity']:
                            # Location is center of sampled region
                            center_i, center_j = sample_size // 2, sample_size // 2
                            try:
                                lon = transform.c + center_j * transform.a + center_i * transform.b
                                lat = transform.f + center_j * transform.d + center_i * transform.e
                            except:
                                lat, lon = center_i, center_j

                            bbox_size = sample_size * abs(transform.a)
                            bbox = (lon - bbox_size/2, lat - bbox_size/2, lon + bbox_size/2, lat + bbox_size/2)

                            signatures.append(RedactionSignature(
                                signature_type='pattern_overlay',
                                confidence=confidence,
                                location=(lat, lon),
                                bounding_box=bbox,
                                size_pixels=sample_size * sample_size,
                                size_meters_sq=bbox_size * bbox_size,
                                technique_used='artificial_pattern_overlay',
                                evidence={
                                    'period_x': float(period_x),
                                    'period_y': float(period_y),
                                    'max_correlation': float(max_corr),
                                    'mean_correlation': float(mean_corr)
                                }
                            ))

        except ImportError:
            # scipy not available, skip pattern detection
            pass

        return signatures

    def _identify_redactors(self, signatures: List[RedactionSignature]) -> List[RedactionSignature]:
        """Attempt to identify different redactors based on signature patterns"""
        if len(signatures) < 2:
            return signatures

        # Group signatures by technique and location patterns
        technique_groups = {}
        for sig in signatures:
            key = sig.technique_used
            if key not in technique_groups:
                technique_groups[key] = []
            technique_groups[key].append(sig)

        # Assign redactor IDs based on technique clustering
        redactor_id = 1
        for technique, sigs in technique_groups.items():
            if len(sigs) >= 3:  # Multiple signatures of same technique
                # Calculate centroid of signature locations
                lats = [s.location[0] for s in sigs]
                lons = [s.location[1] for s in sigs]
                centroid_lat = sum(lats) / len(lats)
                centroid_lon = sum(lons) / len(lons)

                # Assign redactor ID to signatures near centroid
                for sig in sigs:
                    distance = ((sig.location[0] - centroid_lat)**2 +
                              (sig.location[1] - centroid_lon)**2)**0.5
                    if distance < 0.01:  # Within ~1km
                        sig.redactor_id = f"redactor_{redactor_id}"

                redactor_id += 1

        return signatures

    def _correlate_candidates_signatures(self, candidates: List[WreckCandidate],
                                       signatures: List[RedactionSignature]) -> List[WreckCandidate]:
        """Correlate wreck candidates with redaction signatures"""
        for candidate in candidates:
            candidate.redaction_signatures = []

            # Find signatures within candidate bounding box
            min_lon, min_lat, max_lon, max_lat = candidate.bounding_box

            for sig in signatures:
                sig_lat, sig_lon = sig.location
                if (min_lon <= sig_lon <= max_lon and
                    min_lat <= sig_lat <= max_lat):
                    candidate.redaction_signatures.append(sig)

        return candidates

    def _filter_and_rank_candidates(self, candidates: List[WreckCandidate]) -> List[WreckCandidate]:
        """Filter and rank candidates by confidence and redaction evidence"""
        # Boost confidence for candidates with redaction signatures
        for candidate in candidates:
            if candidate.redaction_signatures:
                # Increase confidence based on signature strength
                signature_boost = sum(s.confidence for s in candidate.redaction_signatures) / len(candidate.redaction_signatures)
                candidate.confidence = min(1.0, candidate.confidence + signature_boost * 0.3)

        # Filter by minimum confidence
        filtered = [c for c in candidates if c.confidence >= self.config['min_confidence']]

        # Sort by confidence (highest first)
        filtered.sort(key=lambda c: c.confidence, reverse=True)

        return filtered

    def _store_scan_results(self, bag_path: str, candidates: List[WreckCandidate],
                          signatures: List[RedactionSignature], processing_time: int) -> int:
        """Store scan results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert scan record
        cursor.execute('''
            INSERT INTO bag_scans (file_name, scan_timestamp, total_candidates, redaction_signatures, processing_time_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            Path(bag_path).name,
            datetime.now().isoformat(),
            len(candidates),
            len(signatures),
            processing_time
        ))

        scan_id = cursor.lastrowid

        # Insert candidates
        for candidate in candidates:
            cursor.execute('''
                INSERT INTO wreck_candidates
                (scan_id, latitude, longitude, size_sq_meters, size_sq_feet, width_meters, height_meters,
                 confidence, method, anomaly_score, shape_complexity, depth_gradient)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_id,
                candidate.latitude,
                candidate.longitude,
                candidate.size_sq_meters,
                candidate.size_sq_feet,
                candidate.width_meters,
                candidate.height_meters,
                candidate.confidence,
                candidate.method,
                candidate.anomaly_score,
                candidate.shape_complexity,
                candidate.depth_gradient
            ))

            candidate_id = cursor.lastrowid

            # Insert redaction signatures for this candidate
            if candidate.redaction_signatures:
                for sig in candidate.redaction_signatures:
                    cursor.execute('''
                        INSERT INTO redaction_signatures
                        (candidate_id, signature_type, confidence, redactor_id, technique_used, evidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        candidate_id,
                        sig.signature_type,
                        sig.confidence,
                        sig.redactor_id,
                        sig.technique_used,
                        json.dumps(sig.evidence) if sig.evidence else None
                    ))

        conn.commit()
        conn.close()

        return scan_id

    def _generate_outputs(self, bag_path: str, candidates: List[WreckCandidate],
                        signatures: List[RedactionSignature],
                        uncertainty: np.ndarray = None,
                        elevation: np.ndarray = None) -> Dict[str, str]:
        """Generate KML and KMZ outputs"""
        outputs = {}

        if not candidates:
            return outputs

        base_name = Path(bag_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate KML
        if self.config['output_kml']:
            kml_path = f"{base_name}_advanced_scan_{timestamp}.kml"
            self._generate_kml(kml_path, candidates, signatures, base_name)
            outputs['kml'] = kml_path

        # Generate KMZ
        if self.config['output_kmz']:
            kmz_path = f"{base_name}_advanced_scan_{timestamp}.kmz"
            self._generate_kmz(kmz_path, candidates, signatures, base_name,
                              uncertainty=uncertainty, elevation=elevation)
            outputs['kmz'] = kmz_path

        return outputs

    def _generate_kml(self, kml_path: str, candidates: List[WreckCandidate],
                     signatures: List[RedactionSignature], file_name: str,
                     viewer_indices: set = None):
        """Generate KML file with candidates and signatures"""
        kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Advanced BAG Scan Results - {file_name}</name>
    <description>Enhanced wreck detection with redaction signature analysis.
IMPORTANT: All positions are PREDICTED / ESTIMATED from BAG anomaly detection.
True wreck locations are unknown. Field verification is required to confirm any location.</description>

    <!-- Styles -->
    <Style id="wreckStyle">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/danger.png</href></Icon>
        <scale>1.2</scale>
      </IconStyle>
      <LabelStyle><scale>0.8</scale></LabelStyle>
    </Style>

    <Style id="highConfidenceStyle">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/danger.png</href></Icon>
        <scale>1.5</scale>
        <color>ff0000ff</color>
      </IconStyle>
      <LabelStyle><scale>1.0</scale><color>ff0000ff</color></LabelStyle>
    </Style>

    <Style id="signatureStyle">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href></Icon>
        <scale>0.8</scale>
        <color>ffff00ff</color>
      </IconStyle>
      <LabelStyle><scale>0.6</scale><color>ffff00ff</color></LabelStyle>
    </Style>
'''

        # Add folders for organization
        kml_content += '''
    <Folder>
      <name>Wreck Candidates</name>
      <description>Potential shipwreck locations with enhanced analysis</description>
'''

        for i, candidate in enumerate(candidates):
            style = "#highConfidenceStyle" if candidate.confidence > 0.8 else "#wreckStyle"

            # Build description
            desc = f"""
      <![CDATA[
      <div style="font-family: Arial; max-width: 400px;">
        <h3>Wreck Candidate #{i+1}</h3>
        <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:4px;padding:6px 10px;margin-bottom:8px;color:#856404;font-size:12px;">
          \u26a0\ufe0f <b>PREDICTED / ESTIMATED LOCATION</b> &mdash; true position unknown. Field verification required.
        </div>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td><b>Location Status:</b></td><td>Estimated (algorithmic detection)</td></tr>
          <tr><td><b>Confidence:</b></td><td>{candidate.confidence:.3f}</td></tr>
          <tr><td><b>Size:</b></td><td>{candidate.size_sq_meters:.1f} m² ({candidate.size_sq_feet:.1f} ft²)</td></tr>
          <tr><td><b>Dimensions:</b></td><td>{candidate.width_meters:.1f}m × {candidate.height_meters:.1f}m</td></tr>
          <tr><td><b>Anomaly Score:</b></td><td>{candidate.anomaly_score:.3f}</td></tr>
          <tr><td><b>Shape Complexity:</b></td><td>{candidate.shape_complexity:.3f}</td></tr>
          <tr><td><b>Depth Gradient:</b></td><td>{candidate.depth_gradient:.3f}</td></tr>
"""

            if candidate.elevation_stats:
                desc += f"""
          <tr><td colspan="2"><b>Elevation Stats:</b></td></tr>
          <tr><td>Mean:</td><td>{candidate.elevation_stats.get('mean', 'N/A'):.2f}</td></tr>
          <tr><td>Range:</td><td>{candidate.elevation_stats.get('range', 'N/A'):.2f}</td></tr>
"""

            if candidate.redaction_signatures:
                desc += f"""
          <tr><td colspan="2"><b>Redaction Signatures: {len(candidate.redaction_signatures)}</b></td></tr>
"""
                for sig in candidate.redaction_signatures[:3]:  # Show top 3
                    desc += f"""
          <tr><td>{sig.signature_type}:</td><td>{sig.confidence:.3f}</td></tr>
"""

            desc += """
        </table>
"""
            # Add 3D viewer link if available for this candidate
            if viewer_indices and i in viewer_indices:
                desc += f"""
        <div style="margin-top:10px;text-align:center;padding:8px;background:#0a1628;border:1px solid #1a5276;border-radius:6px;">
          <a href="viewers/candidate_{i}.html"
             style="color:#5dade2;text-decoration:none;font-weight:bold;font-size:13px;">
            &#x1f50d; Open Interactive 3D Model Viewer
          </a>
          <div style="font-size:10px;color:#7f8c8d;margin-top:4px;">
            Opens in browser &bull; Babylon.js WebGL &bull; Orbit/rotate/zoom
          </div>
        </div>
"""

            desc += """
      </div>
      ]]>
"""

            kml_content += f"""
      <Placemark>
        <name>Candidate #{i+1} ({candidate.confidence:.2f}) [ESTIMATED]</name>
        <description>{desc}</description>
        <styleUrl>{style}</styleUrl>
        <Point>
          <coordinates>{candidate.longitude},{candidate.latitude},0</coordinates>
        </Point>
      </Placemark>
"""

        kml_content += """
    </Folder>
"""

        # Add redaction signatures folder
        if signatures:
            kml_content += """
    <Folder>
      <name>Redaction Signatures</name>
      <description>Detected redaction and data manipulation signatures</description>
"""

            for i, sig in enumerate(signatures):
                desc = f"""
      <![CDATA[
      <div style="font-family: Arial; max-width: 350px;">
        <h3>Redaction Signature #{i+1}</h3>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td><b>Type:</b></td><td>{sig.signature_type}</td></tr>
          <tr><td><b>Confidence:</b></td><td>{sig.confidence:.3f}</td></tr>
          <tr><td><b>Technique:</b></td><td>{sig.technique_used}</td></tr>
"""
                if sig.redactor_id:
                    desc += f"""
          <tr><td><b>Redactor ID:</b></td><td>{sig.redactor_id}</td></tr>
"""
                if sig.evidence:
                    desc += f"""
          <tr><td colspan="2"><b>Evidence:</b></td></tr>
"""
                    for key, value in list(sig.evidence.items())[:5]:  # Show top 5 evidence items
                        desc += f"""
          <tr><td>{key}:</td><td>{value}</td></tr>
"""

                desc += """
        </table>
      </div>
      ]]>
"""

                kml_content += f"""
      <Placemark>
        <name>Signature #{i+1} ({sig.signature_type})</name>
        <description>{desc}</description>
        <styleUrl>#signatureStyle</styleUrl>
        <Point>
          <coordinates>{sig.location[1]},{sig.location[0]},0</coordinates>
        </Point>
      </Placemark>
"""

            kml_content += """
    </Folder>
"""

        kml_content += """
  </Document>
</kml>
"""

        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)

        logger.info(f"Generated KML file: {kml_path}")

    def _generate_kmz(self, kmz_path: str, candidates: List[WreckCandidate],
                     signatures: List[RedactionSignature], file_name: str,
                     uncertainty: np.ndarray = None, elevation: np.ndarray = None):
        """Generate KMZ file with embedded 3D Babylon.js viewers.

        If uncertainty and elevation grids are provided, generates interactive
        3D model viewers (Babylon.js + GLB) for each candidate and bundles
        them inside the KMZ alongside the KML.

        KMZ structure:
          doc.kml             — main KML with placemarks
          viewers/candidate_N.html — self-contained Babylon.js 3D viewers
          models/candidate_N.glb  — standalone GLB mesh files
        """
        from bag_processor.kml_3d_viewer import (
            build_wreck_mesh, build_seafloor_mesh, generate_viewer_html
        )

        viewer_data = {}  # idx -> (html, glb_bytes)
        viewer_indices = set()

        # Generate 3D viewers if grids are available
        if uncertainty is not None and elevation is not None:
            try:
                NODATA = 999_000.0
                valid = (uncertainty > 0) & (uncertainty < NODATA) & np.isfinite(uncertainty)
                vals = uncertainty[valid]
                if len(vals) > 100:
                    p5 = np.percentile(vals, 5)

                    # Detect low-uncertainty regions (masked objects)
                    mask = valid & (uncertainty < p5)
                    mask = ndimage.binary_dilation(mask, iterations=3)
                    mask = ndimage.binary_erosion(mask, iterations=3)
                    labels, n_labels = ndimage.label(mask)
                    all_slices = ndimage.find_objects(labels)
                    sizes = ndimage.sum(mask, labels, range(1, n_labels + 1))

                    cell_size = 0.5  # default BAG cell

                    for i, cand in enumerate(candidates):
                        try:
                            # Convert candidate lat/lon to row/col
                            # Use rasterio-style: row = (lat - origin_lat) / cell
                            # For now, search all labels for nearest centroid
                            best_label = None
                            best_dist = float('inf')

                            cand_row = None
                            cand_col = None

                            # Try using the candidate's known grid position if available
                            # Otherwise estimate from relative position in grid
                            for lid in range(1, min(n_labels + 1, 2000)):
                                sl = all_slices[lid - 1]
                                if sl is None:
                                    continue
                                sz = sizes[lid - 1]
                                # Filter: 50-5000 m2 and reasonable aspect ratio
                                area_m2 = sz * cell_size * cell_size
                                if area_m2 < 30 or area_m2 > 10000:
                                    continue
                                h = sl[0].stop - sl[0].start
                                w = sl[1].stop - sl[1].start
                                aspect = max(h, w) / max(min(h, w), 1)
                                if aspect > 15:
                                    continue

                                # Centroid of this label region
                                cr = (sl[0].start + sl[0].stop) / 2
                                cc = (sl[1].start + sl[1].stop) / 2

                                # Match by candidate size similarity
                                cand_area = cand.size_sq_meters
                                area_ratio = area_m2 / max(cand_area, 1)
                                if area_ratio < 0.1 or area_ratio > 10:
                                    continue

                                # Use dimensional similarity as distance
                                cand_dim = max(cand.width_meters, cand.height_meters)
                                label_dim = max(h, w) * cell_size
                                dim_diff = abs(cand_dim - label_dim)

                                if dim_diff < best_dist:
                                    best_dist = dim_diff
                                    best_label = lid

                            if best_label is None or best_dist > 50:
                                continue

                            sl = all_slices[best_label - 1]
                            pad = 8
                            r0 = max(0, sl[0].start - pad)
                            r1 = min(uncertainty.shape[0], sl[0].stop + pad)
                            c0 = max(0, sl[1].start - pad)
                            c1 = min(uncertainty.shape[1], sl[1].stop + pad)

                            local_u = uncertainty[r0:r1, c0:c1].copy()
                            local_mask = (labels[r0:r1, c0:c1] == best_label)

                            # Build wreck mesh
                            result = build_wreck_mesh(local_u, local_mask, cell_size)
                            if result is None:
                                continue
                            wreck_glb, stats = result

                            # Build seafloor from surrounding elevation
                            sf_pad = 80
                            sf_r0 = max(0, sl[0].start - sf_pad)
                            sf_r1 = min(elevation.shape[0], sl[0].stop + sf_pad)
                            sf_c0 = max(0, sl[1].start - sf_pad)
                            sf_c1 = min(elevation.shape[1], sl[1].stop + sf_pad)
                            seafloor_glb = build_seafloor_mesh(
                                elevation, slice(sf_r0, sf_r1), slice(sf_c0, sf_c1),
                                cell_size, subsample=4,
                            )

                            # Generate Babylon.js HTML viewer
                            depth = cand.elevation_stats.get('mean') if cand.elevation_stats else None
                            label = f"Candidate #{i+1} (conf {cand.confidence:.2f})"
                            html = generate_viewer_html(wreck_glb, seafloor_glb, label, stats, depth)

                            viewer_data[i] = (html, wreck_glb)
                            viewer_indices.add(i)

                            logger.info(
                                f"3D viewer for candidate {i}: "
                                f"{stats['length_m']:.0f}m x {stats['width_m']:.0f}m, "
                                f"{stats['n_verts']} verts"
                            )

                        except Exception as e:
                            logger.warning(f"Failed to generate 3D viewer for candidate {i}: {e}")

                logger.info(f"Generated {len(viewer_data)} 3D viewers out of {len(candidates)} candidates")

            except Exception as e:
                logger.warning(f"3D viewer generation failed (KML still generated): {e}")

        # Generate KML (with viewer link info)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kml', delete=False) as temp_kml:
            temp_kml_path = temp_kml.name
            self._generate_kml(temp_kml_path, candidates, signatures, file_name,
                             viewer_indices=viewer_indices if viewer_indices else None)

        # Create KMZ (zip containing KML + viewers + models)
        with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.write(temp_kml_path, 'doc.kml')

            # Bundle 3D viewer HTML files and GLB models
            for idx, (html, glb_bytes) in viewer_data.items():
                kmz.writestr(f'viewers/candidate_{idx}.html', html)
                kmz.writestr(f'models/candidate_{idx}.glb', glb_bytes)

        os.unlink(temp_kml_path)

        if viewer_data:
            logger.info(
                f"Generated KMZ with {len(viewer_data)} embedded 3D viewers: {kmz_path}"
            )
        else:
            logger.info(f"Generated KMZ file: {kmz_path}")

    def batch_scan(self, bag_files: List[str], output_dir: str = None, progress_callback=None) -> Dict:
        """Batch scan multiple BAG files"""
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)

        results = []
        total_candidates = 0
        total_signatures = 0
        total_files = len(bag_files)
        completed = 0
        from concurrent.futures import as_completed

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            future_to_file = {executor.submit(self.scan_bag_file, bag_file): bag_file for bag_file in bag_files}
            start_time = datetime.now()

            for future in as_completed(future_to_file):
                bag_file = future_to_file[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    total_candidates += result.get('total_candidates', 0)
                    total_signatures += result.get('total_signatures', 0)
                except Exception as e:
                    logger.error(f"Failed to scan {bag_file}: {e}")
                    results.append({
                        'file': Path(bag_file).name,
                        'error': str(e),
                        'success': False
                    })
                
                if progress_callback:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta = (elapsed / completed) * (total_files - completed) if completed > 0 else 0
                    progress_callback({
                        "step": f"Scanning file {completed}/{total_files}: {Path(bag_file).name}",
                        "completed": completed,
                        "total": total_files,
                        "eta_seconds": eta,
                        "candidates_so_far": total_candidates
                    })

        summary = {
            'total_files': len(bag_files),
            'successful_scans': len([r for r in results if r.get('success', False)]),
            'total_candidates': total_candidates,
            'total_signatures': total_signatures,
            'results': results,
            'scan_timestamp': datetime.now().isoformat(),
            'config': self.config
        }

        # Save summary
        if output_dir:
            summary_file = Path(output_dir) / f"batch_scan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Batch summary saved to {summary_file}")

        return summary

    def get_database_summary(self) -> Dict:
        """Get summary of all scans in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get scan statistics
        cursor.execute('''
            SELECT COUNT(*) as total_scans,
                   SUM(total_candidates) as total_candidates,
                   SUM(redaction_signatures) as total_signatures,
                   AVG(processing_time_ms) as avg_processing_time
            FROM bag_scans
        ''')

        stats = cursor.fetchone()

        # Get recent scans
        cursor.execute('''
            SELECT file_name, scan_timestamp, total_candidates, redaction_signatures
            FROM bag_scans
            ORDER BY scan_timestamp DESC
            LIMIT 10
        ''')

        recent_scans = cursor.fetchall()

        conn.close()

        return {
            'total_scans': stats[0] or 0,
            'total_candidates': stats[1] or 0,
            'total_signatures': stats[2] or 0,
            'avg_processing_time_ms': stats[3] or 0,
            'recent_scans': [
                {
                    'file_name': row[0],
                    'timestamp': row[1],
                    'candidates': row[2],
                    'signatures': row[3]
                } for row in recent_scans
            ]
        }