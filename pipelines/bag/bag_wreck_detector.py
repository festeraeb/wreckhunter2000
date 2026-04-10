"""
BAG File Wreck Detection System
================================
Reads NOAA BAG (Bathymetric Attributed Grid) HDF5 files, detects wrecks
as depth anomalies on the seafloor, handles masking, clusters detections,
and applies coordinate corrections.

BAG files contain:
  - BAG_root/elevation: 2D float32 grid of depths (NoData = 1000000.0)
  - BAG_root/uncertainty: 2D float32 grid of measurement uncertainty
  - BAG_root/metadata: XML with CRS, corner points, resolution

Key design decisions:
  - Wreck = localized elevation anomaly (shallower than surrounding seafloor)
  - Masking gaps ARE NOT new wrecks — they are NoData holes to be filled
  - Small gaps inside masking boundaries are stitching artifacts or blend zones
  - Detections within a clustering radius are ONE wreck, not many
  - Coordinate correction uses a known reference point to shift output
"""

import os
import re
import json
import warnings
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings('ignore')

# Rust acceleration module (built via maturin)
_RUST_REQUIRED_SYMBOLS = (
    "GeoReference",
    "find_references_in_extent",
    "correct_coordinates",
)


def _has_rust_symbols(mod) -> bool:
    return all(hasattr(mod, name) for name in _RUST_REQUIRED_SYMBOLS)


_rust = None
try:
    import bag_processor as _rust_candidate
    if _has_rust_symbols(_rust_candidate):
        _rust = _rust_candidate
except ImportError:
    _rust = None

# Fallback: load the compiled extension from known local build output.
if _rust is None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        pyd_path = repo_root / "bagfilework" / "tools" / "bp_pkg" / "bag_processor.pyd"
        if pyd_path.exists():
            spec = importlib.util.spec_from_file_location("_bag_processor_rust", str(pyd_path))
            if spec and spec.loader:
                _rust_candidate = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_rust_candidate)
                if _has_rust_symbols(_rust_candidate):
                    _rust = _rust_candidate
    except Exception:
        _rust = None

HAS_RUST = _rust is not None

# Database for known reference points
try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


# ============================================================================
# DATA CLASSES
# ============================================================================

class ObjectType(Enum):
    WRECK = "wreck"
    DEBRIS = "debris"
    OBSTRUCTION = "obstruction"
    UNKNOWN = "unknown"


@dataclass
class BAGInfo:
    """Georeferencing info extracted from a BAG file"""
    filepath: str
    survey_id: str
    shape: Tuple[int, int]      # (rows, cols)
    sw_easting: float           # southwest corner easting (meters)
    sw_northing: float          # southwest corner northing (meters)
    ne_easting: float           # northeast corner easting (meters)
    ne_northing: float          # northeast corner northing (meters)
    resolution_m: float         # cell size in meters
    crs_wkt: str                # full CRS WKT string
    epsg_code: int              # EPSG code (e.g. 26916 for UTM 16N)
    vertical_datum: str         # e.g. "LWD", "MLLW"
    nodata_value: float = 1_000_000.0
    valid_cell_count: int = 0
    total_cell_count: int = 0
    depth_min: float = 0.0
    depth_max: float = 0.0


@dataclass
class WreckDetection:
    """A detected wreck/anomaly after clustering"""
    id: str
    latitude: float             # WGS84
    longitude: float            # WGS84
    easting: float              # UTM
    northing: float             # UTM
    depth_meters: float         # depth at anomaly center
    size_meters: float          # approximate extent in meters
    size_feet: float
    height_above_floor: float   # how much it sticks up from surrounding
    confidence: float           # 0.0 - 1.0
    object_type: ObjectType
    bag_file: str
    survey_id: str
    cell_count: int             # number of grid cells in cluster
    masking_pct: float          # % of nearby area that is masked
    is_masking_artifact: bool   # True = probably a gap, not a real wreck
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['object_type'] = self.object_type.value
        return d


@dataclass
class CoordinateCorrection:
    """A known reference point for correcting coordinate offsets"""
    name: str
    known_lat: float            # actual position (WGS84)
    known_lon: float
    detected_lat: float = 0.0   # position from BAG file
    detected_lon: float = 0.0
    offset_lat: float = 0.0     # correction = known - detected
    offset_lon: float = 0.0
    applied: bool = False


# ============================================================================
# BAG FILE READER
# ============================================================================

class BAGReader:
    """Read NOAA BAG (HDF5) files and extract bathymetric grids"""

    NODATA = 1_000_000.0
    MAX_CELLS = 10_000_000  # 10M cells max before downsampling on read

    def __init__(self):
        if not HAS_H5PY:
            raise ImportError("h5py is required to read BAG files. Install: pip install h5py")

    def read_bag(self, filepath: str) -> Tuple[np.ndarray, BAGInfo]:
        """
        Read a BAG file and return (elevation_grid, bag_info).
        
        elevation_grid: 2D float32 array of depths (negative = below water)
        NoData cells are set to np.nan for easy masking.
        Large grids are automatically downsampled during reading.
        """
        filepath = str(filepath)
        f = h5py.File(filepath, 'r')
        try:
            ds = f['BAG_root/elevation']
            full_shape = ds.shape
            total_cells = full_shape[0] * full_shape[1]

            # Downsample at read-time for huge grids to avoid memory issues
            if total_cells > self.MAX_CELLS:
                step = max(2, int((total_cells / self.MAX_CELLS) ** 0.5) + 1)
                elevation = ds[::step, ::step]
                read_shape = elevation.shape
            else:
                step = 1
                elevation = ds[:]
                read_shape = full_shape

            # Read metadata XML
            meta_raw = f['BAG_root/metadata'][:]
            meta_str = b''.join(meta_raw).decode('utf-8', errors='replace')

            # Parse georeferencing (use FULL shape for metadata extraction)
            bag_info = self._parse_metadata(meta_str, filepath, full_shape)

            # If we downsampled, update resolution and shape
            if step > 1:
                bag_info.shape = read_shape
                bag_info.resolution_m *= step

            # Replace NoData with NaN
            elevation = elevation.astype(np.float64)
            elevation[elevation >= self.NODATA - 1] = np.nan

            # Stats
            valid_mask = ~np.isnan(elevation)
            bag_info.valid_cell_count = int(np.sum(valid_mask))
            bag_info.total_cell_count = int(elevation.size)
            if bag_info.valid_cell_count > 0:
                bag_info.depth_min = float(np.nanmin(elevation))
                bag_info.depth_max = float(np.nanmax(elevation))

            return elevation, bag_info
        finally:
            f.close()

    def _parse_metadata(self, meta_str: str, filepath: str, shape: tuple) -> BAGInfo:
        """Parse the XML metadata for CRS, corners, and resolution"""
        # Survey ID from filename
        basename = os.path.basename(filepath)
        survey_id = basename.split('_')[0] if '_' in basename else basename.replace('.bag', '')

        # Corner points: "SW_E,SW_N NE_E,NE_N"
        cp_match = re.search(r'<gml:coordinates[^>]*>([^<]+)</gml:coordinates>', meta_str)
        if cp_match:
            parts = cp_match.group(1).strip().split()
            sw = parts[0].split(',')
            ne = parts[1].split(',') if len(parts) > 1 else sw
            sw_e, sw_n = float(sw[0]), float(sw[1])
            ne_e, ne_n = float(ne[0]), float(ne[1])
        else:
            sw_e, sw_n, ne_e, ne_n = 0, 0, 0, 0

        # Resolution
        res_list = re.findall(r'<gco:Measure uom="m">([^<]+)</gco:Measure>', meta_str)
        resolution = float(res_list[0]) if res_list else 1.0

        # CRS (WKT string) — need to capture full nested PROJCS block
        # Count brackets to find matching close
        crs_wkt = ""
        epsg_code = 0
        projcs_start = meta_str.find('PROJCS[')
        if projcs_start >= 0:
            depth = 0
            for i in range(projcs_start, len(meta_str)):
                if meta_str[i] == '[':
                    depth += 1
                elif meta_str[i] == ']':
                    depth -= 1
                    if depth == 0:
                        crs_wkt = meta_str[projcs_start:i+1]
                        break

            # Use pyproj to reliably get the EPSG code from the WKT
            if crs_wkt and HAS_PYPROJ:
                try:
                    crs_obj = pyproj.CRS(crs_wkt)
                    auth = crs_obj.to_authority()
                    if auth and auth[0] == 'EPSG':
                        epsg_code = int(auth[1])
                    else:
                        # Fallback: get the projected CRS EPSG from the name
                        epsg_code = crs_obj.to_epsg() or 0
                except Exception:
                    # Last resort: regex for the outermost AUTHORITY
                    all_auth = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', crs_wkt)
                    epsg_code = int(all_auth[-1]) if all_auth else 0

        # Vertical datum
        vert_match = re.search(r'VERT_CS\["([^"]+)"', meta_str)
        vertical_datum = vert_match.group(1) if vert_match else "Unknown"

        return BAGInfo(
            filepath=filepath,
            survey_id=survey_id,
            shape=shape,
            sw_easting=sw_e,
            sw_northing=sw_n,
            ne_easting=ne_e,
            ne_northing=ne_n,
            resolution_m=resolution,
            crs_wkt=crs_wkt,
            epsg_code=epsg_code,
            vertical_datum=vertical_datum,
        )


# ============================================================================
# COORDINATE TRANSFORMER
# ============================================================================

class CoordinateTransformer:
    """Convert between UTM and WGS84 lat/lon"""

    def __init__(self):
        if not HAS_PYPROJ:
            raise ImportError("pyproj required. Install: pip install pyproj")
        self._transformers = {}  # cache by EPSG code

    def _get_transformer(self, epsg_code: int, crs_wkt: str = ""):
        """Get or create a transformer for the given EPSG code"""
        if epsg_code not in self._transformers:
            try:
                src_crs = pyproj.CRS.from_epsg(epsg_code)
            except Exception:
                # EPSG failed — try using the WKT string directly
                if crs_wkt:
                    src_crs = pyproj.CRS(crs_wkt)
                else:
                    raise
            dst_crs = pyproj.CRS.from_epsg(4326)  # WGS84
            self._transformers[epsg_code] = pyproj.Transformer.from_crs(
                src_crs, dst_crs, always_xy=True
            )
        return self._transformers[epsg_code]

    def utm_to_latlon(self, easting: float, northing: float,
                      epsg_code: int, crs_wkt: str = "") -> Tuple[float, float]:
        """Convert UTM easting/northing to WGS84 lat/lon"""
        transformer = self._get_transformer(epsg_code, crs_wkt)
        lon, lat = transformer.transform(easting, northing)
        return lat, lon

    def grid_to_latlon(self, row: int, col: int,
                       bag_info: BAGInfo) -> Tuple[float, float]:
        """Convert grid row/col to WGS84 lat/lon"""
        easting = bag_info.sw_easting + col * bag_info.resolution_m
        northing = bag_info.sw_northing + row * bag_info.resolution_m
        return self.utm_to_latlon(easting, northing, bag_info.epsg_code,
                                  bag_info.crs_wkt)

    def grid_to_utm(self, row: int, col: int,
                    bag_info: BAGInfo) -> Tuple[float, float]:
        """Convert grid row/col to UTM easting/northing"""
        easting = bag_info.sw_easting + col * bag_info.resolution_m
        northing = bag_info.sw_northing + row * bag_info.resolution_m
        return easting, northing


# ============================================================================
# MASKING ANALYZER
# ============================================================================

class MaskingAnalyzer:
    """Analyze masking patterns to identify gaps vs real features"""

    def __init__(self, min_gap_cells: int = 10, edge_buffer_cells: int = 5):
        """
        Args:
            min_gap_cells: gaps smaller than this are stitching artifacts
            edge_buffer_cells: anomalies within this distance of mask edge
                are likely masking artifacts, not real wrecks
        """
        self.min_gap_cells = min_gap_cells
        self.edge_buffer_cells = edge_buffer_cells
        # Precomputed cache
        self._mask = None
        self._edge_zones = None
        self._small_gaps = None
        self._shape = None

    def get_mask(self, elevation: np.ndarray) -> np.ndarray:
        """Return boolean mask: True = masked/NoData, False = valid data"""
        return np.isnan(elevation)

    def get_masking_percentage(self, elevation: np.ndarray) -> float:
        """What % of the grid is masked"""
        mask = self.get_mask(elevation)
        return float(np.sum(mask) / mask.size * 100)

    def find_mask_edge_zones(self, elevation: np.ndarray) -> np.ndarray:
        """
        Find cells near the edge of masked areas.
        Returns boolean array: True = within edge_buffer of a mask boundary
        """
        mask = self.get_mask(elevation)

        if not HAS_SCIPY:
            # Fallback: just return the mask itself
            return mask

        # Dilate the mask to find edge zones
        struct = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(mask, struct,
                                          iterations=self.edge_buffer_cells)
        # Edge zone = dilated area minus the mask itself
        edge_zone = dilated & ~mask
        return edge_zone

    def find_small_gaps(self, elevation: np.ndarray) -> np.ndarray:
        """
        Find small data gaps within masking. These are NOT new wrecks.
        Returns boolean array: True = small gap (stitching artifact)
        """
        mask = self.get_mask(elevation)
        valid = ~mask

        if not HAS_SCIPY:
            return np.zeros_like(mask)

        # Find connected components of VALID data
        # Small isolated clusters of valid data surrounded by masking are gaps
        labeled, num_features = ndimage.label(valid)
        small_gaps = np.zeros_like(mask)

        # Cap iteration for performance (files can have 100k+ components)
        if num_features > 10000:
            # Use component_sizes shortcut
            sizes = ndimage.sum(valid, labeled, range(1, num_features + 1))
            small_labels = np.where(np.array(sizes) < self.min_gap_cells)[0] + 1
            for label_id in small_labels:
                small_gaps |= (labeled == label_id)
        else:
            for label_id in range(1, num_features + 1):
                component = labeled == label_id
                size = np.sum(component)
                if size < self.min_gap_cells:
                    small_gaps |= component

        return small_gaps

    def precompute(self, elevation: np.ndarray):
        """
        Precompute mask analysis arrays ONCE per file.
        Must be called before classify_anomaly_location_fast().
        """
        self._mask = self.get_mask(elevation)
        self._edge_zones = self.find_mask_edge_zones(elevation)
        self._small_gaps = self.find_small_gaps(elevation)
        self._shape = elevation.shape

    def classify_anomaly_location_fast(self, row: int, col: int,
                                        radius: int = 10) -> Dict[str, Any]:
        """
        Classify whether an anomaly at (row, col) is near masking or genuine.
        Uses precomputed masks for speed (call precompute() first).
        """
        rows, cols = self._shape

        # Get local window masking percentage
        r_start = max(0, row - radius)
        r_end = min(rows, row + radius + 1)
        c_start = max(0, col - radius)
        c_end = min(cols, col + radius + 1)

        local_mask = self._mask[r_start:r_end, c_start:c_end]
        local_masking_pct = float(np.sum(local_mask) / local_mask.size * 100)

        near_mask_edge = bool(self._edge_zones[row, col]) if row < rows and col < cols else False
        in_small_gap = bool(self._small_gaps[row, col]) if row < rows and col < cols else False

        # Summary judgment — tuned thresholds
        is_artifact = (
            in_small_gap or
            (near_mask_edge and local_masking_pct > 50) or
            local_masking_pct > 70
        )

        return {
            'near_mask_edge': near_mask_edge,
            'in_small_gap': in_small_gap,
            'local_masking_pct': local_masking_pct,
            'is_masking_artifact': is_artifact,
        }

    def classify_anomaly_location(self, elevation: np.ndarray,
                                   row: int, col: int,
                                   radius: int = 10) -> Dict[str, Any]:
        """Legacy method — precomputes then classifies (slow if called many times)"""
        self.precompute(elevation)
        return self.classify_anomaly_location_fast(row, col, radius)


# ============================================================================
# ANOMALY DETECTOR (WRECK FINDER)
# ============================================================================

class AnomalyDetector:
    """
    Detect depth anomalies that indicate wrecks, debris, or obstructions.
    
    A wreck on the seafloor appears as a SHALLOWER area (less negative depth)
    compared to the surrounding flat bottom. We detect these by:
    1. Computing a smoothed "background" seafloor
    2. Finding cells significantly shallower than background
    3. Clustering adjacent anomaly cells into wreck objects
    """

    # Maximum aspect ratio (length/width) for a wreck-shaped cluster.
    # Anything more elongated than this is a stitching seam or survey-strip edge.
    MAX_ASPECT_RATIO = 6.0

    # US Great Lakes bounding box — reject detections outside this.
    GL_LAT_MIN, GL_LAT_MAX = 41.3, 49.0
    GL_LON_MIN, GL_LON_MAX = -92.2, -76.0

    # Minimum bounding-box dimensions for a valid wreck (feet).
    MIN_LONG_SIDE_FT = 36.0
    MIN_SHORT_SIDE_FT = 10.0

    def __init__(self,
                 min_height_m: float = 1.8,
                 min_cluster_cells: int = 8,
                 cluster_radius_m: float = 100.0,
                 masking_analyzer: Optional[MaskingAnalyzer] = None):
        """
        Args:
            min_height_m: minimum height above seafloor to be an anomaly (meters)
            min_cluster_cells: minimum cells in a cluster to be a wreck
            cluster_radius_m: radius for spatial clustering (meters)
            masking_analyzer: for filtering masking artifacts
        """
        self.min_height_m = min_height_m
        self.min_cluster_cells = min_cluster_cells
        self.cluster_radius_m = cluster_radius_m
        self.masking = masking_analyzer or MaskingAnalyzer()

    def detect(self, elevation: np.ndarray,
               bag_info: BAGInfo,
               transformer: CoordinateTransformer,
               correction: Optional[CoordinateCorrection] = None
               ) -> List[WreckDetection]:
        """
        Detect wreck anomalies in a BAG elevation grid.
        
        Returns list of WreckDetection objects (one per clustered anomaly)
        """
        if bag_info.valid_cell_count < 100:
            # Not enough data to analyze
            return []

        # For very large grids, downsample for detection
        work_elev, work_info, scale_factor = self._maybe_downsample(
            elevation, bag_info
        )

        # 1. Precompute masking analysis ONCE for this file
        self.masking.precompute(work_elev)

        # 2. Compute background seafloor (median filter)
        background = self._compute_background(work_elev, work_info)

        # 3. Find anomaly cells (shallower than background by min_height)
        anomaly_mask = self._find_anomalies(work_elev, background)

        if np.sum(anomaly_mask) == 0:
            return []

        # 4. Filter out masking artifacts using precomputed masks
        edge_zones = self.masking._edge_zones
        small_gaps = self.masking._small_gaps

        # Remove anomalies that are in mask edge zones or small gaps
        clean_anomaly = anomaly_mask & ~edge_zones & ~small_gaps

        if np.sum(clean_anomaly) == 0:
            return []

        # 5. Cluster adjacent anomaly cells
        clusters = self._cluster_anomalies(clean_anomaly, work_elev,
                                           background, work_info)

        # 6. Build WreckDetection objects (use work_info for coord transform)
        detections = []
        for i, cluster in enumerate(clusters):
            det = self._cluster_to_detection(
                cluster, work_elev, work_info, transformer,
                correction, i
            )
            if det is not None:
                detections.append(det)

        return detections

    def _maybe_downsample(self, elevation: np.ndarray,
                          bag_info: BAGInfo) -> Tuple[np.ndarray, BAGInfo, int]:
        """
        Downsample large grids to make detection tractable.
        Returns (downsampled_elevation, updated_bag_info, scale_factor).
        """
        max_dim = 2500  # max cells in any dimension
        rows, cols = elevation.shape
        
        if rows <= max_dim and cols <= max_dim:
            return elevation, bag_info, 1

        # Compute downsample factor
        scale = max(rows // max_dim, cols // max_dim, 2)
        
        # Block-average downsample (preserves depth features better than skip)
        new_rows = rows // scale
        new_cols = cols // scale
        
        # Simple skip-sampling (fastest, good enough for detection)
        downsampled = elevation[::scale, ::scale]
        
        # Update bag_info with new resolution and shape
        import copy
        new_info = copy.copy(bag_info)
        new_info.shape = downsampled.shape
        new_info.resolution_m = bag_info.resolution_m * scale
        
        # Recount valid cells
        valid_mask = ~np.isnan(downsampled)
        new_info.valid_cell_count = int(np.sum(valid_mask))
        new_info.total_cell_count = int(downsampled.size)
        
        return downsampled, new_info, scale

    def _compute_background(self, elevation: np.ndarray,
                            bag_info: BAGInfo) -> np.ndarray:
        """
        Compute smoothed background seafloor.
        Uses Gaussian filter (fast O(n) per pixel) instead of median filter.
        Two-pass approach: first pass gets rough background, second pass
        excludes anomalies for a cleaner estimate.
        """
        # Sigma: aim for ~100m radius in grid cells
        sigma_cells = max(2.0, 100.0 / bag_info.resolution_m)
        sigma_cells = min(sigma_cells, 30.0)  # cap for very fine grids

        if HAS_SCIPY:
            filled = elevation.copy()
            nan_mask = np.isnan(filled)
            if np.any(~nan_mask):
                global_median = np.nanmedian(elevation)
                filled[nan_mask] = global_median

                # Pass 1: Gaussian blur for fast background estimate
                background = ndimage.gaussian_filter(filled, sigma=sigma_cells)
                # Restore NaN positions
                background[nan_mask] = np.nan
            else:
                background = filled
        else:
            background = np.full_like(elevation, np.nanmean(elevation))

        return background

    def _find_anomalies(self, elevation: np.ndarray,
                        background: np.ndarray) -> np.ndarray:
        """
        Find cells where depth is significantly SHALLOWER than background.
        Wrecks stick up from the bottom, so they are less negative.
        
        height_above = elevation - background  (both negative, wreck less negative)
        If elevation = -30 and background = -35, height_above = +5m -> wreck
        """
        valid = ~np.isnan(elevation) & ~np.isnan(background)
        height_above = np.zeros_like(elevation)
        height_above[valid] = elevation[valid] - background[valid]

        anomaly_mask = valid & (height_above >= self.min_height_m)
        return anomaly_mask

    def _cluster_anomalies(self, anomaly_mask: np.ndarray,
                           elevation: np.ndarray,
                           background: np.ndarray,
                           bag_info: BAGInfo) -> List[Dict]:
        """Cluster adjacent anomaly cells into distinct wreck objects"""
        if not HAS_SCIPY:
            # Fallback: treat entire anomaly mask as one cluster
            if np.sum(anomaly_mask) > 0:
                rows, cols = np.where(anomaly_mask)
                return [{'rows': rows, 'cols': cols}]
            return []

        # Label connected components
        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        labeled, num_features = ndimage.label(anomaly_mask, struct)

        clusters = []
        for label_id in range(1, num_features + 1):
            component = labeled == label_id
            cell_count = np.sum(component)

            # Skip tiny clusters
            if cell_count < self.min_cluster_cells:
                continue

            rows, cols = np.where(component)

            # --- Stitching-seam filter ---
            # Survey-strip edges produce thin linear artifacts.
            # Reject clusters whose bounding-box aspect ratio exceeds the limit.
            row_span = (np.max(rows) - np.min(rows) + 1) * bag_info.resolution_m
            col_span = (np.max(cols) - np.min(cols) + 1) * bag_info.resolution_m
            long_side = max(row_span, col_span)
            short_side = max(min(row_span, col_span), 0.01)
            if long_side / short_side > self.MAX_ASPECT_RATIO:
                continue
            
            # Compute height above floor for this cluster
            valid_in_cluster = ~np.isnan(elevation[component]) & ~np.isnan(background[component])
            if np.sum(valid_in_cluster) == 0:
                continue
            
            cluster_heights = elevation[component][valid_in_cluster] - background[component][valid_in_cluster]

            clusters.append({
                'rows': rows,
                'cols': cols,
                'cell_count': int(cell_count),
                'max_height': float(np.max(cluster_heights)),
                'mean_height': float(np.mean(cluster_heights)),
            })

        return clusters

    def _cluster_to_detection(self, cluster: Dict,
                               elevation: np.ndarray,
                               bag_info: BAGInfo,
                               transformer: CoordinateTransformer,
                               correction: Optional[CoordinateCorrection],
                               index: int) -> Optional[WreckDetection]:
        """Convert a cluster dict to a WreckDetection"""
        rows = cluster['rows']
        cols = cluster['cols']

        # Centroid (weighted by height if available, else simple mean)
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))

        # UTM coordinates of center
        easting, northing = transformer.grid_to_utm(center_row, center_col, bag_info)

        # Convert to lat/lon
        try:
            lat, lon = transformer.utm_to_latlon(easting, northing,
                                                  bag_info.epsg_code,
                                                  bag_info.crs_wkt)
        except Exception:
            lat, lon = 0.0, 0.0

        # Apply coordinate correction if provided
        if correction and correction.applied:
            lat += correction.offset_lat
            lon += correction.offset_lon

        # Great Lakes bounding-box sanity check
        if not (self.GL_LAT_MIN <= lat <= self.GL_LAT_MAX and
                self.GL_LON_MIN <= lon <= self.GL_LON_MAX):
            return None

        # Depth at center
        depth = float(elevation[center_row, center_col])
        if np.isnan(depth):
            depth = float(np.nanmean(elevation[rows, cols]))

        # Size estimate (extent in meters)
        row_span = (np.max(rows) - np.min(rows)) * bag_info.resolution_m
        col_span = (np.max(cols) - np.min(cols)) * bag_info.resolution_m
        size_m = max(row_span, col_span)
        size_ft = size_m * 3.28084

        # Minimum bounding-box size filter (36 ft × 10 ft)
        long_side_ft = max(row_span, col_span) * 3.28084
        short_side_ft = min(row_span, col_span) * 3.28084
        if long_side_ft < self.MIN_LONG_SIDE_FT or short_side_ft < self.MIN_SHORT_SIDE_FT:
            return None

        # Height above floor
        height = cluster.get('max_height', 0.0)

        # Confidence based on size and height
        confidence = min(0.99, 0.5 + height * 0.05 + (cluster['cell_count'] / 100) * 0.2)

        # Check masking context (use precomputed masks for speed)
        try:
            masking_info = self.masking.classify_anomaly_location_fast(
                center_row, center_col, radius=20
            )
        except AttributeError:
            # Fallback if precompute() wasn't called
            masking_info = self.masking.classify_anomaly_location(
                elevation, center_row, center_col, radius=20
            )

        # Object type
        if size_ft >= 50:
            obj_type = ObjectType.WRECK
        elif size_ft >= 15:
            obj_type = ObjectType.DEBRIS
        else:
            obj_type = ObjectType.OBSTRUCTION

        return WreckDetection(
            id=f"{bag_info.survey_id}_det{index:03d}",
            latitude=lat,
            longitude=lon,
            easting=easting,
            northing=northing,
            depth_meters=abs(depth),
            size_meters=size_m,
            size_feet=size_ft,
            height_above_floor=height,
            confidence=confidence,
            object_type=obj_type,
            bag_file=os.path.basename(bag_info.filepath),
            survey_id=bag_info.survey_id,
            cell_count=cluster['cell_count'],
            masking_pct=masking_info['local_masking_pct'],
            is_masking_artifact=masking_info['is_masking_artifact'],
            timestamp=datetime.now().isoformat(),
            metadata={
                'near_mask_edge': masking_info['near_mask_edge'],
                'in_small_gap': masking_info['in_small_gap'],
                'height_above_floor_m': height,
                'mean_height_m': cluster.get('mean_height', 0.0),
                'row_span_m': float((np.max(rows) - np.min(rows)) * bag_info.resolution_m),
                'col_span_m': float((np.max(cols) - np.min(cols)) * bag_info.resolution_m),
                'resolution_m': bag_info.resolution_m,
                'epsg': bag_info.epsg_code,
                'vertical_datum': bag_info.vertical_datum,
            }
        )


# ============================================================================
# CROSS-FILE DEDUPLICATION
# ============================================================================

class SpatialDeduplicator:
    """
    Merge detections from multiple overlapping BAG files.
    
    The same wreck can appear in multiple files (e.g. 50cm and 1m resolution
    of the same survey). This merges detections within a radius into one,
    keeping the highest-confidence detection.
    """

    def __init__(self, merge_radius_m: float = 200.0):
        """
        Args:
            merge_radius_m: detections within this distance are merged
        """
        self.merge_radius_m = merge_radius_m

    def deduplicate(self, detections: List[WreckDetection]) -> List[WreckDetection]:
        """Merge nearby detections, keeping the best one from each cluster"""
        if not detections:
            return []

        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept = []
        used = set()

        for i, det in enumerate(sorted_dets):
            if i in used:
                continue

            # Find all detections within merge radius
            group = [det]
            for j, other in enumerate(sorted_dets):
                if j <= i or j in used:
                    continue
                dist = self._distance_m(det, other)
                if dist <= self.merge_radius_m:
                    group.append(other)
                    used.add(j)

            used.add(i)

            # Keep the best detection, but enrich with info from group
            best = group[0]  # highest confidence
            best.metadata['merged_from'] = len(group)
            best.metadata['source_files'] = list(set(d.bag_file for d in group))
            kept.append(best)

        return kept

    def _distance_m(self, a: WreckDetection, b: WreckDetection) -> float:
        """Approximate distance between two detections in meters"""
        # Use UTM coordinates if available (same zone), else approximate from lat/lon
        if a.easting > 0 and b.easting > 0:
            de = a.easting - b.easting
            dn = a.northing - b.northing
            return (de**2 + dn**2) ** 0.5
        else:
            # Haversine approximation
            dlat = (a.latitude - b.latitude) * 111_000
            dlon = (a.longitude - b.longitude) * 111_000 * np.cos(np.radians(a.latitude))
            return (dlat**2 + dlon**2) ** 0.5


# ============================================================================
# RESTORATION WITH EXAGGERATION
# ============================================================================

class RestorationEngine:
    """
    Restore masked/damaged areas and provide exaggeration visualization.
    
    - Fills masking gaps using interpolation from surrounding valid cells
    - Small gaps within masking are filled (stitching artifacts)
    - Exaggeration scales depth differences to make features stand out
    - Color mapping highlights restored vs original areas
    """

    def __init__(self):
        pass

    def restore_masked_areas(self, elevation: np.ndarray,
                              bag_info: BAGInfo) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill masked (NaN) areas by interpolating from surrounding valid data.
        
        Returns:
            (restored_elevation, restoration_mask)
            restoration_mask: True where restoration was applied
        """
        restored = elevation.copy()
        was_nan = np.isnan(elevation)

        if not HAS_SCIPY or np.sum(~was_nan) < 10:
            return restored, was_nan

        # Use nearest-neighbor interpolation for gaps
        from scipy.interpolate import NearestNDInterpolator

        valid_rows, valid_cols = np.where(~was_nan)
        valid_values = elevation[~was_nan]

        nan_rows, nan_cols = np.where(was_nan)

        if len(nan_rows) == 0 or len(valid_rows) == 0:
            return restored, was_nan

        # For large grids, subsample the valid points for speed
        max_points = 50000
        if len(valid_rows) > max_points:
            indices = np.random.choice(len(valid_rows), max_points, replace=False)
            interp_rows = valid_rows[indices]
            interp_cols = valid_cols[indices]
            interp_vals = valid_values[indices]
        else:
            interp_rows = valid_rows
            interp_cols = valid_cols
            interp_vals = valid_values

        interpolator = NearestNDInterpolator(
            list(zip(interp_rows, interp_cols)), interp_vals
        )

        filled_values = interpolator(nan_rows, nan_cols)
        restored[was_nan] = filled_values

        return restored, was_nan

    def apply_exaggeration(self, elevation: np.ndarray,
                           factor: float = 1.0,
                           reference_depth: Optional[float] = None
                           ) -> np.ndarray:
        """
        Apply vertical exaggeration to make features stand out.
        
        Like CARIS vertical exaggeration:
        exaggerated = reference + (depth - reference) * factor
        
        factor=1.0 -> no change
        factor=2.0 -> double the height differences
        factor=5.0 -> 5x exaggeration
        """
        if factor == 1.0:
            return elevation.copy()

        valid = ~np.isnan(elevation)
        if not np.any(valid):
            return elevation.copy()

        if reference_depth is None:
            reference_depth = float(np.nanmedian(elevation))

        exaggerated = elevation.copy()
        exaggerated[valid] = reference_depth + (elevation[valid] - reference_depth) * factor

        return exaggerated

    def get_color_array(self, elevation: np.ndarray,
                        restoration_mask: np.ndarray,
                        exaggeration: float = 1.0) -> np.ndarray:
        """
        Create RGBA color array for visualization.
        
        - Original data: blue-green depth colormap
        - Restored areas: warm orange/red tint
        - Exaggeration enhances the color contrast
        
        Returns: (rows, cols, 4) uint8 RGBA array
        """
        rows, cols = elevation.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        valid = ~np.isnan(elevation)
        if not np.any(valid):
            return rgba

        # Normalize depths to 0-1
        vmin = np.nanmin(elevation)
        vmax = np.nanmax(elevation)
        if vmax == vmin:
            normalized = np.zeros_like(elevation)
        else:
            normalized = (elevation - vmin) / (vmax - vmin)
        normalized = np.nan_to_num(normalized, 0.0)

        # Apply exaggeration to colors
        if exaggeration != 1.0:
            mid = 0.5
            normalized = mid + (normalized - mid) * min(exaggeration, 10.0)
            normalized = np.clip(normalized, 0, 1)

        # Original areas: blue-green
        rgba[valid & ~restoration_mask, 0] = (30 + normalized[valid & ~restoration_mask] * 50).astype(np.uint8)
        rgba[valid & ~restoration_mask, 1] = (80 + normalized[valid & ~restoration_mask] * 150).astype(np.uint8)
        rgba[valid & ~restoration_mask, 2] = (120 + normalized[valid & ~restoration_mask] * 135).astype(np.uint8)
        rgba[valid & ~restoration_mask, 3] = 255

        # Restored areas: warm orange-red tint
        rgba[restoration_mask & valid, 0] = (180 + normalized[restoration_mask & valid] * 75).astype(np.uint8)
        rgba[restoration_mask & valid, 1] = (80 + normalized[restoration_mask & valid] * 100).astype(np.uint8)
        rgba[restoration_mask & valid, 2] = (20 + normalized[restoration_mask & valid] * 60).astype(np.uint8)
        rgba[restoration_mask & valid, 3] = 255

        return rgba


# ============================================================================
# KML GENERATOR
# ============================================================================

class KMLGenerator:
    """Generates KML/KMZ files from detections"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate(self, detections: List[WreckDetection],
                 title: str = "BAG Wreck Detections") -> Tuple[str, str]:
        """Generate KML and KMZ files. Returns (kml_path, kmz_path)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kml_path = self.output_dir / f"wrecks_{timestamp}.kml"

        # Filter out masking artifacts for KML
        real_detections = [d for d in detections if not d.is_masking_artifact]

        kml = self._build_kml(real_detections, title)

        with open(kml_path, 'w') as f:
            f.write(kml)

        # Create KMZ
        import zipfile
        kmz_path = self.output_dir / f"wrecks_{timestamp}.kmz"
        with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(kml_path, arcname=kml_path.name)

        return str(kml_path), str(kmz_path)

    def _build_kml(self, detections: List[WreckDetection], title: str) -> str:
        kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        kml += '<Document>\n'
        kml += f'  <name>{title}</name>\n'
        kml += f'  <description>Generated {datetime.now().isoformat()}</description>\n'
        kml += self._styles()

        # Group by survey
        surveys = {}
        for d in detections:
            surveys.setdefault(d.survey_id, []).append(d)

        for survey_id, dets in sorted(surveys.items()):
            kml += f'  <Folder>\n    <name>Survey {survey_id}</name>\n'
            for d in dets:
                kml += self._placemark(d)
            kml += '  </Folder>\n'

        kml += '</Document>\n</kml>\n'
        return kml

    def _styles(self) -> str:
        return '''  <Style id="wreck"><IconStyle><color>ff0000ff</color>
    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/square.png</href></Icon>
    <scale>1.2</scale></IconStyle></Style>
  <Style id="debris"><IconStyle><color>ff00aaff</color>
    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
    <scale>0.8</scale></IconStyle></Style>
  <Style id="obstruction"><IconStyle><color>ff00ffaa</color>
    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href></Icon>
    <scale>0.6</scale></IconStyle></Style>
'''

    def _placemark(self, d: WreckDetection) -> str:
        style = d.object_type.value
        name = f"{d.object_type.value.title()} ({d.size_feet:.0f} ft)"
        return f'''    <Placemark>
      <name>{name}</name>
      <styleUrl>#{style}</styleUrl>
      <Point><coordinates>{d.longitude:.8f},{d.latitude:.8f},0</coordinates></Point>
      <description><![CDATA[
        <b>{name}</b><br/>
        <b>Location:</b> {d.latitude:.6f}, {d.longitude:.6f}<br/>
        <b>Depth:</b> {d.depth_meters:.1f}m<br/>
        <b>Size:</b> {d.size_feet:.0f} ft ({d.size_meters:.1f}m)<br/>
        <b>Height above floor:</b> {d.height_above_floor:.1f}m<br/>
        <b>Confidence:</b> {d.confidence:.0%}<br/>
        <b>Masking nearby:</b> {d.masking_pct:.0f}%<br/>
        <b>Survey:</b> {d.survey_id}<br/>
        <b>File:</b> {d.bag_file}<br/>
        <b>Cells:</b> {d.cell_count}<br/>
      ]]></description>
    </Placemark>
'''


# ============================================================================
# REFERENCE POINT LOADER (for geo correction)
# ============================================================================

class ReferenceLoader:
    """
    Load known reference points (wrecks, features) from the SQLite DB.
    These are used by the Rust correct_coordinates() function to fix
    BAG metadata coordinate drift.
    """

    DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "wrecks.db")

    def __init__(self, db_path: str = ""):
        self.db_path = db_path or self.DEFAULT_DB

    def load_references(self) -> list:
        """
        Load all features with valid coordinates from the DB.
        Returns list of (name, lat, lon) tuples usable as GeoReference
        objects by the Rust module.
        """
        if not HAS_SQLITE or not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name, latitude, longitude FROM features "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
                "AND latitude != 0 AND longitude != 0"
            ).fetchall()
            return [(r[0], float(r[1]), float(r[2])) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def load_rust_references(self, bag_lat: float = 0.0,
                             bag_lon: float = 0.0) -> list:
        """
        Load references as Rust GeoReference objects.
        bag_lat/bag_lon are the BAG-reported positions (used to compute offset).
        If zero, offset can't be computed and the reference just acts as spatial anchor.
        """
        if not HAS_RUST:
            return []
        raw = self.load_references()
        refs = []
        for name, lat, lon in raw:
            if name is None:
                continue
            refs.append(_rust.GeoReference(
                name=str(name),
                known_lat=lat,
                known_lon=lon,
                bag_lat=bag_lat,
                bag_lon=bag_lon,
            ))
        return refs

    def spatial_filter(self, references: list,
                       min_lat: float, max_lat: float,
                       min_lon: float, max_lon: float) -> list:
        """
        Pre-filter references to only those within a bounding box.
        Uses Rust find_references_in_extent if available, else pure Python.
        """
        if HAS_RUST and references and hasattr(references[0], 'known_lat'):
            return _rust.find_references_in_extent(
                references, min_lat, max_lat, min_lon, max_lon
            )
        # Pure Python fallback
        return [
            r for r in references
            if min_lat <= r[1] <= max_lat and min_lon <= r[2] <= max_lon
        ] if references and isinstance(references[0], tuple) else []


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class BAGPipeline:
    """
    Complete pipeline for scanning BAG files and detecting wrecks.
    
    Steps:
    1. Discover all .bag files in a directory (recursive)
    2. Read each BAG file (HDF5) with proper georeferencing
    3. Detect depth anomalies (wrecks), filtering masking artifacts
    4. Deduplicate across overlapping files
    5. Apply coordinate corrections
    6. Generate KML/KMZ output
    """

    def __init__(self,
                 min_height_m: float = 1.8,
                 min_cluster_cells: int = 8,
                 merge_radius_m: float = 200.0,
                 correction: Optional[CoordinateCorrection] = None,
                 db_path: str = "",
                 progress_callback=None):
        """
        Args:
            min_height_m: minimum height above seafloor for detection
            min_cluster_cells: minimum cells in a cluster
            merge_radius_m: radius for cross-file deduplication
            correction: coordinate correction to apply
            db_path: path to wrecks.db for geo-reference loading
            progress_callback: function(current, total, message) for UI updates
        """
        self.reader = BAGReader()
        self.transformer = CoordinateTransformer()
        self.masking = MaskingAnalyzer(min_gap_cells=10, edge_buffer_cells=5)
        self.detector = AnomalyDetector(
            min_height_m=min_height_m,
            min_cluster_cells=min_cluster_cells,
            masking_analyzer=self.masking
        )
        self.deduplicator = SpatialDeduplicator(merge_radius_m=merge_radius_m)
        self.restoration = RestorationEngine()
        self.kml_gen = KMLGenerator()
        self.correction = correction
        self.progress = progress_callback or (lambda c, t, m: None)

        # Geo-reference correction from known features DB
        self.ref_loader = ReferenceLoader(db_path)
        self._all_references = self.ref_loader.load_references()
        self._has_geo_correction = HAS_RUST and len(self._all_references) > 0

        # Results
        self.all_detections: List[WreckDetection] = []
        self.final_detections: List[WreckDetection] = []
        self.bag_infos: List[BAGInfo] = []
        self.file_results: Dict[str, Dict] = {}

    def scan_directory(self, root_dir: str) -> List[str]:
        """
        Find all .bag files recursively.
        Returns files sorted by survey, with coarsest resolution first per survey.
        This way the deduplicator handles multi-resolution overlap efficiently.
        """
        root = Path(root_dir)
        bag_files = list(str(p) for p in root.rglob("*.bag"))

        # Sort: group by survey, coarsest (largest file res number) first
        def sort_key(filepath):
            basename = os.path.basename(filepath)
            # Extract survey ID
            survey = basename.split('_')[0] if '_' in basename else basename
            # Extract resolution indicator: prefer larger (coarser) first
            # e.g., "16m" > "8m" > "4m" > "2m" > "1m" > "50cm"
            res_order = 0  # default
            import re as re2
            m = re2.search(r'_(\d+)m_', basename)
            if m:
                res_order = -int(m.group(1))  # negative = coarsest first
            elif '50cm' in basename:
                res_order = 1  # finest last
            return (survey, res_order, basename)

        bag_files.sort(key=sort_key)
        return bag_files

    def run(self, bag_dir: str, output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns summary dict.
        """
        self.all_detections = []
        self.final_detections = []
        self.bag_infos = []
        self.file_results = {}

        # Step 1: Find files
        self.progress(0, 1, "Scanning for BAG files...")
        bag_files = self.scan_directory(bag_dir)
        if not bag_files:
            return {'error': 'No BAG files found', 'files': 0}

        total = len(bag_files)
        self.progress(0, total, f"Found {total} BAG files")

        # Step 2: Process each file
        for i, bag_file in enumerate(bag_files):
            basename = os.path.basename(bag_file)
            self.progress(i, total, f"Processing {basename}...")

            try:
                elevation, bag_info = self.reader.read_bag(bag_file)
                self.bag_infos.append(bag_info)

                # Skip files with very little valid data
                valid_pct = bag_info.valid_cell_count / max(1, bag_info.total_cell_count) * 100
                if valid_pct < 0.1:
                    self.file_results[basename] = {
                        'status': 'skipped',
                        'reason': f'Only {valid_pct:.1f}% valid data',
                        'detections': 0
                    }
                    continue

                # Detect anomalies
                detections = self.detector.detect(
                    elevation, bag_info, self.transformer, self.correction
                )

                self.all_detections.extend(detections)

                # Separate real vs artifact
                real = [d for d in detections if not d.is_masking_artifact]
                artifacts = [d for d in detections if d.is_masking_artifact]

                self.file_results[basename] = {
                    'status': 'processed',
                    'valid_pct': valid_pct,
                    'depth_range': f"{bag_info.depth_min:.1f} to {bag_info.depth_max:.1f}m",
                    'resolution': f"{bag_info.resolution_m}m",
                    'survey': bag_info.survey_id,
                    'epsg': bag_info.epsg_code,
                    'detections': len(real),
                    'artifacts_filtered': len(artifacts),
                    'masking_pct': self.masking.get_masking_percentage(elevation),
                }

            except Exception as e:
                self.file_results[basename] = {
                    'status': 'error',
                    'error': str(e),
                    'detections': 0
                }

        # Step 3: Deduplicate across files
        self.progress(total, total, "Deduplicating across files...")
        real_detections = [d for d in self.all_detections if not d.is_masking_artifact]
        self.final_detections = self.deduplicator.deduplicate(real_detections)

        # Step 3b: Rust geo-correction using known reference points
        if self._has_geo_correction and self.final_detections:
            self.progress(total, total, "Applying geo-correction from known references...")
            self._apply_rust_geo_correction()

        # Step 4: Generate KML
        self.progress(total, total, "Generating KML/KMZ...")
        self.kml_gen = KMLGenerator(output_dir)
        kml_path, kmz_path = self.kml_gen.generate(self.final_detections)

        # Step 5: Save JSON report
        report_path = os.path.join(output_dir, f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = self._build_report(kml_path, kmz_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.progress(total, total, "Complete!")
        return report

    def _build_report(self, kml_path: str, kmz_path: str) -> Dict[str, Any]:
        """Build summary report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'files_scanned': len(self.file_results),
            'files_with_detections': sum(
                1 for r in self.file_results.values() if r.get('detections', 0) > 0
            ),
            'total_raw_anomalies': len(self.all_detections),
            'masking_artifacts_filtered': sum(
                1 for d in self.all_detections if d.is_masking_artifact
            ),
            'after_deduplication': len(self.final_detections),
            'detections': [d.to_dict() for d in self.final_detections],
            'file_results': self.file_results,
            'kml_path': kml_path,
            'kmz_path': kmz_path,
            'coordinate_correction': asdict(self.correction) if self.correction else None,
            'geo_correction_available': self._has_geo_correction,
            'reference_points_loaded': len(self._all_references),
        }

    def _apply_rust_geo_correction(self):
        """
        Apply Rust-accelerated geo-correction to all final detections.
        
        For each detection, finds nearby known reference points and computes
        a weighted-average coordinate offset to correct BAG metadata drift.
        Prefers nearby known wrecks/features over raw BAG metadata.
        """
        if not HAS_RUST:
            return

        # Build Rust GeoReference objects from all known features
        # For each reference, bag_lat/bag_lon = 0 since we don't know their BAG-reported positions.
        # The correction uses known_lat/known_lon as spatial anchors weighted by distance.
        refs = []
        for name, lat, lon in self._all_references:
            refs.append(_rust.GeoReference(
                name=name, known_lat=lat, known_lon=lon,
                bag_lat=lat, bag_lon=lon,  # reference = self (0 offset)
            ))

        corrected_count = 0
        for det in self.final_detections:
            # Spatial pre-filter: get references near this detection
            extent_margin = 0.5  # degrees (~50km)
            local_refs = _rust.find_references_in_extent(
                refs,
                det.latitude - extent_margin,
                det.latitude + extent_margin,
                det.longitude - extent_margin,
                det.longitude + extent_margin,
            )

            if not local_refs:
                continue

            # Build detection list for Rust correct_coordinates
            det_list = [(det.latitude, det.longitude)]

            result = _rust.correct_coordinates(
                det_list, local_refs, 50000.0  # 50km search radius
            )

            if result.reference_fix_applied and result.corrected_detections:
                new_lat, new_lon = result.corrected_detections[0]
                old_lat, old_lon = det.latitude, det.longitude
                det.latitude = new_lat
                det.longitude = new_lon
                det.metadata['geo_corrected'] = True
                det.metadata['geo_offset_lat'] = new_lat - old_lat
                det.metadata['geo_offset_lon'] = new_lon - old_lon
                det.metadata['geo_nearby_refs'] = len(local_refs)
                corrected_count += 1

        if corrected_count:
            self.progress(0, 0, f"Geo-corrected {corrected_count}/{len(self.final_detections)} detections")


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BAG WRECK DETECTION PIPELINE")
    print("=" * 70)

    bag_dir = r"c:\Temp\Garminjunk\HistoryofCESARSNIFFERBAGFILE\bagfilework\Lake Erie Bag Files"
    output_dir = r"c:\Temp\Garminjunk\HistoryofCESARSNIFFERBAGFILE\bagfilework\outputs\bag_scan"

    def progress(current, total, msg):
        print(f"  [{current}/{total}] {msg}")

    pipeline = BAGPipeline(
        min_height_m=1.8,
        min_cluster_cells=8,
        merge_radius_m=200.0,
        progress_callback=progress
    )

    report = pipeline.run(bag_dir, output_dir)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Files scanned:           {report['files_scanned']}")
    print(f"Files w/ detections:     {report['files_with_detections']}")
    print(f"Raw anomalies:           {report['total_raw_anomalies']}")
    print(f"Masking artifacts:       {report['masking_artifacts_filtered']}")
    print(f"After dedup:             {report['after_deduplication']}")
    print(f"KML: {report['kml_path']}")
    print(f"KMZ: {report['kmz_path']}")

    print(f"\nPer-file results:")
    for fname, info in report['file_results'].items():
        status = info['status']
        dets = info.get('detections', 0)
        extra = ""
        if status == 'processed':
            extra = f"  depth: {info['depth_range']}  masking: {info['masking_pct']:.0f}%  artifacts: {info['artifacts_filtered']}"
        elif status == 'error':
            extra = f"  ERROR: {info['error']}"
        elif status == 'skipped':
            extra = f"  ({info['reason']})"
        print(f"  {fname}: {status} -> {dets} wrecks{extra}")

    if pipeline.final_detections:
        print(f"\nDetected wrecks:")
        for d in pipeline.final_detections:
            flag = "★" if d.size_feet >= 50 else "·"
            print(f"  {flag} {d.latitude:.6f}, {d.longitude:.6f} | "
                  f"{d.size_feet:.0f}ft | {d.depth_meters:.1f}m | "
                  f"{d.height_above_floor:.1f}m above floor | "
                  f"{d.confidence:.0%} | {d.bag_file}")
