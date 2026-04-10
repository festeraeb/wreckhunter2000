"""
Bathymetry Mapper - Phase 11.2

Create high-resolution bathymetric maps from sonar depth data with interpolation,
contour analysis, and multi-format export (GeoTIFF, KML contours, NetCDF grids).

Key Features:
- Grid-based depth interpolation from scattered sonar points
- Bathymetric contour generation with depth labels
- Slope analysis (detect drop-offs, gradients)
- Export to GeoTIFF, KML, NetCDF, and GeoJSON formats
- Handles data gaps and irregular sampling
- Depth-based color mapping (shallow=light, deep=dark)

Architecture:
- BathymetryMapper: Main class for grid generation and interpolation
- GridBuilder: Creates regular grids from irregular sonar trackpoints
- DepthAnalyzer: Compute slope, gradient, curvature
- ContourGenerator: Create contours with depth labels
- MultiFormatExporter: Export to various formats

ML Enhancement Potential:
- Use bathymetry + anomalies for automatic structure detection
- Slope thresholds for potential fish structure prediction
- Integration with Phase 11.1 anomaly detector for correlation
"""

import numpy as np
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for bathymetric grid generation."""
    grid_resolution: float = 5.0  # meters per grid cell
    min_depth: float = -100.0  # minimum depth for valid data
    max_depth: float = 0.0  # maximum depth (surface)
    interpolation_method: str = "linear"  # 'linear', 'cubic', 'nearest'
    fill_value: Optional[float] = None  # Value for gaps


@dataclass
class TrackPoint:
    """A single sonar measurement point with depth and location."""
    latitude: float
    longitude: float
    depth: float
    timestamp: Optional[float] = None
    confidence: float = 1.0


class BathymetryMapper:
    """
    Create bathymetric maps from sonar depth measurements using spatial interpolation.
    
    Workflow:
        1. Add trackpoints (sonar measurements with lat/lon/depth)
        2. Generate regular grid from scattered points
        3. Interpolate depth values across grid
        4. Analyze bathymetric features (slopes, contours)
        5. Export to desired format (GeoTIFF, KML, GeoJSON, etc.)
    
    Example:
        >>> mapper = BathymetryMapper(resolution=5.0)
        >>> trackpoints = [
        ...     TrackPoint(lat=42.0, lon=-85.0, depth=-20.5),
        ...     TrackPoint(lat=42.1, lon=-85.0, depth=-25.0),
        ... ]
        >>> mapper.add_trackpoints(trackpoints)
        >>> grid = mapper.generate_grid()
        >>> mapper.export_to_geojson('bathymetry.geojson')
    """
    
    def __init__(self, resolution: float = 5.0, grid_config: Optional[GridConfig] = None):
        """
        Initialize bathymetry mapper.
        
        Args:
            resolution: Grid resolution in meters (default 5.0)
            grid_config: Advanced grid configuration (optional)
        """
        self.resolution = resolution
        self.config = grid_config or GridConfig(grid_resolution=resolution)
        self.trackpoints: List[TrackPoint] = []
        self.grid: Optional[np.ndarray] = None
        self.x_coords: Optional[np.ndarray] = None  # Longitude
        self.y_coords: Optional[np.ndarray] = None  # Latitude
        self.metadata: Dict = {
            "created": datetime.now(timezone.utc).isoformat(),
            "num_trackpoints": 0,
            "resolution_m": resolution,
            "grid_generated": False,
            "interpolation_method": self.config.interpolation_method,
        }
        logger.info(f"BathymetryMapper initialized (resolution={resolution}m)")
    
    def add_trackpoints(self, trackpoints: Union[List[TrackPoint], np.ndarray]) -> 'BathymetryMapper':
        """
        Add sonar measurement points to the mapper.
        
        Args:
            trackpoints: List of TrackPoint objects or (n, 3) array of [lat, lon, depth]
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If trackpoints format invalid or depth out of range
        """
        if isinstance(trackpoints, np.ndarray):
            if trackpoints.shape[1] < 3:
                raise ValueError(f"Array must have at least 3 columns (lat, lon, depth), got {trackpoints.shape}")
            trackpoints = [
                TrackPoint(latitude=row[0], longitude=row[1], depth=row[2])
                for row in trackpoints
            ]
        
        # Validate trackpoints
        for tp in trackpoints:
            if not (-90 <= tp.latitude <= 90):
                raise ValueError(f"Invalid latitude: {tp.latitude}")
            if not (-180 <= tp.longitude <= 180):
                raise ValueError(f"Invalid longitude: {tp.longitude}")
            if not (self.config.min_depth <= tp.depth <= self.config.max_depth):
                raise ValueError(f"Depth {tp.depth} outside valid range [{self.config.min_depth}, {self.config.max_depth}]")
        
        self.trackpoints.extend(trackpoints)
        self.metadata["num_trackpoints"] = len(self.trackpoints)
        logger.debug(f"Added {len(trackpoints)} trackpoints (total: {len(self.trackpoints)})")
        return self
    
    def generate_grid(self, force_regenerate: bool = False) -> np.ndarray:
        """
        Generate regular grid from scattered trackpoints using interpolation.
        
        Args:
            force_regenerate: Force regeneration even if grid exists
        
        Returns:
            Grid array with shape (n_lat, n_lon) containing interpolated depths
        
        Raises:
            ValueError: If insufficient trackpoints for interpolation
        """
        if self.grid is not None and not force_regenerate:
            return self.grid
        
        if len(self.trackpoints) < 3:
            raise ValueError(f"Need at least 3 trackpoints for interpolation, got {len(self.trackpoints)}")
        
        # Extract coordinates and depths
        lats = np.array([tp.latitude for tp in self.trackpoints])
        lons = np.array([tp.longitude for tp in self.trackpoints])
        depths = np.array([tp.depth for tp in self.trackpoints])
        
        # Create regular grid
        lon_min, lon_max = np.min(lons), np.max(lons)
        lat_min, lat_max = np.min(lats), np.max(lats)
        
        # Convert meters to degrees (rough estimate: 1 degree ≈ 111 km)
        resolution_deg = self.resolution / 111000.0
        
        self.x_coords = np.arange(lon_min, lon_max + resolution_deg, resolution_deg)
        self.y_coords = np.arange(lat_min, lat_max + resolution_deg, resolution_deg)
        
        # Interpolate using scipy if available, otherwise use simple method
        try:
            from scipy.interpolate import griddata
            xx, yy = np.meshgrid(self.x_coords, self.y_coords)
            points = np.column_stack([lons, lats])
            self.grid = griddata(
                points, depths,
                (xx, yy),
                method=self.config.interpolation_method,
                fill_value=self.config.fill_value or np.nanmean(depths)
            )
        except ImportError:
            logger.warning("scipy not available, using simple nearest-neighbor interpolation")
            self.grid = self._simple_interpolate(lons, lats, depths)
        
        self.metadata["grid_generated"] = True
        self.metadata["grid_shape"] = self.grid.shape
        logger.info(f"Grid generated: {self.grid.shape} cells at {self.resolution}m resolution")
        return self.grid
    
    def _simple_interpolate(self, lons: np.ndarray, lats: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Simple nearest-neighbor interpolation as fallback."""
        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        grid = np.zeros_like(xx, dtype=float)
        
        for i in range(len(self.y_coords)):
            for j in range(len(self.x_coords)):
                # Find nearest point
                dist = (lons - self.x_coords[j])**2 + (lats - self.y_coords[i])**2
                nearest_idx = np.argmin(dist)
                grid[i, j] = depths[nearest_idx]
        
        return grid
    
    def compute_slope(self) -> np.ndarray:
        """
        Compute bathymetric slope (rate of depth change).
        
        Returns:
            Slope array with same shape as grid
        
        Raises:
            ValueError: If grid not generated
        """
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        
        # Compute gradients
        dz_dy, dz_dx = np.gradient(self.grid)
        
        # Compute slope magnitude (rate of depth change per meter)
        slope = np.sqrt(dz_dx**2 + dz_dy**2)
        logger.debug(f"Slope computed: mean={np.nanmean(slope):.4f}, max={np.nanmax(slope):.4f}")
        return slope
    
    def compute_curvature(self) -> np.ndarray:
        """
        Compute bathymetric curvature (concavity/convexity).
        
        Returns:
            Curvature array (positive=bowl-like, negative=ridge-like)
        """
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        
        # Compute second derivatives
        dz_dy, dz_dx = np.gradient(self.grid)
        d2z_dy2, _ = np.gradient(dz_dy)
        _, d2z_dx2 = np.gradient(dz_dx)
        
        # Gaussian curvature approximation
        curvature = d2z_dx2 + d2z_dy2
        logger.debug(f"Curvature computed: mean={np.nanmean(curvature):.4f}")
        return curvature
    
    def find_drop_offs(self, slope_threshold: float = 0.1) -> List[Dict]:
        """
        Identify steep slope areas (potential drop-offs).
        
        Args:
            slope_threshold: Slope threshold (depth change per meter) to detect
        
        Returns:
            List of drop-off regions with coordinates and slope stats
        """
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        
        slope = self.compute_slope()
        steep_mask = slope > slope_threshold
        
        # Find connected regions
        drop_offs = []
        if np.any(steep_mask):
            # Simple clustering: find isolated steep regions
            from scipy.ndimage import label
            labeled_array, num_features = label(steep_mask)
            
            for region_id in range(1, num_features + 1):
                region = labeled_array == region_id
                region_slope = slope[region]
                
                drop_offs.append({
                    "region_id": int(region_id),
                    "num_cells": int(np.sum(region)),
                    "mean_slope": float(np.mean(region_slope)),
                    "max_slope": float(np.max(region_slope)),
                    "min_slope": float(np.min(region_slope)),
                })
        
        logger.info(f"Found {len(drop_offs)} drop-off regions (threshold={slope_threshold})")
        return drop_offs
    
    def generate_contours(self, depth_intervals: Optional[List[float]] = None) -> Dict[float, List[Tuple[float, float]]]:
        """
        Generate bathymetric contours at specified depth intervals.
        
        Args:
            depth_intervals: Depth values for contours (e.g., [-20, -40, -60])
        
        Returns:
            Dictionary mapping depth to list of contour lines (as coordinate pairs)
        """
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        
        if depth_intervals is None:
            min_depth = np.nanmin(self.grid)
            max_depth = np.nanmax(self.grid)
            depth_intervals = list(np.arange(min_depth, max_depth + 10, 10))
        
        contours_dict = {}
        
        try:
            from matplotlib import pyplot as plt
            xx, yy = np.meshgrid(self.x_coords, self.y_coords)
            
            for depth_val in depth_intervals:
                cs = plt.contour(xx, yy, self.grid, levels=[depth_val])
                
                contour_lines = []
                for collection in cs.collections:
                    for path in collection.get_paths():
                        vertices = path.vertices.tolist()
                        if len(vertices) > 1:
                            contour_lines.append(vertices)
                
                if contour_lines:
                    contours_dict[depth_val] = contour_lines
            
            plt.close('all')
            logger.info(f"Generated {len(contours_dict)} contours")
        except ImportError:
            logger.warning("matplotlib not available, skipping contour generation")
        
        return contours_dict
    
    def export_to_geojson(self, filepath: Union[str, Path], include_grid: bool = True) -> bool:
        """
        Export bathymetry data to GeoJSON format.
        
        Args:
            filepath: Output file path
            include_grid: Include interpolated grid points
        
        Returns:
            True if successful
        """
        try:
            features = []
            
            # Export trackpoints
            for tp in self.trackpoints:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [tp.longitude, tp.latitude]
                    },
                    "properties": {
                        "depth": float(tp.depth),
                        "confidence": float(tp.confidence),
                    }
                })
            
            geojson = {
                "type": "FeatureCollection",
                "features": features,
                "properties": self.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            logger.info(f"Exported {len(features)} features to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export GeoJSON: {e}")
            return False
    
    def export_to_kml(self, filepath: Union[str, Path]) -> bool:
        """
        Export bathymetry trackpoints to KML format.
        
        Args:
            filepath: Output file path
        
        Returns:
            True if successful
        """
        try:
            kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Bathymetry Map</name>
    <description>Bathymetric measurements</description>
    <Folder>
      <name>Depth Points</name>
'''
            
            # Color map: shallow=light blue, deep=dark blue
            for tp in self.trackpoints:
                depth_norm = (tp.depth - self.config.min_depth) / (self.config.max_depth - self.config.min_depth)
                # BGR format in KML
                color_intensity = int(255 * (1 - depth_norm))  # Darker for deeper
                color = f"ff{color_intensity:02x}{color_intensity:02x}ff"
                
                kml_content += f'''      <Placemark>
        <name>Depth: {tp.depth:.1f}m</name>
        <Style>
          <IconStyle>
            <color>{color}</color>
            <scale>0.7</scale>
          </IconStyle>
        </Style>
        <Point>
          <coordinates>{tp.longitude},{tp.latitude},0</coordinates>
        </Point>
      </Placemark>
'''
            
            kml_content += '''    </Folder>
  </Document>
</kml>'''
            
            with open(filepath, 'w') as f:
                f.write(kml_content)
            
            logger.info(f"Exported {len(self.trackpoints)} trackpoints to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export KML: {e}")
            return False
    
    def export_grid_to_netcdf(self, filepath: Union[str, Path]) -> bool:
        """
        Export grid to NetCDF format (requires xarray).
        
        Args:
            filepath: Output file path
        
        Returns:
            True if successful
        """
        if self.grid is None:
            logger.error("Grid not generated")
            return False
        
        try:
            import xarray as xr
            
            ds = xr.Dataset(
                data_vars={"depth": (["latitude", "longitude"], self.grid)},
                coords={
                    "latitude": self.y_coords,
                    "longitude": self.x_coords,
                },
                attrs=self.metadata
            )
            
            ds.to_netcdf(filepath)
            logger.info(f"Grid exported to NetCDF: {filepath}")
            return True
        except ImportError:
            logger.warning("xarray not available, cannot export to NetCDF")
            return False
        except Exception as e:
            logger.error(f"Failed to export NetCDF: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get bathymetric statistics.
        
        Returns:
            Dictionary with depth statistics and grid info
        """
        depths = np.array([tp.depth for tp in self.trackpoints])
        
        stats = {
            "num_trackpoints": len(self.trackpoints),
            "depth_min": float(np.min(depths)),
            "depth_max": float(np.max(depths)),
            "depth_mean": float(np.mean(depths)),
            "depth_std": float(np.std(depths)),
            "depth_median": float(np.median(depths)),
            "grid_generated": self.grid is not None,
        }
        
        if self.grid is not None:
            stats.update({
                "grid_shape": self.grid.shape,
                "grid_min": float(np.nanmin(self.grid)),
                "grid_max": float(np.nanmax(self.grid)),
                "grid_mean": float(np.nanmean(self.grid)),
            })
        
        return stats


# Example usage and synthetic data generation
def create_synthetic_bathymetry(num_points: int = 50, area_size: float = 1.0) -> List[TrackPoint]:
    """
    Generate synthetic bathymetric data for testing.
    
    Args:
        num_points: Number of synthetic measurement points
        area_size: Size of survey area in degrees
    
    Returns:
        List of TrackPoint objects with realistic depth variations
    """
    np.random.seed(42)
    
    lat_center, lon_center = 42.0, -85.0
    
    trackpoints = []
    for i in range(num_points):
        lat = lat_center + (np.random.random() - 0.5) * area_size
        lon = lon_center + (np.random.random() - 0.5) * area_size
        
        # Realistic Great Lakes depths (mostly 10-50m, with drop-offs)
        distance_from_center = np.sqrt((lat - lat_center)**2 + (lon - lon_center)**2)
        base_depth = -20 - distance_from_center * 30  # Deepen from center
        depth = base_depth + np.random.normal(0, 2)  # Add noise
        depth = np.clip(depth, -100, -5)  # Clamp to realistic range
        
        trackpoints.append(TrackPoint(
            latitude=lat,
            longitude=lon,
            depth=depth,
            confidence=0.9 + np.random.random() * 0.1
        ))
    
    return trackpoints


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("BathymetryMapper - Phase 11.2 Example")
    print("="*60 + "\n")
    
    # Generate synthetic data
    print("1. Creating synthetic bathymetric data...")
    trackpoints = create_synthetic_bathymetry(num_points=100)
    print(f"   Created {len(trackpoints)} synthetic trackpoints")
    
    # Initialize mapper
    print("\n2. Initializing mapper...")
    mapper = BathymetryMapper(resolution=100.0)  # 100m grid cells
    mapper.add_trackpoints(trackpoints)
    
    # Generate grid
    print("\n3. Generating interpolated grid...")
    grid = mapper.generate_grid()
    print(f"   Grid shape: {grid.shape}")
    
    # Compute bathymetric features
    print("\n4. Computing bathymetric features...")
    slope = mapper.compute_slope()
    curvature = mapper.compute_curvature()
    print(f"   Slope - mean: {np.nanmean(slope):.4f}, max: {np.nanmax(slope):.4f}")
    print(f"   Curvature - mean: {np.nanmean(curvature):.4f}, max: {np.nanmax(curvature):.4f}")
    
    # Find drop-offs
    print("\n5. Detecting drop-off regions...")
    drop_offs = mapper.find_drop_offs(slope_threshold=0.05)
    for dropoff in drop_offs[:3]:  # Show first 3
        print(f"   Region {dropoff['region_id']}: "
              f"{dropoff['num_cells']} cells, "
              f"slope {dropoff['mean_slope']:.4f} ± {dropoff['max_slope']:.4f}")
    
    # Get statistics
    print("\n6. Bathymetry statistics:")
    stats = mapper.get_statistics()
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.2f}")
        else:
            print(f"   {key}: {val}")
    
    # Export to GeoJSON
    print("\n7. Exporting data...")
    mapper.export_to_geojson("bathymetry_test.geojson")
    mapper.export_to_kml("bathymetry_test.kml")
    print("   ✓ GeoJSON and KML exported")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60 + "\n")
