"""
Sonar Visualization Module (Phase 12.2)

WebGL-based 3D visualization for real-time sonar data.
Renders bathymetry, anomalies, and composite mosaics with interactive controls.

Author: CESARops Development
Date: January 27, 2026
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import threading
import time


class BlendMode(Enum):
    """Blend modes for layer composition."""
    NORMAL = "normal"
    OVERLAY = "overlay"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    ADD = "add"


@dataclass
class Color:
    """RGB color representation."""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_hex(self) -> str:
        """Convert to hex color string."""
        r = int(self.r * 255)
        g = int(self.g * 255)
        b = int(self.b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict."""
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}


@dataclass
class CameraState:
    """Camera position and orientation."""
    position: Tuple[float, float, float]  # (x, y, z)
    target: Tuple[float, float, float]    # (x, y, z)
    up: Tuple[float, float, float] = (0, 0, 1)
    fov: float = 60.0  # Field of view
    near: float = 0.1
    far: float = 10000.0


@dataclass
class ViewportSettings:
    """Viewport configuration."""
    width: int
    height: int
    background_color: Color = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.background_color is None:
            self.background_color = Color(0.1, 0.1, 0.1, 1.0)


class BathymetryLayer:
    """
    Renders bathymetric data as 3D mesh.
    
    Converts depth grid to mesh with color-coded elevation.
    """
    
    def __init__(self, grid: np.ndarray, resolution: float = 1.0):
        """
        Initialize bathymetry layer.
        
        Args:
            grid: 2D depth array (positive values downward)
            resolution: Grid cell size in meters
        """
        self.grid = grid
        self.resolution = resolution
        self.visible = True
        self.opacity = 1.0
        self.colormap = self._default_colormap()
        self._mesh_data = None
    
    def _default_colormap(self) -> Dict[str, Tuple[float, float, float]]:
        """Get default depth colormap (shallow to deep)."""
        return {
            "0": (0.2, 0.8, 1.0),      # Light blue - shallow
            "5": (0.1, 0.6, 0.8),      # Blue
            "10": (0.0, 0.4, 0.6),     # Dark blue
            "20": (0.0, 0.2, 0.4),     # Very dark blue
            "50": (0.0, 0.1, 0.2),     # Nearly black - deep
        }
    
    def get_depth_color(self, depth: float) -> Color:
        """Get color for depth value."""
        depth_abs = abs(depth)
        
        if depth_abs < 5:
            color = self.colormap["0"]
        elif depth_abs < 10:
            color = self.colormap["5"]
        elif depth_abs < 20:
            color = self.colormap["10"]
        elif depth_abs < 50:
            color = self.colormap["20"]
        else:
            color = self.colormap["50"]
        
        return Color(color[0], color[1], color[2], self.opacity)
    
    def get_mesh_data(self) -> Dict[str, Any]:
        """
        Generate mesh data for rendering.
        
        Returns:
            Dict with vertices, indices, and colors
        """
        if self._mesh_data is not None:
            return self._mesh_data
        
        height, width = self.grid.shape
        vertices = []
        colors = []
        indices = []
        
        # Create vertices
        for y in range(height):
            for x in range(width):
                depth = self.grid[y, x]
                vertices.append([x * self.resolution, y * self.resolution, depth])
                
                color = self.get_depth_color(depth)
                colors.append(color.to_dict())
        
        # Create triangle indices
        for y in range(height - 1):
            for x in range(width - 1):
                v0 = y * width + x
                v1 = v0 + 1
                v2 = v0 + width
                v3 = v2 + 1
                
                # Two triangles per quad
                indices.extend([v0, v1, v2, v1, v3, v2])
        
        self._mesh_data = {
            "vertices": vertices,
            "colors": colors,
            "indices": indices,
            "visible": self.visible,
            "opacity": self.opacity
        }
        
        return self._mesh_data


class AnomalyLayer:
    """
    Renders sonar anomalies as point cloud or markers.
    """
    
    def __init__(self, anomalies: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize anomaly layer.
        
        Args:
            anomalies: List of anomaly dicts with position and type
        """
        self.anomalies = anomalies or []
        self.visible = True
        self.opacity = 1.0
        self.point_size = 3.0
        self.type_colors = self._default_type_colors()
    
    def _default_type_colors(self) -> Dict[str, Color]:
        """Get color mapping for anomaly types."""
        return {
            "strong_return": Color(1.0, 0.0, 0.0),          # Red
            "structured_pattern": Color(1.0, 1.0, 0.0),     # Yellow
            "persistent_feature": Color(0.0, 1.0, 0.0),     # Green
            "unusual_pattern": Color(1.0, 0.5, 0.0),        # Orange
        }
    
    def add_anomaly(self, x: float, y: float, z: float, 
                   anomaly_type: str, confidence: float = 1.0) -> None:
        """
        Add anomaly point.
        
        Args:
            x, y, z: Position coordinates
            anomaly_type: Type of anomaly
            confidence: Confidence score (0-1)
        """
        self.anomalies.append({
            "position": [x, y, z],
            "type": anomaly_type,
            "confidence": confidence
        })
        self._invalidate_cache()
    
    def get_point_cloud_data(self) -> Dict[str, Any]:
        """
        Generate point cloud data for rendering.
        
        Returns:
            Dict with points and colors
        """
        points = []
        colors = []
        sizes = []
        
        for anom in self.anomalies:
            points.append(anom["position"])
            
            # Get color by type
            anom_type = anom.get("type", "unusual_pattern")
            color = self.type_colors.get(anom_type, Color(0.5, 0.5, 0.5))
            colors.append(color.to_dict())
            
            # Size by confidence
            confidence = anom.get("confidence", 1.0)
            sizes.append(self.point_size * confidence)
        
        return {
            "points": points,
            "colors": colors,
            "sizes": sizes,
            "visible": self.visible,
            "opacity": self.opacity
        }
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached mesh data."""
        pass


class MosaicLayer:
    """
    Renders composite mosaic as textured surface.
    """
    
    def __init__(self, mosaic_data: np.ndarray, width_m: float, height_m: float):
        """
        Initialize mosaic layer.
        
        Args:
            mosaic_data: 2D array of mosaic values
            width_m: Width in meters
            height_m: Height in meters
        """
        self.mosaic_data = mosaic_data
        self.width_m = width_m
        self.height_m = height_m
        self.visible = True
        self.opacity = 0.7
        self.blend_mode = BlendMode.NORMAL
        self.elevation = 0.0
    
    def get_texture_data(self) -> Dict[str, Any]:
        """
        Generate texture data for rendering.
        
        Returns:
            Dict with texture and geometry
        """
        # Normalize mosaic to 0-1 range
        min_val = np.min(self.mosaic_data)
        max_val = np.max(self.mosaic_data)
        
        if max_val > min_val:
            normalized = (self.mosaic_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(self.mosaic_data)
        
        # Convert to color texture (grayscale)
        height, width = self.mosaic_data.shape
        texture = np.zeros((height, width, 4), dtype=np.uint8)
        
        for c in range(3):
            texture[:, :, c] = (normalized * 255).astype(np.uint8)
        texture[:, :, 3] = int(self.opacity * 255)
        
        return {
            "texture": texture.tolist(),
            "width": width,
            "height": height,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "elevation": self.elevation,
            "blend_mode": self.blend_mode.value,
            "visible": self.visible,
            "opacity": self.opacity
        }


class WebGLRenderer:
    """
    Main WebGL rendering engine for sonar visualization.
    
    Manages camera, lighting, layers, and composite rendering.
    """
    
    def __init__(self, viewport: ViewportSettings):
        """
        Initialize renderer.
        
        Args:
            viewport: Viewport configuration
        """
        self.viewport = viewport
        self.camera = self._default_camera()
        
        self.layers: Dict[str, Any] = {}
        self.layer_order: List[str] = []
        
        self.lighting_settings = self._default_lighting()
        self.render_stats = {
            "frame_time_ms": 0,
            "fps": 0,
            "vertices_rendered": 0,
            "triangles_rendered": 0
        }
        
        self._lock = threading.RLock()
        self._frame_times = []
    
    def _default_camera(self) -> CameraState:
        """Get default camera position."""
        return CameraState(
            position=(0, 0, 50),
            target=(0, 0, 0),
            fov=60.0
        )
    
    def _default_lighting(self) -> Dict[str, Any]:
        """Get default lighting settings."""
        return {
            "ambient": 0.6,
            "directional": 0.4,
            "direction": [1, 1, 1],
            "shadows_enabled": False
        }
    
    def add_bathymetry_layer(self, name: str, grid: np.ndarray, 
                            resolution: float = 1.0) -> None:
        """Add bathymetry layer."""
        with self._lock:
            layer = BathymetryLayer(grid, resolution)
            self.layers[name] = layer
            if name not in self.layer_order:
                self.layer_order.append(name)
    
    def add_anomaly_layer(self, name: str) -> None:
        """Add anomaly layer."""
        with self._lock:
            self.layers[name] = AnomalyLayer()
            if name not in self.layer_order:
                self.layer_order.append(name)
    
    def add_mosaic_layer(self, name: str, mosaic_data: np.ndarray,
                        width_m: float, height_m: float) -> None:
        """Add mosaic layer."""
        with self._lock:
            layer = MosaicLayer(mosaic_data, width_m, height_m)
            self.layers[name] = layer
            if name not in self.layer_order:
                self.layer_order.append(name)
    
    def set_camera_position(self, position: Tuple[float, float, float]) -> None:
        """Set camera position."""
        with self._lock:
            self.camera.position = position
    
    def set_camera_target(self, target: Tuple[float, float, float]) -> None:
        """Set camera target/look-at point."""
        with self._lock:
            self.camera.target = target
    
    def rotate_camera(self, delta_yaw: float, delta_pitch: float) -> None:
        """Rotate camera around target."""
        # Simplified camera rotation
        x, y, z = self.camera.position
        target_x, target_y, target_z = self.camera.target
        
        # Convert to spherical coordinates
        dx = x - target_x
        dy = y - target_y
        dz = z - target_z
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Rotate
        angle_xy = np.arctan2(dy, dx) + delta_yaw * np.pi / 180
        angle_z = np.arcsin(dz / distance) + delta_pitch * np.pi / 180
        
        # Clamp pitch
        angle_z = np.clip(angle_z, -np.pi/2, np.pi/2)
        
        # Convert back to Cartesian
        self.camera.position = (
            target_x + distance * np.cos(angle_z) * np.cos(angle_xy),
            target_y + distance * np.cos(angle_z) * np.sin(angle_xy),
            target_z + distance * np.sin(angle_z)
        )
    
    def zoom_camera(self, zoom_factor: float) -> None:
        """Zoom camera (positive = zoom in)."""
        x, y, z = self.camera.position
        target_x, target_y, target_z = self.camera.target
        
        dx = x - target_x
        dy = y - target_y
        dz = z - target_z
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        new_distance = distance * (1 - zoom_factor * 0.1)
        
        if new_distance > 1:  # Minimum distance
            scale = new_distance / distance
            self.camera.position = (
                target_x + dx * scale,
                target_y + dy * scale,
                target_z + dz * scale
            )
    
    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Toggle layer visibility."""
        with self._lock:
            if layer_name in self.layers:
                self.layers[layer_name].visible = visible
    
    def render_frame(self) -> Dict[str, Any]:
        """
        Render current frame.
        
        Returns:
            Frame data with all layers
        """
        start_time = time.time()
        
        with self._lock:
            frame_data = {
                "viewport": {
                    "width": self.viewport.width,
                    "height": self.viewport.height,
                    "background": self.viewport.background_color.to_dict()
                },
                "camera": {
                    "position": self.camera.position,
                    "target": self.camera.target,
                    "fov": self.camera.fov
                },
                "lighting": self.lighting_settings,
                "layers": {}
            }
            
            total_vertices = 0
            total_triangles = 0
            
            # Render layers in order
            for layer_name in self.layer_order:
                if layer_name not in self.layers:
                    continue
                
                layer = self.layers[layer_name]
                if not layer.visible:
                    continue
                
                if isinstance(layer, BathymetryLayer):
                    mesh_data = layer.get_mesh_data()
                    frame_data["layers"][layer_name] = mesh_data
                    total_vertices += len(mesh_data.get("vertices", []))
                    total_triangles += len(mesh_data.get("indices", [])) // 3
                
                elif isinstance(layer, AnomalyLayer):
                    cloud_data = layer.get_point_cloud_data()
                    frame_data["layers"][layer_name] = cloud_data
                    total_vertices += len(cloud_data.get("points", []))
                
                elif isinstance(layer, MosaicLayer):
                    texture_data = layer.get_texture_data()
                    frame_data["layers"][layer_name] = texture_data
            
            # Update statistics
            frame_time = (time.time() - start_time) * 1000
            self._frame_times.append(frame_time)
            
            if len(self._frame_times) > 60:
                self._frame_times.pop(0)
            
            avg_frame_time = np.mean(self._frame_times)
            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            
            self.render_stats = {
                "frame_time_ms": frame_time,
                "fps": fps,
                "vertices_rendered": total_vertices,
                "triangles_rendered": total_triangles
            }
            
            frame_data["stats"] = self.render_stats
            
            return frame_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        with self._lock:
            return self.render_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._frame_times.clear()
            self.render_stats = {
                "frame_time_ms": 0,
                "fps": 0,
                "vertices_rendered": 0,
                "triangles_rendered": 0
            }


class VisualizationPipeline:
    """
    Complete visualization pipeline from streaming data to rendered frames.
    
    Combines streaming (Phase 12.1), visualization (Phase 12.2),
    and analysis modules (Phase 11).
    """
    
    def __init__(self, renderer: WebGLRenderer):
        """
        Initialize pipeline.
        
        Args:
            renderer: WebGLRenderer instance
        """
        self.renderer = renderer
        self.frame_queue = []
        self.lock = threading.RLock()
    
    def add_frame_data(self, frame_data: Dict[str, Any]) -> None:
        """Add new frame data to pipeline."""
        with self.lock:
            self.frame_queue.append({
                "timestamp": time.time(),
                "data": frame_data
            })
    
    def get_next_frame(self) -> Optional[Dict[str, Any]]:
        """Get next frame to render."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue.pop(0)
            return None
    
    def render(self) -> Dict[str, Any]:
        """Render current frame."""
        return self.renderer.render_frame()
