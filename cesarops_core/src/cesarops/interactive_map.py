"""
Interactive Map Module for Phase 12.3

Provides interactive user controls for sonar visualization:
- Layer toggling
- Camera controls
- Statistics display
- Snapshot export

Author: CESARops Development
Date: January 27, 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import threading
import json
from datetime import datetime
from collections import deque


@dataclass
class MapSettings:
    """Interactive map configuration."""
    width: int = 1024
    height: int = 768
    enable_keyboard: bool = True
    enable_mouse: bool = True
    enable_touch: bool = False
    statistics_update_hz: int = 30
    export_format: str = "png"  # png, jpg, webm


class LayerToggle:
    """Manages layer visibility toggles."""
    
    def __init__(self):
        """Initialize layer toggles."""
        self.layers: Dict[str, bool] = {}
        self._lock = threading.RLock()
    
    def add_layer(self, name: str, visible: bool = True) -> None:
        """Add a layer to toggles."""
        with self._lock:
            self.layers[name] = visible
    
    def toggle_layer(self, name: str) -> bool:
        """Toggle layer visibility."""
        with self._lock:
            if name in self.layers:
                self.layers[name] = not self.layers[name]
                return self.layers[name]
            return False
    
    def set_layer_visible(self, name: str, visible: bool) -> None:
        """Set layer visibility."""
        with self._lock:
            if name in self.layers:
                self.layers[name] = visible
    
    def get_layer_state(self, name: str) -> Optional[bool]:
        """Get layer visibility state."""
        with self._lock:
            return self.layers.get(name)
    
    def get_visible_layers(self) -> List[str]:
        """Get list of visible layers."""
        with self._lock:
            return [name for name, visible in self.layers.items() if visible]
    
    def get_all_layers(self) -> Dict[str, bool]:
        """Get all layer states."""
        with self._lock:
            return self.layers.copy()


class StatisticsPanel:
    """Real-time statistics display."""
    
    def __init__(self, update_hz: int = 30):
        """Initialize statistics panel."""
        self.update_hz = update_hz
        self.update_interval = 1.0 / update_hz
        
        self.fps = 0.0
        self.frame_time_ms = 0.0
        self.vertices_rendered = 0
        self.triangles_rendered = 0
        self.memory_usage_mb = 0.0
        self.streaming_rate_hz = 0.0
        self.anomalies_detected = 0
        
        self._history: Dict[str, deque] = {
            'fps': deque(maxlen=100),
            'frame_time': deque(maxlen=100),
            'vertices': deque(maxlen=100),
        }
        self._lock = threading.RLock()
        self._last_update = 0.0
    
    def update_from_renderer(self, render_stats: Dict) -> None:
        """Update stats from renderer."""
        with self._lock:
            now = datetime.now().timestamp()
            if now - self._last_update < self.update_interval:
                return
            
            self.fps = render_stats.get("fps", 0.0)
            self.frame_time_ms = render_stats.get("frame_time_ms", 0.0)
            self.vertices_rendered = render_stats.get("vertices_rendered", 0)
            
            # Track history
            self._history['fps'].append(self.fps)
            self._history['frame_time'].append(self.frame_time_ms)
            self._history['vertices'].append(self.vertices_rendered)
            
            self._last_update = now
    
    def update_streaming_stats(self, rate_hz: float, anomalies: int) -> None:
        """Update streaming statistics."""
        with self._lock:
            self.streaming_rate_hz = rate_hz
            self.anomalies_detected = anomalies
    
    def get_stats_dict(self) -> Dict:
        """Get statistics as dictionary."""
        with self._lock:
            return {
                'fps': round(self.fps, 2),
                'frame_time_ms': round(self.frame_time_ms, 2),
                'vertices_rendered': self.vertices_rendered,
                'triangles_rendered': self.triangles_rendered,
                'memory_usage_mb': round(self.memory_usage_mb, 2),
                'streaming_rate_hz': round(self.streaming_rate_hz, 2),
                'anomalies_detected': self.anomalies_detected,
            }
    
    def get_average_fps(self) -> float:
        """Get average FPS over history."""
        with self._lock:
            if self._history['fps']:
                return np.mean(list(self._history['fps']))
            return 0.0
    
    def get_average_frame_time(self) -> float:
        """Get average frame time."""
        with self._lock:
            if self._history['frame_time']:
                return np.mean(list(self._history['frame_time']))
            return 0.0


class SnapshotExporter:
    """Handle snapshot and video export."""
    
    def __init__(self, export_dir: str = "outputs"):
        """Initialize exporter."""
        self.export_dir = export_dir
        self.snapshot_format = "png"
        self.video_format = "webm"
        self.video_bitrate = "8M"
        self.video_fps = 30
        
        self._lock = threading.RLock()
        self._export_queue: List[Dict] = []
        self._export_history: List[Dict] = []
    
    def export_snapshot(self, frame_data: np.ndarray, filename: Optional[str] = None) -> str:
        """Export current frame as image."""
        with self._lock:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.{self.snapshot_format}"
            
            filepath = f"{self.export_dir}/{filename}"
            
            # In real implementation, would save frame data
            export_info = {
                'type': 'snapshot',
                'filename': filename,
                'filepath': filepath,
                'timestamp': datetime.now().isoformat(),
                'format': self.snapshot_format,
            }
            self._export_history.append(export_info)
            
            return filepath
    
    def start_video_export(self, output_filename: Optional[str] = None) -> str:
        """Start recording video."""
        with self._lock:
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"video_{timestamp}.{self.video_format}"
            
            filepath = f"{self.export_dir}/{output_filename}"
            
            export_info = {
                'type': 'video',
                'filename': output_filename,
                'filepath': filepath,
                'timestamp': datetime.now().isoformat(),
                'format': self.video_format,
                'status': 'recording',
            }
            self._export_queue.append(export_info)
            
            return filepath
    
    def stop_video_export(self) -> Optional[str]:
        """Stop video recording."""
        with self._lock:
            if self._export_queue:
                export_info = self._export_queue.pop(0)
                export_info['status'] = 'completed'
                export_info['completed_at'] = datetime.now().isoformat()
                self._export_history.append(export_info)
                return export_info['filepath']
            return None
    
    def get_export_history(self) -> List[Dict]:
        """Get export history."""
        with self._lock:
            return self._export_history.copy()


class NavigationControls:
    """Handle camera navigation."""
    
    def __init__(self):
        """Initialize navigation."""
        self.camera_speed = 10.0  # units per key press
        self.rotation_speed = 5.0  # degrees per pixel
        self.zoom_speed = 0.1
        
        self.pan_enabled = True
        self.rotate_enabled = True
        self.zoom_enabled = True
        
        self._key_state: Dict[str, bool] = {}
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def register_callback(self, callback: Callable) -> None:
        """Register camera change callback."""
        with self._lock:
            self._callbacks.append(callback)
    
    def on_key_down(self, key: str) -> None:
        """Handle key press."""
        with self._lock:
            self._key_state[key] = True
            self._trigger_callbacks(f"key_down:{key}")
    
    def on_key_up(self, key: str) -> None:
        """Handle key release."""
        with self._lock:
            self._key_state[key] = False
            self._trigger_callbacks(f"key_up:{key}")
    
    def on_mouse_drag(self, dx: float, dy: float) -> Dict:
        """Handle mouse drag for camera control."""
        with self._lock:
            rotation = {
                'yaw': dx * self.rotation_speed,
                'pitch': dy * self.rotation_speed,
            }
            self._trigger_callbacks("mouse_drag")
            return rotation
    
    def on_mouse_scroll(self, delta: float) -> float:
        """Handle mouse scroll for zoom."""
        with self._lock:
            zoom_factor = 1.0 - (delta * self.zoom_speed)
            self._trigger_callbacks("mouse_scroll")
            return zoom_factor
    
    def get_movement_vector(self) -> Tuple[float, float, float]:
        """Get movement vector from key state."""
        with self._lock:
            x = (float(self._key_state.get('d', False)) - 
                 float(self._key_state.get('a', False)))
            y = (float(self._key_state.get('w', False)) - 
                 float(self._key_state.get('s', False)))
            z = (float(self._key_state.get('e', False)) - 
                 float(self._key_state.get('q', False)))
            
            return x * self.camera_speed, y * self.camera_speed, z * self.camera_speed
    
    def _trigger_callbacks(self, event: str) -> None:
        """Trigger registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass


class InteractiveMap:
    """Main interactive map interface."""
    
    def __init__(self, settings: Optional[MapSettings] = None):
        """Initialize interactive map."""
        self.settings = settings or MapSettings()
        
        self.layer_toggle = LayerToggle()
        self.statistics = StatisticsPanel(self.settings.statistics_update_hz)
        self.exporter = SnapshotExporter()
        self.navigation = NavigationControls()
        
        self._active = False
        self._lock = threading.RLock()
        
        # Data references
        self._renderer = None
        self._stream_manager = None
    
    def setup(self, renderer, stream_manager) -> None:
        """Setup with renderer and stream manager."""
        with self._lock:
            self._renderer = renderer
            self._stream_manager = stream_manager
            
            # Register layers
            if hasattr(renderer, 'layers'):
                for layer_name in renderer.layers.keys():
                    self.layer_toggle.add_layer(layer_name)
    
    def activate(self) -> None:
        """Activate interactive map."""
        with self._lock:
            self._active = True
    
    def deactivate(self) -> None:
        """Deactivate interactive map."""
        with self._lock:
            self._active = False
    
    def is_active(self) -> bool:
        """Check if active."""
        with self._lock:
            return self._active
    
    def update(self) -> Dict:
        """Update map state and return current state."""
        with self._lock:
            if not self._active or self._renderer is None:
                return {}
            
            # Get render stats
            render_stats = self._renderer.get_stats()
            self.statistics.update_from_renderer(render_stats)
            
            # Get streaming stats if available
            if self._stream_manager:
                stream_stats = getattr(self._stream_manager, 'get_stats', lambda: {})()
                rate = stream_stats.get('samples_per_second', 0)
                anomalies = stream_stats.get('anomalies_detected', 0)
                self.statistics.update_streaming_stats(rate, anomalies)
            
            # Build state dict
            state = {
                'timestamp': datetime.now().isoformat(),
                'layers': self.layer_toggle.get_all_layers(),
                'statistics': self.statistics.get_stats_dict(),
                'camera': {
                    'position': self._renderer.camera.position,
                    'target': self._renderer.camera.target,
                    'fov': self._renderer.camera.fov,
                },
            }
            
            return state
    
    def handle_layer_toggle(self, layer_name: str) -> bool:
        """Handle layer toggle from UI."""
        with self._lock:
            visible = self.layer_toggle.toggle_layer(layer_name)
            if self._renderer:
                self._renderer.set_layer_visibility(layer_name, visible)
            return visible
    
    def handle_camera_pan(self, dx: float, dy: float) -> None:
        """Handle camera pan."""
        with self._lock:
            if self._renderer:
                rotation = self.navigation.on_mouse_drag(dx, dy)
                self._renderer.rotate_camera(rotation['yaw'], rotation['pitch'])
    
    def handle_camera_zoom(self, delta: float) -> None:
        """Handle camera zoom."""
        with self._lock:
            if self._renderer:
                zoom_factor = self.navigation.on_mouse_scroll(delta)
                self._renderer.zoom_camera(zoom_factor)
    
    def handle_snapshot_export(self) -> str:
        """Handle snapshot export request."""
        with self._lock:
            frame_data = None  # Would get from renderer
            return self.exporter.export_snapshot(frame_data)
    
    def handle_video_export_start(self) -> str:
        """Start video export."""
        with self._lock:
            return self.exporter.start_video_export()
    
    def handle_video_export_stop(self) -> Optional[str]:
        """Stop video export."""
        with self._lock:
            return self.exporter.stop_video_export()
    
    def get_ui_state(self) -> Dict:
        """Get complete UI state for rendering."""
        with self._lock:
            return {
                'map_active': self._active,
                'layers': self.layer_toggle.get_all_layers(),
                'statistics': self.statistics.get_stats_dict(),
                'export_history': self.exporter.get_export_history(),
                'settings': {
                    'width': self.settings.width,
                    'height': self.settings.height,
                    'keyboard_enabled': self.settings.enable_keyboard,
                    'mouse_enabled': self.settings.enable_mouse,
                },
            }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        with self._lock:
            return {
                'type': 'interactive_map',
                'active': self._active,
                'settings': {
                    'width': self.settings.width,
                    'height': self.settings.height,
                    'enable_keyboard': self.settings.enable_keyboard,
                    'enable_mouse': self.settings.enable_mouse,
                    'enable_touch': self.settings.enable_touch,
                    'statistics_update_hz': self.settings.statistics_update_hz,
                },
                'layers': self.layer_toggle.get_all_layers(),
                'statistics': self.statistics.get_stats_dict(),
            }
