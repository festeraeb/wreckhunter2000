"""
Phase 13.2: Multi-Sensor Fusion Module

Combines sonar data with bathymetry, currents, temperature, and other sensors
for enhanced understanding of underwater environments.

Classes:
    BaseSensor: Abstract base for all sensor implementations
    SensorReading: Data class for single sensor reading
    SensorReadings: Container for multiple synchronized readings
    SensorDataBus: Central hub for all sensor data
    SensorQualityEstimator: Quality and reliability metrics
    CoordinateSystemManager: Coordinate system transformations
    FusedData: Fused multi-sensor data
    CorrelationResult: Results from multi-sensor correlations
    EnvironmentalContext: Complete environmental picture
    FusionEngine: Core multi-sensor analysis engine
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from collections import defaultdict


# ============================================================================
# Enumerations
# ============================================================================

class SensorType(Enum):
    """Supported sensor types."""
    SONAR = "sonar"
    BATHYMETRY = "bathymetry"
    CURRENT = "current"
    TEMPERATURE = "temperature"
    WIND = "wind"
    WAVE = "wave"
    TIDE = "tide"
    SALINITY = "salinity"


class CoordinateSystem(Enum):
    """Supported coordinate systems."""
    WGS84 = "WGS84"  # Global GPS
    GREAT_LAKES_LOCAL = "great_lakes_local"  # Local Great Lakes
    VESSEL_RELATIVE = "vessel_relative"  # Relative to vessel
    SONAR_LOCAL = "sonar_local"  # Sonar transducer frame


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Location:
    """Geographic location with coordinate system."""
    x: float  # Longitude or local X
    y: float  # Latitude or local Y
    z: float = 0.0  # Depth or elevation
    coordinate_system: CoordinateSystem = CoordinateSystem.WGS84
    uncertainty: float = 0.0  # Meters of uncertainty
    
    def __hash__(self):
        return hash((self.x, self.y, self.z, self.coordinate_system.value))


@dataclass
class SensorReading:
    """Single sensor reading with metadata."""
    sensor_type: SensorType
    value: Any  # Can be float, Dict, array, etc.
    location: Optional[Location] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    quality: float = 1.0  # 0.0-1.0 quality metric
    confidence: float = 1.0  # 0.0-1.0 confidence
    units: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReadings:
    """Collection of synchronized sensor readings."""
    readings: Dict[SensorType, SensorReading] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    synchronization_error_ms: float = 0.0
    
    def get_reading(self, sensor_type: SensorType) -> Optional[SensorReading]:
        """Get reading for specific sensor type."""
        return self.readings.get(sensor_type)
    
    def has_reading(self, sensor_type: SensorType) -> bool:
        """Check if reading exists for sensor type."""
        return sensor_type in self.readings
    
    def add_reading(self, reading: SensorReading) -> None:
        """Add reading to collection."""
        self.readings[reading.sensor_type] = reading


@dataclass
class FusedData:
    """Result of multi-sensor fusion."""
    timestamp: float
    location: Location
    
    # Fused values
    sonar_intensity: Optional[float] = None
    depth: Optional[float] = None
    current_velocity: Optional[Tuple[float, float]] = None  # (u, v)
    temperature: Optional[float] = None
    wind_velocity: Optional[Tuple[float, float]] = None
    wave_height: Optional[float] = None
    
    # Quality metrics
    fusion_confidence: float = 0.8
    sensor_quality_weights: Dict[str, float] = field(default_factory=dict)
    synchronization_quality: float = 1.0
    
    # Source information
    contributing_sensors: List[SensorType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Results from multi-sensor correlation analysis."""
    primary_anomaly_id: str
    timestamp: float
    
    # Correlated observations
    sonar_characteristics: Dict[str, float] = field(default_factory=dict)
    bathymetric_features: List[str] = field(default_factory=list)
    environmental_conditions: Dict[str, float] = field(default_factory=dict)
    
    # Correlation strength
    correlation_score: float = 0.0  # 0.0-1.0
    confidence: float = 0.0
    
    # Interpretation
    interpretation: str = ""
    likely_cause: str = "unknown"
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class EnvironmentalContext:
    """Complete environmental picture for analysis."""
    location: Location
    timestamp: float
    
    # Physical parameters
    depth: float = 0.0
    temperature: float = 0.0
    current_strength: float = 0.0
    current_direction: float = 0.0  # Degrees
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    wave_height: float = 0.0
    water_level: float = 0.0
    
    # Bathymetric context
    bathymetric_slope: float = 0.0  # Degrees
    nearby_features: List[str] = field(default_factory=list)
    
    # Quality
    overall_quality: float = 0.8
    data_freshness_seconds: float = 0.0
    sensor_count: int = 0


# ============================================================================
# Sensor Abstraction
# ============================================================================

class BaseSensor(ABC):
    """Abstract base class for all sensors."""
    
    def __init__(self, sensor_type: SensorType, name: str = ""):
        self.sensor_type = sensor_type
        self.name = name or sensor_type.value
        self.last_reading: Optional[SensorReading] = None
        self.read_count = 0
    
    @abstractmethod
    def read(self) -> Optional[SensorReading]:
        """Read current sensor value."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if sensor is available and functioning."""
        pass
    
    def get_quality(self) -> float:
        """Get estimated quality of sensor (0.0-1.0)."""
        if self.last_reading is None:
            return 0.0
        return self.last_reading.quality


# ============================================================================
# Sensor Quality Estimation
# ============================================================================

class SensorQualityEstimator:
    """Estimates reliability and quality of each sensor."""
    
    def __init__(self):
        self.quality_history: Dict[SensorType, List[float]] = defaultdict(list)
        self.max_history = 100
    
    def estimate_sonar_quality(self, signal_strength: float = 0.8,
                               noise_level: float = 0.1,
                               multipath_error: float = 0.05) -> float:
        """
        Estimate sonar quality.
        
        Args:
            signal_strength: 0.0-1.0
            noise_level: 0.0-1.0
            multipath_error: 0.0-1.0
        
        Returns:
            Quality score 0.0-1.0
        """
        # Strong signal + low noise + low multipath = high quality
        quality = (signal_strength * 0.5 + 
                  (1.0 - noise_level) * 0.3 +
                  (1.0 - multipath_error) * 0.2)
        
        self.quality_history[SensorType.SONAR].append(quality)
        if len(self.quality_history[SensorType.SONAR]) > self.max_history:
            self.quality_history[SensorType.SONAR].pop(0)
        
        return min(1.0, max(0.0, quality))
    
    def estimate_bathymetry_quality(self, coverage: float = 0.9,
                                    resolution: float = 0.5,
                                    age_hours: float = 24) -> float:
        """
        Estimate bathymetry quality.
        
        Args:
            coverage: 0.0-1.0 data coverage
            resolution: 0.0-1.0 relative resolution
            age_hours: Age of data in hours
        
        Returns:
            Quality score 0.0-1.0
        """
        # Coverage + resolution - age penalty
        freshness = max(0.0, 1.0 - (age_hours / 168.0))  # 1 week = 0.0
        quality = coverage * 0.4 + resolution * 0.4 + freshness * 0.2
        
        self.quality_history[SensorType.BATHYMETRY].append(quality)
        if len(self.quality_history[SensorType.BATHYMETRY]) > self.max_history:
            self.quality_history[SensorType.BATHYMETRY].pop(0)
        
        return min(1.0, max(0.0, quality))
    
    def estimate_current_quality(self, station_proximity_km: float = 0.0,
                                 data_freshness_minutes: float = 30) -> float:
        """
        Estimate current quality.
        
        Args:
            station_proximity_km: Distance to reference station
            data_freshness_minutes: Minutes since last update
        
        Returns:
            Quality score 0.0-1.0
        """
        # Closer station + fresher data = higher quality
        proximity_quality = max(0.0, 1.0 - (station_proximity_km / 50.0))
        freshness_quality = max(0.0, 1.0 - (data_freshness_minutes / 180.0))
        
        quality = proximity_quality * 0.5 + freshness_quality * 0.5
        
        self.quality_history[SensorType.CURRENT].append(quality)
        if len(self.quality_history[SensorType.CURRENT]) > self.max_history:
            self.quality_history[SensorType.CURRENT].pop(0)
        
        return min(1.0, max(0.0, quality))
    
    def estimate_temperature_quality(self, measurement_error: float = 0.1,
                                     calibration_days_ago: int = 30) -> float:
        """
        Estimate temperature quality.
        
        Args:
            measurement_error: Absolute error in degrees C
            calibration_days_ago: Days since last calibration
        
        Returns:
            Quality score 0.0-1.0
        """
        error_quality = max(0.0, 1.0 - (measurement_error / 2.0))
        calibration_quality = max(0.0, 1.0 - (calibration_days_ago / 180.0))
        
        quality = error_quality * 0.6 + calibration_quality * 0.4
        
        self.quality_history[SensorType.TEMPERATURE].append(quality)
        if len(self.quality_history[SensorType.TEMPERATURE]) > self.max_history:
            self.quality_history[SensorType.TEMPERATURE].pop(0)
        
        return min(1.0, max(0.0, quality))
    
    def get_average_quality(self, sensor_type: SensorType) -> float:
        """Get average quality for sensor type."""
        history = self.quality_history.get(sensor_type, [])
        if not history:
            return 0.5
        return np.mean(history)


# ============================================================================
# Coordinate System Management
# ============================================================================

class CoordinateSystemManager:
    """Manages coordinate systems and transformations."""
    
    def __init__(self):
        self.systems: Dict[str, Dict[str, Any]] = {}
        self._register_default_systems()
    
    def _register_default_systems(self) -> None:
        """Register standard coordinate systems."""
        self.systems[CoordinateSystem.WGS84.value] = {
            "name": "WGS84",
            "description": "Global GPS coordinates",
            "origin": (0.0, 0.0),
            "units": "degrees"
        }
        self.systems[CoordinateSystem.GREAT_LAKES_LOCAL.value] = {
            "name": "Great Lakes Local",
            "description": "Local Great Lakes coordinate system",
            "origin": (42.0, -87.0),  # Lake Michigan center
            "units": "kilometers"
        }
        self.systems[CoordinateSystem.VESSEL_RELATIVE.value] = {
            "name": "Vessel Relative",
            "description": "Relative to vessel position",
            "origin": (0.0, 0.0),
            "units": "meters"
        }
        self.systems[CoordinateSystem.SONAR_LOCAL.value] = {
            "name": "Sonar Local",
            "description": "Sonar transducer frame",
            "origin": (0.0, 0.0),
            "units": "meters"
        }
    
    def register_coordinate_system(self, name: str, definition: Dict[str, Any]) -> None:
        """Register new coordinate system."""
        self.systems[name] = definition
    
    def transform(self, point: Location, to_system: CoordinateSystem) -> Location:
        """
        Transform point between coordinate systems.
        
        Note: Full geodetic transformations would require more complex math.
        This is a simplified version for demonstration.
        """
        if point.coordinate_system == to_system:
            return point
        
        # Simplified transformation
        x, y = point.x, point.y
        
        # Convert from source system to WGS84 (intermediate)
        if point.coordinate_system == CoordinateSystem.GREAT_LAKES_LOCAL:
            # Convert from km offset to degrees (approximate)
            x = 42.0 + (x / 111.0)  # 1 degree ≈ 111 km
            y = -87.0 + (y / 111.0)
        
        # Convert from WGS84 to target system
        if to_system == CoordinateSystem.GREAT_LAKES_LOCAL:
            x = (x - 42.0) * 111.0  # Convert back to km
            y = (y - (-87.0)) * 111.0
        
        return Location(x, y, point.z, to_system, point.uncertainty)
    
    def get_system_info(self, system: CoordinateSystem) -> Optional[Dict]:
        """Get information about coordinate system."""
        return self.systems.get(system.value)


# ============================================================================
# Sensor Data Bus
# ============================================================================

class SensorDataBus:
    """Central hub for all sensor data integration."""
    
    def __init__(self):
        self.sensors: Dict[SensorType, BaseSensor] = {}
        self.last_readings: Dict[SensorType, SensorReading] = {}
        self.read_history: Dict[SensorType, List[SensorReading]] = defaultdict(list)
        self.max_history = 100
        self.quality_estimator = SensorQualityEstimator()
        self.coordinate_manager = CoordinateSystemManager()
    
    def register_sensor(self, sensor: BaseSensor) -> None:
        """Register new sensor."""
        self.sensors[sensor.sensor_type] = sensor
    
    def unregister_sensor(self, sensor_type: SensorType) -> None:
        """Remove sensor from bus."""
        if sensor_type in self.sensors:
            del self.sensors[sensor_type]
    
    def poll_all_sensors(self) -> SensorReadings:
        """Get synchronized data from all sensors."""
        readings = SensorReadings()
        min_timestamp = None
        max_timestamp = None
        
        for sensor_type, sensor in self.sensors.items():
            if not sensor.is_available():
                continue
            
            reading = sensor.read()
            if reading is None:
                continue
            
            readings.add_reading(reading)
            self.last_readings[sensor_type] = reading
            
            # Track history
            self.read_history[sensor_type].append(reading)
            if len(self.read_history[sensor_type]) > self.max_history:
                self.read_history[sensor_type].pop(0)
            
            # Calculate synchronization error
            if min_timestamp is None or reading.timestamp < min_timestamp:
                min_timestamp = reading.timestamp
            if max_timestamp is None or reading.timestamp > max_timestamp:
                max_timestamp = reading.timestamp
        
        if min_timestamp is not None and max_timestamp is not None:
            readings.synchronization_error_ms = (max_timestamp - min_timestamp) * 1000
        
        return readings
    
    def get_sensor(self, sensor_type: SensorType) -> Optional[BaseSensor]:
        """Retrieve specific sensor."""
        return self.sensors.get(sensor_type)
    
    def get_data_quality(self) -> Dict[str, float]:
        """Get quality metrics for each sensor."""
        quality = {}
        for sensor_type in self.sensors:
            if sensor_type in self.last_readings:
                quality[sensor_type.value] = self.last_readings[sensor_type].quality
            else:
                quality[sensor_type.value] = 0.0
        return quality
    
    def get_available_sensors(self) -> List[SensorType]:
        """Get list of available sensors."""
        return [st for st, s in self.sensors.items() if s.is_available()]


# ============================================================================
# Fusion Engine
# ============================================================================

class FusionEngine:
    """Fuses data from multiple sensors for enhanced analysis."""
    
    def __init__(self, sensor_bus: SensorDataBus,
                 coordinate_system: CoordinateSystem = CoordinateSystem.WGS84):
        self.sensor_bus = sensor_bus
        self.coordinate_system = coordinate_system
        self.fusion_history: List[FusedData] = []
        self.max_history = 100
    
    def fuse_readings(self, readings: Optional[SensorReadings] = None) -> FusedData:
        """
        Combine all sensor readings with weighting.
        
        Args:
            readings: SensorReadings to fuse. If None, polls all sensors.
        
        Returns:
            FusedData with combined values
        """
        if readings is None:
            readings = self.sensor_bus.poll_all_sensors()
        
        fused = FusedData(
            timestamp=readings.timestamp,
            location=Location(0.0, 0.0, 0.0, self.coordinate_system)
        )
        
        # Process each available reading
        quality_weights = {}
        
        if readings.has_reading(SensorType.SONAR):
            reading = readings.get_reading(SensorType.SONAR)
            if reading:
                fused.sonar_intensity = float(reading.value)
                quality_weights[SensorType.SONAR.value] = reading.quality
                fused.contributing_sensors.append(SensorType.SONAR)
        
        if readings.has_reading(SensorType.BATHYMETRY):
            reading = readings.get_reading(SensorType.BATHYMETRY)
            if reading:
                if isinstance(reading.value, dict):
                    fused.depth = reading.value.get("depth", 0.0)
                else:
                    fused.depth = float(reading.value)
                quality_weights[SensorType.BATHYMETRY.value] = reading.quality
                fused.contributing_sensors.append(SensorType.BATHYMETRY)
        
        if readings.has_reading(SensorType.CURRENT):
            reading = readings.get_reading(SensorType.CURRENT)
            if reading:
                if isinstance(reading.value, (list, tuple)):
                    fused.current_velocity = tuple(reading.value[:2])
                quality_weights[SensorType.CURRENT.value] = reading.quality
                fused.contributing_sensors.append(SensorType.CURRENT)
        
        if readings.has_reading(SensorType.TEMPERATURE):
            reading = readings.get_reading(SensorType.TEMPERATURE)
            if reading:
                fused.temperature = float(reading.value)
                quality_weights[SensorType.TEMPERATURE.value] = reading.quality
                fused.contributing_sensors.append(SensorType.TEMPERATURE)
        
        if readings.has_reading(SensorType.WIND):
            reading = readings.get_reading(SensorType.WIND)
            if reading:
                if isinstance(reading.value, (list, tuple)):
                    fused.wind_velocity = tuple(reading.value[:2])
                quality_weights[SensorType.WIND.value] = reading.quality
                fused.contributing_sensors.append(SensorType.WIND)
        
        if readings.has_reading(SensorType.WAVE):
            reading = readings.get_reading(SensorType.WAVE)
            if reading:
                fused.wave_height = float(reading.value)
                quality_weights[SensorType.WAVE.value] = reading.quality
                fused.contributing_sensors.append(SensorType.WAVE)
        
        # Calculate overall fusion confidence
        if quality_weights:
            fused.fusion_confidence = np.mean(list(quality_weights.values()))
        else:
            fused.fusion_confidence = 0.0
        
        fused.sensor_quality_weights = quality_weights
        fused.synchronization_quality = max(0.0, 1.0 - readings.synchronization_error_ms / 1000.0)
        
        # Track history
        self.fusion_history.append(fused)
        if len(self.fusion_history) > self.max_history:
            self.fusion_history.pop(0)
        
        return fused
    
    def correlate_anomalies(self, sonar_anomaly_id: str,
                           sonar_location: Location) -> CorrelationResult:
        """
        Find correlations across multiple sensors.
        
        Example: sonar anomaly + bathymetry change + current shift
        """
        result = CorrelationResult(
            primary_anomaly_id=sonar_anomaly_id,
            timestamp=datetime.now().timestamp()
        )
        
        # Get current fused data
        fused = self.fuse_readings()
        
        # Correlate with bathymetry
        if fused.depth is not None:
            result.sonar_characteristics["relative_to_depth"] = \
                (sonar_location.z / fused.depth) if fused.depth > 0 else 0.0
        
        # Correlate with current
        if fused.current_velocity:
            u, v = fused.current_velocity
            current_speed = np.sqrt(u**2 + v**2)
            result.environmental_conditions["current_speed"] = current_speed
        
        # Correlate with temperature
        if fused.temperature is not None:
            result.environmental_conditions["temperature"] = fused.temperature
        
        # Calculate correlation score
        correlation_count = len(result.environmental_conditions) + \
                          len(result.sonar_characteristics)
        result.correlation_score = min(1.0, correlation_count / 5.0)
        result.confidence = fused.fusion_confidence
        
        return result
    
    def get_environmental_context(self, location: Location) -> EnvironmentalContext:
        """Get full environmental picture for analysis."""
        fused = self.fuse_readings()
        
        context = EnvironmentalContext(
            location=location,
            timestamp=fused.timestamp
        )
        
        if fused.depth is not None:
            context.depth = fused.depth
        
        if fused.temperature is not None:
            context.temperature = fused.temperature
        
        if fused.current_velocity:
            u, v = fused.current_velocity
            context.current_strength = np.sqrt(u**2 + v**2)
            context.current_direction = np.degrees(np.arctan2(v, u))
        
        if fused.wind_velocity:
            u, v = fused.wind_velocity
            context.wind_speed = np.sqrt(u**2 + v**2)
            context.wind_direction = np.degrees(np.arctan2(v, u))
        
        if fused.wave_height is not None:
            context.wave_height = fused.wave_height
        
        context.overall_quality = fused.fusion_confidence
        context.sensor_count = len(fused.contributing_sensors)
        
        return context
    
    def get_fusion_history(self, limit: int = 10) -> List[FusedData]:
        """Get recent fusion results."""
        return self.fusion_history[-limit:]
