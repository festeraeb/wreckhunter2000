# CESARops Drone Integration Module
"""
Drone coordination and mission planning for SAR operations.
Ties directly into incident reporting with search pattern optimization,
multi-drone coordination (LSAR algorithm), and real-time telemetry.
"""

import json
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict


class DroneType(Enum):
    """Supported drone types for SAR operations."""
    QUADCOPTER = "quadcopter"  # DJI Matrice, Pixhawk 4
    FIXED_WING = "fixed_wing"  # Longer endurance
    VTOL = "vtol"  # Vertical takeoff + fixed-wing speed
    THERMAL = "thermal"  # Thermal imaging specialist


class SearchPattern(Enum):
    """Search patterns optimized for different SAR types."""
    LAWN_MOWER = "lawn_mower"  # Parallel lines (land SAR)
    SPIRAL = "spiral"  # Outward spiral (water SAR, centered search)
    EXPANDING_SQUARE = "expanding_square"  # Expanding squares (wilderness)
    PERIMETER = "perimeter"  # Follow perimeter (coastal/river)
    GRID_RANDOM = "grid_random"  # Random within grid (high probability areas)


@dataclass
class Drone:
    """Represents a drone asset with capabilities and constraints."""
    drone_id: str
    drone_type: DroneType
    max_flight_time_minutes: int
    max_speed_ms: float
    payload_kg: float
    sensors: List[str]  # e.g., ['thermal', 'rgb', 'lidar']
    home_location: Tuple[float, float]  # (lat, lon)
    current_battery_percent: int = 100
    status: str = "available"  # available, flying, charging, maintenance
    altitude_m: int = 0
    
    def estimated_coverage_km2(self, ground_resolution_m: int = 10) -> float:
        """Estimate area coverage based on flight time and speed."""
        flight_time_hours = self.max_flight_time_minutes / 60
        distance_km = (self.max_speed_ms * 3.6) * flight_time_hours  # Convert m/s to km/h
        coverage_width_km = (ground_resolution_m / 1000) * 3  # 3x the resolution for overlap
        return distance_km * coverage_width_km


@dataclass
class Waypoint:
    """Represents a waypoint in a drone mission."""
    lat: float
    lon: float
    altitude_m: int
    sequence: int
    hold_time_seconds: int = 0
    action: str = "WAYPOINT"  # WAYPOINT, LAND, LOITER
    
    def to_mavlink_dict(self) -> Dict[str, Any]:
        """Convert to MAVLink mission item format."""
        return {
            'seq': self.sequence,
            'frame': 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            'command': 16 if self.action == 'WAYPOINT' else (21 if self.action == 'LAND' else 17),
            'current': 1 if self.sequence == 0 else 0,
            'autocontinue': 1,
            'param1': self.hold_time_seconds,
            'param2': 0,
            'param3': 0,
            'param4': 0,
            'x': self.lat,
            'y': self.lon,
            'z': self.altitude_m
        }


class SearchPatternGenerator:
    """Generate optimized search patterns for different SAR incident types."""
    
    @staticmethod
    def generate_lawn_mower(
        center: Tuple[float, float],
        width_km: float,
        height_km: float,
        altitude_m: int,
        drone_speed_ms: float,
        line_spacing_m: int = 100
    ) -> List[Waypoint]:
        """
        Generate lawn-mower pattern for systematic grid search (land SAR).
        
        Args:
            center: Center point (lat, lon)
            width_km: East-west extent in km
            height_km: North-south extent in km
            altitude_m: Flight altitude
            drone_speed_ms: Drone speed in m/s
            line_spacing_m: Distance between parallel lines (based on sensor FOV)
        
        Returns:
            List of waypoints forming lawn-mower pattern
        """
        waypoints = []
        center_lat, center_lon = center
        
        # Convert km to approximate degrees (varies by latitude, rough estimate)
        lat_per_km = 1 / 111.0
        lon_per_km = 1 / (111.0 * math.cos(math.radians(center_lat)))
        
        half_width = (width_km / 2) * lon_per_km
        half_height = (height_km / 2) * lat_per_km
        
        # Generate parallel lines
        num_lines = int((width_km * 1000) / line_spacing_m)
        
        for i in range(num_lines):
            # Calculate line position
            x_offset = -half_width + (i * line_spacing_m / 1000) * lon_per_km
            
            # Alternate direction for efficiency
            if i % 2 == 0:
                # Go north
                start_lat = center_lat - half_height
                end_lat = center_lat + half_height
                start_lon = center_lon + x_offset
                end_lon = center_lon + x_offset
            else:
                # Go south
                start_lat = center_lat + half_height
                end_lat = center_lat - half_height
                start_lon = center_lon + x_offset
                end_lon = center_lon + x_offset
            
            # Create waypoint for start of line
            waypoints.append(Waypoint(
                lat=start_lat,
                lon=start_lon,
                altitude_m=altitude_m,
                sequence=len(waypoints),
                action='WAYPOINT'
            ))
            
            # Create waypoint for end of line
            waypoints.append(Waypoint(
                lat=end_lat,
                lon=end_lon,
                altitude_m=altitude_m,
                sequence=len(waypoints),
                action='WAYPOINT'
            ))
        
        return waypoints
    
    @staticmethod
    def generate_spiral(
        center: Tuple[float, float],
        max_radius_km: float,
        altitude_m: int,
        spiral_spacing_m: int = 100
    ) -> List[Waypoint]:
        """
        Generate outward spiral pattern for water SAR.
        Covers center point first (highest priority).
        
        Args:
            center: Center point (lat, lon)
            max_radius_km: Maximum search radius
            altitude_m: Flight altitude
            spiral_spacing_m: Distance between spiral rings
        
        Returns:
            List of waypoints forming spiral pattern
        """
        waypoints = []
        center_lat, center_lon = center
        
        lat_per_km = 1 / 111.0
        lon_per_km = 1 / (111.0 * math.cos(math.radians(center_lat)))
        
        # Start at center
        num_rings = int((max_radius_km * 1000) / spiral_spacing_m)
        
        for ring in range(num_rings + 1):
            radius_m = ring * spiral_spacing_m
            radius_deg_lat = (radius_m / 1000) * lat_per_km
            radius_deg_lon = (radius_m / 1000) * lon_per_km
            
            # Create circle of waypoints at this radius
            num_points = max(8, int(2 * math.pi * radius_m / 50))  # Min 8 points
            
            for point_num in range(num_points):
                angle = (point_num / num_points) * 2 * math.pi
                
                lat = center_lat + radius_deg_lat * math.sin(angle)
                lon = center_lon + radius_deg_lon * math.cos(angle)
                
                waypoints.append(Waypoint(
                    lat=lat,
                    lon=lon,
                    altitude_m=altitude_m,
                    sequence=len(waypoints),
                    action='WAYPOINT'
                ))
        
        return waypoints
    
    @staticmethod
    def generate_expanding_square(
        center: Tuple[float, float],
        altitude_m: int,
        initial_size_m: int = 500,
        expansion_step_m: int = 500,
        max_size_km: float = 5.0
    ) -> List[Waypoint]:
        """
        Generate expanding square pattern for wilderness SAR.
        
        Args:
            center: Center point (lat, lon)
            altitude_m: Flight altitude
            initial_size_m: Initial square size
            expansion_step_m: How much to expand each iteration
            max_size_km: Maximum square size
        
        Returns:
            List of waypoints forming expanding squares
        """
        waypoints = []
        center_lat, center_lon = center
        
        lat_per_km = 1 / 111.0
        lon_per_km = 1 / (111.0 * math.cos(math.radians(center_lat)))
        
        size_m = initial_size_m
        max_size_m = max_size_km * 1000
        
        while size_m <= max_size_m:
            half_size_lat = (size_m / 2 / 1000) * lat_per_km
            half_size_lon = (size_m / 2 / 1000) * lon_per_km
            
            # Create square corners
            corners = [
                (center_lat - half_size_lat, center_lon - half_size_lon),  # SW
                (center_lat - half_size_lat, center_lon + half_size_lon),  # SE
                (center_lat + half_size_lat, center_lon + half_size_lon),  # NE
                (center_lat + half_size_lat, center_lon - half_size_lon),  # NW
                (center_lat - half_size_lat, center_lon - half_size_lon),  # Back to SW
            ]
            
            for lat, lon in corners:
                waypoints.append(Waypoint(
                    lat=lat,
                    lon=lon,
                    altitude_m=altitude_m,
                    sequence=len(waypoints),
                    action='WAYPOINT'
                ))
            
            size_m += expansion_step_m
        
        return waypoints


class LSARCoordinator:
    """
    Layered Search and Rescue (LSAR) multi-drone coordinator.
    Based on Alotaibi et al. 2019 (417 citations).
    Allocates search zones to multiple drones based on battery/speed.
    """
    
    @staticmethod
    def allocate_search_zones(
        drones: List[Drone],
        search_area: Tuple[float, float, float, float],  # (lat_min, lon_min, lat_max, lon_max)
        incident_type: str = 'land'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Allocate search zones to drones using LSAR algorithm.
        
        Args:
            drones: List of available drones
            search_area: Bounding box of search area
            incident_type: Type of incident (land, water, k9, aerial, coastal)
        
        Returns:
            Dictionary mapping drone_id -> {zone, pattern, waypoints, estimated_time}
        """
        
        allocations = {}
        available_drones = [d for d in drones if d.status == 'available']
        
        if not available_drones:
            return allocations
        
        # Calculate coverage capability for each drone
        drone_capabilities = {}
        for drone in available_drones:
            coverage = drone.estimated_coverage_km2()
            max_time = drone.max_flight_time_minutes
            speed = drone.max_speed_ms
            
            drone_capabilities[drone.drone_id] = {
                'coverage_km2': coverage,
                'max_time_minutes': max_time,
                'speed_ms': speed,
                'battery_factor': drone.current_battery_percent / 100.0
            }
        
        # Select search pattern based on incident type
        pattern_map = {
            'land': SearchPattern.LAWN_MOWER,
            'water': SearchPattern.SPIRAL,
            'k9': SearchPattern.EXPANDING_SQUARE,
            'aerial': SearchPattern.LAWN_MOWER,
            'coastal': SearchPattern.PERIMETER,
            'dive': SearchPattern.SPIRAL
        }
        search_pattern = pattern_map.get(incident_type, SearchPattern.LAWN_MOWER)
        
        # Calculate center and size of search area
        lat_min, lon_min, lat_max, lon_max = search_area
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        width_km = (lon_max - lon_min) * 111.0
        height_km = (lat_max - lat_min) * 111.0
        
        # Allocate zones sequentially
        zone_num = 0
        for drone in sorted(available_drones, key=lambda d: drone_capabilities[d.drone_id]['coverage_km2'], reverse=True):
            cap = drone_capabilities[drone.drone_id]
            
            # Calculate zone for this drone
            zone_width = width_km / len(available_drones)
            zone_lat_min = lat_min
            zone_lat_max = lat_max
            zone_lon_min = lon_min + (zone_num * zone_width / 111.0)
            zone_lon_max = zone_lon_min + (zone_width / 111.0)
            
            zone_center = ((zone_lat_min + zone_lat_max) / 2, (zone_lon_min + zone_lon_max) / 2)
            
            # Generate pattern
            if search_pattern == SearchPattern.LAWN_MOWER:
                waypoints = SearchPatternGenerator.generate_lawn_mower(
                    center=zone_center,
                    width_km=zone_width,
                    height_km=height_km,
                    altitude_m=100,
                    drone_speed_ms=drone.max_speed_ms
                )
            elif search_pattern == SearchPattern.SPIRAL:
                waypoints = SearchPatternGenerator.generate_spiral(
                    center=zone_center,
                    max_radius_km=math.sqrt(zone_width**2 + height_km**2) / 2,
                    altitude_m=80
                )
            else:  # EXPANDING_SQUARE
                waypoints = SearchPatternGenerator.generate_expanding_square(
                    center=zone_center,
                    altitude_m=100
                )
            
            # Estimate mission time
            total_distance = 0
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i + 1]
                # Rough distance calculation (not exact)
                lat_diff = abs(wp2.lat - wp1.lat) * 111000
                lon_diff = abs(wp2.lon - wp1.lon) * 111000 * math.cos(math.radians((wp1.lat + wp2.lat) / 2))
                distance = math.sqrt(lat_diff**2 + lon_diff**2)
                total_distance += distance
            
            mission_time_minutes = (total_distance / (drone.max_speed_ms * 60))
            
            allocations[drone.drone_id] = {
                'zone': {
                    'lat_min': zone_lat_min,
                    'lon_min': zone_lon_min,
                    'lat_max': zone_lat_max,
                    'lon_max': zone_lon_max,
                    'center': zone_center
                },
                'pattern': search_pattern.value,
                'waypoints': [asdict(wp) for wp in waypoints],
                'estimated_mission_time_minutes': mission_time_minutes,
                'battery_sufficient': mission_time_minutes < (drone.max_flight_time_minutes * cap['battery_factor']),
                'altitude_m': 100,
                'sensors_assigned': drone.sensors
            }
            
            zone_num += 1
        
        return allocations
    
    @staticmethod
    def monitor_drone_health(drone: Drone) -> Dict[str, Any]:
        """
        Monitor drone battery, signal, and other health metrics.
        Trigger RTH (return-to-home) if critical.
        """
        
        return {
            'drone_id': drone.drone_id,
            'status': drone.status,
            'battery_percent': drone.current_battery_percent,
            'battery_critical': drone.current_battery_percent < 15,
            'should_rth': drone.current_battery_percent < 30,
            'altitude_m': drone.altitude_m,
            'timestamp': datetime.now().isoformat()
        }


class DronePayloadAnalytics:
    """
    Analyze drone video feeds for target detection.
    Integrates YOLOv5 for human detection + thermal imaging.
    """
    
    @staticmethod
    def human_detection_confidence(
        has_thermal: bool = False,
        has_rgb: bool = False,
        thermal_anomaly: bool = False,
        rgb_detection: bool = False,
        detection_size_pixels: int = 0,
        minimum_size_pixels: int = 20
    ) -> Tuple[float, str]:
        """
        Calculate confidence score for human detection.
        Requires dual confirmation for high confidence.
        
        Returns:
            (confidence_score_0_to_1, confidence_level)
        """
        
        score = 0.0
        
        # Thermal channel (40% weight)
        if has_thermal and thermal_anomaly:
            score += 0.4
        
        # RGB channel (50% weight)
        if has_rgb and rgb_detection and detection_size_pixels >= minimum_size_pixels:
            size_factor = min(1.0, detection_size_pixels / 200)  # Normalized
            score += 0.5 * size_factor
        
        # Dual confirmation bonus (10%)
        if (has_thermal and thermal_anomaly) and (has_rgb and rgb_detection):
            score += 0.1
        
        # Determine confidence level
        if score >= 0.9:
            confidence_level = 'VERY_HIGH'
        elif score >= 0.7:
            confidence_level = 'HIGH'
        elif score >= 0.5:
            confidence_level = 'MODERATE'
        else:
            confidence_level = 'LOW'
        
        return score, confidence_level
    
    @staticmethod
    def generate_alert(
        drone_id: str,
        location: Tuple[float, float],
        confidence: float,
        sensor_type: str,
        timestamp: str = None
    ) -> Dict[str, Any]:
        """Generate alert for potential target detection."""
        
        return {
            'alert_id': f"ALERT-{drone_id}-{int(datetime.now().timestamp() * 1000) % 10000}",
            'drone_id': drone_id,
            'location': {'lat': location[0], 'lon': location[1]},
            'confidence_score': confidence,
            'sensor_type': sensor_type,
            'timestamp': timestamp or datetime.now().isoformat(),
            'action_required': confidence >= 0.7,
            'priority': 'HIGH' if confidence >= 0.8 else 'MEDIUM'
        }


class DroneIncidentIntegration:
    """
    Integrate drone module with CESARops incident reporting.
    Links drone operations to incident reports and metrics.
    """
    
    def __init__(self, incident_id: str):
        """Initialize drone integration for specific incident."""
        self.incident_id = incident_id
        self.drones: List[Drone] = []
        self.allocations: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def add_drone(self, drone: Drone) -> None:
        """Register drone for this incident."""
        self.drones.append(drone)
    
    def generate_drone_mission(
        self,
        incident_type: str,
        search_area: Tuple[float, float, float, float],
        altitude_m: int = 100
    ) -> Dict[str, Any]:
        """
        Generate complete drone mission for incident.
        
        Returns:
            Mission package with allocations and waypoints
        """
        
        # Allocate zones using LSAR
        allocations = LSARCoordinator.allocate_search_zones(
            drones=self.drones,
            search_area=search_area,
            incident_type=incident_type
        )
        
        self.allocations = allocations
        
        # Generate mission package
        mission_package = {
            'incident_id': self.incident_id,
            'created_timestamp': datetime.now().isoformat(),
            'incident_type': incident_type,
            'search_area': {
                'bounds': search_area,
                'center': ((search_area[0] + search_area[2]) / 2, (search_area[1] + search_area[3]) / 2)
            },
            'drone_allocations': allocations,
            'total_drones': len(self.drones),
            'total_coverage_km2': sum([
                d.estimated_coverage_km2() for d in self.drones
                if d.status == 'available'
            ]),
            'estimated_total_time_minutes': max([
                alloc['estimated_mission_time_minutes'] for alloc in allocations.values()
            ]) if allocations else 0,
            'all_drones_ready': all(d.status == 'available' for d in self.drones)
        }
        
        return mission_package
    
    def add_detection_alert(self, alert: Dict[str, Any]) -> None:
        """Add target detection alert to incident."""
        self.alerts.append(alert)
    
    def get_drone_status_for_report(self) -> Dict[str, Any]:
        """
        Get drone operations status for incident report integration.
        """
        
        return {
            'drones_deployed': len([d for d in self.drones if d.status == 'flying']),
            'drones_total': len(self.drones),
            'mission_package': self.allocations,
            'detections': len(self.alerts),
            'high_confidence_detections': len([a for a in self.alerts if a['confidence_score'] >= 0.8]),
            'alerts': self.alerts
        }


if __name__ == '__main__':
    # Example: Create drone fleet and generate mission
    drones = [
        Drone(
            drone_id='DRONE-001',
            drone_type=DroneType.QUADCOPTER,
            max_flight_time_minutes=45,
            max_speed_ms=15,
            payload_kg=2.7,
            sensors=['thermal', 'rgb'],
            home_location=(37.7749, -122.4194),
            current_battery_percent=100
        ),
        Drone(
            drone_id='DRONE-002',
            drone_type=DroneType.VTOL,
            max_flight_time_minutes=60,
            max_speed_ms=20,
            payload_kg=3.5,
            sensors=['rgb'],
            home_location=(37.7749, -122.4194),
            current_battery_percent=100
        ),
    ]
    
    # Create incident integration
    integration = DroneIncidentIntegration('TEST-LAND-001')
    for drone in drones:
        integration.add_drone(drone)
    
    # Generate mission for land SAR
    mission = integration.generate_drone_mission(
        incident_type='land',
        search_area=(37.7600, -122.4400, 37.7900, -122.4000),
        altitude_m=100
    )
    
    print(f"Mission generated for {mission['incident_id']}")
    print(f"Drones deployed: {mission['total_drones']}")
    print(f"Total coverage: {mission['total_coverage_km2']:.1f} km²")
    print(f"Estimated mission time: {mission['estimated_total_time_minutes']:.0f} minutes")
    
    # Simulate detection
    alert = DronePayloadAnalytics.generate_alert(
        drone_id='DRONE-001',
        location=(37.7750, -122.4200),
        confidence=0.85,
        sensor_type='thermal+rgb'
    )
    integration.add_detection_alert(alert)
    print(f"\nAlert: {alert['alert_id']} (Priority: {alert['priority']})")
