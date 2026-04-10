# CESARops Drone-Reporting Integration Bridge
"""
Integration layer connecting drone operations with incident reporting.
Updates incident reports with drone mission data, detections, and metrics.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from drone_module import (
    DroneIncidentIntegration, Drone, DroneType, 
    DronePayloadAnalytics, LSARCoordinator
)


class DroneReportingBridge:
    """
    Bridge between drone operations and incident reporting.
    Updates incident reports with drone-specific data.
    """
    
    def __init__(self, reporting_api):
        """
        Initialize bridge with reference to reporting API.
        
        Args:
            reporting_api: Instance of CESAROpsReportingAPI
        """
        self.reporting_api = reporting_api
        self.drone_integrations: Dict[str, DroneIncidentIntegration] = {}
    
    def enhance_incident_report_with_drones(
        self,
        incident_analysis: Dict[str, Any],
        drone_integration: DroneIncidentIntegration
    ) -> Dict[str, Any]:
        """
        Enhance incident analysis with drone operations data.
        
        Args:
            incident_analysis: Incident analysis from reporting API
            drone_integration: Drone operations for incident
        
        Returns:
            Enhanced incident analysis with drone data
        """
        
        # Get drone status
        drone_status = drone_integration.get_drone_status_for_report()
        
        # Add drone operations section to report
        incident_analysis['report']['drone_operations'] = {
            'drones_deployed': drone_status['drones_deployed'],
            'drones_total': drone_status['drones_total'],
            'mission_allocations': drone_integration.allocations,
            'detections': drone_status['detections'],
            'high_confidence_detections': drone_status['high_confidence_detections'],
            'detection_alerts': drone_status['alerts']
        }
        
        # Update search coverage metrics with drone data
        if drone_integration.allocations:
            estimated_coverage = sum([
                alloc.get('estimated_mission_time_minutes', 0) * 0.25  # Rough estimation
                for alloc in drone_integration.allocations.values()
            ])
            incident_analysis['metrics']['drone_estimated_coverage_km2'] = estimated_coverage
        
        # Update findings with drone detections
        for alert in drone_status['alerts']:
            if alert['confidence_score'] >= 0.7:  # Only high-confidence
                incident_analysis['report']['findings']['findings'].append({
                    'type': 'drone_detection',
                    'location': alert['location'],
                    'confidence': alert['confidence_score'],
                    'timestamp': alert['timestamp'],
                    'drone_id': alert['drone_id'],
                    'sensor': alert['sensor_type']
                })
        
        # Add drone metrics to statistics
        incident_analysis['metrics']['drone_alerts'] = drone_status['detections']
        incident_analysis['metrics']['confirmed_targets'] = drone_status['high_confidence_detections']
        
        return incident_analysis
    
    def generate_drone_mission_from_incident(
        self,
        incident_data: Dict[str, Any],
        available_drones: List[Drone]
    ) -> Dict[str, Any]:
        """
        Generate drone mission directly from incident report data.
        
        Args:
            incident_data: Incident data from incident report
            available_drones: List of available drones
        
        Returns:
            Complete drone mission package
        """
        
        incident_id = incident_data.get('incident_id', 'UNKNOWN')
        incident_type = incident_data.get('incident_type', 'land')
        location = incident_data.get('location', (0, 0))
        
        # Define search area around incident location
        lat, lon = location
        search_radius_km = 2.0  # Start with 2km radius
        
        # Convert to bounding box
        lat_delta = search_radius_km / 111.0
        lon_delta = search_radius_km / (111.0 * 1.0)  # Rough estimate
        
        search_area = (
            lat - lat_delta,
            lon - lon_delta,
            lat + lat_delta,
            lon + lon_delta
        )
        
        # Create drone integration
        integration = DroneIncidentIntegration(incident_id)
        for drone in available_drones:
            integration.add_drone(drone)
        
        # Generate mission
        mission = integration.generate_drone_mission(
            incident_type=incident_type,
            search_area=search_area,
            altitude_m=self._get_altitude_for_incident_type(incident_type)
        )
        
        # Store for later reference
        self.drone_integrations[incident_id] = integration
        
        return mission
    
    def _get_altitude_for_incident_type(self, incident_type: str) -> int:
        """Get recommended altitude for incident type."""
        altitude_map = {
            'land': 100,  # 100m for terrain detail
            'water': 80,  # Lower for water clarity
            'k9': 100,  # Standard
            'aerial': 300,  # Higher to avoid aircraft
            'coastal': 100,  # Standard
            'dive': 50,  # Very low for water clarity
            'equine': 100  # Standard
        }
        return altitude_map.get(incident_type, 100)
    
    def update_incident_with_drone_detections(
        self,
        incident_id: str,
        detections: List[Dict[str, Any]]
    ) -> None:
        """
        Update incident with new drone detections.
        
        Args:
            incident_id: Incident identifier
            detections: List of detection alerts
        """
        
        if incident_id not in self.drone_integrations:
            return
        
        integration = self.drone_integrations[incident_id]
        
        for detection in detections:
            alert = DronePayloadAnalytics.generate_alert(
                drone_id=detection.get('drone_id', 'UNKNOWN'),
                location=detection.get('location', (0, 0)),
                confidence=detection.get('confidence', 0.5),
                sensor_type=detection.get('sensor', 'unknown')
            )
            integration.add_detection_alert(alert)
    
    def generate_drone_operations_dashboard(
        self,
        incident_id: str
    ) -> Dict[str, Any]:
        """
        Generate dashboard data for drone operations monitoring.
        
        Args:
            incident_id: Incident identifier
        
        Returns:
            Dashboard data for visualization
        """
        
        if incident_id not in self.drone_integrations:
            return {'error': 'No drone operations for incident'}
        
        integration = self.drone_integrations[incident_id]
        status = integration.get_drone_status_for_report()
        
        return {
            'incident_id': incident_id,
            'drones': {
                'total': status['drones_total'],
                'deployed': status['drones_deployed'],
                'available': status['drones_total'] - status['drones_deployed']
            },
            'mission': {
                'zones_assigned': len(status['mission_package']),
                'total_coverage_estimate': sum([
                    alloc.get('estimated_mission_time_minutes', 0) * 0.25
                    for alloc in status['mission_package'].values()
                ]) if status['mission_package'] else 0
            },
            'detections': {
                'total_alerts': status['detections'],
                'high_confidence': status['high_confidence_detections'],
                'latest_alerts': status['alerts'][-5:]  # Last 5
            },
            'map_data': {
                'drone_positions': self._get_drone_positions(integration),
                'detection_markers': self._get_detection_markers(status['alerts']),
                'search_zones': self._extract_search_zones(status['mission_package'])
            }
        }
    
    def _get_drone_positions(self, integration: DroneIncidentIntegration) -> List[Dict[str, Any]]:
        """Extract current drone positions."""
        positions = []
        for drone in integration.drones:
            if drone.status == 'flying' and drone.altitude_m > 0:
                # In production, get actual GPS position from telemetry
                positions.append({
                    'drone_id': drone.drone_id,
                    'lat': drone.home_location[0],
                    'lon': drone.home_location[1],
                    'altitude_m': drone.altitude_m,
                    'battery_percent': drone.current_battery_percent,
                    'status': drone.status
                })
        return positions
    
    def _get_detection_markers(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert alerts to map markers."""
        markers = []
        for alert in alerts:
            markers.append({
                'lat': alert['location']['lat'],
                'lon': alert['location']['lon'],
                'confidence': alert['confidence_score'],
                'priority': alert['priority'],
                'drone_id': alert['drone_id'],
                'timestamp': alert['timestamp']
            })
        return markers
    
    def _extract_search_zones(self, allocations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search zones for map display."""
        zones = []
        for drone_id, alloc in allocations.items():
            zone = alloc.get('zone', {})
            zones.append({
                'drone_id': drone_id,
                'bounds': [
                    [zone.get('lat_min'), zone.get('lon_min')],
                    [zone.get('lat_max'), zone.get('lon_min')],
                    [zone.get('lat_max'), zone.get('lon_max')],
                    [zone.get('lat_min'), zone.get('lon_max')]
                ],
                'center': zone.get('center'),
                'pattern': alloc.get('pattern')
            })
        return zones


class ReportWithDrones:
    """
    Wrapper that extends incident reports with drone operations.
    Used by reporting API to provide complete incident picture.
    """
    
    def __init__(self, reporting_api, bridge: DroneReportingBridge):
        """Initialize report wrapper."""
        self.reporting_api = reporting_api
        self.bridge = bridge
    
    def create_complete_incident_analysis(
        self,
        incident_data: Dict[str, Any],
        available_drones: List[Drone] = None
    ) -> Dict[str, Any]:
        """
        Create complete incident analysis with drone integration.
        
        Args:
            incident_data: Incident data
            available_drones: Optional list of drones to deploy
        
        Returns:
            Complete analysis with drone operations
        """
        
        # Generate base incident analysis
        analysis = self.reporting_api.create_incident_analysis(incident_data)
        
        # Add drone operations if drones available
        if available_drones:
            drone_mission = self.bridge.generate_drone_mission_from_incident(
                incident_data,
                available_drones
            )
            
            # Create drone integration
            incident_id = incident_data.get('incident_id')
            integration = self.bridge.drone_integrations.get(incident_id)
            
            if integration:
                # Enhance report with drone data
                analysis = self.bridge.enhance_incident_report_with_drones(
                    analysis,
                    integration
                )
                
                # Add drone mission to report
                analysis['report']['drone_mission'] = drone_mission
        
        return analysis
    
    def export_with_drones(
        self,
        analysis: Dict[str, Any],
        formats: List[str] = None,
        output_dir: str = 'reports/generated'
    ) -> Dict[str, str]:
        """Export incident analysis with drone data included."""
        
        # Use standard export from reporting API
        return self.reporting_api.export_incident_analysis(
            analysis,
            formats=formats,
            output_dir=output_dir
        )


# Example usage
if __name__ == '__main__':
    # Optional: Import reporting API if available
    reporting_api = None
    bridge = None
    try:
        from reporting_api import CESAROpsReportingAPI
        reporting_api = CESAROpsReportingAPI()
        bridge = DroneReportingBridge(reporting_api)
    except ImportError:
        print("Note: reporting_api module not found. Using drone module in standalone mode.")
    
    # Create sample drones
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
        )
    ]
    
    # Create incident data
    incident_data = {
        'incident_id': 'TEST-LAND-001',
        'incident_type': 'land',
        'location': (37.7749, -122.4194),
        'subject_profile': {'name': 'Test Subject', 'age': 35},
        'search_resources': {'personnel': 10, 'equipment': 3, 'duration_hours': 8}
    }
    
    # Create complete analysis with drones
    if reporting_api and bridge:
        complete_analysis = ReportWithDrones(reporting_api, bridge).create_complete_incident_analysis(
            incident_data,
            drones
        )
        print(f"Complete analysis created for {incident_data['incident_id']}")
        print(f"Drones integrated: {len(complete_analysis.get('report', {}).get('drone_operations', {}).get('detection_alerts', []))} operations")
    else:
        # Standalone drone test
        print(f"Testing drone operations for {incident_data['incident_id']}")
        integration = DroneIncidentIntegration(incident_data['incident_id'])
        for drone in drones:
            integration.add_drone(drone)
        mission = integration.generate_drone_mission('land', (37.7600, -122.4400, 37.7900, -122.4000))
        print(f"Drone mission generated: {mission['total_drones']} drones, {mission['estimated_total_time_minutes']:.0f} min estimated")
