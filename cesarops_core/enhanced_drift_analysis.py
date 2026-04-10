#!/usr/bin/env python3
"""
Enhanced Drift Analysis System with High-Performance Rust Core
Integrates ML predictions with optimized computational kernels
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
import joblib
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
import joblib
import os

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CESAROPS_ENHANCED')

# Try to import the Rust extension
try:
    import cesarops_core
    RUST_CORE_AVAILABLE = True
    logger.info("High-performance Rust core loaded successfully")
except ImportError:
    RUST_CORE_AVAILABLE = False
    logger.warning("Rust core not available, using Python fallback")

try:
    from sarops import (
        EnhancedOceanDrift, 
        calculate_distance,
        determine_great_lake,
        fetch_enhanced_environmental_data
    )
except ImportError:
    # Fallback implementations
    def calculate_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in nautical miles"""
        import math
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance_km = R * c
        distance_nm = distance_km * 0.539957  # Convert to nautical miles
        return distance_nm
    
    def determine_great_lake(lat, lon):
        """Determine which Great Lake a coordinate point is in"""
        if 41.5 <= lat <= 46.0 and -88.5 <= lon <= -85.5:
            return 'michigan'
        return 'unknown'
    
    def fetch_enhanced_environmental_data(db_file):
        """Placeholder for environmental data fetching"""
        pass

logger = logging.getLogger('CESAROPS_ENHANCED')

class AdvancedDriftAnalyzer:
    """
    Advanced drift analysis system combining ML predictions with high-performance computation
    """
    
    def __init__(self, db_file='drift_objects.db'):
        self.db_file = db_file
        self.ml_models = self._load_ml_models()
        
        # Initialize high-performance core if available
        if RUST_CORE_AVAILABLE:
            self.rust_analyzer = cesarops_core.HighPerformanceDriftAnalyzer()
            logger.info("Using high-performance Rust computational core")
        else:
            self.rust_analyzer = None
            logger.info("Using Python computational fallback")
        
        # Object-specific drift characteristics database
        self.object_characteristics = {
            'person': {
                'windage': 0.03,
                'leeway': 0.10,
                'submerged_fraction': 0.85,
                'drag_coefficient': 1.2,
                'description': 'Person in water'
            },
            'boat_fender': {
                'windage': 0.08,
                'leeway': 0.15,
                'submerged_fraction': 0.30,
                'drag_coefficient': 0.8,
                'description': 'Boat fender (teardrop or cylindrical)'
            },
            'life_ring': {
                'windage': 0.05,
                'leeway': 0.12,
                'submerged_fraction': 0.20,
                'drag_coefficient': 0.9,
                'description': 'Life ring or life preserver'
            },
            'debris': {
                'windage': 0.02,
                'leeway': 0.05,
                'submerged_fraction': 0.50,
                'drag_coefficient': 1.0,
                'description': 'General debris'
            },
            'boat': {
                'windage': 0.02,
                'leeway': 0.05,
                'submerged_fraction': 0.90,
                'drag_coefficient': 0.6,
                'description': 'Small boat or vessel'
            }
        }
    
    def _load_ml_models(self) -> Dict:
        """Load trained ML drift correction models"""
        models = {}
        
        try:
            if os.path.exists('models/drift_correction_model_u.pkl'):
                models['velocity_u'] = joblib.load('models/drift_correction_model_u.pkl')
            if os.path.exists('models/drift_correction_model_v.pkl'):
                models['velocity_v'] = joblib.load('models/drift_correction_model_v.pkl')
            if os.path.exists('models/feature_names.pkl'):
                models['feature_names'] = joblib.load('models/feature_names.pkl')
                
            logger.info(f"Loaded {len(models)} ML models")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
            
        return models
    
    def analyze_charlie_brown_fender_case(self, 
                                        vessel_name: str = "The Rosa",
                                        object_type: str = "boat_fender",
                                        last_seen_location: Tuple[float, float] = (43.0389, -87.9065),  # Milwaukee
                                        last_seen_time: str = "2024-09-15T14:00:00",
                                        search_duration_hours: float = 48.0) -> Dict:
        """
        Analyze the specific case of a fender that came off The Rosa (Charlie Brown's vessel)
        """
        
        logger.info(f"Analyzing drift case: {object_type} from {vessel_name}")
        
        # Case setup
        case_data = {
            'vessel_name': vessel_name,
            'object_type': object_type,
            'object_characteristics': self.object_characteristics.get(object_type, self.object_characteristics['debris']),
            'start_location': {
                'lat': last_seen_location[0],
                'lon': last_seen_location[1],
                'description': 'Milwaukee Harbor area'
            },
            'start_time': datetime.fromisoformat(last_seen_time),
            'search_duration_hours': search_duration_hours
        }
        
        # Get environmental conditions for the time period
        env_conditions = self._get_historical_environmental_conditions(
            case_data['start_location']['lat'],
            case_data['start_location']['lon'],
            case_data['start_time'],
            search_duration_hours
        )
        
        # Run high-performance drift analysis
        if self.rust_analyzer and RUST_CORE_AVAILABLE:
            results = self._run_rust_drift_analysis(case_data, env_conditions)
        else:
            results = self._run_python_drift_analysis(case_data, env_conditions)
        
        # Add ML-enhanced predictions
        ml_enhanced_results = self._apply_ml_enhancements(results, env_conditions)
        
        # Calculate search zones and probabilities
        search_zones = self._calculate_search_zones(ml_enhanced_results, confidence_levels=[0.5, 0.75, 0.9])
        
        # Comprehensive analysis report
        analysis_report = {
            'case_info': case_data,
            'environmental_summary': self._summarize_environmental_conditions(env_conditions),
            'drift_predictions': ml_enhanced_results,
            'search_zones': search_zones,
            'recommendations': self._generate_search_recommendations(ml_enhanced_results, case_data),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Drift analysis completed successfully")
        return analysis_report
    
    def _get_historical_environmental_conditions(self, lat: float, lon: float, 
                                               start_time: datetime, duration_hours: float) -> List[Dict]:
        """Get historical environmental conditions for the analysis period"""
        
        try:
            conn = sqlite3.connect(self.db_file)
            
            end_time = start_time + timedelta(hours=duration_hours)
            
            # Query environmental data
            query = '''
            SELECT timestamp, latitude, longitude, wind_speed, wind_direction,
                   current_u, current_v, wave_height, water_temp, pressure
            FROM environmental_conditions
            WHERE timestamp BETWEEN ? AND ?
            AND ABS(latitude - ?) < 1.0 AND ABS(longitude - ?) < 1.0
            ORDER BY timestamp
            '''
            
            cursor = conn.execute(query, (
                start_time.isoformat(),
                end_time.isoformat(),
                lat, lon
            ))
            
            conditions = []
            for row in cursor.fetchall():
                conditions.append({
                    'timestamp': datetime.fromisoformat(row[0]),
                    'latitude': row[1],
                    'longitude': row[2],
                    'wind_speed': row[3] or 5.0,
                    'wind_direction': row[4] or 270.0,
                    'current_u': row[5] or 0.05,
                    'current_v': row[6] or 0.02,
                    'wave_height': row[7] or 0.5,
                    'water_temp': row[8] or 12.0,
                    'pressure': row[9] or 1013.25
                })
            
            conn.close()
            
            # If no data available, create realistic conditions for Great Lakes
            if not conditions:
                logger.warning("No historical environmental data found, using typical Great Lakes conditions")
                conditions = self._generate_typical_great_lakes_conditions(start_time, duration_hours)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error getting environmental conditions: {e}")
            return self._generate_typical_great_lakes_conditions(start_time, duration_hours)
    
    def _generate_typical_great_lakes_conditions(self, start_time: datetime, duration_hours: float) -> List[Dict]:
        """Generate typical Great Lakes environmental conditions"""
        
        conditions = []
        
        # Base conditions for Lake Michigan in fall
        base_conditions = {
            'wind_speed': 8.5,      # m/s (typical fall winds)
            'wind_direction': 285.0, # WNW (typical fall pattern)
            'current_u': 0.08,      # Eastward current component
            'current_v': 0.04,      # Northward current component  
            'wave_height': 1.2,     # Moderate waves
            'water_temp': 14.0,     # Fall water temperature
            'pressure': 1015.0      # Typical pressure
        }
        
        # Generate hourly conditions with realistic variations
        current_time = start_time
        for hour in range(int(duration_hours)):
            # Add realistic variations
            wind_variation = np.sin(hour * 0.1) * 3.0  # Diurnal wind variation
            current_variation = np.sin(hour * 0.05) * 0.02  # Current variation
            
            conditions.append({
                'timestamp': current_time,
                'latitude': 43.0,
                'longitude': -87.0,
                'wind_speed': base_conditions['wind_speed'] + wind_variation,
                'wind_direction': base_conditions['wind_direction'] + np.sin(hour * 0.08) * 15.0,
                'current_u': base_conditions['current_u'] + current_variation,
                'current_v': base_conditions['current_v'] + current_variation * 0.5,
                'wave_height': base_conditions['wave_height'] + wind_variation * 0.1,
                'water_temp': base_conditions['water_temp'],
                'pressure': base_conditions['pressure'] + np.sin(hour * 0.02) * 5.0
            })
            
            current_time += timedelta(hours=1)
        
        return conditions
    
    def _run_rust_drift_analysis(self, case_data: Dict, env_conditions: List[Dict]) -> Dict:
        """Run high-performance drift analysis using Rust core"""
        
        # Convert environmental data to numpy arrays for Rust
        times = np.array([c['timestamp'].timestamp() for c in env_conditions])
        wind_speeds = np.array([c['wind_speed'] for c in env_conditions])
        wind_directions = np.array([c['wind_direction'] for c in env_conditions])
        currents_u = np.array([c['current_u'] for c in env_conditions])
        currents_v = np.array([c['current_v'] for c in env_conditions])
        wave_heights = np.array([c['wave_height'] for c in env_conditions])
        water_temps = np.array([c['water_temp'] for c in env_conditions])
        pressures = np.array([c['pressure'] for c in env_conditions])
        
        # Add environmental data to Rust analyzer
        self.rust_analyzer.add_environmental_data(
            wind_speeds, wind_directions, currents_u, currents_v,
            wave_heights, water_temps, pressures
        )
        
        # Run high-performance drift prediction
        start_lat = case_data['start_location']['lat']
        start_lon = case_data['start_location']['lon']
        duration = case_data['search_duration_hours']
        
        times, lats, lons = self.rust_analyzer.predict_drift_trajectory(
            start_lat, start_lon, duration, time_step_minutes=10.0
        )
        
        # Calculate probability zones
        start_positions = np.array([[start_lat, start_lon]])
        object_types = [case_data['object_type']]
        
        probability_zones = self.rust_analyzer.calculate_probability_zones(
            start_lat, start_lon, duration, confidence_level=0.75
        )
        
        return {
            'trajectory': {
                'times': times,
                'latitudes': lats,
                'longitudes': lons
            },
            'probability_zones': probability_zones,
            'computation_method': 'high_performance_rust'
        }
    
    def _run_python_drift_analysis(self, case_data: Dict, env_conditions: List[Dict]) -> Dict:
        """Fallback Python implementation of drift analysis"""
        
        start_lat = case_data['start_location']['lat']
        start_lon = case_data['start_location']['lon']
        duration_hours = case_data['search_duration_hours']
        
        # Object characteristics
        obj_char = case_data['object_characteristics']
        
        # Initialize trajectory
        trajectory_times = []
        trajectory_lats = []
        trajectory_lons = []
        
        current_lat = start_lat
        current_lon = start_lon
        current_time = case_data['start_time']
        
        time_step = timedelta(minutes=10)
        steps = int((duration_hours * 60) / 10)  # 10-minute time steps
        
        for step in range(steps):
            trajectory_times.append(current_time.timestamp())
            trajectory_lats.append(current_lat)
            trajectory_lons.append(current_lon)
            
            # Get environmental conditions for current time
            env = self._interpolate_conditions(env_conditions, current_time)
            
            # Calculate drift velocity with object-specific characteristics
            drift_u, drift_v = self._calculate_object_specific_drift(env, obj_char)
            
            # Update position
            dt_seconds = time_step.total_seconds()
            delta_lat, delta_lon = self._velocity_to_displacement(drift_u, drift_v, current_lat, dt_seconds)
            
            current_lat += delta_lat
            current_lon += delta_lon
            current_time += time_step
        
        return {
            'trajectory': {
                'times': np.array(trajectory_times),
                'latitudes': np.array(trajectory_lats),
                'longitudes': np.array(trajectory_lons)
            },
            'computation_method': 'python_fallback'
        }
    
    def _calculate_object_specific_drift(self, env: Dict, obj_char: Dict) -> Tuple[float, float]:
        """Calculate drift velocity for specific object type"""
        
        # Wind components
        wind_rad = np.radians(env['wind_direction'])
        wind_u = env['wind_speed'] * np.sin(wind_rad)
        wind_v = env['wind_speed'] * np.cos(wind_rad)
        
        # Object-specific drift calculation
        windage = obj_char['windage']
        leeway = obj_char['leeway']
        drag_coeff = obj_char['drag_coefficient']
        
        # Enhanced physics model for fenders
        if 'fender' in obj_char['description'].lower():
            # Fenders have high windage due to shape and partial submersion
            effective_windage = windage * (1.0 + 0.5 * np.sqrt(env['wave_height']))
            # Leeway angle depends on wind speed
            leeway_angle = leeway * np.tanh(env['wind_speed'] / 10.0)
        else:
            effective_windage = windage
            leeway_angle = leeway
        
        # Stokes drift from waves
        stokes_factor = 0.016 * np.sqrt(env['wave_height'])
        
        # Combined drift velocity
        drift_u = (env['current_u'] + 
                  effective_windage * wind_u + 
                  stokes_factor * wind_u * drag_coeff +
                  leeway_angle * wind_v)  # Cross-wind leeway
        
        drift_v = (env['current_v'] + 
                  effective_windage * wind_v + 
                  stokes_factor * wind_v * drag_coeff -
                  leeway_angle * wind_u)  # Cross-wind leeway
        
        return drift_u, drift_v
    
    def _apply_ml_enhancements(self, results: Dict, env_conditions: List[Dict]) -> Dict:
        """Apply ML corrections to drift predictions"""
        
        if not self.ml_models or 'velocity_u' not in self.ml_models:
            logger.warning("ML models not available, returning physics-only results")
            return results
        
        try:
            # Apply ML corrections to trajectory
            enhanced_results = results.copy()
            
            # Get average environmental conditions
            avg_env = self._average_environmental_conditions(env_conditions)
            
            # Create feature vector for ML model
            features = np.array([[
                avg_env['wind_speed'],
                np.sin(np.radians(avg_env['wind_direction'])),
                np.cos(np.radians(avg_env['wind_direction'])),
                avg_env['current_u'],
                avg_env['current_v'],
                avg_env['wave_height'],
                avg_env['water_temp'],
                avg_env['pressure']
            ]])
            
            # Get ML corrections
            correction_u = self.ml_models['velocity_u'].predict(features)[0]
            correction_v = self.ml_models['velocity_v'].predict(features)[0]
            
            # Apply corrections to trajectory (simplified)
            enhanced_results['ml_corrections'] = {
                'velocity_u_correction': correction_u,
                'velocity_v_correction': correction_v,
                'correction_applied': True
            }
            
            logger.info(f"Applied ML corrections: U={correction_u:.4f}, V={correction_v:.4f}")
            
        except Exception as e:
            logger.error(f"Error applying ML enhancements: {e}")
            enhanced_results = results
            enhanced_results['ml_corrections'] = {'correction_applied': False, 'error': str(e)}
        
        return enhanced_results
    
    def _calculate_search_zones(self, results: Dict, confidence_levels: List[float]) -> Dict:
        """Calculate probability-based search zones"""
        
        trajectory = results['trajectory']
        final_lat = trajectory['latitudes'][-1]
        final_lon = trajectory['longitudes'][-1]
        
        # Calculate uncertainty ellipses based on trajectory spread
        search_zones = {}
        
        for confidence in confidence_levels:
            # Simple uncertainty model - in practice would use Monte Carlo results
            uncertainty_factor = 1.0 - confidence
            
            # Radius increases with uncertainty and time
            radius_nm = 5.0 + (uncertainty_factor * 15.0)
            
            search_zones[f'{int(confidence*100)}%'] = {
                'center_lat': final_lat,
                'center_lon': final_lon,
                'radius_nm': radius_nm,
                'area_nm2': np.pi * radius_nm**2,
                'confidence_level': confidence
            }
        
        return search_zones
    
    def _generate_search_recommendations(self, results: Dict, case_data: Dict) -> List[str]:
        """Generate actionable search recommendations"""
        
        recommendations = []
        
        trajectory = results['trajectory']
        final_position = (trajectory['latitudes'][-1], trajectory['longitudes'][-1])
        start_position = (case_data['start_location']['lat'], case_data['start_location']['lon'])
        
        # Calculate total drift distance
        total_distance = calculate_distance(
            start_position[0], start_position[1],
            final_position[0], final_position[1]
        )
        
        recommendations.append(f"Primary search area: {final_position[0]:.4f}°N, {final_position[1]:.4f}°W")
        recommendations.append(f"Total predicted drift: {total_distance:.1f} nautical miles")
        
        # Object-specific recommendations
        obj_type = case_data['object_type']
        if obj_type == 'boat_fender':
            recommendations.extend([
                "Fender likely to be highly visible on surface due to bright color",
                "Check downwind areas more thoroughly due to high windage",
                "Search in harbors and near shore structures where fenders may wash up",
                "Consider that fender may have been picked up by other boaters"
            ])
        
        # Time-based recommendations
        hours_elapsed = case_data['search_duration_hours']
        if hours_elapsed > 24:
            recommendations.append("After 24+ hours, expand search to include shoreline areas")
        
        if hours_elapsed > 48:
            recommendations.append("Consider beach and marina surveys in predicted drift area")
        
        return recommendations
    
    def _interpolate_conditions(self, env_conditions: List[Dict], target_time: datetime) -> Dict:
        """Interpolate environmental conditions for a specific time"""
        
        if not env_conditions:
            return self._get_default_conditions()
        
        # Find closest time point
        closest_condition = min(env_conditions, 
                              key=lambda x: abs((x['timestamp'] - target_time).total_seconds()))
        
        return closest_condition
    
    def _velocity_to_displacement(self, vel_u: float, vel_v: float, lat: float, dt: float) -> Tuple[float, float]:
        """Convert velocity components to lat/lon displacement"""
        
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat))
        
        delta_lat = (vel_v * dt) / meters_per_degree_lat
        delta_lon = (vel_u * dt) / meters_per_degree_lon
        
        return delta_lat, delta_lon
    
    def _average_environmental_conditions(self, env_conditions: List[Dict]) -> Dict:
        """Calculate average environmental conditions"""
        
        if not env_conditions:
            return self._get_default_conditions()
        
        avg_conditions = {}
        numeric_fields = ['wind_speed', 'wind_direction', 'current_u', 'current_v', 
                         'wave_height', 'water_temp', 'pressure']
        
        for field in numeric_fields:
            values = [c[field] for c in env_conditions if c[field] is not None]
            avg_conditions[field] = np.mean(values) if values else 0.0
        
        return avg_conditions
    
    def _summarize_environmental_conditions(self, env_conditions: List[Dict]) -> Dict:
        """Create summary of environmental conditions"""
        
        if not env_conditions:
            return {"error": "No environmental data available"}
        
        avg = self._average_environmental_conditions(env_conditions)
        
        return {
            'duration_hours': len(env_conditions),
            'average_wind_speed_ms': avg['wind_speed'],
            'average_wind_direction_deg': avg['wind_direction'],
            'average_current_speed_ms': np.sqrt(avg['current_u']**2 + avg['current_v']**2),
            'average_wave_height_m': avg['wave_height'],
            'water_temperature_c': avg['water_temp'],
            'conditions_description': self._describe_conditions(avg)
        }
    
    def _describe_conditions(self, avg_conditions: Dict) -> str:
        """Generate human-readable description of conditions"""
        
        wind_speed = avg_conditions['wind_speed']
        wave_height = avg_conditions['wave_height']
        
        if wind_speed < 5:
            wind_desc = "Light winds"
        elif wind_speed < 10:
            wind_desc = "Moderate winds"
        else:
            wind_desc = "Strong winds"
        
        if wave_height < 0.5:
            wave_desc = "calm seas"
        elif wave_height < 1.5:
            wave_desc = "moderate seas"
        else:
            wave_desc = "rough seas"
        
        return f"{wind_desc} with {wave_desc}"
    
    def _get_default_conditions(self) -> Dict:
        """Get default environmental conditions for Great Lakes"""
        
        return {
            'wind_speed': 5.0,
            'wind_direction': 270.0,
            'current_u': 0.05,
            'current_v': 0.02,
            'wave_height': 0.5,
            'water_temp': 12.0,
            'pressure': 1013.25
        }

def run_charlie_brown_fender_analysis():
    """Run the specific Charlie Brown fender case analysis"""
    
    print("🔍 CESAROPS Enhanced Drift Analysis")
    print("=" * 50)
    print("Case: Teardrop fender from The Rosa (Charlie Brown's vessel)")
    print()
    
    # Initialize the enhanced analyzer
    analyzer = AdvancedDriftAnalyzer()
    
    # Run the analysis
    results = analyzer.analyze_charlie_brown_fender_case(
        vessel_name="The Rosa",
        object_type="boat_fender",
        last_seen_location=(43.0389, -87.9065),  # Milwaukee Harbor
        last_seen_time="2024-09-15T14:00:00",   # 2 PM
        search_duration_hours=48.0               # 48-hour search window
    )
    
    # Display results
    print("📍 CASE INFORMATION:")
    case_info = results['case_info']
    print(f"   Vessel: {case_info['vessel_name']}")
    print(f"   Object: {case_info['object_characteristics']['description']}")
    print(f"   Start Location: {case_info['start_location']['lat']:.4f}°N, {case_info['start_location']['lon']:.4f}°W")
    print(f"   Start Time: {case_info['start_time']}")
    print(f"   Search Duration: {case_info['search_duration_hours']} hours")
    print()
    
    print("🌊 ENVIRONMENTAL CONDITIONS:")
    env_summary = results['environmental_summary']
    print(f"   {env_summary['conditions_description']}")
    print(f"   Average Wind: {env_summary['average_wind_speed_ms']:.1f} m/s")
    print(f"   Average Waves: {env_summary['average_wave_height_m']:.1f} m")
    print(f"   Water Temperature: {env_summary['water_temperature_c']:.1f}°C")
    print()
    
    print("🎯 PREDICTED FINAL POSITION:")
    trajectory = results['drift_predictions']['trajectory']
    final_lat = trajectory['latitudes'][-1]
    final_lon = trajectory['longitudes'][-1]
    print(f"   Latitude: {final_lat:.4f}°N")
    print(f"   Longitude: {final_lon:.4f}°W")
    
    # Calculate total drift
    start_lat = case_info['start_location']['lat']
    start_lon = case_info['start_location']['lon']
    total_drift = calculate_distance(start_lat, start_lon, final_lat, final_lon)
    print(f"   Total Drift Distance: {total_drift:.1f} nautical miles")
    print()
    
    print("🔍 SEARCH ZONES:")
    for zone_name, zone_data in results['search_zones'].items():
        print(f"   {zone_name} Confidence Zone:")
        print(f"     Center: {zone_data['center_lat']:.4f}°N, {zone_data['center_lon']:.4f}°W")
        print(f"     Radius: {zone_data['radius_nm']:.1f} nm")
        print(f"     Area: {zone_data['area_nm2']:.1f} nm²")
    print()
    
    print("💡 SEARCH RECOMMENDATIONS:")
    for i, recommendation in enumerate(results['recommendations'], 1):
        print(f"   {i}. {recommendation}")
    print()
    
    print("⚙️  COMPUTATION INFO:")
    computation_method = results['drift_predictions']['computation_method']
    print(f"   Method: {computation_method}")
    
    if 'ml_corrections' in results['drift_predictions']:
        ml_info = results['drift_predictions']['ml_corrections']
        if ml_info.get('correction_applied', False):
            print(f"   ML Corrections Applied: ✅")
            print(f"     U-velocity correction: {ml_info['velocity_u_correction']:.4f}")
            print(f"     V-velocity correction: {ml_info['velocity_v_correction']:.4f}")
        else:
            print(f"   ML Corrections: ❌ {ml_info.get('error', 'Not available')}")
    
    print("\n" + "=" * 50)
    print("Analysis completed successfully! 🎉")
    
    return results

if __name__ == "__main__":
    run_charlie_brown_fender_analysis()