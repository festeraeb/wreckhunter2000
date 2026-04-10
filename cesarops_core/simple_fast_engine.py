#!/usr/bin/env python3
"""
Fast Drift Engine - Simplified Version
High-performance drift analysis without JIT complications
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional
import sqlite3
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger('FAST_DRIFT_ENGINE')

class SimpleFastDriftEngine:
    """
    Simplified high-performance drift analysis
    Optimized for immediate deployment without JIT issues
    """
    
    def __init__(self, use_multiprocessing=True, n_workers=None):
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers or min(8, (os.cpu_count() or 1))
        
        logger.info(f"SimpleFastDriftEngine initialized with {self.n_workers} workers")
    
    def simulate_drift_ensemble(self, 
                               release_lat: float, 
                               release_lon: float,
                               release_time: datetime,
                               duration_hours: float,
                               environmental_data: List[Dict],
                               n_particles: int = 1000,
                               time_step_minutes: int = 10,
                               object_specs: Optional[Dict] = None) -> Dict:
        """
        High-performance ensemble drift simulation
        """
        
        logger.info(f"Starting ensemble simulation: {n_particles} particles, {duration_hours}h")
        
        # Convert environmental data to arrays
        env_df = pd.DataFrame(environmental_data)
        if 'timestamp' in env_df.columns:
            env_df['time_offset'] = env_df['timestamp'].apply(
                lambda x: (x - release_time).total_seconds() if isinstance(x, datetime) else float(x)
            )
        
        # Initialize particles
        particles = []
        for i in range(n_particles):
            particles.append({
                'id': i,
                'lat': release_lat + np.random.normal(0, 0.001),
                'lon': release_lon + np.random.normal(0, 0.001),
                'trajectory': [(release_lat, release_lon, 0.0)]
            })
        
        # Object specifications
        if object_specs is None:
            object_specs = self._get_default_object_specs()
        
        # Time parameters
        time_step_seconds = time_step_minutes * 60
        n_steps = int(duration_hours * 3600 / time_step_seconds)
        
        # Run simulation
        for step in range(n_steps):
            current_time = step * time_step_seconds
            
            # Update all particles
            for particle in particles:
                # Get environmental conditions
                env_conditions = self._interpolate_conditions(env_df, current_time)
                
                # Calculate drift
                dlat, dlon = self._calculate_drift_step(
                    particle['lat'], particle['lon'], 
                    env_conditions, time_step_seconds, object_specs
                )
                
                # Update position
                particle['lat'] += dlat
                particle['lon'] += dlon
                particle['trajectory'].append((particle['lat'], particle['lon'], current_time))
        
        # Calculate statistics
        final_positions = [(p['lat'], p['lon']) for p in particles]
        statistics = self._calculate_statistics(final_positions, release_time, duration_hours)
        
        return {
            'particles': particles,
            'statistics': statistics,
            'simulation_info': {
                'n_particles': n_particles,
                'duration_hours': duration_hours,
                'time_step_minutes': time_step_minutes,
                'n_steps': n_steps,
                'object_specs': object_specs
            }
        }
    
    def _interpolate_conditions(self, env_df: pd.DataFrame, target_time: float) -> Dict:
        """Interpolate environmental conditions for specific time"""
        
        if 'time_offset' not in env_df.columns:
            # Use first record if no time data
            return env_df.iloc[0].to_dict()
        
        # Find closest records
        time_diffs = np.abs(env_df['time_offset'] - target_time)
        closest_idx = time_diffs.idxmin()
        
        return env_df.loc[closest_idx].to_dict()
    
    def _calculate_drift_step(self, lat: float, lon: float, 
                             env_conditions: Dict, dt: float, 
                             object_specs: Dict) -> Tuple[float, float]:
        """Calculate single drift step"""
        
        # Extract environmental conditions
        wind_speed = env_conditions.get('wind_speed', 0.0)
        wind_direction = np.radians(env_conditions.get('wind_direction', 0.0))
        current_u = env_conditions.get('current_u', 0.0)
        current_v = env_conditions.get('current_v', 0.0)
        wave_height = env_conditions.get('wave_height', 0.0)
        
        # Wind components
        wind_u = wind_speed * np.sin(wind_direction)
        wind_v = wind_speed * np.cos(wind_direction)
        
        # Object-specific parameters
        windage = object_specs.get('windage', 0.08)
        leeway = object_specs.get('leeway', 0.15)
        stokes_factor = object_specs.get('stokes_factor', 0.02) * np.sqrt(max(0.1, wave_height))
        
        # Calculate drift velocity components
        velocity_u = (current_u + 
                     windage * wind_u + 
                     stokes_factor * wind_u +
                     leeway * wind_v)
        
        velocity_v = (current_v + 
                     windage * wind_v + 
                     stokes_factor * wind_v -
                     leeway * wind_u)
        
        # Convert to displacement
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat))
        
        dlat = (velocity_v * dt) / meters_per_degree_lat
        dlon = (velocity_u * dt) / meters_per_degree_lon
        
        return dlat, dlon
    
    def _calculate_statistics(self, final_positions: List[Tuple], 
                            release_time: datetime, duration_hours: float) -> Dict:
        """Calculate ensemble statistics"""
        
        lats = [pos[0] for pos in final_positions]
        lons = [pos[1] for pos in final_positions]
        
        # Center of mass
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Spread statistics
        lat_std = np.std(lats)
        lon_std = np.std(lons)
        
        # Confidence intervals
        lat_conf_95 = 1.96 * lat_std
        lon_conf_95 = 1.96 * lon_std
        
        # Distance statistics
        distances = []
        for lat, lon in final_positions:
            dist = self._calculate_distance(lats[0], lons[0], lat, lon)
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'center_position': {'lat': center_lat, 'lon': center_lon},
            'spread': {'lat_std': lat_std, 'lon_std': lon_std},
            'confidence_95': {'lat_range': lat_conf_95, 'lon_range': lon_conf_95},
            'distance_stats': {
                'mean_nm': np.mean(distances),
                'std_nm': np.std(distances),
                'max_nm': np.max(distances),
                'percentile_95_nm': np.percentile(distances, 95)
            },
            'particle_count': len(final_positions),
            'final_time': release_time + timedelta(hours=duration_hours)
        }
    
    def _get_default_object_specs(self) -> Dict:
        """Get default object specifications for fender"""
        return {
            'windage': 0.08,        # High windage for orange teardrop fender
            'leeway': 0.15,         # Cross-wind drift
            'stokes_factor': 0.02,  # Wave-induced drift
            'drag_coefficient': 0.8
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in nautical miles"""
        
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance_km = R * c
        distance_nm = distance_km * 0.539957
        
        return distance_nm

def test_simple_engine():
    """Test the simplified fast engine"""
    
    print("🚀 Testing Simplified Fast Drift Engine")
    print("=" * 45)
    
    # Initialize engine
    engine = SimpleFastDriftEngine(use_multiprocessing=False, n_workers=1)
    
    # Create test environmental data
    release_time = datetime(2025, 8, 22, 15, 0, 0)
    duration = 48  # hours
    
    env_data = []
    for hour in range(duration + 12):
        time_point = release_time + timedelta(hours=hour)
        env_data.append({
            'timestamp': time_point,
            'wind_speed': 8.0 + 3.0 * np.sin(hour * 0.1),
            'wind_direction': 240.0 + 20.0 * np.sin(hour * 0.05),
            'current_u': 0.05 + 0.02 * np.cos(hour * 0.08),
            'current_v': 0.03 + 0.01 * np.sin(hour * 0.12),
            'wave_height': 1.0 + 0.5 * np.sin(hour * 0.15),
            'water_temp': 20.0,
            'pressure': 1015.0
        })
    
    # Run simulation
    start_time = datetime.now()
    
    result = engine.simulate_drift_ensemble(
        release_lat=43.0389,   # Milwaukee area
        release_lon=-87.9065,
        release_time=release_time,
        duration_hours=duration,
        environmental_data=env_data,
        n_particles=100,  # Reduced for testing
        time_step_minutes=30,
        object_specs={
            'windage': 0.08,    # Orange teardrop fender
            'leeway': 0.15,     # High cross-wind drift
            'stokes_factor': 0.02
        }
    )
    
    end_time = datetime.now()
    simulation_time = (end_time - start_time).total_seconds()
    
    # Display results
    stats = result['statistics']
    center = stats['center_position']
    
    print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
    print(f"Particles: {stats['particle_count']}")
    print(f"Final center: {center['lat']:.4f}°N, {center['lon']:.4f}°W")
    print(f"Spread: ±{stats['spread']['lat_std']*60:.1f}' lat, ±{stats['spread']['lon_std']*60:.1f}' lon")
    print(f"95% confidence: ±{stats['confidence_95']['lat_range']*60:.1f}' lat")
    print(f"Max drift distance: {stats['distance_stats']['max_nm']:.1f} nm")
    
    # Calculate total drift from start
    start_lat, start_lon = 43.0389, -87.9065
    total_drift = engine._calculate_distance(start_lat, start_lon, center['lat'], center['lon'])
    print(f"Total drift distance: {total_drift:.1f} nm")
    
    return result

if __name__ == "__main__":
    test_simple_engine()