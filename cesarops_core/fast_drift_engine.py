#!/usr/bin/env python3
"""
High-Performance Drift Analysis Engine (Pure Python)
Optimized alternative to Rust engine for immediate deployment
"""

import numpy as np
import pandas as pd
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional
import sqlite3
import logging
from datetime import datetime, timedelta
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='numba')

logger = logging.getLogger('FAST_DRIFT_ENGINE')

class FastDriftEngine:
    """
    High-performance drift analysis using optimized Python
    Provides near-Rust performance using Numba JIT compilation
    """
    
    def __init__(self, use_multiprocessing=True, n_workers=None):
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers or min(8, (os.cpu_count() or 1))
        
        logger.info(f"FastDriftEngine initialized with {self.n_workers} workers")
        
        # Pre-compile JIT functions
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Pre-compile JIT functions for optimal performance"""
        logger.info("Warming up JIT compiler...")
        
        # Dummy data for compilation
        dummy_particles = np.random.rand(100, 3)  # lat, lon, time
        dummy_env = np.random.rand(50, 8)  # environmental data
        dummy_times = np.linspace(0, 86400, 50)
        
        # Trigger compilation
        try:
            self._jit_drift_step(dummy_particles, dummy_env, dummy_times, 0, 600)
            self._jit_interpolate_conditions(dummy_env, dummy_times, 3600)
            logger.info("✅ JIT compilation completed")
        except Exception as e:
            logger.warning(f"JIT warmup failed: {e}")
    
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
        
        Args:
            release_lat: Release latitude
            release_lon: Release longitude  
            release_time: Release time
            duration_hours: Simulation duration
            environmental_data: Environmental conditions
            n_particles: Number of particles
            time_step_minutes: Time step in minutes
            object_specs: Object-specific parameters
            
        Returns:
            Simulation results with statistics
        """
        
        logger.info(f"Starting ensemble simulation: {n_particles} particles, {duration_hours}h")
        
        # Convert environmental data to optimized arrays
        env_array, time_array = self._prepare_environmental_arrays(environmental_data, release_time)
        
        # Initialize particle array [lat, lon, time_index]
        particles = np.zeros((n_particles, 3), dtype=np.float64)
        particles[:, 0] = release_lat + np.random.normal(0, 0.001, n_particles)  # Small initial spread
        particles[:, 1] = release_lon + np.random.normal(0, 0.001, n_particles)
        particles[:, 2] = 0  # Start at time index 0
        
        # Object specifications
        if object_specs is None:
            object_specs = self._get_default_object_specs()
        
        # Time parameters
        time_step_seconds = time_step_minutes * 60
        n_steps = int(duration_hours * 3600 / time_step_seconds)
        
        # Run simulation
        if self.use_multiprocessing and n_particles > 500:
            result = self._run_parallel_simulation(particles, env_array, time_array, 
                                                 n_steps, time_step_seconds, object_specs)
        else:
            result = self._run_sequential_simulation(particles, env_array, time_array,
                                                   n_steps, time_step_seconds, object_specs)
        
        # Calculate statistics
        statistics = self._calculate_ensemble_statistics(result, release_time, time_step_minutes)
        
        return {
            'particles': result,
            'statistics': statistics,
            'simulation_info': {
                'n_particles': n_particles,
                'duration_hours': duration_hours,
                'time_step_minutes': time_step_minutes,
                'n_steps': n_steps,
                'object_specs': object_specs
            }
        }
    
    def _prepare_environmental_arrays(self, env_data: List[Dict], release_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Convert environmental data to optimized numpy arrays"""
        
        n_records = len(env_data)
        
        # Environmental array: [time, wind_speed, wind_dir, current_u, current_v, wave_height, temp, pressure]
        env_array = np.zeros((n_records, 8), dtype=np.float64)
        time_array = np.zeros(n_records, dtype=np.float64)
        
        for i, record in enumerate(env_data):
            if isinstance(record['timestamp'], datetime):
                time_offset = (record['timestamp'] - release_time).total_seconds()
            else:
                time_offset = float(record['timestamp'])
            
            time_array[i] = time_offset
            env_array[i, 0] = record.get('wind_speed', 0.0)
            env_array[i, 1] = record.get('wind_direction', 0.0)
            env_array[i, 2] = record.get('current_u', 0.0)
            env_array[i, 3] = record.get('current_v', 0.0)
            env_array[i, 4] = record.get('wave_height', 0.0)
            env_array[i, 5] = record.get('water_temp', 15.0)
            env_array[i, 6] = record.get('pressure', 1013.0)
            env_array[i, 7] = 0.0  # Reserved for future use
        
        return env_array, time_array
    
    def _run_parallel_simulation(self, particles: np.ndarray, env_array: np.ndarray, 
                               time_array: np.ndarray, n_steps: int, 
                               time_step_seconds: float, object_specs: Dict) -> np.ndarray:
        """Run simulation using parallel processing"""
        
        # Split particles among workers
        chunk_size = len(particles) // self.n_workers
        particle_chunks = [particles[i:i+chunk_size] for i in range(0, len(particles), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for chunk in particle_chunks:
                future = executor.submit(
                    self._simulate_particle_chunk,
                    chunk, env_array, time_array, n_steps, time_step_seconds, object_specs
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                results.append(future.result())
        
        # Combine results
        return np.vstack(results) if results else particles
    
    def _run_sequential_simulation(self, particles: np.ndarray, env_array: np.ndarray,
                                 time_array: np.ndarray, n_steps: int,
                                 time_step_seconds: float, object_specs: Dict) -> np.ndarray:
        """Run simulation sequentially (single-threaded)"""
        
        return self._simulate_particle_chunk(particles, env_array, time_array, 
                                           n_steps, time_step_seconds, object_specs)
    
    def _simulate_particle_chunk(self, particles: np.ndarray, env_array: np.ndarray,
                               time_array: np.ndarray, n_steps: int,
                               time_step_seconds: float, object_specs: Dict) -> np.ndarray:
        """Simulate a chunk of particles"""
        
        # Create trajectory array: [particle_id, step, lat, lon, time]
        n_particles = len(particles)
        trajectories = np.zeros((n_particles, n_steps + 1, 5), dtype=np.float64)
        
        # Initialize trajectories
        for i in range(n_particles):
            trajectories[i, 0, 0] = i  # particle ID
            trajectories[i, 0, 1] = 0  # step
            trajectories[i, 0, 2] = particles[i, 0]  # lat
            trajectories[i, 0, 3] = particles[i, 1]  # lon
            trajectories[i, 0, 4] = 0  # time
        
        # Run simulation steps
        for step in range(n_steps):
            current_time = step * time_step_seconds
            
            for particle_id in range(n_particles):
                current_lat = trajectories[particle_id, step, 2]
                current_lon = trajectories[particle_id, step, 3]
                
                # Get environmental conditions
                env_conditions = _jit_interpolate_conditions(env_array, time_array, current_time)
                
                # Calculate drift
                dlat, dlon = _jit_drift_step(
                    np.array([[current_lat, current_lon, current_time]]),
                    env_array, time_array, current_time, time_step_seconds
                )[0]
                
                # Update position
                new_lat = current_lat + dlat
                new_lon = current_lon + dlon
                
                # Store trajectory point
                trajectories[particle_id, step + 1, 0] = particle_id
                trajectories[particle_id, step + 1, 1] = step + 1
                trajectories[particle_id, step + 1, 2] = new_lat
                trajectories[particle_id, step + 1, 3] = new_lon
                trajectories[particle_id, step + 1, 4] = current_time + time_step_seconds
        
        return trajectories
    
    @staticmethod
    @njit(parallel=True)
    def _jit_drift_step(particles: np.ndarray, env_array: np.ndarray, 
                       time_array: np.ndarray, current_time: float, 
                       dt: float) -> np.ndarray:
        """JIT-compiled drift calculation for maximum performance"""
        
        n_particles = particles.shape[0]
        displacements = np.zeros((n_particles, 2), dtype=np.float64)
        
        for i in prange(n_particles):
            lat = particles[i, 0]
            lon = particles[i, 1]
            
            # Interpolate environmental conditions
            env_conditions = _jit_interpolate_conditions(env_array, time_array, current_time)
            
            # Extract conditions
            wind_speed = env_conditions[0]
            wind_direction = env_conditions[1] * np.pi / 180.0  # Convert to radians
            current_u = env_conditions[2]
            current_v = env_conditions[3]
            wave_height = env_conditions[4]
            
            # Wind components
            wind_u = wind_speed * np.sin(wind_direction)
            wind_v = wind_speed * np.cos(wind_direction)
            
            # Object-specific drift (using typical fender parameters)
            windage = 0.08  # High windage for fender
            leeway = 0.15   # Cross-wind drift
            stokes_factor = 0.02 * np.sqrt(max(0.1, wave_height))
            
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
            meters_per_degree_lon = 111320.0 * np.cos(lat * np.pi / 180.0)
            
            dlat = (velocity_v * dt) / meters_per_degree_lat
            dlon = (velocity_u * dt) / meters_per_degree_lon
            
            displacements[i, 0] = dlat
            displacements[i, 1] = dlon
        
        return displacements
    
    @staticmethod
    @njit
    def _jit_interpolate_conditions(env_array: np.ndarray, time_array: np.ndarray, 
                                   target_time: float) -> np.ndarray:
        """JIT-compiled environmental data interpolation"""
        
        n_records = len(time_array)
        conditions = np.zeros(8, dtype=np.float64)
        
        if target_time <= time_array[0]:
            conditions[:] = env_array[0, :]
            return conditions
        
        if target_time >= time_array[-1]:
            conditions[:] = env_array[-1, :]
            return conditions
        
        # Find surrounding time indices
        for i in range(n_records - 1):
            if time_array[i] <= target_time <= time_array[i + 1]:
                # Linear interpolation
                t1, t2 = time_array[i], time_array[i + 1]
                weight = (target_time - t1) / (t2 - t1)
                
                for j in range(8):
                    conditions[j] = env_array[i, j] + weight * (env_array[i + 1, j] - env_array[i, j])
                
                return conditions
        
        # Fallback
        conditions[:] = env_array[0, :]
        return conditions
    
    def _calculate_ensemble_statistics(self, trajectories: np.ndarray, 
                                     release_time: datetime,
                                     time_step_minutes: int) -> Dict:
        """Calculate ensemble statistics from simulation results"""
        
        n_particles = trajectories.shape[0]
        n_steps = trajectories.shape[1]
        
        # Final positions
        final_lats = trajectories[:, -1, 2]
        final_lons = trajectories[:, -1, 3]
        
        # Center of mass
        center_lat = np.mean(final_lats)
        center_lon = np.mean(final_lons)
        
        # Spread statistics
        lat_std = np.std(final_lats)
        lon_std = np.std(final_lons)
        
        # Confidence ellipse (95%)
        confidence_factor = 1.96  # 95% confidence
        lat_conf = confidence_factor * lat_std
        lon_conf = confidence_factor * lon_std
        
        # Distance statistics
        distances = []
        for i in range(n_particles):
            dist = self._calculate_distance(
                final_lats[0], final_lons[0],  # Use first particle as reference
                final_lats[i], final_lons[i]
            )
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'center_position': {'lat': center_lat, 'lon': center_lon},
            'spread': {'lat_std': lat_std, 'lon_std': lon_std},
            'confidence_95': {'lat_range': lat_conf, 'lon_range': lon_conf},
            'distance_stats': {
                'mean_nm': np.mean(distances),
                'std_nm': np.std(distances),
                'max_nm': np.max(distances),
                'percentile_95_nm': np.percentile(distances, 95)
            },
            'particle_count': n_particles,
            'final_time': release_time + timedelta(minutes=time_step_minutes * (n_steps - 1))
        }
    
    def _get_default_object_specs(self) -> Dict:
        """Get default object specifications"""
        return {
            'windage': 0.08,
            'leeway': 0.15,
            'stokes_factor': 0.02,
            'drag_coefficient': 0.8,
            'submerged_fraction': 0.25
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance_km = R * c
        distance_nm = distance_km * 0.539957  # Convert to nautical miles
        
        return distance_nm

# Import os for CPU count
import os

def test_fast_engine():
    """Test the fast drift engine"""
    
    print("🚀 Testing Fast Drift Engine")
    print("=" * 40)
    
    # Initialize engine
    engine = FastDriftEngine(use_multiprocessing=True, n_workers=4)
    
    # Create test environmental data
    release_time = datetime(2025, 8, 22, 15, 0, 0)
    duration = 48  # hours
    
    env_data = []
    for hour in range(duration + 12):  # Extra data for interpolation
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
        n_particles=500,
        time_step_minutes=15
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
    
    return result

if __name__ == "__main__":
    test_fast_engine()