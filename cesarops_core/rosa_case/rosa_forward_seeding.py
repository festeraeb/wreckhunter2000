#!/usr/bin/env python3
"""
ROSA CASE - FORWARD SEEDING PATTERN ANALYSIS
Test multiple release points 5+ miles off Milwaukee Harbor to find optimal release
scenarios that result in South Haven shoreline impacts by August 28, 4 PM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_fast_engine import SimpleFastDriftEngine
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import matplotlib.pyplot as plt
from geopy.distance import geodesic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ROSA_FORWARD_SEEDING')

class RosaForwardSeeding:
    """
    Forward seeding analysis to find optimal Rosa release scenarios
    """
    
    def __init__(self):
        self.engine = SimpleFastDriftEngine()
        
        # Rosa recovery information
        self.target_recovery = {
            'position': {'lat': 42.40, 'lon': -86.28},  # South Haven area
            'time': datetime(2025, 8, 28, 16, 0, 0),    # August 28, 4 PM
            'shoreline_tolerance': 20.0  # Within 20 nm of shoreline
        }
        
        # Milwaukee Harbor reference point
        self.milwaukee_harbor = {'lat': 43.0389, 'lon': -87.9065}
        
        # User's optimal parameters from GLOS analysis
        self.optimal_params = {
            'windage': 0.06,      # A = 0.06 (user specified)
            'leeway': 0.06,       # Same as windage for fender
            'stokes_factor': 0.0045  # Stokes = 0.0045 (user specified)
        }
        
        os.makedirs('outputs/rosa_forward_seeding', exist_ok=True)
    
    def generate_release_grid(self):
        """Generate release points 5+ miles straight off Milwaukee Harbor"""
        
        logger.info("🌊 Generating release grid 5+ miles off Milwaukee Harbor...")
        
        release_points = []
        
        # Distances from harbor (5-15 miles offshore)
        distances_nm = [5, 7, 10, 12, 15]
        
        # Directions from harbor (roughly straight out into lake)
        # Milwaukee Harbor faces roughly east, so 60° to 120° covers the offshore area
        bearings = [70, 80, 90, 100, 110]
        
        for distance_nm in distances_nm:
            for bearing in bearings:
                # Calculate position using distance and bearing from Milwaukee Harbor
                lat, lon = self.calculate_position_from_bearing(
                    self.milwaukee_harbor['lat'], 
                    self.milwaukee_harbor['lon'],
                    bearing, 
                    distance_nm
                )
                
                release_points.append({
                    'lat': lat,
                    'lon': lon,
                    'distance_nm': distance_nm,
                    'bearing': bearing,
                    'name': f"MKE_{distance_nm}nm_{bearing}deg"
                })
        
        logger.info(f"📍 Generated {len(release_points)} release points")
        return release_points
    
    def calculate_position_from_bearing(self, start_lat, start_lon, bearing, distance_nm):
        """Calculate lat/lon from starting point, bearing, and distance"""
        from geopy import distance
        from geopy.distance import geodesic
        
        # Convert nautical miles to kilometers
        distance_km = distance_nm * 1.852
        
        # Calculate destination point
        start_point = (start_lat, start_lon)
        destination = geodesic(kilometers=distance_km).destination(start_point, bearing)
        
        return destination.latitude, destination.longitude
    
    def generate_release_times(self):
        """Generate release times from 3 PM to midnight on August 22nd (date written on fender)"""
        
        logger.info("⏰ Generating release time scenarios for August 22nd only...")
        logger.info("📝 Date was written on the suicide fender - August 22, 2025")
        
        release_times = []
        
        # Only August 22, 2025 - the date written on the fender
        release_date = datetime(2025, 8, 22)
        
        # Release times: 3 PM to midnight (user specified)
        release_hours = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # 3 PM to 11 PM
        
        for hour in release_hours:
            release_time = release_date.replace(hour=hour, minute=0, second=0)
            
            # Calculate drift duration to target recovery time
            duration = self.target_recovery['time'] - release_time
            duration_hours = duration.total_seconds() / 3600
            
            if duration_hours > 0:  # Valid forward drift
                release_times.append({
                    'time': release_time,
                    'duration_hours': duration_hours,
                    'date_str': release_time.strftime('%b %d %H:%M'),
                    'hours_before': duration_hours
                })
        
        logger.info(f"⏰ Generated {len(release_times)} release time scenarios for August 22nd")
        logger.info(f"📊 Time range: 3 PM to 11 PM (date written on fender)")
        return release_times
    
    def test_forward_scenarios(self):
        """Test all combinations of release points and times for August 22nd"""
        
        logger.info("🔄 Running forward seeding analysis...")
        logger.info(f"📝 Date written on fender: August 22, 2025")
        logger.info(f"⏰ Release times: 3 PM to 11 PM")
        logger.info(f"🎯 Target: South Haven area by Aug 28, 4 PM")
        logger.info("=" * 70)
        
        release_points = self.generate_release_grid()
        release_times = self.generate_release_times()
        
        successful_scenarios = []
        all_results = []
        
        scenario_count = 0
        total_scenarios = len(release_points) * len(release_times)
        
        for release_point in release_points:
            for release_time_info in release_times:
                scenario_count += 1
                
                if scenario_count % 10 == 0:
                    logger.info(f"Progress: {scenario_count}/{total_scenarios} scenarios tested")
                
                # Run drift simulation
                result = self.run_scenario(release_point, release_time_info)
                all_results.append(result)
                
                # Check if this scenario hits the target area
                if result['hits_target']:
                    successful_scenarios.append(result)
                    logger.info(f"🎯 HIT: {result['scenario_name']} -> {result['final_distance_nm']:.1f} nm from target")
        
        logger.info(f"✅ Analysis complete: {len(successful_scenarios)}/{total_scenarios} scenarios successful")
        
        return successful_scenarios, all_results
    
    def run_scenario(self, release_point, release_time_info):
        """Run single drift scenario"""
        
        scenario_name = f"{release_point['name']}_{release_time_info['date_str']}"
        
        # Create simple environmental conditions (using real GLOS patterns)
        # From our previous analysis: 3.1 m/s wind, 93° direction
        env_conditions = []
        num_steps = int(release_time_info['duration_hours'] / 0.5) + 1  # 30-minute steps
        
        for i in range(num_steps):
            time_offset = i * 1800  # 30 minutes in seconds
            env_conditions.append({
                'timestamp': release_time_info['time'] + timedelta(seconds=time_offset),
                'wind_speed': 3.1,  # From real GLOS data
                'wind_direction': 93,  # From real GLOS data
                'current_u': 0.0,   # Simplified for now
                'current_v': 0.0,
                'wave_height': 1.0
            })
        
        # Run simulation
        try:
            results = self.engine.simulate_drift_ensemble(
                release_lat=release_point['lat'],
                release_lon=release_point['lon'],
                release_time=release_time_info['time'],
                duration_hours=release_time_info['duration_hours'],
                environmental_data=env_conditions,
                n_particles=50,  # Reduced for faster processing
                object_specs={
                    'windage': self.optimal_params['windage'],
                    'leeway': self.optimal_params['leeway'],
                    'stokes_factor': self.optimal_params['stokes_factor']
                }
            )
            
            # Calculate distance to target
            center_position = results['statistics']['center_position']
            mean_lat = center_position['lat']
            mean_lon = center_position['lon']
            
            target_pos = (self.target_recovery['position']['lat'], self.target_recovery['position']['lon'])
            final_pos = (mean_lat, mean_lon)
            distance_nm = geodesic(target_pos, final_pos).nautical
            
            # Check if hits target (within tolerance)
            hits_target = distance_nm <= self.target_recovery['shoreline_tolerance']
            
            return {
                'scenario_name': scenario_name,
                'release_point': release_point,
                'release_time': release_time_info,
                'final_lat': mean_lat,
                'final_lon': mean_lon,
                'final_distance_nm': distance_nm,
                'hits_target': hits_target,
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"❌ Scenario {scenario_name} failed: {e}")
            return {
                'scenario_name': scenario_name,
                'release_point': release_point,
                'release_time': release_time_info,
                'final_distance_nm': 999.0,
                'hits_target': False,
                'success': False,
                'error': str(e)
            }
    
    def analyze_successful_scenarios(self, successful_scenarios):
        """Analyze and rank successful scenarios"""
        
        if not successful_scenarios:
            logger.warning("❌ No successful scenarios found!")
            return
        
        logger.info(f"📊 Analyzing {len(successful_scenarios)} successful scenarios...")
        
        # Sort by accuracy (closest to target)
        successful_scenarios.sort(key=lambda x: x['final_distance_nm'])
        
        logger.info("🏆 TOP SUCCESSFUL SCENARIOS:")
        logger.info("=" * 60)
        
        for i, scenario in enumerate(successful_scenarios[:10]):  # Top 10
            release_info = scenario['release_point']
            time_info = scenario['release_time']
            
            logger.info(f"{i+1:2d}. {scenario['scenario_name']}")
            logger.info(f"    📍 Release: {release_info['distance_nm']} nm off Milwaukee at {release_info['bearing']}°")
            logger.info(f"    ⏰ Time: {time_info['date_str']} (Aug 22 - date on fender)")
            logger.info(f"    🎯 Final distance: {scenario['final_distance_nm']:.1f} nm from target")
            logger.info(f"    📊 Duration: {time_info['duration_hours']:.1f} hours")
            logger.info()
        
        # Create summary report
        self.create_forward_analysis_report(successful_scenarios)
    
    def create_forward_analysis_report(self, successful_scenarios):
        """Create detailed analysis report"""
        
        report = {
            'analysis_type': 'rosa_forward_seeding',
            'target_recovery': self.target_recovery,
            'parameters_used': self.optimal_params,
            'total_successful_scenarios': len(successful_scenarios),
            'top_scenarios': []
        }
        
        # Add top 10 scenarios
        for scenario in successful_scenarios[:10]:
            report['top_scenarios'].append({
                'scenario_name': scenario['scenario_name'],
                'release_distance_nm': scenario['release_point']['distance_nm'],
                'release_bearing': scenario['release_point']['bearing'],
                'release_time': scenario['release_time']['time'].isoformat(),
                'hours_before_recovery': scenario['release_time']['hours_before'],
                'duration_hours': scenario['release_time']['duration_hours'],
                'final_distance_nm': scenario['final_distance_nm'],
                'final_position': {
                    'lat': scenario['final_lat'],
                    'lon': scenario['final_lon']
                }
            })
        
        # Save report
        report_file = 'outputs/rosa_forward_seeding/forward_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📁 Detailed report saved: {report_file}")
        
        return report
    
    def run_forward_analysis(self):
        """Main forward seeding analysis"""
        
        logger.info("🌊 ROSA CASE - FORWARD SEEDING ANALYSIS")
        logger.info(f"Using optimal parameters: A={self.optimal_params['windage']}, Stokes={self.optimal_params['stokes_factor']}")
        logger.info("=" * 70)
        
        # Run all scenarios
        successful_scenarios, all_results = self.test_forward_scenarios()
        
        # Analyze results
        self.analyze_successful_scenarios(successful_scenarios)
        
        logger.info("✅ Forward seeding analysis completed!")
        return successful_scenarios

if __name__ == "__main__":
    analyzer = RosaForwardSeeding()
    results = analyzer.run_forward_analysis()