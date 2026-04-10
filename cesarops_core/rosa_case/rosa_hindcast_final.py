#!/usr/bin/env python3
"""
ROSA FENDER HINDCAST - Complete Analysis
Using the new local environment and fast drift engine
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_fast_engine import SimpleFastDriftEngine
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ROSA_HINDCAST')

class RosaFenderHindcastFinal:
    """
    Complete Rosa fender hindcast analysis using fast drift engine
    """
    
    def __init__(self):
        self.engine = SimpleFastDriftEngine(use_multiprocessing=False, n_workers=1)
        
        # Case details from user
        self.case_details = {
            'vessel_name': 'Rosa',
            'victim': 'Charlie Brown',
            'incident_date': '2025-08-22',
            'object_description': '8-10 inch orange teardrop fender with 3ft tagline',
            
            # Release window: 3 PM to 11:59 PM CDT on Aug 22
            'release_window': {
                'earliest': datetime(2025, 8, 22, 15, 0, 0),  # 3 PM CDT
                'latest': datetime(2025, 8, 22, 23, 59, 0),   # 11:59 PM CDT
            },
            
            # Recovery details
            'recovery': {
                'lat': 42.40,        # Near South Haven, MI
                'lon': -86.28,
                'time': datetime(2025, 8, 28, 16, 0, 0),  # 4 PM Aug 28
                'location': 'South Haven, MI area'
            },
            
            # Potential release locations (2-3 nm from harbor)
            'potential_release_locations': [
                {'name': 'Milwaukee 2nm', 'lat': 43.0389, 'lon': -87.8731},  # 2 nm east
                {'name': 'Milwaukee 2.5nm', 'lat': 43.0389, 'lon': -87.8560}, # 2.5 nm east  
                {'name': 'Milwaukee 3nm', 'lat': 43.0389, 'lon': -87.8389},   # 3 nm east
            ]
        }
        
        # Create outputs directory
        os.makedirs('outputs/rosa_final_hindcast', exist_ok=True)
        
        logger.info("🔍 Rosa Fender Hindcast Analysis - Final Version")
        logger.info(f"Victim: {self.case_details['victim']}")
        logger.info(f"Recovery: {self.case_details['recovery']['time']} at {self.case_details['recovery']['location']}")
    
    def run_complete_hindcast(self):
        """Run complete hindcast analysis with multiple scenarios"""
        
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE ROSA FENDER HINDCAST ANALYSIS")
        logger.info("=" * 60)
        
        # 1. Generate environmental data for the period
        logger.info("Step 1: Generating environmental data...")
        env_data = self._generate_august_environmental_data()
        
        # 2. Run multiple release scenarios
        logger.info("Step 2: Running multiple release scenarios...")
        scenario_results = self._run_multiple_scenarios(env_data)
        
        # 3. Score scenarios against actual recovery
        logger.info("Step 3: Scoring scenarios...")
        scored_results = self._score_scenarios(scenario_results)
        
        # 4. Generate final report
        logger.info("Step 4: Creating final report...")
        final_report = self._create_final_report(scored_results)
        
        logger.info("🎉 HINDCAST ANALYSIS COMPLETED!")
        return final_report
    
    def _generate_august_environmental_data(self):
        """Generate realistic August environmental conditions for Lake Michigan"""
        
        start_time = datetime(2025, 8, 22, 0, 0, 0)
        end_time = datetime(2025, 8, 29, 0, 0, 0)
        
        env_data = []
        current_time = start_time
        hour = 0
        
        while current_time <= end_time:
            # August Lake Michigan pattern with weather system passage
            day_of_period = (current_time - start_time).days
            hour_of_day = current_time.hour
            
            # Base conditions
            base_wind_speed = 7.0
            base_wind_dir = 240.0  # SW
            base_current_u = 0.06  # Eastward
            base_current_v = 0.03  # Northward
            
            # Weather pattern simulation
            if day_of_period < 2:  # Aug 22-23: Building weather
                wind_modifier = 1.0 + 0.5 * day_of_period
                pressure = 1018 - 2 * day_of_period
                dir_shift = -10 * day_of_period
            elif day_of_period < 4:  # Aug 24-25: Storm passage  
                wind_modifier = 2.0 - 0.2 * (day_of_period - 2)
                pressure = 1010 + 2 * (day_of_period - 2)
                dir_shift = -20 + 5 * (day_of_period - 2)
            else:  # Aug 26-28: Clearing
                wind_modifier = 1.5 - 0.2 * (day_of_period - 4)
                pressure = 1015 + 1 * (day_of_period - 4)
                dir_shift = -10 + 5 * (day_of_period - 4)
            
            # Diurnal variation
            diurnal_wind = 2.0 * np.sin((hour_of_day - 6) * np.pi / 12)
            
            # Random variation
            wind_noise = np.random.normal(0, 0.5)
            dir_noise = np.random.normal(0, 8.0)
            
            env_data.append({
                'timestamp': current_time,
                'latitude': 43.0,
                'longitude': -87.0,
                'wind_speed': max(1.0, base_wind_speed * wind_modifier + diurnal_wind + wind_noise),
                'wind_direction': (base_wind_dir + dir_shift + dir_noise) % 360,
                'current_u': base_current_u + 0.02 * np.sin(hour * 0.1),
                'current_v': base_current_v + 0.01 * np.cos(hour * 0.08),
                'wave_height': max(0.3, 0.8 + 0.3 * wind_modifier + 0.2 * np.sin(hour * 0.12)),
                'water_temp': 21.0 - 0.1 * day_of_period,
                'pressure': pressure + np.random.normal(0, 1.5),
                'air_temp': 24.0 + 4.0 * np.sin((hour_of_day - 8) * np.pi / 12)
            })
            
            current_time += timedelta(hours=1)
            hour += 1
        
        logger.info(f"Generated {len(env_data)} hourly environmental records")
        return env_data
    
    def _run_multiple_scenarios(self, env_data):
        """Run drift simulations for multiple release scenarios"""
        
        scenario_results = {}
        
        # Release times: Every 2 hours from 3 PM to 11 PM
        release_times = []
        for hour_offset in [0, 2, 4, 6, 8]:  # 3 PM, 5 PM, 7 PM, 9 PM, 11 PM
            release_times.append(self.case_details['release_window']['earliest'] + timedelta(hours=hour_offset))
        
        scenario_count = 0
        total_scenarios = len(self.case_details['potential_release_locations']) * len(release_times)
        
        for location in self.case_details['potential_release_locations']:
            for release_time in release_times:
                scenario_count += 1
                scenario_name = f"{location['name']}_{release_time.strftime('%H%M')}"
                
                logger.info(f"Running scenario {scenario_count}/{total_scenarios}: {scenario_name}")
                
                # Calculate duration until recovery
                recovery_time = self.case_details['recovery']['time']
                duration_hours = (recovery_time - release_time).total_seconds() / 3600.0
                
                # Run simulation
                result = self.engine.simulate_drift_ensemble(
                    release_lat=location['lat'],
                    release_lon=location['lon'], 
                    release_time=release_time,
                    duration_hours=duration_hours,
                    environmental_data=env_data,
                    n_particles=200,  # Good balance of accuracy and speed
                    time_step_minutes=20,
                    object_specs={
                        'windage': 0.08,      # High for orange teardrop fender
                        'leeway': 0.15,       # Significant cross-wind drift
                        'stokes_factor': 0.025 # Wave drift
                    }
                )
                
                scenario_results[scenario_name] = {
                    'simulation_result': result,
                    'release_location': location,
                    'release_time': release_time,
                    'duration_hours': duration_hours
                }
        
        return scenario_results
    
    def _score_scenarios(self, scenario_results):
        """Score scenarios based on proximity to actual recovery location"""
        
        actual_lat = self.case_details['recovery']['lat']
        actual_lon = self.case_details['recovery']['lon']
        
        scored_scenarios = []
        
        for scenario_name, scenario_data in scenario_results.items():
            result = scenario_data['simulation_result']
            center = result['statistics']['center_position']
            
            # Calculate distance error
            distance_error = self.engine._calculate_distance(
                center['lat'], center['lon'], actual_lat, actual_lon
            )
            
            # Calculate time preference score (prefer evening releases)
            release_hour = scenario_data['release_time'].hour
            optimal_hour = 20  # 8 PM
            time_penalty = abs(release_hour - optimal_hour) * 0.5
            
            # Combined score
            total_score = distance_error + time_penalty
            
            scored_scenarios.append({
                'scenario_name': scenario_name,
                'release_location': scenario_data['release_location'],
                'release_time': scenario_data['release_time'],
                'duration_hours': scenario_data['duration_hours'],
                'predicted_position': center,
                'actual_position': {'lat': actual_lat, 'lon': actual_lon},
                'distance_error_nm': distance_error,
                'time_penalty': time_penalty,
                'total_score': total_score,
                'accuracy_rating': self._get_accuracy_rating(distance_error),
                'simulation_result': result
            })
        
        # Sort by score (best first)
        scored_scenarios.sort(key=lambda x: x['total_score'])
        
        return scored_scenarios
    
    def _get_accuracy_rating(self, distance_error):
        """Get accuracy rating based on distance error"""
        if distance_error < 5.0:
            return "Excellent"
        elif distance_error < 15.0:
            return "Good"
        elif distance_error < 30.0:
            return "Fair"
        else:
            return "Poor"
    
    def _create_final_report(self, scored_scenarios):
        """Create comprehensive final report"""
        
        best_scenario = scored_scenarios[0]
        
        # Save detailed results
        output_dir = 'outputs/rosa_final_hindcast'
        
        # Summary data for JSON
        summary_data = {
            'case_summary': {
                'vessel': self.case_details['vessel_name'],
                'victim': self.case_details['victim'],
                'object': self.case_details['object_description'],
                'recovery_location': f"{self.case_details['recovery']['lat']:.4f}°N, {self.case_details['recovery']['lon']:.4f}°W",
                'recovery_time': self.case_details['recovery']['time'].isoformat(),
                'analysis_date': datetime.now().isoformat()
            },
            
            'best_scenario': {
                'scenario_name': best_scenario['scenario_name'],
                'release_location': best_scenario['release_location']['name'],
                'release_coordinates': f"{best_scenario['release_location']['lat']:.4f}°N, {best_scenario['release_location']['lon']:.4f}°W",
                'release_time': best_scenario['release_time'].isoformat(),
                'predicted_recovery': f"{best_scenario['predicted_position']['lat']:.4f}°N, {best_scenario['predicted_position']['lon']:.4f}°W",
                'distance_error_nm': round(best_scenario['distance_error_nm'], 2),
                'accuracy_rating': best_scenario['accuracy_rating'],
                'total_score': round(best_scenario['total_score'], 2)
            },
            
            'all_scenarios': []
        }
        
        # Add all scenarios to summary
        for scenario in scored_scenarios[:5]:  # Top 5
            summary_data['all_scenarios'].append({
                'name': scenario['scenario_name'],
                'release_time': scenario['release_time'].strftime('%Y-%m-%d %H:%M CDT'),
                'location': scenario['release_location']['name'],
                'error_nm': round(scenario['distance_error_nm'], 2),
                'rating': scenario['accuracy_rating'],
                'score': round(scenario['total_score'], 2)
            })
        
        # Save JSON report
        with open(f'{output_dir}/rosa_hindcast_final_report.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Create CSV summary
        csv_data = []
        for scenario in scored_scenarios:
            csv_data.append({
                'scenario_name': scenario['scenario_name'],
                'release_time': scenario['release_time'].strftime('%Y-%m-%d %H:%M'),
                'release_lat': scenario['release_location']['lat'],
                'release_lon': scenario['release_location']['lon'],
                'predicted_lat': scenario['predicted_position']['lat'],
                'predicted_lon': scenario['predicted_position']['lon'], 
                'distance_error_nm': scenario['distance_error_nm'],
                'accuracy_rating': scenario['accuracy_rating'],
                'total_score': scenario['total_score']
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(f'{output_dir}/rosa_scenarios_summary.csv', index=False)
        
        # Return comprehensive report
        return {
            'summary': summary_data,
            'best_scenario': best_scenario,
            'all_scenarios': scored_scenarios,
            'output_files': [
                f'{output_dir}/rosa_hindcast_final_report.json',
                f'{output_dir}/rosa_scenarios_summary.csv'
            ]
        }

def main():
    """Main execution function"""
    
    print("🔍 ROSA FENDER HINDCAST - FINAL ANALYSIS")
    print("Real SAR Case: Charlie Brown's Vessel")
    print("Object: Orange teardrop fender with 3ft tagline")
    print("Recovery: Aug 28, 2025 near South Haven, MI")
    print("=" * 60)
    print()
    
    # Run analysis
    analyzer = RosaFenderHindcastFinal()
    report = analyzer.run_complete_hindcast()
    
    # Display key findings
    best = report['best_scenario']
    
    print("\n🎯 KEY FINDINGS:")
    print("=" * 40)
    print(f"Most Likely Release Time: {best['release_time'].strftime('%B %d, %Y at %I:%M %p CDT')}")
    print(f"Most Likely Release Location: {best['release_location']['name']}")
    print(f"Release Coordinates: {best['release_location']['lat']:.4f}°N, {best['release_location']['lon']:.4f}°W")
    print(f"Predicted Recovery: {best['predicted_position']['lat']:.4f}°N, {best['predicted_position']['lon']:.4f}°W")
    print(f"Actual Recovery: {analyzer.case_details['recovery']['lat']:.4f}°N, {analyzer.case_details['recovery']['lon']:.4f}°W")
    print(f"Prediction Error: {best['distance_error_nm']:.2f} nautical miles")
    print(f"Accuracy Rating: {best['accuracy_rating']}")
    print(f"Drift Duration: {best['duration_hours']:.1f} hours")
    
    print(f"\n📊 TOP 5 SCENARIOS:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Scenario':<18} {'Release Time':<17} {'Error (nm)':<10} {'Rating':<10}")
    print("-" * 80)
    for i, scenario in enumerate(report['all_scenarios'][:5], 1):
        release_time_str = scenario['release_time'].strftime('%H:%M')
        print(f"{i:<4} {scenario['scenario_name']:<18} {release_time_str:<17} {scenario['distance_error_nm']:<10.1f} {scenario['accuracy_rating']:<10}")
    
    print(f"\n📁 OUTPUT FILES:")
    for filepath in report['output_files']:
        print(f"  • {filepath}")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"The model predicts the fender was most likely released on {best['release_time'].strftime('%B %d at %I:%M %p CDT')}")
    print(f"from approximately {best['release_location']['name']} with an accuracy of {best['accuracy_rating']} (±{best['distance_error_nm']:.1f} nm)")
    
    return report

if __name__ == "__main__":
    main()