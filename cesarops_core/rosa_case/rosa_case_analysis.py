#!/usr/bin/env python3
"""
Rosa Fender Case Analysis - Comprehensive Hindcast and Forecast
==============================================================

Real-world validation using the Rosa fender case where the object was found
in South Haven, MI. This implements:

1. Hindcast analysis: Work backwards from South Haven to find likely drop zones
2. Forward cast analysis: From drop zones, predict drift to shore
3. Search grid optimization: Combine both to narrow probable search areas
4. ML model validation against known outcome

Author: GitHub Copilot
Date: January 7, 2025
"""

import sys
import numpy as np
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RosaFenderAnalysis:
    """Comprehensive analysis of the Rosa fender case"""
    
    def __init__(self, models_dir='models', db_file='drift_objects.db'):
        self.models_dir = Path(models_dir)
        self.db_file = db_file
        
        # Rosa case known facts
        self.known_facts = {
            'found_location': {'lat': 42.4030, 'lon': -86.2750},  # South Haven, MI
            'found_date': datetime(2025, 8, 23, 8, 0),  # Found morning after
            'incident_date': datetime(2025, 8, 22, 20, 0),  # 8 PM August 22
            'original_coordinates': {'lat': 42.995, 'lon': -87.845},  # Original coordinates
            'drift_duration_hours': 12,  # Approximate drift time
            'object_type': 'life_jacket_fender',
            'windage': 0.06,  # Calculated windage from our analysis
            'stokes_drift': 0.0045  # Calculated Stokes drift
        }
        
        # Load ML models
        self.load_ml_models()
        
        # Great Lakes bounds for analysis
        self.michigan_bounds = {
            'min_lat': 41.5, 'max_lat': 46.0,
            'min_lon': -88.5, 'max_lon': -85.5
        }
    
    def load_ml_models(self):
        """Load trained ML models"""
        self.models = {}
        
        # Load simple regression
        simple_path = self.models_dir / 'simple_regression_model.json'
        if simple_path.exists():
            with open(simple_path, 'r') as f:
                self.models['simple_regression'] = json.load(f)
            logger.info("✅ Simple regression model loaded")
        
        # Load Random Forest models
        try:
            rf_lat_path = self.models_dir / 'random_forest_lat.pkl'
            rf_lon_path = self.models_dir / 'random_forest_lon.pkl'
            
            if rf_lat_path.exists() and rf_lon_path.exists():
                with open(rf_lat_path, 'rb') as f:
                    rf_lat = pickle.load(f)
                with open(rf_lon_path, 'rb') as f:
                    rf_lon = pickle.load(f)
                
                self.models['random_forest'] = {'lat': rf_lat, 'lon': rf_lon}
                logger.info("✅ Random Forest models loaded")
        except Exception as e:
            logger.warning(f"Random Forest models not available: {e}")
        
        logger.info(f"📊 Loaded {len(self.models)} model types")
    
    def prepare_features(self, lat, lon, timestamp):
        """Prepare features for ML prediction"""
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        month = timestamp.month
        
        lat_sin = np.sin(np.radians(lat))
        lat_cos = np.cos(np.radians(lat))
        lon_sin = np.sin(np.radians(lon))
        lon_cos = np.cos(np.radians(lon))
        
        # Base features (matching training format)
        features = [lat, lon, hour, day_of_year, month, lat_sin, lat_cos, lon_sin, lon_cos]
        
        # Add velocity and temperature defaults
        features.extend([0.1, 0.05, 15.0])  # Default velocity and temperature
        
        return np.array(features).reshape(1, -1)
    
    def predict_next_position(self, lat, lon, timestamp, model_type='ensemble'):
        """Predict next position using ML models"""
        features = self.prepare_features(lat, lon, timestamp)
        
        predictions = {}
        
        # Simple regression prediction
        if 'simple_regression' in self.models and model_type in ['simple_regression', 'ensemble']:
            model = self.models['simple_regression']
            features_bias = np.column_stack([np.ones(features.shape[0]), features])
            
            weights_lat = np.array(model['weights_lat'])
            weights_lon = np.array(model['weights_lon'])
            
            pred_lat = features_bias @ weights_lat
            pred_lon = features_bias @ weights_lon
            
            predictions['simple_regression'] = (pred_lat[0], pred_lon[0])
        
        # Random Forest prediction
        if 'random_forest' in self.models and model_type in ['random_forest', 'ensemble']:
            rf_models = self.models['random_forest']
            
            pred_lat = rf_models['lat'].predict(features)[0]
            pred_lon = rf_models['lon'].predict(features)[0]
            
            predictions['random_forest'] = (pred_lat, pred_lon)
        
        # Ensemble prediction
        if len(predictions) > 1:
            lats = [pred[0] for pred in predictions.values()]
            lons = [pred[1] for pred in predictions.values()]
            return np.mean(lats), np.mean(lons)
        elif len(predictions) == 1:
            return list(predictions.values())[0]
        else:
            # Fallback to simple drift calculation
            return self.simple_drift_step(lat, lon, timestamp)
    
    def simple_drift_step(self, lat, lon, timestamp, hours=1):
        """Simple drift calculation as fallback"""
        # Lake Michigan typical current patterns
        # General northward flow in summer
        lat_drift = 0.005 * hours  # ~0.5 km/hour northward
        lon_drift = 0.001 * hours  # slight eastward
        
        # Add some randomness for wind effects
        wind_effect_lat = np.random.normal(0, 0.002)
        wind_effect_lon = np.random.normal(0, 0.002)
        
        new_lat = lat + lat_drift + wind_effect_lat
        new_lon = lon + lon_drift + wind_effect_lon
        
        return new_lat, new_lon
    
    def run_hindcast_analysis(self, hours_back=24, grid_size=0.1):
        """
        Run hindcast analysis from South Haven to find likely drop zones
        """
        logger.info(f"🔄 Running hindcast analysis from South Haven ({hours_back} hours back)")
        
        found_loc = self.known_facts['found_location']
        found_time = self.known_facts['found_date']
        
        # Create grid of potential starting positions
        hindcast_results = []
        
        # Grid search around the known original coordinates
        center_lat = self.known_facts['original_coordinates']['lat']
        center_lon = self.known_facts['original_coordinates']['lon']
        
        # Create search grid
        lat_range = np.arange(center_lat - 0.5, center_lat + 0.5, grid_size)
        lon_range = np.arange(center_lon - 0.5, center_lon + 0.5, grid_size)
        
        best_scenarios = []
        
        for start_lat in lat_range:
            for start_lon in lon_range:
                # Skip if outside Lake Michigan
                if not self.is_in_lake_michigan(start_lat, start_lon):
                    continue
                
                # Run forward simulation from this starting point
                trajectory = self.run_forward_simulation(
                    start_lat, start_lon, 
                    self.known_facts['incident_date'],
                    hours_back
                )
                
                if trajectory:
                    # Check how close the end point is to South Haven
                    end_lat, end_lon = trajectory[-1][1], trajectory[-1][2]
                    distance_to_found = self.calculate_distance(
                        end_lat, end_lon,
                        found_loc['lat'], found_loc['lon']
                    )
                    
                    scenario = {
                        'start_lat': start_lat,
                        'start_lon': start_lon,
                        'end_lat': end_lat,
                        'end_lon': end_lon,
                        'distance_to_found_km': distance_to_found,
                        'trajectory': trajectory,
                        'accuracy_score': max(0, 50 - distance_to_found)  # Better score for closer matches
                    }
                    
                    hindcast_results.append(scenario)
        
        # Sort by accuracy (closest to South Haven)
        hindcast_results.sort(key=lambda x: x['distance_to_found_km'])
        best_scenarios = hindcast_results[:10]  # Top 10 scenarios
        
        logger.info(f"✅ Hindcast complete: {len(hindcast_results)} scenarios analyzed")
        logger.info(f"🎯 Best scenario: {best_scenarios[0]['distance_to_found_km']:.2f} km from South Haven")
        
        return best_scenarios
    
    def run_forward_simulation(self, start_lat, start_lon, start_time, duration_hours):
        """Run forward drift simulation"""
        trajectory = [(start_time, start_lat, start_lon)]
        
        current_lat = start_lat
        current_lon = start_lon
        current_time = start_time
        
        steps = int(duration_hours)  # 1-hour steps
        
        for step in range(steps):
            # Predict next position
            next_lat, next_lon = self.predict_next_position(
                current_lat, current_lon, current_time
            )
            
            # Check if still in bounds
            if not self.is_in_lake_michigan(next_lat, next_lon):
                break
                
            current_lat = next_lat
            current_lon = next_lon
            current_time += timedelta(hours=1)
            
            trajectory.append((current_time, current_lat, current_lon))
        
        return trajectory
    
    def analyze_shoreline_arrival(self, scenarios):
        """Analyze when and where objects would reach shoreline"""
        logger.info("🏖️ Analyzing shoreline arrival patterns...")
        
        shoreline_analysis = []
        
        for scenario in scenarios:
            trajectory = scenario['trajectory']
            
            # Check each point for proximity to shore
            for i, (timestamp, lat, lon) in enumerate(trajectory):
                distance_to_shore = self.distance_to_nearest_shore(lat, lon)
                
                if distance_to_shore < 2.0:  # Within 2 km of shore
                    arrival_analysis = {
                        'scenario_id': scenarios.index(scenario),
                        'arrival_time': timestamp,
                        'arrival_lat': lat,
                        'arrival_lon': lon,
                        'distance_to_shore_km': distance_to_shore,
                        'drift_duration_hours': (timestamp - scenario['trajectory'][0][0]).total_seconds() / 3600,
                        'nearest_location': self.identify_nearest_location(lat, lon)
                    }
                    shoreline_analysis.append(arrival_analysis)
                    break  # First shore contact
        
        return shoreline_analysis
    
    def distance_to_nearest_shore(self, lat, lon):
        """Estimate distance to nearest shore (simplified)"""
        # Lake Michigan shore approximations
        # East shore (Michigan)
        east_shore_lon = -86.0
        # West shore (Wisconsin/Illinois)
        west_shore_lon = -87.8
        
        # Distance to east or west shore
        east_distance = abs(lon - east_shore_lon) * 111.0  # Rough km conversion
        west_distance = abs(lon - west_shore_lon) * 111.0
        
        return min(east_distance, west_distance)
    
    def identify_nearest_location(self, lat, lon):
        """Identify nearest major location"""
        locations = {
            'South Haven, MI': (42.4030, -86.2750),
            'St. Joseph, MI': (42.1070, -86.4950),
            'Michigan City, IN': (41.7072, -86.8950),
            'Milwaukee, WI': (43.0389, -87.9065),
            'Racine, WI': (42.7261, -87.7829),
            'Kenosha, WI': (42.5847, -87.8211)
        }
        
        min_distance = float('inf')
        nearest_location = 'Unknown'
        
        for location, (loc_lat, loc_lon) in locations.items():
            distance = self.calculate_distance(lat, lon, loc_lat, loc_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        
        return f"{nearest_location} ({min_distance:.1f} km)"
    
    def is_in_lake_michigan(self, lat, lon):
        """Check if coordinates are within Lake Michigan bounds"""
        bounds = self.michigan_bounds
        return (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon'])
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        # Haversine formula
        R = 6371  # Earth radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def generate_search_grid(self, scenarios, confidence_threshold=0.7):
        """Generate optimized search grid based on analysis"""
        logger.info("🎯 Generating optimized search grid...")
        
        # Get high-confidence scenarios
        good_scenarios = [s for s in scenarios if s['accuracy_score'] > confidence_threshold * 50]
        
        if not good_scenarios:
            good_scenarios = scenarios[:5]  # Use top 5 if none meet threshold
        
        # Calculate centroid of likely drop zones
        lats = [s['start_lat'] for s in good_scenarios]
        lons = [s['start_lon'] for s in good_scenarios]
        
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Calculate search area size based on scenario spread
        lat_std = np.std(lats)
        lon_std = np.std(lons)
        
        search_grid = {
            'center': {'lat': center_lat, 'lon': center_lon},
            'search_radius_km': max(lat_std, lon_std) * 111.0 * 2,  # 2 std deviations
            'confidence_zones': [],
            'recommended_search_pattern': 'expanding_square'
        }
        
        # Create confidence zones
        for i, scenario in enumerate(good_scenarios):
            zone = {
                'priority': i + 1,
                'center_lat': scenario['start_lat'],
                'center_lon': scenario['start_lon'],
                'radius_km': 5.0,  # 5 km search radius
                'confidence': scenario['accuracy_score'] / 50.0,
                'predicted_shore_arrival': self.identify_nearest_location(
                    scenario['end_lat'], scenario['end_lon']
                )
            }
            search_grid['confidence_zones'].append(zone)
        
        return search_grid
    
    def save_analysis_results(self, hindcast_scenarios, shoreline_analysis, search_grid):
        """Save comprehensive analysis results"""
        results = {
            'analysis_date': datetime.now().isoformat(),
            'rosa_case_facts': self.known_facts,
            'hindcast_scenarios': hindcast_scenarios,
            'shoreline_analysis': shoreline_analysis,
            'search_grid': search_grid,
            'model_performance': {
                'models_used': list(self.models.keys()),
                'scenarios_analyzed': len(hindcast_scenarios),
                'best_accuracy_km': hindcast_scenarios[0]['distance_to_found_km'] if hindcast_scenarios else None
            }
        }
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_datetime(value) for key, value in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(convert_datetime(item) for item in obj)
            return obj
        
        results = convert_datetime(results)
        
        # Save to JSON
        with open('rosa_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable report
        self.generate_analysis_report(hindcast_scenarios, shoreline_analysis, search_grid)
        
        logger.info("📄 Analysis results saved to rosa_analysis_results.json")
    
    def generate_analysis_report(self, hindcast_scenarios, shoreline_analysis, search_grid):
        """Generate human-readable analysis report"""
        with open('Rosa_Fender_Analysis_Report.txt', 'w') as f:
            f.write("ROSA FENDER CASE - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("KNOWN FACTS:\n")
            f.write("-" * 12 + "\n")
            f.write(f"Found Location: {self.known_facts['found_location']['lat']:.4f}°N, {self.known_facts['found_location']['lon']:.4f}°W (South Haven, MI)\n")
            f.write(f"Found Date: {self.known_facts['found_date']}\n")
            f.write(f"Incident Date: {self.known_facts['incident_date']}\n")
            f.write(f"Original Coordinates: {self.known_facts['original_coordinates']['lat']:.4f}°N, {self.known_facts['original_coordinates']['lon']:.4f}°W\n")
            f.write(f"Drift Duration: ~{self.known_facts['drift_duration_hours']} hours\n\n")
            
            f.write("HINDCAST ANALYSIS RESULTS:\n")
            f.write("-" * 28 + "\n")
            f.write(f"Scenarios Analyzed: {len(hindcast_scenarios)}\n")
            f.write(f"Best Match Distance: {hindcast_scenarios[0]['distance_to_found_km']:.2f} km from South Haven\n\n")
            
            f.write("TOP 5 MOST LIKELY DROP ZONES:\n")
            f.write("-" * 30 + "\n")
            for i, scenario in enumerate(hindcast_scenarios[:5]):
                f.write(f"{i+1}. Start: {scenario['start_lat']:.4f}°N, {scenario['start_lon']:.4f}°W\n")
                f.write(f"   End: {scenario['end_lat']:.4f}°N, {scenario['end_lon']:.4f}°W\n")
                f.write(f"   Distance to South Haven: {scenario['distance_to_found_km']:.2f} km\n")
                f.write(f"   Accuracy Score: {scenario['accuracy_score']:.1f}/50\n\n")
            
            f.write("SEARCH GRID RECOMMENDATIONS:\n")
            f.write("-" * 29 + "\n")
            f.write(f"Primary Search Center: {search_grid['center']['lat']:.4f}°N, {search_grid['center']['lon']:.4f}°W\n")
            f.write(f"Search Radius: {search_grid['search_radius_km']:.1f} km\n")
            f.write(f"Recommended Pattern: {search_grid['recommended_search_pattern']}\n\n")
            
            f.write("PRIORITY SEARCH ZONES:\n")
            f.write("-" * 21 + "\n")
            for zone in search_grid['confidence_zones']:
                f.write(f"Priority {zone['priority']}: {zone['center_lat']:.4f}°N, {zone['center_lon']:.4f}°W\n")
                f.write(f"  Radius: {zone['radius_km']:.1f} km, Confidence: {zone['confidence']:.2f}\n")
                f.write(f"  Predicted Shore: {zone['predicted_shore_arrival']}\n\n")
            
            if shoreline_analysis:
                f.write("SHORELINE ARRIVAL ANALYSIS:\n")
                f.write("-" * 28 + "\n")
                for arrival in shoreline_analysis[:5]:
                    f.write(f"Scenario {arrival['scenario_id']}: {arrival['nearest_location']}\n")
                    f.write(f"  Arrival Time: {arrival['arrival_time']}\n")
                    f.write(f"  Drift Duration: {arrival['drift_duration_hours']:.1f} hours\n\n")
        
        logger.info("📄 Human-readable report saved to Rosa_Fender_Analysis_Report.txt")
    
    def run_complete_analysis(self):
        """Run complete Rosa fender case analysis"""
        logger.info("🚀 Starting complete Rosa fender case analysis...")
        start_time = datetime.now()
        
        # Step 1: Hindcast analysis
        hindcast_scenarios = self.run_hindcast_analysis(hours_back=24, grid_size=0.05)
        
        # Step 2: Shoreline arrival analysis
        shoreline_analysis = self.analyze_shoreline_arrival(hindcast_scenarios[:10])
        
        # Step 3: Generate search grid
        search_grid = self.generate_search_grid(hindcast_scenarios)
        
        # Step 4: Save results
        self.save_analysis_results(hindcast_scenarios, shoreline_analysis, search_grid)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 Complete analysis finished in {duration:.1f} seconds")
        
        return hindcast_scenarios, shoreline_analysis, search_grid

def main():
    """Main function"""
    print("ROSA FENDER CASE - Comprehensive Analysis")
    print("=" * 45)
    print("🔍 Hindcast & Forecast Analysis with ML Models")
    print("📍 Working backwards from South Haven, MI")
    print()
    
    analyzer = RosaFenderAnalysis()
    
    if not analyzer.models:
        print("❌ No ML models found. Run training first.")
        return 1
    
    # Run complete analysis
    hindcast_scenarios, shoreline_analysis, search_grid = analyzer.run_complete_analysis()
    
    # Display key results
    print("\n🎯 KEY FINDINGS:")
    print("-" * 15)
    
    best_scenario = hindcast_scenarios[0]
    print(f"📍 Most Likely Drop Zone:")
    print(f"   Coordinates: {best_scenario['start_lat']:.4f}°N, {best_scenario['start_lon']:.4f}°W")
    print(f"   Distance from original: {analyzer.calculate_distance(best_scenario['start_lat'], best_scenario['start_lon'], 42.995, -87.845):.2f} km")
    print(f"   Predicted end: {best_scenario['distance_to_found_km']:.2f} km from South Haven")
    
    print(f"\n🎯 Search Grid Center:")
    print(f"   Coordinates: {search_grid['center']['lat']:.4f}°N, {search_grid['center']['lon']:.4f}°W")
    print(f"   Search Radius: {search_grid['search_radius_km']:.1f} km")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Scenarios Analyzed: {len(hindcast_scenarios)}")
    print(f"   ML Models Used: {', '.join(analyzer.models.keys())}")
    print(f"   Best Accuracy: {best_scenario['distance_to_found_km']:.2f} km from known location")
    
    print(f"\n📁 Results saved to:")
    print(f"   • rosa_analysis_results.json")
    print(f"   • Rosa_Fender_Analysis_Report.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())