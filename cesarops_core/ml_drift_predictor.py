#!/usr/bin/env python3
"""
ML-Enhanced Drift Predictor
===========================

Uses trained ML models to improve drift predictions for Great Lakes SAR operations.
Integrates with the existing CESAROPS system to provide ML-enhanced forecasts.

Author: GitHub Copilot
Date: January 7, 2025
"""

import sys
import json
import pickle
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MLDriftPredictor:
    """ML-enhanced drift predictor using trained models"""
    
    def __init__(self, models_dir='models', db_file='drift_objects.db'):
        self.models_dir = Path(models_dir)
        self.db_file = db_file
        self.loaded_models = {}
        self.model_metadata = {}
        
        self.load_available_models()
    
    def load_available_models(self):
        """Load all available trained models"""
        logger.info("📂 Loading available ML models...")
        
        # Load simple regression model
        simple_model_path = self.models_dir / 'simple_regression_model.json'
        if simple_model_path.exists():
            try:
                with open(simple_model_path, 'r') as f:
                    self.loaded_models['simple_regression'] = json.load(f)
                logger.info("✅ Simple regression model loaded")
            except Exception as e:
                logger.error(f"Failed to load simple regression: {e}")
        
        # Load Random Forest models
        rf_lat_path = self.models_dir / 'random_forest_lat.pkl'
        rf_lon_path = self.models_dir / 'random_forest_lon.pkl'
        
        if rf_lat_path.exists() and rf_lon_path.exists() and SKLEARN_AVAILABLE:
            try:
                with open(rf_lat_path, 'rb') as f:
                    rf_lat = pickle.load(f)
                with open(rf_lon_path, 'rb') as f:
                    rf_lon = pickle.load(f)
                
                self.loaded_models['random_forest'] = {
                    'lat_model': rf_lat,
                    'lon_model': rf_lon
                }
                logger.info("✅ Random Forest models loaded")
            except Exception as e:
                logger.error(f"Failed to load Random Forest: {e}")
        
        # Load training report for metadata
        report_path = self.models_dir / 'training_report.json'
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("✅ Model metadata loaded")
            except Exception as e:
                logger.error(f"Failed to load model metadata: {e}")
        
        logger.info(f"📊 Loaded {len(self.loaded_models)} model types")
    
    def prepare_features(self, lat, lon, timestamp=None, velocity_u=None, velocity_v=None, sst=None):
        """Prepare features for ML prediction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Time-based features
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        month = timestamp.month
        
        # Position-based features
        lat_sin = np.sin(np.radians(lat))
        lat_cos = np.cos(np.radians(lat))
        lon_sin = np.sin(np.radians(lon))
        lon_cos = np.cos(np.radians(lon))
        
        # Base features (same order as training)
        features = [lat, lon, hour, day_of_year, month, lat_sin, lat_cos, lon_sin, lon_cos]
        
        # Add velocity if available
        if velocity_u is not None and velocity_v is not None:
            features.extend([velocity_u, velocity_v])
        else:
            features.extend([0.0, 0.0])  # Default values
        
        # Add temperature if available
        if sst is not None:
            features.append(sst)
        else:
            features.append(15.0)  # Default Great Lakes temperature
        
        return np.array(features).reshape(1, -1)
    
    def predict_with_simple_regression(self, features):
        """Predict next position using simple regression"""
        if 'simple_regression' not in self.loaded_models:
            return None
        
        model = self.loaded_models['simple_regression']
        
        # Add bias term
        features_bias = np.column_stack([np.ones(features.shape[0]), features])
        
        # Predict
        weights_lat = np.array(model['weights_lat'])
        weights_lon = np.array(model['weights_lon'])
        
        pred_lat = features_bias @ weights_lat
        pred_lon = features_bias @ weights_lon
        
        return pred_lat[0], pred_lon[0]
    
    def predict_with_random_forest(self, features):
        """Predict next position using Random Forest"""
        if 'random_forest' not in self.loaded_models:
            return None
        
        models = self.loaded_models['random_forest']
        
        pred_lat = models['lat_model'].predict(features)[0]
        pred_lon = models['lon_model'].predict(features)[0]
        
        return pred_lat, pred_lon
    
    def predict_drift_position(self, current_lat, current_lon, time_step_hours=1, 
                             velocity_u=None, velocity_v=None, sst=None, 
                             model_type='ensemble'):
        """
        Predict drift position after time_step_hours
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            time_step_hours: Time step in hours for prediction
            velocity_u: Current velocity U component (m/s)
            velocity_v: Current velocity V component (m/s)
            sst: Sea surface temperature (°C)
            model_type: 'simple_regression', 'random_forest', or 'ensemble'
        
        Returns:
            (predicted_lat, predicted_lon, confidence)
        """
        logger.info(f"🎯 Predicting drift from ({current_lat:.4f}, {current_lon:.4f})")
        
        # Prepare features
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=time_step_hours)
        
        features = self.prepare_features(
            current_lat, current_lon, future_time, 
            velocity_u, velocity_v, sst
        )
        
        predictions = {}
        
        # Get predictions from available models
        if model_type in ['simple_regression', 'ensemble']:
            simple_pred = self.predict_with_simple_regression(features)
            if simple_pred:
                predictions['simple_regression'] = simple_pred
        
        if model_type in ['random_forest', 'ensemble']:
            rf_pred = self.predict_with_random_forest(features)
            if rf_pred:
                predictions['random_forest'] = rf_pred
        
        if not predictions:
            logger.error("No model predictions available")
            return None, None, 0.0
        
        # Combine predictions
        if model_type == 'ensemble' and len(predictions) > 1:
            # Simple ensemble average
            lats = [pred[0] for pred in predictions.values()]
            lons = [pred[1] for pred in predictions.values()]
            
            pred_lat = np.mean(lats)
            pred_lon = np.mean(lons)
            
            # Confidence based on agreement between models
            lat_std = np.std(lats)
            lon_std = np.std(lons)
            disagreement = np.sqrt(lat_std**2 + lon_std**2)
            confidence = max(0.0, 1.0 - disagreement * 100)  # Scale disagreement to confidence
            
            logger.info(f"📊 Ensemble prediction: {len(predictions)} models")
            
        else:
            # Single model prediction
            model_name = list(predictions.keys())[0]
            pred_lat, pred_lon = predictions[model_name]
            
            # Confidence based on model training metrics
            if model_name in self.model_metadata.get('training_metrics', {}):
                r2_score = self.model_metadata['training_metrics'][model_name].get('r2_score', 0.5)
                confidence = r2_score
            else:
                confidence = 0.8  # Default confidence
            
            logger.info(f"📊 Single model prediction: {model_name}")
        
        logger.info(f"🎯 Predicted position: ({pred_lat:.4f}, {pred_lon:.4f})")
        logger.info(f"📈 Confidence: {confidence:.2f}")
        
        return pred_lat, pred_lon, confidence
    
    def predict_trajectory(self, start_lat, start_lon, duration_hours=24, 
                          time_step_hours=1, initial_velocity_u=None, 
                          initial_velocity_v=None, model_type='ensemble'):
        """
        Predict full trajectory over duration
        
        Returns:
            List of (timestamp, lat, lon, confidence) tuples
        """
        logger.info(f"🛤️ Predicting trajectory for {duration_hours} hours")
        
        trajectory = []
        current_lat = start_lat
        current_lon = start_lon
        current_time = datetime.now()
        
        # Add starting point
        trajectory.append((current_time, current_lat, current_lon, 1.0))
        
        steps = int(duration_hours / time_step_hours)
        
        for step in range(steps):
            # Predict next position
            pred_lat, pred_lon, confidence = self.predict_drift_position(
                current_lat, current_lon, time_step_hours,
                initial_velocity_u, initial_velocity_v, 
                model_type=model_type
            )
            
            if pred_lat is None:
                logger.error(f"Prediction failed at step {step}")
                break
            
            # Update position and time
            current_lat = pred_lat
            current_lon = pred_lon
            current_time += timedelta(hours=time_step_hours)
            
            trajectory.append((current_time, current_lat, current_lon, confidence))
            
            # Simple velocity decay (in real implementation, would use environmental data)
            if initial_velocity_u is not None:
                initial_velocity_u *= 0.95  # Velocity decay
            if initial_velocity_v is not None:
                initial_velocity_v *= 0.95
        
        logger.info(f"✅ Generated trajectory with {len(trajectory)} points")
        return trajectory
    
    def save_trajectory_to_kml(self, trajectory, filename='ml_predicted_trajectory.kml'):
        """Save trajectory to KML file for visualization"""
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>CESAROPS ML-Enhanced Drift Prediction</name>
    <description>Machine learning enhanced drift trajectory prediction</description>
    
    <Style id="trajectoryLine">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
    </Style>
    
    <Style id="startPoint">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
    
    <Style id="endPoint">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
    
    <Placemark>
      <name>ML Predicted Trajectory</name>
      <styleUrl>#trajectoryLine</styleUrl>
      <LineString>
        <coordinates>
'''
        
        # Add coordinates
        for timestamp, lat, lon, confidence in trajectory:
            kml_content += f"          {lon:.6f},{lat:.6f},0\n"
        
        kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
    
    <Placemark>
      <name>Start Position</name>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <coordinates>{},{},0</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>Predicted End Position</name>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <coordinates>{},{},0</coordinates>
      </Point>
    </Placemark>
    
  </Document>
</kml>'''.format(
            trajectory[0][2], trajectory[0][1],  # Start point
            trajectory[-1][2], trajectory[-1][1]  # End point
        )
        
        with open(filename, 'w') as f:
            f.write(kml_content)
        
        logger.info(f"💾 Trajectory saved to {filename}")

def main():
    """Main function - demonstrate ML-enhanced drift prediction"""
    print("CESAROPS ML-Enhanced Drift Predictor")
    print("=" * 40)
    
    # Initialize predictor
    predictor = MLDriftPredictor()
    
    if not predictor.loaded_models:
        print("❌ No trained models found")
        print("Run: python simple_ml_trainer.py")
        return 1
    
    # Show available models
    print(f"📊 Available models: {list(predictor.loaded_models.keys())}")
    
    # Example prediction - Rosa case coordinates
    print(f"\n🎯 Example: Rosa Fender Case")
    print(f"Location: 42.995°N, 87.845°W")
    print(f"Time: August 22, 8 PM")
    
    start_lat = 42.995
    start_lon = -87.845
    
    # Single step prediction
    pred_lat, pred_lon, confidence = predictor.predict_drift_position(
        start_lat, start_lon, time_step_hours=6,
        velocity_u=0.1, velocity_v=0.05,  # Some initial velocity
        model_type='ensemble'
    )
    
    if pred_lat is not None:
        print(f"✅ 6-hour prediction:")
        print(f"   Position: {pred_lat:.4f}°N, {pred_lon:.4f}°W")
        print(f"   Confidence: {confidence:.2f}")
        
        # Calculate distance moved
        distance_km = np.sqrt((pred_lat - start_lat)**2 + (pred_lon - start_lon)**2) * 111.0
        print(f"   Distance: {distance_km:.2f} km")
    
    # Full trajectory prediction
    print(f"\n🛤️ Generating 24-hour trajectory...")
    trajectory = predictor.predict_trajectory(
        start_lat, start_lon, duration_hours=24, 
        time_step_hours=2, model_type='ensemble'
    )
    
    if trajectory:
        print(f"✅ Trajectory generated with {len(trajectory)} points")
        
        # Save to KML
        predictor.save_trajectory_to_kml(trajectory, 'outputs/ml_trajectory_rosa_case.kml')
        
        # Show key points
        print(f"\nKey trajectory points:")
        for i, (timestamp, lat, lon, conf) in enumerate(trajectory):
            if i % 6 == 0:  # Every 12 hours
                print(f"  {timestamp.strftime('%m/%d %H:%M')}: {lat:.4f}°N, {lon:.4f}°W (conf: {conf:.2f})")
        
        # Final position
        final_time, final_lat, final_lon, final_conf = trajectory[-1]
        total_distance = np.sqrt((final_lat - start_lat)**2 + (final_lon - start_lon)**2) * 111.0
        print(f"\n📍 24-hour prediction:")
        print(f"   Final position: {final_lat:.4f}°N, {final_lon:.4f}°W")
        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Final confidence: {final_conf:.2f}")
    
    print(f"\n💡 Models trained on {predictor.model_metadata.get('training_metrics', {}).get('simple_regression', {}).get('training_samples', 'unknown')} trajectory points")
    print(f"📁 KML output: outputs/ml_trajectory_rosa_case.kml")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())