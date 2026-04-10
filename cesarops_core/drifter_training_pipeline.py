#!/usr/bin/env python3
"""
Comprehensive Drifter Buoy Data Collection and ML Training System
================================================================

Collects data from multiple sources and trains ML models:
- NOAA Global Drifter Program (GDP) - 1500+ active drifters
- NDBC buoys - Great Lakes and coastal stations
- GLOS Seagull - Real-time Great Lakes observations
- Historical archives for training data compilation

Features:
- Multi-source data integration
- Automated quality control
- ML model training pipeline
- Performance validation
- Real-time data streaming

Author: GitHub Copilot
Date: January 7, 2025
License: MIT
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drifter_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NOAAGDPCollector:
    """Collects data from NOAA Global Drifter Program"""
    
    def __init__(self, db_path: str = 'drift_objects.db'):
        self.db_path = db_path
        self.base_url = "https://www.aoml.noaa.gov/phod/gdp"
        self.api_endpoints = {
            'active_drifters': 'https://www.aoml.noaa.gov/phod/gdp/json/gdp_active_drifters.json',
            'drifter_data': 'https://www.aoml.noaa.gov/phod/gdp/data/{drifter_id}/',
            'interpolated': 'https://www.aoml.noaa.gov/phod/gdp/interpolated/data/{year}/',
            'hourly': 'https://www.aoml.noaa.gov/phod/gdp/hourly/data/{year}/'
        }
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables for GDP data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # GDP drifters metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gdp_drifters (
                drifter_id INTEGER PRIMARY KEY,
                wmo_id TEXT,
                deploy_date TEXT,
                end_date TEXT,
                deploy_lat REAL,
                deploy_lon REAL,
                last_lat REAL,
                last_lon REAL,
                drogue_status TEXT,
                death_code TEXT,
                data_quality TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # GDP trajectory data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gdp_trajectories (
                id INTEGER PRIMARY KEY,
                drifter_id INTEGER,
                timestamp TEXT,
                latitude REAL,
                longitude REAL,
                u_velocity REAL,
                v_velocity REAL,
                temperature REAL,
                drogue_status INTEGER,
                position_quality INTEGER,
                velocity_quality INTEGER,
                temp_quality INTEGER,
                FOREIGN KEY (drifter_id) REFERENCES gdp_drifters (drifter_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("GDP database tables initialized")
    
    def fetch_active_drifters(self) -> List[Dict]:
        """Fetch list of currently active GDP drifters"""
        try:
            response = requests.get(self.api_endpoints['active_drifters'], timeout=30)
            response.raise_for_status()
            
            drifters = response.json()
            logger.info(f"Retrieved {len(drifters)} active GDP drifters")
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for drifter in drifters:
                cursor.execute("""
                    INSERT OR REPLACE INTO gdp_drifters 
                    (drifter_id, wmo_id, deploy_date, deploy_lat, deploy_lon, 
                     last_lat, last_lon, drogue_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    drifter.get('id'),
                    drifter.get('wmo'),
                    drifter.get('deploy_date'),
                    drifter.get('deploy_lat'),
                    drifter.get('deploy_lon'),
                    drifter.get('last_lat'),
                    drifter.get('last_lon'),
                    drifter.get('drogue')
                ))
            
            conn.commit()
            conn.close()
            
            return drifters
            
        except Exception as e:
            logger.error(f"Failed to fetch active GDP drifters: {e}")
            return []
    
    def fetch_drifter_trajectory(self, drifter_id: int, 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch trajectory data for specific drifter"""
        try:
            # Use ERDDAP endpoint for better data access
            erddap_url = f"https://coastwatch.pfeg.noaa.gov/erddap/tabledap/gdp_interpolated_drifter_data.csv"
            
            params = {
                'trajectory_id': drifter_id,
                'time>=': start_date or '2020-01-01T00:00:00Z',
                'time<=': end_date or datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            response = requests.get(erddap_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), skiprows=1)
            
            if not df.empty:
                # Store in database
                conn = sqlite3.connect(self.db_path)
                
                # Prepare data for database insertion
                trajectory_data = []
                for _, row in df.iterrows():
                    trajectory_data.append((
                        drifter_id,
                        row.get('time'),
                        row.get('latitude'),
                        row.get('longitude'),
                        row.get('u'),
                        row.get('v'),
                        row.get('temperature'),
                        row.get('drogue_status'),
                        row.get('position_quality'),
                        row.get('velocity_quality'),
                        row.get('temp_quality')
                    ))
                
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO gdp_trajectories 
                    (drifter_id, timestamp, latitude, longitude, u_velocity, v_velocity,
                     temperature, drogue_status, position_quality, velocity_quality, temp_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, trajectory_data)
                
                conn.commit()
                conn.close()
                
                logger.info(f"Stored {len(df)} trajectory points for drifter {drifter_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch trajectory for drifter {drifter_id}: {e}")
            return pd.DataFrame()

class NDBCCollector:
    """Collects data from NDBC buoy network"""
    
    def __init__(self, db_path: str = 'drift_objects.db'):
        self.db_path = db_path
        self.base_url = "https://www.ndbc.noaa.gov"
        self.great_lakes_buoys = [
            '45001', '45002', '45003', '45004', '45005', '45006', '45007', '45008',
            '45161', '45162', '45163', '45164', '45165', '45166', '45167', '45168',
            '45169', '45170', '45171', '45172', '45173', '45174', '45175', '45176'
        ]
        self._init_database()
    
    def _init_database(self):
        """Initialize NDBC data tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ndbc_buoys (
                station_id TEXT PRIMARY KEY,
                station_name TEXT,
                latitude REAL,
                longitude REAL,
                water_depth REAL,
                station_type TEXT,
                lake TEXT,
                active BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ndbc_observations (
                id INTEGER PRIMARY KEY,
                station_id TEXT,
                timestamp TEXT,
                wind_direction REAL,
                wind_speed REAL,
                wind_gust REAL,
                wave_height REAL,
                wave_period REAL,
                wave_direction REAL,
                water_temp REAL,
                air_temp REAL,
                pressure REAL,
                FOREIGN KEY (station_id) REFERENCES ndbc_buoys (station_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_buoy_metadata(self) -> Dict:
        """Fetch NDBC buoy station metadata"""
        try:
            # Download station metadata
            metadata_url = f"{self.base_url}/data/stations/station_table.txt"
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            
            # Parse fixed-width format
            lines = response.text.split('\n')
            buoys = {}
            
            for line in lines[2:]:  # Skip header lines
                if len(line) > 50:  # Valid data line
                    station_id = line[0:5].strip()
                    if station_id in self.great_lakes_buoys:
                        buoys[station_id] = {
                            'station_id': station_id,
                            'station_name': line[6:24].strip(),
                            'latitude': float(line[26:33].strip()) if line[26:33].strip() else None,
                            'longitude': float(line[34:42].strip()) if line[34:42].strip() else None,
                            'station_type': line[43:].strip()
                        }
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for buoy_id, data in buoys.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO ndbc_buoys 
                    (station_id, station_name, latitude, longitude, station_type, active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    data['station_id'],
                    data['station_name'],
                    data['latitude'],
                    data['longitude'],
                    data['station_type'],
                    True
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Retrieved metadata for {len(buoys)} Great Lakes NDBC buoys")
            return buoys
            
        except Exception as e:
            logger.error(f"Failed to fetch NDBC metadata: {e}")
            return {}
    
    def fetch_buoy_data(self, station_id: str, year: int = None) -> pd.DataFrame:
        """Fetch historical data for specific buoy"""
        try:
            if year is None:
                year = datetime.now().year
            
            # Standard meteorological data
            data_url = f"{self.base_url}/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/"
            
            response = requests.get(data_url, timeout=60)
            response.raise_for_status()
            
            # Parse text data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), delim_whitespace=True, skiprows=2)
            
            if not df.empty and len(df.columns) >= 8:
                # Rename columns to standard names
                df.columns = ['year', 'month', 'day', 'hour', 'minute', 
                             'wind_dir', 'wind_speed', 'wind_gust', 'wave_height',
                             'wave_period', 'wave_dir', 'pressure', 'air_temp', 'water_temp'][:len(df.columns)]
                
                # Create timestamp
                df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
                
                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO ndbc_observations 
                        (station_id, timestamp, wind_direction, wind_speed, wind_gust,
                         wave_height, wave_period, wave_direction, pressure, air_temp, water_temp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        station_id,
                        row['timestamp'].isoformat(),
                        row.get('wind_dir'),
                        row.get('wind_speed'),
                        row.get('wind_gust'),
                        row.get('wave_height'),
                        row.get('wave_period'),
                        row.get('wave_dir'),
                        row.get('pressure'),
                        row.get('air_temp'),
                        row.get('water_temp')
                    ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Stored {len(df)} observations for NDBC buoy {station_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch NDBC data for {station_id}: {e}")
            return pd.DataFrame()

class GLOSSeagullCollector:
    """Enhanced GLOS Seagull collector from existing analysis"""
    
    def __init__(self, db_path: str = 'drift_objects.db'):
        self.db_path = db_path
        self.base_url = "https://seagull.glos.us/api"
        self.stations = {
            'obs_2': 'ATW20 Milwaukee',
            'obs_181': 'NDBC 45007', 
            'obs_37': 'South Haven'
        }
        self._init_database()
    
    def _init_database(self):
        """Initialize GLOS data tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS glos_observations (
                id INTEGER PRIMARY KEY,
                station_id TEXT,
                timestamp TEXT,
                parameter TEXT,
                value REAL,
                units TEXT,
                quality TEXT,
                depth REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_station_data(self, station_id: str, 
                          start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch GLOS station data (enhanced from existing code)"""
        try:
            url = f"{self.base_url}/obs/{station_id}/"
            params = {
                'start_time': start_time,
                'end_time': end_time,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            observations = []
            for obs in data.get('observations', []):
                observations.append({
                    'station_id': station_id,
                    'timestamp': obs.get('time'),
                    'parameter': obs.get('parameter'),
                    'value': obs.get('value'),
                    'units': obs.get('units'),
                    'quality': obs.get('quality'),
                    'depth': obs.get('depth')
                })
            
            df = pd.DataFrame(observations)
            
            if not df.empty:
                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO glos_observations 
                        (station_id, timestamp, parameter, value, units, quality, depth)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['station_id'],
                        row['timestamp'],
                        row['parameter'],
                        row['value'],
                        row['units'],
                        row['quality'],
                        row['depth']
                    ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Stored {len(df)} GLOS observations for {station_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch GLOS data for {station_id}: {e}")
            return pd.DataFrame()

class DrifterMLTrainer:
    """ML model trainer for drifter trajectory prediction"""
    
    def __init__(self, db_path: str = 'drift_objects.db'):
        self.db_path = db_path
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Try to import ML libraries
        try:
            import tensorflow as tf
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            import joblib
            
            self.tf = tf
            self.RandomForestRegressor = RandomForestRegressor
            self.StandardScaler = StandardScaler
            self.train_test_split = train_test_split
            self.joblib = joblib
            self.ml_available = True
            
        except ImportError as e:
            logger.warning(f"ML libraries not available: {e}")
            self.ml_available = False
    
    def prepare_training_data(self, min_trajectory_length: int = 24) -> pd.DataFrame:
        """Prepare training data from collected drifter trajectories"""
        conn = sqlite3.connect(self.db_path)
        
        # Get GDP trajectory data
        query = """
        SELECT 
            t.drifter_id,
            t.timestamp,
            t.latitude,
            t.longitude,
            t.u_velocity,
            t.v_velocity,
            t.temperature,
            t.drogue_status,
            d.deploy_lat,
            d.deploy_lon
        FROM gdp_trajectories t
        JOIN gdp_drifters d ON t.drifter_id = d.drifter_id
        WHERE t.position_quality <= 2 AND t.velocity_quality <= 2
        ORDER BY t.drifter_id, t.timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("No GDP trajectory data available for training")
            return pd.DataFrame()
        
        # Filter trajectories by minimum length
        trajectory_lengths = df.groupby('drifter_id').size()
        valid_drifters = trajectory_lengths[trajectory_lengths >= min_trajectory_length].index
        df = df[df['drifter_id'].isin(valid_drifters)]
        
        logger.info(f"Prepared {len(df)} trajectory points from {len(valid_drifters)} drifters")
        return df
    
    def create_training_sequences(self, df: pd.DataFrame, 
                                 sequence_length: int = 12,
                                 prediction_horizon: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and prediction targets"""
        if not self.ml_available:
            logger.error("ML libraries not available")
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for drifter_id in df['drifter_id'].unique():
            drifter_data = df[df['drifter_id'] == drifter_id].sort_values('timestamp')
            
            if len(drifter_data) < sequence_length + prediction_horizon:
                continue
            
            # Create sequences
            for i in range(len(drifter_data) - sequence_length - prediction_horizon + 1):
                # Input sequence (past observations)
                input_seq = drifter_data.iloc[i:i+sequence_length]
                
                # Prediction target (future positions)
                target_seq = drifter_data.iloc[i+sequence_length:i+sequence_length+prediction_horizon]
                
                # Features: lat, lon, u_vel, v_vel, temp, drogue_status
                features = input_seq[['latitude', 'longitude', 'u_velocity', 'v_velocity', 
                                    'temperature', 'drogue_status']].values
                
                # Targets: future lat, lon
                targets = target_seq[['latitude', 'longitude']].values
                
                # Handle missing values
                if not (np.isnan(features).any() or np.isnan(targets).any()):
                    X.append(features)
                    y.append(targets)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} training sequences")
        return X, y
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train LSTM model for trajectory prediction"""
        if not self.ml_available:
            logger.error("TensorFlow not available")
            return False
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build LSTM model
            model = self.tf.keras.Sequential([
                self.tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                self.tf.keras.layers.Dropout(0.2),
                self.tf.keras.layers.LSTM(32, return_sequences=False),
                self.tf.keras.layers.Dropout(0.2),
                self.tf.keras.layers.Dense(32, activation='relu'),
                self.tf.keras.layers.Dense(y.shape[1] * y.shape[2])  # Flatten output
            ])
            
            # Reshape y for training
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            y_test_flat = y_test.reshape(y_test.shape[0], -1)
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train_flat,
                validation_data=(X_test, y_test_flat),
                epochs=50,
                batch_size=32,
                verbose=1,
                callbacks=[
                    self.tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            # Save model
            model_path = self.models_dir / 'drifter_lstm_model.h5'
            model.save(str(model_path))
            
            # Save training metrics
            metrics = {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_mae': float(history.history['mae'][-1]),
                'final_val_mae': float(history.history['val_mae'][-1]),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            metrics_path = self.models_dir / 'lstm_training_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"LSTM model trained and saved to {model_path}")
            logger.info(f"Final validation loss: {metrics['final_val_loss']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train LSTM model: {e}")
            return False
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train Random Forest model for comparison"""
        if not self.ml_available:
            logger.error("Scikit-learn not available")
            return False
        
        try:
            # Flatten input sequences for Random Forest
            X_flat = X.reshape(X.shape[0], -1)
            y_flat = y.reshape(y.shape[0], -1)
            
            # Split data
            X_train, X_test, y_train, y_test = self.train_test_split(
                X_flat, y_flat, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = self.StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = self.RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            # Save model and scaler
            model_path = self.models_dir / 'drifter_random_forest.pkl'
            scaler_path = self.models_dir / 'rf_scaler.pkl'
            
            self.joblib.dump(rf_model, model_path)
            self.joblib.dump(scaler, scaler_path)
            
            # Save metrics
            metrics = {
                'train_score': float(train_score),
                'test_score': float(test_score),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            metrics_path = self.models_dir / 'rf_training_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Random Forest model trained and saved to {model_path}")
            logger.info(f"Test R² score: {test_score:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Random Forest model: {e}")
            return False

class DrifterTrainingPipeline:
    """Complete pipeline for drifter data collection and ML training"""
    
    def __init__(self, db_path: str = 'drift_objects.db'):
        self.db_path = db_path
        
        # Initialize collectors
        self.gdp_collector = NOAAGDPCollector(db_path)
        self.ndbc_collector = NDBCCollector(db_path)
        self.glos_collector = GLOSSeagullCollector(db_path)
        self.ml_trainer = DrifterMLTrainer(db_path)
        
        logger.info("Drifter training pipeline initialized")
    
    def collect_all_data(self, max_workers: int = 5) -> Dict:
        """Collect data from all sources in parallel"""
        logger.info("Starting comprehensive drifter data collection...")
        
        results = {
            'gdp_drifters': 0,
            'gdp_trajectories': 0,
            'ndbc_buoys': 0,
            'ndbc_observations': 0,
            'glos_observations': 0,
            'errors': []
        }
        
        # Step 1: Collect GDP active drifters
        try:
            active_drifters = self.gdp_collector.fetch_active_drifters()
            results['gdp_drifters'] = len(active_drifters)
            
            # Collect trajectories for a subset (to avoid overwhelming the system)
            logger.info("Collecting trajectories for recent drifters...")
            
            # Get recently deployed drifters
            recent_drifters = [d for d in active_drifters 
                             if d.get('deploy_date') and 
                             datetime.strptime(d['deploy_date'], '%Y-%m-%d') > 
                             datetime.now() - timedelta(days=365)][:50]  # Limit to 50
            
            trajectory_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_drifter = {
                    executor.submit(
                        self.gdp_collector.fetch_drifter_trajectory, 
                        drifter['id'],
                        (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d')
                    ): drifter['id'] 
                    for drifter in recent_drifters
                }
                
                for future in as_completed(future_to_drifter):
                    drifter_id = future_to_drifter[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            trajectory_count += len(df)
                    except Exception as e:
                        results['errors'].append(f"GDP trajectory {drifter_id}: {e}")
            
            results['gdp_trajectories'] = trajectory_count
            
        except Exception as e:
            results['errors'].append(f"GDP collection: {e}")
        
        # Step 2: Collect NDBC data
        try:
            buoy_metadata = self.ndbc_collector.fetch_buoy_metadata()
            results['ndbc_buoys'] = len(buoy_metadata)
            
            # Collect recent data for Great Lakes buoys
            observation_count = 0
            current_year = datetime.now().year
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_buoy = {
                    executor.submit(
                        self.ndbc_collector.fetch_buoy_data, 
                        buoy_id, 
                        current_year
                    ): buoy_id 
                    for buoy_id in buoy_metadata.keys()
                }
                
                for future in as_completed(future_to_buoy):
                    buoy_id = future_to_buoy[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            observation_count += len(df)
                    except Exception as e:
                        results['errors'].append(f"NDBC buoy {buoy_id}: {e}")
            
            results['ndbc_observations'] = observation_count
            
        except Exception as e:
            results['errors'].append(f"NDBC collection: {e}")
        
        # Step 3: Collect GLOS data
        try:
            glos_count = 0
            end_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            start_time = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            for station_id in self.glos_collector.stations.keys():
                try:
                    df = self.glos_collector.fetch_station_data(station_id, start_time, end_time)
                    if not df.empty:
                        glos_count += len(df)
                except Exception as e:
                    results['errors'].append(f"GLOS station {station_id}: {e}")
            
            results['glos_observations'] = glos_count
            
        except Exception as e:
            results['errors'].append(f"GLOS collection: {e}")
        
        logger.info("Data collection completed:")
        logger.info(f"  GDP drifters: {results['gdp_drifters']}")
        logger.info(f"  GDP trajectories: {results['gdp_trajectories']} points")
        logger.info(f"  NDBC buoys: {results['ndbc_buoys']}")
        logger.info(f"  NDBC observations: {results['ndbc_observations']} points")
        logger.info(f"  GLOS observations: {results['glos_observations']} points")
        if results['errors']:
            logger.warning(f"  Errors: {len(results['errors'])}")
        
        return results
    
    def train_all_models(self) -> Dict:
        """Train all ML models on collected data"""
        logger.info("Starting ML model training...")
        
        results = {
            'data_prepared': False,
            'lstm_trained': False,
            'rf_trained': False,
            'training_samples': 0,
            'errors': []
        }
        
        try:
            # Prepare training data
            df = self.ml_trainer.prepare_training_data(min_trajectory_length=24)
            
            if df.empty:
                results['errors'].append("No training data available")
                return results
            
            results['data_prepared'] = True
            
            # Create training sequences
            X, y = self.ml_trainer.create_training_sequences(df)
            
            if len(X) == 0:
                results['errors'].append("Failed to create training sequences")
                return results
            
            results['training_samples'] = len(X)
            
            # Train LSTM model
            try:
                if self.ml_trainer.train_lstm_model(X, y):
                    results['lstm_trained'] = True
                    logger.info("✓ LSTM model training completed")
                else:
                    results['errors'].append("LSTM training failed")
            except Exception as e:
                results['errors'].append(f"LSTM training error: {e}")
            
            # Train Random Forest model
            try:
                if self.ml_trainer.train_random_forest(X, y):
                    results['rf_trained'] = True
                    logger.info("✓ Random Forest model training completed")
                else:
                    results['errors'].append("Random Forest training failed")
            except Exception as e:
                results['errors'].append(f"Random Forest training error: {e}")
            
        except Exception as e:
            results['errors'].append(f"Training pipeline error: {e}")
        
        return results
    
    def run_complete_pipeline(self) -> Dict:
        """Run complete data collection and training pipeline"""
        logger.info("Starting complete drifter training pipeline...")
        
        # Step 1: Data collection
        collection_results = self.collect_all_data()
        
        # Step 2: Model training
        training_results = self.train_all_models()
        
        # Combine results
        complete_results = {
            'collection': collection_results,
            'training': training_results,
            'pipeline_success': (
                collection_results['gdp_trajectories'] > 0 and
                (training_results['lstm_trained'] or training_results['rf_trained'])
            )
        }
        
        # Generate summary report
        self._generate_pipeline_report(complete_results)
        
        return complete_results
    
    def _generate_pipeline_report(self, results: Dict):
        """Generate comprehensive pipeline report"""
        report_lines = [
            "DRIFTER ML TRAINING PIPELINE REPORT",
            "=" * 50,
            f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATA COLLECTION SUMMARY:",
            f"  GDP Drifters: {results['collection']['gdp_drifters']}",
            f"  GDP Trajectory Points: {results['collection']['gdp_trajectories']}",
            f"  NDBC Buoys: {results['collection']['ndbc_buoys']}",
            f"  NDBC Observations: {results['collection']['ndbc_observations']}",
            f"  GLOS Observations: {results['collection']['glos_observations']}",
            "",
            "MODEL TRAINING SUMMARY:",
            f"  Training Data Prepared: {'✓' if results['training']['data_prepared'] else '✗'}",
            f"  Training Samples: {results['training']['training_samples']}",
            f"  LSTM Model: {'✓ Trained' if results['training']['lstm_trained'] else '✗ Failed'}",
            f"  Random Forest: {'✓ Trained' if results['training']['rf_trained'] else '✗ Failed'}",
            "",
            "OVERALL STATUS:",
            f"  Pipeline Success: {'✓ COMPLETE' if results['pipeline_success'] else '✗ INCOMPLETE'}",
            ""
        ]
        
        # Add errors if any
        all_errors = results['collection']['errors'] + results['training']['errors']
        if all_errors:
            report_lines.extend([
                "ERRORS ENCOUNTERED:",
                *[f"  - {error}" for error in all_errors[:10]],  # Show first 10 errors
                f"  ... and {len(all_errors) - 10} more" if len(all_errors) > 10 else "",
                ""
            ])
        
        report_lines.extend([
            "NEXT STEPS:",
            "  1. Check model files in models/ directory",
            "  2. Validate model performance with test data",
            "  3. Integrate models into SAROPS prediction system",
            "  4. Monitor real-time prediction accuracy",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = Path('drifter_training_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Print to console
        print("\n" + report_text)
        
        logger.info(f"Pipeline report saved to {report_path}")

def main():
    """Main execution function"""
    print("CESAROPS Drifter Data Collection and ML Training System")
    print("=" * 60)
    print("This will collect data from multiple sources and train ML models:")
    print("• NOAA Global Drifter Program (1500+ active drifters)")
    print("• NDBC Great Lakes buoy network")  
    print("• GLOS Seagull real-time observations")
    print("• Train LSTM and Random Forest models")
    print("")
    
    # Check if user wants to proceed
    response = input("Start comprehensive data collection and training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Initialize and run pipeline
    pipeline = DrifterTrainingPipeline()
    
    start_time = time.time()
    results = pipeline.run_complete_pipeline()
    end_time = time.time()
    
    print(f"\nPipeline completed in {end_time - start_time:.1f} seconds")
    
    if results['pipeline_success']:
        print("🎉 Training pipeline completed successfully!")
        print("✓ Models saved to models/ directory")
        print("✓ Training data stored in database")
        print("✓ Ready for integration with SAROPS")
    else:
        print("⚠️ Pipeline completed with some issues")
        print("Check drifter_training.log for detailed error information")
    
    return results

if __name__ == "__main__":
    main()