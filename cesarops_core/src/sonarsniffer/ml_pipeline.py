"""
Machine Learning Pipeline for CESAROPS Drift Correction
Implements real-time drift prediction with incremental learning and feedback loops
"""

import numpy as np
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import threading
import joblib

logger = logging.getLogger(__name__)


class DriftCorrectionPipeline:
    """Production ML pipeline for drift correction with online learning"""

    def __init__(self, great_lake: str = "Michigan", model_dir: str = "models"):
        """
        Initialize drift correction pipeline for Great Lakes

        Args:
            great_lake: Lake name (Michigan, Erie, Huron, Ontario, Superior)
            model_dir: Directory to store trained models
        """
        self.lake = great_lake
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Models per lake
        self.u_velocity_model = None  # U component (east-west)
        self.v_velocity_model = None  # V component (north-south)
        self.scaler = StandardScaler()

        # Online learning buffer
        self.training_buffer = deque(maxlen=100)  # 100 samples before retrain
        self.lock = threading.Lock()

        # Model metadata
        self.model_version = 1
        self.last_retraining = None
        self.prediction_count = 0
        self.performance_metrics = {
            "mean_absolute_error": 0.0,
            "rmse": 0.0,
            "predictions_since_retrain": 0,
        }

        logger.info(f"Initialized ML Pipeline for {great_lake}")

    def load_or_train_models(self, training_data_file: Optional[str] = None):
        """Load existing models or train new ones"""
        model_path = self.model_dir / f"{self.lake.lower()}_u_model.pkl"

        if model_path.exists():
            self._load_models()
            logger.info(f"Loaded existing models for {self.lake}")
        elif training_data_file:
            self._train_models(training_data_file)
            logger.info(f"Trained new models for {self.lake}")
        else:
            self._initialize_placeholder_models()
            logger.warning(
                f"No training data. Using placeholder models for {self.lake}"
            )

    def _load_models(self):
        """Load pre-trained models from disk"""
        u_path = self.model_dir / f"{self.lake.lower()}_u_model.pkl"
        v_path = self.model_dir / f"{self.lake.lower()}_v_model.pkl"
        scaler_path = self.model_dir / f"{self.lake.lower()}_scaler.pkl"

        try:
            self.u_velocity_model = joblib.load(u_path)
            self.v_velocity_model = joblib.load(v_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Successfully loaded models from {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._initialize_placeholder_models()

    def _initialize_placeholder_models(self):
        """Initialize placeholder models (empirical fallback)"""
        self.u_velocity_model = RandomForestRegressor(
            n_estimators=10, random_state=42, max_depth=5
        )
        self.v_velocity_model = RandomForestRegressor(
            n_estimators=10, random_state=42, max_depth=5
        )

        # Fit with dummy data to initialize
        dummy_X = np.array(
            [[0, 0, 0, 0, 0]]
        )  # wind_u, wind_v, depth, latitude, longitude
        dummy_y = np.array([0])

        self.u_velocity_model.fit(dummy_X, dummy_y)
        self.v_velocity_model.fit(dummy_X, dummy_y)
        self.scaler.fit(dummy_X)

        logger.info("Initialized placeholder models")

    def _train_models(self, training_data_file: str):
        """Train models on historical data"""
        try:
            with open(training_data_file, "r") as f:
                data = json.load(f)

            X = np.array(data["features"])  # [wind_u, wind_v, depth, lat, lon]
            u_velocities = np.array(data["u_velocities"])
            v_velocities = np.array(data["v_velocities"])

            # Normalize features
            X_scaled = self.scaler.fit_transform(X)

            # Train models
            self.u_velocity_model = RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )
            self.v_velocity_model = RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )

            self.u_velocity_model.fit(X_scaled, u_velocities)
            self.v_velocity_model.fit(X_scaled, v_velocities)

            # Validate
            u_score = cross_val_score(
                self.u_velocity_model, X_scaled, u_velocities, cv=5, scoring="r2"
            )
            v_score = cross_val_score(
                self.v_velocity_model, X_scaled, v_velocities, cv=5, scoring="r2"
            )

            logger.info(
                f"Model training complete. U R²: {u_score.mean():.3f}, V R²: {v_score.mean():.3f}"
            )

            # Save models
            self._save_models()

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self._initialize_placeholder_models()

    def predict(
        self,
        wind_u: float,
        wind_v: float,
        depth: float,
        latitude: float,
        longitude: float,
    ) -> Tuple[float, float]:
        """
        Predict drift velocity components

        Args:
            wind_u, wind_v: Wind velocity components (m/s)
            depth: Water depth (m)
            latitude, longitude: Position (degrees)

        Returns:
            (u_velocity, v_velocity) components in m/s
        """
        try:
            features = np.array([[wind_u, wind_v, depth, latitude, longitude]])
            X_scaled = self.scaler.transform(features)

            u_pred = self.u_velocity_model.predict(X_scaled)[0]
            v_pred = self.v_velocity_model.predict(X_scaled)[0]

            self.prediction_count += 1
            return float(u_pred), float(v_pred)

        except Exception as e:
            logger.error(f"Prediction failed: {e}. Using fallback.")
            return 0.0, 0.0

    def add_training_sample(
        self,
        wind_u: float,
        wind_v: float,
        depth: float,
        latitude: float,
        longitude: float,
        actual_u: float,
        actual_v: float,
    ):
        """Add sample to training buffer for online learning"""
        with self.lock:
            self.training_buffer.append(
                {
                    "features": [wind_u, wind_v, depth, latitude, longitude],
                    "u_actual": actual_u,
                    "v_actual": actual_v,
                }
            )

            # Auto-retrain when buffer fills
            if len(self.training_buffer) >= 100:
                self._retrain_from_buffer()

    def _retrain_from_buffer(self):
        """Retrain models incrementally from buffer"""
        try:
            X = np.array([s["features"] for s in self.training_buffer])
            u_actual = np.array([s["u_actual"] for s in self.training_buffer])
            v_actual = np.array([s["v_actual"] for s in self.training_buffer])

            X_scaled = self.scaler.fit_transform(X)

            # Warm-start retrain (update existing models)
            self.u_velocity_model.fit(X_scaled, u_actual)
            self.v_velocity_model.fit(X_scaled, v_actual)

            self.model_version += 1
            self.last_retraining = datetime.now()
            self.training_buffer.clear()

            logger.info(f"Models retrained. Version {self.model_version}")
            self._save_models()

        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    def _save_models(self):
        """Persist models to disk"""
        try:
            u_path = self.model_dir / f"{self.lake.lower()}_u_model.pkl"
            v_path = self.model_dir / f"{self.lake.lower()}_v_model.pkl"
            scaler_path = self.model_dir / f"{self.lake.lower()}_scaler.pkl"
            meta_path = self.model_dir / f"{self.lake.lower()}_metadata.json"

            joblib.dump(self.u_velocity_model, u_path)
            joblib.dump(self.v_velocity_model, v_path)
            joblib.dump(self.scaler, scaler_path)

            metadata = {
                "version": self.model_version,
                "lake": self.lake,
                "last_retraining": str(self.last_retraining),
                "created_date": str(datetime.now()),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Models saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def get_metrics(self) -> Dict:
        """Get current pipeline metrics"""
        return {
            "lake": self.lake,
            "model_version": self.model_version,
            "predictions_since_retrain": self.prediction_count,
            "last_retraining": (
                str(self.last_retraining) if self.last_retraining else "Never"
            ),
            "training_buffer_size": len(self.training_buffer),
            "models_trained": self.u_velocity_model is not None,
        }


class EnhancedDriftSimulator:
    """Drift simulation with ML-enhanced velocity correction"""

    def __init__(self, ml_pipeline: Optional[DriftCorrectionPipeline] = None):
        self.ml_pipeline = ml_pipeline
        self.use_ml = (
            ml_pipeline is not None and ml_pipeline.u_velocity_model is not None
        )

    def get_corrected_velocity(
        self,
        base_u: float,
        base_v: float,
        wind_u: float,
        wind_v: float,
        depth: float,
        latitude: float,
        longitude: float,
    ) -> Tuple[float, float]:
        """
        Get velocity with ML correction applied

        Args:
            base_u, base_v: Base velocity from environment
            wind_u, wind_v: Wind velocity
            depth, latitude, longitude: Position info

        Returns:
            Corrected (u, v) velocities
        """
        if not self.use_ml:
            return base_u, base_v

        try:
            # Get ML-predicted correction
            ml_u, ml_v = self.ml_pipeline.predict(
                wind_u, wind_v, depth, latitude, longitude
            )

            # Apply correction with 70% blend (conservative)
            corrected_u = base_u + 0.7 * ml_u
            corrected_v = base_v + 0.7 * ml_v

            return corrected_u, corrected_v

        except Exception as e:
            logger.warning(f"ML correction failed: {e}. Using base velocity.")
            return base_u, base_v


# Global pipeline instance (one per lake)
_pipelines: Dict[str, DriftCorrectionPipeline] = {}


def get_pipeline(lake: str) -> DriftCorrectionPipeline:
    """Get or create ML pipeline for lake"""
    if lake not in _pipelines:
        _pipelines[lake] = DriftCorrectionPipeline(lake)
        _pipelines[lake].load_or_train_models()
    return _pipelines[lake]
