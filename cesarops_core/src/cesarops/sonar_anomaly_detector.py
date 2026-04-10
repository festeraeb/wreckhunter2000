"""
SonarAnomalyDetector - Phase 11.1 Anomaly Detection Module

Detect anomalies in sonar data that indicate potential targets using ML.
Uses Isolation Forest for unsupervised anomaly detection.

Author: CESARops Development Team
Date: January 2026
Status: Phase 11.1 Implementation
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SonarAnomalyDetector:
    """
    Detect anomalies in sonar returns using Machine Learning.
    
    Workflow:
    1. Train on baseline (clean water with no targets)
    2. Apply to new data to find anomalies
    3. Export results as KML or JSON
    
    Attributes:
        detector: IsolationForest ML model
        scaler: StandardScaler for feature normalization
        trained: Whether model has been trained
        baseline_stats: Statistics from baseline data
        
    Example:
        >>> detector = SonarAnomalyDetector()
        >>> detector.train_on_baseline(baseline_pings)
        >>> results = detector.detect_anomalies(new_pings)
        >>> print(f"Found {len(results['anomalies'])} anomalies")
    """
    
    def __init__(self, contamination_factor: float = 0.05, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination_factor: Expected fraction of anomalies (5% default)
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If contamination_factor not in (0, 1)
        """
        if not (0 < contamination_factor < 1):
            raise ValueError("contamination_factor must be between 0 and 1")
            
        self.detector = IsolationForest(
            contamination=contamination_factor,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1  # Use all available cores
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.baseline_stats = None
        logger.info(f"SonarAnomalyDetector initialized (contamination={contamination_factor})")
        
    def train_on_baseline(self, baseline_pings: np.ndarray) -> 'SonarAnomalyDetector':
        """
        Train detector on clean sonar data (baseline/background).
        
        This establishes what "normal" sonar looks like, so anomalies
        can be detected by deviation from this baseline.
        
        Args:
            baseline_pings: Array of shape (n_pings, n_samples)
                Each row is one sonar ping from clean water
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If baseline_pings shape invalid or has NaN values
            
        Example:
            >>> baseline = np.random.randn(100, 512)  # 100 pings, 512 samples each
            >>> detector.train_on_baseline(baseline)
            >>> assert detector.trained
        """
        baseline_pings = np.asarray(baseline_pings)
        
        # Validate input
        if baseline_pings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {baseline_pings.shape}")
        if np.isnan(baseline_pings).any():
            logger.warning("Baseline contains NaN values, will be replaced with 0")
            baseline_pings = np.nan_to_num(baseline_pings, nan=0.0)
        if len(baseline_pings) < 10:
            raise ValueError(f"Need at least 10 baseline pings, got {len(baseline_pings)}")
        
        logger.info(f"Training on {len(baseline_pings)} baseline pings "
                   f"(shape: {baseline_pings.shape})")
        
        try:
            # Extract features from baseline
            features = self._extract_features(baseline_pings)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train isolation forest
            self.detector.fit(features_scaled)
            self.trained = True
            
            # Store baseline statistics for reference
            self.baseline_stats = {
                'num_pings': int(len(baseline_pings)),
                'num_samples_per_ping': int(baseline_pings.shape[1]),
                'mean_amplitude': float(np.mean(baseline_pings)),
                'std_amplitude': float(np.std(baseline_pings)),
                'min_amplitude': float(np.min(baseline_pings)),
                'max_amplitude': float(np.max(baseline_pings)),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"✓ Training complete. Model ready for predictions.")
            logger.debug(f"Baseline stats: {self.baseline_stats}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.trained = False
            raise
            
        return self
    
    def detect_anomalies(self, sonar_pings: np.ndarray) -> Dict:
        """
        Detect anomalies in new sonar data.
        
        Args:
            sonar_pings: Array of shape (n_pings, n_samples)
        
        Returns:
            Dictionary with keys:
                - 'anomaly_scores': scores for each ping (-1 to 1)
                - 'anomalies': list of detected anomalies with details
                - 'num_anomalies': count of anomalies found
                - 'anomaly_density': percentage of anomalies
                - 'timestamp': when detection was run
                - 'status': 'success' or error message
        
        Raises:
            ValueError: If not trained or invalid input
            
        Example:
            >>> new_pings = np.random.randn(50, 512)
            >>> results = detector.detect_anomalies(new_pings)
            >>> for anom in results['anomalies'][:5]:
            ...     print(f"Ping {anom['ping_idx']}: {anom['type']}")
        """
        if not self.trained:
            raise ValueError("Must call train_on_baseline() first!")
        
        sonar_pings = np.asarray(sonar_pings)
        
        # Validate input
        if sonar_pings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {sonar_pings.shape}")
        if sonar_pings.shape[1] != self.baseline_stats['num_samples_per_ping']:
            logger.warning(f"Ping width mismatch: expected "
                          f"{self.baseline_stats['num_samples_per_ping']}, "
                          f"got {sonar_pings.shape[1]}")
        if np.isnan(sonar_pings).any():
            logger.warning("Input contains NaN values, replacing with 0")
            sonar_pings = np.nan_to_num(sonar_pings, nan=0.0)
        
        try:
            logger.debug(f"Detecting anomalies in {len(sonar_pings)} pings")
            
            # Extract features
            features = self._extract_features(sonar_pings)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            # -1 = anomaly, 1 = normal
            predictions = self.detector.predict(features_scaled)
            anomaly_scores = self.detector.score_samples(features_scaled)
            
            # Identify anomalies
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1:  # Is anomaly
                    anomaly_type = self._classify_anomaly(sonar_pings[i])
                    # Normalize score to 0-1 confidence
                    confidence = float(1 - (score + 1) / 2)
                    
                    anomalies.append({
                        'ping_idx': int(i),
                        'score': float(score),
                        'type': anomaly_type,
                        'confidence': confidence,
                    })
            
            # Sort by confidence (highest first)
            anomalies = sorted(anomalies, key=lambda x: x['confidence'], reverse=True)
            
            result = {
                'anomaly_scores': anomaly_scores.tolist(),
                'anomalies': anomalies,
                'num_anomalies': len(anomalies),
                'anomaly_density': len(anomalies) / len(sonar_pings),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"✓ Detection complete: {len(anomalies)} anomalies found "
                       f"({result['anomaly_density']:.1%} density)")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'anomaly_scores': [],
                'anomalies': [],
                'num_anomalies': 0,
                'anomaly_density': 0,
                'timestamp': datetime.now().isoformat(),
                'status': f'error: {str(e)}'
            }
    
    def _extract_features(self, pings: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from sonar pings.
        
        Features extracted (10 per ping):
        - Mean (average amplitude)
        - Std (variability)
        - Max/Min (extreme values)
        - Range (max - min)
        - Energy (sum of squares)
        - Entropy (randomness/structure)
        - RMS (root mean square)
        - Skewness (asymmetry)
        - Kurtosis (sharpness of peaks)
        
        These features capture different aspects of the sonar signal
        that can distinguish targets from background noise.
        
        Args:
            pings: Array of shape (n_pings, n_samples)
            
        Returns:
            Array of shape (n_pings, 10) with extracted features
        """
        features = []
        
        for i, ping in enumerate(pings):
            try:
                # Basic statistics
                mean_val = float(np.mean(ping))
                std_val = float(np.std(ping))
                max_val = float(np.max(ping))
                min_val = float(np.min(ping))
                
                # Derived statistics
                range_val = max_val - min_val
                energy = float(np.sum(ping ** 2))
                entropy = self._entropy(ping)
                rms = float(np.sqrt(np.mean(ping ** 2)))
                
                # Higher-order statistics
                if std_val > 1e-10:
                    skewness = float(np.mean((ping - mean_val) ** 3) / (std_val ** 3))
                    kurtosis = float(np.mean((ping - mean_val) ** 4) / (std_val ** 4))
                else:
                    skewness = 0.0
                    kurtosis = 0.0
                
                # Package all features
                feature_vector = [
                    mean_val, std_val, max_val, min_val, range_val,
                    energy, entropy, rms, skewness, kurtosis
                ]
                features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error extracting features from ping {i}: {e}")
                # Append zeros if extraction fails
                features.append([0.0] * 10)
        
        return np.array(features)
    
    @staticmethod
    def _entropy(signal_data: np.ndarray) -> float:
        """
        Calculate Shannon entropy (measure of randomness).
        
        High entropy = noisy/random signal
        Low entropy = structured/organized signal
        
        Targets often show lower entropy (more structured returns)
        
        Args:
            signal_data: 1D array of signal values
            
        Returns:
            Float entropy value (typically 0-10)
        """
        # Bin the signal to create a histogram for discrete entropy
        # Use 10 bins for entropy calculation
        if len(signal_data) < 2:
            return 0.0
            
        sig_min = float(np.min(signal_data))
        sig_max = float(np.max(signal_data))
        
        # Handle constant signal
        if sig_max == sig_min:
            return 0.0
        
        hist, _ = np.histogram(signal_data, bins=10, range=(sig_min, sig_max + 1e-10))
        hist = hist[hist > 0]  # Only non-zero bins
        
        if len(hist) == 0:
            return 0.0
            
        # Normalize to probability distribution
        prob = hist / np.sum(hist)
        # Calculate Shannon entropy
        entropy_val = -np.sum(prob * np.log2(prob + 1e-10))
        return float(entropy_val)
    
    def _classify_anomaly(self, ping: np.ndarray) -> str:
        """
        Classify type of anomaly detected.
        
        Returns one of:
        - 'strong_return': Hard object (boat, rock, wreck)
        - 'structured_pattern': Organized pattern
        - 'persistent_feature': Sustained signal
        - 'unusual_pattern': Something else unusual
        
        Args:
            ping: 1D array of sonar values
            
        Returns:
            String classification of anomaly type
        """
        mean_val = np.mean(ping)
        std_val = np.std(ping)
        max_val = np.max(ping)
        min_val = np.min(ping)
        dynamic_range = max_val - min_val
        
        # Check for strong localized return
        if dynamic_range > 0:
            # Normalize to 0-1 range
            ping_norm = (ping - min_val) / dynamic_range
            # If one sample is significantly higher than average
            if np.max(ping_norm) > 0.8 and np.mean(ping_norm) < 0.3:
                return "strong_return"
        
        # Check for structured pattern (monotonic or smooth variation)
        diffs = np.abs(np.diff(ping))
        if len(diffs) > 0:
            avg_diff = np.mean(diffs)
            if std_val > 0 and avg_diff > std_val * 0.5:  # Smooth changes
                return "structured_pattern"
        
        # Check for persistent feature (sustained elevation)
        if mean_val > 1e-10 and np.sum(ping > mean_val * 0.8) > 0.6 * len(ping):
            return "persistent_feature"
        
        return "unusual_pattern"
    
    def export_detections_to_json(self, results: Dict, filepath: str) -> bool:
        """
        Export detections as JSON for sharing/archival.
        
        Args:
            results: Results dictionary from detect_anomalies()
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Exported {len(results['anomalies'])} detections to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def export_detections_to_kml(self, results: Dict, trackpoints: List[Dict], 
                                  filepath: str) -> bool:
        """
        Export detected anomalies as KML with placemarks.
        
        Args:
            results: Results dictionary from detect_anomalies()
            trackpoints: List of GPS positions [{'lat': float, 'lon': float}, ...]
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Sonar Anomalies</name>
    <description>Anomaly detections from SonarAnomalyDetector</description>
    <Style id="anomaly_style">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href>
        </Icon>
      </IconStyle>
    </Style>
"""
            
            for anom in results['anomalies']:
                ping_idx = anom['ping_idx']
                if ping_idx < len(trackpoints):
                    lat = trackpoints[ping_idx].get('lat', 0)
                    lon = trackpoints[ping_idx].get('lon', 0)
                    kml_content += f"""    <Placemark>
      <name>Anomaly {ping_idx}</name>
      <description>Type: {anom['type']}<br/>Confidence: {anom['confidence']:.2%}<br/>Score: {anom['score']:.4f}</description>
      <styleUrl>#anomaly_style</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
"""
            
            kml_content += """  </Document>
</kml>"""
            
            with open(filepath, 'w') as f:
                f.write(kml_content)
            
            logger.info(f"✓ Exported {len(results['anomalies'])} anomalies to KML: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to KML: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'trained': self.trained,
            'baseline_stats': self.baseline_stats,
            'n_estimators': self.detector.n_estimators,
            'contamination': self.detector.contamination,
            'random_state': self.detector.random_state,
        }


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def create_synthetic_baseline(num_pings: int = 100, 
                              ping_width: int = 256,
                              noise_level: float = 20.0) -> np.ndarray:
    """
    Create synthetic baseline data (clean water) for testing.
    
    Args:
        num_pings: Number of pings to generate
        ping_width: Samples per ping
        noise_level: Gaussian noise standard deviation
        
    Returns:
        Array of shape (num_pings, ping_width)
    """
    np.random.seed(42)
    baseline = np.random.normal(loc=100, scale=noise_level, 
                               size=(num_pings, ping_width))
    # Clip to realistic values
    baseline = np.clip(baseline, 0, 255)
    return baseline


def create_synthetic_test_data(num_pings: int = 50,
                               ping_width: int = 256,
                               num_anomalies: int = 5,
                               noise_level: float = 20.0) -> Tuple[np.ndarray, List[int]]:
    """
    Create synthetic test data with injected anomalies.
    
    Args:
        num_pings: Number of pings to generate
        ping_width: Samples per ping
        num_anomalies: Number of anomalies to inject
        noise_level: Gaussian noise standard deviation
        
    Returns:
        Tuple of (test_data array, list of anomaly indices)
    """
    np.random.seed(42)
    test_data = np.random.normal(loc=100, scale=noise_level,
                                size=(num_pings, ping_width))
    test_data = np.clip(test_data, 0, 255)
    
    # Inject anomalies
    anomaly_indices = np.random.choice(num_pings, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        if idx % 3 == 0:
            # Strong return anomaly
            test_data[idx] = np.ones(ping_width) * 200
        elif idx % 3 == 1:
            # Structured pattern
            test_data[idx] = np.linspace(50, 200, ping_width)
        else:
            # Shifted distribution
            test_data[idx] = np.random.normal(150, 10, ping_width)
    
    return test_data, sorted(anomaly_indices.tolist())


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("SonarAnomalyDetector - Example Usage")
    print("=" * 70)
    
    # Create synthetic data
    print("\n1. Generating synthetic baseline data...")
    baseline = create_synthetic_baseline(num_pings=100)
    print(f"   Created baseline: {baseline.shape}")
    
    # Train detector
    print("\n2. Training anomaly detector...")
    detector = SonarAnomalyDetector(contamination_factor=0.1)
    detector.train_on_baseline(baseline)
    print(f"   Baseline stats: {detector.baseline_stats}")
    
    # Generate test data with anomalies
    print("\n3. Generating test data with anomalies...")
    test_data, true_anomalies = create_synthetic_test_data(num_pings=50, 
                                                           num_anomalies=5)
    print(f"   Created test data: {test_data.shape}")
    print(f"   Injected anomalies at indices: {true_anomalies}")
    
    # Run detection
    print("\n4. Running anomaly detection...")
    results = detector.detect_anomalies(test_data)
    
    # Display results
    print(f"\n5. Results:")
    print(f"   Total anomalies found: {results['num_anomalies']}")
    print(f"   Anomaly density: {results['anomaly_density']:.1%}")
    print(f"\n   Detailed findings:")
    for anom in results['anomalies']:
        print(f"      Ping {anom['ping_idx']:2d}: {anom['type']:20s} "
              f"(confidence: {anom['confidence']:5.1%})")
    
    # Export
    print("\n6. Exporting results...")
    detector.export_detections_to_json(results, "test_anomalies.json")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
