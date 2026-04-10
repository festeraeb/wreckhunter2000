"""
CESARops - Great Lakes Search and Rescue Drift Modeling System

A comprehensive drift modeling system combining OpenDrift Lagrangian particle tracking
with machine learning enhancements for offline-capable SAR operations across all five
Great Lakes (Michigan, Erie, Huron, Ontario, Superior).

Author: CESAROPS Development Team
Version: 2.0.0-beta
License: Proprietary
"""

__version__ = "2.0.0-beta"
__author__ = "CESAROPS Development Team"
__license__ = "Proprietary"

# Core ML and drift correction modules
try:
    from .ml_pipeline import DriftCorrectionPipeline, EnhancedDriftSimulator

    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False

# Geospatial export modules
try:
    from .geospatial_export import (
        GeoTIFFGenerator,
        KMLNetworkLinkGenerator,
        DriftSimulationExporter,
    )

    GEOSPATIAL_EXPORT_AVAILABLE = True
except ImportError:
    GEOSPATIAL_EXPORT_AVAILABLE = False

# Incremental loading for memory efficiency
try:
    from .incremental_loading import (
        StreamingDataLoader,
        ChunkedDataProcessor,
        ProgressiveDataLoader,
    )

    INCREMENTAL_LOADING_AVAILABLE = True
except ImportError:
    INCREMENTAL_LOADING_AVAILABLE = False

# Phase 13: Advanced Sonar Intelligence Analysis
try:
    from .sonar_classification import (
        SignalType,
        SonarSignalFeatures,
        SonarSignalFeatureExtractor,
        SonarSignalClassifier,
    )

    SONAR_CLASSIFICATION_AVAILABLE = True
except ImportError:
    SONAR_CLASSIFICATION_AVAILABLE = False

try:
    from .sensor_fusion import (
        SensorType,
        CoordinateSystem,
        Location,
        SensorReading,
        FusedData,
        CorrelationResult,
        EnvironmentalContext,
        BaseSensor,
        SensorDataBus,
        SensorQualityEstimator,
        CoordinateSystemManager,
        FusionEngine,
    )

    SENSOR_FUSION_AVAILABLE = True
except ImportError:
    SENSOR_FUSION_AVAILABLE = False

try:
    from .contextual_anomaly_detection import (
        RiskLevel,
        AnomalyType,
        Pattern,
        Explanation,
        ContextualAnomaly,
        AnomalyPatternLearner,
        AnomalyExplainer,
        ContextualAnomalyDetector,
    )

    CONTEXTUAL_ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    CONTEXTUAL_ANOMALY_DETECTION_AVAILABLE = False

__all__ = [
    # Core ML and drift correction
    "DriftCorrectionPipeline",
    "EnhancedDriftSimulator",
    # Geospatial export
    "GeoTIFFGenerator",
    "KMLNetworkLinkGenerator",
    "DriftSimulationExporter",
    # Incremental loading
    "StreamingDataLoader",
    "ChunkedDataProcessor",
    "ProgressiveDataLoader",
    # Phase 13: Sonar Classification
    "SignalType",
    "SonarSignalFeatures",
    "SonarSignalFeatureExtractor",
    "SonarSignalClassifier",
    # Phase 13: Sensor Fusion
    "SensorType",
    "CoordinateSystem",
    "Location",
    "SensorReading",
    "FusedData",
    "CorrelationResult",
    "EnvironmentalContext",
    "BaseSensor",
    "SensorDataBus",
    "SensorQualityEstimator",
    "CoordinateSystemManager",
    "FusionEngine",
    # Phase 13: Contextual Anomaly Detection
    "RiskLevel",
    "AnomalyType",
    "Pattern",
    "Explanation",
    "ContextualAnomaly",
    "AnomalyPatternLearner",
    "AnomalyExplainer",
    "ContextualAnomalyDetector",
    # Feature flags
    "ML_PIPELINE_AVAILABLE",
    "GEOSPATIAL_EXPORT_AVAILABLE",
    "INCREMENTAL_LOADING_AVAILABLE",
    "SONAR_CLASSIFICATION_AVAILABLE",
    "SENSOR_FUSION_AVAILABLE",
    "CONTEXTUAL_ANOMALY_DETECTION_AVAILABLE",
]
