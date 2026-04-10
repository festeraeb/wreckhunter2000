"""
SonarSniffer - Professional Marine Survey Analysis Software

A comprehensive tool for analyzing marine sonar data with AI-enhanced target detection,
web-based visualization, and professional reporting capabilities.

Author: NautiDog Inc.
Contact: festeraeb@yahoo.com
Version: 1.0.0-beta
"""

__version__ = "1.0.0-beta"
__author__ = "NautiDog Inc."
__email__ = "festeraeb@yahoo.com"
__license__ = "Proprietary"

from .license_manager import LicenseManager
from .sonar_parser import SonarParser
from .web_dashboard_generator import WebDashboardGenerator

# Optional optimization modules
try:
    from .incremental_loading import IncrementalLoader
    INCREMENTAL_LOADING_AVAILABLE = True
except ImportError:
    INCREMENTAL_LOADING_AVAILABLE = False

try:
    from .ml_pipeline import DriftCorrectionModel
    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False

try:
    from .geospatial_export import GeoTIFFTileExporter
    GEOSPATIAL_EXPORT_AVAILABLE = True
except ImportError:
    GEOSPATIAL_EXPORT_AVAILABLE = False

__all__ = [
    "LicenseManager",
    "SonarParser",
    "WebDashboardGenerator",
    "IncrementalLoader",
    "DriftCorrectionModel",
    "GeoTIFFTileExporter",
    "INCREMENTAL_LOADING_AVAILABLE",
    "ML_PIPELINE_AVAILABLE",
    "GEOSPATIAL_EXPORT_AVAILABLE",
]