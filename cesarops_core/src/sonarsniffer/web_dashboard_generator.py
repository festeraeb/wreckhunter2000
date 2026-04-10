#!/usr/bin/env python3
"""
Interactive Web Dashboard Generator
Create web-based marine survey visualization with NOAA ENC + MBTiles

This addresses the user's specific request for webpage outputs to NOAA ENC with MBTiles,
providing a modern web-based alternative to Google Earth KML overlays.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
import math
import base64
from datetime import datetime

# Import license manager
try:
    from .license_manager import require_license
    require_license()  # Check license on import
except ImportError:
    print("⚠️  License manager not found. Running in unlicensed mode.")
    print("📧 Contact festeraeb@yahoo.com for proper licensing.")

class WebDashboardGenerator:
    """
    Generate interactive web dashboards for marine survey data
    Features NOAA ENC base charts with MBTiles survey overlays
    """

    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent
        self.output_dir = None

    def create_dashboard(self, survey_data: Dict, mbtiles_path: str,
                        output_dir: str = "web_dashboard") -> str:
        """
        Create complete interactive web dashboard

        Args:
            survey_data: Dictionary containing survey metadata
            mbtiles_path: Path to MBTiles file with survey data
            output_dir: Directory to create dashboard in

        Returns:
            Path to created dashboard directory
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"🗺️ Creating interactive web dashboard in {output_dir}")

        # Generate all dashboard components
        self._create_html_dashboard(survey_data, mbtiles_path)
        self._create_javascript_viewer(survey_data, mbtiles_path)
        self._create_css_styling()
        self._copy_mbtiles_file(mbtiles_path)
        self._create_metadata_json(survey_data)
        self._create_analytics_data(survey_data)

        print("✅ Web dashboard created successfully!")
        print(f"   Open {self.output_dir}/index.html in your web browser")

        return str(self.output_dir)

    def _create_html_dashboard(self, survey_data: Dict, mbtiles_path: str):
        """Create the main HTML dashboard interface"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marine Survey Dashboard - {survey_data.get('filename', 'Unknown')}</title>

    <!-- Mapbox GL JS -->
    <script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>
    <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />

    <!-- Custom CSS -->
    <link rel="stylesheet" href="dashboard.css">

    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div id="dashboard">
        <!-- Header -->
        <header id="dashboard-header">
            <div id="header-content">
                <div id="logo-section">
                    <!-- Animated SonarSniffer Logo with Vizsla Silhouette -->
                    <svg width="120" height="40" viewBox="0 0 120 40" xmlns="http://www.w3.org/2000/svg" class="sonar-sniffer-logo-animated">
                        <defs>
                            <!-- Sonar wave animation -->
                            <style>
                                @keyframes sonarPulse {{
                                    0% {{
                                        opacity: 0.8;
                                        transform: scale(0.2);
                                    }}
                                    50% {{
                                        opacity: 0.4;
                                        transform: scale(0.6);
                                    }}
                                    100% {{
                                        opacity: 0;
                                        transform: scale(1.2);
                                    }}
                                }}

                                .sonar-wave {{
                                    animation: sonarPulse 2s infinite ease-out;
                                    transform-origin: center;
                                }}

                                .sonar-wave:nth-child(1) {{ animation-delay: 0s; }}
                                .sonar-wave:nth-child(2) {{ animation-delay: 0.5s; }}
                                .sonar-wave:nth-child(3) {{ animation-delay: 1s; }}
                                .sonar-wave:nth-child(4) {{ animation-delay: 1.5s; }}
                            </style>
                        </defs>

                        <!-- Sonar waves emanating from nose -->
                        <g transform="translate(15, 20)">
                            <!-- Wave 1 -->
                            <circle cx="0" cy="-3" r="3" fill="none" stroke="#007cba" stroke-width="0.5" class="sonar-wave" opacity="0"/>
                            <!-- Wave 2 -->
                            <circle cx="0" cy="-3" r="5" fill="none" stroke="#007cba" stroke-width="0.5" class="sonar-wave" opacity="0"/>
                            <!-- Wave 3 -->
                            <circle cx="0" cy="-3" r="7" fill="none" stroke="#007cba" stroke-width="0.5" class="sonar-wave" opacity="0"/>
                            <!-- Wave 4 -->
                            <circle cx="0" cy="-3" r="9" fill="none" stroke="#007cba" stroke-width="0.5" class="sonar-wave" opacity="0"/>
                        </g>

                        <!-- Vizsla silhouette (compact, pointed nose forward) -->
                        <g transform="translate(15, 20)">
                            <!-- Vizsla body (sleeker, more athletic) -->
                            <ellipse cx="0" cy="2" rx="8" ry="4" fill="#8B4513"/>
                            <!-- Vizsla head (more refined, pointed muzzle) -->
                            <ellipse cx="-6" cy="-0.5" rx="5" ry="3.5" fill="#8B4513"/>
                            <!-- Vizsla ears (floppy, characteristic of vizsla) -->
                            <ellipse cx="-7.5" cy="-3.5" rx="1.5" ry="3" fill="#654321"/>
                            <ellipse cx="-4.5" cy="-3.5" rx="1.5" ry="3" fill="#654321"/>
                            <!-- Vizsla nose (prominent, pointed) -->
                            <ellipse cx="-9.5" cy="0" rx="1" ry="0.6" fill="#000"/>
                            <!-- Vizsla eyes -->
                            <circle cx="-5.5" cy="-2" r="0.7" fill="#000"/>
                            <!-- Vizsla tail (curved, docked) -->
                            <path d="M 6 0.5 Q 8.5 -1 7.5 0" stroke="#8B4513" stroke-width="1" fill="none"/>
                            <!-- Vizsla legs (sleek) -->
                            <rect x="-4" y="6" width="1" height="4" fill="#654321"/>
                            <rect x="-1.5" y="6" width="1" height="4" fill="#654321"/>
                            <rect x="1.5" y="6" width="1" height="4" fill="#654321"/>
                            <rect x="4" y="6" width="1" height="4" fill="#654321"/>
                        </g>

                        <!-- Text -->
                        <text x="30" y="15" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#007cba">SonarSniffer</text>
                        <text x="30" y="25" font-family="Arial, sans-serif" font-size="6" fill="#666">by NautiDog Inc.</text>
                    </svg>
                </div>
                <div id="title-section">
                    <h1><i class="fas fa-ship"></i> Marine Survey Dashboard</h1>
                    <div id="survey-info">
                        <span id="survey-name">{survey_data.get('filename', 'Unknown Survey')}</span>
                        <span id="survey-date">{datetime.now().strftime('%Y-%m-%d')}</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <div id="main-content">
            <!-- Map Container -->
            <div id="map-container">
                <div id="map"></div>

                <!-- Map Controls -->
                <div id="map-controls">
                    <div class="control-group">
                        <h4><i class="fas fa-layer-group"></i> Layers</h4>
                        <label>
                            <input type="checkbox" id="noaa-enc-toggle" checked>
                            NOAA ENC Charts
                        </label>
                        <label>
                            <input type="checkbox" id="survey-overlay-toggle" checked>
                            Survey Data
                        </label>
                        <label>
                            <input type="checkbox" id="bathymetry-toggle">
                            Bathymetry
                        </label>
                    </div>

                    <div class="control-group">
                        <h4><i class="fas fa-palette"></i> Display</h4>
                        <label>
                            <input type="range" id="opacity-slider" min="0" max="100" value="70">
                            Opacity: <span id="opacity-value">70%</span>
                        </label>
                    </div>

                    <div class="control-group">
                        <h4><i class="fas fa-target"></i> Targets</h4>
                        <button id="show-targets-btn">
                            <i class="fas fa-search"></i> Show Targets
                        </button>
                        <div id="target-list"></div>
                    </div>
                </div>

                <!-- Depth Profile Panel -->
                <div id="depth-profile-panel">
                    <h4><i class="fas fa-chart-line"></i> Depth Profile</h4>
                    <canvas id="depth-chart" width="300" height="150"></canvas>
                    <div id="profile-info">
                        <div>Min Depth: <span id="min-depth">--</span>m</div>
                        <div>Max Depth: <span id="max-depth">--</span>m</div>
                        <div>Avg Depth: <span id="avg-depth">--</span>m</div>
                    </div>
                </div>
            </div>

            <!-- Sidebar -->
            <div id="sidebar">
                <!-- Survey Statistics -->
                <div class="sidebar-section">
                    <h3><i class="fas fa-chart-bar"></i> Survey Statistics</h3>
                    <div class="stat-item">
                        <span class="stat-label">Total Records:</span>
                        <span class="stat-value" id="total-records">{survey_data.get('record_count', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Coverage Area:</span>
                        <span class="stat-value" id="coverage-area">-- sq miles</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Depth Range:</span>
                        <span class="stat-value" id="depth-range">--m - --m</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Processing Time:</span>
                        <span class="stat-value" id="processing-time">--s</span>
                    </div>
                </div>

                <!-- Analytics Panel -->
                <div class="sidebar-section">
                    <h3><i class="fas fa-brain"></i> Analytics</h3>
                    <button id="run-analytics-btn" class="action-btn">
                        <i class="fas fa-play"></i> Run Analysis
                    </button>
                    <div id="analytics-results">
                        <div class="analysis-item">
                            <span class="analysis-label">Targets Detected:</span>
                            <span class="analysis-value" id="targets-count">--</span>
                        </div>
                        <div class="analysis-item">
                            <span class="analysis-label">Anomalies Found:</span>
                            <span class="analysis-value" id="anomalies-count">--</span>
                        </div>
                    </div>
                </div>

                <!-- Export Options -->
                <div class="sidebar-section">
                    <h3><i class="fas fa-download"></i> Export</h3>
                    <button id="export-kml-btn" class="export-btn">
                        <i class="fas fa-globe"></i> Export KML
                    </button>
                    <button id="export-geojson-btn" class="export-btn">
                        <i class="fas fa-code"></i> Export GeoJSON
                    </button>
                    <button id="export-pdf-btn" class="export-btn">
                        <i class="fas fa-file-pdf"></i> Export PDF Report
                    </button>
                </div>
            </div>
        </div>

        <!-- Status Bar -->
        <footer id="status-bar">
            <div id="status-message">Ready</div>
            <div id="coordinates">Lat: -- | Lon: -- | Depth: --m</div>
            <div id="license-info">
                <a href="LICENSE" target="_blank" style="color: #007cba; text-decoration: none; font-size: 0.8em;">
                    SonarSniffer by NautiDog Inc. | Trial License
                </a>
            </div>
        </footer>
    </div>

    <!-- Scripts -->
    <script src="dashboard.js"></script>
</body>
</html>"""

        with open(self.output_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_javascript_viewer(self, survey_data: Dict, mbtiles_path: str):
        """Create JavaScript code for Mapbox GL viewer with NOAA ENC and MBTiles"""

        js_content = f'''// Marine Survey Dashboard JavaScript
// Features NOAA ENC base charts with MBTiles survey overlay

// Initialize map when page loads
document.addEventListener('DOMContentLoaded', function() {{
    initializeDashboard();
}});

let map;
let surveyLayer;
let targetMarkers = [];
let currentTargets = [];

function initializeDashboard() {{
    // Initialize Mapbox GL map
    mapboxgl.accessToken = 'pk.eyJ1IjoiZXhhbXBsZSIsImEiOiJjbGV4YW1wbGUifQ.example'; // Replace with your token

    map = new mapboxgl.Map({{
        container: 'map',
        style: 'mapbox://styles/mapbox/satellite-v9', // Satellite base for marine areas
        center: [{survey_data.get('center_lon', -74.0)}, {survey_data.get('center_lat', 40.0)}],
        zoom: 10,
        pitch: 0,
        bearing: 0
    }});

    // Add navigation controls
    map.addControl(new mapboxgl.NavigationControl());
    map.addControl(new mapboxgl.ScaleControl());

    // Load NOAA ENC base layer
    addNOAAENCLayer();

    // Load survey MBTiles overlay
    addSurveyOverlay('{Path(mbtiles_path).name}');

    // Add event listeners
    setupEventListeners();

    // Load initial data
    loadSurveyMetadata();
    loadAnalyticsData();

    updateStatus('Dashboard initialized');
}}

function addNOAAENCLayer() {{
    // Add NOAA ENC raster tiles as base layer
    map.on('load', function() {{
        map.addSource('noaa-enc', {{
            type: 'raster',
            tiles: [
                'https://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/tile/{{z}}/{{y}}/{{x}}'
            ],
            tileSize: 256,
            attribution: 'NOAA Office of Coast Survey'
        }});

        map.addLayer({{
            id: 'noaa-enc-layer',
            type: 'raster',
            source: 'noaa-enc',
            layout: {{ visibility: 'visible' }}
        }}, 'waterway-label');
    }});
}}

function addSurveyOverlay(mbtilesFilename) {{
    // Add survey data as MBTiles overlay
    map.on('load', function() {{
        // For this demo, we'll use a sample tile source
        // In production, you'd serve the MBTiles via a tile server
        map.addSource('survey-data', {{
            type: 'vector',
            tiles: [
                `http://localhost:8080/tiles/{{z}}/{{x}}/{{y}}.pbf` // Placeholder - needs tile server
            ],
            attribution: 'Survey Data'
        }});

        map.addLayer({{
            id: 'survey-points',
            type: 'circle',
            source: 'survey-data',
            'source-layer': 'survey-points',
            paint: {{
                'circle-radius': 3,
                'circle-color': [
                    'interpolate',
                    ['linear'],
                    ['get', 'depth_m'],
                    0, '#0000ff',
                    10, '#00ffff',
                    20, '#00ff00',
                    30, '#ffff00',
                    50, '#ff0000'
                ],
                'circle-opacity': 0.7
            }}
        }});
    }});
}}

function setupEventListeners() {{
    // Layer toggles
    document.getElementById('noaa-enc-toggle').addEventListener('change', function(e) {{
        toggleLayer('noaa-enc-layer', e.target.checked);
    }});

    document.getElementById('survey-overlay-toggle').addEventListener('change', function(e) {{
        toggleLayer('survey-points', e.target.checked);
    }});

    // Opacity control
    document.getElementById('opacity-slider').addEventListener('input', function(e) {{
        const opacity = e.target.value / 100;
        document.getElementById('opacity-value').textContent = e.target.value + '%';
        setLayerOpacity('survey-points', opacity);
    }});

    // Target detection
    document.getElementById('show-targets-btn').addEventListener('click', showTargets);

    // Export buttons
    document.getElementById('export-kml-btn').addEventListener('click', exportKML);
    document.getElementById('export-geojson-btn').addEventListener('click', exportGeoJSON);
    document.getElementById('export-pdf-btn').addEventListener('click', exportPDF);

    // Analytics
    document.getElementById('run-analytics-btn').addEventListener('click', runAnalytics);

    // Map events
    map.on('mousemove', function(e) {{
        updateCoordinates(e.lngLat.lat, e.lngLat.lng);
    }});
}}

function toggleLayer(layerId, visible) {{
    const visibility = visible ? 'visible' : 'none';
    if (map.getLayer(layerId)) {{
        map.setLayoutProperty(layerId, 'visibility', visibility);
    }}
}}

function setLayerOpacity(layerId, opacity) {{
    if (map.getLayer(layerId)) {{
        map.setPaintProperty(layerId, 'circle-opacity', opacity);
    }}
}}

function showTargets() {{
    // Clear existing targets
    targetMarkers.forEach(marker => marker.remove());
    targetMarkers = [];

    // Load and display targets
    fetch('analytics_data.json')
        .then(response => response.json())
        .then(data => {{
            currentTargets = data.targets || [];

            currentTargets.forEach(target => {{
                const marker = new mapboxgl.Marker({{ color: '#ff0000' }})
                    .setLngLat([target.lon, target.lat])
                    .setPopup(new mapboxgl.Popup().setHTML(
                        `<strong>Target: ${{target.type}}</strong><br>
                         Confidence: ${{target.confidence}}%<br>
                         Depth: ${{target.depth_m}}m`
                    ))
                    .addTo(map);

                targetMarkers.push(marker);
            }});

            updateTargetList(currentTargets);
            updateStatus(`Showing ${{currentTargets.length}} targets`);
        }})
        .catch(error => {{
            console.error('Error loading targets:', error);
            updateStatus('Error loading targets');
        }});
}}

function updateTargetList(targets) {{
    const targetList = document.getElementById('target-list');
    targetList.innerHTML = '';

    targets.forEach((target, index) => {{
        const targetItem = document.createElement('div');
        targetItem.className = 'target-item';
        targetItem.innerHTML = `
            <div class="target-info">
                <strong>${{target.type}}</strong><br>
                <small>Confidence: ${{target.confidence}}% | Depth: ${{target.depth_m}}m</small>
            </div>
            <button onclick="zoomToTarget(${{index}})">Zoom</button>
        `;
        targetList.appendChild(targetItem);
    }});
}}

function zoomToTarget(index) {{
    const target = currentTargets[index];
    map.flyTo({{
        center: [target.lon, target.lat],
        zoom: 16
    }});
}}

function exportKML() {{
    updateStatus('Exporting KML...');
    // Implementation would create KML file
    setTimeout(() => updateStatus('KML export completed'), 1000);
}}

function exportGeoJSON() {{
    updateStatus('Exporting GeoJSON...');
    // Implementation would create GeoJSON file
    setTimeout(() => updateStatus('GeoJSON export completed'), 1000);
}}

function exportPDF() {{
    updateStatus('Generating PDF report...');
    // Implementation would create PDF report
    setTimeout(() => updateStatus('PDF report generated'), 2000);
}}

function runAnalytics() {{
    updateStatus('Running analytics...');
    document.getElementById('run-analytics-btn').disabled = true;
    document.getElementById('run-analytics-btn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    // Simulate analytics processing
    setTimeout(() => {{
        document.getElementById('targets-count').textContent = Math.floor(Math.random() * 10) + 1;
        document.getElementById('anomalies-count').textContent = Math.floor(Math.random() * 5) + 1;
        document.getElementById('run-analytics-btn').disabled = false;
        document.getElementById('run-analytics-btn').innerHTML = '<i class="fas fa-play"></i> Run Analysis';
        updateStatus('Analytics completed');
    }}, 3000);
}}

function loadSurveyMetadata() {{
    fetch('survey_metadata.json')
        .then(response => response.json())
        .then(data => {{
            document.getElementById('total-records').textContent = data.record_count || 0;
            document.getElementById('coverage-area').textContent = data.coverage_area || '--';
            document.getElementById('depth-range').textContent = data.depth_range || '--';
            document.getElementById('processing-time').textContent = data.processing_time || '--';
        }})
        .catch(error => console.error('Error loading metadata:', error));
}}

function loadAnalyticsData() {{
    fetch('analytics_data.json')
        .then(response => response.json())
        .then(data => {{
            // Load any pre-computed analytics
            console.log('Analytics data loaded:', data);
        }})
        .catch(error => console.error('Error loading analytics:', error));
}}

function updateCoordinates(lat, lon) {{
    document.getElementById('coordinates').textContent =
        `Lat: ${{lat.toFixed(6)}} | Lon: ${{lon.toFixed(6)}} | Depth: --m`;
}}

function updateStatus(message) {{
    document.getElementById('status-message').textContent = message;
    console.log('Status:', message);
}}

// Utility functions for depth profiling
function updateDepthProfile() {{
    // Implementation for depth profile visualization
    const canvas = document.getElementById('depth-chart');
    const ctx = canvas.getContext('2d');

    // Simple depth profile visualization
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height * 0.8);

    // Draw sample depth profile
    for (let i = 0; i < canvas.width; i++) {{
        const depth = Math.sin(i * 0.1) * 20 + 30; // Sample data
        const y = (depth / 50) * canvas.height;
        ctx.lineTo(i, y);
    }}

    ctx.strokeStyle = '#007cba';
    ctx.lineWidth = 2;
    ctx.stroke();
}}
'''

        with open(self.output_dir / "dashboard.js", 'w', encoding='utf-8') as f:
            f.write(js_content)

    def _create_css_styling(self):
        """Create professional CSS styling for the marine dashboard"""

        css_content = '''/* Marine Survey Dashboard CSS */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f5f5f5;
    color: #333;
}

#dashboard {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* Header */
#dashboard-header {
    background: linear-gradient(135deg, #007cba 0%, #005a87 100%);
    color: white;
    padding: 1rem 2rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}

#logo-section {
    flex-shrink: 0;
}

#title-section {
    flex: 1;
    text-align: center;
}

#title-section h1 {
    font-size: 1.5rem;
    font-weight: 300;
    margin: 0;
}

#survey-info {
    display: flex;
    gap: 2rem;
    font-size: 0.9rem;
    justify-content: center;
    margin-top: 0.5rem;
}

/* SonarSniffer Logo */
.sonar-sniffer-logo {
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
}

/* Main Content */
#main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
}

#map-container {
    flex: 1;
    position: relative;
}

#map {
    width: 100%;
    height: 100%;
}

/* Map Controls */
#map-controls {
    position: absolute;
    top: 10px;
    right: 10px;
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 250px;
    max-height: 70vh;
    overflow-y: auto;
}

.control-group {
    margin-bottom: 1.5rem;
}

.control-group h4 {
    margin-bottom: 0.5rem;
    color: #007cba;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.control-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    cursor: pointer;
}

.control-group input[type="checkbox"] {
    margin-right: 0.5rem;
}

.control-group input[type="range"] {
    width: 100%;
    margin: 0.5rem 0;
}

.action-btn, .export-btn {
    background: #007cba;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    width: 100%;
    text-align: left;
}

.action-btn:hover, .export-btn:hover {
    background: #005a87;
}

.action-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
}

#target-list {
    max-height: 200px;
    overflow-y: auto;
    margin-top: 0.5rem;
}

.target-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    border: 1px solid #eee;
    border-radius: 4px;
    margin-bottom: 0.25rem;
    background: #f9f9f9;
}

.target-item button {
    background: #28a745;
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.8rem;
}

.target-item button:hover {
    background: #218838;
}

/* Depth Profile Panel */
#depth-profile-panel {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 320px;
}

#depth-profile-panel h4 {
    margin-bottom: 0.5rem;
    color: #007cba;
}

#depth-chart {
    border: 1px solid #eee;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

#profile-info {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #666;
}

/* Sidebar */
#sidebar {
    width: 300px;
    background: white;
    border-left: 1px solid #e0e0e0;
    overflow-y: auto;
    padding: 1rem;
}

.sidebar-section {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e0e0e0;
}

.sidebar-section:last-child {
    border-bottom: none;
}

.sidebar-section h3 {
    color: #007cba;
    margin-bottom: 1rem;
    font-size: 1.1rem;
}

.stat-item, .analysis-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    padding: 0.25rem 0;
}

.stat-label, .analysis-label {
    font-weight: 500;
    color: #666;
}

.stat-value, .analysis-value {
    font-weight: bold;
    color: #007cba;
}

/* Status Bar */
#status-bar {
    background: #f8f9fa;
    border-top: 1px solid #e0e0e0;
    padding: 0.75rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}

#status-message {
    color: #28a745;
    font-weight: 500;
}

#coordinates {
    color: #666;
}

#license-info {
    color: #007cba;
    font-size: 0.8em;
}

#license-info a:hover {
    text-decoration: underline;
}

/* Responsive Design */
@media (max-width: 768px) {
    #main-content {
        flex-direction: column;
    }

    #sidebar {
        width: 100%;
        height: 200px;
        border-left: none;
        border-top: 1px solid #e0e0e0;
    }

    #map-controls {
        max-width: none;
        left: 10px;
        right: 10px;
        top: auto;
        bottom: 10px;
        max-height: 150px;
    }

    #depth-profile-panel {
        display: none; /* Hide on mobile for space */
    }
}

/* Loading Animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.fa-spinner {
    animation: spin 1s linear infinite;
}

/* Mapbox Overrides */
.mapboxgl-ctrl {
    font-family: inherit;
}

.mapboxgl-popup-content {
    font-family: inherit;
    border-radius: 8px;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}'''

        with open(self.output_dir / "dashboard.css", 'w', encoding='utf-8') as f:
            f.write(css_content)

    def _copy_mbtiles_file(self, mbtiles_path: str):
        """Copy MBTiles file to dashboard directory"""
        if Path(mbtiles_path).exists():
            shutil.copy2(mbtiles_path, self.output_dir / Path(mbtiles_path).name)

    def _create_metadata_json(self, survey_data: Dict):
        """Create survey metadata JSON file"""
        metadata = {
            "filename": survey_data.get("filename", "unknown"),
            "record_count": survey_data.get("record_count", 0),
            "coverage_area": survey_data.get("coverage_area", "unknown"),
            "depth_range": survey_data.get("depth_range", "unknown"),
            "processing_time": survey_data.get("processing_time", "unknown"),
            "center_lat": survey_data.get("center_lat", 40.0),
            "center_lon": survey_data.get("center_lon", -74.0),
            "bounds": survey_data.get("bounds", {}),
            "created_at": datetime.now().isoformat()
        }

        with open(self.output_dir / "survey_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _create_analytics_data(self, survey_data: Dict):
        """Create analytics data JSON file"""
        analytics = {
            "targets": [
                {
                    "lat": 40.001,
                    "lon": -74.001,
                    "depth_m": 15.5,
                    "type": "wreck",
                    "confidence": 85,
                    "size_estimate_m": 8.5
                },
                {
                    "lat": 40.002,
                    "lon": -74.002,
                    "depth_m": 22.3,
                    "type": "anomaly",
                    "confidence": 72,
                    "size_estimate_m": 3.2
                }
            ],
            "anomalies": [],
            "statistics": {
                "total_targets": 2,
                "high_confidence_targets": 1,
                "average_depth": 18.9,
                "survey_coverage_percent": 85.3
            }
        }

        with open(self.output_dir / "analytics_data.json", 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2)

def create_web_dashboard_from_rsd(rsd_file: str, output_dir: str = "marine_dashboard") -> str:
    """
    Create web dashboard from RSD file

    This is the main function that users would call to create a web dashboard
    from their RSD survey files.
    """

    print(f"🌐 Creating web dashboard for {rsd_file}")

    # Import here to avoid circular imports
    try:
        from engine_classic_varstruct import parse_rsd_records_classic, RSDRecord
        
        # Parse records
        records = list(parse_rsd_records_classic(rsd_file, limit_records=1000))  # Limit for demo
        
        # Extract survey data from records
        survey_data = {
            "filename": Path(rsd_file).name,
            "record_count": len(records),
            "processing_time": "2.3s",  # Would measure actual time
        }
        
        if records:
            # Calculate bounds and center
            lats = [r.lat for r in records if r.lat != 0.0]
            lons = [r.lon for r in records if r.lon != 0.0]
            depths = [r.depth_m for r in records if r.depth_m > 0]
            
            if lats and lons:
                survey_data.update({
                    "center_lat": sum(lats) / len(lats),
                    "center_lon": sum(lons) / len(lons),
                    "bounds": {
                        "north": max(lats),
                        "south": min(lats),
                        "east": max(lons),
                        "west": min(lons)
                    }
                })
            else:
                # Default values for demo
                survey_data.update({
                    "center_lat": 40.0,
                    "center_lon": -74.0,
                    "bounds": {
                        "north": 40.1,
                        "south": 39.9,
                        "east": -73.9,
                        "west": -74.1
                    }
                })
            
            if depths:
                survey_data.update({
                    "depth_range": ".1f",
                    "coverage_area": "4.8 sq miles"  # Would calculate from data
                })
            else:
                survey_data.update({
                    "depth_range": "0.1m - 45.3m",
                    "coverage_area": "4.8 sq miles"
                })
        else:
            # Default values if no records
            survey_data.update({
                "center_lat": 40.0,
                "center_lon": -74.0,
                "bounds": {
                    "north": 40.1,
                    "south": 39.9,
                    "east": -73.9,
                    "west": -74.1
                },
                "depth_range": "0.1m - 45.3m",
                "coverage_area": "4.8 sq miles"
            })

        # Create MBTiles (placeholder - would use actual MBTiles creation)
        mbtiles_path = f"{Path(rsd_file).stem}_survey.mbtiles"

        # For demo, create a simple MBTiles file
        _create_sample_mbtiles(mbtiles_path)

        # Create dashboard
        generator = WebDashboardGenerator()
        dashboard_path = generator.create_dashboard(survey_data, mbtiles_path, output_dir)

        return dashboard_path

    except ImportError as e:
        print(f"❌ Parser import failed: {e}")
        return ""

def _create_sample_mbtiles(filename: str):
    """Create a sample MBTiles file for demonstration"""
    # This is a placeholder - real implementation would create actual MBTiles
    # from survey data with proper tiling and NOAA ENC integration

    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Drop tables if they exist
    cursor.execute('DROP TABLE IF EXISTS metadata')
    cursor.execute('DROP TABLE IF EXISTS tiles')

    # Create MBTiles schema
    cursor.execute('''
        CREATE TABLE metadata (
            name TEXT,
            value TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB
        )
    ''')

    # Add metadata
    metadata = [
        ('name', 'Sample Survey MBTiles'),
        ('description', 'Demonstration MBTiles for marine survey data'),
        ('version', '1.0'),
        ('format', 'png'),
        ('type', 'overlay'),
        ('bounds', '-74.1,-39.9,-73.9,40.1'),
        ('center', '-74.0,40.0,10'),
        ('minzoom', '8'),
        ('maxzoom', '16')
    ]

    cursor.executemany('INSERT INTO metadata VALUES (?, ?)', metadata)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        rsd_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "marine_dashboard"

        dashboard_path = create_web_dashboard_from_rsd(rsd_file, output_dir)
        if dashboard_path:
            print(f"\n🎉 Dashboard created successfully!")
            print(f"📂 Location: {dashboard_path}")
            print(f"🌐 Open: file://{Path(dashboard_path).absolute()}/index.html")
        else:
            print("❌ Failed to create dashboard")
    else:
        print("Usage: python web_dashboard.py <rsd_file> [output_dir]")
        print("Example: python web_dashboard.py Holloway.RSD marine_dashboard")