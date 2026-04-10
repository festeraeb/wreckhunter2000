#!/usr/bin/env python3
"""
Rosa Case Visualization and Model Calibration
=============================================

Create visualization of Rosa case analysis results and identify
issues with our current model predictions.

Author: GitHub Copilot
Date: January 7, 2025
"""

import json
import numpy as np
from datetime import datetime

def create_rosa_visualization():
    """Create HTML visualization of Rosa case analysis"""
    
    # Load analysis results
    try:
        with open('rosa_real_analysis_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Analysis results not found. Run rosa_real_analysis.py first.")
        return
    
    # Extract key data
    rosa_facts = results['rosa_case_facts']
    best_scenarios = results['hindcast_analysis']['best_scenarios']
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Rosa Fender Case Analysis Results</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 85vh; width: 100%; }}
        .header {{
            background: #2c3e50; color: white; padding: 15px; text-align: center;
        }}
        .info-panel {{
            position: absolute; top: 10px; right: 10px; background: white;
            padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000; max-width: 350px;
        }}
        .issue-panel {{
            position: absolute; top: 10px; left: 10px; background: #e74c3c;
            color: white; padding: 15px; border-radius: 5px; z-index: 1000; max-width: 300px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🌊 Rosa Fender Case Analysis Results</h2>
        <p>Hindcast Analysis: Off Milwaukee → South Haven, MI</p>
    </div>
    
    <div class="issue-panel">
        <h3>⚠️ Model Issue Detected</h3>
        <p><strong>Problem:</strong> Model predicts ending ~144 km from South Haven</p>
        <p><strong>Expected:</strong> Should predict arrival at South Haven</p>
        <p><strong>Likely cause:</strong> Environmental data or drift parameters need calibration</p>
    </div>
    
    <div class="info-panel">
        <h3>📊 Rosa Case Facts</h3>
        <p><strong>Last Known:</strong> 42.995°N, 87.845°W</p>
        <p><strong>Incident:</strong> Aug 22, 8 PM</p>
        <p><strong>Found:</strong> South Haven, MI</p>
        <p><strong>Found Time:</strong> Aug 23, 8 AM</p>
        <p><strong>Drift Time:</strong> ~12 hours</p>
        <hr>
        <p><strong>Analysis Results:</strong></p>
        <p>• Scenarios: 900</p>
        <p>• Best accuracy: 144.2 km</p>
        <p>• Search radius: 2.8 km</p>
        <hr>
        <p><strong>🔵 Blue:</strong> Last known position</p>
        <p><strong>🔴 Red:</strong> Found location (South Haven)</p>
        <p><strong>🟡 Yellow:</strong> Top predicted drop zones</p>
        <p><strong>🟢 Green:</strong> Model predictions</p>
    </div>
    
    <div id="map"></div>

    <script>
        var map = L.map('map').setView([43.0, -87.0], 8);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Last known position (blue)
        var lastKnown = L.marker([{rosa_facts['last_known_location']['lat']}, {rosa_facts['last_known_location']['lon']}], {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
            }})
        }}).addTo(map);
        
        lastKnown.bindPopup(`
            <b>🌊 Last Known Position</b><br>
            <strong>Location:</strong> {rosa_facts['last_known_location']['lat']:.4f}°N, {rosa_facts['last_known_location']['lon']:.4f}°W<br>
            <strong>Time:</strong> {rosa_facts['incident_time']}<br>
            <strong>Description:</strong> Rosa fender off Milwaukee
        `);

        // Found location (red) 
        var foundLocation = L.marker([{rosa_facts['found_location']['lat']}, {rosa_facts['found_location']['lon']}], {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
            }})
        }}).addTo(map);
        
        foundLocation.bindPopup(`
            <b>🏖️ Found Location</b><br>
            <strong>Location:</strong> South Haven, MI<br>
            <strong>Coordinates:</strong> {rosa_facts['found_location']['lat']:.4f}°N, {rosa_facts['found_location']['lon']:.4f}°W<br>
            <strong>Time:</strong> {rosa_facts['found_time']}<br>
            <strong>Description:</strong> Where Rosa fender was actually found
        `);

        // Top 5 predicted drop zones (yellow)'''
    
    # Add top scenarios to map
    for i, scenario in enumerate(best_scenarios[:5]):
        html_content += f'''
        var dropZone{i+1} = L.marker([{scenario['start_lat']}, {scenario['start_lon']}], {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-yellow.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [20, 32], iconAnchor: [10, 32], popupAnchor: [1, -28], shadowSize: [32, 32]
            }})
        }}).addTo(map);
        
        dropZone{i+1}.bindPopup(`
            <b>📍 Drop Zone #{i+1}</b><br>
            <strong>Coordinates:</strong> {scenario['start_lat']:.4f}°N, {scenario['start_lon']:.4f}°W<br>
            <strong>Distance to South Haven:</strong> {scenario['distance_to_south_haven_km']:.1f} km<br>
            <strong>Distance from known:</strong> {scenario['distance_from_known_position_km']:.1f} km<br>
            <strong>Accuracy Score:</strong> {scenario['accuracy_score']:.1f}/25
        `);'''
    
    html_content += '''
        
        // Draw line from last known to found location
        var actualDrift = L.polyline([
            [42.995, -87.845],
            [42.4030, -86.2750]
        ], {color: 'red', weight: 3, opacity: 0.7, dashArray: '5, 5'}).addTo(map);
        
        actualDrift.bindPopup('<b>🎯 Actual Drift Path</b><br>Distance: ~145 km<br>Direction: SE');
        
        // Fit map to show all points
        var group = new L.featureGroup([lastKnown, foundLocation]);
        map.fitBounds(group.getBounds().pad(0.1));
        
        // Add scale
        L.control.scale().addTo(map);
        
        // Add legend
        var legend = L.control({position: 'bottomleft'});
        legend.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
                    <h4>📍 Legend</h4>
                    <p><span style="color: blue;">🔵</span> Last Known Position</p>
                    <p><span style="color: red;">🔴</span> Found Location</p>
                    <p><span style="color: orange;">🟡</span> Predicted Drop Zones</p>
                    <p><span style="color: red;">---</span> Actual Drift (estimated)</p>
                </div>
            `;
            return div;
        };
        legend.addTo(map);
        
    </script>
</body>
</html>'''
    
    # Save visualization
    with open('outputs/rosa_case_analysis_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Visualization saved to outputs/rosa_case_analysis_visualization.html")

def analyze_model_issues():
    """Analyze why our model isn't predicting the correct drift"""
    
    print("ROSA CASE MODEL ANALYSIS")
    print("=" * 25)
    
    # Known facts
    last_known = (42.995, -87.845)  # Off Milwaukee
    found_location = (42.4030, -86.2750)  # South Haven, MI
    
    # Calculate actual drift
    lat_diff = found_location[0] - last_known[0]
    lon_diff = found_location[1] - last_known[1]
    
    # Convert to km (approximate)
    lat_km = lat_diff * 111.0
    lon_km = lon_diff * 111.0 * np.cos(np.radians(last_known[0]))
    
    total_distance = np.sqrt(lat_km**2 + lon_km**2)
    bearing = np.degrees(np.arctan2(lon_km, lat_km))
    
    print(f"📍 ACTUAL DRIFT ANALYSIS:")
    print(f"   From: {last_known[0]:.4f}°N, {last_known[1]:.4f}°W")
    print(f"   To: {found_location[0]:.4f}°N, {found_location[1]:.4f}°W")
    print(f"   Distance: {total_distance:.1f} km")
    print(f"   Bearing: {bearing:.1f}° (South-Southeast)")
    print(f"   Time: ~12 hours")
    print(f"   Speed: {total_distance/12:.1f} km/hr")
    
    print(f"\n⚠️ MODEL ISSUES:")
    print(f"   1. Model predicts ending ~144 km from South Haven")
    print(f"   2. Should predict arrival AT South Haven")
    print(f"   3. Indicates our drift parameters are incorrect")
    
    print(f"\n🔧 CALIBRATION NEEDED:")
    print(f"   1. Increase southward drift component")
    print(f"   2. Adjust wind/current coefficients")
    print(f"   3. Account for Lake Michigan circulation patterns")
    print(f"   4. Validate environmental data for August 22-23, 2025")
    
    # Calculate required drift components
    required_lat_speed = lat_km / 12  # km/hr southward
    required_lon_speed = lon_km / 12  # km/hr eastward
    
    print(f"\n📊 REQUIRED DRIFT COMPONENTS:")
    print(f"   Southward: {required_lat_speed:.2f} km/hr")
    print(f"   Eastward: {required_lon_speed:.2f} km/hr")
    print(f"   Suggests strong southward current/wind component")

def main():
    """Main function"""
    print("Rosa Case Analysis Visualization & Calibration")
    print("=" * 46)
    
    # Create visualization
    create_rosa_visualization()
    
    # Analyze model issues
    analyze_model_issues()
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. View visualization: outputs/rosa_case_analysis_visualization.html")
    print(f"   2. Calibrate drift parameters based on actual Rosa case")
    print(f"   3. Re-run analysis with corrected parameters")
    print(f"   4. Validate against other known cases")

if __name__ == "__main__":
    main()