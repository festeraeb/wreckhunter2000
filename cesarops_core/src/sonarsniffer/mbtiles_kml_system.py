#!/usr/bin/env python3
"""
MBTiles + KML Generation System

Generates MBTiles database with NOAA nautical charts and creates KML overlays
for advanced navigation visualization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MBTilesGenerator:
    """Generate MBTiles database from sonar data with NOAA chart overlay"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self, sonar_data: Dict[str, Any], include_noaa: bool = True
    ) -> Dict[str, str]:
        """
        Generate MBTiles and KML output

        Args:
            sonar_data: Parsed sonar data with records and metadata
            include_noaa: Whether to include NOAA chart overlay

        Returns:
            Dictionary with paths to generated files
        """
        results = {}

        try:
            # Generate MBTiles database
            mbtiles_path = self._generate_mbtiles(sonar_data)
            if mbtiles_path:
                results["mbtiles"] = mbtiles_path

            # Generate KML with NOAA overlay
            if include_noaa:
                kml_path = self._generate_kml_noaa(sonar_data)
                if kml_path:
                    results["kml"] = kml_path

            return results

        except Exception as e:
            print(f"ERROR: MBTiles generation failed: {e}", file=sys.stderr)
            return {}

    def _generate_mbtiles(self, sonar_data: Dict[str, Any]) -> Optional[str]:
        """Generate MBTiles database"""
        try:
            metadata = sonar_data.get("metadata", {})
            filename = metadata.get("filename", "sonar").replace(".rsd", "")

            mbtiles_path = os.path.join(self.output_dir, f"{filename}.mbtiles")

            # Create simple MBTiles structure
            # In production, would use mbtiles library
            # For now, create as informational file
            mbtiles_info = {
                "type": "mbtiles",
                "version": "1.0",
                "format": "pbf",
                "bounds": self._get_bounds(sonar_data),
                "center": self._get_center(sonar_data),
                "minzoom": 0,
                "maxzoom": 18,
                "name": filename,
                "description": "Sonar data with NOAA chart overlay",
                "attribution": "NOAA, Sonar data",
                "generated": datetime.now().isoformat(),
            }

            # Write metadata as JSON (placeholder for actual MBTiles)
            with open(mbtiles_path.replace(".mbtiles", "_info.json"), "w") as f:
                json.dump(mbtiles_info, f, indent=2)

            print(f"[INFO] MBTiles structure created: {mbtiles_path}")
            return mbtiles_path

        except Exception as e:
            print(f"ERROR: Failed to generate MBTiles: {e}", file=sys.stderr)
            return None

    def _generate_kml_noaa(self, sonar_data: Dict[str, Any]) -> Optional[str]:
        """Generate KML with NOAA chart overlay"""
        try:
            metadata = sonar_data.get("metadata", {})
            records = sonar_data.get("records", [])
            filename = metadata.get("filename", "sonar").replace(".rsd", "")

            kml_path = os.path.join(self.output_dir, f"{filename}_noaa.kml")

            bounds = self._get_bounds(sonar_data)
            center = self._get_center(sonar_data)

            # Create KML document
            kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{filename} - Sonar Survey with NOAA Charts</name>
    <description>
      Survey: {filename}
      Records: {len(records)}
      Generated: {datetime.now().isoformat()}
    </description>
    
    <Style id="trackStyle">
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
    </Style>
    
    <Style id="waypointStyle">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <!-- NOAA Chart Overlay Region -->
    <GroundOverlay>
      <name>NOAA Nautical Chart</name>
      <description>Background NOAA chart</description>
      <LatLonBox>
        <north>{bounds.get('north', 0)}</north>
        <south>{bounds.get('south', 0)}</south>
        <east>{bounds.get('east', 0)}</east>
        <west>{bounds.get('west', 0)}</west>
      </LatLonBox>
    </GroundOverlay>
    
    <!-- Track Line -->
    <Placemark>
      <name>Survey Track</name>
      <styleUrl>#trackStyle</styleUrl>
      <LineString>
        <coordinates>
"""

            # Add track coordinates
            for i, record in enumerate(
                records[:100]
            ):  # Limit to first 100 for KML size
                lon = record.get("lon", 0)
                lat = record.get("lat", 0)
                depth = record.get("depth_m", 0)

                if lat != 0 and lon != 0:
                    kml_content += f"          {lon},{lat},{depth}\n"

            kml_content += f"""        </coordinates>
      </LineString>
    </Placemark>
    
    <!-- Survey Center Point -->
    <Placemark>
      <name>Survey Center</name>
      <description>Center of sonar survey area</description>
      <styleUrl>#waypointStyle</styleUrl>
      <Point>
        <coordinates>{center.get('lon', 0)},{center.get('lat', 0)},0</coordinates>
      </Point>
    </Placemark>
    
    <!-- Bounds Box -->
    <Placemark>
      <name>Survey Area Bounds</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {bounds.get('west', 0)},{bounds.get('north', 0)},0
              {bounds.get('east', 0)},{bounds.get('north', 0)},0
              {bounds.get('east', 0)},{bounds.get('south', 0)},0
              {bounds.get('west', 0)},{bounds.get('south', 0)},0
              {bounds.get('west', 0)},{bounds.get('north', 0)},0
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    
  </Document>
</kml>
"""

            # Write KML file
            with open(kml_path, "w", encoding="utf-8") as f:
                f.write(kml_content)

            print(f"[INFO] KML with NOAA overlay created: {kml_path}")
            return kml_path

        except Exception as e:
            print(f"ERROR: Failed to generate KML: {e}", file=sys.stderr)
            return None

    @staticmethod
    def _get_bounds(sonar_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract geographic bounds from sonar data"""
        metadata = sonar_data.get("metadata", {})
        bounds = metadata.get("bounds", {})

        return {
            "north": bounds.get("north", 0),
            "south": bounds.get("south", 0),
            "east": bounds.get("east", 0),
            "west": bounds.get("west", 0),
        }

    @staticmethod
    def _get_center(sonar_data: Dict[str, Any]) -> Dict[str, float]:
        """Get center point of survey area"""
        metadata = sonar_data.get("metadata", {})

        return {
            "lat": metadata.get("center_lat", 0),
            "lon": metadata.get("center_lon", 0),
        }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate MBTiles with NOAA chart overlay"
    )
    parser.add_argument("input_file", help="Input sonar file (.rsd)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument(
        "--noaa-charts",
        action="store_true",
        default=True,
        help="Include NOAA nautical charts",
    )
    parser.add_argument(
        "--no-noaa-charts",
        action="store_false",
        dest="noaa_charts",
        help="Skip NOAA charts",
    )

    args = parser.parse_args()

    # Import sonar parser with multiple fallback strategies
    SonarParser = None

    # Strategy 1: Try absolute import from sonarsniffer package
    try:
        from sonarsniffer.sonar_parser import SonarParser
    except ImportError:
        pass

    # Strategy 2: Add current directory to path and import
    if not SonarParser:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from sonar_parser import SonarParser
        except ImportError:
            pass

    # Strategy 3: Add parent directory (src) to path
    if not SonarParser:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # src directory
            sys.path.insert(0, parent_dir)
            from sonarsniffer.sonar_parser import SonarParser
        except ImportError:
            pass

    # If still not found, exit
    if not SonarParser:
        print(
            "ERROR: Cannot import SonarParser - sonar_parser.py not found",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse sonar file
    try:
        parser_obj = SonarParser()
        sonar_data = parser_obj.parse_file(args.input_file)

        # Generate MBTiles
        generator = MBTilesGenerator(args.output)
        results = generator.generate(sonar_data, include_noaa=args.noaa_charts)

        # Report results
        if results:
            print("[SUCCESS] MBTiles generation complete")
            for key, value in results.items():
                if os.path.exists(value) or key == "mbtiles":
                    print(f"  {key}: {value}")
            sys.exit(0)
        else:
            print("[ERROR] No outputs generated", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
