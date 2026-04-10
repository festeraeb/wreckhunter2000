"""
Geospatial Export Module for CESAROPS
Handles GeoTIFF creation, KML export with LOD, and Google Earth integration
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
import simplekml
from osgeo import gdal, gdalconst

logger = logging.getLogger(__name__)


class GeoTIFFGenerator:
    """Generate Cloud-Optimized GeoTIFF from mosaic data"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_geotiff(
        self,
        data: np.ndarray,
        bounds: Tuple[float, float, float, float],
        filename: str,
        compress: str = "DEFLATE",
    ) -> Path:
        """
        Create GeoTIFF from numpy array

        Args:
            data: 2D numpy array (mosaic or bathymetry)
            bounds: (minlon, minlat, maxlon, maxlat)
            filename: Output filename
            compress: Compression (DEFLATE, JPEG, ZSTD)

        Returns:
            Path to created GeoTIFF
        """
        try:
            output_path = self.output_dir / filename

            # Create driver and dataset
            driver = gdal.GetDriverByName("GTiff")
            height, width = data.shape

            ds = driver.Create(
                str(output_path),
                width,
                height,
                1,
                gdal.GDT_Float32,
                options=[
                    "COMPRESS=" + compress,
                    "BIGTIFF=YES",
                    "TILED=YES",
                    "BLOCKXSIZE=512",
                    "BLOCKYSIZE=512",
                ],
            )

            # Set geotransform
            minlon, minlat, maxlon, maxlat = bounds
            pixel_width = (maxlon - minlon) / width
            pixel_height = (maxlat - minlat) / height

            ds.SetGeoTransform([minlon, pixel_width, 0, maxlat, 0, -pixel_height])

            # Set projection (WGS84)
            from osgeo import osr

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())

            # Write data
            band = ds.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            band.FlushCache()

            ds = None  # Close dataset
            logger.info(f"Created GeoTIFF: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"GeoTIFF creation failed: {e}")
            return None

    def create_pyramid(
        self, geotiff_path: Path, levels: List[int] = [2, 4, 8, 16]
    ) -> bool:
        """Build overview pyramids for efficient viewing"""
        try:
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            ds = gdal.Open(str(geotiff_path), gdal.GA_ReadOnly)

            if ds is None:
                logger.error(f"Failed to open {geotiff_path}")
                return False

            # Build overviews
            ds.BuildOverviews("AVERAGE", levels)
            ds = None

            logger.info(f"Built pyramids for {geotiff_path}")
            return True

        except Exception as e:
            logger.error(f"Pyramid creation failed: {e}")
            return False


class KMLNetworkLinkGenerator:
    """Generate KML with NetworkLink and LOD for efficient Google Earth viewing"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_network_link_kml(
        self,
        name: str,
        bounds: Tuple[float, float, float, float],
        tile_url_pattern: str,
        description: str = "",
    ) -> Path:
        """
        Create KML with NetworkLink for tiled data

        Args:
            name: KML file name
            bounds: (minlon, minlat, maxlon, maxlat)
            tile_url_pattern: URL pattern for tiles (e.g., 'tiles/{z}/{x}/{y}.png')
            description: Description text

        Returns:
            Path to created KML
        """
        try:
            kml = simplekml.Kml()

            # Create document
            doc = kml.newdocument(name=name, description=description)

            # Add network link with region-based LOD
            netlink = doc.newnetworklink(
                name=f"{name} (Tiled)", description="Region-based progressive loading"
            )

            minlon, minlat, maxlon, maxlat = bounds

            # Create region with LOD
            region = netlink.region
            latlonaltbox = region.latlonaltbox
            latlonaltbox.north = maxlat
            latlonaltbox.south = minlat
            latlonaltbox.east = maxlon
            latlonaltbox.west = minlon

            # LOD (Level of Detail) - load when visible and pixels > 128
            region.lod.minlodpixels = 128
            region.lod.maxlodpixels = -1
            region.lod.minfadeextent = 0
            region.lod.maxfadeextent = 0

            # Link to tiles
            netlink.link.href = tile_url_pattern
            netlink.link.refreshmode = simplekml.RefreshMode.onregion

            # Save
            output_path = self.output_dir / f"{name}.kml"
            kml.save(str(output_path))

            logger.info(f"Created NetworkLink KML: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"KML generation failed: {e}")
            return None

    def create_placemark_kml(
        self, name: str, placemarks: List[dict], description: str = ""
    ) -> Path:
        """
        Create KML with placemarks (targets, anomalies)

        Args:
            name: KML file name
            placemarks: List of dicts with 'lat', 'lon', 'name', 'description', 'confidence'
            description: Document description

        Returns:
            Path to created KML
        """
        try:
            kml = simplekml.Kml()
            doc = kml.newdocument(name=name, description=description)

            # Color scale by confidence (red=low, green=high)
            def get_color_by_confidence(conf: float) -> str:
                # AABBGGRR format
                if conf >= 0.8:
                    return "ff00ff00"  # Green
                elif conf >= 0.6:
                    return "ff00ffff"  # Yellow
                elif conf >= 0.4:
                    return "ff0088ff"  # Orange
                else:
                    return "ff0000ff"  # Red

            for pm in placemarks:
                pnt = doc.newpoint(
                    name=pm.get("name", "Target"), coords=[(pm["lon"], pm["lat"])]
                )
                pnt.description = pm.get("description", "")

                # Style by confidence
                style = simplekml.Style()
                style.iconstyle.color = get_color_by_confidence(
                    pm.get("confidence", 0.5)
                )
                style.iconstyle.scale = 0.8 + pm.get("confidence", 0.5) * 0.4
                pnt.style = style

            output_path = self.output_dir / f"{name}_placemarks.kml"
            kml.save(str(output_path))

            logger.info(
                f"Created placemark KML: {output_path} ({len(placemarks)} marks)"
            )
            return output_path

        except Exception as e:
            logger.error(f"Placemark KML generation failed: {e}")
            return None


class DriftSimulationExporter:
    """Export drift simulation results to KML/GeoTIFF for Google Earth"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.geotiff_gen = GeoTIFFGenerator(output_dir)
        self.kml_gen = KMLNetworkLinkGenerator(output_dir)

    def export_drift_particles(
        self, particles: List[dict], lake: str, simulation_name: str
    ) -> dict:
        """
        Export drift simulation particles to KML

        Args:
            particles: List of particle dicts with 'lon', 'lat', 'time'
            lake: Lake name
            simulation_name: Simulation identifier

        Returns:
            Dict with export results
        """
        try:
            placemarks = []

            # Convert particles to placemarks
            for i, p in enumerate(particles[:1000]):  # Limit to 1000 for performance
                placemarks.append(
                    {
                        "lat": p["lat"],
                        "lon": p["lon"],
                        "name": f"Particle {i}",
                        "description": f"Time: {p.get('time', 'Unknown')}",
                        "confidence": 0.9,
                    }
                )

            # Create KML
            kml_path = self.kml_gen.create_placemark_kml(
                f"{lake}_{simulation_name}_particles",
                placemarks,
                description=f"Drift simulation: {simulation_name}",
            )

            return {
                "status": "success",
                "kml_file": str(kml_path),
                "particle_count": len(placemarks),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Particle export failed: {e}")
            return {"status": "failed", "error": str(e)}

    def export_heatmap(
        self,
        heatmap_data: np.ndarray,
        bounds: Tuple[float, float, float, float],
        lake: str,
        name: str,
    ) -> dict:
        """
        Export heatmap (concentration grid) to GeoTIFF

        Args:
            heatmap_data: 2D concentration grid
            bounds: (minlon, minlat, maxlon, maxlat)
            lake: Lake name
            name: Heatmap name

        Returns:
            Dict with export results
        """
        try:
            filename = f"{lake}_{name}_heatmap.tif"
            geotiff_path = self.geotiff_gen.create_geotiff(
                heatmap_data, bounds, filename
            )

            if geotiff_path:
                self.geotiff_gen.create_pyramid(geotiff_path)

            return {
                "status": "success",
                "geotiff_file": str(geotiff_path),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Heatmap export failed: {e}")
            return {"status": "failed", "error": str(e)}


def export_simulation_to_kml(
    simulation_result: dict, lake: str, output_dir: str = "outputs"
) -> dict:
    """Convenience function to export complete simulation"""
    exporter = DriftSimulationExporter(output_dir)

    results = {}

    # Export particles if available
    if "particles" in simulation_result:
        results["particles"] = exporter.export_drift_particles(
            simulation_result["particles"],
            lake,
            simulation_result.get("name", "simulation"),
        )

    # Export heatmap if available
    if "heatmap" in simulation_result:
        results["heatmap"] = exporter.export_heatmap(
            simulation_result["heatmap"],
            simulation_result.get("bounds", (-90, 40, -82, 48)),
            lake,
            simulation_result.get("name", "simulation"),
        )

    return results
