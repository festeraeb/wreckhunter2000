import geopandas as gpd
import pandas as pd

def fuse_magnetic_data(ml_results_path, magnetic_data_path, output_path):
    """
    Fuse ML probabilities with magnetic data using 1200m offset logic.

    Args:
        ml_results_path (str): Path to the ML results GeoJSON.
        magnetic_data_path (str): Path to the magnetic data GeoJSON.
        output_path (str): Path to save the fused GeoJSON.
    """
    ml_results = gpd.read_file(ml_results_path)
    magnetic_data = gpd.read_file(magnetic_data_path)

    # Buffer magnetic data by 1200m
    magnetic_buffer = magnetic_data.copy()
    magnetic_buffer['geometry'] = magnetic_buffer.geometry.buffer(1200)

    # Spatial join to find overlaps
    fused = gpd.sjoin(ml_results, magnetic_buffer, how="inner", predicate="intersects")

    # Save the fused results
    fused.to_file(output_path, driver="GeoJSON")
    print(f"Fused data saved to {output_path}")

if __name__ == "__main__":
    ml_results_path = "../sample_data/ml_results.geojson"  # Replace with actual path
    magnetic_data_path = "../sample_data/magnetic_data.geojson"  # Replace with actual path
    output_path = "../sample_data/fused_results.geojson"

    fuse_magnetic_data(ml_results_path, magnetic_data_path, output_path)