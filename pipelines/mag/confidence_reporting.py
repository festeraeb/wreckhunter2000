import geopandas as gpd

def generate_confidence_report(fused_data_path, output_path):
    """
    Generate a GeoJSON with "Sensor Agreement" breakdown for each target.

    Args:
        fused_data_path (str): Path to the fused GeoJSON data.
        output_path (str): Path to save the confidence report GeoJSON.
    """
    fused_data = gpd.read_file(fused_data_path)

    # Calculate sensor agreement breakdown
    fused_data['sensor_agreement'] = fused_data.apply(
        lambda row: {
            "ML_Probability": row.get("ml_probability", 0),
            "Magnetic_Score": row.get("magnetic_score", 0),
            "SAR_Persistence": row.get("sar_persistence", 0)
        }, axis=1
    )

    # Save the confidence report
    fused_data.to_file(output_path, driver="GeoJSON")
    print(f"Confidence report saved to {output_path}")

if __name__ == "__main__":
    fused_data_path = "../sample_data/fused_results.geojson"  # Replace with actual path
    output_path = "../sample_data/confidence_report.geojson"

    generate_confidence_report(fused_data_path, output_path)