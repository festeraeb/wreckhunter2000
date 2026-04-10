import geopandas as gpd
import re

def parse_kml(file_path):
    """
    Parse a KML file to extract magnetic target coordinates and scores.

    Args:
        file_path (str): Path to the KML file.

    Returns:
        list: A list of dictionaries containing target name, coordinates, and magnetic score.
    """
    gdf = gpd.read_file(file_path, driver='KML')

    print("DEBUG: GeoDataFrame columns:", gdf.columns)
    print("DEBUG: GeoDataFrame head:")
    print(gdf.head())

    targets = []
    for _, row in gdf.iterrows():
        name = row.get('Name', 'Unknown')
        description = row.get('description', '')
        coordinates = row.geometry.coords[0] if row.geometry else None

        # Extract magnetic score from the description using regex
        score_match = re.search(r'Magnetic Score:\s*(\d+)', description)
        magnetic_score = int(score_match.group(1)) if score_match else None

        targets.append({
            'name': name,
            'coordinates': coordinates,
            'magnetic_score': magnetic_score
        })

    return targets

if __name__ == "__main__":
    kml_file = "c:/Users/thomf/programming/Bagrecovery/sample_data/magnetic_targets.kml"
    results = parse_kml(kml_file)
    for target in results:
        print(target)