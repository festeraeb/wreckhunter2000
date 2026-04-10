import os
from pathlib import Path
from typing import Tuple

from scripts.nasa_earthdata_client import EarthdataClient, EarthdataAPIError

OPERA_DSWX_COLLECTION_ID = "C2617126679-POCLOUD"


def fetch_opera_dswx(client: EarthdataClient, bbox: Tuple[float, float, float, float], time_range: Tuple[str, str], output_dir: str) -> str:
    """Fetch OPERA DSWx data for the given bounding box and time range."""
    granules = client.search_granules(
        collection_concept_id=OPERA_DSWX_COLLECTION_ID,
        bbox=bbox,
        start_date=time_range[0],
        end_date=time_range[1],
    )

    if not granules:
        raise EarthdataAPIError("No granules found for the specified parameters")

    urls = client.extract_download_urls(granules[0])
    if not urls:
        raise EarthdataAPIError("No download URLs found for the first granule")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "opera_dswx_data.nc")
    client.download_url(urls[0], output_path)
    return output_path

if __name__ == "__main__":
    # Example usage
    # Token will be read automatically from NASA_EARTHDATA_TOKEN env var or .earthdata_token file.
    bbox = (-82.0, 45.0, -81.0, 46.0)  # Example bounding box
    time_range = ("2026-03-01", "2026-03-19")  # Example time range
    output_dir = "../sample_data/opera_dswx"

    os.makedirs(output_dir, exist_ok=True)

    # EarthdataClient will automatically read the token from the
    # NASA_EARTHDATA_TOKEN env var or the repository's .earthdata_token file.
    client = EarthdataClient()

    # Use session for API requests, similar to fetch_swot_data.py
    try:
        file_path = fetch_opera_dswx(client, bbox, time_range, output_dir)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error fetching OPERA DSWx data: {e}")