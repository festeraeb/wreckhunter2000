import os
import logging
from datetime import datetime, timedelta

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic, reader_global_landmask
import xarray as xr
print("Xarray IO engines available:", xr.backends.list_engines())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CESAROPS")

# Ensure output folders exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ERDDAP URLs for GLERL GLCFS Great Lakes hydrodynamic model data
erddap_urls = {
    'michigan': 'https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_MICHIGAN_3D.nc?U[(last)][(0.0)][(41.5):(46.0)][(-88.5):(-85.5)],V[(last)][(0.0)][(41.5):(46.0)][(-88.5):(-85.5)]',
    'erie': 'https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_ERIE_3D.nc?U[(last)][(0.0)][(41.2):(42.9)][(-83.7):(-78.8)],V[(last)][(0.0)][(41.2):(42.9)][(-83.7):(-78.8)]',
    'huron': 'https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_HURON_3D.nc?U[(last)][(0.0)][(43.2):(46.3)][(-85.0):(-80.5)],V[(last)][(0.0)][(43.2):(46.3)][(-85.0):(-80.5)]',
    'ontario': 'https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_ONTARIO_3D.nc?U[(last)][(0.0)][(43.2):(44.3)][(-80.0):(-76.0)],V[(last)][(0.0)][(43.2):(44.3)][(-80.0):(-76.0)]',
    'superior': 'https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_SUPERIOR_3D.nc?U[(last)][(0.0)][(46.0):(49.5)][(-92.5):(-84.5)],V[(last)][(0.0)][(46.0):(49.5)][(-92.5):(-84.5)]'
}

# ----------------------------------------
# Main simulation logic
# ----------------------------------------

def run_drift_simulation():
    lake = 'michigan'
    erddap_url = erddap_urls.get(lake, erddap_urls['michigan'])

    # Create OpenDrift model
    o = OceanDrift(loglevel=0)

    # Add environmental readers
    reader_current = reader_netCDF_CF_generic.Reader(erddap_url)
    reader_landmask = reader_global_landmask.Reader()
    o.add_reader([reader_current, reader_landmask])

    # Seed elements
    o.seed_elements(
        lon=-86.5,
        lat=43.0,
        number=100,
        time=datetime.utcnow()
    )

    # Run the model
    o.run(duration=timedelta(hours=12))

    # Output
    o.plot(filename=f'outputs/sim_{lake}.html')
    logger.info("Simulation complete")

if __name__ == '__main__':
    run_drift_simulation()
