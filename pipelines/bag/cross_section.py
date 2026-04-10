import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
from pyproj import Transformer

target_lat = 45.78725  
target_lon = -84.6708
files = glob.glob(r'C:\Users\thomf\Downloads\H13255*.bag')
files = [f for f in files if '(1)' not in f]
target_file = next(f for f in files if '5of6' in f)

with rasterio.open(target_file) as src:
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    t_x, t_y = transformer.transform(target_lon, target_lat)
    py, px = src.index(t_x, t_y)

    r_start, r_end = py - 100, py + 100
    c_start, c_end = px - 100, px + 100
    
    uncert = src.read(2, window=((r_start, r_end), (c_start, c_end)))
    elev = src.read(1, window=((r_start, r_end), (c_start, c_end)))

row_slice_elev = elev[100, :]
row_slice_uncert = uncert[100, :]

valid = (row_slice_elev < 10000)
row_slice_elev = np.where(valid, row_slice_elev, np.nan)
row_slice_uncert = np.where(row_slice_uncert < 10000, row_slice_uncert, np.nan)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(row_slice_elev, 'b-', label='Elevation (Z)')
ax1.set_ylabel('Depth (meters) - Negative is down', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(row_slice_uncert, 'r-', label='Uncertainty')
ax2.set_ylabel('Uncertainty Level', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Cross-section East-West across the Wreck')
fig.tight_layout()
plt.savefig('cross_section.png')
print("Saved cross_section.png")
