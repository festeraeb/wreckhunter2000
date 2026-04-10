import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import copy
import glob
from pyproj import Transformer
import math

target_lat = 45.78725  
target_lon = -84.6708
files = glob.glob(r'C:\Users\thomf\Downloads\H13255*.bag')
files = [f for f in files if '(1)' not in f]
target_file = next(f for f in files if '5of6' in f)

box_size = 500  
with rasterio.open(target_file) as src:
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    t_x, t_y = transformer.transform(target_lon, target_lat)
    py, px = src.index(t_x, t_y)

    r_start = max(0, py - box_size // 2)
    r_end = min(src.height, py + box_size // 2)
    c_start = max(0, px - box_size // 2)
    c_end = min(src.width, px + box_size // 2)
    
    uncert = src.read(2, window=((r_start, r_end), (c_start, c_end)))
    tform = src.transform

nodata_val = 1000000.0
valid_uncert = uncert[(uncert > 0) & (uncert < nodata_val)]

ship_mask = (uncert < np.percentile(valid_uncert, 4)) & (uncert > 0)
contours = measure.find_contours(ship_mask, 0.5)
large_contours = [c for c in contours if len(c) > 50] 

# Find the biggest contour (the main black part)
main_contour = max(large_contours, key=len)

min_r, min_c = np.min(main_contour, axis=0)
max_r, max_c = np.max(main_contour, axis=0)

points = np.column_stack((main_contour[:, 1], main_contour[:, 0])) # (x, y)
cov_mat = np.cov(points.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

# The eigenvector with the largest eigenvalue is the principal axis (the main orientation)
principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

# Calculate angle in degrees
angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
angle_deg = np.degrees(angle_rad)

# Correcting to a 0-360 true north compass bearing
# array Y goes down, X goes right. 
# angle_rad is from the positive X axis going towards positive Y (downwards)
# Map to compass: North is up (-Y), East is right (+X)
dx = principal_axis[0]
dy = -principal_axis[1]  # flip Y since array goes down
compass_rad = math.atan2(dx, dy)
compass_bearing = math.degrees(compass_rad) % 360

print(f'Compass Bearing Orientation: {compass_bearing:.1f} degrees / {(compass_bearing+180)%360:.1f} degrees')

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(ship_mask, cmap='gray')
ax.plot(main_contour[:, 1], main_contour[:, 0], color='red', linewidth=2)

# Draw the orientation line
center_x, center_y = np.mean(points, axis=0)
scale = 100
ax.plot([center_x - principal_axis[0]*scale, center_x + principal_axis[0]*scale],
        [center_y - principal_axis[1]*scale, center_y + principal_axis[1]*scale], 
        color='gold', linewidth=3, linestyle='--')

ax.set_title(f"Orientation Check: {compass_bearing:.0f} / {(compass_bearing+180)%360:.0f} degrees", color='white')
fig.patch.set_facecolor('black')
ax.axis('off')
plt.savefig('orientation_check.png', facecolor='black', bbox_inches='tight')
