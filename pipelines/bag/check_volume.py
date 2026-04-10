import rasterio, numpy as np
import json
from pyproj import Transformer
from scipy.interpolate import griddata

target_file = r'C:\Users\thomf\Downloads\H13255_MB_1m_LWD_5of6.bag' 
t_lat, t_lon = 45.86, -84.45

with rasterio.open(target_file) as src:
    trans = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    t_x, t_y = trans.transform(t_lon, t_lat)
    py, px = src.index(t_x, t_y)
    
    box = 400
    elev = src.read(1, window=((py-box//2, py+box//2), (px-box//2, px+box//2)))
    uncert = src.read(2, window=((py-box//2, py+box//2), (px-box//2, px+box//2)))
    
    valid_u = uncert[(uncert > 0) & (uncert < 100)]
    mask = (uncert > 0) & (uncert < np.percentile(valid_u, 5)) & (elev > -100)
    
    y_valid, x_valid = np.where(~mask & (elev > -100))
    z_valid = elev[y_valid, x_valid]
    y_grid, x_grid = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    
    interp_elev = griddata((y_valid[::2], x_valid[::2]), z_valid[::2], (y_grid, x_grid), method='nearest')
    diff = interp_elev - elev
    diff[~mask] = 0
    diff[diff < 0] = 0
    diff_ft = diff * 3.28084
    
    total_mask_area_ft = int(np.sum(mask) * (3.28084*3.28084))
    ship_core_area = int(np.sum(diff_ft > 5.0) * (3.28084*3.28084))
    huge_core_area = int(np.sum(diff_ft > 15.0) * (3.28084*3.28084))
    peak_height = float(np.max(diff_ft))
    
    out = {
        "mask_area": total_mask_area_ft,
        "over_5ft": ship_core_area,
        "over_15ft": huge_core_area,
        "peak_height": peak_height
    }
    with open("volume_stats.json", "w") as f:
        json.dump(out, f)
