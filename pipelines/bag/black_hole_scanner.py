import os
import logging
from pathlib import Path
from datetime import datetime
import rasterio
import numpy as np
import plotly.graph_objects as go
from pyproj import Transformer
from skimage import measure
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

def run_black_hole_scan(bag_files, output_dir, config):
    start_time = datetime.now()
    results_list = []
    total_candidates = 0
    os.makedirs(output_dir, exist_ok=True)
    
    # config tuning
    scan_mode = config.get('scan_mode', 'masked')
    scan_mode = config.get('scan_mode', 'masked')
    min_area_ft = config.get('min_wreck_size_sq_ft', 100.0)
    max_area_ft = config.get('max_wreck_size_sq_ft', 500000.0)
    plot_visuals = config.get('plot_visuals', True)
    
    for path in bag_files:
        _start = datetime.now()
        candidates = []
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_json_path = os.path.join(output_dir, f"{base_name}_results.json")
        if os.path.exists(out_json_path):
            logging.info(f"Skipping {base_name}, already processed.")
            # Read existing to append to results list
            try:
                import json
                with open(out_json_path, 'r') as jf:
                    existing_data = json.load(jf)
                    results_list.append(existing_data)
                    total_candidates += existing_data.get('total_hits', 0)
            except Exception:
                pass
            continue

        try:
            if scan_mode in ('unmasked', 'both'):
                from standalone_bag_scanner import StandaloneBagScanner
                scanner = StandaloneBagScanner(config=config)
                unmasked_candidates = scanner.scan_bag_file(path)
                if unmasked_candidates:
                    candidates.extend(unmasked_candidates)
                    
            if scan_mode in ('masked', 'both'):
                with rasterio.open(path) as src:
                    trans = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                    
                    # Use block-based or full read strategy depending on size.
                    # Assuming these files fit in memory for now based on Rust scanner approach
                    elev_full = src.read(1)
                    uncert_full = src.read(2)
                
                    valid_u = uncert_full[(uncert_full > 0) & (uncert_full < 100)]
                    if len(valid_u) == 0:
                        continue
                    
                    p5 = np.percentile(valid_u, 5)
                    mask_full = (uncert_full > 0) & (uncert_full < p5) & (elev_full > -1000) & (elev_full < 1000)
                
                    # Merge nearby masked pieces to avoid duplicate detections for the same wreck
                    from scipy.ndimage import binary_closing, binary_dilation
                    # Dilate and then erode (closing) over a 20-pixel radius (approx 10m on 50cm grids)
                    mask_full = binary_closing(mask_full, iterations=15)
                
                    # Label contiguous masked areas
                    labeled_masks = measure.label(mask_full, connectivity=2)
                    regions = measure.regionprops(labeled_masks)
                
                    # We sort by region size descending to catch the big ones
                    regions = sorted(regions, key=lambda x: x.area, reverse=True)
                
                    for idx, region in enumerate(regions):
                        # Filter out tiny pixels immediately
                        if region.area < 10: 
                            continue
                        
                        # Calculate bounding box with padding, limit excessive padding to prevent OOM/hang
                        min_row, min_col, max_row, max_col = region.bbox
                        pad_val = int(max(max_row - min_row, max_col - min_col) * 0.3)
                        pad_val = min(pad_val, 400) # prevent hanging on multi-mile patches
                        pad = max(40, pad_val)
                    
                        r_start = max(0, min_row - pad)
                        r_end = min(src.height, max_row + pad)
                        c_start = max(0, min_col - pad)
                        c_end = min(src.width, max_col + pad)
                    
                        elev = elev_full[r_start:r_end, c_start:c_end].copy()
                        uncert = uncert_full[r_start:r_end, c_start:c_end].copy()
                    
                        # Local mask
                        l_mask = (uncert > 0) & (uncert < p5) & (elev > -1000) & (elev < 1000)
                    
                        y_valid, x_valid = np.where(~l_mask & (elev > -1000) & (elev < 1000))
                        if len(y_valid) < 50:
                            continue # Not enough edge data to interpolate
                        
                        z_valid = elev[y_valid, x_valid]
                        y_grid, x_grid = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
                    
                        # Interpolate using nearest to reconstruct lakebed
                        stride = max(1, len(y_valid) // 10000)  # Downsample if too large
                        interp_elev = griddata(
                            (y_valid[::stride], x_valid[::stride]), 
                            z_valid[::stride], 
                            (y_grid, x_grid), 
                            method='nearest'
                        )
                    
                        diff = interp_elev - elev
                        diff[(diff > 100) | (diff < -100)] = 0
                        diff[~l_mask] = 0
                        diff[diff < 0] = 0
                        diff_ft = diff * 3.28084
                    
                        # Detect physical structure inside
                        core_mask = diff_ft > 5.0 # Height > 5ft
                    
                        physical_area_ft = np.sum(core_mask) * (3.28084*3.28084)
                    
                        # Size constraints based on the specific physical object, not the mask
                        if physical_area_ft < min_area_ft or physical_area_ft > max_area_ft:
                            continue
                        
                        peak_height_ft = float(np.max(diff_ft))
                    
                        # Get center cord
                        cy, cx = np.mean(region.coords, axis=0) # Relative to full image
                        pt_x, pt_y = src.xy(cy, cx)
                        lon, lat = trans.transform(pt_x, pt_y)
                    
                        # Get Mask bounding/length/width (The 'Cliff' / Red Box)
                        masked_length_ft = 0.0
                        masked_width_ft = 0.0
                        mask_contours = measure.find_contours(l_mask, 0.5)
                        if mask_contours:
                            mc = max(mask_contours, key=len)
                            if len(mc) > 2:
                                stride = max(1, len(mc) // 80)
                                mc_small = mc[::stride]
                                mdists = []
                                for pt in mc_small: mdists.append(np.linalg.norm(mc_small - pt, axis=1))
                                m_max_dist_idx = np.unravel_index(np.argmax(mdists), np.array(mdists).shape)
                                mpt1 = mc_small[m_max_dist_idx[0]]; mpt2 = mc_small[m_max_dist_idx[1]]
                                masked_length_ft = np.linalg.norm(mpt1 - mpt2) * 3.28084
                                m_line_vec = mpt2 - mpt1
                                if np.linalg.norm(m_line_vec) > 0:
                                    m_line_vec_norm = m_line_vec / np.linalg.norm(m_line_vec)
                                    m_perp_vec = np.array([-m_line_vec_norm[1], m_line_vec_norm[0]])
                                    m_projections = np.dot(mc - mpt1, m_perp_vec)
                                    masked_width_ft = (np.max(m_projections) - np.min(m_projections)) * 3.28084
                        masked_area_sq_feet = float(np.sum(l_mask) * (3.28084*3.28084))

                        # Get True Object bounding/length/width
                        object_length_ft = 0.0
                        object_width_ft = 0.0
                        pt1_plot = None
                        pt2_plot = None
                    
                        contours = measure.find_contours(core_mask, 0.5)
                        if contours:
                            c = max(contours, key=len)
                            if len(c) > 2:
                                stride = max(1, len(c) // 80)
                                c_small = c[::stride]
                                dists = []
                                for pt in c_small: dists.append(np.linalg.norm(c_small - pt, axis=1))
                                max_dist_idx = np.unravel_index(np.argmax(dists), np.array(dists).shape)
                                pt1 = c_small[max_dist_idx[0]]; pt2 = c_small[max_dist_idx[1]]
                                object_length_ft = np.linalg.norm(pt1 - pt2) * 3.28084
                                pt1_plot = pt1
                                pt2_plot = pt2
                            
                                line_vec = pt2 - pt1
                                if np.linalg.norm(line_vec) > 0:
                                    line_vec_norm = line_vec / np.linalg.norm(line_vec)
                                    perp_vec = np.array([-line_vec_norm[1], line_vec_norm[0]])
                                    projections = np.dot(c - pt1, perp_vec)
                                    object_width_ft = (np.max(projections) - np.min(projections)) * 3.28084

                        candidate = {
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'masked_area_sq_feet': masked_area_sq_feet,
                            'masked_length_feet': masked_length_ft,
                            'masked_width_feet': masked_width_ft,
                            'size_sq_feet': float(physical_area_ft),
                            'size_sq_meters': float(physical_area_ft / 10.7639),
                            'width_meters': float(object_width_ft / 3.28084),
                            'height_meters': float(peak_height_ft / 3.28084),
                            'length_feet': float(object_length_ft),
                            'width_feet': float(object_width_ft),
                            'peak_height_feet': peak_height_ft,
                            'confidence': 1.0, # Black holes that pass height constraints are highly anomalous
                            'method': 'deep_interpolation',
                            'location_status': '',
                            'location_accuracy': '',
                            'anomaly_score': 100.0,
                            'redaction_signatures': ['extreme_edge_variance_reconstructed']
                        }
                        candidates.append(candidate)
                    
                        if plot_visuals:
                            # Output interactive 3D HTML plot
                            try:
                                # Zoom limits to highlight the object
                                y_obj, x_obj = np.where(core_mask)
                                if len(y_obj) > 0:
                                    y_min, y_max = max(0, int(np.min(y_obj)) - 80), len(elev) - 1
                                    y_max = min(y_max, int(np.max(y_obj)) + 80)
                                    x_min, x_max = max(0, int(np.min(x_obj)) - 80), len(elev[0]) - 1
                                    x_max = min(x_max, int(np.max(x_obj)) + 80)
                                else:
                                    y_min, y_max = 0, elev.shape[0]-1
                                    x_min, x_max = 0, elev.shape[1]-1

                                # Downsample massive meshes for Plotly HTML so they don't break the browser
                                row_count = y_max - y_min + 1
                                col_count = x_max - x_min + 1
                                stride_y = max(1, row_count // 150)
                                stride_x = max(1, col_count // 150)

                                elev_crop = elev[y_min:y_max+1:stride_y, x_min:x_max+1:stride_x] * 3.28084 
                                interp_crop = interp_elev[y_min:y_max+1:stride_y, x_min:x_max+1:stride_x] * 3.28084 
                            
                                c_grid, r_grid = np.meshgrid(
                                    np.arange(c_start + x_min, c_start + x_max + 1, stride_x),
                                    np.arange(r_start + y_min, r_start + y_max + 1, stride_y)
                                )
                                x_proj, y_proj = src.transform * (c_grid, r_grid)
                                lon_grid, lat_grid = trans.transform(x_proj, y_proj)
                            
                                fig = go.Figure()
                            
                                # Lakebed Mesh
                                fig.add_trace(go.Surface(
                                    z=interp_crop, x=lon_grid, y=lat_grid,
                                    colorscale='Viridis', opacity=0.6,
                                    name='Natural Lakebed', showscale=False
                                ))
                            
                                # Actual Hole
                                fig.add_trace(go.Surface(
                                    z=elev_crop, x=lon_grid, y=lat_grid,
                                    colorscale='Hot', name='Target / Masked Surface'
                                ))
                            
                                title_text = f"Anomalous Volume at {lat:.5f}, {lon:.5f}<br>Object: Length {object_length_ft:.0f}ft Width {object_width_ft:.0f}ft Height {peak_height_ft:.1f}ft<br>Redacted Mask: Length {masked_length_ft:.0f}ft Width {masked_width_ft:.0f}ft Area {masked_area_sq_feet:.0f} sqft<br>Water Depth: {-np.max(interp_crop):.0f}ft"
                                if candidate['size_sq_feet'] > 5000:
                                    title_text = "<b>[WRECK DETECTION]</b> " + title_text
                                
                                lat_range = np.max(lat_grid) - np.min(lat_grid)
                                lon_range = np.max(lon_grid) - np.min(lon_grid)
                            
                                fig.update_layout(
                                    title=title_text,
                                    scene=dict(
                                        xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Depth (Feet)',
                                        aspectmode="manual",
                                        aspectratio=dict(x=1, y=lat_range/lon_range if lon_range>0 else 1, z=0.5)
                                    ),
                                    margin=dict(l=0, r=0, b=0, t=50)
                                )
                            
                                safe_name = Path(path).name.replace('.bag', '')
                                out_html = os.path.join(output_dir, f"{safe_name}_{idx}_anomaly.html")
                                fig.write_html(out_html)
                                candidate['plot_path'] = out_html
                            except Exception as plot_err:
                                logger.error(f"Plotting failed for anomaly {idx}: {plot_err}")

                file_res = {
                    'file': Path(path).name,
                    'candidates': candidates,
                    'total_candidates': len(candidates),
                    'success': True,
                    'error': None,
                    'processing_time_ms': int((datetime.now() - _start).total_seconds() * 1000)
                }
                results_list.append(file_res)
            
                # Write intermediate output so we can restart!
                with open(out_json_path, 'w') as jf:
                    import json
                    json.dump(file_res, jf)

            total_candidates += len(candidates)
        except Exception as e:
            logger.error(f"Failed black hole scan on {path}: {e}")
            file_res = {
                'file': Path(path).name,
                'candidates': [],
                'total_candidates': 0,
                'success': False,
                'error': str(e)
            }
            results_list.append(file_res)

    total_time = (datetime.now() - start_time).total_seconds()
    return {
        'results': results_list,
        'total_files': len(bag_files),
        'successful_scans': sum(1 for x in results_list if x['success']),
        'total_candidates': total_candidates,
        'total_signatures': 0,
        'processing_time_ms': int(total_time * 1000)
    }
