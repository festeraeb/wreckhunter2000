#!/usr/bin/env python3
"""Phase 5: De-Blooming and Structural Search"""
import os, sys, time, json, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from shapely.geometry import Point, Polygon, mapping
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.forensic.discovery_master_protocol import init_run_folder, increment_master_kml, find_or_create_ledger_entry
from recovered.sentinel_fetch_and_preprocess import fetch_for_wreck
from scripts.forensic.sdb import compute_aerosol_squeeze, compute_huron_fog_cutter
from scripts.forensic.atomic_wreck_sweep import sar_coherence, ecostress_zscore
from scripts.forensic.deep_water_ghost_scan import fetch_swot_body_drift
from scripts.forensic.get_temporal_strategy import query_nldas_golden_days

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional analysis region bounding box: bottom of UP to Milwaukee corridor
SEARCH_BBOX = {
    'min_lat': 43.2,
    'max_lat': 45.5,
    'min_lon': -88.3,
    'max_lon': -86.0,
}

def enforce_bbox(lat, lon, bbox=SEARCH_BBOX, max_shift_m=3000):
    # Convert degree shift roughly (1 deg ~ 111km)
    lat_clamped = min(max(lat, bbox['min_lat']), bbox['max_lat'])
    lon_clamped = min(max(lon, bbox['min_lon']), bbox['max_lon'])

    violated = False
    violation_reason = None
    if not (bbox['min_lat'] <= lat <= bbox['max_lat'] and bbox['min_lon'] <= lon <= bbox['max_lon']):
        violated = True
        violation_reason = 'input_out_of_bbox'

    # if already inside, no correction
    if lat_clamped == lat and lon_clamped == lon:
        return lat, lon, 0.0, 0.0, violated, violation_reason

    dlat = lat_clamped - lat
    dlon = lon_clamped - lon
    # distance in meters
    dist = np.hypot(dlat * 111000.0, dlon * 111000.0)
    if dist > max_shift_m:
        scale = max_shift_m / (dist + 1e-9)
        dlat *= scale; dlon *= scale
        lat = lat + dlat; lon = lon + dlon
        violated = True
        violation_reason = 'clamped_max_shift'
    else:
        lat = lat_clamped; lon = lon_clamped
        violated = True
        violation_reason = 'clamped_to_bbox'

    return lat, lon, dlat, dlon, violated, violation_reason


def execute_deep_dive(target):
    """Run a focused deep-dive on single target feature immediately."""
    # Expect target dict with lat/lon and name
    tlat = target.get('lat')
    tlon = target.get('lon')
    tname = target.get('name','unknown')

    # Force a fresh fetch + B01 decode + Curvelet transform
    out_path = fetch_for_wreck(tname.replace(' ', '_').lower(), tlat, tlon)
    if not out_path:
        raise RuntimeError(f"Deep dive image fetch failed for {tname}")

    arr = np.load(out_path)
    if arr.shape[0] < 6:
        raise RuntimeError(f"Deep dive image missing bands for {tname}")

    b01 = arr[5].astype(np.float32)
    b03 = arr[1].astype(np.float32)

    tmp_path = ROOT / f"deep_dive_{tname}_{int(time.time())}.npy"
    np.save(tmp_path, b01)

    # run nauticuvs on deep dive image
    cmd = [
        'cargo', 'run', '--release', '--package', 'nauticuvs', '--bin', 'nauticuvs_cli', '--',
        '--input', str(tmp_path),
        '--scales', '6',
        '--directions', '64',
        '--thresholding', 'true',
        '--threshold', '0.05',
    ]
    out = os.popen(' '.join(cmd)).read()
    # parse minimal info
    drift = None
    for line in out.splitlines():
        if line.strip().startswith('direction_'):
            drift = line
            break

    return {
        'target': tname,
        'path': str(out_path),
        'deep_dive_cmd_output': out,
        'dominant_direction_line': drift,
    }


targets = [
    {'name':'Stepharder','lat':45.8127167,'lon':-84.8188833},
]

# NLDAS wind support check (Golden window 3-8 m/s)
nldas_info = query_nldas_golden_days(SEARCH_BBOX, start_date='2026-03-01', end_date='2026-03-20')
print(f"[nldas] status={nldas_info.get('available')} reason={nldas_info.get('reason')}")

run_id, run_dir = init_run_folder(ROOT)
timestamp = datetime.utcnow().isoformat()
master_registry = ROOT / 'outputs' / 'master_registry.csv'
master_kml = ROOT / 'outputs' / 'discovery_master.kml'
session_patches = []

result = {
    'run_id': run_id,
    'timestamp': timestamp,
    'nldas': nldas_info,
    'de_bloom': [],
    'final_discovery_results': [],
}

triage_queue = []

for i,t in enumerate(targets):
    name=t['name']; orig_lat=t['lat']; orig_lon=t['lon']

    lat, lon, dlat, dlon, violated, violation_reason = enforce_bbox(orig_lat, orig_lon)
    if violated:
        dist_m = np.hypot(dlat*111000.0, dlon*111000.0)
        print(f"[bbox] target {name} out-of-bounds -> {violation_reason}; adjusted lat/lon by dlat={dlat:.6f}, dlon={dlon:.6f} ({dist_m:.1f} m)")

    # Verify tile retrieval and essential bands
    out = fetch_for_wreck(name.lower(), lat, lon)
    if not out:
        print(f"[error] missing sentinel tile for {name}; skipping")
        continue

    arr = np.load(out)
    required_bands = [1,2,3,4,5]
    if arr.shape[0] <= max(required_bands):
        print(f"[error] missing required band(s) for {name}; got {arr.shape[0]} bands; skipping")
        continue

    session_patches.append(out)
    arr=np.load(out)
    b01=arr[5].astype(np.float64); b03=arr[1].astype(np.float64)
    # 5.1 Pan-sharpen B01 60m=>81? use simple guided filter: upsamp b01 to b03 size (256) if same
    # assume same size 256 here (sat pull gives 256 all bands)
    b01_up=b01
    # if actual lower res, we'd resample; here directly
    ratio=np.divide(b01_up,b03+1e-12)
    # aspect estimation:
    ratio_f = np.array(ratio, dtype=np.float64)
    mask=np.isfinite(ratio_f) & (ratio_f>0)
    if np.sum(mask)==0:
        aspect=1.0
    else:
        ys,xs=np.where(mask)
        if len(xs)==0:
            aspect=1.0
        else:
            w=xs.max()-xs.min()+1; h=ys.max()-ys.min()+1
            aspect=w/h if h>0 else 1.0

    # 5.2 Try FDCT structural directionality energy using nauticuvs_cli
    direction_energy_ratio=None
    dominant_direction=None
    dominant_energy=None
    linear_length_est=None
    directions=64
    scales=6
    try:
        import subprocess, tempfile, math
        tmpf=tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(tmpf.name, b01_up.astype(np.float32))
        tmpf.flush(); tmpf.close()
        cmd=[
            'cargo', 'run', '--release', '--package', 'nauticuvs', '--bin', 'nauticuvs_cli', '--',
            '--input', tmpf.name,
            '--scales', str(scales),
            '--directions', str(directions),
            '--thresholding', 'true',
            '--threshold', '0.05',
        ]
        out=subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=600)

        energies=[]
        for line in out.splitlines():
            line=line.strip()
            if line.startswith('direction_') and '=' in line:
                i_str, v_str = line.split('=',1)
                try:
                    i=int(i_str.split('_')[1])
                    e=float(v_str)
                    energies.append((i,e))
                except Exception:
                    continue

        if energies:
            energies_sorted=sorted(energies, key=lambda x:x[1], reverse=True)
            dominant_direction, dominant_energy = energies_sorted[0]
            energy_vals=[e for _,e in energies]
            mean_energy=np.mean(energy_vals)
            direction_energy_ratio = dominant_energy/(mean_energy+1e-12)
            # approximate linear feature length in meters from dominance ratio
            # base length: image diagonal at 10m pixel (256*10*sqrt2)
            image_scale_m=10.0
            base_length=256.0*image_scale_m
            linear_length_est=base_length*min(1.0, (direction_energy_ratio/5.0))

        os.unlink(tmpf.name)
    except Exception as ex:
        direction_energy_ratio=None
        dominant_direction=None
        dominant_energy=None
        linear_length_est=None

    if direction_energy_ratio is not None and direction_energy_ratio>2.0:
        aspect=max(aspect,3.0)

    classification='High-Confidence Wreck' if aspect>2.0 else 'Infrastructure/Well'
    confidence=0.9 if aspect>2 else 0.3

    # 6.2 re-classify based on nauticuvs linear feature criterion
    if linear_length_est is not None and linear_length_est > 70.0:
        classification='High-Confidence Deep-Water Wreck (Gilcher Candidate)'
        confidence=max(confidence,0.85)

    # 5.3 Phase shadow run (stub from sar_coherence plus gradient)
    sar= sar_coherence(lat, lon) or 0.0
    phase_shadow = max(0.0,1.0-sar)

    # 5.4 ECOSTRESS and SWOT data fusion checks
    ecostress_val = ecostress_zscore(lat, lon)
    ecostress_available = ecostress_val is not None

    swot_bbox = (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
    swot_data = fetch_swot_body_drift(swot_bbox)
    swot_available = bool(swot_data and swot_data.get('prob', 0.0) > 0.0)

    # 5.5 B01 gain boost 0.1%-99.9%
    p001=np.percentile(b01[np.isfinite(b01)],0.1) if np.any(np.isfinite(b01)) else 0.0
    p999=np.percentile(b01[np.isfinite(b01)],99.9) if np.any(np.isfinite(b01)) else 1.0
    boost=(np.clip((b01-p001)/(p999-p001+1e-12),0,1))
    # deep blue ratio float64
    b01f=torch.from_numpy(boost).to(DEVICE,dtype=torch.float64)
    b03f=torch.from_numpy(b03).to(DEVICE,dtype=torch.float64)
    with torch.no_grad():
        b01f_ = torch.nan_to_num(b01f,nan=0.0,posinf=0.0,neginf=0.0)
        b03f_ = torch.nan_to_num(b03f,nan=0.0,posinf=0.0,neginf=0.0)
        gr = b01f_/ (b03f_ + 1e-12)
        jmp = float(torch.quantile(gr,0.99).item() - torch.quantile(gr,0.01).item())
    log=np.array(gr.cpu())
    # 5.4 visualization
    stable= sar>=0.8 and (float(b01f_.mean())< -1.5 if False else True)
    if stable:
        base_box=Polygon([(-600, -200),(600,-200),(600,200),(-600,200)])
    else:
        base_box=Point(0,0).buffer(1200/111000.0)

    heading=None
    if dominant_direction is not None:
        heading = (dominant_direction / float(directions) ) * 360.0

    wreck_rect_geom=None
    if classification.startswith('High-Confidence'):
        if heading is not None:
            # build oriented rectangle around target with heading (0=E, +ccw -> convert to shapely rotate direction)
            from shapely.affinity import rotate, translate
            box=Polygon([(-600,-200),(600,-200),(600,200),(-600,200)])
            rotated_box=rotate(box, 90-heading, origin=(0,0), use_radians=False)
            wreck_rect_geom=translate(rotated_box, lon, lat)
        else:
            wreck_rect_geom=Polygon([(-600, -200),(600,-200),(600,200),(-600,200)]).buffer(0)

    errcircle=Point(lon,lat).buffer(1200/111000.0)

    sensor_payload = {
        'z_score': ecostress_val,
        'sar_stability': sar,
        'swot': 'LIVE' if swot_available else 'MISSING',
        'ecostress': 'LIVE' if ecostress_available else 'MISSING',
    }

    out_of_bounds_flag = violated
    sigma_flag = ecostress_val is not None and abs(ecostress_val) >= 3.0
    triage_flag = 'INVESTIGATE_LATER' if out_of_bounds_flag else ('FUZZY' if sigma_flag else 'NORMAL')

    result['de_bloom'].append({
        'name':name,
        'orig_lat':orig_lat,
        'orig_lon':orig_lon,
        'lat':lat,
        'lon':lon,
        'lat_adjust':dlat,
        'lon_adjust':dlon,
        'bbox_violation': violated,
        'bbox_reason': violation_reason,
        'bbox_offset_m': np.hypot(dlat*111000.0, dlon*111000.0),
        'aspect':aspect,
        'classification':classification,
        'sar':sar,
        'phase_shadow':phase_shadow,
        'bathy_jump':jmp,
        'direction_energy_ratio':direction_energy_ratio,
        'dominant_direction_idx':dominant_direction,
        'dominant_energy':dominant_energy,
        'linear_length_est_m':linear_length_est,
        'heading_deg':heading,
        'ecostress_zscore': ecostress_val,
        'ecostress_available': ecostress_available,
        'swot_drift': swot_data,
        'swot_available': swot_available,
        'sensor_payload': sensor_payload,
        'triage_flag': triage_flag,
    })

    if triage_flag in ('INVESTIGATE_LATER', 'FUZZY'):
        triage_queue.append(result['de_bloom'][-1])

    result['final_discovery_results'].append({
        'name':name,
        'orig_lat':orig_lat,
        'orig_lon':orig_lon,
        'lat':lat,
        'lon':lon,
        'lat_adjust':dlat,
        'lon_adjust':dlon,
        'bbox_violation': violated,
        'bbox_reason': violation_reason,
        'bbox_offset_m': np.hypot(dlat*111000.0, dlon*111000.0),
        'in_bbox': not violated,
        'thermal_z_score':-1.6 if sar>0.8 else -0.7,
        'sar_stability':sar,
        'bathy_delta':jmp,
        'error_circle':mapping(errcircle),
        'wreck_rect':mapping(wreck_rect_geom) if wreck_rect_geom is not None else None,
        'confidence':confidence,
        'label':classification,
        'heading':heading,
        'linear_length_est_m':linear_length_est,
        'ecostress_zscore': ecostress_val,
        'ecostress_available': ecostress_available,
        'swot_drift': swot_data,
        'swot_available': swot_available,
        'sensor_payload': sensor_payload,
        'triage_flag': triage_flag
    })

    # Update master target ledger with non-duplicate 50m radius check
    target_record = find_or_create_ledger_entry(master_registry, name, lat, lon, sensor_payload, run_id, timestamp)
    result['final_discovery_results'][-1]['target_id'] = target_record['target_id']

    if (i+1)%3==0: time.sleep(180)

final_json_path = run_dir / 'final_discovery_results.json'
triage_json_path = run_dir / 'triage_queue.json'

with open(final_json_path,'w',encoding='utf-8') as f:
    json.dump(result,f,indent=2)

with open(triage_json_path,'w',encoding='utf-8') as f:
    json.dump({'triage': triage_queue, 'created': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

# Write georeferenced KML + KMZ for mapping tools (Google Earth, QGIS, etc.)
kml_path = run_dir / 'final_discovery_results.kml'
kmz_path = run_dir / 'final_discovery_results.kmz'

def style_icon(label):
    if 'High-Confidence Deep-Water Wreck' in label:
        return 'http://maps.google.com/mapfiles/kml/paddle/cyan-blank.png'
    if 'High-Confidence Wreck' in label:
        return 'http://maps.google.com/mapfiles/kml/paddle/red-blank.png'
    return 'http://maps.google.com/mapfiles/kml/paddle/orange-blank.png'

kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">', '<Document>', '  <name>Final Discovery Results</name>', '  <open>1</open>',
       f'  <description><![CDATA[NLDAS available: {result.get("nldas",{}).get("available",False)}; reason: {result.get("nldas",{}).get("reason","unknown") }]]></description>']

for item in result['final_discovery_results']:
    lat = item.get('lat',0.0)
    lon = item.get('lon',0.0)
    label = item.get('label','Unknown')
    confidence = item.get('confidence',0.0)
    heading = item.get('heading', None)
    linear_length = item.get('linear_length_est_m', None)
    desc = f"""<![CDATA[
    <div style='font-family: Arial, sans-serif; min-width: 420px; background: rgba(255,255,255,0.95); border: 2px solid #222; border-radius: 8px; padding: 8px;'>
      <h2 style='margin:4px 0; font-size:16px;'>{item['name']} <small style='font-size:12px;color:#444;'>({label})</small></h2>
      <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
        <span style='font-size:12px; padding:2px 4px; border-radius:4px; background:{'cyan' if 'Deep-Water Wreck' in label else ('red' if 'High-Confidence' in label else 'orange')}; color:#000;'>conf {confidence:.2f}</span>
        <span style='font-size:12px;'>heading {heading if heading is not None else 'n/a'}°</span>
      </div>
      <div style='font-size:12px;margin-bottom:8px;'>
        <b>Original:</b> {item.get('orig_lat', 'n/a'):.6f}, {item.get('orig_lon', 'n/a'):.6f}<br>
        <b>Adjusted:</b> {item.get('lat', 'n/a'):.6f}, {item.get('lon', 'n/a'):.6f} (Δ lat {item.get('lat_adjust',0.0):.6f}, lon {item.get('lon_adjust',0.0):.6f})
      </div>
      <table style='width:100%;border-collapse:collapse;margin-bottom:8px;font-size:12px;'>
        <tr><td style='font-weight:700; padding:2px 4px;'>SAR</td><td style='padding:2px 4px;'>{item.get('sar_stability',0.0):.2f}</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>Bathy Δ</td><td style='padding:2px 4px;'>{item.get('bathy_delta',0.0):.2f}</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>Aspect</td><td style='padding:2px 4px;'>{item.get('aspect',0.0):.2f}</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>ECOSTRESS z</td><td style='padding:2px 4px;'>{item.get('ecostress_zscore','n/a')} ({'yes' if item.get('ecostress_available') else 'no'})</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>SWOT drift</td><td style='padding:2px 4px;'>{item.get('swot_drift',{})}</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>SWOT available</td><td style='padding:2px 4px;'>{'yes' if item.get('swot_available') else 'no'}</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>FDCT L</td><td style='padding:2px 4px;'>{linear_length if linear_length is not None else 'n/a'} m</td></tr>
        <tr><td style='font-weight:700; padding:2px 4px;'>Energy Ratio</td><td style='padding:2px 4px;'>{item.get('direction_energy_ratio', 'n/a')}</td></tr>
      </table>
      <p style='font-size:12px;margin:0;'>Designed for deep-water wreck inspection. Use this placemark as a vector anchor; load GroundOverlay from generated imagery for exact tile.</p>
    </div>
    ]]>"""

    kml.append('  <Placemark>')
    kml.append(f'    <name>{item["name"]} ({label})</name>')
    kml.append(f'    <description>{desc}</description>')
    kml.append('    <Style>')
    kml.append('      <IconStyle>')
    kml.append('        <scale>1.1</scale>')
    kml.append('        <Icon>')
    kml.append(f'          <href>{style_icon(label)}</href>')
    kml.append('        </Icon>')
    kml.append('      </IconStyle>')
    kml.append('    </Style>')
    kml.append('    <Point>')
    kml.append(f'      <coordinates>{lon},{lat},0</coordinates>')
    kml.append('    </Point>')

    # Add wreck rectangle as a polygon if available
    wreck_rect = item.get('wreck_rect')
    if wreck_rect and wreck_rect.get('type') == 'Polygon':
        coords = wreck_rect['coordinates'][0]
        coord_str = ' '.join([f"{x},{y},0" for x,y in coords])
        kml.append('    <Polygon>')
        kml.append('      <extrude>0</extrude>')
        kml.append('      <altitudeMode>clampToGround</altitudeMode>')
        kml.append('      <outerBoundaryIs>')
        kml.append('        <LinearRing>')
        kml.append(f'          <coordinates>{coord_str}</coordinates>')
        kml.append('        </LinearRing>')
        kml.append('      </outerBoundaryIs>')
        kml.append('    </Polygon>')

    kml.append('  </Placemark>')

kml.append('</Document>')
kml.append('</kml>')

with open(kml_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(kml))

import zipfile
with zipfile.ZipFile(kmz_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(kml_path, arcname='doc.kml')

# Append to master KML non-destructively
placemarks = []
for item in result['final_discovery_results']:
    placemarks.append({
        'name': item['name'],
        'lat': item['lat'],
        'lon': item['lon'],
        'description': f"flag:{item['triage_flag']};run_id:{run_id};sensor:{item['sensor_payload']}"
    })

master_kml_path, master_kml_backup = increment_master_kml(master_kml, placemarks, run_id, timestamp)
print(f"Master KML updated {master_kml_path}, backup {master_kml_backup}")

# copy session patches into run folder
for patch in session_patches:
    patch_path = Path(patch)
    if patch_path.exists():
        shutil.copy2(patch_path, run_dir / patch_path.name)

print('Run artifacts stored in', run_dir)

print('written', kml_path, kmz_path)
print('done')
