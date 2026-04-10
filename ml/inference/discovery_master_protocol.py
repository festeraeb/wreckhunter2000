import csv
import os
import json
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import math


def get_run_id():
    return 'RUN_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def init_run_folder(root=None):
    if root is None:
        root = Path(__file__).resolve().parents[2]
    run_id = get_run_id()
    run_dir = ensure_dir(Path(root) / 'outputs' / 'runs' / run_id)
    return run_id, run_dir


def haversine_distance_m(lat1, lon1, lat2, lon2):
    # meters
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c


def ensure_master_registry(ledger_path):
    ledger_path = Path(ledger_path)
    ensure_dir(ledger_path.parent)
    if not ledger_path.exists():
        with open(ledger_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['target_id','name','lat','lon','records'])
            writer.writeheader()
    return ledger_path


def find_or_create_ledger_entry(ledger_path, target_name, lat, lon, sensor_payload, run_id, timestamp):
    ledger_path = ensure_master_registry(ledger_path)
    rows = []
    matched = None
    with open(ledger_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r_lat = float(r.get('lat', '0'))
            r_lon = float(r.get('lon', '0'))
            if haversine_distance_m(lat, lon, r_lat, r_lon) <= 50.0:
                matched = r
            rows.append(r)

    if matched is None:
        target_id = f"TGT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        record = {
            'target_id': target_id,
            'name': target_name,
            'lat': f'{lat:.6f}',
            'lon': f'{lon:.6f}',
            'records': json.dumps([{'run_id': run_id, 'timestamp': timestamp, 'sensor_payload': sensor_payload}])
        }
        rows.append(record)
        matched = record
    else:
        # append to records list
        recs = json.loads(matched.get('records', '[]') or '[]')
        recs.append({'run_id': run_id, 'timestamp': timestamp, 'sensor_payload': sensor_payload})
        matched['records'] = json.dumps(recs)

    with open(ledger_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['target_id','name','lat','lon','records'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return matched


def increment_master_kml(master_path, placemarks, run_id, timestamp):
    master_path = Path(master_path)
    ensure_dir(master_path.parent)
    if master_path.exists():
        tree = ET.parse(master_path)
        root = tree.getroot()
    else:
        root = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
        document = ET.SubElement(root, 'Document')
        tree = ET.ElementTree(root)
        doc = document
        q = ET.SubElement(doc, 'name')
        q.text = 'discovery_master'
    # find Document element
    document = root.find('{http://www.opengis.net/kml/2.2}Document')
    if document is None:
        document = ET.SubElement(root, 'Document')
    for p in placemarks:
        pm = ET.SubElement(document, 'Placemark')
        nm = ET.SubElement(pm, 'name')
        nm.text = p.get('name', 'target')
        desc = ET.SubElement(pm, 'description')
        desc.text = f"Run: {run_id} | {timestamp} | {p.get('description','') }"
        point = ET.SubElement(pm, 'Point')
        coords = ET.SubElement(point, 'coordinates')
        coords.text = f"{p['lon']},{p['lat']},0"
    # write back, keep historical by writing new filename
    backup_path = master_path.parent / f"{master_path.stem}_{run_id}{master_path.suffix}"
    tree.write(str(backup_path), encoding='utf-8', xml_declaration=True)
    tree.write(str(master_path), encoding='utf-8', xml_declaration=True)
    return str(master_path), str(backup_path)
