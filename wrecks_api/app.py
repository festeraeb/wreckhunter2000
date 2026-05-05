"""
Swayze Great Lakes Wrecks - REST API
Serves the enhanced wrecks database (db/wrecks.db) via FastAPI.

Endpoints:
  GET /health
  GET /stats
  GET /wrecks            ?page&limit&lake&has_coords&magnetic_potential&is_steel&name
  GET /wrecks/{id}
  GET /wrecks/search     ?q=<text>&limit=50
  GET /wrecks/steel-freighters
  GET /wrecks/magnetic
"""
import os
import math
import json
import sqlite3
from typing import Optional
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
import sys
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# â”€â”€ Inject pipeline source directories so lazy `from X import Y` calls work â”€â”€
# Repo root = two levels up from wrecks_api/app.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("pipelines/mag", "pipelines/satellite", "pipelines/bag",
             "ml/training", "ml/inference", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel
import threading
import multiprocessing as mp
import uuid
import time

# Make root project importable for scanner
sys.path.insert(0, str(Path(__file__).parent.parent))
# Delay importing heavy scanner modules until job runtime to avoid import-time dependency errors
def _import_scanner():
    from bag_processor.rust_bag_scanner_runner import run_advanced_scan, find_bag_files
    return run_advanced_scan, find_bag_files


def _advanced_scan_worker(queue, paths: list, output_dir: str, config: dict):
    """Run advanced scan in a separate process so we can enforce a timeout."""
    try:
        run_advanced_scan, find_bag_files = _import_scanner()
        bag_files = find_bag_files(paths)
        results = run_advanced_scan(bag_files, output_dir, config or {})
        queue.put({"ok": True, "results": results})
    except Exception as e:
        queue.put({"ok": False, "error": str(e)})

# â”€â”€ DB path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inside Docker: DB is copied to /app/db/wrecks.db
# Local dev: relative path from project root
_HERE = Path(__file__).parent
_DB_PATH = os.environ.get("DB_PATH", str(_HERE.parent / "db" / "wrecks.db"))

# Base URL used in NetworkLink KML hrefs.
# Set API_BASE_URL to your Cloudflare/ngrok/tunnel public URL so Google Earth
# can reach the live feed from anywhere.  Falls back to localhost.
_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5001").rstrip("/")

app = FastAPI(
    title="Great Lakes Wrecks API",
    description="Enhanced Swayze Great Lakes shipwreck database â€” 9,784 wrecks with NAMAG magnetic signatures, steel freighter ML classification, and BGSU hull material data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# â”€â”€ DB helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@contextmanager
def get_db():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# â”€â”€ Scan audit database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_SCAN_AUDIT_DB_PATH = os.environ.get(
    "SCAN_AUDIT_DB_PATH",
    str(Path(__file__).parent.parent / "db" / "scan_audit.db"),
)


@contextmanager
def _get_scan_audit_db(write: bool = False):
    audit_path = Path(_SCAN_AUDIT_DB_PATH)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    init = not audit_path.exists()
    conn = sqlite3.connect(str(audit_path), check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    if write:
        conn.execute("PRAGMA journal_mode=WAL")
    else:
        conn.execute("PRAGMA query_only = ON")
    if init:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS scan_audit (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                status TEXT,
                paths TEXT,
                output_dir TEXT,
                config TEXT,
                swayze_match INTEGER,
                swayze_radius_m REAL,
                result_summary TEXT,
                total_candidates INTEGER,
                total_signatures INTEGER,
                error TEXT,
                research_mode INTEGER,
                research_notes TEXT
            );
        ''')
        conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _update_scan_audit(job_id: str, **fields):
    if not fields:
        return
    with _get_scan_audit_db(write=True) as conn:
        placeholders = ", ".join([f"{k}=?" for k in fields.keys()])
        params = list(fields.values()) + [job_id]
        conn.execute(f"UPDATE scan_audit SET {placeholders} WHERE id=?", params)
        conn.commit()


def _insert_scan_audit(job_id: str, created_at: str, paths: list, output_dir: str, config: dict,
                       swayze_match: bool, swayze_radius_m: float,
                       research_mode: bool, research_notes: str):
    with _get_scan_audit_db(write=True) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scan_audit (id, created_at, status, paths, output_dir, config, swayze_match, swayze_radius_m, research_mode, research_notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                created_at,
                'queued',
                json.dumps(paths),
                output_dir,
                json.dumps(config or {}),
                int(bool(swayze_match)),
                float(swayze_radius_m),
                int(bool(research_mode)),
                research_notes or "",
            ),
        )
        conn.commit()


# â”€â”€ Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/health", tags=["meta"])
def health():
    with get_db() as conn:
        conn.execute("SELECT 1 FROM features LIMIT 1")
    return {"status": "ok", "db": _DB_PATH}


# â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/stats", tags=["meta"])
def stats():
    with get_db() as conn:
        c = conn.cursor()
        def q(sql): c.execute(sql); return c.fetchone()[0]
        return {
            "total_wrecks":          q("SELECT COUNT(*) FROM features"),
            "with_coordinates":      q("SELECT COUNT(*) FROM features WHERE latitude IS NOT NULL"),
            "with_hull_material":    q("SELECT COUNT(*) FROM features WHERE hull_material IS NOT NULL"),
            "with_namag_features":   q("SELECT COUNT(*) FROM features WHERE mag_mean IS NOT NULL"),
            "steel_freighters":      q("SELECT COUNT(*) FROM features WHERE is_steel_freighter=1"),
            "iron_ore_carriers":     q("SELECT COUNT(*) FROM features WHERE is_iron_ore_carrier=1"),
            "strong_mag_potential":  q("SELECT COUNT(*) FROM features WHERE magnetic_potential='strong'"),
            "moderate_mag_potential":q("SELECT COUNT(*) FROM features WHERE magnetic_potential='moderate'"),
        }


# â”€â”€ List wrecks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks", tags=["wrecks"])
def list_wrecks(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=500, description="Results per page"),
    name: Optional[str] = Query(None, description="Name contains (case-insensitive)"),
    has_coords: Optional[bool] = Query(None, description="Filter to wrecks with lat/lon"),
    is_steel: Optional[bool] = Query(None, description="Filter steel freighters only"),
    magnetic_potential: Optional[str] = Query(None, description="strong|moderate|weak|unknown"),
    has_mag: Optional[bool] = Query(None, description="Has NAMAG magnetic features"),
    min_lat: Optional[float] = Query(None, description="Bounding box south"),
    max_lat: Optional[float] = Query(None, description="Bounding box north"),
    min_lon: Optional[float] = Query(None, description="Bounding box west"),
    max_lon: Optional[float] = Query(None, description="Bounding box east"),
):
    where = []
    params = []

    if name:
        where.append("UPPER(name) LIKE UPPER(?)")
        params.append(f"%{name}%")
    if has_coords is True:
        where.append("latitude IS NOT NULL")
    elif has_coords is False:
        where.append("latitude IS NULL")
    if is_steel is True:
        where.append("is_steel_freighter=1")
    elif is_steel is False:
        where.append("(is_steel_freighter=0 OR is_steel_freighter IS NULL)")
    if magnetic_potential:
        where.append("magnetic_potential=?")
        params.append(magnetic_potential)
    if has_mag is True:
        where.append("mag_mean IS NOT NULL")
    elif has_mag is False:
        where.append("mag_mean IS NULL")
    if min_lat is not None:
        where.append("latitude >= ?"); params.append(min_lat)
    if max_lat is not None:
        where.append("latitude <= ?"); params.append(max_lat)
    if min_lon is not None:
        where.append("longitude >= ?"); params.append(min_lon)
    if max_lon is not None:
        where.append("longitude <= ?"); params.append(max_lon)

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    offset = (page - 1) * limit

    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) FROM features {where_clause}", params)
        total = c.fetchone()[0]

        c.execute(
            f"SELECT id,name,date,latitude,longitude,depth,feature_type,source,"
            f"magnetic_potential,is_steel_freighter,is_iron_ore_carrier,"
            f"hull_material,size_category,salvage_status,"
            f"mag_mean,mag_label,training_confidence,coord_quality "
            f"FROM features {where_clause} ORDER BY id LIMIT ? OFFSET ?",
            params + [limit, offset]
        )
        items = [row_to_dict(r) for r in c.fetchall()]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": math.ceil(total / limit),
        "results": items,
    }


# â”€â”€ Single wreck â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks/{wreck_id:int}", tags=["wrecks"])
def get_wreck(wreck_id: int):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM features WHERE id=?", (wreck_id,))
        row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Wreck {wreck_id} not found")
    return row_to_dict(row)


# â”€â”€ Full-text search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks/search/query", tags=["wrecks"])
def search_wrecks(
    q: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(50, ge=1, le=200),
):
    """Search across name and historical_place_names."""
    pat = f"%{q}%"
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id,name,date,latitude,longitude,depth,feature_type,"
            "magnetic_potential,is_steel_freighter,hull_material,"
            "historical_place_names,coord_quality "
            "FROM features "
            "WHERE UPPER(name) LIKE UPPER(?) "
            "   OR UPPER(historical_place_names) LIKE UPPER(?) "
            "ORDER BY "
            "   CASE WHEN UPPER(name) LIKE UPPER(?) THEN 0 ELSE 1 END, name "
            "LIMIT ?",
            (pat, pat, pat, limit)
        )
        results = [row_to_dict(r) for r in c.fetchall()]
    return {"query": q, "count": len(results), "results": results}


# â”€â”€ Steel freighters shortcut â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks/steel-freighters/list", tags=["wrecks"])
def steel_freighters(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
):
    offset = (page - 1) * limit
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM features WHERE is_steel_freighter=1")
        total = c.fetchone()[0]
        c.execute(
            "SELECT id,name,date,latitude,longitude,depth,"
            "magnetic_potential,hull_material,size_category,salvage_status,"
            "training_confidence,mag_mean,mag_label,coord_quality "
            "FROM features WHERE is_steel_freighter=1 "
            "ORDER BY training_confidence DESC NULLS LAST, name "
            "LIMIT ? OFFSET ?",
            (limit, offset)
        )
        items = [row_to_dict(r) for r in c.fetchall()]
    return {"total": total, "page": page, "limit": limit, "results": items}


# â”€â”€ NAMAG magnetic wrecks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks/magnetic/list", tags=["wrecks"])
def magnetic_wrecks(
    only_positive: bool = Query(False, description="Only mag_label=1 (positive anomaly)"),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
):
    offset = (page - 1) * limit
    where = "mag_mean IS NOT NULL"
    if only_positive:
        where += " AND mag_label=1"
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) FROM features WHERE {where}")
        total = c.fetchone()[0]
        c.execute(
            f"SELECT id,name,date,latitude,longitude,depth,magnetic_potential,"
            f"mag_mean,mag_std,mag_max,mag_min,mag_median,mag_label,"
            f"mag_as_peak,mag_vd_peak,mag_tmi_peak,mag_spike_w_m,mag_polarity,coord_quality "
            f"FROM features WHERE {where} "
            f"ORDER BY ABS(COALESCE(mag_mean,0)) DESC "
            f"LIMIT ? OFFSET ?",
            (limit, offset)
        )
        items = [row_to_dict(r) for r in c.fetchall()]
    return {"total": total, "page": page, "limit": limit, "results": items}


# â”€â”€ Bounding box spatial query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/wrecks/bbox/query", tags=["wrecks"])
def bbox_query(
    min_lat: float = Query(...),
    max_lat: float = Query(...),
    min_lon: float = Query(...),
    max_lon: float = Query(...),
    limit: int = Query(500, ge=1, le=2000),
):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id,name,date,latitude,longitude,depth,feature_type,"
            "magnetic_potential,is_steel_freighter,hull_material,mag_mean,coord_quality "
            "FROM features "
            "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? "
            "LIMIT ?",
            (min_lat, max_lat, min_lon, max_lon, limit)
        )
        results = [row_to_dict(r) for r in c.fetchall()]
    return {"bbox": [min_lat, min_lon, max_lat, max_lon], "count": len(results), "results": results}


# -------------------- Scan helpers ----------------------------------

def _match_swayze_wrecks(candidates: list, search_radius_m: float = 2000.0) -> list:
    """Cross-reference scan candidates against Swayze wrecks DB."""
    import math as _math
    matches = []
    lat_offset = search_radius_m / 111_000.0
    with get_db() as conn:
        for idx, cand in enumerate(candidates):
            lat, lon = cand.get('latitude', 0), cand.get('longitude', 0)
            if lat == 0 and lon == 0:
                continue
            lon_offset = search_radius_m / (111_000.0 * max(_math.cos(_math.radians(lat)), 0.01))
            rows = conn.execute(
                "SELECT id,name,date,latitude,longitude,depth,feature_type,"
                "hull_material,vessel_class,magnetic_weight,length_ft "
                "FROM features "
                "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? "
                "AND latitude IS NOT NULL AND longitude IS NOT NULL",
                (lat - lat_offset, lat + lat_offset, lon - lon_offset, lon + lon_offset),
            ).fetchall()
            for r in rows:
                rlat, rlon = r['latitude'], r['longitude']
                # Haversine
                dlat = _math.radians(rlat - lat)
                dlon = _math.radians(rlon - lon)
                a = _math.sin(dlat/2)**2 + _math.cos(_math.radians(lat)) * _math.cos(_math.radians(rlat)) * _math.sin(dlon/2)**2
                dist_m = 6_371_000 * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1-a))
                if dist_m > search_radius_m:
                    continue
                # Confidence scoring
                dist_score = 0.5 * (1 - min(dist_m / search_radius_m, 1.0))
                size_score = 0.2 if (r['length_ft'] or 0) > 50 else 0.1
                mat_score = 0.2 if (r['hull_material'] or '').lower() in ('steel', 'iron') else 0.1
                mag_score = 0.1 if (r['magnetic_weight'] or 0) > 30 else 0.0
                matches.append({
                    'candidate_index': idx,
                    'wreck_id': r['id'],
                    'name': r['name'],
                    'date': r['date'],
                    'latitude': rlat,
                    'longitude': rlon,
                    'depth': r['depth'],
                    'feature_type': r['feature_type'],
                    'hull_material': r['hull_material'],
                    'vessel_class': r['vessel_class'],
                    'magnetic_weight': r['magnetic_weight'],
                    'length_ft': r['length_ft'],
                    'distance_m': round(dist_m, 1),
                    'match_score': round(dist_score + size_score + mat_score + mag_score, 3),
                    'location_status': 'PREDICTED â€” true location unknown',
                    'location_accuracy': 'estimated',
                    'swayze_location_note': 'Swayze DB coordinates are estimated positions; true wreck location may differ',
                })
    # Sort best matches first
    matches.sort(key=lambda m: m['match_score'], reverse=True)
    return matches


def _write_scan_exports(output_dir: str, results: dict, swayze_matches: list):
    """Write JSON summary and CSV table to output dir."""
    import csv
    from pathlib import Path as _P
    _P(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    # â”€â”€ JSON summary â”€â”€
    export = {
        'scan_timestamp': ts,
        'total_files': results.get('total_files', 0),
        'successful_scans': results.get('successful_scans', 0),
        'total_candidates': results.get('total_candidates', 0),
        'total_signatures': results.get('total_signatures', 0),
        'swayze_matches': len(swayze_matches),
        'location_disclaimer': (
            'All candidate positions are PREDICTED / ESTIMATED from BAG anomaly detection. '
            'True wreck locations are unknown. Swayze DB positions are also estimated. '
            'Field verification is required to confirm any location.'
        ),
        'candidates': [],
        'matches': swayze_matches,
    }
    # Flatten candidates across files
    for fr in results.get('results', []):
        for c in fr.get('candidates', []):
            c_copy = {**c, 'source_file': fr.get('file', '')}
            # Attach matches for this candidate
            cidx = export['candidates'].__len__()
            c_copy['swayze_matches'] = [m for m in swayze_matches if m.get('candidate_index') == cidx]
            export['candidates'].append(c_copy)

    json_path = str(_P(output_dir) / f'scan_results_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(export, f, indent=2, default=str)

    # â”€â”€ CSV table â”€â”€
    csv_path = str(_P(output_dir) / f'scan_results_{ts}.csv')
    all_cands = export['candidates']
    if all_cands:
        fieldnames = ['source_file', 'latitude', 'longitude', 'location_status',
                      'location_accuracy', 'confidence',
                      'size_sq_meters', 'size_sq_feet', 'width_meters', 'height_meters',
                      'anomaly_score', 'method', 'best_swayze_match', 'match_score',
                      'match_distance_m']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for i, c in enumerate(all_cands):
                row = {k: c.get(k, '') for k in fieldnames}
                # Find best Swayze match for this candidate
                cand_matches = [m for m in swayze_matches if m.get('candidate_index') == i]
                if cand_matches:
                    best = cand_matches[0]
                    row['best_swayze_match'] = best.get('name', '')
                    row['match_score'] = best.get('match_score', '')
                    row['match_distance_m'] = best.get('distance_m', '')
                writer.writerow(row)

    return {'json': json_path, 'csv': csv_path}


# -------------------- Scan job APIs ---------------------------------


class ScanRequest(BaseModel):
    paths: list
    output_dir: str = "advanced_scan_results"
    config: dict = None
    swayze_match: bool = True
    swayze_radius_m: float = 2000.0
    research_mode: bool = False
    research_notes: str = ""


JOBS = {}


def _run_scan_job(job_id: str, paths: list, output_dir: str, config: dict,
                  swayze_match: bool = True, swayze_radius_m: float = 2000.0):
    JOBS[job_id]['status'] = 'running'
    JOBS[job_id]['start_time'] = time.time()
    _update_scan_audit(job_id, status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        cfg = config or {}

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_advanced_scan_worker, args=(q, paths, output_dir, cfg), daemon=True)
        p.start()
        p.join()  # no timeout â€” let scans run to completion

        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
        else:
            if not q.empty():
                msg = q.get()
                if msg.get("ok"):
                    results = msg.get("results")
                else:
                    raise RuntimeError(msg.get("error") or "Advanced scan worker failed")
            else:
                raise RuntimeError("Advanced scan worker returned no result")

        # â”€â”€ Stamp every candidate with location_status â”€â”€
        for fr in results.get('results', []):
            for cand in fr.get('candidates', []):
                cand['location_status'] = 'PREDICTED â€” true location unknown'
                cand['location_accuracy'] = 'estimated'

        # â”€â”€ Swayze cross-reference â”€â”€
        swayze_matches = []
        if swayze_match:
            all_candidates = []
            for fr in results.get('results', []):
                all_candidates.extend(fr.get('candidates', []))
            if all_candidates:
                swayze_matches = _match_swayze_wrecks(all_candidates, swayze_radius_m)

        # â”€â”€ Export JSON + CSV + already-generated KML/KMZ â”€â”€
        export_paths = _write_scan_exports(output_dir, results, swayze_matches)

        # Enrich the results object
        results['swayze_matches'] = swayze_matches
        results['export_files'] = export_paths
        # Collect KML/KMZ paths from individual scan results
        for fr in results.get('results', []):
            outputs = fr.get('outputs', {})
            if outputs.get('kml'):
                export_paths['kml'] = str(Path(output_dir) / outputs['kml']) if not os.path.isabs(outputs['kml']) else outputs['kml']
            if outputs.get('kmz'):
                export_paths['kmz'] = str(Path(output_dir) / outputs['kmz']) if not os.path.isabs(outputs['kmz']) else outputs['kmz']

        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result'] = results
        JOBS[job_id]['end_time'] = time.time()
        total_candidates = sum(len(fr.get('candidates', [])) for fr in results.get('results', []))
        total_signatures = results.get('total_signatures', 0) if isinstance(results.get('total_signatures', 0), int) else 0
        _update_scan_audit(
            job_id,
            status='completed',
            finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            result_summary=json.dumps({
                'results': results.get('results', []),
                'swayze_matches': results.get('swayze_matches', []),
            }),
            total_candidates=total_candidates,
            total_signatures=total_signatures,
            error=None,
        )
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        JOBS[job_id]['end_time'] = time.time()
        _update_scan_audit(
            job_id,
            status='failed',
            finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            error=str(e)[:2000],
        )


@app.post('/scan/start', tags=['scan'])
def start_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'id': job_id,
        'status': 'queued',
        'paths': req.paths,
        'output_dir': req.output_dir,
        'config': req.config,
        'research_mode': req.research_mode,
        'research_notes': req.research_notes,
        'created': time.time()
    }

    _insert_scan_audit(
        job_id=job_id,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        paths=req.paths,
        output_dir=req.output_dir,
        config=req.config or {},
        swayze_match=req.swayze_match,
        swayze_radius_m=req.swayze_radius_m,
        research_mode=req.research_mode,
        research_notes=req.research_notes,
    )

    # Start background thread
    t = threading.Thread(
        target=_run_scan_job,
        args=(job_id, req.paths, req.output_dir, req.config, req.swayze_match, req.swayze_radius_m),
        daemon=True,
    )
    t.start()

    return {'job_id': job_id, 'status': 'queued'}


@app.get('/scan/status/{job_id}', tags=['scan'])
def scan_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    # Provide lightweight status
    resp = {k: job[k] for k in ['id', 'status', 'created'] if k in job}
    if 'start_time' in job:
        resp['start_time'] = job['start_time']
    if 'end_time' in job:
        resp['end_time'] = job['end_time']
    if 'error' in job:
        resp['error'] = job['error']
    if 'pipeline' in job:
        resp['pipeline'] = job['pipeline']
    return resp


@app.get('/scan/results/{job_id}', tags=['scan'])
def scan_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('status') != 'completed':
        return {'status': job.get('status'), 'message': 'Results not yet available'}
    result = job.get('result', {})
    return {
        'status': 'completed',
        'result': result,
        'export_files': result.get('export_files', {}),
        'swayze_matches': result.get('swayze_matches', []),
        'total_candidates': result.get('total_candidates', 0),
        'total_signatures': result.get('total_signatures', 0),
    }


@app.get('/scan/audit/yesterday', tags=['scan'])
def scan_audit_yesterday():
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    start_ts = datetime(yesterday.year, yesterday.month, yesterday.day)
    end_ts = start_ts + timedelta(days=1)
    start_key = start_ts.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_key = end_ts.strftime('%Y-%m-%dT%H:%M:%SZ')

    with _get_scan_audit_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_audit WHERE created_at >= ? AND created_at < ? ORDER BY total_candidates DESC",
            (start_key, end_key)
        ).fetchall()

    tool_counts = {}
    candidate_list = []
    jobs = []

    for row in rows:
        result_summary = {}
        try:
            result_summary = json.loads(row['result_summary'] or '{}')
        except Exception:
            result_summary = {}

        total_candidates = row['total_candidates'] or 0
        total_signatures = row['total_signatures'] or 0
        jobs.append({
            'job_id': row['id'],
            'status': row['status'],
            'created_at': row['created_at'],
            'finished_at': row['finished_at'],
            'output_dir': row['output_dir'],
            'research_mode': bool(row['research_mode']),
            'research_notes': row['research_notes'] or '',
            'config': json.loads(row['config'] or '{}'),
            'total_candidates': total_candidates,
            'total_signatures': total_signatures,
        })

        for fr in result_summary.get('results', []):
            for cand in fr.get('candidates', []):
                method = cand.get('method') or cand.get('source_file') or 'unknown'
                tool_counts[method] = tool_counts.get(method, 0) + 1
                candidate_list.append({
                    'job_id': row['id'],
                    'source_file': fr.get('file') or cand.get('source_file') or 'unknown',
                    'latitude': cand.get('latitude'),
                    'longitude': cand.get('longitude'),
                    'confidence': cand.get('confidence'),
                    'anomaly_score': cand.get('anomaly_score'),
                    'method': method,
                    'size_sq_feet': cand.get('size_sq_feet'),
                    'size_sq_meters': cand.get('size_sq_meters'),
                })

    candidate_list.sort(key=lambda c: ((c['confidence'] or 0) * 1000 + (c['anomaly_score'] or 0)), reverse=True)
    top_candidates = candidate_list[:8]
    top_tools = [
        {'tool': tool, 'count': count}
        for tool, count in sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    ]

    return {
        'date': yesterday.isoformat(),
        'job_count': len(jobs),
        'total_candidates': sum(j['total_candidates'] for j in jobs),
        'total_signatures': sum(j['total_signatures'] for j in jobs),
        'top_tools': top_tools,
        'top_candidates': top_candidates,
        'research_jobs': jobs,
    }


@app.post('/scan/results/{job_id}/restore', tags=['scan'])
def scan_to_restore(job_id: str, candidate_index: int = 0):
    """Kick off restoration for a candidate found during a scan."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('status') != 'completed':
        raise HTTPException(status_code=400, detail='Scan not completed yet')
    result = job.get('result', {})
    # Find the source BAG file and candidate bounds
    file_results = result.get('results', [])
    cand_cursor = 0
    bag_path = None
    candidate = None
    for fr in file_results:
        cands = fr.get('candidates', [])
        for c in cands:
            if cand_cursor == candidate_index:
                candidate = c
                # Resolve BAG path from the scan paths
                for p in job.get('paths', []):
                    if fr.get('file', '') in p or os.path.isfile(p):
                        bag_path = p
                        break
                break
            cand_cursor += 1
        if candidate:
            break
    if not candidate or not bag_path:
        raise HTTPException(status_code=404, detail=f'Candidate {candidate_index} not found')

    # Start restoration job targeting the candidate region
    rest_job_id = str(uuid.uuid4())
    rest_output = str(Path(job.get('output_dir', 'advanced_scan_results')) / 'restoration')
    TOOL_JOBS[rest_job_id] = {
        'id': rest_job_id,
        'tool': 'bag_restoration',
        'status': 'queued',
        'bag_path': bag_path,
        'output_dir': rest_output,
        'source_scan_job': job_id,
        'candidate_index': candidate_index,
        'location_status': candidate.get('location_status', 'PREDICTED â€” true location unknown'),
        'location_accuracy': candidate.get('location_accuracy', 'estimated'),
        'created': time.time(),
    }
    req_dict = {
        'bag_path': bag_path,
        'output_dir': rest_output,
        'amplification': 3.0,
        'sigma': 2.0,
    }
    t = threading.Thread(target=_run_restoration_job, args=(rest_job_id, req_dict), daemon=True)
    t.start()
    return {'job_id': rest_job_id, 'status': 'queued', 'bag_path': bag_path, 'candidate': candidate}


# â”€â”€ Standalone tool endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# These let the frontend invoke Mag Pipeline independently
# of the BAG scan pipeline.

TOOL_JOBS = {}  # Separate dict for standalone tool runs


class SwarmMagRequest(BaseModel):
    start_time: str = "2023-05-01T00:00:00"
    end_time: str = "2023-05-07T00:00:00"
    bbox: list[float] = None
    output_file: str = "swarm_mag_data.csv"

def _run_swarm_mag_job(job_id: str, req: SwarmMagRequest):
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        import subprocess
        import sys
        
        cmd = [
            sys.executable, "tools/swarm_mag_fetcher.py",
            "--start", req.start_time,
            "--end", req.end_time,
            "--out", req.output_file
        ]
        if req.bbox and len(req.bbox) == 4:
            bbox_str = f"{req.bbox[0]},{req.bbox[1]},{req.bbox[2]},{req.bbox[3]}"
            cmd.extend(["--bbox", bbox_str])
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            TOOL_JOBS[job_id]['status'] = 'completed'
            TOOL_JOBS[job_id]['result'] = f"Data saved to {req.output_file}"
            TOOL_JOBS[job_id]['output'] = result.stdout
        else:
            TOOL_JOBS[job_id]['status'] = 'failed'
            TOOL_JOBS[job_id]['error'] = result.stderr or result.stdout
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
    TOOL_JOBS[job_id]['end_time'] = time.time()

@app.post('/tools/swarm-mag/fetch', tags=['tools'])
def fetch_swarm_mag(req: SwarmMagRequest):
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'swarm_mag',
        'status': 'queued',
        'output_file': req.output_file,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_swarm_mag_job,
        args=(job_id, req),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}

@app.get('/tools/swarm-mag/status/{job_id}', tags=['tools'])
def swarm_mag_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


class MagPipelineRequest(BaseModel):
    output_dir: str = "mag_pipeline_output"
    sources: list[str] = ["usgs_namag", "usgs_usmag"]
    bbox: list[float] = [-92.5, 41.0, -75.0, 49.0]
    stages: str = "all"
    threshold: float = 0.3
    mode: str = "full"  # "full" | "validate"
    config: dict = None


def _run_mag_pipeline_job(job_id: str, req: MagPipelineRequest):
    from wrecks_api.stages.mag_pipeline_stage import run_mag_pipeline_stage
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        cfg = req.config.copy() if req.config else {}
        cfg.setdefault("run_mag_pipeline", True)
        cfg.setdefault("sources", req.sources)
        cfg.setdefault("bbox", req.bbox)
        cfg.setdefault("stages", req.stages)
        cfg.setdefault("threshold", req.threshold)
        cfg.setdefault("mode", req.mode)
        result = run_mag_pipeline_stage([], req.output_dir, cfg)
        TOOL_JOBS[job_id]['status'] = result.get('status', 'completed')
        TOOL_JOBS[job_id]['result'] = result
        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/mag-pipeline/start', tags=['tools'])
def start_mag_pipeline(req: MagPipelineRequest):
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'mag_pipeline',
        'status': 'queued',
        'output_dir': req.output_dir,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_mag_pipeline_job,
        args=(job_id, req),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/mag-pipeline/status/{job_id}', tags=['tools'])
def mag_pipeline_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


@app.get('/tools/mag-data/status', tags=['tools'])
def mag_data_status():
    """Return status of the persistent magnetic data lake."""
    from mag_data_manager import data_status
    return data_status()


# â”€â”€ Mag detection labeling (for refinement training) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DetectionLabel(BaseModel):
    patch_file: str
    label: int  # 1 = wreck, 0 = not-wreck
    notes: str = ""


class DetectionLabelBatch(BaseModel):
    output_dir: str = "mag_pipeline_output"
    labels: list[DetectionLabel]


@app.get('/tools/mag-pipeline/detections', tags=['tools'])
def list_detections(output_dir: str = "mag_pipeline_output"):
    """Return current detections and any existing labels."""
    det_path = Path(output_dir) / "detections" / "detections.json"
    confirmed_path = Path(output_dir) / "detections" / "confirmed_detections.json"

    detections = []
    if det_path.exists():
        with open(det_path) as f:
            detections = json.load(f)

    confirmed = {}
    if confirmed_path.exists():
        with open(confirmed_path) as f:
            for item in json.load(f):
                confirmed[item.get("patch_file", "")] = item

    # Merge label info into detections
    for d in detections:
        pf = d.get("patch_file", "")
        if pf in confirmed:
            d["label"] = confirmed[pf].get("label")
            d["notes"] = confirmed[pf].get("notes", "")
        else:
            d["label"] = None

    return {"detections": detections, "total": len(detections),
            "labeled": sum(1 for d in detections if d.get("label") is not None)}


@app.post('/tools/mag-pipeline/label', tags=['tools'])
def label_detections(req: DetectionLabelBatch):
    """Save user labels for detections (wreck / not-wreck) for refinement training."""
    confirmed_path = Path(req.output_dir) / "detections" / "confirmed_detections.json"
    confirmed_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    existing = []
    if confirmed_path.exists():
        with open(confirmed_path) as f:
            existing = json.load(f)

    # Index by patch_file for upsert
    by_patch = {item["patch_file"]: item for item in existing}
    for lbl in req.labels:
        by_patch[lbl.patch_file] = {
            "patch_file": lbl.patch_file,
            "label": lbl.label,
            "notes": lbl.notes,
            "labeled_at": time.time(),
        }

    merged = list(by_patch.values())
    with open(confirmed_path, "w") as f:
        json.dump(merged, f, indent=2)

    return {
        "saved": len(req.labels),
        "total_labeled": len(merged),
        "confirmed_path": str(confirmed_path),
    }


# â”€â”€ Lake Erie Focused Scanner endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ErieScanRequest(BaseModel):
    candidates_csv: str = None
    wells_csv: str = None
    output_dir: str = "erie_scanner_output"
    wellhead_radius_m: float = 2000.0
    satellite_sources: list[str] = None
    apply_loran_correction: bool = True
    retrain: bool = False


def _run_erie_scan_job(job_id: str, req: ErieScanRequest):
    """Run Lake Erie focused scanner in a background thread."""
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        from erie_scanner_pipeline import run_erie_scan
        result = run_erie_scan(
            candidates_csv=req.candidates_csv,
            wells_csv=req.wells_csv,
            output_dir=req.output_dir,
            wellhead_radius_m=req.wellhead_radius_m,
            satellite_sources=req.satellite_sources,
            apply_loran_correction=req.apply_loran_correction,
            retrain=req.retrain,
        )
        if "error" in result:
            TOOL_JOBS[job_id]['status'] = 'failed'
            TOOL_JOBS[job_id]['error'] = result['error']
        else:
            TOOL_JOBS[job_id]['status'] = 'completed'
            TOOL_JOBS[job_id]['result'] = result
        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/erie-scanner/start', tags=['tools'])
def start_erie_scan(req: ErieScanRequest):
    """Start Lake Erie focused scanner with wellhead discrimination."""
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'erie_scanner',
        'status': 'queued',
        'output_dir': req.output_dir,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_erie_scan_job,
        args=(job_id, req),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/erie-scanner/status/{job_id}', tags=['tools'])
def erie_scan_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


@app.get('/tools/erie-scanner/results', tags=['tools'])
def erie_scan_results(output_dir: str = "erie_scanner_output"):
    """Return latest Erie scanner results."""
    results_path = Path(output_dir) / "erie_scan_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail='No Erie scan results found. Run a scan first.')
    with open(results_path) as f:
        return json.load(f)


@app.get('/tools/erie-scanner/wellheads', tags=['tools'])
def erie_wellheads(wells_csv: str = None, lake_erie_only: bool = True):
    """Return loaded wellhead positions for map overlay."""
    from erie_wellhead_discriminator import load_ogsr_wells
    from dataclasses import asdict
    if wells_csv:
        wells = load_ogsr_wells(wells_csv, lake_erie_only=lake_erie_only)
    else:
        for p in [Path("data/wells.csv"), Path("reference/wells.csv"), Path("erie_scanner_output/wells.csv")]:
            if p.exists():
                wells = load_ogsr_wells(p, lake_erie_only=lake_erie_only)
                break
        else:
            wells = []
    return {"wells": [asdict(w) for w in wells[:5000]], "total": len(wells)}


@app.get('/tools/erie-scanner/known-wrecks', tags=['tools'])
def erie_known_wrecks():
    """Return compiled known wreck positions for Lake Erie."""
    from erie_wellhead_discriminator import get_all_known_wrecks
    from dataclasses import asdict
    wrecks = get_all_known_wrecks()
    return {"wrecks": [asdict(w) for w in wrecks], "total": len(wrecks)}


# â”€â”€ Lake Erie XGBoost Training endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ErieTrainRequest(BaseModel):
    candidates_csv: str = None
    wells_csv: str = None
    output_dir: str = "models/erie"
    n_synth_wreck: int = 10000
    n_synth_wellhead: int = 3000
    n_synth_geological: int = 2000


def _run_erie_training_job(job_id: str, req: ErieTrainRequest):
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        from train_lake_erie_offaxis import train_all_models
        result = train_all_models(
            candidates_csv=req.candidates_csv,
            wells_csv=req.wells_csv,
            output_dir=req.output_dir,
            n_synth_wreck=req.n_synth_wreck,
            n_synth_wellhead=req.n_synth_wellhead,
            n_synth_geological=req.n_synth_geological,
        )
        if "error" in result:
            TOOL_JOBS[job_id]['status'] = 'failed'
            TOOL_JOBS[job_id]['error'] = result['error']
        else:
            TOOL_JOBS[job_id]['status'] = 'completed'
            TOOL_JOBS[job_id]['result'] = result
        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/erie-scanner/train', tags=['tools'])
def start_erie_training(req: ErieTrainRequest):
    """Start XGBoost model training for Lake Erie off-axis detector."""
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'erie_training',
        'status': 'queued',
        'output_dir': req.output_dir,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_erie_training_job,
        args=(job_id, req),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/erie-scanner/training-status/{job_id}', tags=['tools'])
def erie_training_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


@app.get('/tools/erie-scanner/training-report', tags=['tools'])
def erie_training_report(output_dir: str = "models/erie"):
    """Return the latest training report."""
    report_path = Path(output_dir) / "training_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail='No training report found. Train models first.')
    with open(report_path) as f:
        return json.load(f)


@app.get('/tools/erie-scanner/feedback-report', tags=['tools'])
def erie_feedback_report(model_dir: str = "models/erie", feedback_dir: str = "models/erie/feedback"):
    """Return the feedback loop status and label summary."""
    from erie_feedback_loop import feedback_report
    return feedback_report(model_dir, feedback_dir)


# â”€â”€ BAG Depth Restoration endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Operates on BAG depth grids directly â€” no PDF dependency.
# PDFs are optional cross-reference only.

class RestorationRequest(BaseModel):
    bag_path: str
    output_dir: str = "restoration_output"
    # Bounding box to restrict restoration (optional, otherwise full grid)
    min_row: int = None
    max_row: int = None
    min_col: int = None
    max_col: int = None
    # Restoration parameters
    amplification: float = 3.0
    sigma: float = 2.0
    techniques: list = None  # None = run all


def _run_restoration_job(job_id: str, req_dict: dict):
    """Run BAG depth restoration in a background thread."""
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        import numpy as np
        import rasterio
        import json as _json
        from bag_processor.unmasking_restoration import (
            multi_technique_restore,
            over_exaggerate_smoothing,
        )
        from pathlib import Path as _Path

        bag_path = req_dict['bag_path'].strip().strip('"').strip("'")
        output_dir = req_dict['output_dir'].strip().strip('"').strip("'")
        _Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Read the BAG depth grid
        with rasterio.open(bag_path) as src:
            elev = src.read(1).astype('float64')
            nodata = src.nodata
            if nodata is not None:
                elev[elev == nodata] = np.nan

        # Optional sub-region
        r0 = req_dict.get('min_row') or 0
        r1 = req_dict.get('max_row') or elev.shape[0]
        c0 = req_dict.get('min_col') or 0
        c1 = req_dict.get('max_col') or elev.shape[1]
        sub = elev[r0:r1, c0:c1]

        # Build mask: areas with suspiciously low variance (likely masked)
        from scipy import ndimage
        local_std = ndimage.generic_filter(
            np.where(np.isfinite(sub), sub, 0.0), np.nanstd, size=15
        )
        mask = (local_std < 0.1) & np.isfinite(sub)

        if mask.sum() == 0:
            # Fallback: use all valid data as "mask" for exploration
            mask = np.isfinite(sub)

        # Run all techniques
        results = multi_technique_restore(
            sub, mask,
            sigma=req_dict.get('sigma', 2.0),
            amplification=req_dict.get('amplification', 3.0),
        )

        bag_stem = _Path(bag_path).stem
        output_files = {}

        for technique, restored in results.items():
            # Save as numpy binary for downstream use
            npy_path = str(_Path(output_dir) / f"{bag_stem}_{technique}.npy")
            np.save(npy_path, restored)
            output_files[technique] = npy_path

            # Also save a PNG visualization
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                vmin = np.nanpercentile(sub, 2)
                vmax = np.nanpercentile(sub, 98)

                axes[0].imshow(sub, cmap='terrain', vmin=vmin, vmax=vmax)
                axes[0].set_title(f'Before â€” {bag_stem}')
                axes[0].axis('off')

                axes[1].imshow(restored, cmap='terrain', vmin=vmin, vmax=vmax)
                axes[1].set_title(f'After â€” {technique}')
                axes[1].axis('off')

                png_path = str(_Path(output_dir) / f"{bag_stem}_{technique}.png")
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                output_files[f"{technique}_png"] = png_path
            except Exception:
                pass  # matplotlib optional for visualization

        TOOL_JOBS[job_id]['status'] = 'completed'
        TOOL_JOBS[job_id]['result'] = {
            'bag_path': bag_path,
            'region': {'rows': [r0, r1], 'cols': [c0, c1]},
            'mask_pixels': int(mask.sum()),
            'techniques_run': list(results.keys()),
            'output_files': output_files,
        }
        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/restoration/start', tags=['tools'])
def start_restoration(req: RestorationRequest):
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'bag_restoration',
        'status': 'queued',
        'bag_path': req.bag_path,
        'output_dir': req.output_dir,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_restoration_job,
        args=(job_id, req.dict()),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/restoration/status/{job_id}', tags=['tools'])
def restoration_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


# â”€â”€ Azure AI Vision endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get('/tools/azure-vision/status', tags=['tools'])
def azure_vision_status():
    """Check whether Azure AI Vision is configured and reachable."""
    from bag_processor.azure_vision_analyzer import check_available
    return check_available()


class VisionConfigureRequest(BaseModel):
    key: str
    endpoint: str = "https://wreckhunter2000.cognitiveservices.azure.com/"
    region: str = "eastus"


@app.post('/tools/azure-vision/configure', tags=['tools'])
def configure_azure_vision(req: VisionConfigureRequest):
    """Save the Azure Vision API key (persisted to .azure_vision_key)."""
    from bag_processor.azure_vision_analyzer import set_key, check_available
    if not req.key or len(req.key) < 10:
        raise HTTPException(status_code=400, detail='Invalid API key')
    set_key(req.key, req.endpoint)
    return check_available()


class VisionAnalyzeRequest(BaseModel):
    output_dir: str
    bag_stem: str


def _run_vision_job(job_id: str, output_dir: str, bag_stem: str):
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        from bag_processor.azure_vision_analyzer import analyze_restoration_set
        results = analyze_restoration_set(output_dir, bag_stem)
        TOOL_JOBS[job_id]['status'] = 'completed'
        TOOL_JOBS[job_id]['result'] = results
        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/azure-vision/analyze', tags=['tools'])
def start_vision_analysis(req: VisionAnalyzeRequest):
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'azure_vision',
        'status': 'queued',
        'output_dir': req.output_dir,
        'bag_stem': req.bag_stem,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_vision_job,
        args=(job_id, req.output_dir, req.bag_stem),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/azure-vision/status/{job_id}', tags=['tools'])
def vision_analysis_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


# â”€â”€ Datum Correction / Loran-C Warp endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DatumCorrectionRequest(BaseModel):
    lat: float
    lon: float
    datum: str = "nad27"  # "nad27" | "wgs84"


class DatumBatchRequest(BaseModel):
    candidates: list       # list of {center_lat, center_lon, ...}
    datum: str = "nad27"
    input_file: str = ""   # optional: path to JSON candidates file


@app.post('/tools/datum/correct', tags=['tools'])
def datum_correct_single(req: DatumCorrectionRequest):
    """Apply NAD27â†’WGS84 Molodensky + Loran-C rubber-sheet correction to a single point."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from datum_correction import correct_candidate, load_anchors
    anchors = load_anchors()
    result = correct_candidate(req.lat, req.lon, anchors, req.datum)
    return result


@app.post('/tools/datum/batch', tags=['tools'])
def datum_correct_batch(req: DatumBatchRequest):
    """Apply datum correction to a batch of candidates."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from datum_correction import batch_correct, load_anchors
    anchors = load_anchors()

    candidates = req.candidates
    if not candidates and req.input_file:
        input_path = Path(req.input_file)
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f'File not found: {req.input_file}')
        candidates = json.loads(input_path.read_text())

    if not candidates:
        raise HTTPException(status_code=400, detail='No candidates provided')

    corrected = batch_correct(candidates, anchors, req.datum)
    return {
        'total': len(corrected),
        'datum': req.datum,
        'results': corrected,
    }


@app.get('/tools/datum/anchors', tags=['tools'])
def datum_list_anchors():
    """List all datum correction anchors."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from datum_correction import load_anchors
    anchors = load_anchors()
    n_ready = sum(1 for a in anchors if a.get("verified") and a.get("survey_pos"))
    return {
        'total': len(anchors),
        'ready': n_ready,
        'anchors': anchors,
    }


# â”€â”€ Extended Sensors (erie_remote) endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SensorRunRequest(BaseModel):
    lat: float
    lon: float
    amplitude_nt: float = 0.0
    depth_m: float = 24.0
    label: str = "GUI-candidate"
    sensors: list = []      # empty = "all"
    dry_run: bool = True    # default dry_run for GUI (no download)
    data_dir: str = "erie_remote_data"
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    earthdata_token: str = ""


def _run_sensor_job(job_id: str, req: SensorRunRequest):
    """Run erie_remote sensors in a background thread via subprocess."""
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        import subprocess
        root = Path(__file__).parent.parent
        erie_exe = root / "erie_remote" / "target" / "release" / "erie_remote.exe"
        if not erie_exe.exists():
            erie_exe = root / "erie_remote" / "target" / "debug" / "erie_remote.exe"

        data_dir = Path(req.data_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir

        # Build candidate JSON for the CLI to consume
        cand_file = data_dir / "gui_candidate.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        cand_data = {
            "lat": req.lat, "lon": req.lon,
            "amplitude_nt": req.amplitude_nt,
            "depth_m": req.depth_m,
            "label": req.label,
        }
        cand_file.write_text(json.dumps(cand_data, indent=2))

        if erie_exe.exists():
            # Use compiled binary
            cmd = [str(erie_exe), "--data-dir", str(data_dir), "all"]
            if req.dry_run:
                cmd.append("--dry-run")
            env = dict(os.environ)
            if req.earthdata_token:
                env["NASA_EARTHDATA_TOKEN"] = req.earthdata_token
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, env=env,
                cwd=str(root / "erie_remote"),
            )
            # Try to load combined report
            reports = list(data_dir.glob("combined_report_*.json"))
            if reports:
                reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                result = json.loads(reports[0].read_text())
            else:
                result = {"stdout": proc.stdout[-2000:] if proc.stdout else "",
                          "stderr": proc.stderr[-2000:] if proc.stderr else ""}

            TOOL_JOBS[job_id]['status'] = 'completed' if proc.returncode == 0 else 'failed'
            TOOL_JOBS[job_id]['result'] = result
            if proc.returncode != 0:
                TOOL_JOBS[job_id]['error'] = proc.stderr[-500:] if proc.stderr else f'exit code {proc.returncode}'
        else:
            TOOL_JOBS[job_id]['status'] = 'failed'
            TOOL_JOBS[job_id]['error'] = (
                f'erie_remote binary not found at {erie_exe}. '
                'Build with: cd erie_remote && cargo build --release'
            )

        TOOL_JOBS[job_id]['end_time'] = time.time()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
        TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/sensors/run', tags=['tools'])
def start_sensor_run(req: SensorRunRequest):
    """Run extended sensor pipeline against a candidate location."""
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'extended_sensors',
        'status': 'queued',
        'candidate': {'lat': req.lat, 'lon': req.lon, 'label': req.label},
        'created': time.time(),
    }
    t = threading.Thread(target=_run_sensor_job, args=(job_id, req), daemon=True)
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/sensors/status/{job_id}', tags=['tools'])
def sensor_run_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


@app.get('/tools/sensors/reports', tags=['tools'])
def list_sensor_reports(data_dir: str = "erie_remote_data"):
    """List available combined sensor reports."""
    root = Path(__file__).parent.parent
    d = Path(data_dir)
    if not d.is_absolute():
        d = root / d
    reports = []
    for p in sorted(d.glob("combined_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            reports.append({
                'file': str(p),
                'candidate': data.get('candidate', {}),
                'sensors_total': data.get('sensors_total', 0),
                'sensors_flagged': data.get('sensors_flagged', 0),
                'generated': data.get('generated_utc', ''),
            })
        except Exception:
            pass
    return {'reports': reports, 'total': len(reports)}


# ── PDF Breaker endpoints ─────────────────────────────────────────────────────

class PdfBreakerStartRequest(BaseModel):
    paths: list[str]
    output_dir: str = "pdf_breaker_output"
    config: dict = {}


def _run_pdf_breaker_job(job_id: str, req: PdfBreakerStartRequest):
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        from wrecks_api.stages.pdf_breaker_stage import run_pdf_breaker_stage
        cfg = dict(req.config)
        cfg['run_pdf_breaker'] = True
        result = run_pdf_breaker_stage(req.paths, req.output_dir, cfg)
        TOOL_JOBS[job_id]['status'] = result.get('status', 'completed')
        TOOL_JOBS[job_id]['result'] = result
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
    TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/pdf-breaker/start', tags=['tools'])
def start_pdf_breaker(req: PdfBreakerStartRequest):
    # Validate output_dir is not escaping to unexpected locations
    out = Path(req.output_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    if not str(out).startswith(str(root)):
        raise HTTPException(status_code=400, detail='output_dir must be within the project directory')

    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'pdf_breaker',
        'status': 'queued',
        'output_dir': req.output_dir,
        'created': time.time(),
    }
    t = threading.Thread(
        target=_run_pdf_breaker_job,
        args=(job_id, req),
        daemon=True,
    )
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/pdf-breaker/status/{job_id}', tags=['tools'])
def pdf_breaker_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: job[k] for k in job}


class AutoBagRequest(BaseModel):
    throttle_mode: str  # "unfettered", "half", "custom"
    custom_kbps: Optional[int] = None
    scan_mode: Optional[str] = "masked" # "masked", "unmasked", "both"

import subprocess

@app.post('/tools/auto-bag/start', tags=['tools'])
def start_auto_bag(req: AutoBagRequest):
    import time
    job_id = str(uuid.uuid4())

    # Validate scan_mode against allowlist
    allowed_scan_modes = {"masked", "unmasked", "both"}
    if req.scan_mode and req.scan_mode not in allowed_scan_modes:
        raise HTTPException(status_code=400, detail=f'Invalid scan_mode. Must be one of: {", ".join(sorted(allowed_scan_modes))}')

        # Calculate throttle if needed
    throttle_kbps = None
    if req.throttle_mode == "half":
        try:
            import time, requests
            from requests.exceptions import RequestException
            start_t = time.time()
            r = requests.get('https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/nos_hydro_dynamic/MapServer/0/query?f=json&where=1=1', timeout=10)
            elapsed = time.time() - start_t
            kb = len(r.content) / 1024
            speed_kbps = kb / max(elapsed, 0.01)
            throttle_kbps = int(speed_kbps / 2) # Half bandwidth
        except Exception:
            throttle_kbps = 2000 # Fallback
    elif req.throttle_mode == "custom" and req.custom_kbps:
        throttle_kbps = req.custom_kbps

    cmd = [sys.executable, "bag_auto_pipeline.py"]
    if throttle_kbps:
        cmd.extend(["--throttle-kbps", str(throttle_kbps)])
    if req.scan_mode:
        cmd.extend(["--scan-mode", req.scan_mode])

    # We use subprocess.Popen to let it run in background safely
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent.parent))
    
    TOOL_JOBS[job_id] = {
        'id': job_id,
        'tool': 'auto_bag_pipeline',
        'status': 'running',
        'pid': proc.pid,
        'throttle_mode': req.throttle_mode,
        'throttle_kbps': throttle_kbps,
        'created': time.time(),
    }
    return {'job_id': job_id, 'status': 'running', 'pid': proc.pid}

@app.get('/tools/auto-bag/status/{job_id}', tags=['tools'])
def auto_bag_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    
    # Check if process is still running (this is a simple un-managed PID check)
    try:
        import psutil
        if not psutil.pid_exists(job.get('pid')):
            job['status'] = 'completed'
    except Exception:
        pass
        
    return {k: job[k] for k in job}


# â”€â”€ Raw Harvester endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Manages the full raw-ping ingestion pipeline:
#   wh2k_warp_field_export.py  â†’ builds loran_warp_field.json (IDW grid)
#   wh2k_harvester.py          â†’ fetches NGDC/ScienceBase/NRCan/Swarm, applies warp
#
# Job lifecycle:  queued â†’ running â†’ completed|failed
# Progress stored per-source in job['pipeline'] so the UI can show per-source state.

class WarpFieldRequest(BaseModel):
    lake:       str   = "erie"
    spacing_km: float = 2.0


class HarvesterRequest(BaseModel):
    lake:        str        = "erie"
    sources:     list[str]  = ["ngdc", "sciencebase", "nrcan"]
    apply_warp:  bool       = True
    max_surveys: int        = 0        # 0 = unlimited
    dry_run:     bool       = False
    swarm_token: str        = ""


def _run_warp_export_job(job_id: str, req: WarpFieldRequest) -> None:
    """Background thread: run wh2k_warp_field_export.py and update job state."""
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    try:
        _root = Path(__file__).parent.parent
        sys.path.insert(0, str(_root / 'scripts'))
        from wh2k_warp_field_export import build_warp_field
        out_path = build_warp_field(
            lake=req.lake,
            spacing_km=req.spacing_km,
        )
        TOOL_JOBS[job_id]['status'] = 'completed'
        TOOL_JOBS[job_id]['result'] = {
            'warp_json': str(out_path),
            'lake': req.lake,
            'spacing_km': req.spacing_km,
            'size_kb': out_path.stat().st_size // 1024,
        }
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
    TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/warp-field/export', tags=['tools'])
def start_warp_export(req: WarpFieldRequest):
    """
    Export the LORAN-C IDW warp field to loran_warp_field.json.
    Must have verified anchors in scripts/datum_anchors.json first.
    """
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id, 'tool': 'warp_field_export',
        'status': 'queued', 'lake': req.lake, 'created': time.time(),
    }
    t = threading.Thread(target=_run_warp_export_job, args=(job_id, req), daemon=True)
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/warp-field/status/{job_id}', tags=['tools'])
def warp_export_status(job_id: str):
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {k: v for k, v in job.items()}


@app.get('/tools/warp-field/info', tags=['tools'])
def warp_field_info():
    """Return info about the currently built warp field, if any."""
    warp_path = Path(__file__).parent.parent / 'scripts' / 'loran_warp_field.json'
    if not warp_path.exists():
        return {'exists': False, 'message': 'Run /tools/warp-field/export first.'}
    try:
        with open(warp_path, encoding='utf-8') as f:
            wf = json.load(f)
        return {
            'exists': True,
            'generated': wf.get('generated'),
            'lake': wf.get('lake'),
            'anchor_count': wf.get('anchor_count'),
            'grid_spacing_km': wf.get('grid_spacing_km'),
            'bbox': wf.get('bbox'),
            'size_kb': warp_path.stat().st_size // 1024,
            'anchors_used': wf.get('anchors_used', []),
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def _run_harvester_job(job_id: str, req: HarvesterRequest) -> None:
    """Background thread: run wh2k_harvester.run_harvester() with live progress."""
    TOOL_JOBS[job_id]['status'] = 'running'
    TOOL_JOBS[job_id]['start_time'] = time.time()
    TOOL_JOBS[job_id]['pipeline'] = {src: 'pending' for src in req.sources}
    TOOL_JOBS[job_id]['progress_pct'] = 0.0
    TOOL_JOBS[job_id]['progress_msg'] = 'Startingâ€¦'

    def _on_progress(msg: str, pct: float) -> None:
        TOOL_JOBS[job_id]['progress_pct'] = round(pct, 1)
        TOOL_JOBS[job_id]['progress_msg'] = msg
        # Mark source as active/done based on message keywords
        for src in req.sources:
            if src.lower() in msg.lower():
                TOOL_JOBS[job_id]['pipeline'][src] = (
                    'done' if pct > 85 else 'running'
                )

    try:
        _root = Path(__file__).parent.parent
        sys.path.insert(0, str(_root / 'scripts'))
        from wh2k_harvester import run_harvester, HarvestConfig
        cfg = HarvestConfig(
            lake=req.lake,
            sources=req.sources,
            apply_warp=req.apply_warp,
            max_surveys=req.max_surveys,
            dry_run=req.dry_run,
            swarm_token=req.swarm_token,
        )
        result = run_harvester(cfg, on_progress=_on_progress)
        for src in req.sources:
            TOOL_JOBS[job_id]['pipeline'][src] = 'done'
        TOOL_JOBS[job_id]['status'] = 'completed'
        TOOL_JOBS[job_id]['result'] = result.to_dict()
    except Exception as e:
        TOOL_JOBS[job_id]['status'] = 'failed'
        TOOL_JOBS[job_id]['error'] = str(e)
    TOOL_JOBS[job_id]['end_time'] = time.time()


@app.post('/tools/harvester/start', tags=['tools'])
def start_harvester(req: HarvesterRequest):
    """
    Start the raw magnetometer harvester.
    Downloads pings from NGDC WFS, USGS ScienceBase, NRCan, and optionally Swarm.
    Applies LORAN-C warp if loran_warp_field.json exists.
    Poll /tools/harvester/status/{job_id} for live progress.
    """
    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        'id': job_id, 'tool': 'harvester',
        'status': 'queued', 'lake': req.lake,
        'sources': req.sources, 'created': time.time(),
    }
    t = threading.Thread(target=_run_harvester_job, args=(job_id, req), daemon=True)
    t.start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/tools/harvester/status/{job_id}', tags=['tools'])
def harvester_status(job_id: str):
    """Lightweight status + per-source pipeline state. Safe to poll every 2 s."""
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    keys = ['id', 'tool', 'status', 'lake', 'sources', 'created',
            'start_time', 'end_time', 'error', 'pipeline',
            'progress_pct', 'progress_msg']
    return {k: job[k] for k in keys if k in job}


@app.get('/tools/harvester/results/{job_id}', tags=['tools'])
def harvester_results(job_id: str):
    """Full results once status == completed."""
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('status') != 'completed':
        return {'status': job.get('status'), 'message': 'Not yet completed.'}
    return {'status': 'completed', 'result': job.get('result', {})}


@app.get('/tools/harvester/catalog', tags=['tools'])
def harvester_catalog(lake: str = 'erie'):
    """Return the most recent harvest catalog JSON for a lake."""
    catalog_path = (Path(__file__).parent.parent / 'magnetic_data' / 'raw'
                    / 'normalised' / f'{lake}_harvest_catalog.json')
    if not catalog_path.exists():
        raise HTTPException(status_code=404, detail=f'No catalog for {lake}. Run a harvest first.')
    with open(catalog_path, encoding='utf-8') as f:
        return json.load(f)


# ── Live KML / Google Earth NetworkLink ─────────────────────────────────────
# GET /wrecks/live.kml          — placemarks feed (Google Earth refreshes this)
# GET /wrecks/networklink.kmz   — download-once KMZ containing the NetworkLink
#
# Usage:
#   1.  Open networklink.kmz in Google Earth.  It stores the public URL so GE
#       periodically re-fetches live.kml as the DB updates.
#   2.  Set API_BASE_URL env-var to your Cloudflare / ngrok tunnel URL so the
#       NetworkLink href resolves from anywhere (school, etc.).

from fastapi.responses import Response as _Response
import zipfile as _zipfile
import io as _io
import html as _html
import xml.etree.ElementTree as _ET

_KML_NS = 'http://www.opengis.net/kml/2.2'

_ICON_BY_MAG = {
    'strong':   'http://maps.google.com/mapfiles/kml/paddle/red-circle.png',
    'moderate': 'http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png',
    'weak':     'http://maps.google.com/mapfiles/kml/paddle/grn-circle.png',
}
_ICON_DEFAULT = 'http://maps.google.com/mapfiles/kml/paddle/wht-circle.png'


def _build_wrecks_kml(
    base_url: str,
    limit: int = 2000,
    magnetic_potential: str = None,
    is_steel: bool = None,
    min_lat: float = None, max_lat: float = None,
    min_lon: float = None, max_lon: float = None,
) -> str:
    """Build a KML string with placemarks from the wrecks DB."""
    where, params = ["latitude IS NOT NULL", "longitude IS NOT NULL"], []
    if magnetic_potential:
        where.append("magnetic_potential=?"); params.append(magnetic_potential)
    if is_steel is True:
        where.append("is_steel_freighter=1")
    elif is_steel is False:
        where.append("(is_steel_freighter=0 OR is_steel_freighter IS NULL)")
    if min_lat is not None: where.append("latitude >= ?"); params.append(min_lat)
    if max_lat is not None: where.append("latitude <= ?"); params.append(max_lat)
    if min_lon is not None: where.append("longitude >= ?"); params.append(min_lon)
    if max_lon is not None: where.append("longitude <= ?"); params.append(max_lon)

    sql = (
        "SELECT name,latitude,longitude,depth,date,hull_material,"
        "magnetic_potential,is_steel_freighter,feature_type "
        "FROM features WHERE " + " AND ".join(where) +
        " ORDER BY CASE magnetic_potential WHEN 'strong' THEN 0 WHEN 'moderate' THEN 1 ELSE 2 END"
        " LIMIT ?"
    )
    params.append(limit)

    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<kml xmlns="{_KML_NS}">',
        '<Document>',
        f'  <name>Great Lakes Wrecks — live ({len(rows)} records)</name>',
        '  <Style id="s_strong"><IconStyle><Icon><href>'
            + _ICON_BY_MAG["strong"] + '</href></Icon><scale>1.1</scale></IconStyle></Style>',
        '  <Style id="s_moderate"><IconStyle><Icon><href>'
            + _ICON_BY_MAG["moderate"] + '</href></Icon><scale>1.0</scale></IconStyle></Style>',
        '  <Style id="s_weak"><IconStyle><Icon><href>'
            + _ICON_BY_MAG["weak"] + '</href></Icon><scale>0.9</scale></IconStyle></Style>',
        '  <Style id="s_unknown"><IconStyle><Icon><href>'
            + _ICON_DEFAULT + '</href></Icon><scale>0.8</scale></IconStyle></Style>',
    ]
    for r in rows:
        mag  = r.get('magnetic_potential') or 'unknown'
        style = f"s_{mag}" if mag in _ICON_BY_MAG else "s_unknown"
        depth = f"{r['depth']} ft" if r.get('depth') else 'depth unknown'
        steel = ' ★ Steel freighter' if r.get('is_steel_freighter') else ''
        desc = _html.escape(
            f"Date: {r.get('date') or '?'}  Depth: {depth}  "
            f"Hull: {r.get('hull_material') or '?'}  Mag: {mag}{steel}"
        )
        name = _html.escape(r.get('name') or 'Unknown wreck')
        lines += [
            '  <Placemark>',
            f'    <name>{name}</name>',
            f'    <description>{desc}</description>',
            f'    <styleUrl>#{style}</styleUrl>',
            '    <Point>',
            f'      <coordinates>{r["longitude"]},{r["latitude"]},0</coordinates>',
            '    </Point>',
            '  </Placemark>',
        ]
    lines += ['</Document>', '</kml>']
    return '\n'.join(lines)


@app.get('/wrecks/live.kml', tags=['google-earth'],
         response_class=_Response,
         summary='Live KML placemark feed — add as NetworkLink in Google Earth')
def wrecks_live_kml(
    limit: int = Query(2000, ge=1, le=10000),
    magnetic_potential: Optional[str] = Query(None),
    is_steel: Optional[bool] = Query(None),
    min_lat: Optional[float] = Query(None), max_lat: Optional[float] = Query(None),
    min_lon: Optional[float] = Query(None), max_lon: Optional[float] = Query(None),
    base_url: Optional[str] = Query(None, description="Override API base URL in NetworkLink href"),
):
    kml = _build_wrecks_kml(
        base_url=base_url or _API_BASE_URL,
        limit=limit,
        magnetic_potential=magnetic_potential,
        is_steel=is_steel,
        min_lat=min_lat, max_lat=max_lat,
        min_lon=min_lon, max_lon=max_lon,
    )
    return _Response(content=kml, media_type='application/vnd.google-earth.kml+xml')


@app.get('/wrecks/networklink.kmz', tags=['google-earth'],
         response_class=_Response,
         summary='Download this KMZ once — Google Earth will auto-refresh the live feed')
def wrecks_networklink_kmz(
    refresh_seconds: int = Query(300, ge=30, le=3600,
                                  description="How often GE re-fetches the live feed (seconds)"),
    base_url: Optional[str] = Query(None, description="Public API URL — set to your tunnel URL"),
    limit: int = Query(2000, ge=1, le=10000),
    magnetic_potential: Optional[str] = Query(None),
    is_steel: Optional[bool] = Query(None),
):
    effective_base = (base_url or _API_BASE_URL).rstrip('/')
    # Build query string for the live.kml href
    qs_parts = [f"limit={limit}"]
    if magnetic_potential: qs_parts.append(f"magnetic_potential={magnetic_potential}")
    if is_steel is not None: qs_parts.append(f"is_steel={str(is_steel).lower()}")
    live_url = f"{effective_base}/wrecks/live.kml?" + "&".join(qs_parts)

    nl_kml = '\n'.join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<kml xmlns="{_KML_NS}">',
        '<NetworkLink>',
        '  <name>Great Lakes Wrecks — Live DB</name>',
        '  <description>Auto-refreshes from the wreckhunter API every '
            f'{refresh_seconds}s. Red=strong mag, Yellow=moderate, Green=weak.</description>',
        '  <open>1</open>',
        '  <Link>',
        f'    <href>{_html.escape(live_url)}</href>',
        '    <refreshMode>onInterval</refreshMode>',
        f'    <refreshInterval>{refresh_seconds}</refreshInterval>',
        '  </Link>',
        '</NetworkLink>',
        '</kml>',
    ])

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, 'w', _zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('wrecks_live.kml', nl_kml)
    buf.seek(0)
    return _Response(
        content=buf.read(),
        media_type='application/vnd.google-earth.kmz',
        headers={'Content-Disposition': 'attachment; filename="wrecks_live_networklink.kmz"'},
    )


# ── Scan Queue API ─────────────────────────────────────────────────────────────
# These endpoints let you push/inspect scan jobs from anywhere (e.g. phone at sea).
# The background scan_worker.py on i7/Xeon picks them up automatically.
# Priority 0 = urgent user request (preempts everything).

import importlib.util as _ilu

def _queue() -> Optional[object]:
    """Lazy-load scan_queue from repo root so the API stays importable even if
    the file is absent (e.g. in older Docker images)."""
    root = Path(__file__).resolve().parents[1]
    spec = _ilu.spec_from_file_location("scan_queue", root / "scan_queue.py")
    if spec is None:
        return None
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.init_db()
        return mod
    except Exception:
        return None


class _PushJobRequest(BaseModel):
    label: str
    bbox: list          # [lat_min, lon_min, lat_max, lon_max]
    sensors: list       # e.g. ["thermal", "optical"]
    priority: int = 0   # 0=urgent, 1=directed, 2=idle
    params: dict = {}


@app.get("/scan/queue", tags=["scan"])
def scan_queue_list(limit: int = Query(default=30, le=200)):
    """List recent scan jobs (most urgent first)."""
    q = _queue()
    if q is None:
        raise HTTPException(503, "scan_queue module not available")
    return q.list_jobs(limit=limit)


@app.get("/scan/queue/depth", tags=["scan"])
def scan_queue_depth():
    """Return counts by status (QUEUED / RUNNING / DONE / FAILED)."""
    q = _queue()
    if q is None:
        raise HTTPException(503, "scan_queue module not available")
    return q.queue_depth()


@app.post("/scan/queue", tags=["scan"], status_code=201)
def scan_queue_push(job: _PushJobRequest):
    """Push a new scan job. Priority 0 = interrupt current scan."""
    if len(job.bbox) != 4:
        raise HTTPException(400, "bbox must be [lat_min, lon_min, lat_max, lon_max]")
    if not job.sensors:
        raise HTTPException(400, "sensors list cannot be empty")
    allowed = {"thermal", "optical", "triple_lock", "swot", "sar", "nir_swir"}
    bad = set(job.sensors) - allowed
    if bad:
        raise HTTPException(400, f"Unknown sensors: {bad}. Allowed: {allowed}")
    q = _queue()
    if q is None:
        raise HTTPException(503, "scan_queue module not available")
    jid = q.push(label=job.label, bbox=job.bbox, sensors=job.sensors,
                 priority=job.priority, params=job.params)
    return {"job_id": jid, "priority": job.priority, "label": job.label}


@app.delete("/scan/queue/{job_id}", tags=["scan"])
def scan_queue_cancel(job_id: str):
    """Cancel a queued job (has no effect on already-running jobs)."""
    q = _queue()
    if q is None:
        raise HTTPException(503, "scan_queue module not available")
    q.cancel(job_id)
    return {"cancelled": job_id}


@app.get("/scan/worker/state", tags=["scan"])
def scan_worker_state():
    """Return the on-disk state file written by scan_worker.py."""
    state_path = Path(__file__).resolve().parents[1] / "db" / "worker_state.json"
    if not state_path.exists():
        return {"status": "no worker state found"}
    try:
        return json.loads(state_path.read_text())
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Worker / job dashboard ──────────────────────────────────────────────────

_QUEUE_DB_PATH = os.environ.get(
    "QUEUE_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "db" / "scan_queue.db"),
)


@contextmanager
def _get_queue_db(write: bool = False):
    if not Path(_QUEUE_DB_PATH).exists():
        yield None
        return
    conn = sqlite3.connect(_QUEUE_DB_PATH, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    if not write:
        conn.execute("PRAGMA query_only = ON")
    else:
        conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


_queue_claim_lock = __import__("threading").Lock()  # serialise concurrent claims


@app.get("/workers", tags=["workers"])
def get_workers():
    """
    Return active workers (nodes with at least one RUNNING job)
    and a summary of all job statuses.
    """
    with _get_queue_db() as conn:
        if conn is None:
            return {"workers": [], "summary": {}, "queue_available": False}

        # Per-worker active jobs
        rows = conn.execute(
            "SELECT worker_id, id, label, sensors, params, started_at, bbox "
            "FROM scan_jobs WHERE status='RUNNING' ORDER BY started_at ASC"
        ).fetchall()

        workers: dict[str, dict] = {}
        for r in rows:
            wid = r["worker_id"] or "unknown"
            if wid not in workers:
                workers[wid] = {"worker_id": wid, "jobs": []}
            try:
                params = json.loads(r["params"] or "{}")
            except Exception:
                params = {}
            workers[wid]["jobs"].append({
                "id": r["id"],
                "label": r["label"],
                "sensors": json.loads(r["sensors"] or "[]"),
                "mission": params.get("mission_name", ""),
                "target_type": params.get("target_type", ""),
                "started_at": r["started_at"],
                "bbox": json.loads(r["bbox"] or "[]"),
            })

        # Status summary counts
        summary_rows = conn.execute(
            "SELECT status, COUNT(*) n FROM scan_jobs GROUP BY status"
        ).fetchall()
        summary = {r["status"]: r["n"] for r in summary_rows}

        return {
            "workers": list(workers.values()),
            "summary": summary,
            "queue_available": True,
        }


@app.get("/jobs", tags=["workers"])
def get_jobs(
    status: Optional[str] = Query(None, description="QUEUED|RUNNING|DONE|FAILED"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Return recent jobs, optionally filtered by status.
    Sorted newest first.
    """
    with _get_queue_db() as conn:
        if conn is None:
            return {"jobs": [], "total": 0, "queue_available": False}

        where = "WHERE status=?" if status else ""
        params_q: list = [status] if status else []
        params_q.append(limit)

        rows = conn.execute(
            f"SELECT id, priority, status, label, sensors, params, "
            f"created_at, started_at, finished_at, worker_id, error_msg, result_path "
            f"FROM scan_jobs {where} "
            f"ORDER BY COALESCE(started_at, created_at) DESC "
            f"LIMIT ?",
            params_q,
        ).fetchall()

        total_row = conn.execute(
            f"SELECT COUNT(*) FROM scan_jobs {where}",
            [status] if status else [],
        ).fetchone()

        jobs = []
        for r in rows:
            try:
                p = json.loads(r["params"] or "{}")
            except Exception:
                p = {}
            jobs.append({
                "id": r["id"],
                "priority": r["priority"],
                "status": r["status"],
                "label": r["label"],
                "sensors": json.loads(r["sensors"] or "[]"),
                "mission": p.get("mission_name", ""),
                "target_type": p.get("target_type", ""),
                "bbox": json.loads(r["bbox"] if "bbox" in r.keys() else "[]") if "bbox" in r.keys() else [],
                "worker_id": r["worker_id"],
                "created_at": r["created_at"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "error_msg": r["error_msg"],
                "result_path": r["result_path"],
            })

        return {"jobs": jobs, "total": total_row[0], "queue_available": True}


# ── Node worker HTTP API (claim / finish / submit) ──────────────────────────

class _ClaimBody(BaseModel):
    worker_id: str
    has_gpu: bool = False
    has_tpu: bool = False
    vram_gb: float = 0.0
    max_jobs: int = 1  # how many RUNNING jobs this worker already has (for server-side info)


class _FinishBody(BaseModel):
    success: bool
    result_path: str = ""
    error_msg: str = ""


class _SubmitJobBody(BaseModel):
    label: str
    bbox: list
    sensors: list
    params: dict = {}
    priority: int = 1
    # Legacy per-params flags kept for compatibility;
    # prefer job_type for new submissions.
    requires_gpu: bool = False
    requires_tpu: bool = False
    min_vram_gb: float = 0.0
    # Job routing
    job_type: str = "cpu"       # cpu | gpu | gpu_tpu
    pipeline_stage: str = "process"  # process | postprocess
    parent_id: str = ""


class _HeartbeatBody(BaseModel):
    worker_id: str
    has_gpu: bool = False
    has_tpu: bool = False
    vram_gb: float = 0.0
    gpu_label: str = ""


@app.post("/jobs/claim", tags=["workers"])
def claim_job_http(body: _ClaimBody):
    """
    Atomically claim the best matching QUEUED job for this worker.
    Capability filtering: jobs with requires_gpu/requires_tpu in their params
    are only returned to workers that advertise those capabilities.
    Returns the full job dict, or {"job": null} if nothing suitable.
    """
    with _queue_claim_lock:
        with _get_queue_db(write=True) as conn:
            if conn is None:
                raise HTTPException(503, "Queue DB not available")

            # Fetch all QUEUED jobs in priority order, filter by capabilities.
            # job_type column (added by migration) drives routing:
            #   cpu      → any worker
            #   gpu      → only workers with has_gpu
            #   gpu_tpu  → only workers with both has_gpu and has_tpu
            rows = conn.execute(
                "SELECT * FROM scan_jobs WHERE status='QUEUED' "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()

            chosen = None
            for r in rows:
                cols = r.keys()
                # job_type column may not exist on older DBs — fall back to params flags
                job_type = r["job_type"] if "job_type" in cols else None
                if job_type is None:
                    try:
                        p = json.loads(r["params"] or "{}")
                    except Exception:
                        p = {}
                    if p.get("requires_tpu"):
                        job_type = "gpu_tpu"
                    elif p.get("requires_gpu"):
                        job_type = "gpu"
                    else:
                        job_type = "cpu"

                if job_type == "gpu_tpu" and not (body.has_gpu and body.has_tpu):
                    continue
                if job_type == "gpu" and not body.has_gpu:
                    continue
                # cpu jobs — allow any worker
                chosen = dict(r)
                break

            if chosen is None:
                return {"job": None}

            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            affected = conn.execute(
                "UPDATE scan_jobs SET status='RUNNING', started_at=?, worker_id=? "
                "WHERE id=? AND status='QUEUED'",
                (now, body.worker_id, chosen["id"]),
            ).rowcount
            conn.commit()

            if affected == 0:
                return {"job": None}  # race condition — someone else got it

            chosen["status"] = "RUNNING"
            chosen["started_at"] = now
            chosen["worker_id"] = body.worker_id
            # Decode JSON text fields
            for field in ("bbox", "sensors", "params"):
                val = chosen.get(field)
                if isinstance(val, str):
                    try:
                        chosen[field] = json.loads(val)
                    except Exception:
                        pass
            return {"job": chosen}


@app.post("/jobs/{job_id}/finish", tags=["workers"])
def finish_job_http(job_id: str, body: _FinishBody):
    """Mark a job DONE or FAILED."""
    from datetime import datetime, timezone
    status = "DONE" if body.success else "FAILED"
    now = datetime.now(timezone.utc).isoformat()
    with _get_queue_db(write=True) as conn:
        if conn is None:
            raise HTTPException(503, "Queue DB not available")
        affected = conn.execute(
            "UPDATE scan_jobs SET status=?, finished_at=?, result_path=?, error_msg=? "
            "WHERE id=?",
            (
                status, now,
                body.result_path if body.success else None,
                None if body.success else body.error_msg[:4000],
                job_id,
            ),
        ).rowcount
        conn.commit()
    if affected == 0:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"ok": True, "status": status}


@app.post("/jobs", tags=["workers"])
def submit_job(body: _SubmitJobBody):
    """Submit a new job to the queue."""
    import uuid
    from datetime import datetime, timezone
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    params = dict(body.params)
    # Derive job_type from legacy flags when caller didn't set it explicitly
    job_type = body.job_type
    if job_type == "cpu" and (body.requires_tpu or params.get("requires_tpu")):
        job_type = "gpu_tpu"
    elif job_type == "cpu" and (body.requires_gpu or params.get("requires_gpu")):
        job_type = "gpu"

    with _get_queue_db(write=True) as conn:
        if conn is None:
            raise HTTPException(503, "Queue DB not available")
        conn.execute(
            "INSERT INTO scan_jobs "
            "(id, priority, status, label, bbox, sensors, params, created_at, "
            " job_type, pipeline_stage, parent_id) "
            "VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, body.priority, body.label,
             json.dumps(body.bbox), json.dumps(body.sensors),
             json.dumps(params), now,
             job_type,
             body.pipeline_stage,
             body.parent_id or None),
        )
        conn.commit()
    return {"ok": True, "id": job_id}


# ── Worker heartbeat / online roster ─────────────────────────────────────────

@app.post("/workers/heartbeat", tags=["workers"])
def worker_heartbeat(body: _HeartbeatBody):
    """Workers call this every 30 s to advertise themselves as online."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _get_queue_db(write=True) as conn:
        if conn is None:
            raise HTTPException(503, "Queue DB not available")
        conn.execute(
            """
            INSERT INTO worker_heartbeats (worker_id, has_gpu, has_tpu, vram_gb, gpu_label, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                has_gpu   = excluded.has_gpu,
                has_tpu   = excluded.has_tpu,
                vram_gb   = excluded.vram_gb,
                gpu_label = excluded.gpu_label,
                last_seen = excluded.last_seen
            """,
            (body.worker_id,
             int(body.has_gpu), int(body.has_tpu),
             body.vram_gb, body.gpu_label, now),
        )
        conn.commit()
    return {"ok": True}


@app.get("/workers/online", tags=["workers"])
def workers_online():
    """Return workers that have sent a heartbeat within the last 90 s."""
    with _get_queue_db() as conn:
        if conn is None:
            return {"workers": [], "queue_available": False}
        # SQLite datetime comparison (ISO strings sort lexicographically)
        rows = conn.execute(
            """
            SELECT worker_id, has_gpu, has_tpu, vram_gb, gpu_label, last_seen
            FROM worker_heartbeats
            WHERE last_seen >= datetime('now', '-90 seconds')
            ORDER BY last_seen DESC
            """
        ).fetchall()
        workers = [
            {
                "worker_id": r["worker_id"],
                "has_gpu":   bool(r["has_gpu"]),
                "has_tpu":   bool(r["has_tpu"]),
                "vram_gb":   r["vram_gb"],
                "gpu_label": r["gpu_label"],
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]
    return {"workers": workers, "count": len(workers), "queue_available": True}

# ── Hardware Telemetry (Bypass Tauri) ────────────────────────────────────────

TELEMETRY_STATE = {}
_telemetry_lock = threading.Lock()

class TelemetryPayload(BaseModel):
    worker_id: str
    platform: str
    cpu_percent: float
    ram_percent: float
    gpus: list = []  # e.g. [{"name": "TITAN X", "load": 45.0, "memory_used": 4096, "memory_total": 12288}]
    timestamp: float

@app.post("/telemetry", tags=["workers"])
def submit_telemetry(payload: TelemetryPayload):
    """Store hardware telemetry from a worker node."""
    with _telemetry_lock:
        TELEMETRY_STATE[payload.worker_id] = payload.dict()
    return {"ok": True}

@app.get("/telemetry", tags=["workers"])
def get_telemetry():
    """Retrieve all recent hardware telemetry for the web dashboard."""
    with _telemetry_lock:
        return {"telemetry": list(TELEMETRY_STATE.values())}

# ── Agent / AI Director / Mission REST endpoints (web-mode equivalents) ───────
# These mirror the Tauri Rust commands so the browser build can call them via
# fetch() instead of Tauri's invoke() IPC.  All subprocesses run from _REPO_ROOT.

def _run_script(
    script: str,
    args: list,
    cwd: Optional[str] = None,
    timeout: int = 300,
) -> dict:
    """Run a Python script and return a TaskOutput-compatible dict."""
    import time as _time
    start = _time.time()
    task_id = str(uuid.uuid4())
    work_dir = Path(cwd) if cwd else _REPO_ROOT
    cmd = [sys.executable, script] + [str(a) for a in (args or [])]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "id": task_id,
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(_time.time() - start, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "id": task_id, "status": "error",
            "stdout": "", "stderr": f"Timeout after {timeout}s",
            "duration_s": float(timeout),
        }
    except Exception as exc:
        return {
            "id": task_id, "status": "error",
            "stdout": "", "stderr": str(exc),
            "duration_s": round(_time.time() - start, 2),
        }


class _AgentRequestBody(BaseModel):
    request: str
    provider: Optional[str] = "qwen"
    model_override: Optional[str] = None


class _MissionRunBody(BaseModel):
    task_id: Optional[str] = None
    mission_json: str


class _NasaSearchBody(BaseModel):
    bbox: list          # [lat_min, lon_min, lat_max, lon_max]
    start_date: str
    end_date: str
    sensor: str = "hls"


class _RunTaskBody(BaseModel):
    task_id: Optional[str] = None
    script: str
    args: Optional[list] = []
    cwd: Optional[str] = None


@app.get("/tools/work-dir", tags=["agent"])
def get_work_dir():
    """Return the repo root used by agent scripts (web-mode equivalent of Tauri get_work_dir)."""
    return {"work_dir": str(_REPO_ROOT)}


@app.post("/tools/agent/request", tags=["agent"])
def agent_request(body: _AgentRequestBody):
    """Web-mode equivalent of Tauri ai_direct_request — runs ai_director.py."""
    args = ["--request", body.request, "--execute"]
    # Native providers need --provider flag; qwen/koboldcpp/github_sdk use env vars
    if body.provider and body.provider not in ("qwen", "koboldcpp", "github_sdk"):
        args += ["--provider", body.provider]
    return _run_script("ai_director.py", args)


@app.post("/tools/agent/probe", tags=["agent"])
def agent_probe():
    """Web-mode equivalent of Tauri run_background_probe — runs background_probe.py --once."""
    return _run_script("background_probe.py", ["--once"])


@app.get("/tools/nodes/check", tags=["agent"])
def nodes_check():
    """Web-mode equivalent of Tauri check_nodes — runs cesarops_orchestrator.py --status."""
    return _run_script("cesarops_orchestrator.py", ["--status"])


@app.get("/tools/agent/provider-status", tags=["agent"])
def agent_provider_status_web(
    provider: str = Query("qwen"),
    model_override: Optional[str] = Query(None),
):
    """Web-mode equivalent of Tauri agent_provider_status — reads .env and checks provider."""
    import socket
    # Load .env from repo root
    env_path = _REPO_ROOT / ".env"
    dotenv: dict = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                dotenv[k.strip()] = v.strip()

    def _env(key: str) -> Optional[str]:
        return os.environ.get(key) or dotenv.get(key)

    p = provider
    m = (model_override or "").strip() or None
    if p == "qwen":
        base = _env("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = m or _env("QWEN_MODEL") or "qwen-plus"
        has_key = bool(_env("QWEN_API_KEY"))
    elif p == "koboldcpp":
        ts_ip = _env("KOBOLDCPP_TAILSCALE_IP") or _env("I7_TAILSCALE") or "100.85.138.4"
        base = _env("KOBOLDCPP_BASE_URL") or f"http://{ts_ip}:5001/v1"
        model = m or _env("KOBOLDCPP_MODEL") or "DeepSeek-R1-Distill-Qwen-7B"
        has_key = bool(_env("KOBOLDCPP_API_KEY"))
    elif p == "github_sdk":
        base = _env("GITHUB_MODELS_BASE_URL") or "https://models.inference.ai.azure.com"
        model = m or _env("GITHUB_MODEL") or "gpt-4.1"
        has_key = bool(_env("GITHUB_TOKEN") or _env("GITHUB_PAT"))
    elif p == "gemini":
        base = _env("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
        model = m or _env("GEMINI_MODEL") or "gemini-2.5-flash"
        has_key = bool(_env("GEMINI_API_KEY"))
    elif p == "anthropic":
        base = _env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
        model = m or _env("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
        has_key = bool(_env("ANTHROPIC_API_KEY"))
    elif p == "groq":
        base = _env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        model = m or _env("GROQ_MODEL") or "llama-3.3-70b-versatile"
        has_key = bool(_env("GROQ_API_KEY"))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")

    def tcp_ok(url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((parsed.hostname, port), timeout=1.2):
                return True
        except Exception:
            return False

    reachable = tcp_ok(base)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        f"Provider: {p}\n"
        f"Base URL: {base}\n"
        f"Model: {model}\n"
        f"API key: {'present' if has_key else 'missing (local default)'}\n"
        f"Endpoint: {'reachable \u2713' if reachable else 'not reachable \u2717'}"
    )


@app.post("/tools/mission/run", tags=["agent"])
def mission_run(body: _MissionRunBody):
    """Web-mode equivalent of Tauri run_mission — runs cesarops_mission.py."""
    return _run_script("cesarops_mission.py", ["--mission-json", body.mission_json], timeout=600)


@app.post("/tools/nasa/search", tags=["agent"])
def nasa_search(body: _NasaSearchBody):
    """Web-mode equivalent of Tauri search_nasa_granules — runs cmr_search.py."""
    if len(body.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must have 4 elements: [lat_min, lon_min, lat_max, lon_max]")
    bbox_str = ",".join(str(v) for v in body.bbox)
    args = [
        "--bbox", bbox_str,
        "--start", body.start_date,
        "--end", body.end_date,
        "--sensor", body.sensor,
        "--max-results", "50",
    ]
    result = _run_script("cmr_search.py", args)
    if result["status"] == "success":
        try:
            return json.loads(result["stdout"])
        except Exception:
            raise HTTPException(status_code=502, detail=f"CMR parse error: {result['stdout'][:200]}")
    raise HTTPException(status_code=500, detail=result["stderr"][:400])


@app.post("/tools/run-task", tags=["agent"])
def run_task_web(body: _RunTaskBody):
    """Web-mode equivalent of Tauri run_task — run an arbitrary script from repo root."""
    return _run_script(body.script, body.args or [], cwd=body.cwd)

# ── Hardware info endpoint ───────────────────────────────────────────────────
@app.get("/hardware", tags=["system"])
def get_system_hardware():
    """Get system hardware status including CPU, GPU temperatures, and NVIDIA SMI info."""
    hardware_info = {
        "cpu": {},
        "gpu": {},
        "nvidia_smi": {}
    }

    # CPU info using psutil
    if HAS_PSUTIL:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            hardware_info["cpu"] = {
                "percent": cpu_percent,
                "count": cpu_count,
                "freq_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
                "memory_total_gb": round(memory.total / (1024**3), 1),
                "memory_used_gb": round(memory.used / (1024**3), 1),
                "memory_percent": memory.percent,
            }
        except Exception as e:
            hardware_info["cpu"]["error"] = str(e)
    else:
        hardware_info["cpu"]["error"] = "psutil not available"

    # GPU info using nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "temperature_c": float(parts[0]),
                        "utilization_percent": float(parts[1]),
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "power_draw_w": float(parts[4]),
                        "power_limit_w": float(parts[5])
                    })
            hardware_info["gpu"]["nvidia"] = gpus
        else:
            hardware_info["gpu"]["error"] = result.stderr.strip()
    except FileNotFoundError:
        hardware_info["gpu"]["error"] = "nvidia-smi not found"
    except Exception as e:
        hardware_info["gpu"]["error"] = str(e)

    # Full NVIDIA SMI output
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            hardware_info["nvidia_smi"]["output"] = result.stdout
        else:
            hardware_info["nvidia_smi"]["error"] = result.stderr.strip()
    except Exception as e:
        hardware_info["nvidia_smi"]["error"] = str(e)

    return hardware_info




# ── Kobold Agent endpoints ────────────────────────────────────────────────────
# These endpoints let the frontend launch KoboldCPP and download models.
# The frontend calls these on the wrecks API (port 8099); the backend
# proxies launch commands to the local KoboldCPP process.

_KOBOLD_PROCESSES: dict = {}  # track launched kobold subprocesses by key

_KOBOLD_BASE_URL = os.environ.get(
    "KOBOLD_BASE_URL",
    _dotenv.get("KOBOLD_BASE_URL", "http://localhost:5001/v1")
    if "_dotenv" in dir()
    else "http://localhost:5001/v1",
)

# Resolve _dotenv safely (it's defined at module top in ai_director.py but not here)
def _load_dotenv_safe() -> dict:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    result = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result

_kobold_env = _load_dotenv_safe()
_KOBOLD_BASE_URL = os.environ.get(
    "KOBOLD_BASE_URL",
    _kobold_env.get("KOBOLD_BASE_URL", "http://localhost:5001/v1"),
)
_KOBOLD_HOST = _KOBOLD_BASE_URL.split("/v1")[0]  # e.g. http://localhost:5001


# Allowlists for security
_ALLOWED_GPU_MODES = {"p100_single", "p100_dual", "mixed_p100_1070"}
_ALLOWED_MODELS = {
    "Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-14B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-Coder-V2-Lite-Instruct",
    "CodeLlama-7B-Instruct",
    "CodeLlama-13B-Instruct",
    "StarCoder2-7B",
    "StarCoder2-3B",
    "phi-3-mini",
    "tinyllama",
    "qwen1.5-0.5b",
}


class KoboldLaunchRequest(BaseModel):
    model: str
    gpu_mode: str = "p100_single"
    model_path: str = "/mnt/garmour/models"
    port: int = 5001
    reasoning_model: str = ""


class KoboldDownloadRequest(BaseModel):
    model: str          # HuggingFace repo id, e.g. "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    save_path: str = "/mnt/garmour/models"


@app.get("/kobold/status", tags=["kobold"])
def kobold_status():
    """Check whether KoboldCPP is reachable and return its model info."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            f"{_KOBOLD_HOST}/api/v1/model",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        return {"online": True, "kobold_url": _KOBOLD_HOST, "model": data}
    except Exception as e:
        return {"online": False, "kobold_url": _KOBOLD_HOST, "error": str(e)}


@app.post("/kobold/launch", tags=["kobold"])
def kobold_launch(req: KoboldLaunchRequest):
    """
    Launch KoboldCPP with the requested model and GPU configuration.
    Searches common Linux install locations for the binary.
    """
    # Validate inputs
    if req.gpu_mode not in _ALLOWED_GPU_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid gpu_mode. Allowed: {sorted(_ALLOWED_GPU_MODES)}")

    allowed_prefixes = ("/mnt/garmour", "/mnt/data", "/home", "/models", "/opt", "/root", "C:\\", "D:\\")
    if not any(req.model_path.startswith(p) for p in allowed_prefixes):
        raise HTTPException(status_code=400, detail="model_path must be within an allowed directory")

    if not (1024 <= req.port <= 65535):
        raise HTTPException(status_code=400, detail="port must be between 1024 and 65535")

    root = Path(__file__).resolve().parents[1]

    # ── Check if already running ──────────────────────────────────────────────
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(f"http://localhost:{req.port}/api/v1/model", timeout=2)
        return {
            "status": "already_running",
            "message": f"KoboldCPP is already running on port {req.port}",
            "kobold_url": f"http://localhost:{req.port}/v1",
        }
    except Exception:
        pass

    # ── Find KoboldCPP binary (Linux search order) ────────────────────────────
    home = Path.home()
    kobold_candidates = [
        home / "ai_coding" / "koboldcpp",
        home / "koboldcpp" / "koboldcpp",
        home / "koboldcpp",
        Path("/opt/koboldcpp/koboldcpp-linux-x64"),
        Path("/opt/koboldcpp/koboldcpp"),
        Path("/usr/local/bin/koboldcpp"),
        Path("/usr/bin/koboldcpp"),
        root / "koboldcpp",
        root / "koboldcpp-linux-x64",
    ]
    kobold_bin = next((p for p in kobold_candidates if p.exists() and p.is_file()), None)

    # ── Find model file ───────────────────────────────────────────────────────
    model_file = None
    model_dir = Path(req.model_path)
    if model_dir.exists():
        model_lower = req.model.lower().replace("-", "").replace("_", "")
        # Try exact name match first
        for ext in ("*.gguf", "*.onnx"):
            for f in model_dir.glob(ext):
                fname_lower = f.name.lower().replace("-", "").replace("_", "")
                if model_lower[:12] in fname_lower:
                    model_file = str(f)
                    break
            if model_file:
                break
        # Fall back to first .gguf found
        if not model_file:
            gguf_files = sorted(model_dir.glob("*.gguf"))
            if gguf_files:
                model_file = str(gguf_files[0])

    # ── GPU layer count ───────────────────────────────────────────────────────
    gpu_layers = {"p100_single": 28, "p100_dual": 56, "mixed_p100_1070": 32}.get(req.gpu_mode, 28)

    try:
        if kobold_bin:
            # Direct binary launch
            cmd = [str(kobold_bin), "--port", str(req.port),
                   "--gpulayers", str(gpu_layers),
                   "--contextsize", "4096", "--threads", "8"]
            if model_file:
                cmd += ["--model", model_file]

            proc = subprocess.Popen(
                cmd, cwd=str(root),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _KOBOLD_PROCESSES[req.port] = proc
            return {
                "status": "launching",
                "pid": proc.pid,
                "binary": str(kobold_bin),
                "model": req.model,
                "model_file": model_file,
                "gpu_mode": req.gpu_mode,
                "gpu_layers": gpu_layers,
                "kobold_url": f"http://localhost:{req.port}/v1",
                "message": f"KoboldCPP launching on port {req.port} with {gpu_layers} GPU layers. Poll /kobold/status to confirm.",
            }

        # ── Launcher script fallback ──────────────────────────────────────────
        launcher = root / "scripts" / "launch_koboldcpp.py"
        if launcher.exists():
            cmd = [sys.executable, str(launcher), "--port", str(req.port)]
            if req.gpu_mode == "p100_single":
                cmd += ["--gpu-ids", "0"]
            elif req.gpu_mode in ("p100_dual", "mixed_p100_1070"):
                cmd += ["--gpu-ids", "0,1"]
            if model_file:
                cmd += ["--model", model_file]

            proc = subprocess.Popen(cmd, cwd=str(root),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _KOBOLD_PROCESSES[req.port] = proc
            return {
                "status": "launching",
                "pid": proc.pid,
                "model_file": model_file,
                "kobold_url": f"http://localhost:{req.port}/v1",
                "message": f"KoboldCPP launching via launcher script on port {req.port}.",
            }

        # ── Nothing found — give install instructions ─────────────────────────
        return {
            "status": "not_installed",
            "kobold_url": f"http://localhost:{req.port}/v1",
            "message": (
                "KoboldCPP binary not found. Install it:\n"
                "  bash install_kobold.sh\n"
                "or download from https://github.com/LostRuins/koboldcpp/releases\n"
                "and place at ~/koboldcpp or /opt/koboldcpp/koboldcpp-linux-x64"
            ),
            "searched": [str(p) for p in kobold_candidates],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch KoboldCPP: {e}")


@app.post("/kobold/download-model", tags=["kobold"])
def kobold_download_model(req: KoboldDownloadRequest):
    """
    Download a GGUF model from HuggingFace to the specified path.
    Uses huggingface_hub if available, otherwise falls back to wget/curl.
    """
    # Validate save_path
    allowed_prefixes = ("/mnt/garmour", "/mnt/data", "/home", "/models", "C:\\", "D:\\")
    if not any(req.save_path.startswith(p) for p in allowed_prefixes):
        raise HTTPException(status_code=400, detail="save_path must be within an allowed directory")

    # Validate model name — must look like a HuggingFace repo id (owner/repo)
    import re as _re
    if not _re.match(r'^[\w\-\.]+/[\w\-\.]+$', req.model):
        raise HTTPException(
            status_code=400,
            detail="model must be a valid HuggingFace repo id (e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF')"
        )

    save_dir = Path(req.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())
    TOOL_JOBS[job_id] = {
        "id": job_id,
        "tool": "kobold_download",
        "status": "queued",
        "model": req.model,
        "save_path": req.save_path,
        "created": time.time(),
    }

    def _do_download(job_id: str, model: str, save_path: str):
        TOOL_JOBS[job_id]["status"] = "running"
        TOOL_JOBS[job_id]["start_time"] = time.time()
        try:
            try:
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(
                    repo_id=model,
                    local_dir=save_path,
                    ignore_patterns=["*.bin", "*.pt", "*.safetensors"],  # GGUF only
                )
                TOOL_JOBS[job_id]["status"] = "completed"
                TOOL_JOBS[job_id]["result"] = {"local_dir": local_dir}
            except ImportError:
                # Fall back to subprocess wget
                url = f"https://huggingface.co/{model}/resolve/main"
                cmd = ["wget", "-P", save_path, "-r", "-l1", "--no-parent",
                       "-A", "*.gguf", url]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if proc.returncode == 0:
                    TOOL_JOBS[job_id]["status"] = "completed"
                    TOOL_JOBS[job_id]["result"] = {"save_path": save_path}
                else:
                    TOOL_JOBS[job_id]["status"] = "failed"
                    TOOL_JOBS[job_id]["error"] = proc.stderr[-500:]
        except Exception as e:
            TOOL_JOBS[job_id]["status"] = "failed"
            TOOL_JOBS[job_id]["error"] = str(e)
        TOOL_JOBS[job_id]["end_time"] = time.time()

    t = threading.Thread(
        target=_do_download,
        args=(job_id, req.model, req.save_path),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "model": req.model, "save_path": req.save_path}


@app.get("/kobold/download-status/{job_id}", tags=["kobold"])
def kobold_download_status(job_id: str):
    """Poll download job status."""
    job = TOOL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: job[k] for k in job}


@app.post("/kobold/upload-model", tags=["kobold"])
async def kobold_upload_model(
    file: UploadFile = File(...),
    save_path: str = Form("/mnt/garmour/models"),
):
    """
    Accept a .gguf or .onnx file upload from the browser and save it to
    the server's model directory (save_path).
    """
    import shutil as _shutil

    # Validate extension
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("gguf", "onnx"):
        raise HTTPException(status_code=400, detail="Only .gguf and .onnx files are accepted")

    # Validate save_path stays within allowed mounts
    allowed_prefixes = ("/mnt/garmour", "/mnt/data", "/home", "/models", "C:\\", "D:\\")
    if not any(save_path.startswith(p) for p in allowed_prefixes):
        raise HTTPException(status_code=400, detail="save_path must be within an allowed directory")

    dest_dir = Path(save_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    # Stream to disk
    try:
        with open(dest_file, "wb") as out:
            _shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    size_mb = dest_file.stat().st_size / (1024 * 1024)
    return {
        "saved_path": str(dest_file),
        "filename": filename,
        "size_mb": round(size_mb, 2),
        "message": f"Saved {filename} ({size_mb:.1f} MB) to {save_path}",
    }


# ── Unified Search / Orchestrator endpoint ───────────────────────────────────
# This is the primary entry point for the "type what you're looking for" UI.
# It accepts natural language, parses intent (with LLM or keyword fallback),
# queues a scan job, and returns a job_id to poll.
#
# Workers are pure knob-turners — they don't write new code, they just receive
# a mission spec with tuned parameters and execute it.

import re as _re

# ── Lake bounding boxes ───────────────────────────────────────────────────────
_LAKE_BBOXES = {
    "superior":  [46.5, -92.0, 48.5, -84.5],
    "michigan":  [41.5, -88.0, 46.0, -84.5],
    "huron":     [43.0, -84.5, 46.5, -79.5],
    "erie":      [41.3, -83.5, 42.9, -78.8],
    "ontario":   [43.2, -79.9, 44.3, -76.0],
    "straits":   [45.6, -85.0, 46.1, -84.0],
    "mackinac":  [45.6, -85.0, 46.1, -84.0],
}

# ── Target type → sensor/pass mapping ────────────────────────────────────────
_TARGET_SENSORS = {
    "wreck":      ["thermal", "optical", "sar"],
    "ship":       ["thermal", "optical", "sar"],
    "vessel":     ["thermal", "optical", "sar"],
    "freighter":  ["thermal", "optical", "sar"],
    "aircraft":   ["optical", "sar", "ndvi"],
    "plane":      ["optical", "sar", "ndvi"],
    "car":        ["optical", "sar"],
    "vehicle":    ["optical", "sar"],
    "person":     ["thermal", "optical"],
    "missing":    ["thermal", "optical", "sar"],
    "hydrocarbon":["hls", "swir"],
    "oil":        ["hls", "swir"],
    "leak":       ["hls", "swir"],
    "magnetic":   ["magnetics"],
    "anomaly":    ["thermal", "optical", "sar", "magnetics"],
}

_TARGET_PASSES = {
    "wreck":      [1, 2, 3, 4, 7],
    "ship":       [1, 2, 3, 4, 7],
    "vessel":     [1, 2, 3, 4, 7],
    "freighter":  [1, 2, 3, 4, 7],
    "aircraft":   [1, 3, 4],
    "plane":      [1, 3, 4],
    "car":        [1, 3],
    "vehicle":    [1, 3],
    "person":     [1, 3],
    "missing":    [1, 2, 3, 4, 7],
    "hydrocarbon":[2, 5, 6],
    "oil":        [2, 5, 6],
    "leak":       [2, 5, 6],
    "magnetic":   [4],
    "anomaly":    [1, 2, 3, 4, 5, 6, 7],
}

# ── Sensitivity keywords ──────────────────────────────────────────────────────
_SENSITIVITY_KEYWORDS = {
    "aggressive": 1.2,
    "sensitive":  1.3,
    "thorough":   1.4,
    "deep":       1.5,
    "standard":   2.0,
    "normal":     2.0,
    "conservative": 2.5,
    "strict":     2.8,
}


def _parse_search_intent(query: str) -> dict:
    """
    Parse a natural language search query into a structured mission spec.
    Uses keyword matching — no LLM required, but LLM can override if available.

    Returns a dict compatible with mission_control.py's mission spec schema.
    """
    q = query.lower()

    # ── Detect lake / area ────────────────────────────────────────────────────
    bbox = None
    area_label = "Great Lakes"
    for lake, bb in _LAKE_BBOXES.items():
        if lake in q:
            bbox = bb
            area_label = f"Lake {lake.title()}"
            break

    # ── Detect target type ────────────────────────────────────────────────────
    target_type = "wreck"  # default
    sensors = _TARGET_SENSORS["wreck"]
    passes = _TARGET_PASSES["wreck"]
    for kw in _TARGET_SENSORS:
        if kw in q:
            target_type = kw
            sensors = _TARGET_SENSORS[kw]
            passes = _TARGET_PASSES[kw]
            break

    # ── Detect named target (quoted or capitalized) ───────────────────────────
    target_name = None
    # Look for quoted name
    m = _re.search(r'"([^"]+)"', query)
    if m:
        target_name = m.group(1)
    else:
        # Look for known wreck names in the DB
        try:
            with get_db() as conn:
                # Extract capitalized words as potential names
                words = _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
                for phrase in words:
                    if len(phrase) > 3:
                        row = conn.execute(
                            "SELECT name FROM features WHERE UPPER(name) LIKE UPPER(?) LIMIT 1",
                            (f"%{phrase}%",)
                        ).fetchone()
                        if row:
                            target_name = row["name"]
                            break
        except Exception:
            pass

    # ── Detect sensitivity ────────────────────────────────────────────────────
    sensitivity = 2.0
    for kw, val in _SENSITIVITY_KEYWORDS.items():
        if kw in q:
            sensitivity = val
            break

    # ── Detect date range ─────────────────────────────────────────────────────
    # Look for year mentions
    years = _re.findall(r'\b(19[0-9]{2}|20[0-2][0-9])\b', query)
    if len(years) >= 2:
        date_range = [f"{min(years)}-01-01", f"{max(years)}-12-31"]
    elif len(years) == 1:
        date_range = [f"{years[0]}-01-01", f"{years[0]}-12-31"]
    else:
        # Default: last 2 years
        from datetime import datetime as _dt
        now = _dt.utcnow()
        date_range = [f"{now.year - 2}-01-01", f"{now.year}-12-31"]

    # ── Weather filter ────────────────────────────────────────────────────────
    weather_filter = "calm_and_post_storm"
    if "storm" in q:
        weather_filter = "post_storm"
    elif "calm" in q:
        weather_filter = "calm"

    # ── Build mission spec ────────────────────────────────────────────────────
    mission_id = f"SEARCH_{int(time.time())}"
    spec = {
        "mission_id": mission_id,
        "query": query,
        "target_name": target_name or target_type.title(),
        "target_type": target_type,
        "area_label": area_label,
        "bbox": bbox or [41.3, -92.0, 48.5, -76.0],  # all Great Lakes fallback
        "date_range": date_range,
        "sensors": sensors,
        "weather_filter": weather_filter,
        "passes": passes,
        "knobs": {
            "hc_threshold": sensitivity,
            "silt_erasure_threshold": sensitivity + 0.5,
            "displacement_min_delta": sensitivity,
            "mussel_clearspot_top_n": 30,
            "max_download_results": 200,
            "post_storm_days": 3,
            "calm_max_wind_kmh": 15.0,
            "storm_min_wind_kmh": 28.0,
        },
        "output": {
            "db_path": None,
            "scan_group": target_type,
        },
    }
    return spec


def _try_llm_parse(query: str, spec: dict) -> dict:
    """
    Optionally refine the keyword-parsed spec using the local LLM.
    Falls back to the keyword spec if LLM is unavailable or slow.
    """
    kobold_url = _kobold_env.get("KOBOLD_BASE_URL", "http://localhost:5001/v1")
    try:
        import urllib.request as _ur
        prompt = (
            "You are a search-and-rescue mission planner. "
            "Given this user query, output ONLY a JSON object with these fields: "
            "target_name (string), target_type (string), area_label (string), "
            "bbox ([lat_min,lon_min,lat_max,lon_max]), date_range ([start,end]), "
            "sensors (list), passes (list of ints 1-7), weather_filter (string), "
            "sensitivity (float 1.0-3.0).\n\n"
            f"Query: {query}\n\n"
            "JSON:"
        )
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.1,
            "stop": ["\n\n", "```"],
        }).encode()
        req = _ur.Request(
            f"{kobold_url.rstrip('/v1').rstrip('/')}/api/v1/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        text = data.get("results", [{}])[0].get("text", "")
        # Extract JSON from response
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            llm_spec = json.loads(m.group(0))
            # Merge LLM refinements into keyword spec
            for key in ("target_name", "target_type", "area_label", "bbox",
                        "date_range", "sensors", "passes", "weather_filter"):
                if key in llm_spec and llm_spec[key]:
                    spec[key] = llm_spec[key]
            if "sensitivity" in llm_spec:
                s = float(llm_spec["sensitivity"])
                spec["knobs"]["hc_threshold"] = s
                spec["knobs"]["silt_erasure_threshold"] = s + 0.5
                spec["knobs"]["displacement_min_delta"] = s
    except Exception:
        pass  # LLM unavailable — keyword spec is fine
    return spec


class SearchRequest(BaseModel):
    query: str
    use_llm: bool = True   # attempt LLM refinement of the parsed spec


class SearchJobStatus(BaseModel):
    job_id: str
    status: str
    query: str
    spec: dict
    created: float
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    result_summary: Optional[dict] = None


# In-memory search job store (survives process lifetime)
_SEARCH_JOBS: dict = {}


def _run_search_job(job_id: str, spec: dict):
    """
    Background thread: run mission_control.py with the parsed spec.
    Workers just turn knobs — they receive the spec and execute.
    """
    _SEARCH_JOBS[job_id]["status"] = "running"
    _SEARCH_JOBS[job_id]["started"] = time.time()

    root = Path(__file__).resolve().parents[1]
    mission_control = root / "mission_control.py"

    try:
        # Write spec to a temp file
        spec_path = root / "outputs" / f"search_{job_id}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        if mission_control.exists():
            cmd = [
                sys.executable, str(mission_control),
                "--spec", str(spec_path),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            stdout = proc.stdout[-4000:] if proc.stdout else ""
            stderr = proc.stderr[-2000:] if proc.stderr else ""

            if proc.returncode == 0:
                _SEARCH_JOBS[job_id]["status"] = "completed"
                _SEARCH_JOBS[job_id]["result_summary"] = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "spec": spec,
                    "output_dir": str(root / "outputs" / spec["mission_id"]),
                }
            else:
                _SEARCH_JOBS[job_id]["status"] = "failed"
                _SEARCH_JOBS[job_id]["error"] = stderr or f"exit code {proc.returncode}"
                _SEARCH_JOBS[job_id]["result_summary"] = {"stdout": stdout, "stderr": stderr}
        else:
            # mission_control.py not found — push to scan queue as fallback
            import scan_queue as _sq
            _sq.init_db()
            qjob_id = _sq.push(
                label=spec.get("target_name", spec["query"][:60]),
                bbox=spec["bbox"],
                sensors=spec.get("sensors", ["thermal", "optical", "sar"]),
                priority=_sq.PRIORITY_USER,
                params={
                    "query": spec["query"],
                    "passes": spec.get("passes", [1, 2, 3]),
                    "knobs": spec.get("knobs", {}),
                    "weather_filter": spec.get("weather_filter", "calm_and_post_storm"),
                    "date_range": spec.get("date_range", []),
                },
            )
            _SEARCH_JOBS[job_id]["status"] = "queued_to_worker"
            _SEARCH_JOBS[job_id]["result_summary"] = {
                "queue_job_id": qjob_id,
                "message": "Queued to scan worker daemon (mission_control.py not found)",
                "spec": spec,
            }

    except subprocess.TimeoutExpired:
        _SEARCH_JOBS[job_id]["status"] = "failed"
        _SEARCH_JOBS[job_id]["error"] = "Mission timed out after 1 hour"
    except Exception as e:
        _SEARCH_JOBS[job_id]["status"] = "failed"
        _SEARCH_JOBS[job_id]["error"] = str(e)

    _SEARCH_JOBS[job_id]["finished"] = time.time()


@app.post("/search", tags=["search"], status_code=202)
def submit_search(req: SearchRequest):
    """
    Primary entry point: user types what they're looking for.
    Parses intent, builds a mission spec, queues it to the worker pipeline.
    Returns a job_id to poll for status and results.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Parse intent from natural language
    spec = _parse_search_intent(req.query.strip())

    # Optionally refine with LLM (non-blocking — falls back to keyword spec)
    if req.use_llm:
        spec = _try_llm_parse(req.query.strip(), spec)

    job_id = spec["mission_id"]
    _SEARCH_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "query": req.query,
        "spec": spec,
        "created": time.time(),
        "started": None,
        "finished": None,
        "error": None,
        "result_summary": None,
    }

    t = threading.Thread(target=_run_search_job, args=(job_id, spec), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "spec": spec,
        "message": (
            f"Searching for {spec['target_type']} in {spec['area_label']} "
            f"({spec['date_range'][0]} – {spec['date_range'][1]}). "
            f"Poll /search/{job_id} for status."
        ),
    }


@app.get("/search/{job_id}", tags=["search"])
def get_search_status(job_id: str):
    """Poll search job status and results."""
    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Search job not found")
    return job


@app.get("/search", tags=["search"])
def list_searches(limit: int = Query(20, ge=1, le=100)):
    """List recent search jobs, newest first."""
    jobs = sorted(_SEARCH_JOBS.values(), key=lambda j: j["created"], reverse=True)
    return {"total": len(jobs), "jobs": jobs[:limit]}


@app.get("/search/{job_id}/report", tags=["search"])
def get_search_report(job_id: str):
    """
    Return a human-readable report for a completed search job.
    Aggregates: spec, detections from DB, wreck DB matches, scan output.
    """
    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Search job not found")

    spec = job.get("spec", {})
    result = job.get("result_summary", {})
    status = job.get("status", "unknown")

    # Try to load detections from the output directory
    detections = []
    output_dir = result.get("output_dir") if result else None
    if output_dir:
        out_path = Path(output_dir)
        for jf in sorted(out_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if "candidates" in data or "detections" in data:
                    detections = data.get("candidates", data.get("detections", []))
                    break
            except Exception:
                pass

    # Cross-reference detections against wreck DB
    wreck_matches = []
    if detections:
        try:
            wreck_matches = _match_swayze_wrecks(detections[:50], search_radius_m=3000.0)
        except Exception:
            pass

    # Build elapsed time
    elapsed = None
    if job.get("started") and job.get("finished"):
        elapsed = round(job["finished"] - job["started"], 1)
    elif job.get("started"):
        elapsed = round(time.time() - job["started"], 1)

    return {
        "job_id": job_id,
        "status": status,
        "query": job.get("query", ""),
        "elapsed_seconds": elapsed,
        "spec": {
            "target": spec.get("target_name"),
            "target_type": spec.get("target_type"),
            "area": spec.get("area_label"),
            "bbox": spec.get("bbox"),
            "date_range": spec.get("date_range"),
            "sensors": spec.get("sensors"),
            "passes": spec.get("passes"),
            "sensitivity": spec.get("knobs", {}).get("hc_threshold", 2.0),
        },
        "detections": {
            "count": len(detections),
            "top": detections[:10],
        },
        "wreck_db_matches": {
            "count": len(wreck_matches),
            "top": wreck_matches[:5],
        },
        "output": result or {},
        "idle_status": _get_idle_status(),
    }


def _get_idle_status() -> dict:
    """Return current idle worker state from the state file."""
    state_path = Path(__file__).resolve().parents[1] / "db" / "worker_state.json"
    if not state_path.exists():
        return {"running": False, "message": "Worker not started"}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False, "message": "Could not read worker state"}


@app.get("/search/idle/status", tags=["search"])
def idle_status():
    """Return the current idle scan worker state."""
    return _get_idle_status()


# ── Watchdog status endpoints ─────────────────────────────────────────────────
# These let the frontend read the watchdog state file and trigger restarts.

@app.get("/watchdog/status", tags=["watchdog"])
def watchdog_status():
    """Return the current watchdog state (written by watchdog.py)."""
    state_path = Path(__file__).resolve().parents[1] / "db" / "watchdog_state.json"
    if not state_path.exists():
        return {"running": False, "message": "Watchdog not running. Deploy with: bash scripts/deploy_services.sh"}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"running": False, "error": str(e)}


@app.post("/watchdog/start-api", tags=["watchdog"])
def watchdog_start_api():
    """
    Ask systemd to start the wrecks-api service.
    Only works if the calling process has sudo rights or the service is user-level.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", "wrecks-api"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return {"status": "started", "message": "wrecks-api started via systemd --user"}
        # Try system-level (requires sudo or polkit)
        result2 = subprocess.run(
            ["sudo", "-n", "systemctl", "start", "wrecks-api"],
            capture_output=True, text=True, timeout=10,
        )
        if result2.returncode == 0:
            return {"status": "started", "message": "wrecks-api started via systemd"}
        return {"status": "failed", "message": result2.stderr.strip() or result.stderr.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/watchdog/restart", tags=["watchdog"])
def watchdog_restart_all():
    """Restart all CESAROPS services via systemd."""
    results = {}
    for svc in ("wrecks-api", "koboldcpp", "scan-worker"):
        try:
            r = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", svc],
                capture_output=True, text=True, timeout=15,
            )
            results[svc] = "restarted" if r.returncode == 0 else f"failed: {r.stderr.strip()[:100]}"
        except Exception as e:
            results[svc] = f"error: {e}"
    return {"results": results}
