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
import sys

# â”€â”€ Inject pipeline source directories so lazy `from X import Y` calls work â”€â”€
# Repo root = two levels up from wrecks_api/app.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("pipelines/mag", "pipelines/satellite", "pipelines/bag",
             "ml/training", "ml/inference", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Query
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


# â”€â”€ Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
@app.get("/wrecks/{wreck_id}", tags=["wrecks"])
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


JOBS = {}


def _run_scan_job(job_id: str, paths: list, output_dir: str, config: dict,
                  swayze_match: bool = True, swayze_radius_m: float = 2000.0):
    JOBS[job_id]['status'] = 'running'
    JOBS[job_id]['start_time'] = time.time()
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
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        JOBS[job_id]['end_time'] = time.time()


@app.post('/scan/start', tags=['scan'])
def start_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'id': job_id,
        'status': 'queued',
        'paths': req.paths,
        'output_dir': req.output_dir,
        'config': req.config,
        'created': time.time()
    }

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

class AutoBagRequest(BaseModel):
    throttle_mode: str  # "unfettered", "half", "custom"
    custom_kbps: Optional[int] = None
    scan_mode: Optional[str] = "masked" # "masked", "unmasked", "both"

import subprocess

@app.post('/tools/auto-bag/start', tags=['tools'])
def start_auto_bag(req: AutoBagRequest):
    import time
    job_id = str(uuid.uuid4())

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

    cmd = ["python", "bag_auto_pipeline.py"]
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

