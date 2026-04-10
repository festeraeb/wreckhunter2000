use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::config::{DetectionMethod, Lake};

// ── Known wreck for validation ───────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownWreck {
    pub id: i64,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub lake: String,
    pub depth_m: Option<f64>,
    pub is_steel: bool,
    pub has_coords: bool,
    pub coord_quality: Option<String>,
    pub spatial_extent_m: Option<f64>,
}

// ── Detection match ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMatch {
    pub wreck: KnownWreck,
    pub detection_lat: f64,
    pub detection_lon: f64,
    pub distance_m: f64,
    pub bearing_deg: f64,
    pub method: DetectionMethod,
    pub confidence: f64,
}

// ── Validation report ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub lake: Lake,
    pub method: DetectionMethod,
    pub total_detections: usize,
    pub total_known_wrecks: usize,
    pub matches: Vec<ValidationMatch>,
    pub unmatched_detections: usize,
    pub detection_rate: f64,           // matches / known_wrecks_in_area
    pub false_positive_rate: f64,      // unmatched / total_detections
    pub mean_offset_m: f64,
}

// ── Database queries ─────────────────────────────────────────────────

/// Load known wrecks from the wrecks.db SQLite database.
pub fn load_known_wrecks(
    db_path: &str,
    lake: Lake,
    require_coords: bool,
) -> Result<Vec<KnownWreck>> {
    let conn = Connection::open(db_path)
        .context(format!("open wrecks db: {db_path}"))?;

    let lake_name = lake.label();
    let sql = if require_coords {
        "SELECT id, name, lat, lon, lake, depth_m, is_steel, has_coords, coord_quality, spatial_extent_m \
         FROM features WHERE lake = ?1 AND has_coords = 1 AND lat IS NOT NULL AND lon IS NOT NULL"
    } else {
        "SELECT id, name, lat, lon, lake, depth_m, is_steel, has_coords, coord_quality, spatial_extent_m \
         FROM features WHERE lake = ?1"
    };

    let mut stmt = conn.prepare(sql)?;
    let wrecks = stmt
        .query_map([lake_name], |row| {
            Ok(KnownWreck {
                id: row.get(0)?,
                name: row.get(1)?,
                lat: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                lon: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                lake: row.get(4)?,
                depth_m: row.get(5)?,
                is_steel: row.get::<_, Option<bool>>(6)?.unwrap_or(false),
                has_coords: row.get::<_, Option<bool>>(7)?.unwrap_or(false),
                coord_quality: row.get(8)?,
                spatial_extent_m: row.get(9)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(wrecks)
}

/// Load wrecks within a bounding box (for ROI-specific validation).
pub fn load_wrecks_in_bbox(
    db_path: &str,
    bbox: [f64; 4], // [west, south, east, north]
) -> Result<Vec<KnownWreck>> {
    let conn = Connection::open(db_path)?;

    let sql = "SELECT id, name, lat, lon, lake, depth_m, is_steel, has_coords, coord_quality, spatial_extent_m \
               FROM features \
               WHERE has_coords = 1 AND lat IS NOT NULL AND lon IS NOT NULL \
               AND lon >= ?1 AND lat >= ?2 AND lon <= ?3 AND lat <= ?4";

    let mut stmt = conn.prepare(sql)?;
    let wrecks = stmt
        .query_map(rusqlite::params![bbox[0], bbox[1], bbox[2], bbox[3]], |row| {
            Ok(KnownWreck {
                id: row.get(0)?,
                name: row.get(1)?,
                lat: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                lon: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                lake: row.get(4)?,
                depth_m: row.get(5)?,
                is_steel: row.get::<_, Option<bool>>(6)?.unwrap_or(false),
                has_coords: row.get::<_, Option<bool>>(7)?.unwrap_or(false),
                coord_quality: row.get(8)?,
                spatial_extent_m: row.get(9)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(wrecks)
}

// ── Matching logic ───────────────────────────────────────────────────

/// Match detections against known wrecks.
///
/// A detection "matches" a wreck if it's within `match_radius_m` meters.
pub fn validate_detections(
    detections: &[(f64, f64, f64)],  // (lat, lon, confidence)
    known_wrecks: &[KnownWreck],
    method: DetectionMethod,
    lake: Lake,
    match_radius_m: f64,
) -> ValidationReport {
    let mut matches = Vec::new();
    let mut matched_wrecks = std::collections::HashSet::new();

    for &(det_lat, det_lon, confidence) in detections {
        let mut best_match: Option<(usize, f64, f64)> = None; // (wreck_idx, distance, bearing)

        for (i, wreck) in known_wrecks.iter().enumerate() {
            if !wreck.has_coords || wreck.lat == 0.0 {
                continue;
            }

            let (dist, bearing) = haversine_distance_bearing(
                det_lat, det_lon, wreck.lat, wreck.lon,
            );

            if dist <= match_radius_m {
                if best_match.is_none() || dist < best_match.unwrap().1 {
                    best_match = Some((i, dist, bearing));
                }
            }
        }

        if let Some((idx, dist, bearing)) = best_match {
            matched_wrecks.insert(idx);
            matches.push(ValidationMatch {
                wreck: known_wrecks[idx].clone(),
                detection_lat: det_lat,
                detection_lon: det_lon,
                distance_m: dist,
                bearing_deg: bearing,
                method,
                confidence,
            });
        }
    }

    let unmatched = detections.len() - matches.len();
    let coordinated_wrecks = known_wrecks.iter()
        .filter(|w| w.has_coords && w.lat != 0.0)
        .count();

    let mean_offset = if matches.is_empty() {
        0.0
    } else {
        matches.iter().map(|m| m.distance_m).sum::<f64>() / matches.len() as f64
    };

    ValidationReport {
        lake,
        method,
        total_detections: detections.len(),
        total_known_wrecks: coordinated_wrecks,
        detection_rate: if coordinated_wrecks > 0 {
            matched_wrecks.len() as f64 / coordinated_wrecks as f64
        } else {
            0.0
        },
        false_positive_rate: if detections.is_empty() {
            0.0
        } else {
            unmatched as f64 / detections.len() as f64
        },
        mean_offset_m: mean_offset,
        matches,
        unmatched_detections: unmatched,
    }
}

// ── Haversine ────────────────────────────────────────────────────────

fn haversine_distance_bearing(
    lat1: f64, lon1: f64,
    lat2: f64, lon2: f64,
) -> (f64, f64) {
    let r = 6_371_000.0; // Earth radius in meters

    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a = (dlat / 2.0).sin().powi(2)
        + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = r * c;

    // Bearing
    let y = dlon.sin() * lat2_r.cos();
    let x = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.cos();
    let bearing = y.atan2(x).to_degrees();
    let bearing = (bearing + 360.0) % 360.0;

    (distance, bearing)
}

/// Score matching by wreck characteristics (hull type, depth, etc.)
pub fn score_by_characteristics(report: &ValidationReport) -> serde_json::Value {
    let mut steel_matches = 0;
    let _steel_total = 0;
    let mut shallow_matches = 0; // < 30m
    let _shallow_total = 0;
    let mut deep_matches = 0; // >= 30m
    let _deep_total = 0;

    for m in &report.matches {
        if m.wreck.is_steel {
            steel_matches += 1;
        }
        match m.wreck.depth_m {
            Some(d) if d < 30.0 => shallow_matches += 1,
            Some(_) => deep_matches += 1,
            None => {}
        }
    }

    // Count totals from all known wrecks in the lake (approximation from matches)
    serde_json::json!({
        "method": report.method,
        "steel_hull_matches": steel_matches,
        "shallow_matches": shallow_matches,
        "deep_matches": deep_matches,
        "detection_rate": report.detection_rate,
        "false_positive_rate": report.false_positive_rate,
        "mean_offset_m": report.mean_offset_m,
    })
}
