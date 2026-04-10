use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::bridge::PythonBridge;
use crate::config::{DetectionMethod, Lake, LakeConfig};
use crate::stac::{self, StacClient, StacItem};
use crate::weather::{WeatherAssessment, WeatherClient};

// ── Detection result ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub scene_id: String,
    pub method: DetectionMethod,
    pub lake: Lake,
    pub datetime: String,
    pub detections: Vec<Detection>,
    pub weather: Option<WeatherAssessment>,
    pub orbit_info: Option<OrbitInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub lat: f64,
    pub lon: f64,
    pub confidence: f64,
    pub anomaly_sigma: f64,
    pub radius_m: f64,
    pub classification: String, // "laminar", "karman_vortex", "clarity_hole", "sediment_plume", etc.
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitInfo {
    pub relative_orbit: u32,
    pub direction: String,
    pub incidence_deg: f64,
    pub is_far_range: bool,
}

// ── Detection pipeline ───────────────────────────────────────────────

pub struct DetectionPipeline {
    stac: StacClient,
    weather: WeatherClient,
    bridge: PythonBridge,
    cache_dir: PathBuf,
    db_path: String,
}

impl DetectionPipeline {
    pub fn new(
        bridge: PythonBridge,
        cache_dir: PathBuf,
        db_path: String,
    ) -> Self {
        Self {
            stac: StacClient::new(),
            weather: WeatherClient::new(),
            bridge,
            cache_dir,
            db_path,
        }
    }

    /// Full pipeline: search → weather-check → fetch → detect.
    pub async fn run(
        &self,
        lake_cfg: &LakeConfig,
        method: DetectionMethod,
        start: NaiveDate,
        end: NaiveDate,
        max_scenes: u32,
    ) -> Result<Vec<DetectionResult>> {
        tracing::info!(
            "pipeline: {} {} {}-{}",
            lake_cfg.lake, method.label(), start, end
        );

        // 1. Search STAC for matching scenes
        let items = self.stac
            .search_for_method(lake_cfg, method, start, end, max_scenes)
            .await
            .context("STAC search")?;

        tracing::info!("found {} candidate scenes", items.len());

        // 2. Process each scene
        let mut results = Vec::new();
        for item in &items {
            match self.process_scene(lake_cfg, method, item).await {
                Ok(r) => results.push(r),
                Err(e) => tracing::warn!("scene {} failed: {}", item.id, e),
            }
        }

        tracing::info!("{} scenes produced results", results.len());
        Ok(results)
    }

    /// Process a single scene: fetch bands → run detector.
    async fn process_scene(
        &self,
        lake_cfg: &LakeConfig,
        method: DetectionMethod,
        item: &StacItem,
    ) -> Result<DetectionResult> {
        tracing::info!("processing scene {} ({})", item.id, item.datetime);

        // Collect asset URLs for the required bands
        let required_bands = method.bands();
        let mut asset_urls = std::collections::HashMap::new();
        for band in required_bands {
            if let Some(url) = stac::asset_url(item, band) {
                asset_urls.insert(band.to_string(), url);
            }
        }

        // ASF fallback: if no individual band URLs, use "product" asset (ZIP download)
        if asset_urls.is_empty() {
            if let Some(url) = stac::asset_url(item, "product") {
                tracing::info!("using ASF product download for {}", item.id);
                asset_urls.insert("product".to_string(), url);
            }
        }

        // Fetch bands via Python (windowed to lake/ROI bounds)
        let scene_cache = self.cache_dir.join(&item.id);
        let fetch_result = self.bridge.fetch_bands(
            &item.id,
            required_bands,
            &asset_urls,
            &scene_cache,
            Some(lake_cfg.bounds),
        )?;

        if !fetch_result.success {
            anyhow::bail!(
                "band fetch failed: {}",
                fetch_result.error.unwrap_or_default()
            );
        }

        // Build band_paths from fetch result metadata
        let band_paths: std::collections::HashMap<String, String> = fetch_result
            .metadata
            .get("band_paths")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Run the appropriate detector
        let detector_name = method.label();
        let det_result = self.bridge.run_detector(
            detector_name,
            &band_paths,
            Some(lake_cfg.bounds),
            None,
        )?;

        if !det_result.success {
            anyhow::bail!(
                "detector {} failed: {}",
                detector_name,
                det_result.error.unwrap_or_default()
            );
        }

        // Parse detections
        let detections: Vec<Detection> = det_result
            .detections
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();

        // Extract orbit info for SAR scenes
        let orbit_info = if method == DetectionMethod::DarkSpot {
            let rel_orbit = stac::relative_orbit(item);
            let direction = stac::orbit_direction(item);
            rel_orbit.map(|o| OrbitInfo {
                relative_orbit: o,
                direction: direction
                    .map(|d| format!("{:?}", d))
                    .unwrap_or("unknown".into()),
                incidence_deg: 37.0, // fallback; refine with orbits::classify_swath_position
                is_far_range: false,
            })
        } else {
            None
        };

        Ok(DetectionResult {
            scene_id: item.id.clone(),
            method,
            lake: lake_cfg.lake,
            datetime: item.datetime.clone(),
            detections,
            weather: None, // caller can enrich with weather assessment
            orbit_info,
        })
    }

    /// Weather-filtered scan: only process scenes with matching conditions.
    pub async fn run_weather_filtered(
        &self,
        lake_cfg: &LakeConfig,
        method: DetectionMethod,
        start: NaiveDate,
        end: NaiveDate,
        max_scenes: u32,
    ) -> Result<Vec<DetectionResult>> {
        // First find historical weather-matching dates
        let weather_dates = self.weather
            .find_historical_windows(lake_cfg, method, start.year())
            .await
            .unwrap_or_default();

        tracing::info!(
            "found {} weather-matching dates for {} {}",
            weather_dates.len(), lake_cfg.lake, method.label()
        );

        if weather_dates.is_empty() {
            tracing::warn!("no weather matches found, running unfiltered");
            return self.run(lake_cfg, method, start, end, max_scenes).await;
        }

        // Search STAC for scenes on those specific dates
        let mut all_results = Vec::new();
        for date in weather_dates.iter().take(max_scenes as usize) {
            let day_start = *date;
            let day_end = *date + chrono::Duration::days(1);

            let items = self.stac
                .search_for_method(lake_cfg, method, day_start, day_end, 5)
                .await
                .unwrap_or_default();

            for item in &items {
                match self.process_scene(lake_cfg, method, item).await {
                    Ok(r) => all_results.push(r),
                    Err(e) => tracing::warn!("scene {} failed: {}", item.id, e),
                }
            }
        }

        Ok(all_results)
    }
}

/// Merge results from multiple detection runs, deduplicating nearby detections.
pub fn merge_detections(
    results: &[DetectionResult],
    dedup_radius_m: f64,
) -> Vec<Detection> {
    let mut all_dets: Vec<Detection> = results
        .iter()
        .flat_map(|r| r.detections.clone())
        .collect();

    // Sort by confidence (highest first)
    all_dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate: keep highest-confidence detection within radius
    let mut kept = Vec::new();
    let mut suppressed = vec![false; all_dets.len()];

    for i in 0..all_dets.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(all_dets[i].clone());

        // Suppress nearby lower-confidence detections
        for j in (i + 1)..all_dets.len() {
            if suppressed[j] {
                continue;
            }
            let dist = haversine_m(
                all_dets[i].lat, all_dets[i].lon,
                all_dets[j].lat, all_dets[j].lon,
            );
            if dist < dedup_radius_m {
                suppressed[j] = true;
            }
        }
    }

    kept
}

fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    r * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}
