//! GLOS (Great Lakes Observing System) ERDDAP client.
//!
//! Complements the NDBC atmospheric data in `weather.rs` with water-column
//! observations from GLOS Seagull:
//!   - Turbidity / total suspended solids  → calibrates Sediment Trap
//!   - Secchi depth / water clarity         → calibrates Clear Hole
//!   - Chlorophyll-a                        → context for optical detection
//!   - Surface currents (HF radar)          → predicts wreck-induced flow anomalies
//!   - Water temperature profiles           → thermocline affects mussel distribution
//!
//! GLOS ERDDAP base: <https://seagull.glos.org/erddap/>
//! Docs: <https://coastwatch.pfeg.noaa.gov/erddap/information.html>
//!
//! ERDDAP REST pattern:
//!   GET /erddap/tabledap/{dataset}.json?{vars}&{constraints}
//!   GET /erddap/search/index.json?searchFor={keyword}

use anyhow::{Context, Result};
use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

use crate::config::{Lake, LakeConfig};

// ── Constants ────────────────────────────────────────────────────────

/// Primary GLOS ERDDAP endpoint (Seagull platform).
const GLOS_ERDDAP: &str = "https://seagull.glos.org/erddap";

/// Fallback ERDDAP (older GLOS server, still active).
const GLOS_ERDDAP_FALLBACK: &str = "https://erddap.glos.org/erddap";

/// NDBC-compatible station IDs that also appear in GLOS with extra sensors.
/// GLOS adds water-quality sensors to many NDBC platforms.
/// Maps lake → Vec of GLOS platform IDs that carry turbidity/clarity sensors.
fn glos_water_quality_platforms(lake: Lake) -> &'static [&'static str] {
    match lake {
        // GLOS deploys water quality sensors on these platforms.
        // Platform IDs follow GLOS naming: "GLERL-{station}" or NDBC-style.
        Lake::Erie => &[
            "NDBC-ERIESP",   // Erie shore platforms
            "GLERL-ERB",     // GLERL central Erie buoy
            "45005",         // NDBC W Erie (GLOS augmented)
            "LEOBS",         // Lake Erie Observing Station
        ],
        Lake::Huron => &[
            "45003",         // NDBC Huron (GLOS augmented)
            "GLERL-HNB",     // GLERL Huron north buoy
            "45149",         // Saginaw Bay
        ],
        Lake::Michigan => &[
            "45007",         // NDBC S Michigan
            "45002",         // NDBC N Michigan
            "GLERL-MKB",     // GLERL Milwaukee buoy
        ],
        Lake::Superior => &[
            "45006",         // NDBC W Superior
            "45004",         // NDBC E Superior
        ],
        Lake::Ontario => &[
            "45012",         // NDBC Ontario
            "C45135",        // Canadian Ontario
        ],
    }
}

// ── ERDDAP response structures ──────────────────────────────────────

/// Raw ERDDAP tabledap JSON response.
#[derive(Debug, Deserialize)]
struct ErddapResponse {
    table: ErddapTable,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ErddapTable {
    column_names: Vec<String>,
    column_types: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
}

/// ERDDAP dataset search result.
#[derive(Debug, Deserialize)]
struct ErddapSearchResponse {
    table: ErddapSearchTable,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ErddapSearchTable {
    column_names: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
}

// ── Observation types ────────────────────────────────────────────────

/// Water quality observation from a GLOS sensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterQualityObs {
    pub platform_id: String,
    pub timestamp: NaiveDateTime,
    pub lat: f64,
    pub lon: f64,
    /// Turbidity in NTU (Nephelometric Turbidity Units).
    /// Low = clear water. Erie baseline: 3-10 NTU; post-storm: 50-200+ NTU.
    pub turbidity_ntu: Option<f64>,
    /// Total Suspended Solids (mg/L). Correlated with turbidity.
    pub tss_mg_l: Option<f64>,
    /// Secchi depth (m). Disk-measured water clarity.
    /// Erie: 1-3m typical; "clear hole" over mussels: 5-8m.
    pub secchi_depth_m: Option<f64>,
    /// Chlorophyll-a (µg/L). Phytoplankton proxy.
    /// Low chlorophyll + high clarity → mussel filtration zone.
    pub chlorophyll_ug_l: Option<f64>,
    /// Dissolved oxygen (mg/L). Context only.
    pub dissolved_oxygen_mg_l: Option<f64>,
    /// Water temperature at sensor depth (°C).
    pub water_temp_c: Option<f64>,
    /// Sensor depth below surface (m).
    pub sensor_depth_m: Option<f64>,
}

/// Surface current observation from GLOS HF radar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceCurrentObs {
    pub timestamp: NaiveDateTime,
    pub lat: f64,
    pub lon: f64,
    /// Eastward current velocity (m/s, positive = east).
    pub u_ms: f64,
    /// Northward current velocity (m/s, positive = north).
    pub v_ms: f64,
}

impl SurfaceCurrentObs {
    /// Current speed magnitude (m/s).
    pub fn speed_ms(&self) -> f64 {
        (self.u_ms.powi(2) + self.v_ms.powi(2)).sqrt()
    }

    /// Current direction (degrees, oceanographic convention: direction flow is GOING TO).
    pub fn direction_deg(&self) -> f64 {
        let deg = self.u_ms.atan2(self.v_ms).to_degrees();
        if deg < 0.0 { deg + 360.0 } else { deg }
    }
}

/// Combined assessment of in-water conditions at a site.
/// Use with WeatherAssessment (atmosphere) for full detection scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterConditions {
    pub lake: Lake,
    pub assessed_at: NaiveDateTime,
    /// Is the water abnormally clear? (Secchi or turbidity vs. baseline)
    pub clarity_anomaly: Option<ClarityAnomaly>,
    /// Is there elevated turbidity suggesting recent storm mixing?
    pub turbidity_elevated: bool,
    /// Surface current speed near ROI (m/s).
    pub surface_current_ms: Option<f64>,
    /// Overall suitability score for satellite detection (0-1).
    pub detection_suitability: f64,
    pub reason: String,
    pub observations_used: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarityAnomaly {
    /// Measured Secchi depth (m) or turbidity-derived equivalent.
    pub secchi_m: f64,
    /// Expected baseline Secchi depth for this lake/season (m).
    pub baseline_secchi_m: f64,
    /// How many sigma above baseline clarity?
    pub sigma: f64,
}

// ── GLOS client ──────────────────────────────────────────────────────

pub struct GlosClient {
    http: reqwest::Client,
    base_url: String,
}

impl GlosClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .build()
                .expect("http client"),
            base_url: GLOS_ERDDAP.to_string(),
        }
    }

    /// Search GLOS ERDDAP for datasets matching a keyword.
    /// Useful for discovering available sensors in a lake.
    pub async fn search_datasets(&self, keyword: &str) -> Result<Vec<DatasetInfo>> {
        let url = format!(
            "{}/search/index.json?searchFor={}&page=1&itemsPerPage=100",
            self.base_url,
            urlencoding::encode(keyword)
        );

        let resp = self.http.get(&url).send().await;
        let resp = match resp {
            Ok(r) => r,
            Err(_) => {
                // Fallback to secondary ERDDAP
                tracing::info!("primary GLOS ERDDAP unavailable, trying fallback");
                self.http
                    .get(url.replace(GLOS_ERDDAP, GLOS_ERDDAP_FALLBACK))
                    .send()
                    .await
                    .context("GLOS ERDDAP search (fallback)")?
            }
        };

        if !resp.status().is_success() {
            anyhow::bail!("GLOS search returned {}", resp.status());
        }

        let body: ErddapSearchResponse = resp.json().await?;
        parse_search_results(&body)
    }

    /// Fetch water quality observations for a lake within a date range.
    /// Tries known platform IDs for that lake.
    pub async fn fetch_water_quality(
        &self,
        lake: Lake,
        bbox: [f64; 4],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<WaterQualityObs>> {
        let platforms = glos_water_quality_platforms(lake);
        let mut all_obs = Vec::new();

        for platform_id in platforms {
            match self
                .fetch_platform_wq(platform_id, bbox, start, end)
                .await
            {
                Ok(obs) => {
                    tracing::info!(
                        "GLOS {}: {} water quality obs",
                        platform_id,
                        obs.len()
                    );
                    all_obs.extend(obs);
                }
                Err(e) => {
                    tracing::debug!("GLOS {} water quality unavailable: {}", platform_id, e);
                }
            }
        }

        Ok(all_obs)
    }

    /// Query a single GLOS platform for water quality data.
    async fn fetch_platform_wq(
        &self,
        platform_id: &str,
        bbox: [f64; 4],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<WaterQualityObs>> {
        // ERDDAP tabledap query — request common water quality variables.
        // Not all platforms have all variables; ERDDAP returns what's available.
        // We try the most common GLOS dataset naming patterns.
        let dataset_candidates = vec![
            format!("obs_{}", platform_id.to_lowercase().replace('-', "_")),
            format!("{}_water_quality", platform_id.to_lowercase().replace('-', "_")),
            platform_id.to_lowercase(),
        ];

        let vars = "time,latitude,longitude,turbidity,tss,secchi_depth,chlorophyll,dissolved_oxygen,water_temperature,depth";
        let start_str = start.format("%Y-%m-%dT00:00:00Z");
        let end_str = end.format("%Y-%m-%dT23:59:59Z");

        for dataset_id in &dataset_candidates {
            let url = format!(
                "{}/tabledap/{}.json?{}&time>={}&time<={}&latitude>={}&latitude<={}&longitude>={}&longitude<={}",
                self.base_url, dataset_id, vars,
                start_str, end_str,
                bbox[1], bbox[3], bbox[0], bbox[2]
            );

            let resp = match self.http.get(&url).send().await {
                Ok(r) if r.status().is_success() => r,
                _ => continue, // Try next dataset name
            };

            let body: ErddapResponse = match resp.json().await {
                Ok(b) => b,
                Err(_) => continue,
            };

            let obs = parse_water_quality(&body, platform_id)?;
            if !obs.is_empty() {
                return Ok(obs);
            }
        }

        anyhow::bail!("no water quality data found for {}", platform_id)
    }

    /// Fetch HF radar surface currents near a point.
    /// GLOS HF radar covers parts of Lakes Michigan, Huron, and Erie.
    pub async fn fetch_surface_currents(
        &self,
        lake: Lake,
        center_lat: f64,
        center_lon: f64,
        radius_deg: f64,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<SurfaceCurrentObs>> {
        // GLOS HF radar dataset IDs follow lake naming
        let dataset_id = match lake {
            Lake::Erie => "glos_hfr_erie",
            Lake::Huron => "glos_hfr_huron",
            Lake::Michigan => "glos_hfr_michigan",
            _ => anyhow::bail!("no HF radar coverage for {}", lake),
        };

        let start_str = start.format("%Y-%m-%dT00:00:00Z");
        let end_str = end.format("%Y-%m-%dT23:59:59Z");

        let url = format!(
            "{}/tabledap/{}.json?time,latitude,longitude,u,v&time>={}&time<={}&latitude>={}&latitude<={}&longitude>={}&longitude<={}",
            self.base_url, dataset_id,
            start_str, end_str,
            center_lat - radius_deg, center_lat + radius_deg,
            center_lon - radius_deg, center_lon + radius_deg,
        );

        let resp = self.http.get(&url).send().await
            .context("GLOS HF radar fetch")?;

        if !resp.status().is_success() {
            anyhow::bail!("HF radar query returned {}", resp.status());
        }

        let body: ErddapResponse = resp.json().await?;
        parse_surface_currents(&body)
    }

    /// Assess in-water conditions for detection suitability.
    /// Combine with WeatherAssessment for complete picture.
    pub async fn assess_water_conditions(
        &self,
        lake_cfg: &LakeConfig,
        method: crate::config::DetectionMethod,
    ) -> Result<WaterConditions> {
        let now = chrono::Utc::now().naive_utc();
        let today = now.date();
        let week_ago = today - chrono::Duration::days(7);

        let obs = self
            .fetch_water_quality(lake_cfg.lake, lake_cfg.bounds, week_ago, today)
            .await
            .unwrap_or_default();

        if obs.is_empty() {
            return Ok(WaterConditions {
                lake: lake_cfg.lake,
                assessed_at: now,
                clarity_anomaly: None,
                turbidity_elevated: false,
                surface_current_ms: None,
                detection_suitability: 0.5, // neutral — no data
                reason: "no recent GLOS water quality data available".into(),
                observations_used: 0,
            });
        }

        // Get recent turbidity and clarity stats
        let recent_turb: Vec<f64> = obs.iter().filter_map(|o| o.turbidity_ntu).collect();
        let recent_secchi: Vec<f64> = obs.iter().filter_map(|o| o.secchi_depth_m).collect();

        let baseline = lake_baselines(lake_cfg.lake);

        // Check clarity anomaly (relevant for ClearHole)
        let clarity_anomaly = if !recent_secchi.is_empty() {
            let avg_secchi = recent_secchi.iter().sum::<f64>() / recent_secchi.len() as f64;
            let sigma = (avg_secchi - baseline.secchi_m) / baseline.secchi_std.max(0.1);
            if sigma > 1.5 {
                Some(ClarityAnomaly {
                    secchi_m: avg_secchi,
                    baseline_secchi_m: baseline.secchi_m,
                    sigma,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Check turbidity relative to baseline (relevant for SedimentTrap)
        let turbidity_elevated = if !recent_turb.is_empty() {
            let avg_turb = recent_turb.iter().sum::<f64>() / recent_turb.len() as f64;
            avg_turb > baseline.turbidity_ntu * 2.0
        } else {
            false
        };

        // Score detection suitability based on method
        use crate::config::DetectionMethod;
        let suitability = match method {
            DetectionMethod::ClearHole => {
                // Want high clarity — clear holes show best when surrounding water is clear too
                if clarity_anomaly.is_some() {
                    0.9
                } else if recent_secchi.iter().any(|&s| s > baseline.secchi_m) {
                    0.7
                } else {
                    0.3
                }
            }
            DetectionMethod::SedimentTrap => {
                // Want elevated turbidity — post-storm plumes need suspended material
                if turbidity_elevated {
                    0.9
                } else if recent_turb.iter().any(|&t| t > baseline.turbidity_ntu) {
                    0.6
                } else {
                    0.2
                }
            }
            DetectionMethod::DarkSpot => {
                // SAR doesn't directly use water quality, but currents help
                0.5
            }
        };

        Ok(WaterConditions {
            lake: lake_cfg.lake,
            assessed_at: now,
            clarity_anomaly,
            turbidity_elevated,
            surface_current_ms: None, // TODO: add HF radar query
            detection_suitability: suitability,
            reason: format!(
                "turbidity: {} obs (elevated={}), secchi: {} obs",
                recent_turb.len(),
                turbidity_elevated,
                recent_secchi.len()
            ),
            observations_used: obs.len(),
        })
    }
}

// ── Lake baselines ───────────────────────────────────────────────────
// Typical water clarity / turbidity values for each lake (summer).
// These are approximate; eventually could be computed from GLOS historical data.

struct LakeBaseline {
    secchi_m: f64,
    secchi_std: f64,
    turbidity_ntu: f64,
    chlorophyll_ug_l: f64,
}

fn lake_baselines(lake: Lake) -> LakeBaseline {
    match lake {
        Lake::Erie => LakeBaseline {
            // Western basin: 0.5-2m; Central: 2-4m; Eastern: 3-6m
            secchi_m: 2.5,
            secchi_std: 1.5,
            turbidity_ntu: 8.0,
            chlorophyll_ug_l: 5.0,
        },
        Lake::Huron => LakeBaseline {
            // Dreissenid-cleared, quite clear
            secchi_m: 6.0,
            secchi_std: 2.0,
            turbidity_ntu: 2.0,
            chlorophyll_ug_l: 1.5,
        },
        Lake::Michigan => LakeBaseline {
            // Southern basin murkier, northern very clear
            secchi_m: 5.0,
            secchi_std: 2.5,
            turbidity_ntu: 3.0,
            chlorophyll_ug_l: 2.0,
        },
        Lake::Superior => LakeBaseline {
            // Naturally ultra-oligotrophic, very clear
            secchi_m: 10.0,
            secchi_std: 3.0,
            turbidity_ntu: 0.5,
            chlorophyll_ug_l: 0.5,
        },
        Lake::Ontario => LakeBaseline {
            secchi_m: 4.0,
            secchi_std: 2.0,
            turbidity_ntu: 3.5,
            chlorophyll_ug_l: 2.5,
        },
    }
}

// ── Parsers ──────────────────────────────────────────────────────────

fn parse_search_results(resp: &ErddapSearchResponse) -> Result<Vec<DatasetInfo>> {
    let cols = &resp.table.column_names;

    let idx = |name: &str| cols.iter().position(|c| c == name);
    let id_col = idx("datasetID").or_else(|| idx("Dataset ID"));
    let title_col = idx("Title").or_else(|| idx("title"));
    let summary_col = idx("Summary").or_else(|| idx("summary"));

    let mut datasets = Vec::new();
    for row in &resp.table.rows {
        let get_str = |col: Option<usize>| -> String {
            col.and_then(|i| row.get(i))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        datasets.push(DatasetInfo {
            dataset_id: get_str(id_col),
            title: get_str(title_col),
            summary: get_str(summary_col),
        });
    }

    Ok(datasets)
}

fn parse_water_quality(
    resp: &ErddapResponse,
    platform_id: &str,
) -> Result<Vec<WaterQualityObs>> {
    let cols = &resp.table.column_names;
    let find = |name: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(name));

    let time_col = find("time").ok_or_else(|| anyhow::anyhow!("no time column"))?;
    let lat_col = find("latitude").ok_or_else(|| anyhow::anyhow!("no latitude column"))?;
    let lon_col = find("longitude").ok_or_else(|| anyhow::anyhow!("no longitude column"))?;

    let turb_col = find("turbidity");
    let tss_col = find("tss").or_else(|| find("total_suspended_solids"));
    let secchi_col = find("secchi_depth").or_else(|| find("secchi"));
    let chl_col = find("chlorophyll").or_else(|| find("chlorophyll_a"));
    let do_col = find("dissolved_oxygen");
    let wtemp_col = find("water_temperature").or_else(|| find("sea_water_temperature"));
    let depth_col = find("depth").or_else(|| find("z"));

    let mut obs = Vec::new();
    for row in &resp.table.rows {
        let get_f64 = |col: Option<usize>| -> Option<f64> {
            col.and_then(|i| row.get(i))
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        };

        let time_str = row.get(time_col).and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = parse_erddap_time(time_str);
        let timestamp = match timestamp {
            Some(t) => t,
            None => continue,
        };

        let lat = get_f64(Some(lat_col)).unwrap_or(0.0);
        let lon = get_f64(Some(lon_col)).unwrap_or(0.0);

        obs.push(WaterQualityObs {
            platform_id: platform_id.to_string(),
            timestamp,
            lat,
            lon,
            turbidity_ntu: get_f64(turb_col),
            tss_mg_l: get_f64(tss_col),
            secchi_depth_m: get_f64(secchi_col),
            chlorophyll_ug_l: get_f64(chl_col),
            dissolved_oxygen_mg_l: get_f64(do_col),
            water_temp_c: get_f64(wtemp_col),
            sensor_depth_m: get_f64(depth_col),
        });
    }

    Ok(obs)
}

fn parse_surface_currents(resp: &ErddapResponse) -> Result<Vec<SurfaceCurrentObs>> {
    let cols = &resp.table.column_names;
    let find = |name: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(name));

    let time_col = find("time").ok_or_else(|| anyhow::anyhow!("no time column"))?;
    let lat_col = find("latitude").ok_or_else(|| anyhow::anyhow!("no latitude column"))?;
    let lon_col = find("longitude").ok_or_else(|| anyhow::anyhow!("no longitude column"))?;
    let u_col = find("u").or_else(|| find("u_velocity")).ok_or_else(|| anyhow::anyhow!("no u column"))?;
    let v_col = find("v").or_else(|| find("v_velocity")).ok_or_else(|| anyhow::anyhow!("no v column"))?;

    let mut obs = Vec::new();
    for row in &resp.table.rows {
        let get_f64 = |col: usize| -> Option<f64> {
            row.get(col)
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        };

        let time_str = row.get(time_col).and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = match parse_erddap_time(time_str) {
            Some(t) => t,
            None => continue,
        };

        let lat = get_f64(lat_col).unwrap_or(0.0);
        let lon = get_f64(lon_col).unwrap_or(0.0);
        let u = match get_f64(u_col) { Some(v) => v, None => continue };
        let v = match get_f64(v_col) { Some(v) => v, None => continue };

        obs.push(SurfaceCurrentObs {
            timestamp,
            lat,
            lon,
            u_ms: u,
            v_ms: v,
        });
    }

    Ok(obs)
}

/// Parse ERDDAP ISO-8601 timestamps: "2024-08-15T14:00:00Z"
fn parse_erddap_time(s: &str) -> Option<NaiveDateTime> {
    // Try standard ISO format first
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
        .ok()
}

// ── Dataset info ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub dataset_id: String,
    pub title: String,
    pub summary: String,
}

impl DatasetInfo {
    /// Check if this dataset likely has water quality variables.
    pub fn is_water_quality(&self) -> bool {
        let text = format!("{} {}", self.title, self.summary).to_lowercase();
        text.contains("turbid")
            || text.contains("secchi")
            || text.contains("clarity")
            || text.contains("chlorophyll")
            || text.contains("water quality")
            || text.contains("tss")
    }

    /// Check if this dataset has current data.
    pub fn is_currents(&self) -> bool {
        let text = format!("{} {}", self.title, self.summary).to_lowercase();
        text.contains("current") || text.contains("hf radar") || text.contains("hfr")
    }
}

// ── Utility: combine atmospheric + water-column data ─────────────────

/// Full environmental picture combining NDBC weather + GLOS water column.
/// Use this to make the go/no-go decision for satellite image acquisition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalAssessment {
    pub lake: Lake,
    pub timestamp: NaiveDateTime,
    pub weather: crate::weather::WeatherAssessment,
    pub water: WaterConditions,
    /// Combined suitability score (0-1).
    pub combined_score: f64,
    pub recommendation: String,
}

/// Combine weather and water assessments into a single recommendation.
pub fn combine_assessments(
    weather: &crate::weather::WeatherAssessment,
    water: &WaterConditions,
    method: crate::config::DetectionMethod,
) -> EnvironmentalAssessment {
    let weather_score = weather.confidence;
    let water_score = water.detection_suitability;

    // Weight depends on method:
    //   DarkSpot → weather-dominant (wind transition matters most)
    //   ClearHole → water-dominant (clarity matters most)
    //   SedimentTrap → both matter (need storm + turbidity)
    use crate::config::DetectionMethod;
    let (w_weight, wq_weight) = match method {
        DetectionMethod::DarkSpot => (0.8, 0.2),
        DetectionMethod::ClearHole => (0.3, 0.7),
        DetectionMethod::SedimentTrap => (0.5, 0.5),
    };

    let combined = weather_score * w_weight + water_score * wq_weight;

    let recommendation = if combined >= 0.7 {
        format!("GO — excellent conditions for {} detection", method.label())
    } else if combined >= 0.4 {
        format!("MARGINAL — conditions acceptable for {} but not ideal", method.label())
    } else {
        format!("NO-GO — poor conditions for {} detection", method.label())
    };

    EnvironmentalAssessment {
        lake: water.lake,
        timestamp: water.assessed_at,
        weather: weather.clone(),
        water: water.clone(),
        combined_score: combined,
        recommendation,
    }
}
