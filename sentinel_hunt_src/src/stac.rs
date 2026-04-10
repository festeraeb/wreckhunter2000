use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::config::{DetectionMethod, LakeConfig};
use crate::orbits::OrbitDirection;

const STAC_ROOT: &str = "https://earth-search.aws.element84.com/v1";
const ASF_SEARCH_URL: &str = "https://api.daac.asf.alaska.edu/services/search/param";

// ── STAC search request ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct StacSearch {
    collections: Vec<String>,
    bbox: [f64; 4],
    datetime: String,
    limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<serde_json::Value>,
}

// ── STAC item (simplified) ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacItem {
    pub id: String,
    pub datetime: String,
    pub collection: String,
    pub bbox: Option<[f64; 4]>,
    pub cloud_cover: Option<f64>,
    pub properties: serde_json::Value,
    pub assets: std::collections::HashMap<String, StacAsset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacAsset {
    pub href: String,
    #[serde(rename = "type")]
    pub media_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StacFeatureCollection {
    features: Vec<serde_json::Value>,
    #[serde(default)]
    context: Option<serde_json::Value>,
}

// ── Public API ───────────────────────────────────────────────────────

pub struct StacClient {
    http: reqwest::Client,
}

impl StacClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("http client"),
        }
    }

    /// Search Sentinel-1 GRD scenes for a lake.
    pub async fn search_sar(
        &self,
        lake: &LakeConfig,
        start: NaiveDate,
        end: NaiveDate,
        orbit_direction: Option<OrbitDirection>,
        max_results: u32,
    ) -> Result<Vec<StacItem>> {
        let mut query_filters = serde_json::Map::new();

        // Sentinel-1 specific: filter by polarisation and orbit direction
        query_filters.insert(
            "sar:polarizations".into(),
            serde_json::json!({"contains": ["VV"]}),
        );
        if let Some(dir) = orbit_direction {
            let dir_str = match dir {
                OrbitDirection::Ascending => "ascending",
                OrbitDirection::Descending => "descending",
            };
            query_filters.insert(
                "sat:orbit_state".into(),
                serde_json::json!({"eq": dir_str}),
            );
        }

        let body = StacSearch {
            collections: vec!["sentinel-1-grd".into()],
            bbox: lake.bounds,
            datetime: format!(
                "{}T00:00:00Z/{}T23:59:59Z",
                start.format("%Y-%m-%d"),
                end.format("%Y-%m-%d")
            ),
            limit: max_results,
            query: Some(serde_json::Value::Object(query_filters)),
        };

        self.do_search(body).await
    }

    /// Search Sentinel-2 L2A scenes for a lake.
    pub async fn search_optical(
        &self,
        lake: &LakeConfig,
        start: NaiveDate,
        end: NaiveDate,
        max_cloud_pct: f64,
        max_results: u32,
    ) -> Result<Vec<StacItem>> {
        let mut query_filters = serde_json::Map::new();
        query_filters.insert(
            "eo:cloud_cover".into(),
            serde_json::json!({"lte": max_cloud_pct}),
        );

        let body = StacSearch {
            collections: vec!["sentinel-2-l2a".into()],
            bbox: lake.bounds,
            datetime: format!(
                "{}T00:00:00Z/{}T23:59:59Z",
                start.format("%Y-%m-%d"),
                end.format("%Y-%m-%d")
            ),
            limit: max_results,
            query: Some(serde_json::Value::Object(query_filters)),
        };

        self.do_search(body).await
    }

    /// Search for scenes appropriate to a specific detection method.
    pub async fn search_for_method(
        &self,
        lake: &LakeConfig,
        method: DetectionMethod,
        start: NaiveDate,
        end: NaiveDate,
        max_results: u32,
    ) -> Result<Vec<StacItem>> {
        match method {
            DetectionMethod::DarkSpot => {
                // Use ASF for SAR data (free with Earthdata token)
                self.search_sar_asf(lake, start, end, None, max_results).await
            }
            DetectionMethod::ClearHole | DetectionMethod::SedimentTrap => {
                let max_cloud = match method {
                    DetectionMethod::ClearHole => 10.0,    // need clear skies
                    DetectionMethod::SedimentTrap => 30.0,  // can tolerate some cloud
                    _ => 20.0,
                };
                self.search_optical(lake, start, end, max_cloud, max_results).await
            }
        }
    }

    /// Search Sentinel-1 GRD via ASF (Alaska Satellite Facility).
    /// Search is public; downloads require Earthdata bearer token.
    pub async fn search_sar_asf(
        &self,
        lake: &LakeConfig,
        start: NaiveDate,
        end: NaiveDate,
        orbit_direction: Option<OrbitDirection>,
        max_results: u32,
    ) -> Result<Vec<StacItem>> {
        let [w, s, e, n] = lake.bounds;
        let wkt = format!("POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))");

        let mut params: Vec<(&str, String)> = vec![
            ("platform", "Sentinel-1".into()),
            ("processingLevel", "GRD_HD".into()),
            ("beamMode", "IW".into()),
            ("polarization", "VV,VV+VH".into()),
            ("intersectsWith", wkt),
            ("start", format!("{}T00:00:00UTC", start.format("%Y-%m-%d"))),
            ("end", format!("{}T23:59:59UTC", end.format("%Y-%m-%d"))),
            ("maxResults", max_results.to_string()),
            ("output", "geojson".into()),
        ];

        if let Some(dir) = orbit_direction {
            params.push(("flightDirection", match dir {
                OrbitDirection::Ascending => "ASCENDING".into(),
                OrbitDirection::Descending => "DESCENDING".into(),
            }));
        }

        tracing::info!("ASF search: bbox={:?}", lake.bounds);

        let resp = self
            .http
            .get(ASF_SEARCH_URL)
            .query(&params)
            .send()
            .await
            .context("ASF search request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("ASF search returned {status}: {text}");
        }

        let body: serde_json::Value = resp.json().await.context("parse ASF response")?;
        let features = body["features"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("ASF response has no features array"))?;

        let mut items = Vec::with_capacity(features.len());
        for feat in features {
            let props = &feat["properties"];
            let id = props["sceneName"]
                .as_str()
                .or_else(|| props["fileID"].as_str())
                .unwrap_or("")
                .to_string();
            let datetime = props["startTime"].as_str().unwrap_or("").to_string();
            let download_url = props["url"].as_str().unwrap_or("").to_string();

            let bbox = bbox_from_geojson_geometry(&feat["geometry"]);

            let mut assets = std::collections::HashMap::new();
            if !download_url.is_empty() {
                assets.insert(
                    "product".into(),
                    StacAsset {
                        href: download_url,
                        media_type: Some("application/zip".into()),
                    },
                );
            }

            items.push(StacItem {
                id,
                datetime,
                collection: "sentinel-1-grd-asf".into(),
                bbox,
                cloud_cover: None,
                properties: props.clone(),
                assets,
            });
        }

        tracing::info!("ASF returned {} scenes", items.len());
        Ok(items)
    }

    // ── internal ─────────────────────────────────────────────────

    async fn do_search(&self, body: StacSearch) -> Result<Vec<StacItem>> {
        let url = format!("{STAC_ROOT}/search");
        tracing::info!("STAC search: {} {:?}", body.collections.join(","), body.bbox);

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("STAC search request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("STAC search returned {status}: {text}");
        }

        let fc: StacFeatureCollection = resp.json().await.context("parse STAC response")?;
        let mut items = Vec::with_capacity(fc.features.len());

        for feat in fc.features {
            let id = feat["id"].as_str().unwrap_or("").to_string();
            let dt = feat["properties"]["datetime"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let collection = feat["collection"].as_str().unwrap_or("").to_string();
            let bbox = feat["bbox"].as_array().and_then(|a| {
                if a.len() >= 4 {
                    Some([
                        a[0].as_f64()?,
                        a[1].as_f64()?,
                        a[2].as_f64()?,
                        a[3].as_f64()?,
                    ])
                } else {
                    None
                }
            });
            let cloud_cover = feat["properties"]["eo:cloud_cover"].as_f64();
            let properties = feat["properties"].clone();

            // Parse assets
            let assets_raw = feat["assets"].as_object();
            let mut assets = std::collections::HashMap::new();
            if let Some(obj) = assets_raw {
                for (k, v) in obj {
                    if let Some(href) = v["href"].as_str() {
                        assets.insert(
                            k.clone(),
                            StacAsset {
                                href: href.to_string(),
                                media_type: v["type"].as_str().map(String::from),
                            },
                        );
                    }
                }
            }

            items.push(StacItem {
                id,
                datetime: dt,
                collection,
                bbox,
                cloud_cover,
                properties,
                assets,
            });
        }

        tracing::info!("STAC returned {} items", items.len());
        Ok(items)
    }
}

/// Extract the relative orbit number from a STAC item's properties.
pub fn relative_orbit(item: &StacItem) -> Option<u32> {
    item.properties["sat:relative_orbit"]
        .as_u64()
        .or_else(|| item.properties["pathNumber"].as_u64())
        .map(|v| v as u32)
}

/// Extract orbit direction from a STAC item.
pub fn orbit_direction(item: &StacItem) -> Option<OrbitDirection> {
    item.properties["sat:orbit_state"]
        .as_str()
        .or_else(|| item.properties["flightDirection"].as_str())
        .and_then(|s| match s.to_lowercase().as_str() {
            "ascending" => Some(OrbitDirection::Ascending),
            "descending" => Some(OrbitDirection::Descending),
            _ => None,
        })
}

/// Get the download URL for a specific band/asset.
pub fn asset_url(item: &StacItem, band: &str) -> Option<String> {
    item.assets.get(band).map(|a| a.href.clone())
}

/// Compute bounding box from a GeoJSON geometry object.
fn bbox_from_geojson_geometry(geom: &serde_json::Value) -> Option<[f64; 4]> {
    let coords = geom["coordinates"].as_array()?;
    let ring = coords.first()?.as_array()?;
    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;
    for pt in ring {
        let arr = pt.as_array()?;
        let lon = arr.first()?.as_f64()?;
        let lat = arr.get(1)?.as_f64()?;
        min_lon = min_lon.min(lon);
        min_lat = min_lat.min(lat);
        max_lon = max_lon.max(lon);
        max_lat = max_lat.max(lat);
    }
    Some([min_lon, min_lat, max_lon, max_lat])
}
