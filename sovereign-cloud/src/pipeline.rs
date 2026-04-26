use anyhow::Result;
use chrono::Utc;
use nauticuvs::protocol::{NodeRole, TaskRequest, TaskType};
use nauticuvs::synthetic_grid::SyntheticTile;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::allocation::AllocationEngine;
use crate::tile_store::TileStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassResult {
    pub task_id: String,
    pub pass: String,
    pub success: bool,
    pub anomaly_confidence: f32,
    pub output: serde_json::Value,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    pub tile_id: String,
    pub confidence: f32,
    pub lat: f64,
    pub lon: f64,
    pub timestamp: i64,
    pub analyst_result: Option<serde_json::Value>,
}

pub struct PipelineManager {
    allocation: Arc<AllocationEngine>,
    store: Arc<dyn TileStore>,
    alert_endpoint: Option<String>,
    active_tasks: Arc<RwLock<u32>>,
}

impl PipelineManager {
    pub fn new(
        allocation: Arc<AllocationEngine>,
        store: Arc<dyn TileStore>,
        alert_endpoint: Option<String>,
    ) -> Self {
        Self {
            allocation,
            store,
            alert_endpoint,
            active_tasks: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn dispatch(&self, req: TaskRequest) -> Result<PassResult> {
        *self.active_tasks.write().await += 1;
        let result = self.route(&req).await;
        *self.active_tasks.write().await -= 1;
        result
    }

    async fn route(&self, req: &TaskRequest) -> Result<PassResult> {
        let role = self.allocation.determine_role().await;

        match req.task_type {
            TaskType::ScoutPass => {
                if !matches!(role, NodeRole::Scout | NodeRole::CoderReasoningLead | NodeRole::CoderExecutionWorker | NodeRole::SuperAgent | NodeRole::Analyst) {
                    warn!("Scout pass requested but this node has role {:?} — proceeding anyway (CPU fallback)", role);
                }
                self.pass_scout(req).await
            }
            TaskType::SyntheticTiling => self.pass_tiling(req).await,
            TaskType::AnalystPass => {
                if !matches!(role, NodeRole::Analyst | NodeRole::SuperAgent) {
                    warn!("Analyst pass requested on non-analyst node (role={:?}). Will attempt but performance may degrade.", role);
                }
                self.pass_analyst(req).await
            }
            TaskType::TemporalStacking => self.pass_stitch(req).await,
            TaskType::CodeGeneration => self.pass_codegen(req, &role).await,
        }
    }

    /// Pass 1 — Scout: glint / hydrocarbon / heat-cold sink detection
    async fn pass_scout(&self, req: &TaskRequest) -> Result<PassResult> {
        let start = std::time::Instant::now();
        info!("Pass 1 (Scout) — task {}", req.id);

        let payload = &req.payload;
        let lat = payload["lat"].as_f64().unwrap_or(0.0);
        let lon = payload["lon"].as_f64().unwrap_or(0.0);
        let bands: Vec<f32> = payload["bands"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .unwrap_or_default();

        let glint = detect_glint(&bands);
        let hydrocarbon = detect_hydrocarbon(&bands);
        let thermal = detect_thermal_anomaly(&bands);

        let max_confidence = glint.max(hydrocarbon).max(thermal);
        let tile_id = format!("scout_{:.4}_{:.4}_{}", lat, lon, Utc::now().timestamp());

        let tile = SyntheticTile::new(tile_id.clone(), lat, lon, 30.0, Utc::now().timestamp());
        let _ = self.store.store(&tile);

        Ok(PassResult {
            task_id: req.id.clone(),
            pass: "scout".into(),
            success: true,
            anomaly_confidence: max_confidence,
            output: serde_json::json!({
                "tile_id": tile_id,
                "glint": glint,
                "hydrocarbon": hydrocarbon,
                "thermal": thermal,
                "lat": lat,
                "lon": lon,
            }),
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Pass 2 — Synthetic Tiling: decouple from UTM, convert ROIs to 16-square density tiles
    async fn pass_tiling(&self, req: &TaskRequest) -> Result<PassResult> {
        let start = std::time::Instant::now();
        info!("Pass 2 (SyntheticTiling) — task {}", req.id);

        let payload = &req.payload;
        let lat = payload["lat"].as_f64().unwrap_or(0.0);
        let lon = payload["lon"].as_f64().unwrap_or(0.0);
        let resolution = payload["resolution_meters"].as_f64().unwrap_or(30.0);

        let mut tile = SyntheticTile::new(
            req.id.clone(),
            lat, lon,
            resolution,
            Utc::now().timestamp(),
        );

        // Apply synthetic density values from the incoming band data
        if let Some(arr) = payload["band_values"].as_array() {
            for (i, v) in arr.iter().enumerate().take(16) {
                tile.update_square(i, v.as_f64().unwrap_or(0.0) as f32);
            }
        }

        let _ = self.store.store(&tile);

        Ok(PassResult {
            task_id: req.id.clone(),
            pass: "synthetic_tiling".into(),
            success: true,
            anomaly_confidence: 0.0,
            output: serde_json::to_value(&tile)?,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Pass 3 — Analyst: curvelet filtering, spectral analysis, bathymetry (FP64 / high-VRAM)
    async fn pass_analyst(&self, req: &TaskRequest) -> Result<PassResult> {
        let start = std::time::Instant::now();
        info!("Pass 3 (Analyst) — task {}", req.id);

        let payload = &req.payload;
        let tile_id = payload["tile_id"].as_str().unwrap_or(&req.id);

        // Heavy math stubs — replace with actual wgpu compute shader dispatch
        let curvelet_score = curvelet_filter_stub(payload);
        let spectral_score = spectral_analysis_stub(payload);
        let bathymetry_score = bathymetry_stub(payload);

        let confidence = (curvelet_score + spectral_score + bathymetry_score) / 3.0;

        Ok(PassResult {
            task_id: req.id.clone(),
            pass: "analyst".into(),
            success: true,
            anomaly_confidence: confidence,
            output: serde_json::json!({
                "tile_id": tile_id,
                "curvelet_score": curvelet_score,
                "spectral_score": spectral_score,
                "bathymetry_score": bathymetry_score,
                "final_confidence": confidence,
            }),
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Pass 4 — Stitch: temporal stacking on synthetic grid, then re-georeference
    async fn pass_stitch(&self, req: &TaskRequest) -> Result<PassResult> {
        let start = std::time::Instant::now();
        info!("Pass 4 (Stitch) — task {}", req.id);

        let payload = &req.payload;
        let lat = payload["lat"].as_f64().unwrap_or(0.0);
        let lon = payload["lon"].as_f64().unwrap_or(0.0);
        let radius_km = payload["radius_km"].as_f64().unwrap_or(5.0);

        let historical = self.store.get_by_region(lat, lon, radius_km)?;
        info!("Stitch: found {} historical tiles for region ({:.4}, {:.4})", historical.len(), lat, lon);

        let current_id = req.id.clone();
        let mut stacked = SyntheticTile::new(
            format!("stacked_{}", current_id),
            lat, lon,
            30.0,
            Utc::now().timestamp(),
        );

        for hist_tile in &historical {
            stacked.stack_with(hist_tile);
        }

        let _ = self.store.store(&stacked);

        Ok(PassResult {
            task_id: req.id.clone(),
            pass: "stitch".into(),
            success: true,
            anomaly_confidence: 0.0,
            output: serde_json::json!({
                "stacked_tile_id": stacked.id,
                "historical_count": historical.len(),
                "grid_data": stacked.grid_data,
            }),
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Coding agent pass — routes to Reasoning Lead or Execution Worker based on VRAM
    async fn pass_codegen(&self, req: &TaskRequest, role: &NodeRole) -> Result<PassResult> {
        let start = std::time::Instant::now();
        info!("CodeGen pass — task {} — role {:?}", req.id, role);

        let mode = match role {
            NodeRole::SuperAgent => "super_agent",
            NodeRole::CoderReasoningLead => "reasoning_lead",
            NodeRole::CoderExecutionWorker => "execution_worker",
            _ => "fallback_cpu",
        };

        Ok(PassResult {
            task_id: req.id.clone(),
            pass: "codegen".into(),
            success: true,
            anomaly_confidence: 0.0,
            output: serde_json::json!({
                "mode": mode,
                "note": "Model inference delegated to Candle/llm backend — wire in via InferenceBackend trait",
            }),
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    pub async fn fire_full_pipeline(&self, lat: f64, lon: f64, bands: Vec<f32>) -> Result<Vec<PassResult>> {
        let mut results = Vec::new();
        let task_id = uuid::Uuid::new_v4().to_string();

        // Pass 1
        let scout_req = TaskRequest {
            id: task_id.clone(),
            task_type: TaskType::ScoutPass,
            payload: serde_json::json!({ "lat": lat, "lon": lon, "bands": bands }),
            required_vram_gb: 0,
            required_fp64: false,
            requires_tpu: false,
        };
        let scout_result = self.dispatch(scout_req).await?;
        let scout_confidence = scout_result.anomaly_confidence;
        let tile_id = scout_result.output["tile_id"].as_str().unwrap_or(&task_id).to_string();
        results.push(scout_result);

        if scout_confidence < 0.1 {
            info!("Scout confidence {:.2} below threshold — skipping deeper passes", scout_confidence);
            return Ok(results);
        }

        // Pass 2
        let tiling_req = TaskRequest {
            id: format!("{}_tile", task_id),
            task_type: TaskType::SyntheticTiling,
            payload: serde_json::json!({ "lat": lat, "lon": lon, "resolution_meters": 30.0, "band_values": bands }),
            required_vram_gb: 0,
            required_fp64: false,
            requires_tpu: false,
        };
        results.push(self.dispatch(tiling_req).await?);

        // Pass 3
        let analyst_req = TaskRequest {
            id: format!("{}_analyst", task_id),
            task_type: TaskType::AnalystPass,
            payload: serde_json::json!({ "tile_id": tile_id, "bands": bands }),
            required_vram_gb: 8,
            required_fp64: true,
            requires_tpu: false,
        };
        let analyst_result = self.dispatch(analyst_req).await?;
        let final_confidence = analyst_result.anomaly_confidence;
        results.push(analyst_result);

        // Pass 4
        let stitch_req = TaskRequest {
            id: format!("{}_stitch", task_id),
            task_type: TaskType::TemporalStacking,
            payload: serde_json::json!({ "lat": lat, "lon": lon, "radius_km": 5.0 }),
            required_vram_gb: 4,
            required_fp64: false,
            requires_tpu: false,
        };
        results.push(self.dispatch(stitch_req).await?);

        if final_confidence > 0.65 {
            self.send_alert(AnomalyAlert {
                tile_id,
                confidence: final_confidence,
                lat,
                lon,
                timestamp: Utc::now().timestamp(),
                analyst_result: Some(results[2].output.clone()),
            }).await;
        }

        Ok(results)
    }

    async fn send_alert(&self, alert: AnomalyAlert) {
        if let Some(endpoint) = &self.alert_endpoint {
            let client = reqwest::Client::new();
            match client.post(endpoint).json(&alert).send().await {
                Ok(_) => info!("Alert sent to {} — confidence {:.2}", endpoint, alert.confidence),
                Err(e) => warn!("Alert delivery failed: {}", e),
            }
        } else {
            // n8n not configured — log to console
            tracing::warn!(
                "ANOMALY ALERT: tile={} confidence={:.2} lat={:.4} lon={:.4}",
                alert.tile_id, alert.confidence, alert.lat, alert.lon
            );
        }
    }

    pub async fn active_task_count(&self) -> u32 {
        *self.active_tasks.read().await
    }
}

// --- Scout pass signal detectors (stub implementations) ---
// Replace with actual TPU/Vulkan compute dispatch when Coral SDK / wgpu shaders are wired in

fn detect_glint(bands: &[f32]) -> f32 {
    if bands.len() < 4 { return 0.0; }
    // NIR/SWIR reflectance ratio heuristic for sun glint
    let ratio = bands[3] / (bands[0] + 0.001);
    (ratio - 1.0).clamp(0.0, 1.0)
}

fn detect_hydrocarbon(bands: &[f32]) -> f32 {
    if bands.len() < 6 { return 0.0; }
    // SWIR band suppression signature for hydrocarbon slicks
    let swir_suppression = 1.0 - (bands[5] / (bands[2] + 0.001)).clamp(0.0, 1.0);
    swir_suppression
}

fn detect_thermal_anomaly(bands: &[f32]) -> f32 {
    if bands.is_empty() { return 0.0; }
    // Z-score of last band (thermal) relative to median
    let mean: f32 = bands.iter().sum::<f32>() / bands.len() as f32;
    let last = *bands.last().unwrap();
    ((last - mean) / (mean + 0.001)).abs().clamp(0.0, 1.0)
}

// --- Analyst pass heavy math stubs ---
// Replace with wgpu compute pipeline / Vulkan dispatch

fn curvelet_filter_stub(payload: &serde_json::Value) -> f32 {
    payload["curvelet_hint"].as_f64().unwrap_or(0.3) as f32
}

fn spectral_analysis_stub(payload: &serde_json::Value) -> f32 {
    payload["spectral_hint"].as_f64().unwrap_or(0.3) as f32
}

fn bathymetry_stub(payload: &serde_json::Value) -> f32 {
    payload["depth_hint"].as_f64().unwrap_or(0.2) as f32
}
