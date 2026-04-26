use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use nauticuvs::protocol::{TaskRequest, TaskType};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::allocation::AllocationEngine;
use crate::pipeline::PipelineManager;

// --- Hot-swap mode state ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeMode {
    Llm,
    Vulkan,
}

pub struct NodeState {
    pub allocation: Arc<AllocationEngine>,
    pub pipeline: Arc<PipelineManager>,
    pub mode: RwLock<ComputeMode>,
}

impl NodeState {
    pub fn new(allocation: Arc<AllocationEngine>, pipeline: Arc<PipelineManager>) -> Arc<Self> {
        Arc::new(Self {
            allocation,
            pipeline,
            mode: RwLock::new(ComputeMode::Llm),
        })
    }

    /// Hot-swap: flush current mode and initialize the new one.
    pub async fn swap_mode(&self, new_mode: ComputeMode) {
        let mut mode = self.mode.write().await;
        if *mode == new_mode {
            return;
        }
        info!("Hot-swap: {:?} → {:?}", *mode, new_mode);
        match new_mode {
            ComputeMode::Vulkan => {
                // LLM weights flushed — Vulkan pipeline takes the VRAM
                info!("Hot-swap: flushing LLM weights, initializing Vulkan compute pipeline");
            }
            ComputeMode::Llm => {
                info!("Hot-swap: releasing Vulkan pipeline, loading LLM weights");
            }
        }
        *mode = new_mode;
    }
}

// --- OpenAI-compatible types ---

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

// --- Request / response types for direct pipeline dispatch ---

#[derive(Debug, Serialize)]
pub struct NodeStatus {
    pub node_id: String,
    pub gpu_name: String,
    pub total_vram_gb: u32,
    pub available_vram_gb: u32,
    pub has_fp64: bool,
    pub has_tpu: bool,
    pub role: String,
    pub mode: String,
    pub active_tasks: u32,
}

// --- Router ---

pub fn router(state: Arc<NodeState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/pipeline/dispatch", post(pipeline_dispatch))
        .route("/v1/pipeline/run", post(pipeline_run_full))
        .route("/v1/node/status", get(node_status))
        .route("/v1/node/mode", post(set_mode))
        .with_state(state)
}

// --- Handlers ---

/// OpenAI-compatible chat completions endpoint.
/// Routes to pipeline dispatch or LLM backend depending on message content and current mode.
async fn chat_completions(
    State(state): State<Arc<NodeState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let user_msg = req.messages.iter().rev().find(|m| m.role == "user");
    let content = user_msg.map(|m| m.content.as_str()).unwrap_or("");

    // Detect pipeline dispatch intent in the message
    let task_type = infer_task_from_message(content);

    let reply = if let Some(tt) = task_type {
        // Switch to Vulkan mode for compute tasks
        state.swap_mode(ComputeMode::Vulkan).await;

        let task_req = TaskRequest {
            id: uuid::Uuid::new_v4().to_string(),
            task_type: tt,
            payload: serde_json::json!({ "prompt": content }),
            required_vram_gb: 0,
            required_fp64: false,
            requires_tpu: false,
        };
        match state.pipeline.dispatch(task_req).await {
            Ok(result) => format!(
                "Pipeline pass '{}' completed. Confidence: {:.2}. Output: {}",
                result.pass,
                result.anomaly_confidence,
                serde_json::to_string_pretty(&result.output).unwrap_or_default()
            ),
            Err(e) => format!("Pipeline error: {}", e),
        }
    } else {
        // LLM mode — ensure weights are loaded
        state.swap_mode(ComputeMode::Llm).await;
        // Actual inference delegated to Candle/llm backend (wire in via InferenceBackend trait)
        format!("Received: '{}'. [LLM inference backend not yet connected — wire Candle/llm here]", content)
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: reply,
            },
            finish_reason: "stop".into(),
        }],
    };

    (StatusCode::OK, Json(response))
}

/// Direct pipeline task dispatch endpoint.
async fn pipeline_dispatch(
    State(state): State<Arc<NodeState>>,
    Json(req): Json<TaskRequest>,
) -> impl IntoResponse {
    state.swap_mode(ComputeMode::Vulkan).await;
    match state.pipeline.dispatch(req).await {
        Ok(result) => (StatusCode::OK, Json(serde_json::to_value(result).unwrap())),
        Err(e) => {
            warn!("Pipeline dispatch error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
        }
    }
}

/// Fire the full 4-pass pipeline over a geographic region.
#[derive(Debug, Deserialize)]
struct FullPipelineRequest {
    lat: f64,
    lon: f64,
    bands: Vec<f32>,
}

async fn pipeline_run_full(
    State(state): State<Arc<NodeState>>,
    Json(req): Json<FullPipelineRequest>,
) -> impl IntoResponse {
    state.swap_mode(ComputeMode::Vulkan).await;
    match state.pipeline.fire_full_pipeline(req.lat, req.lon, req.bands).await {
        Ok(results) => (StatusCode::OK, Json(serde_json::to_value(results).unwrap())),
        Err(e) => {
            warn!("Full pipeline error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
        }
    }
}

/// Node capabilities and current status.
async fn node_status(State(state): State<Arc<NodeState>>) -> impl IntoResponse {
    let caps = state.allocation.capabilities.read().await;
    let role = state.allocation.determine_role().await;
    let mode = state.mode.read().await;

    let status = NodeStatus {
        node_id: caps.node_id.clone(),
        gpu_name: caps.gpu_name.clone(),
        total_vram_gb: caps.total_vram_gb,
        available_vram_gb: caps.available_vram_gb,
        has_fp64: caps.has_fp64,
        has_tpu: caps.has_tpu,
        role: format!("{:?}", role),
        mode: format!("{:?}", *mode),
        active_tasks: state.pipeline.active_task_count().await,
    };

    (StatusCode::OK, Json(status))
}

#[derive(Debug, Deserialize)]
struct SetModeRequest {
    mode: String,
}

/// Manually trigger a hot-swap.
async fn set_mode(
    State(state): State<Arc<NodeState>>,
    Json(req): Json<SetModeRequest>,
) -> impl IntoResponse {
    let new_mode = match req.mode.to_lowercase().as_str() {
        "vulkan" | "compute" => ComputeMode::Vulkan,
        "llm" | "inference" => ComputeMode::Llm,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("unknown mode: {}", other) })),
            );
        }
    };
    state.swap_mode(new_mode).await;
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

// --- Helpers ---

fn infer_task_from_message(content: &str) -> Option<TaskType> {
    let lower = content.to_lowercase();
    if lower.contains("scout") || lower.contains("glint") || lower.contains("hydrocarbon") {
        Some(TaskType::ScoutPass)
    } else if lower.contains("tile") || lower.contains("grid") || lower.contains("synthetic") {
        Some(TaskType::SyntheticTiling)
    } else if lower.contains("analyst") || lower.contains("curvelet") || lower.contains("bathymetry") || lower.contains("spectral") {
        Some(TaskType::AnalystPass)
    } else if lower.contains("stitch") || lower.contains("stack") || lower.contains("temporal") {
        Some(TaskType::TemporalStacking)
    } else if lower.contains("code") || lower.contains("rust") || lower.contains("shader") || lower.contains("wgsl") {
        Some(TaskType::CodeGeneration)
    } else {
        None
    }
}
