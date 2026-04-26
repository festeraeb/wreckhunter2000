/// Split-Agent Coder
/// A standalone reasoning+coding model team orchestrator that is hardware-aware.
///
/// Modes:
///   auto     — detect local VRAM, pick role automatically
///   full     — run all three stages locally (plan → code → review)
///   reason   — run only the Planning stage, output a Technical Brief (JSON)
///   code     — receive a Technical Brief, run only the Coding stage
///   daemon   — run as an HTTP server, accept briefs and return code responses
///
/// Distributed usage:
///   Node A (8GB reasoning lead):   split-agent --mode reason --task "..." | curl -X POST http://node-b:8766/brief -d @-
///   Node B (6GB execution worker): split-agent --mode daemon --port 8766
///   Single node (24GB+):           split-agent --mode auto --task "..."

use anyhow::{anyhow, Context, Result};
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use clap::{Parser, ValueEnum};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "split-agent", about = "Hardware-aware split reasoning+coding agent")]
struct Args {
    /// Task description (inline text)
    #[arg(long)]
    task: Option<String>,

    /// Task from file
    #[arg(long)]
    task_file: Option<PathBuf>,

    /// Technical brief JSON file (used in --mode code)
    #[arg(long)]
    brief_file: Option<PathBuf>,

    /// Operating mode
    #[arg(long, value_enum, default_value = "auto")]
    mode: RunMode,

    /// Port for daemon mode
    #[arg(long, default_value_t = 8766)]
    port: u16,

    /// Override reasoning model URL (default: REASONING_BASE_URL or 127.0.0.1:5001/v1)
    #[arg(long)]
    reasoning_url: Option<String>,

    /// Override coding model URL (default: CODING_BASE_URL or 127.0.0.1:5002/v1)
    #[arg(long)]
    coding_url: Option<String>,

    /// Remote reasoning-lead node URL for distributed dispatch
    #[arg(long)]
    remote_reasoning: Option<String>,

    /// Remote execution-worker node URL for distributed dispatch
    #[arg(long)]
    remote_execution: Option<String>,

    /// Write output to this file (JSON)
    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 0.2)]
    reasoning_temperature: f32,

    #[arg(long, default_value_t = 0.1)]
    coding_temperature: f32,

    #[arg(long, default_value_t = 600)]
    reasoning_max_tokens: u32,

    #[arg(long, default_value_t = 1600)]
    coding_max_tokens: u32,
}

#[derive(Debug, Clone, ValueEnum)]
enum RunMode {
    /// Detect VRAM and pick role automatically
    Auto,
    /// Run all three stages locally (plan → code → review)
    Full,
    /// Planning stage only — outputs a TechnicalBrief
    Reason,
    /// Coding stage only — reads a TechnicalBrief, outputs code
    Code,
    /// HTTP daemon — accepts briefs and returns code responses
    Daemon,
}

// ── Hardware detection ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCaps {
    pub node_id: String,
    pub total_vram_gb: u32,
    pub has_fp64: bool,
    pub gpu_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    SuperAgent,          // 24GB+ — reasoning + coding + review in one process
    ReasoningLead,       // 8-23GB — planning only, delegates coding to worker
    ExecutionWorker,     // 6-7GB — coding only, receives briefs
    FallbackCpu,         // no GPU — full pipeline via API only
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentRole::SuperAgent => write!(f, "super_agent"),
            AgentRole::ReasoningLead => write!(f, "reasoning_lead"),
            AgentRole::ExecutionWorker => write!(f, "execution_worker"),
            AgentRole::FallbackCpu => write!(f, "fallback_cpu"),
        }
    }
}

fn detect_hardware() -> HardwareCaps {
    let node_id = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());

    if let Some(caps) = probe_nvml(&node_id) {
        return caps;
    }
    if let Some(caps) = probe_wgpu(&node_id) {
        return caps;
    }

    warn!("No GPU detected — CPU-only mode");
    HardwareCaps { node_id, total_vram_gb: 0, has_fp64: false, gpu_name: "CPU-only".into() }
}

fn probe_nvml(node_id: &str) -> Option<HardwareCaps> {
    use nvml_wrapper::Nvml;
    let nvml = Nvml::init().ok()?;
    let dev = nvml.device_by_index(0).ok()?;
    let mem = dev.memory_info().ok()?;
    let name = dev.name().ok()?;
    let total_gb = (mem.total / (1024 * 1024 * 1024)) as u32;
    let has_fp64 = ["P100", "P40", "V100", "A100", "A40", "Quadro", "Tesla"]
        .iter()
        .any(|tag| name.contains(tag));
    info!("NVML: {} | {}GB | FP64: {}", name, total_gb, has_fp64);
    Some(HardwareCaps { node_id: node_id.into(), total_vram_gb: total_gb, has_fp64, gpu_name: name })
}

fn probe_wgpu(node_id: &str) -> Option<HardwareCaps> {
    use wgpu::{Instance, PowerPreference, RequestAdapterOptions};
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).ok()?;
    let info = adapter.get_info();
    let name = info.name.clone();
    let vram_gb = estimate_vram(&name);
    warn!("NVML unavailable — wgpu fallback: {} (~{}GB estimated)", name, vram_gb);
    Some(HardwareCaps { node_id: node_id.into(), total_vram_gb: vram_gb, has_fp64: false, gpu_name: name })
}

fn estimate_vram(name: &str) -> u32 {
    let n = name.to_uppercase();
    if n.contains("P40") { 24 } else if n.contains("P100") { 16 }
    else if n.contains("1070") || n.contains("1080") { 8 }
    else if n.contains("1060") { 6 }
    else if n.contains("3090") || n.contains("4090") { 24 }
    else if n.contains("3080") { 10 } else { 4 }
}

fn determine_role(caps: &HardwareCaps) -> AgentRole {
    match caps.total_vram_gb {
        v if v >= 24 => AgentRole::SuperAgent,
        v if v >= 8 => AgentRole::ReasoningLead,
        v if v >= 6 => AgentRole::ExecutionWorker,
        _ => AgentRole::FallbackCpu,
    }
}

// ── Model configs ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ModelConfig {
    base_url: String,
    model: String,
    api_key: String,
}

fn load_configs(args: &Args) -> (ModelConfig, ModelConfig) {
    let reasoning = ModelConfig {
        base_url: args.reasoning_url.clone()
            .or_else(|| env::var("REASONING_BASE_URL").ok())
            .unwrap_or_else(|| "http://127.0.0.1:5001/v1".into()),
        model: env::var("REASONING_MODEL").unwrap_or_else(|_| "deepseek-r1-distill-qwen-7b".into()),
        api_key: env::var("REASONING_API_KEY").unwrap_or_else(|_| "not-needed".into()),
    };
    let coding = ModelConfig {
        base_url: args.coding_url.clone()
            .or_else(|| env::var("CODING_BASE_URL").ok())
            .unwrap_or_else(|| "http://127.0.0.1:5002/v1".into()),
        model: env::var("CODING_MODEL").unwrap_or_else(|_| "qwen2.5-coder-7b-instruct".into()),
        api_key: env::var("CODING_API_KEY").unwrap_or_else(|_| "not-needed".into()),
    };
    (reasoning, coding)
}

// ── Chat / inference ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatMessage { role: String, content: String }

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse { choices: Vec<ChatChoice> }

#[derive(Debug, Deserialize)]
struct ChatChoice { message: ChatMessage }

async fn call_model(
    client: &reqwest::Client,
    cfg: &ModelConfig,
    messages: &[ChatMessage],
    temp: f32,
    max_tokens: u32,
) -> Result<String> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let payload = json!({ "model": cfg.model, "messages": messages, "temperature": temp, "max_tokens": max_tokens });

    let resp = client
        .post(&url)
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, format!("Bearer {}", cfg.api_key))
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("request failed for {}", url))?;

    let status = resp.status();
    let body = resp.text().await.with_context(|| format!("failed reading body from {}", url))?;
    if !status.is_success() {
        return Err(anyhow!("model call failed [{}] at {}: {}", status, url, body));
    }
    let parsed: ChatCompletionResponse = serde_json::from_str(&body)
        .with_context(|| format!("bad response from {}: {}", url, body))?;
    Ok(parsed.choices.first().ok_or_else(|| anyhow!("no choices from {}", url))?.message.content.clone())
}

// ── Brief protocol ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TechnicalBrief {
    pub task_id: String,
    pub task: String,
    pub planning_note: String,
    pub output_lang: String,
    pub constraints: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeGenResponse {
    pub task_id: String,
    pub role: String,
    pub hardware: HardwareCaps,
    pub brief: TechnicalBrief,
    pub code_output: Option<String>,
    pub review_note: Option<String>,
}

// ── Stage implementations ─────────────────────────────────────────────────────

fn planning_messages(task: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a senior architect and reasoning lead. \
                      Produce a precise implementation plan: goals, proposed files, \
                      key constraints, and test strategy. Be concise and actionable.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!("Task:\n{}\n\nReturn the implementation plan.", task),
        },
    ]
}

fn coding_messages(brief: &TechnicalBrief) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: format!(
                "You are an expert {} coder. \
                 Implement the code described in the brief. \
                 Output only compilable code with minimal comments. \
                 Constraints: {}.",
                brief.output_lang,
                if brief.constraints.is_empty() { "none".into() } else { brief.constraints.join(", ") }
            ),
        },
        ChatMessage {
            role: "user".into(),
            content: format!(
                "Task:\n{}\n\nArchitect's plan:\n{}\n\nImplement it now.",
                brief.task, brief.planning_note
            ),
        },
    ]
}

fn review_messages(brief: &TechnicalBrief, code: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You are the reviewer. Verify the code satisfies the task and plan. \
                      Return: verdict (pass/fail), risks, and required fixes.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!(
                "Task:\n{}\n\nPlan:\n{}\n\nCode output:\n{}\n\nReview this now.",
                brief.task, brief.planning_note, code
            ),
        },
    ]
}

async fn stage_plan(client: &reqwest::Client, cfg: &ModelConfig, task: &str, args: &Args) -> Result<TechnicalBrief> {
    info!("Planning stage — reasoning model: {}", cfg.model);
    let planning_note = call_model(client, cfg, &planning_messages(task), args.reasoning_temperature, args.reasoning_max_tokens).await?;
    Ok(TechnicalBrief {
        task_id: uuid::Uuid::new_v4().to_string(),
        task: task.to_string(),
        planning_note,
        output_lang: env::var("OUTPUT_LANG").unwrap_or_else(|_| "rust".into()),
        constraints: vec![],
    })
}

async fn stage_code(client: &reqwest::Client, cfg: &ModelConfig, brief: &TechnicalBrief, args: &Args) -> Result<String> {
    info!("Coding stage — coding model: {}", cfg.model);
    call_model(client, cfg, &coding_messages(brief), args.coding_temperature, args.coding_max_tokens).await
}

async fn stage_review(client: &reqwest::Client, cfg: &ModelConfig, brief: &TechnicalBrief, code: &str, args: &Args) -> Result<String> {
    info!("Review stage — reasoning model: {}", cfg.model);
    call_model(client, cfg, &review_messages(brief, code), args.reasoning_temperature, args.reasoning_max_tokens).await
}

// ── Distributed dispatch ──────────────────────────────────────────────────────

async fn dispatch_brief_to_remote(client: &reqwest::Client, url: &str, brief: &TechnicalBrief) -> Result<CodeGenResponse> {
    let resp = client
        .post(format!("{}/brief", url.trim_end_matches('/')))
        .header(CONTENT_TYPE, "application/json")
        .json(brief)
        .send()
        .await
        .with_context(|| format!("failed to dispatch brief to {}", url))?;

    let body = resp.text().await?;
    serde_json::from_str(&body).with_context(|| format!("bad response from remote worker: {}", body))
}

// ── Daemon mode ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct DaemonState {
    client: reqwest::Client,
    coding_cfg: ModelConfig,
    hardware: HardwareCaps,
    coding_temperature: f32,
    coding_max_tokens: u32,
    reasoning_temperature: f32,
    reasoning_max_tokens: u32,
}

// We need a wrapper because ModelConfig isn't Clone-able for State without Arc,
// and axum State requires Clone. We use Arc<DaemonState>.
async fn handle_brief(
    State(state): State<Arc<DaemonState>>,
    Json(brief): Json<TechnicalBrief>,
) -> impl IntoResponse {
    let args_proxy = Args {
        task: None, task_file: None, brief_file: None,
        mode: RunMode::Code, port: 0,
        reasoning_url: None, coding_url: None,
        remote_reasoning: None, remote_execution: None,
        output: None,
        reasoning_temperature: state.reasoning_temperature,
        coding_temperature: state.coding_temperature,
        reasoning_max_tokens: state.reasoning_max_tokens,
        coding_max_tokens: state.coding_max_tokens,
    };

    match stage_code(&state.client, &state.coding_cfg, &brief, &args_proxy).await {
        Ok(code) => {
            let resp = CodeGenResponse {
                task_id: brief.task_id.clone(),
                role: AgentRole::ExecutionWorker.to_string(),
                hardware: state.hardware.clone(),
                brief,
                code_output: Some(code),
                review_note: None,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

async fn run_daemon(state: Arc<DaemonState>, port: u16) -> Result<()> {
    let app = Router::new()
        .route("/brief", post(handle_brief))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    info!("Split-agent daemon listening on http://0.0.0.0:{}", port);
    info!("POST /brief   — send TechnicalBrief JSON, receive code response");
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Task I/O ──────────────────────────────────────────────────────────────────

fn read_task(args: &Args) -> Result<String> {
    if let Some(t) = &args.task {
        let t = t.trim();
        if t.is_empty() { return Err(anyhow!("--task cannot be empty")); }
        return Ok(t.to_string());
    }
    if let Some(p) = &args.task_file {
        let content = fs::read_to_string(p).with_context(|| format!("failed reading {}", p.display()))?;
        let t = content.trim();
        if t.is_empty() { return Err(anyhow!("task file is empty")); }
        return Ok(t.to_string());
    }
    Err(anyhow!("provide --task or --task-file"))
}

fn read_brief(args: &Args) -> Result<TechnicalBrief> {
    let path = args.brief_file.as_deref().ok_or_else(|| anyhow!("--brief-file required for --mode code"))?;
    let content = fs::read_to_string(path).with_context(|| format!("failed reading brief file {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| "failed parsing brief JSON")
}

fn write_output(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?).with_context(|| format!("failed writing {}", path.display()))
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let hardware = detect_hardware();
    let detected_role = determine_role(&hardware);
    info!("Node: {} | GPU: {} | {}GB | Role: {}", hardware.node_id, hardware.gpu_name, hardware.total_vram_gb, detected_role);

    let (reasoning_cfg, coding_cfg) = load_configs(&args);
    let client = reqwest::Client::new();

    let effective_mode = match &args.mode {
        RunMode::Auto => match detected_role {
            AgentRole::SuperAgent => RunMode::Full,
            AgentRole::ReasoningLead => RunMode::Reason,
            AgentRole::ExecutionWorker => RunMode::Daemon,
            AgentRole::FallbackCpu => RunMode::Full,
        },
        other => other.clone(),
    };

    match effective_mode {
        RunMode::Full | RunMode::Auto => {
            let task = read_task(&args)?;
            info!("Mode: full pipeline ({})", detected_role);

            let brief = stage_plan(&client, &reasoning_cfg, &task, &args).await?;
            let code = stage_code(&client, &coding_cfg, &brief, &args).await?;
            let review = stage_review(&client, &reasoning_cfg, &brief, &code, &args).await?;

            let response = CodeGenResponse {
                task_id: brief.task_id.clone(),
                role: detected_role.to_string(),
                hardware,
                brief,
                code_output: Some(code),
                review_note: Some(review),
            };

            if let Some(path) = args.output.as_deref() {
                write_output(path, &response)?;
                println!("Wrote output to {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&response)?);
        }

        RunMode::Reason => {
            let task = read_task(&args)?;
            info!("Mode: reasoning lead only");

            // If a remote execution worker is specified, dispatch brief to it
            if let Some(exec_url) = &args.remote_execution {
                let brief = stage_plan(&client, &reasoning_cfg, &task, &args).await?;
                info!("Dispatching brief to execution worker at {}", exec_url);
                let response = dispatch_brief_to_remote(&client, exec_url, &brief).await?;
                if let Some(path) = args.output.as_deref() {
                    write_output(path, &response)?;
                }
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                let brief = stage_plan(&client, &reasoning_cfg, &task, &args).await?;
                if let Some(path) = args.output.as_deref() {
                    write_output(path, &brief)?;
                }
                println!("{}", serde_json::to_string_pretty(&brief)?);
            }
        }

        RunMode::Code => {
            info!("Mode: execution worker only");
            let brief = read_brief(&args)?;
            let code = stage_code(&client, &coding_cfg, &brief, &args).await?;

            let response = CodeGenResponse {
                task_id: brief.task_id.clone(),
                role: detected_role.to_string(),
                hardware,
                brief,
                code_output: Some(code),
                review_note: None,
            };
            if let Some(path) = args.output.as_deref() {
                write_output(path, &response)?;
            }
            println!("{}", serde_json::to_string_pretty(&response)?);
        }

        RunMode::Daemon => {
            info!("Mode: daemon (execution worker HTTP server)");
            let state = Arc::new(DaemonState {
                client,
                coding_cfg,
                hardware,
                coding_temperature: args.coding_temperature,
                coding_max_tokens: args.coding_max_tokens,
                reasoning_temperature: args.reasoning_temperature,
                reasoning_max_tokens: args.reasoning_max_tokens,
            });
            run_daemon(state, args.port).await?;
        }
    }

    Ok(())
}
