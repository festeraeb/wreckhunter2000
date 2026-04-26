mod allocation;
mod api;
mod discovery;
mod idle_scout;
mod pipeline;
mod tile_store;

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

const API_PORT: u16 = 8765;
const TILE_STORE_PATH: &str = "./data/tiles";
/// n8n webhook URL — set via N8N_ALERT_URL env var or leave None for console alerts
const ENV_ALERT_URL: &str = "N8N_ALERT_URL";

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Init structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("=== CESARops Sovereign Cloud Node ===");

    // 2. Scan hardware — query NVML then wgpu fallback
    let allocation = Arc::new(allocation::AllocationEngine::detect().await?);
    let role = allocation.determine_role().await;
    info!("Assigned role: {:?}", role);

    // 3. Open tile store (sled-backed; swap for LanceDB by implementing TileStore trait)
    std::fs::create_dir_all(TILE_STORE_PATH).ok();
    let store: Arc<dyn tile_store::TileStore> =
        Arc::new(tile_store::SledTileStore::open(TILE_STORE_PATH)?);

    // 4. Build pipeline manager
    let alert_url = std::env::var(ENV_ALERT_URL).ok();
    if alert_url.is_some() {
        info!("Alert endpoint configured: {}", alert_url.as_deref().unwrap_or(""));
    } else {
        info!("No N8N_ALERT_URL set — anomaly alerts will print to console");
    }
    let pipeline = Arc::new(pipeline::PipelineManager::new(
        allocation.clone(),
        store.clone(),
        alert_url,
    ));

    // 5. Broadcast node capabilities via mDNS so Pi dispatcher and peers can discover us
    let discovery = discovery::NodeDiscovery::new()?;
    {
        let caps = allocation.capabilities.read().await;
        discovery.announce(&caps, API_PORT)?;
    }
    discovery.browse_peers().await?;
    info!("mDNS: node announced on port {}", API_PORT);

    // 6. Spawn self-annealing idle scout background task
    idle_scout::spawn(pipeline.clone(), store.clone());
    info!("IdleScout: background loop active");

    // 7. Start Axum API server
    let node_state = api::NodeState::new(allocation.clone(), pipeline.clone());
    let app = api::router(node_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", API_PORT)).await?;
    info!("API listening on http://0.0.0.0:{}", API_PORT);
    info!("OpenAI-compatible endpoint: POST /v1/chat/completions");
    info!("Pipeline dispatch:          POST /v1/pipeline/dispatch");
    info!("Full pipeline run:          POST /v1/pipeline/run");
    info!("Node status:               GET  /v1/node/status");

    axum::serve(listener, app).await?;

    Ok(())
}
