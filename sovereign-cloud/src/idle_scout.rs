use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::pipeline::PipelineManager;
use crate::tile_store::TileStore;

const IDLE_INTERVAL_SECS: u64 = 300; // 5-minute scout cycle
const SCOUT_BATCH_SIZE: usize = 5;
const ALERT_CONFIDENCE_THRESHOLD: f32 = 0.65;

/// Spawns the self-annealing idle scout background task.
/// During idle cycles the node autonomously scouts historical tiles,
/// escalating to the Analyst pass when confidence is high enough.
pub fn spawn(pipeline: Arc<PipelineManager>, store: Arc<dyn TileStore>) {
    tokio::spawn(async move {
        info!("IdleScout: background task started (interval={}s, batch={})", IDLE_INTERVAL_SECS, SCOUT_BATCH_SIZE);

        loop {
            sleep(Duration::from_secs(IDLE_INTERVAL_SECS)).await;

            if pipeline.active_task_count().await > 0 {
                info!("IdleScout: node is busy — skipping scout cycle");
                continue;
            }

            info!("IdleScout: starting autonomous scout cycle");

            let tiles = match store.pull_random(SCOUT_BATCH_SIZE) {
                Ok(t) => t,
                Err(e) => {
                    warn!("IdleScout: failed to pull tiles from store: {}", e);
                    continue;
                }
            };

            if tiles.is_empty() {
                info!("IdleScout: tile store empty — no historical tiles to scout");
                continue;
            }

            info!("IdleScout: scouting {} historical tiles", tiles.len());

            for tile in tiles {
                // Synthesize band data from the stored grid for the scout pass
                let bands: Vec<f32> = tile.grid_data.clone();
                if bands.is_empty() {
                    continue;
                }

                match pipeline.fire_full_pipeline(tile.center_lat, tile.center_lon, bands).await {
                    Ok(results) => {
                        let max_confidence = results.iter().map(|r| r.anomaly_confidence).fold(0.0_f32, f32::max);
                        if max_confidence >= ALERT_CONFIDENCE_THRESHOLD {
                            // Alert is fired inside fire_full_pipeline — just log here
                            info!(
                                "IdleScout: HIGH confidence anomaly on tile {} — conf={:.2}",
                                tile.id, max_confidence
                            );
                        } else {
                            info!("IdleScout: tile {} — max_conf={:.2} (below threshold)", tile.id, max_confidence);
                        }
                    }
                    Err(e) => {
                        warn!("IdleScout: pipeline error for tile {}: {}", tile.id, e);
                    }
                }
            }

            info!("IdleScout: cycle complete");
        }
    });
}
