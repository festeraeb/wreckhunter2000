use anyhow::Result;
use nauticuvs::synthetic_grid::SyntheticTile;
use rand::seq::SliceRandom;
use sled::Db;
use std::sync::Arc;
use tracing::{info, warn};

/// Abstraction over the vector context store (LanceDB when available; sled otherwise).
/// All pipeline passes operate through this trait so the backend can be swapped.
pub trait TileStore: Send + Sync {
    fn store(&self, tile: &SyntheticTile) -> Result<()>;
    fn pull_random(&self, n: usize) -> Result<Vec<SyntheticTile>>;
    fn get_by_region(&self, center_lat: f64, center_lon: f64, radius_km: f64) -> Result<Vec<SyntheticTile>>;
    fn list_ids(&self) -> Result<Vec<String>>;
}

/// Sled-backed implementation for local / edge deployment.
/// Compatible with the LanceDB interface at the trait level — swap by implementing TileStore.
pub struct SledTileStore {
    db: Arc<Db>,
}

impl SledTileStore {
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        info!("TileStore opened at {}", path);
        Ok(Self { db: Arc::new(db) })
    }
}

impl TileStore for SledTileStore {
    fn store(&self, tile: &SyntheticTile) -> Result<()> {
        let key = tile.id.as_bytes();
        let value = serde_json::to_vec(tile)?;
        self.db.insert(key, value)?;
        Ok(())
    }

    fn pull_random(&self, n: usize) -> Result<Vec<SyntheticTile>> {
        let all_ids: Vec<sled::IVec> = self.db.iter().keys().filter_map(|r| r.ok()).collect();
        if all_ids.is_empty() {
            warn!("TileStore is empty — idle scout has no historical tiles yet");
            return Ok(vec![]);
        }

        let mut rng = rand::thread_rng();
        let chosen: Vec<&sled::IVec> = all_ids.choose_multiple(&mut rng, n.min(all_ids.len())).collect();

        let mut tiles = Vec::with_capacity(chosen.len());
        for key in chosen {
            if let Ok(Some(bytes)) = self.db.get(key) {
                if let Ok(tile) = serde_json::from_slice::<SyntheticTile>(&bytes) {
                    tiles.push(tile);
                }
            }
        }
        Ok(tiles)
    }

    fn get_by_region(&self, center_lat: f64, center_lon: f64, radius_km: f64) -> Result<Vec<SyntheticTile>> {
        let degree_approx = radius_km / 111.0;
        let mut results = Vec::new();

        for entry in self.db.iter() {
            let (_, value) = entry?;
            if let Ok(tile) = serde_json::from_slice::<SyntheticTile>(&value) {
                let dlat = (tile.center_lat - center_lat).abs();
                let dlon = (tile.center_lon - center_lon).abs();
                if dlat <= degree_approx && dlon <= degree_approx {
                    results.push(tile);
                }
            }
        }
        Ok(results)
    }

    fn list_ids(&self) -> Result<Vec<String>> {
        let ids = self.db
            .iter()
            .keys()
            .filter_map(|k| k.ok())
            .filter_map(|k| String::from_utf8(k.to_vec()).ok())
            .collect();
        Ok(ids)
    }
}
