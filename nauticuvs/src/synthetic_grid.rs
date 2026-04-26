use serde::{Deserialize, Serialize};

/// Represents a 16-square sub-pixel grid synthetic tile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticTile {
    pub id: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub resolution_meters: f64,
    /// 4x4 grid (16 squares) representing the sub-pixel density
    pub grid_data: Vec<f32>, 
    pub timestamp: i64,
}

impl SyntheticTile {
    pub fn new(id: String, center_lat: f64, center_lon: f64, resolution_meters: f64, timestamp: i64) -> Self {
        Self {
            id,
            center_lat,
            center_lon,
            resolution_meters,
            grid_data: vec![0.0; 16], // 4x4 grid initialized to 0
            timestamp,
        }
    }

    /// Update a specific sub-square (0-15)
    pub fn update_square(&mut self, index: usize, value: f32) {
        if index < 16 {
            self.grid_data[index] = value;
        }
    }

    /// Temporal stack with another tile (weighted average, newer = higher weight).
    /// All stacking is done in synthetic grid space — no UTM projection, no zone-jump drift.
    pub fn stack_with(&mut self, other: &SyntheticTile) {
        let weight_self = if self.timestamp >= other.timestamp { 0.6 } else { 0.4 };
        let weight_other = 1.0 - weight_self;
        for i in 0..16 {
            self.grid_data[i] = self.grid_data[i] * weight_self + other.grid_data[i] * weight_other;
        }
    }

    /// Stack multiple historical tiles. Earlier timestamps contribute less.
    pub fn stack_history(&mut self, history: &[SyntheticTile]) {
        if history.is_empty() {
            return;
        }
        let n = history.len() as f32;
        for tile in history {
            // Recency weight: tiles closer in time to self get more weight
            let age_s = (self.timestamp - tile.timestamp).unsigned_abs() as f32;
            let recency = (-age_s / 86400.0).exp(); // exponential decay over days
            for i in 0..16 {
                self.grid_data[i] = self.grid_data[i] * (1.0 - recency / n)
                    + tile.grid_data[i] * (recency / n);
            }
        }
    }

    /// Convert this tile's center back to a geographic coordinate after stacking.
    /// The lat/lon fields are preserved from the original ROI — this method is a
    /// no-op sentinel confirming we never altered the coordinate during stacking.
    pub fn georef_check(&self) -> (f64, f64) {
        (self.center_lat, self.center_lon)
    }

    /// Compute per-square anomaly scores relative to a baseline tile.
    /// Returns a 16-element vector of delta magnitudes (0.0–1.0).
    pub fn anomaly_delta(&self, baseline: &SyntheticTile) -> Vec<f32> {
        self.grid_data.iter().zip(baseline.grid_data.iter())
            .map(|(a, b)| (a - b).abs().clamp(0.0, 1.0))
            .collect()
    }

    /// Return the index (0–15) and value of the highest-anomaly sub-square.
    pub fn peak_anomaly(&self, baseline: &SyntheticTile) -> (usize, f32) {
        let deltas = self.anomaly_delta(baseline);
        deltas.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0))
    }
}
