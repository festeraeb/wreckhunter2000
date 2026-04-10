use serde::{Deserialize, Serialize};
pub use crate::config::OrbitDirection;

/// A grouped set of STAC items from the same Sentinel-1 relative orbit + direction.
/// ML training must NEVER mix track groups or the model confuses look-angle shading
/// with real surface anomalies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackGroup {
    pub relative_orbit: u32,
    pub direction: OrbitDirection,
    pub center_incidence_deg: f64,
    pub is_far_range: bool,
    pub scene_ids: Vec<String>,
}

/// Classify a Sentinel-1 item's swath position to estimate incidence angle.
/// IW mode swath is ~250km.  Given the item bbox and the lake bounds,
/// we estimate whether the lake center falls in the near-range or far-range half.
///
/// Returns (estimated_incidence_deg, is_far_range).
pub fn classify_swath_position(
    item_bbox: &[f64; 4],
    lake_center_lon: f64,
    direction: OrbitDirection,
) -> (f64, bool) {
    // For descending passes (N→S), radar looks east:
    //   near range = eastern edge of swath, far range = western edge
    // For ascending passes (S→N), radar looks west:
    //   near range = western edge of swath, far range = eastern edge

    let swath_west = item_bbox[0];
    let swath_east = item_bbox[2];
    let swath_width = swath_east - swath_west;

    if swath_width <= 0.0 {
        return (37.0, false); // fallback
    }

    let frac = (lake_center_lon - swath_west) / swath_width; // 0 = west edge, 1 = east edge

    let near_to_far_frac = match direction {
        OrbitDirection::Descending => 1.0 - frac, // east=near, west=far
        OrbitDirection::Ascending => frac,          // west=near, east=far
    };

    // IW incidence angle range: ~29° (near) to ~46° (far)
    let incidence = 29.0 + near_to_far_frac * 17.0;
    let is_far = near_to_far_frac > 0.5;

    (incidence, is_far)
}

/// Group STAC items by relative orbit + direction.
/// Each group should be trained/analyzed independently.
pub fn group_by_track(
    items: &[(String, u32, OrbitDirection)],
) -> Vec<TrackGroup> {
    use std::collections::HashMap;

    let mut groups: HashMap<(u32, OrbitDirection), Vec<String>> = HashMap::new();
    for (id, orbit, dir) in items {
        groups.entry((*orbit, *dir)).or_default().push(id.clone());
    }

    groups
        .into_iter()
        .map(|((orbit, dir), ids)| TrackGroup {
            relative_orbit: orbit,
            direction: dir,
            center_incidence_deg: match dir {
                OrbitDirection::Ascending => 40.0,  // rough defaults
                OrbitDirection::Descending => 36.0,
            },
            is_far_range: false, // caller should refine
            scene_ids: ids,
        })
        .collect()
}

/// For a given lake center and SAR scene, determine if this scene provides
/// optimal shadow-detection geometry (far range, shallow incidence angle).
pub fn is_shadow_optimal(incidence_deg: f64) -> bool {
    // Shallow angles (>38°) give longer, more visible wreck shadows
    incidence_deg >= 38.0
}

/// Sentinel-1 revisit period is 6 days (with 1A+1B constellation).
/// Given a date, estimate the next acquisition for a specific track.
pub fn next_acquisition_estimate(
    last_seen: chrono::NaiveDate,
    revisit_days: u32,
) -> chrono::NaiveDate {
    last_seen + chrono::Duration::days(revisit_days as i64)
}
