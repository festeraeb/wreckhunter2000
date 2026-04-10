use serde::{Deserialize, Serialize};

// ── Detection methods ────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetectionMethod {
    DarkSpot,     // SAR VV – surface-tension / current disruption
    ClearHole,    // Optical – mussel-filtration clarity anomaly
    SedimentTrap, // Optical – post-storm turbidity plume anchor
}

impl DetectionMethod {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DarkSpot => "dark_spot",
            Self::ClearHole => "clear_hole",
            Self::SedimentTrap => "sediment_trap",
        }
    }

    pub fn sentinel(&self) -> &'static str {
        match self {
            Self::DarkSpot => "sentinel-1-grd",
            _ => "sentinel-2-l2a",
        }
    }

    pub fn bands(&self) -> &'static [&'static str] {
        match self {
            Self::DarkSpot => &["vv"],
            Self::ClearHole => &["blue", "green", "red", "nir"],
            Self::SedimentTrap => &["green", "red"],
        }
    }
}

// ── Weather windows ──────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeatherWindow {
    /// Wind dropping from ≥15 mph to ≤5 mph within 6 hours.
    /// Residual current remains but wind-chop dies → wreck shadow pops.
    TransitionDay,
    /// Sustained winds < 5 mph, no precip, July–October.
    /// Mussel-cleared water stays localized over wreck.
    CalmSummer,
    /// First clear-sky day after ≥24 h of N/NW winds ≥ 20 mph.
    /// Suspended silt maximum, wreck acts as sediment fence.
    PostStorm,
}

impl WeatherWindow {
    pub fn for_method(m: DetectionMethod) -> Self {
        match m {
            DetectionMethod::DarkSpot => Self::TransitionDay,
            DetectionMethod::ClearHole => Self::CalmSummer,
            DetectionMethod::SedimentTrap => Self::PostStorm,
        }
    }
}

// ── Season windows ───────────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonWindow {
    pub method: DetectionMethod,
    pub months: Vec<u32>,        // 1-indexed months
    pub description: String,
}

// ── Great Lakes ──────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lake {
    Erie,
    Huron,
    Michigan,
    Superior,
    Ontario,
}

impl Lake {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Erie => "erie",
            Self::Huron => "huron",
            Self::Michigan => "michigan",
            Self::Superior => "superior",
            Self::Ontario => "ontario",
        }
    }
}

impl std::fmt::Display for Lake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for Lake {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "erie" => Ok(Self::Erie),
            "huron" => Ok(Self::Huron),
            "michigan" => Ok(Self::Michigan),
            "superior" => Ok(Self::Superior),
            "ontario" => Ok(Self::Ontario),
            _ => Err(anyhow::anyhow!("unknown lake: {s}")),
        }
    }
}

// ── Sentinel-1 orbit track ──────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitTrack {
    pub relative_orbit: u32,
    pub direction: OrbitDirection,
    /// Approximate incidence angle at lake center (degrees)
    pub center_incidence_deg: f64,
    /// Whether this track gives "far range" (shallow angle, better shadows)
    pub is_far_range: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrbitDirection {
    Ascending,  // S→N, radar looks west
    Descending, // N→S, radar looks east
}

// ── NOAA buoy ────────────────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuoyStation {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
}

// ── Lake configuration ───────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LakeConfig {
    pub lake: Lake,
    pub bounds: [f64; 4], // [west, south, east, north]
    pub avg_depth_m: f32,
    pub max_depth_m: f32,
    pub buoys: Vec<BuoyStation>,
    pub s1_tracks: Vec<OrbitTrack>,
    pub current_features: Vec<String>,
    pub seasons: Vec<SeasonWindow>,
}

impl LakeConfig {
    pub fn best_tracks_for_shadows(&self) -> Vec<&OrbitTrack> {
        self.s1_tracks.iter().filter(|t| t.is_far_range).collect()
    }
}

// ── Hardcoded Great Lakes data ───────────────────────────────────────
// This is fixed geographic / hydrodynamic data that doesn't change.

pub static LAKES: once_cell::sync::Lazy<Vec<LakeConfig>> =
    once_cell::sync::Lazy::new(build_lake_configs);

fn build_lake_configs() -> Vec<LakeConfig> {
    vec![
        // ── LAKE ERIE ────────────────────────────────────────────
        LakeConfig {
            lake: Lake::Erie,
            bounds: [-83.50, 41.35, -78.80, 42.90],
            avg_depth_m: 19.0,
            max_depth_m: 64.0,
            buoys: vec![
                BuoyStation { id: "45005".into(), name: "W Erie".into(), lat: 41.677, lon: -82.398 },
                BuoyStation { id: "45132".into(), name: "C Erie (Canadian)".into(), lat: 42.460, lon: -81.220 },
                BuoyStation { id: "45142".into(), name: "E Erie (Canadian)".into(), lat: 42.730, lon: -79.300 },
                BuoyStation { id: "45164".into(), name: "Cleveland".into(), lat: 41.690, lon: -81.740 },
            ],
            s1_tracks: vec![
                OrbitTrack { relative_orbit: 20,  direction: OrbitDirection::Descending, center_incidence_deg: 39.0, is_far_range: true },
                OrbitTrack { relative_orbit: 122, direction: OrbitDirection::Descending, center_incidence_deg: 33.0, is_far_range: false },
                OrbitTrack { relative_orbit: 49,  direction: OrbitDirection::Ascending,  center_incidence_deg: 41.0, is_far_range: true },
                OrbitTrack { relative_orbit: 151, direction: OrbitDirection::Ascending,  center_incidence_deg: 35.0, is_far_range: false },
            ],
            current_features: vec![
                "seiche (14.2h period, up to 5m amplitude at Buffalo)".into(),
                "Niagara outflow (eastward ~0.05 m/s mean)".into(),
                "Detroit River inflow (western basin)".into(),
                "thermocline-driven upwelling (central basin, summer)".into(),
                "wind-driven Ekman transport (all basins)".into(),
            ],
            seasons: vec![
                SeasonWindow {
                    method: DetectionMethod::DarkSpot,
                    months: vec![3, 4, 5, 9, 10, 11],
                    description: "Spring/fall transition days – thermal mixing + variable winds".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::ClearHole,
                    months: vec![7, 8, 9, 10],
                    description: "Peak mussel filtering, calm summer days, pre-turnover".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::SedimentTrap,
                    months: vec![4, 5, 10, 11],
                    description: "Storm season – silt mobilization from shallow western basin".into(),
                },
            ],
        },

        // ── LAKE HURON ───────────────────────────────────────────
        LakeConfig {
            lake: Lake::Huron,
            bounds: [-84.80, 43.00, -79.70, 46.30],
            avg_depth_m: 59.0,
            max_depth_m: 229.0,
            buoys: vec![
                BuoyStation { id: "45003".into(), name: "N Huron".into(), lat: 45.350, lon: -82.840 },
                BuoyStation { id: "45149".into(), name: "S Huron (Canadian)".into(), lat: 43.800, lon: -82.010 },
                BuoyStation { id: "45008".into(), name: "S Huron".into(), lat: 44.283, lon: -82.417 },
            ],
            s1_tracks: vec![
                OrbitTrack { relative_orbit: 20,  direction: OrbitDirection::Descending, center_incidence_deg: 38.0, is_far_range: true },
                OrbitTrack { relative_orbit: 122, direction: OrbitDirection::Descending, center_incidence_deg: 34.0, is_far_range: false },
                OrbitTrack { relative_orbit: 49,  direction: OrbitDirection::Ascending,  center_incidence_deg: 42.0, is_far_range: true },
            ],
            current_features: vec![
                "St. Marys River discharge (0.06 m/s from Superior)".into(),
                "Straits of Mackinac exchange (bidirectional, wind-driven)".into(),
                "St. Clair River outflow (southward to Port Huron)".into(),
                "Georgian Bay gyre (counterclockwise)".into(),
                "sag-through current (main basin, southward mean flow)".into(),
            ],
            seasons: vec![
                SeasonWindow {
                    method: DetectionMethod::DarkSpot,
                    months: vec![3, 4, 5, 9, 10, 11],
                    description: "Post-ice transition, fall storms – strong residual currents".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::ClearHole,
                    months: vec![7, 8, 9],
                    description: "Dreissenid filtering peak, calm Georgian Bay conditions".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::SedimentTrap,
                    months: vec![4, 5, 10, 11],
                    description: "Clay-bottom resuspension events, Saginaw Bay plumes".into(),
                },
            ],
        },

        // ── LAKE MICHIGAN ────────────────────────────────────────
        LakeConfig {
            lake: Lake::Michigan,
            bounds: [-88.00, 41.60, -84.80, 46.10],
            avg_depth_m: 85.0,
            max_depth_m: 281.0,
            buoys: vec![
                BuoyStation { id: "45002".into(), name: "N Michigan".into(), lat: 45.344, lon: -86.411 },
                BuoyStation { id: "45007".into(), name: "S Michigan".into(), lat: 42.674, lon: -87.026 },
                BuoyStation { id: "45026".into(), name: "C Michigan (Canadian)".into(), lat: 43.800, lon: -86.500 },
            ],
            s1_tracks: vec![
                OrbitTrack { relative_orbit: 122, direction: OrbitDirection::Descending, center_incidence_deg: 36.0, is_far_range: false },
                OrbitTrack { relative_orbit: 49,  direction: OrbitDirection::Ascending,  center_incidence_deg: 40.0, is_far_range: true },
            ],
            current_features: vec![
                "Straits of Mackinac exchange (shared with Huron)".into(),
                "Chicago sanitary canal outflow (negligible volume)".into(),
                "two-gyre circulation (summer stratified)".into(),
                "coastal upwelling (eastern shore, summer)".into(),
            ],
            seasons: vec![
                SeasonWindow {
                    method: DetectionMethod::DarkSpot,
                    months: vec![4, 5, 9, 10, 11],
                    description: "Transition season – thermal bar migration".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::ClearHole,
                    months: vec![7, 8, 9],
                    description: "Quagga mussel deep-water filtering".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::SedimentTrap,
                    months: vec![3, 4, 11],
                    description: "Spring runoff + November gales".into(),
                },
            ],
        },

        // ── LAKE SUPERIOR ────────────────────────────────────────
        LakeConfig {
            lake: Lake::Superior,
            bounds: [-92.20, 46.40, -84.30, 49.00],
            avg_depth_m: 149.0,
            max_depth_m: 406.0,
            buoys: vec![
                BuoyStation { id: "45001".into(), name: "Mid Superior".into(), lat: 48.064, lon: -87.779 },
                BuoyStation { id: "45006".into(), name: "W Superior".into(), lat: 47.336, lon: -89.794 },
                BuoyStation { id: "45004".into(), name: "E Superior".into(), lat: 47.583, lon: -86.583 },
            ],
            s1_tracks: vec![
                OrbitTrack { relative_orbit: 20,  direction: OrbitDirection::Descending, center_incidence_deg: 37.0, is_far_range: true },
                OrbitTrack { relative_orbit: 122, direction: OrbitDirection::Descending, center_incidence_deg: 32.0, is_far_range: false },
            ],
            current_features: vec![
                "counterclockwise gyre (year-round)".into(),
                "St. Marys River outflow to Huron".into(),
                "minimal invasive mussels (cold + oligotrophic)".into(),
                "clay resuspension (south shore, winter storms)".into(),
            ],
            seasons: vec![
                SeasonWindow {
                    method: DetectionMethod::DarkSpot,
                    months: vec![5, 6, 9, 10],
                    description: "Late spring calm after ice-out, early fall gales".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::SedimentTrap,
                    months: vec![4, 5, 11],
                    description: "Spring runoff plumes + November witch storms".into(),
                },
                // ClearHole not viable: too few mussels in Superior
            ],
        },

        // ── LAKE ONTARIO ─────────────────────────────────────────
        LakeConfig {
            lake: Lake::Ontario,
            bounds: [-79.90, 43.15, -76.00, 44.30],
            avg_depth_m: 86.0,
            max_depth_m: 244.0,
            buoys: vec![
                BuoyStation { id: "45012".into(), name: "E Ontario".into(), lat: 43.623, lon: -77.408 },
                BuoyStation { id: "45135".into(), name: "W Ontario (Canadian)".into(), lat: 43.800, lon: -78.980 },
            ],
            s1_tracks: vec![
                OrbitTrack { relative_orbit: 122, direction: OrbitDirection::Descending, center_incidence_deg: 35.0, is_far_range: false },
                OrbitTrack { relative_orbit: 49,  direction: OrbitDirection::Ascending,  center_incidence_deg: 39.0, is_far_range: true },
            ],
            current_features: vec![
                "Niagara River inflow (western end)".into(),
                "St. Lawrence outflow (eastern end)".into(),
                "dreissenid mussel colonies (moderate density)".into(),
            ],
            seasons: vec![
                SeasonWindow {
                    method: DetectionMethod::DarkSpot,
                    months: vec![4, 5, 9, 10, 11],
                    description: "Spring/fall transition winds".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::ClearHole,
                    months: vec![7, 8, 9],
                    description: "Summer calm, moderate mussel filtering".into(),
                },
                SeasonWindow {
                    method: DetectionMethod::SedimentTrap,
                    months: vec![4, 5, 10, 11],
                    description: "Spring snowmelt plumes, fall storms".into(),
                },
            ],
        },
    ]
}

pub fn lake_config(lake: Lake) -> &'static LakeConfig {
    LAKES
        .iter()
        .find(|c| c.lake == lake)
        .expect("all lakes configured")
}
