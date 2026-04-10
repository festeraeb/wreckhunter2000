pub mod config;
pub mod stac;
pub mod weather;
pub mod glos;
pub mod orbits;
pub mod validation;
pub mod bridge;
pub mod detect;
pub mod kml;

pub use config::{DetectionMethod, Lake, LakeConfig, SeasonWindow, WeatherWindow, LAKES};
pub use detect::DetectionResult;
