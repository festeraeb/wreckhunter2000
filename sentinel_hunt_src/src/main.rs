//! sentinel-hunt CLI
//!
//! Satellite-based Great Lakes wreck detection using
//! Sentinel-1 SAR and Sentinel-2 optical imagery.
//!
//! Usage:
//!   sentinel-hunt search  --lake erie --method dark_spot --start 2024-06-01 --end 2024-09-30
//!   sentinel-hunt weather --lake erie --method dark_spot
//!   sentinel-hunt glos    --lake erie
//!   sentinel-hunt detect  --lake erie --method clear_hole --start 2024-07-01 --end 2024-09-30
//!   sentinel-hunt validate --lake erie --results detections.json
//!   sentinel-hunt kml     --results detections.json --output hunt.kmz
//!   sentinel-hunt scan    --lake erie --year 2024

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use clap::{Parser, Subcommand, ValueEnum};

use sentinel_hunt::config::{self, DetectionMethod, Lake, LAKES};
use sentinel_hunt::detect::DetectionPipeline;
use sentinel_hunt::glos::GlosClient;
use sentinel_hunt::kml;
use sentinel_hunt::stac::StacClient;
use sentinel_hunt::validation;
use sentinel_hunt::weather::WeatherClient;

// ── CLI definition ───────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "sentinel-hunt", about = "Great Lakes wreck detection via satellite imagery")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to wreck database (SQLite)
    #[arg(long, default_value = "db/wrecks.db", global = true)]
    db: String,

    /// Cache directory for downloaded bands
    #[arg(long, default_value = "cache", global = true)]
    cache_dir: PathBuf,

    /// Path to Python interpreter (auto-detected if omitted)
    #[arg(long, global = true)]
    python: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Search STAC for available Sentinel scenes
    Search {
        #[arg(long)]
        lake: LakeArg,
        #[arg(long)]
        method: MethodArg,
        #[arg(long)]
        start: String,
        #[arg(long)]
        end: String,
        #[arg(long, default_value = "50")]
        max_scenes: u32,
    },
    /// Check current weather conditions and windows
    Weather {
        #[arg(long)]
        lake: LakeArg,
        #[arg(long)]
        method: MethodArg,
    },
    /// Query GLOS water quality conditions
    Glos {
        #[arg(long)]
        lake: LakeArg,
        /// Search GLOS datasets by keyword
        #[arg(long)]
        search: Option<String>,
    },
    /// Run detection pipeline on satellite scenes
    Detect {
        #[arg(long)]
        lake: LakeArg,
        #[arg(long)]
        method: MethodArg,
        #[arg(long)]
        start: String,
        #[arg(long)]
        end: String,
        #[arg(long, default_value = "20")]
        max_scenes: u32,
        /// Write JSON results to file
        #[arg(long)]
        output: Option<PathBuf>,
        /// Focus bounding box: west,south,east,north (decimal degrees)
        #[arg(long, value_delimiter = ',', num_args = 4, allow_hyphen_values = true)]
        bbox: Option<Vec<f64>>,
    },
    /// Validate detections against known wreck positions
    Validate {
        #[arg(long)]
        lake: LakeArg,
        /// Path to detection results JSON
        #[arg(long)]
        results: PathBuf,
        /// Match radius in meters
        #[arg(long, default_value = "500")]
        radius_m: f64,
    },
    /// Export detections to KML/KMZ
    Kml {
        /// Path to detection results JSON
        #[arg(long)]
        results: PathBuf,
        /// Output .kml or .kmz file
        #[arg(long)]
        output: PathBuf,
    },
    /// Full scan: weather → search → detect → validate → export
    Scan {
        #[arg(long)]
        lake: LakeArg,
        #[arg(long, default_value = "2024")]
        year: i32,
        /// Output directory for results
        #[arg(long, default_value = "output")]
        outdir: PathBuf,
        /// Focus bounding box: west,south,east,north (decimal degrees)
        #[arg(long, value_delimiter = ',', num_args = 4, allow_hyphen_values = true)]
        bbox: Option<Vec<f64>>,
    },
}

// ── Argument enums (mapped to config types) ──────────────────────────

#[derive(Clone, ValueEnum)]
enum LakeArg {
    Erie,
    Huron,
    Michigan,
    Superior,
    Ontario,
}

impl LakeArg {
    fn to_lake(&self) -> Lake {
        match self {
            Self::Erie => Lake::Erie,
            Self::Huron => Lake::Huron,
            Self::Michigan => Lake::Michigan,
            Self::Superior => Lake::Superior,
            Self::Ontario => Lake::Ontario,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum MethodArg {
    DarkSpot,
    ClearHole,
    SedimentTrap,
}

impl MethodArg {
    fn to_method(&self) -> DetectionMethod {
        match self {
            Self::DarkSpot => DetectionMethod::DarkSpot,
            Self::ClearHole => DetectionMethod::ClearHole,
            Self::SedimentTrap => DetectionMethod::SedimentTrap,
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Search {
            lake,
            method,
            start,
            end,
            max_scenes,
        } => {
            cmd_search(lake.to_lake(), method.to_method(), &start, &end, max_scenes).await
        }
        Commands::Weather { lake, method } => {
            cmd_weather(lake.to_lake(), method.to_method()).await
        }
        Commands::Glos { lake, search } => {
            cmd_glos(lake.to_lake(), search).await
        }
        Commands::Detect {
            lake,
            method,
            start,
            end,
            max_scenes,
            output,
            bbox,
        } => {
            cmd_detect(
                lake.to_lake(),
                method.to_method(),
                &start,
                &end,
                max_scenes,
                output,
                &cli.cache_dir,
                &cli.db,
                cli.python.as_deref(),
                bbox,
            )
            .await
        }
        Commands::Validate {
            lake,
            results,
            radius_m,
        } => cmd_validate(lake.to_lake(), &results, radius_m, &cli.db),
        Commands::Kml { results, output } => cmd_kml(&results, &output),
        Commands::Scan {
            lake,
            year,
            outdir,
            bbox,
        } => {
            cmd_scan(
                lake.to_lake(),
                year,
                &outdir,
                &cli.cache_dir,
                &cli.db,
                cli.python.as_deref(),
                bbox,
            )
            .await
        }
    }
}

// ── Subcommand implementations ───────────────────────────────────────

async fn cmd_search(
    lake: Lake,
    method: DetectionMethod,
    start: &str,
    end: &str,
    max_scenes: u32,
) -> Result<()> {
    let lake_cfg = get_lake_config(lake)?;
    let start = parse_date(start)?;
    let end = parse_date(end)?;

    let client = StacClient::new();
    let items = client
        .search_for_method(&lake_cfg, method, start, end, max_scenes)
        .await
        .context("STAC search")?;

    println!("Found {} scenes for {} ({}):", items.len(), lake, method.label());
    for item in &items {
        println!(
            "  {} | {} | cloud={:.0}%",
            item.id,
            item.datetime,
            item.cloud_cover.unwrap_or(-1.0)
        );
    }
    Ok(())
}

async fn cmd_weather(lake: Lake, method: DetectionMethod) -> Result<()> {
    let lake_cfg = get_lake_config(lake)?;
    let client = WeatherClient::new();
    let assessment = client
        .assess_conditions(&lake_cfg, method)
        .await
        .context("weather assessment")?;

    println!("Weather Assessment for {} — {}:", lake, method.label());
    println!("  Station: {}", assessment.station_id);
    println!(
        "  Window: {}",
        assessment
            .window
            .map(|w| format!("{:?}", w))
            .unwrap_or_else(|| "none".into())
    );
    println!("  Confidence: {:.0}%", assessment.confidence * 100.0);
    println!("  Reason: {}", assessment.reason);
    println!("  Observations: {}", assessment.observations_used);

    // Also check GLOS water conditions
    let glos = GlosClient::new();
    let water = glos.assess_water_conditions(&lake_cfg, method).await?;
    println!("\nWater Conditions:");
    println!("  Suitability: {:.0}%", water.detection_suitability * 100.0);
    println!("  Turbidity elevated: {}", water.turbidity_elevated);
    if let Some(ref ca) = water.clarity_anomaly {
        println!(
            "  Clarity anomaly: {:.1}m Secchi ({:.1}σ above {:.1}m baseline)",
            ca.secchi_m, ca.sigma, ca.baseline_secchi_m
        );
    }
    println!("  Obs used: {}", water.observations_used);
    println!("  Note: {}", water.reason);

    // Combined assessment
    let combined = sentinel_hunt::glos::combine_assessments(&assessment, &water, method);
    println!("\nCombined: {:.0}% — {}", combined.combined_score * 100.0, combined.recommendation);

    Ok(())
}

async fn cmd_glos(lake: Lake, search: Option<String>) -> Result<()> {
    let glos = GlosClient::new();

    if let Some(keyword) = search {
        let datasets = glos.search_datasets(&keyword).await?;
        println!("GLOS datasets matching '{}':", keyword);
        for ds in &datasets {
            let tag = if ds.is_water_quality() {
                " [WQ]"
            } else if ds.is_currents() {
                " [HFR]"
            } else {
                ""
            };
            println!("  {}{} — {}", ds.dataset_id, tag, ds.title);
        }
    } else {
        let lake_cfg = get_lake_config(lake)?;
        let today = chrono::Utc::now().naive_utc().date();
        let week_ago = today - chrono::Duration::days(7);
        let obs = glos
            .fetch_water_quality(lake, lake_cfg.bounds, week_ago, today)
            .await?;
        println!("{} water quality observations (last 7 days): {}", lake, obs.len());
        for o in obs.iter().take(20) {
            println!(
                "  {} | {:.4},{:.4} | turb={} secchi={} chl={}",
                o.timestamp.format("%Y-%m-%d %H:%M"),
                o.lat,
                o.lon,
                o.turbidity_ntu
                    .map(|v| format!("{:.1} NTU", v))
                    .unwrap_or_else(|| "-".into()),
                o.secchi_depth_m
                    .map(|v| format!("{:.1} m", v))
                    .unwrap_or_else(|| "-".into()),
                o.chlorophyll_ug_l
                    .map(|v| format!("{:.2} µg/L", v))
                    .unwrap_or_else(|| "-".into()),
            );
        }
    }
    Ok(())
}

async fn cmd_detect(
    lake: Lake,
    method: DetectionMethod,
    start: &str,
    end: &str,
    max_scenes: u32,
    output: Option<PathBuf>,
    cache_dir: &PathBuf,
    db_path: &str,
    python: Option<&str>,
    bbox: Option<Vec<f64>>,
) -> Result<()> {
    let mut lake_cfg = get_lake_config(lake)?;
    let start = parse_date(start)?;
    let end = parse_date(end)?;

    // Override bounds with focused ROI if provided
    if let Some(ref b) = bbox {
        if b.len() == 4 {
            lake_cfg.bounds = [b[0], b[1], b[2], b[3]];
            println!("Focused ROI: [{:.4}, {:.4}, {:.4}, {:.4}]", b[0], b[1], b[2], b[3]);
        }
    }

    let bridge = make_bridge(python, cache_dir)?;
    let pipeline = DetectionPipeline::new(bridge, cache_dir.clone(), db_path.to_string());

    let results = pipeline
        .run(&lake_cfg, method, start, end, max_scenes)
        .await?;

    let total_dets: usize = results.iter().map(|r| r.detections.len()).sum();
    println!(
        "Detection complete: {} scenes, {} detections",
        results.len(),
        total_dets
    );

    if let Some(ref path) = output {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(path, json)?;
        println!("Results written to {}", path.display());
    } else {
        // Print summary
        for r in &results {
            println!(
                "  {} [{}]: {} detections",
                r.scene_id,
                r.datetime,
                r.detections.len()
            );
            for d in &r.detections {
                println!(
                    "    {:.5},{:.5} conf={:.0}% σ={:.1} class={}",
                    d.lat, d.lon, d.confidence * 100.0, d.anomaly_sigma, d.classification
                );
            }
        }
    }

    Ok(())
}

fn cmd_validate(lake: Lake, results_path: &PathBuf, radius_m: f64, db_path: &str) -> Result<()> {
    let results_json = std::fs::read_to_string(results_path)?;
    let results: Vec<sentinel_hunt::DetectionResult> = serde_json::from_str(&results_json)?;

    // Determine method from first result
    let method = results.first()
        .map(|r| r.method)
        .unwrap_or(DetectionMethod::DarkSpot);

    // Convert to (lat, lon, confidence) tuples
    let dets: Vec<(f64, f64, f64)> = results.iter()
        .flat_map(|r| r.detections.iter().map(|d| (d.lat, d.lon, d.confidence)))
        .collect();

    println!("Validating {} detections against wrecks.db...", dets.len());

    let wrecks = validation::load_known_wrecks(db_path, lake, true)?;
    let report = validation::validate_detections(&dets, &wrecks, method, lake, radius_m);
    print_validation_report(&report);

    Ok(())
}

fn cmd_kml(results_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
    let results_json = std::fs::read_to_string(results_path)?;
    let results: Vec<sentinel_hunt::DetectionResult> = serde_json::from_str(&results_json)?;

    let total: usize = results.iter().map(|r| r.detections.len()).sum();

    if output_path.extension().and_then(|e| e.to_str()) == Some("kmz") {
        kml::write_kmz(&results, output_path)?;
    } else {
        kml::write_kml(&results, output_path)?;
    }

    println!(
        "Wrote {} detections from {} scenes to {}",
        total,
        results.len(),
        output_path.display()
    );
    Ok(())
}

async fn cmd_scan(
    lake: Lake,
    year: i32,
    outdir: &PathBuf,
    cache_dir: &PathBuf,
    db_path: &str,
    python: Option<&str>,
    bbox: Option<Vec<f64>>,
) -> Result<()> {
    let mut lake_cfg = get_lake_config(lake)?;

    // Override bounds with focused ROI if provided
    if let Some(ref b) = bbox {
        if b.len() == 4 {
            lake_cfg.bounds = [b[0], b[1], b[2], b[3]];
            println!("Focused ROI: [{:.4}, {:.4}, {:.4}, {:.4}]", b[0], b[1], b[2], b[3]);
        }
    }
    std::fs::create_dir_all(outdir)?;

    let bridge = make_bridge(python, cache_dir)?;
    let pipeline = DetectionPipeline::new(bridge, cache_dir.clone(), db_path.to_string());

    let methods = [
        DetectionMethod::DarkSpot,
        DetectionMethod::ClearHole,
        DetectionMethod::SedimentTrap,
    ];

    let mut all_results = Vec::new();

    for method in &methods {
        // Use seasonal windows from config
        let seasons = &lake_cfg.seasons;
        if let Some(sw) = seasons.iter().find(|s| s.method == *method) {
            let start_month = *sw.months.first().unwrap_or(&1);
            let end_month = *sw.months.last().unwrap_or(&12);
            let start = NaiveDate::from_ymd_opt(year, start_month, 1).unwrap();
            let end_day = if end_month == 12 { 31 } else { 28 };
            let end = NaiveDate::from_ymd_opt(year, end_month, end_day).unwrap();

            println!("\n--- {} ({} → {}) ---", method.label(), start, end);

            match pipeline.run(&lake_cfg, *method, start, end, 50).await {
                Ok(results) => {
                    let n: usize = results.iter().map(|r| r.detections.len()).sum();
                    println!("{}: {} scenes, {} detections", method.label(), results.len(), n);
                    all_results.extend(results);
                }
                Err(e) => {
                    eprintln!("{} failed: {}", method.label(), e);
                }
            }
        }
    }

    // Summary
    let total_dets: usize = all_results.iter().map(|r| r.detections.len()).sum();
    println!("\n=== Scan Complete ===");
    println!("Total: {} scenes, {} detections", all_results.len(), total_dets);

    // Save JSON (full results with scene metadata)
    let json_path = outdir.join(format!("{}_{}_detections.json", lake, year));
    std::fs::write(&json_path, serde_json::to_string_pretty(&all_results)?)?;
    println!("JSON: {}", json_path.display());

    // Save KMZ
    let kmz_path = outdir.join(format!("{}_{}_detections.kmz", lake, year));
    kml::write_kmz(&all_results, &kmz_path)?;
    println!("KMZ: {}", kmz_path.display());

    // Merge + validate against known wrecks
    let merged = sentinel_hunt::detect::merge_detections(&all_results, 200.0);
    println!("Unique detections (200m dedup): {}", merged.len());

    if !merged.is_empty() {
        match validation::load_known_wrecks(db_path, lake, true) {
            Ok(wrecks) => {
                // Validate per method
                for method in &methods {
                    let method_dets: Vec<(f64, f64, f64)> = all_results.iter()
                        .filter(|r| r.method == *method)
                        .flat_map(|r| r.detections.iter().map(|d| (d.lat, d.lon, d.confidence)))
                        .collect();
                    if !method_dets.is_empty() {
                        let report = validation::validate_detections(
                            &method_dets, &wrecks, *method, lake, 500.0,
                        );
                        print_validation_report(&report);
                    }
                }
            }
            Err(e) => eprintln!("wrecks db: {}", e),
        }
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn get_lake_config(lake: Lake) -> Result<config::LakeConfig> {
    LAKES
        .iter()
        .find(|c| c.lake == lake)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no config for {}", lake))
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .with_context(|| format!("invalid date '{}', expected YYYY-MM-DD", s))
}

fn make_bridge(python: Option<&str>, cache_dir: &PathBuf) -> Result<sentinel_hunt::bridge::PythonBridge> {
    let scripts_dir = cache_dir
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("python");
    match python {
        Some(p) => Ok(sentinel_hunt::bridge::PythonBridge::new(p.into(), scripts_dir)),
        None => sentinel_hunt::bridge::PythonBridge::auto_detect(scripts_dir),
    }
}

fn print_validation_report(report: &validation::ValidationReport) {
    println!("\nValidation — {} ({:?}):", report.lake, report.method);
    println!("  Detections:    {}", report.total_detections);
    println!("  Known wrecks:  {}", report.total_known_wrecks);
    println!("  Matches:       {}", report.matches.len());
    println!("  Unmatched:     {}", report.unmatched_detections);
    println!("  Detection rate:     {:.1}%", report.detection_rate * 100.0);
    println!("  False positive rate:{:.1}%", report.false_positive_rate * 100.0);
    println!("  Mean offset:        {:.0}m", report.mean_offset_m);
    for m in &report.matches {
        println!(
            "    {} ({:.5},{:.5}) ← {:.0}m @ {:.0}° conf={:.0}%",
            m.wreck.name, m.wreck.lat, m.wreck.lon,
            m.distance_m, m.bearing_deg, m.confidence * 100.0
        );
    }
}
