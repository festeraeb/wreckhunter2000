use std::fs;
use std::path::Path;
use anyhow::Result;

use nautivecs::{Config, NautivecsEngine, InjectedContextBuilder, ContextFragment};

/// Helper: create a mock Rust project for the AST chunker to parse.
fn create_mock_rust_project(dir_path: &Path) -> Result<()> {
    let src_dir = dir_path.join("src");
    fs::create_dir_all(&src_dir)?;

    // Navigation module with struct + impl (clean single-brace syntax)
    fs::write(
        src_dir.join("navigation.rs"),
        r#"/// Precision geographical position for scan targeting.
pub struct SailCoordinates {
    pub lat: f64,
    pub lon: f64,
}

impl SailCoordinates {
    /// Calculate heading vector toward a target location.
    pub fn calculate_heading(&self, target_lat: f64, target_lon: f64) -> f64 {
        let dlat = target_lat - self.lat;
        let dlon = target_lon - self.lon;
        dlon.atan2(dlat).to_degrees()
    }
}
"#
    )?;

    // Ops module with async function + GPU function
    fs::write(
        src_dir.join("ops.rs"),
        r#"/// Deploy an isolated worker container onto the cluster.
pub async fn deploy_specialist_node(container_id: &str) -> bool {
    let cluster_ip = "100.72.182.77";
    println!("Initializing dipole scan GPU orchestration at {}", cluster_ip);
    true
}

/// Run a parallel dipole scan across the P100 GPU cluster.
pub fn dipole_scan_gpu(grid: &[f32], width: u32, height: u32) -> Vec<f32> {
    grid.iter().map(|v| v * 2.0).collect()
}
"#
    )?;

    Ok(())
}

#[tokio::test]
async fn test_full_pipeline_index_query_format() -> Result<()> {
    // Create temp directory for test
    let tmp_dir = std::env::temp_dir().join("nautivecs_test_integration");
    let _ = fs::remove_dir_all(&tmp_dir); // clean previous runs
    fs::create_dir_all(&tmp_dir)?;

    create_mock_rust_project(&tmp_dir)?;

    let db_path = tmp_dir.join("test_store.json");

    // Configure with fallback embeddings (no real endpoint needed for test)
    let config = Config::builder()
        .embedding_endpoint("http://localhost:99999/v1") // intentionally unreachable — triggers fallback
        .db_path(db_path.to_str().unwrap())
        .vector_dimensions(768)
        .build();

    let mut engine = NautivecsEngine::init(config).await?;

    // Step 1: Index the mock project
    let indexed = engine.index_directory(&tmp_dir.join("src")).await?;
    println!("Indexed {} chunks", indexed);
    assert!(indexed >= 3, "Expected at least 3 chunks (struct + impl + 2 functions), got {}", indexed);

    // Step 2: Keyword search (no embeddings needed)
    let keyword_results = engine.query_keyword("dipole scan GPU", 3);
    assert!(!keyword_results.is_empty(), "Keyword search should find 'dipole_scan_gpu'");
    assert!(
        keyword_results.iter().any(|r| r.function_name.contains("dipole_scan_gpu")),
        "Should find the dipole_scan_gpu function by keyword"
    );

    // Step 3: Full hybrid query (uses fallback embeddings)
    let hybrid_results = engine.query("deploy worker container cluster", 3).await?;
    assert!(!hybrid_results.is_empty(), "Hybrid search should return results");

    // Step 4: Format results through the InjectedContextBuilder
    let fragments: Vec<ContextFragment> = hybrid_results.iter()
        .map(ContextFragment::from)
        .collect();

    let builder = InjectedContextBuilder::new(4096, true);
    let payload = builder.build_system_context(&fragments);

    // Assertions on the formatted output
    assert!(payload.contains("### INJECTED CODEBASE CONTEXT"));
    assert!(payload.contains("FILE:"));
    assert!(payload.contains("```rust"));

    // Cleanup
    let _ = fs::remove_dir_all(&tmp_dir);

    Ok(())
}

#[test]
fn test_context_builder_empty_input() {
    let builder = InjectedContextBuilder::default();
    let payload = builder.build_system_context(&[]);
    assert!(payload.contains("No relevant codebase context"));
}

#[test]
fn test_context_builder_truncation() {
    let builder = InjectedContextBuilder::new(10, true); // tiny budget

    let fragments = vec![ContextFragment {
        text: "fn big() { let x = 1; let y = 2; let z = 3; }".to_string(),
        file_path: "src/big.rs".to_string(),
        function_name: "big".to_string(),
        line_start: 1,
        line_end: 5,
        search_score: 0.9,
    }];

    let payload = builder.build_system_context(&fragments);
    assert!(payload.contains("[Context allocation limit reached"));
}
