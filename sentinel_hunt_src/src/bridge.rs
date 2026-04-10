use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Result from a Python detection script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonResult {
    pub success: bool,
    pub detections: Vec<serde_json::Value>,
    pub metadata: serde_json::Value,
    pub error: Option<String>,
}

/// Configuration for the Python bridge.
pub struct PythonBridge {
    /// Path to python executable (e.g., conda env python).
    python_exe: PathBuf,
    /// Directory containing the Python processing scripts.
    scripts_dir: PathBuf,
    /// Timeout in seconds for script execution.
    timeout_secs: u64,
}

impl PythonBridge {
    pub fn new(python_exe: PathBuf, scripts_dir: PathBuf) -> Self {
        Self {
            python_exe,
            scripts_dir,
            timeout_secs: 600, // 10 minutes default
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Auto-detect Python from common locations.
    pub fn auto_detect(scripts_dir: PathBuf) -> Result<Self> {
        // Try conda env first, then system python
        let candidates = [
            r"C:\Users\thomf\miniconda3\envs\wh2k\python.exe",
            "python",
            "python3",
        ];

        for candidate in &candidates {
            let path = Path::new(candidate);
            if path.exists() || which_python(candidate).is_some() {
                return Ok(Self::new(
                    PathBuf::from(candidate),
                    scripts_dir,
                ));
            }
        }

        Err(anyhow::anyhow!("no Python interpreter found"))
    }

    /// Run a Python detection script with JSON input/output.
    ///
    /// Protocol:
    /// - Rust writes a JSON object to the script's stdin, then closes stdin.
    /// - Script processes and writes a JSON object to stdout.
    /// - Stderr is captured for logging.
    pub fn run_script(
        &self,
        script_name: &str,
        input: &serde_json::Value,
    ) -> Result<PythonResult> {
        let script_path = self.scripts_dir.join(script_name);
        if !script_path.exists() {
            anyhow::bail!("script not found: {}", script_path.display());
        }

        let input_json = serde_json::to_string(input)?;
        tracing::info!(
            "running {} ({} bytes input)",
            script_name,
            input_json.len()
        );

        let mut child = Command::new(&self.python_exe)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(format!("spawn {}", script_name))?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input_json.as_bytes())?;
            // stdin is dropped here, closing the pipe
        }

        // Wait with timeout
        let output = child
            .wait_with_output()
            .context(format!("wait for {}", script_name))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            tracing::warn!("{} stderr: {}", script_name, stderr);
        }

        if !output.status.success() {
            return Ok(PythonResult {
                success: false,
                detections: vec![],
                metadata: serde_json::json!({}),
                error: Some(format!(
                    "exit code {}: {}",
                    output.status.code().unwrap_or(-1),
                    stderr
                )),
            });
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: PythonResult = serde_json::from_str(&stdout)
            .context(format!(
                "parse output from {}: first 200 chars: {}",
                script_name,
                &stdout[..stdout.len().min(200)]
            ))?;

        tracing::info!(
            "{}: {} detections",
            script_name,
            result.detections.len()
        );

        Ok(result)
    }

    /// Run the band_fetch.py script to download satellite bands.
    pub fn fetch_bands(
        &self,
        scene_id: &str,
        bands: &[&str],
        asset_urls: &std::collections::HashMap<String, String>,
        output_dir: &Path,
        bbox: Option<[f64; 4]>,
    ) -> Result<PythonResult> {
        let input = serde_json::json!({
            "command": "fetch_bands",
            "scene_id": scene_id,
            "bands": bands,
            "asset_urls": asset_urls,
            "output_dir": output_dir.to_string_lossy(),
            "bbox": bbox,
        });
        self.run_script("band_fetch.py", &input)
    }

    /// Run a detection script (dark_spot, clear_hole, or sediment_trap).
    pub fn run_detector(
        &self,
        detector: &str,
        band_paths: &std::collections::HashMap<String, String>,
        roi: Option<[f64; 4]>,      // optional focus bbox
        known_wreck: Option<(f64, f64)>,  // optional known wreck position for validation
    ) -> Result<PythonResult> {
        let script = format!("{}.py", detector);
        let input = serde_json::json!({
            "command": "detect",
            "band_paths": band_paths,
            "roi": roi,
            "known_wreck": known_wreck,
        });
        self.run_script(&script, &input)
    }
}

/// Try to find python executable on PATH.
fn which_python(name: &str) -> Option<PathBuf> {
    let output = Command::new("where")
        .arg(name)
        .output()
        .ok()?;

    if output.status.success() {
        let path_str = String::from_utf8_lossy(&output.stdout);
        let first_line = path_str.lines().next()?;
        Some(PathBuf::from(first_line.trim()))
    } else {
        None
    }
}
