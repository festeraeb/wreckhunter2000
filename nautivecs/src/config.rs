//! Configuration for the nautivecs engine.

use std::path::PathBuf;

/// Configuration for the NautivecsEngine.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the LanceDB storage directory
    pub db_path: PathBuf,
    /// Embedding endpoint (OpenAI-compatible /v1/embeddings)
    pub embedding_endpoint: String,
    /// Embedding model name (sent in the API request)
    pub embedding_model: String,
    /// Vector dimensions (must match the embedding model output)
    pub vector_dimensions: usize,
    /// Maximum chunk size in tokens (for fallback text splitting)
    pub max_chunk_tokens: usize,
    /// File extensions to index
    pub index_extensions: Vec<String>,
    /// Directories to skip during indexing
    pub skip_dirs: Vec<String>,
}

impl Config {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data/nautivecs.lance"),
            embedding_endpoint: "http://100.72.182.77:5001/v1/embeddings".to_string(),
            embedding_model: "default".to_string(),
            vector_dimensions: 768,
            max_chunk_tokens: 512,
            index_extensions: vec![
                "rs".into(), "py".into(), "ts".into(), "tsx".into(),
                "wgsl".into(), "toml".into(), "md".into(),
            ],
            skip_dirs: vec![
                "target".into(), "node_modules".into(), ".git".into(),
                ".venv".into(), "dist".into(),
            ],
        }
    }
}

/// Builder pattern for Config.
#[derive(Default)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.db_path = path.into();
        self
    }

    pub fn embedding_endpoint(mut self, url: impl Into<String>) -> Self {
        self.config.embedding_endpoint = url.into();
        self
    }

    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.config.embedding_model = model.into();
        self
    }

    pub fn vector_dimensions(mut self, dims: usize) -> Self {
        self.config.vector_dimensions = dims;
        self
    }

    pub fn max_chunk_tokens(mut self, tokens: usize) -> Self {
        self.config.max_chunk_tokens = tokens;
        self
    }

    pub fn index_extension(mut self, ext: impl Into<String>) -> Self {
        self.config.index_extensions.push(ext.into());
        self
    }

    pub fn skip_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.skip_dirs.push(dir.into());
        self
    }

    pub fn build(self) -> Config {
        self.config
    }
}
