//! The main NautivecsEngine — ties together chunking, embedding, and storage.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::info;

use crate::chunker::{chunk_rust_file, chunk_text_file, CodeChunk};
use crate::config::Config;
use crate::embeddings::EmbeddingClient;
use crate::storage::{SearchResult, VectorStore};

/// The main engine that indexes codebases and retrieves relevant context.
pub struct NautivecsEngine {
    config: Config,
    embeddings: EmbeddingClient,
    store: VectorStore,
}

impl NautivecsEngine {
    /// Initialize the engine with the given configuration.
    pub async fn init(config: Config) -> Result<Self> {
        let embeddings = EmbeddingClient::new(
            &config.embedding_endpoint,
            &config.embedding_model,
            config.vector_dimensions,
        );

        let store = VectorStore::open(
            config.db_path.to_str().unwrap_or("./data/nautivecs.json")
        )?;

        info!("NautivecsEngine: initialized with {} existing chunks", store.len());

        Ok(Self { config, embeddings, store })
    }

    /// Index all supported files in a directory recursively.
    pub async fn index_directory(&mut self, dir: impl AsRef<Path>) -> Result<usize> {
        let dir = dir.as_ref();
        let files = self.discover_files(dir)?;
        info!("NautivecsEngine: found {} files to index in {}", files.len(), dir.display());

        let mut total_indexed = 0;

        for file_path in &files {
            let relative = file_path.strip_prefix(dir)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            match self.index_file(file_path, &relative).await {
                Ok(count) => {
                    total_indexed += count;
                    if count > 0 {
                        info!("  indexed {} chunks from {}", count, relative);
                    }
                }
                Err(e) => {
                    tracing::warn!("  failed to index {}: {}", relative, e);
                }
            }
        }

        // Persist after indexing
        self.store.save()?;
        info!("NautivecsEngine: indexed {} new chunks (total: {})", total_indexed, self.store.len());

        Ok(total_indexed)
    }

    /// Index a single file.
    pub async fn index_file(&mut self, file_path: &Path, relative_path: &str) -> Result<usize> {
        let source = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read {}", file_path.display()))?;

        if source.trim().is_empty() {
            return Ok(0);
        }

        // Remove old chunks for this file (re-index)
        self.store.remove_file(relative_path);

        // Chunk based on file extension
        let ext = file_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let chunks: Vec<CodeChunk> = match ext {
            "rs" => chunk_rust_file(&source, relative_path)?,
            _ => chunk_text_file(&source, relative_path, 50),
        };

        if chunks.is_empty() {
            return Ok(0);
        }

        // Generate embeddings in batch
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&texts).await?;

        // Store
        let inserted = self.store.insert(&chunks, &embeddings);
        Ok(inserted)
    }

    /// Query the index for relevant code/knowledge.
    /// Uses hybrid search (vector + keyword) with Reciprocal Rank Fusion.
    pub async fn query(&self, query_text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embeddings.embed_one(query_text).await?;
        Ok(self.store.search_hybrid(&query_embedding, query_text, top_k))
    }

    /// Query using only vector similarity (no keyword matching).
    pub async fn query_vector(&self, query_text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embeddings.embed_one(query_text).await?;
        Ok(self.store.search_vector(&query_embedding, top_k))
    }

    /// Query using only keyword matching (no embeddings needed).
    pub fn query_keyword(&self, query_text: &str, top_k: usize) -> Vec<SearchResult> {
        self.store.search_keyword(query_text, top_k)
    }

    /// Get the total number of indexed chunks.
    pub fn chunk_count(&self) -> usize {
        self.store.len()
    }

    /// Discover all indexable files in a directory.
    fn discover_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.walk_dir(dir, &mut files)?;
        Ok(files)
    }

    fn walk_dir(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        let dir_name = dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip excluded directories
        if self.config.skip_dirs.iter().any(|s| s == dir_name) {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.walk_dir(&path, files)?;
            } else if path.is_file() {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if self.config.index_extensions.iter().any(|e| e == ext) {
                    files.push(path);
                }
            }
        }

        Ok(())
    }
}
