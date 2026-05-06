//! LanceDB storage layer — embedded vector database (the SQLite of vectors).
//!
//! Stores code chunks with their embeddings and metadata.
//! Supports incremental append, dedup by content hash, and hybrid search.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

use crate::chunker::CodeChunk;

/// A search result returned from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The code text that matched
    pub text: String,
    /// Source file path
    pub file_path: String,
    /// Function/struct/impl name
    pub function_name: String,
    /// Symbol type (function, struct, impl, etc.)
    pub symbol_type: String,
    /// Starting line in the source file
    pub line_start: usize,
    /// Ending line in the source file
    pub line_end: usize,
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
}

/// Stored record in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub text: String,
    pub file_path: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub line_start: u32,
    pub line_end: u32,
    pub content_hash: String,
    pub embedding: Vec<f32>,
}

/// In-memory vector store (MVP — replace with LanceDB for production scale).
/// This allows us to ship without the heavy Arrow/Lance dependency initially,
/// then swap in LanceDB when we're ready for millions of chunks.
pub struct VectorStore {
    chunks: Vec<StoredChunk>,
    db_path: String,
}

impl VectorStore {
    /// Create or open a vector store at the given path.
    pub fn open(db_path: &str) -> Result<Self> {
        let mut store = Self {
            chunks: Vec::new(),
            db_path: db_path.to_string(),
        };

        // Load from disk if exists
        let path = Path::new(db_path);
        if path.exists() {
            let data = std::fs::read_to_string(path)
                .context("Failed to read vector store")?;
            store.chunks = serde_json::from_str(&data)
                .unwrap_or_default();
            info!("VectorStore: loaded {} chunks from {}", store.chunks.len(), db_path);
        }

        Ok(store)
    }

    /// Insert chunks with their embeddings. Deduplicates by content_hash.
    pub fn insert(&mut self, chunks: &[CodeChunk], embeddings: &[Vec<f32>]) -> usize {
        let mut inserted = 0;
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            // Dedup: skip if content_hash already exists
            if self.chunks.iter().any(|c| c.content_hash == chunk.content_hash) {
                continue;
            }

            self.chunks.push(StoredChunk {
                text: chunk.text.clone(),
                file_path: chunk.file_path.clone(),
                symbol_name: chunk.symbol_name.clone(),
                symbol_type: chunk.symbol_type.clone(),
                line_start: chunk.line_start as u32,
                line_end: chunk.line_end as u32,
                content_hash: chunk.content_hash.clone(),
                embedding: embedding.clone(),
            });
            inserted += 1;
        }
        inserted
    }

    /// Remove all chunks from a specific file (for re-indexing after edit).
    pub fn remove_file(&mut self, file_path: &str) {
        self.chunks.retain(|c| c.file_path != file_path);
    }

    /// Search by vector similarity (cosine). Returns top-K results.
    pub fn search_vector(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(usize, f32)> = self.chunks.iter()
            .enumerate()
            .map(|(i, chunk)| {
                let score = cosine_similarity(query_embedding, &chunk.embedding);
                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.iter()
            .take(top_k)
            .filter(|(_, score)| *score > 0.05)
            .map(|(i, score)| {
                let chunk = &self.chunks[*i];
                SearchResult {
                    text: chunk.text.clone(),
                    file_path: chunk.file_path.clone(),
                    function_name: chunk.symbol_name.clone(),
                    symbol_type: chunk.symbol_type.clone(),
                    line_start: chunk.line_start as usize,
                    line_end: chunk.line_end as usize,
                    score: *score,
                }
            })
            .collect()
    }

    /// Search by keyword (BM25-style term frequency). Returns top-K results.
    pub fn search_keyword(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower
            .split_whitespace()
            .collect();

        let mut scored: Vec<(usize, f32)> = self.chunks.iter()
            .enumerate()
            .map(|(i, chunk)| {
                let text_lower = chunk.text.to_lowercase();
                let name_lower = chunk.symbol_name.to_lowercase();
                let mut score = 0.0f32;

                for term in &query_terms {
                    // Exact match in symbol name = high weight
                    if name_lower.contains(term) {
                        score += 3.0;
                    }
                    // Match in code text = lower weight
                    let count = text_lower.matches(term).count() as f32;
                    score += count.min(5.0) * 0.5; // cap at 5 occurrences
                }

                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.iter()
            .take(top_k)
            .filter(|(_, score)| *score > 0.0)
            .map(|(i, score)| {
                let chunk = &self.chunks[*i];
                SearchResult {
                    text: chunk.text.clone(),
                    file_path: chunk.file_path.clone(),
                    function_name: chunk.symbol_name.clone(),
                    symbol_type: chunk.symbol_type.clone(),
                    line_start: chunk.line_start as usize,
                    line_end: chunk.line_end as usize,
                    score: *score,
                }
            })
            .collect()
    }

    /// Hybrid search: Reciprocal Rank Fusion (RRF) combining vector + keyword.
    /// RRF score = sum(1 / (k + rank_i)) across all ranking lists.
    pub fn search_hybrid(&self, query_embedding: &[f32], query_text: &str, top_k: usize) -> Vec<SearchResult> {
        let k = 60.0f32; // RRF constant (standard value)

        let vector_results = self.search_vector(query_embedding, top_k * 2);
        let keyword_results = self.search_keyword(query_text, top_k * 2);

        // Build RRF scores by content_hash
        let mut rrf_scores: std::collections::HashMap<String, (f32, SearchResult)> = std::collections::HashMap::new();

        for (rank, result) in vector_results.iter().enumerate() {
            let hash = format!("{}:{}", result.file_path, result.line_start);
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            rrf_scores.entry(hash)
                .and_modify(|(score, _)| *score += rrf)
                .or_insert((rrf, result.clone()));
        }

        for (rank, result) in keyword_results.iter().enumerate() {
            let hash = format!("{}:{}", result.file_path, result.line_start);
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            rrf_scores.entry(hash)
                .and_modify(|(score, _)| *score += rrf)
                .or_insert((rrf, result.clone()));
        }

        // Sort by fused RRF score
        let mut fused: Vec<(f32, SearchResult)> = rrf_scores.into_values().collect();
        fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        fused.into_iter()
            .take(top_k)
            .map(|(score, mut result)| {
                result.score = score;
                result
            })
            .collect()
    }

    /// Persist the store to disk.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = Path::new(&self.db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string(&self.chunks)?;
        std::fs::write(&self.db_path, json)?;
        info!("VectorStore: saved {} chunks to {}", self.chunks.len(), self.db_path);
        Ok(())
    }

    /// Total number of stored chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
