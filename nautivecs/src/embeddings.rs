//! Embedding generation — supports remote endpoints (OpenAI-compatible).
//!
//! Uses the same KoboldCPP/OpenAI `/v1/embeddings` endpoint that's already
//! running on the cluster. No separate embedding model needed if the main
//! model supports it, or point at a dedicated small embedding model.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Embedding provider that calls an OpenAI-compatible endpoint.
pub struct EmbeddingClient {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    dimensions: usize,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn new(endpoint: &str, model: &str, dimensions: usize) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("http client"),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    /// Generate embeddings for a batch of texts.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.endpoint);
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let response = match self.client
            .post(&url)
            .json(&request)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                // Connection failed — use fallback embeddings
                warn!("Embedding endpoint unreachable ({}): {} — using n-gram fallback", url, e);
                return Ok(texts.iter().map(|t| self.fallback_embedding(t)).collect());
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            // Fall back to simple hash-based embeddings if endpoint doesn't support it
            warn!("Embedding endpoint returned {}: {} — using fallback", status, &body[..body.len().min(200)]);
            return Ok(texts.iter().map(|t| self.fallback_embedding(t)).collect());
        }

        let resp: EmbeddingResponse = response.json().await
            .context("Failed to parse embedding response")?;

        Ok(resp.data.into_iter()
            .map(|d| {
                let mut emb = d.embedding;
                emb.resize(self.dimensions, 0.0);
                emb
            })
            .collect())
    }

    /// Generate embedding for a single text.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))
    }

    /// Fallback: generate a deterministic pseudo-embedding from text features.
    /// Not semantically meaningful but allows the system to function without
    /// a real embedding endpoint. Uses character n-gram hashing.
    fn fallback_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimensions];
        let text_lower = text.to_lowercase();

        // Hash character trigrams into embedding dimensions
        for window in text_lower.as_bytes().windows(3) {
            let hash = (window[0] as u64 * 31 * 31 + window[1] as u64 * 31 + window[2] as u64) as usize;
            let idx = hash % self.dimensions;
            embedding[idx] += 1.0;
        }

        // Normalize to unit vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
