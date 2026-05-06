//! # nautivecs — AST-Aware Code Vectorization & Semantic Retrieval
//!
//! A Rust-native context injection engine that makes small LLMs perform like
//! specialists by providing exactly the right code/knowledge at query time.
//!
//! ## Architecture
//!
//! ```text
//! Source Code → tree-sitter AST → Structural Chunks → Embeddings → LanceDB
//!                                                                      ↓
//! Query → Embed → Cosine Similarity + BM25 Hybrid → Top-K Results → LLM Context
//! ```
//!
//! ## Key Features
//!
//! - **AST-aware chunking**: Splits code by functions/structs/impls, not arbitrary byte offsets
//! - **Hybrid search**: Combines vector similarity with keyword matching (BM25)
//! - **Zero-config storage**: Embedded LanceDB — no external database server
//! - **Flexible embeddings**: Use local llama.cpp or any OpenAI-compatible endpoint
//! - **Incremental indexing**: Only re-embeds changed files
//! - **Metadata-rich**: Each chunk stores file path, function name, line numbers
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use nautivecs::{NautivecsEngine, Config};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = Config::builder()
//!         .embedding_endpoint("http://100.72.182.77:5001/v1/embeddings")
//!         .db_path("./data/nautivecs.lance")
//!         .build();
//!
//!     let mut engine = NautivecsEngine::init(config).await?;
//!
//!     // Index a codebase
//!     engine.index_directory("./src").await?;
//!
//!     // Query for relevant code
//!     let results = engine.query("dipole scan parallel GPU", 5).await?;
//!     for result in &results {
//!         println!("[{:.2}] {}:{} — {}", result.score, result.file_path, result.line_start, result.function_name);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod chunker;
pub mod config;
pub mod embeddings;
pub mod engine;
pub mod pipeline;
pub mod storage;

// Re-exports for convenience
pub use config::Config;
pub use engine::NautivecsEngine;
pub use pipeline::{ContextFragment, InjectedContextBuilder};
pub use storage::SearchResult;
