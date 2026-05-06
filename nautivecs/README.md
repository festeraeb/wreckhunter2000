# ⚓ nautivecs

A high-performance, Rust-native context injection engine designed to navigate massive codebases and deliver hyper-focused contextual grounding to LLMs. `nautivecs` combines syntax-aware Abstract Syntax Tree (AST) parsing with an optimized hybrid retrieval mechanism, enabling instantaneous prompt enrichment directly within developer environments and automated container runtimes.

Developed as a core component of the **CESAROPS** shipwreck detection suite, it sits alongside `nauticuvs` (precision curvelet transforms) and `cesarops-hybrid-engine` (GPU cluster coordination) to form an integrated local AI workspace.

---

## 🏗️ Two-Pronged Architecture

To balance ease of integration with planet-scale indexing performance, `nautivecs` uses a staged evolutionary roadmap:

* **v0.1.0 (Current Stable Release):** A serverless, lightweight implementation utilizing a JSON-backed vector and keyword store. It compiles with zero external native dependency requirements, avoiding painful Arrow/wgpu workspace compilation conflicts.
* **v0.2.0 (Enterprise Scaling Path):** Introduces a serverless **LanceDB** storage backend and local optimized `llama.cpp` vector generation, explicitly quarantined behind a feature flag.

---

## 🛠️ Installation & Feature Flags

Add `nautivecs` to your crate's `Cargo.toml`:

```toml
[dependencies]
# Standard lightweight installation (Recommended for v0.1.0 stability)
nautivecs = "0.1"

# For high-throughput production vectors (Feature-gated)
# nautivecs = { version = "0.1", features = ["lancedb"] }
```

### Building in Shared Workspace Environments

To build without drawing in conflicting dependencies from neighboring GPU engines:

```bash
cargo build -p nautivecs --no-default-features
```

---

## 🚀 Quickstart

```rust
use anyhow::Result;
use nautivecs::{Config, NautivecsEngine, InjectedContextBuilder, ContextFragment};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Configure targeting your local embedding endpoint
    let config = Config::builder()
        .embedding_endpoint("http://100.72.182.77:5001/v1")
        .db_path("./data/nautivecs_store.json")
        .vector_dimensions(768) // nomic-embed-text-v1.5
        .build();

    let mut engine = NautivecsEngine::init(config).await?;

    // 2. Index a codebase directory
    engine.index_directory("./src").await?;

    // 3. Hybrid search (vector + keyword RRF fusion)
    let results = engine.query("dipole scan GPU orchestration", 3).await?;

    // 4. Format results for LLM injection
    let fragments: Vec<ContextFragment> = results.iter()
        .map(ContextFragment::from)
        .collect();

    let builder = InjectedContextBuilder::new(4096, true);
    let context_block = builder.build_system_context(&fragments);

    // 5. Inject into your LLM system message
    println!("{}", context_block);
    Ok(())
}
```

---

## 🎛️ Pipeline Core Mechanics

### 1. AST-Aware Code Chunking (`chunker.rs`)

Instead of slicing text into fixed token boundaries, `nautivecs` leverages **Tree-Sitter** to analyze source files into logical syntax units (`struct`, `enum`, `impl`, `fn`), tracking function boundaries and exact line mappings.

### 2. Dual-Engine Embedding Layer (`embeddings.rs`)

Defaults to an external OpenAI-compatible endpoint (e.g., `nomic-embed-text-v1.5` at 137MB on your GPU node). Falls back to a local n-gram hash embedder if the endpoint is unreachable.

### 3. Fused Hybrid Search (`storage.rs`)

Parallel search: Cosine similarity + BM25 keyword matching, merged via **Reciprocal Rank Fusion (RRF, k=60)**.

### 4. Context Formatting (`pipeline.rs`)

Converts search results into clean markdown blocks with file paths, function names, line numbers, and confidence scores — ready for direct LLM system message injection.

---

## 💻 CLI

```bash
# Index a directory
nautivecs-cli index ./src

# Hybrid search
nautivecs-cli query "process sailing coordinates" --top-k 3

# Keyword-only search (no embeddings needed)
nautivecs-cli keyword "dipole_scan" --top-k 5

# Show stats
nautivecs-cli stats
```

---

## 🔧 Embedding Model Recommendation

For the P100 cluster: **nomic-embed-text-v1.5.Q8_0.gguf** (~140MB)
- 768 dimensions, 8192 context window
- Negligible VRAM usage alongside Qwen3.6-35B
- Serve via a second llama.cpp instance on port 5002

---

## ⚖️ License

MIT OR Apache-2.0
