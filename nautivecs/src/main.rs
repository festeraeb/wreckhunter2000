//! nautivecs-cli — AST-driven codebase indexing and context injection CLI.
//!
//! Usage:
//!   nautivecs-cli index ./src              # Index a directory
//!   nautivecs-cli query "dipole scan GPU"  # Hybrid search
//!   nautivecs-cli stats                    # Show index stats

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use nautivecs::{Config, NautivecsEngine, InjectedContextBuilder, ContextFragment};

#[derive(Parser, Debug)]
#[command(name = "nautivecs-cli")]
#[command(author = "CESAROPS")]
#[command(version = "0.1.0")]
#[command(about = "AST-aware codebase vectorization and semantic retrieval for LLM context injection")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Embedding endpoint URL (OpenAI-compatible /v1/embeddings)
    #[arg(long, global = true, default_value = "http://100.72.182.77:5001/v1")]
    endpoint: String,

    /// Vector store path
    #[arg(long, global = true, default_value = "./data/nautivecs_store.json")]
    db_path: String,

    /// Vector dimensions (768 for nomic-embed, 384 for bge-small)
    #[arg(long, global = true, default_value_t = 768)]
    dimensions: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Index a directory — parses Rust files via tree-sitter AST, embeds chunks
    Index {
        /// Directory to index
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Hybrid search (vector + keyword RRF fusion)
    Query {
        /// Natural language query or code description
        query_string: String,

        /// Maximum results to return
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,

        /// Output raw JSON instead of formatted markdown
        #[arg(long)]
        json: bool,
    },

    /// Show index statistics
    Stats,

    /// Search by keyword only (no embeddings needed)
    Keyword {
        /// Search terms
        query_string: String,

        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config = Config::builder()
        .embedding_endpoint(&cli.endpoint)
        .db_path(&cli.db_path)
        .vector_dimensions(cli.dimensions)
        .build();

    let mut engine = NautivecsEngine::init(config).await
        .context("Failed to initialize nautivecs engine")?;

    match cli.command {
        Commands::Index { path } => {
            println!("⚓ nautivecs: indexing {:?}", path);
            let count = engine.index_directory(&path).await?;
            println!("✅ Indexed {} new chunks (total: {})", count, engine.chunk_count());
        }

        Commands::Query { query_string, top_k, json } => {
            let results = engine.query(&query_string, top_k).await?;

            if results.is_empty() {
                println!("No matches found for: \"{}\"", query_string);
                return Ok(());
            }

            if json {
                // Raw JSON output for piping to other tools
                let json_out = serde_json::to_string_pretty(&results)?;
                println!("{}", json_out);
            } else {
                // Formatted markdown context block (ready for LLM injection)
                let fragments: Vec<ContextFragment> = results.iter()
                    .map(ContextFragment::from)
                    .collect();

                let builder = InjectedContextBuilder::new(4096, true);
                let payload = builder.build_system_context(&fragments);

                println!("{}", payload);
            }
        }

        Commands::Keyword { query_string, top_k } => {
            let results = engine.query_keyword(&query_string, top_k);

            if results.is_empty() {
                println!("No keyword matches for: \"{}\"", query_string);
                return Ok(());
            }

            for (i, result) in results.iter().enumerate() {
                println!(
                    "[{}] {:.3} | {}:{}-{} | {}",
                    i + 1, result.score, result.file_path,
                    result.line_start, result.line_end, result.function_name
                );
            }
        }

        Commands::Stats => {
            println!("⚓ nautivecs store: {}", cli.db_path);
            println!("   chunks indexed: {}", engine.chunk_count());
            println!("   dimensions: {}", cli.dimensions);
            println!("   endpoint: {}", cli.endpoint);
        }
    }

    Ok(())
}
