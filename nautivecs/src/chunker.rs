//! AST-aware code chunking via tree-sitter.
//!
//! Instead of splitting code into arbitrary byte ranges, we parse the AST
//! and extract meaningful structural units: functions, structs, impls, enums.
//! Each chunk is a complete, self-contained code unit that an LLM can understand.

use anyhow::{Context, Result};
use std::path::Path;

/// A structural chunk extracted from source code.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The source file path (relative to index root)
    pub file_path: String,
    /// Name of the function/struct/impl (if identifiable)
    pub symbol_name: String,
    /// Type of symbol: "function", "struct", "impl", "enum", "module", "text"
    pub symbol_type: String,
    /// The actual code text
    pub text: String,
    /// Starting line number (1-indexed)
    pub line_start: usize,
    /// Ending line number (1-indexed)
    pub line_end: usize,
    /// SHA256 hash of the text (for dedup/change detection)
    pub content_hash: String,
}

/// Parse a Rust source file into structural chunks using tree-sitter.
pub fn chunk_rust_file(source: &str, file_path: &str) -> Result<Vec<CodeChunk>> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_rust::language();
    parser.set_language(&language)
        .context("Failed to set tree-sitter Rust language")?;

    let tree = parser.parse(source, None)
        .context("Failed to parse source with tree-sitter")?;

    // Safety check: if tree-sitter detected syntax errors, warn and fall back
    if tree.root_node().has_error() {
        eprintln!(
            "⚠️ [nautivecs] Syntax error detected in file: {}\n\
             [Hint] Check for unmatched braces or malformed macros.\n\
             [Fallback] Using line-based chunking for this file.",
            file_path
        );
        return Ok(chunk_text_file(source, file_path, 50));
    }

    let mut chunks = Vec::new();

    // Query for top-level items: functions, structs, enums, impls, traits, mods
    let query_str = r#"
        (function_item) @item
        (struct_item) @item
        (enum_item) @item
        (impl_item) @item
        (trait_item) @item
        (mod_item) @item
        (const_item) @item
        (static_item) @item
        (type_item) @item
    "#;

    let query = tree_sitter::Query::new(&language, query_str)
        .context("Failed to compile tree-sitter query")?;

    let mut cursor = tree_sitter::QueryCursor::new();
    let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

    for m in matches {
        for capture in m.captures {
            let node = capture.node;
            let text = node.utf8_text(source.as_bytes())
                .unwrap_or("")
                .to_string();

            if text.is_empty() || text.len() < 10 {
                continue;
            }

            let line_start = node.start_position().row + 1;
            let line_end = node.end_position().row + 1;

            // Extract symbol name from the first identifier child
            let symbol_name = extract_symbol_name(&node, source);
            let symbol_type = node.kind().replace("_item", "");

            let content_hash = compute_hash(&text);

            chunks.push(CodeChunk {
                file_path: file_path.to_string(),
                symbol_name,
                symbol_type,
                text,
                line_start,
                line_end,
                content_hash,
            });
        }
    }

    // If no structural items found (e.g., a script or config file),
    // fall back to the whole file as one chunk
    if chunks.is_empty() && !source.trim().is_empty() {
        let line_count = source.lines().count();
        chunks.push(CodeChunk {
            file_path: file_path.to_string(),
            symbol_name: Path::new(file_path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".into()),
            symbol_type: "file".to_string(),
            text: source.to_string(),
            line_start: 1,
            line_end: line_count,
            content_hash: compute_hash(source),
        });
    }

    Ok(chunks)
}

/// Chunk a non-Rust file by splitting on blank lines or fixed size.
/// Used for .md, .toml, .wgsl, .py files.
pub fn chunk_text_file(source: &str, file_path: &str, max_lines: usize) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.lines().collect();

    if lines.is_empty() {
        return chunks;
    }

    // Split on double-newlines (paragraph boundaries) or max_lines
    let mut current_chunk = Vec::new();
    let mut chunk_start = 1usize;

    for (i, line) in lines.iter().enumerate() {
        current_chunk.push(*line);

        let is_boundary = line.is_empty() && current_chunk.len() > 3;
        let is_max = current_chunk.len() >= max_lines;
        let is_last = i == lines.len() - 1;

        if is_boundary || is_max || is_last {
            let text = current_chunk.join("\n");
            if text.trim().len() > 10 {
                chunks.push(CodeChunk {
                    file_path: file_path.to_string(),
                    symbol_name: format!("chunk_{}", chunks.len()),
                    symbol_type: "text".to_string(),
                    text: text.clone(),
                    line_start: chunk_start,
                    line_end: chunk_start + current_chunk.len() - 1,
                    content_hash: compute_hash(&text),
                });
            }
            current_chunk.clear();
            chunk_start = i + 2; // next line (1-indexed)
        }
    }

    chunks
}

/// Extract the symbol name from a tree-sitter node.
fn extract_symbol_name(node: &tree_sitter::Node, source: &str) -> String {
    // Look for the first "identifier" or "type_identifier" child
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" | "type_identifier" => {
                if let Ok(name) = child.utf8_text(source.as_bytes()) {
                    return name.to_string();
                }
            }
            _ => {}
        }
    }
    "anonymous".to_string()
}

/// Compute SHA256 hash of content for change detection.
fn compute_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string() // first 16 hex chars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_rust_functions() {
        let source = r#"
fn add(a: f64, b: f64) -> f64 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
"#;
        let chunks = chunk_rust_file(source, "test.rs").unwrap();
        assert!(chunks.len() >= 3, "Expected at least 3 chunks, got {}", chunks.len());
        assert!(chunks.iter().any(|c| c.symbol_name == "add"));
        assert!(chunks.iter().any(|c| c.symbol_name == "Point"));
    }

    #[test]
    fn test_chunk_empty_file() {
        let chunks = chunk_rust_file("", "empty.rs").unwrap();
        assert!(chunks.is_empty());
    }
}
