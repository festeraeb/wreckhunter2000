//! Context formatting pipeline — converts search results into LLM-ready prompts.
//!
//! Takes raw search hits from the vector store and formats them into clean,
//! structured markdown blocks that can be injected into any LLM system prompt.

use std::fmt::Write;

/// Individual text chunk bound to its structural repository attributes.
#[derive(Debug, Clone)]
pub struct ContextFragment {
    pub text: String,
    pub file_path: String,
    pub function_name: String,
    pub line_start: i32,
    pub line_end: i32,
    pub search_score: f32,
}

/// Converts search results into formatted context blocks for LLM injection.
pub struct InjectedContextBuilder {
    max_tokens_budget: usize,
    strip_empty_lines: bool,
}

impl Default for InjectedContextBuilder {
    fn default() -> Self {
        Self {
            max_tokens_budget: 4096,
            strip_empty_lines: true,
        }
    }
}

impl InjectedContextBuilder {
    pub fn new(max_tokens_budget: usize, strip_empty_lines: bool) -> Self {
        Self {
            max_tokens_budget,
            strip_empty_lines,
        }
    }

    /// Cleans code block anomalies and standardizes spaces for clean prompt embedding.
    fn sanitize_code_block(&self, raw_code: &str) -> String {
        let mut processed = String::with_capacity(raw_code.len());

        for line in raw_code.lines() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() && self.strip_empty_lines {
                continue;
            }
            processed.push_str(trimmed);
            processed.push('\n');
        }

        processed
    }

    /// Takes the retrieved search fragments and renders a formatted text context envelope.
    pub fn build_system_context(&self, fragments: &[ContextFragment]) -> String {
        if fragments.is_empty() {
            return "No relevant codebase context was retrieved for this operation.\n".to_string();
        }

        let mut context_buffer = String::new();

        let _ = writeln!(
            &mut context_buffer,
            "### INJECTED CODEBASE CONTEXT\n\
            The following structural items have been extracted from the local repository via AST analysis.\n\
            Use these precise references to guide code completion, analysis, or generation tasks.\n"
        );

        for (index, fragment) in fragments.iter().enumerate() {
            let cleaned_code = self.sanitize_code_block(&fragment.text);

            let _ = writeln!(
                &mut context_buffer,
                "--- [{}] FILE: {} | FUNCTION: {}() | LINES: {}-{} (Score: {:.4}) ---",
                index + 1,
                fragment.file_path,
                fragment.function_name,
                fragment.line_start,
                fragment.line_end,
                fragment.search_score
            );

            let _ = writeln!(&mut context_buffer, "```rust\n{}```\n", cleaned_code.trim_end());
        }

        // Rough token budget enforcement (~4 chars per token)
        if context_buffer.len() > self.max_tokens_budget * 4 {
            context_buffer.truncate(self.max_tokens_budget * 4);
            context_buffer.push_str("\n[Context allocation limit reached; structural elements truncated...]\n");
        }

        context_buffer
    }
}

/// Convert SearchResults from the vector store into ContextFragments for the builder.
impl From<&crate::storage::SearchResult> for ContextFragment {
    fn from(result: &crate::storage::SearchResult) -> Self {
        Self {
            text: result.text.clone(),
            file_path: result.file_path.clone(),
            function_name: result.function_name.clone(),
            line_start: result.line_start as i32,
            line_end: result.line_end as i32,
            search_score: result.score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_formatting_pipeline() {
        let builder = InjectedContextBuilder::new(2048, true);

        let fragments = vec![
            ContextFragment {
                text: "fn test_harness() {\n\n    println!(\"Nautivecs deployed\");\n}".to_string(),
                file_path: "src/main.rs".to_string(),
                function_name: "test_harness".to_string(),
                line_start: 10,
                line_end: 14,
                search_score: 0.8942,
            }
        ];

        let payload = builder.build_system_context(&fragments);

        assert!(payload.contains("FILE: src/main.rs"));
        assert!(payload.contains("FUNCTION: test_harness()"));
        assert!(payload.contains("println!(\"Nautivecs deployed\");"));
    }

    #[test]
    fn test_empty_fragments() {
        let builder = InjectedContextBuilder::default();
        let payload = builder.build_system_context(&[]);
        assert!(payload.contains("No relevant codebase context"));
    }

    #[test]
    fn test_token_budget_truncation() {
        let builder = InjectedContextBuilder::new(50, true); // Very small budget

        let fragments = vec![
            ContextFragment {
                text: "fn big_function() {\n    // lots of code here\n    let x = 1;\n    let y = 2;\n    let z = x + y;\n    println!(\"{}\", z);\n}".to_string(),
                file_path: "src/big.rs".to_string(),
                function_name: "big_function".to_string(),
                line_start: 1,
                line_end: 100,
                search_score: 0.95,
            }
        ];

        let payload = builder.build_system_context(&fragments);
        assert!(payload.contains("[Context allocation limit reached"));
    }
}
