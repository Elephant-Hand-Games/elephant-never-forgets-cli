use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedDocument {
    pub text: String,
    pub line_count: usize,
}

pub fn extract_text(path: &Path) -> Result<ExtractedDocument> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let line_count = text.lines().count();
    Ok(ExtractedDocument { text, line_count })
}
