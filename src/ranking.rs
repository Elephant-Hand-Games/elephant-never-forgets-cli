use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RankedResult {
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

pub fn keyword_score(query: &str, text: &str) -> f32 {
    if query.trim().is_empty() {
        return 0.0;
    }
    let query = query.to_lowercase();
    let text = text.to_lowercase();
    if text.contains(&query) {
        1.0
    } else {
        0.0
    }
}
