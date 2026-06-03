use std::{cmp::Ordering, collections::BinaryHeap};

use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RankedResult {
    pub path: String,
    pub snippet: String,
    pub score: f32,
    pub kind: String,
    pub file_id: Option<i64>,
    pub chunk_id: Option<i64>,
    pub chunk_index: Option<i64>,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub keyword_score: f32,
    pub vector_score: f32,
    pub metadata_score: f32,
    pub profile_hash: String,
    pub mode: String,
    pub level: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreWeights {
    pub vector: f32,
    pub keyword: f32,
    pub metadata: f32,
}

#[derive(Debug, Clone)]
pub struct TopK {
    limit: usize,
    heap: BinaryHeap<std::cmp::Reverse<HeapEntry>>,
}

#[derive(Debug, Clone)]
struct HeapEntry {
    key: SortKey,
    result: RankedResult,
}

#[derive(Debug, Clone)]
struct SortKey {
    score: f32,
    vector_score: f32,
    keyword_score: f32,
    metadata_score: f32,
    path: String,
    kind: String,
    file_id: Option<i64>,
    chunk_id: Option<i64>,
    chunk_index: Option<i64>,
    start_line: Option<i64>,
    end_line: Option<i64>,
}

impl TopK {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, result: RankedResult) {
        if self.limit == 0 {
            return;
        }
        let entry = HeapEntry::new(result);
        self.heap.push(std::cmp::Reverse(entry));
        if self.heap.len() > self.limit {
            let _ = self.heap.pop();
        }
    }

    pub fn into_sorted_vec(self) -> Vec<RankedResult> {
        let mut results: Vec<_> = self.heap.into_iter().map(|entry| entry.0.result).collect();
        results.sort_by_key(|result| std::cmp::Reverse(sort_key(result)));
        results
    }
}

impl HeapEntry {
    fn new(result: RankedResult) -> Self {
        Self {
            key: sort_key(&result),
            result,
        }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.result == other.result
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SortKey {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.vector_score.to_bits() == other.vector_score.to_bits()
            && self.keyword_score.to_bits() == other.keyword_score.to_bits()
            && self.metadata_score.to_bits() == other.metadata_score.to_bits()
            && self.path == other.path
            && self.kind == other.kind
            && self.file_id == other.file_id
            && self.chunk_id == other.chunk_id
            && self.chunk_index == other.chunk_index
            && self.start_line == other.start_line
            && self.end_line == other.end_line
    }
}

impl Eq for SortKey {}

impl Ord for SortKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.vector_score.total_cmp(&other.vector_score))
            .then_with(|| self.keyword_score.total_cmp(&other.keyword_score))
            .then_with(|| self.metadata_score.total_cmp(&other.metadata_score))
            .then_with(|| other.path.cmp(&self.path))
            .then_with(|| other.kind.cmp(&self.kind))
            .then_with(|| compare_option_desc(self.file_id, other.file_id))
            .then_with(|| compare_option_desc(self.chunk_id, other.chunk_id))
            .then_with(|| compare_option_desc(self.chunk_index, other.chunk_index))
            .then_with(|| compare_option_desc(self.start_line, other.start_line))
            .then_with(|| compare_option_desc(self.end_line, other.end_line))
    }
}

impl PartialOrd for SortKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn normalize_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|part| part.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = dot_product(left, right);
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    (dot / (left_norm * right_norm)).clamp(-1.0, 1.0)
}

pub fn keyword_score(query: &str, text: &str) -> f32 {
    let normalized_query = normalize_query(query);
    if normalized_query.is_empty() {
        return 0.0;
    }

    let query_tokens = unique_tokens(&normalized_query);
    if query_tokens.is_empty() {
        return 0.0;
    }

    let haystack = text.to_lowercase();
    let matched = query_tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .count();

    let mut score = matched as f32 / query_tokens.len() as f32;
    if haystack.contains(&normalized_query) {
        score = score.max(1.0);
    }
    score.clamp(0.0, 1.0)
}

pub fn hybrid_score(
    vector_score: f32,
    keyword_score: f32,
    metadata_score: f32,
    weights: ScoreWeights,
) -> f32 {
    let total_weight = weights.vector + weights.keyword + weights.metadata;
    if total_weight <= 0.0 {
        return 0.0;
    }
    let combined = vector_score * weights.vector
        + keyword_score * weights.keyword
        + metadata_score * weights.metadata;
    (combined / total_weight).clamp(0.0, 1.0)
}

pub fn top_k(results: impl IntoIterator<Item = RankedResult>, limit: usize) -> Vec<RankedResult> {
    let mut heap = TopK::new(limit);
    for result in results {
        heap.push(result);
    }
    heap.into_sorted_vec()
}

fn sort_key(result: &RankedResult) -> SortKey {
    SortKey {
        score: result.score,
        vector_score: result.vector_score,
        keyword_score: result.keyword_score,
        metadata_score: result.metadata_score,
        path: result.path.clone(),
        kind: result.kind.clone(),
        file_id: result.file_id,
        chunk_id: result.chunk_id,
        chunk_index: result.chunk_index,
        start_line: result.start_line,
        end_line: result.end_line,
    }
}

fn compare_option_desc(left: Option<i64>, right: Option<i64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.cmp(&left),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn unique_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for token in text.split(|ch: char| !ch.is_ascii_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        let token = token.to_lowercase();
        if !tokens.iter().any(|existing| existing == &token) {
            tokens.push(token);
        }
    }
    tokens
}
