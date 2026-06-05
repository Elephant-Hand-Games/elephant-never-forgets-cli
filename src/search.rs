use std::collections::{HashMap, HashSet};

use crate::{
    cli::{SearchArgs, SearchKindArg},
    config::{SearchLevel, SearchMode},
    errors::EnfError,
    ranking::{self, RankedResult},
};
use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};

pub fn run(args: SearchArgs, retrieve: bool) -> Result<()> {
    let mut config = crate::config::load()?;
    crate::config::apply_provider_overrides(&mut config, &args.provider);
    crate::config::validate(&config)?;
    let profile = crate::embed::active_profile(&config);
    let cwd = std::env::current_dir()?;
    let conn = crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let profile_id = crate::db::upsert_embedding_profile(&conn, &profile)?;
    let normalized_query = ranking::normalize_query(&args.query);
    let mode = args
        .mode
        .clone()
        .map(Into::into)
        .unwrap_or_else(|| config.search.default_mode.clone());
    let level = args
        .level
        .clone()
        .map(Into::into)
        .unwrap_or_else(|| config.search.default_level.clone());

    if args.cached_query_only
        && mode != SearchMode::Keyword
        && (!config.embedding.query_cache
            || crate::db::query_embedding(&conn, profile_id, &normalized_query)?.is_none())
    {
        return Err(EnfError::QueryEmbeddingNotCached.into());
    }

    let limit = args.limit.unwrap_or(config.search.limit);
    let mut results = Vec::new();
    let mut warnings = Vec::new();
    if mode != SearchMode::Vector && args.kind != SearchKindArg::Image {
        if level != SearchLevel::File {
            results.extend(query_chunks(
                &conn,
                &args.query,
                limit,
                config.search.snippet_chars,
                &profile.profile_hash,
            )?);
        }
        if level != SearchLevel::Chunk {
            results.extend(query_files(
                &conn,
                &args.query,
                limit,
                config.search.snippet_chars,
                &profile.profile_hash,
            )?);
        }
    }
    if mode != SearchMode::Keyword && args.kind != SearchKindArg::Image {
        let has_vectors = active_profile_has_vectors(&conn, profile_id)?;
        if !has_vectors {
            let message = format!(
                "active profile {} has no indexed embeddings; run `enf index .` to populate vectors",
                profile.profile_hash
            );
            if mode == SearchMode::Vector {
                anyhow::bail!("{message}");
            }
            warnings.push(message);
        }
        let vector_search = VectorSearch {
            conn: &conn,
            config: &config,
            profile_id,
            profile_hash: &profile.profile_hash,
            query: &args.query,
            normalized_query: &normalized_query,
            cached_query_only: args.cached_query_only,
            limit,
            mode: &mode,
        };
        if level != SearchLevel::File {
            results.extend(vector_chunk_results(vector_search)?);
        }
        if level != SearchLevel::Chunk {
            results.extend(vector_file_results(vector_search)?);
        }
    }
    if args.kind != SearchKindArg::Text {
        if mode != SearchMode::Vector {
            results.extend(query_images_by_path(
                &conn,
                &args.query,
                limit,
                &profile.profile_hash,
            )?);
        }
        if mode != SearchMode::Keyword && config.image.embedding.enabled {
            match image_vector_results(&conn, &config, &args.query, limit, &mode) {
                Ok(image_results) => results.extend(image_results),
                Err(err) if args.kind == SearchKindArg::Image || mode == SearchMode::Vector => {
                    return Err(err);
                }
                Err(err) => warnings.push(format!("image vector search skipped: {err}")),
            }
        }
    }
    let filters = SearchFilters::from_args(&args)?;
    let results = filters.apply(results);
    let candidate_limit = if config.reranker.enabled {
        config.reranker.candidate_limit.max(limit)
    } else {
        limit
    };
    let results = top_results(results, candidate_limit, config.search.max_chunks_per_file);
    let results = maybe_rerank(&config, &args.query, results, &mut warnings)?;
    let results = top_results(results, limit, config.search.max_chunks_per_file);

    if args.json || retrieve {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "query": args.query,
                "profile_hash": profile.profile_hash,
                "warnings": warnings,
                "results": results,
            }))?
        );
    } else {
        for warning in &warnings {
            eprintln!("warning: {warning}");
        }
        for result in results {
            let label = result_label(&cwd, &result);
            if let (Some(start), Some(end)) = (result.start_line, result.end_line) {
                println!(
                    "{} {label}:{start}-{end}  {:.3}",
                    result_kind_label(&result),
                    result.score
                );
            } else {
                println!(
                    "{} {label}  {:.3}",
                    result_kind_label(&result),
                    result.score
                );
            }
        }
    }
    Ok(())
}

fn active_profile_has_vectors(conn: &rusqlite::Connection, profile_id: i64) -> Result<bool> {
    Ok(!crate::db::vector_chunks_for_profile(conn, profile_id, 1, 0)?.is_empty())
}

fn result_label(cwd: &std::path::Path, result: &RankedResult) -> String {
    let path = cwd.join(&result.path);
    let mut url = format!("file://{}", percent_encode(&path.to_string_lossy()));
    if let Some(line) = result.start_line {
        url.push_str(&format!("#L{line}"));
    }
    format!("\x1b]8;;{url}\x1b\\{}\x1b]8;;\x1b\\", result.path)
}

fn result_kind_label(result: &RankedResult) -> &'static str {
    if result.kind == "image" {
        "[image]"
    } else {
        "[text]"
    }
}

fn percent_encode(input: &str) -> String {
    let mut encoded = String::new();
    for byte in input.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'.' | b'-' | b'_' | b'~' | b':' => {
                encoded.push(*byte as char)
            }
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
}

#[derive(Clone, Copy)]
struct VectorSearch<'a> {
    conn: &'a rusqlite::Connection,
    config: &'a crate::config::Config,
    profile_id: i64,
    profile_hash: &'a str,
    query: &'a str,
    normalized_query: &'a str,
    cached_query_only: bool,
    limit: usize,
    mode: &'a SearchMode,
}

fn vector_chunk_results(search: VectorSearch<'_>) -> Result<Vec<RankedResult>> {
    let has_vectors =
        !crate::db::vector_chunks_for_profile(search.conn, search.profile_id, 1, 0)?.is_empty();
    if !has_vectors {
        return Ok(Vec::new());
    }

    let query_vector = query_vector(search)?;

    let mut top = ranking::TopK::new(search.limit.saturating_mul(2).max(search.limit));
    crate::db::stream_vector_chunks_for_profile(
        search.conn,
        search.profile_id,
        search.config.search.batch_scan_size,
        |rows| {
            for row in rows {
                let vector_score = ranking::cosine_similarity(&query_vector, &row.vector).max(0.0);
                let keyword_score =
                    ranking::keyword_score(search.query, &format!("{}\n{}", row.path, row.text));
                let metadata_score = ranking::keyword_score(search.query, &row.path);
                let score = weighted_score(
                    search.mode,
                    search.config,
                    vector_score,
                    keyword_score,
                    metadata_score,
                );
                top.push(RankedResult {
                    path: row.path.clone(),
                    snippet: snippet(&row.text, search.config.search.snippet_chars),
                    score,
                    rerank_score: None,
                    kind: "chunk".into(),
                    file_type: file_type(&row.path),
                    file_id: Some(row.file_id),
                    chunk_id: Some(row.chunk_id),
                    chunk_index: Some(row.chunk_index),
                    start_line: row.start_line,
                    end_line: row.end_line,
                    keyword_score,
                    vector_score,
                    metadata_score,
                    profile_hash: search.profile_hash.to_string(),
                    mode: mode_label(search.mode).into(),
                    level: "chunk".into(),
                });
            }
            Ok(())
        },
    )?;
    Ok(top.into_sorted_vec())
}

fn vector_file_results(search: VectorSearch<'_>) -> Result<Vec<RankedResult>> {
    let has_vectors =
        !crate::db::vector_chunks_for_profile(search.conn, search.profile_id, 1, 0)?.is_empty();
    if !has_vectors {
        return Ok(Vec::new());
    }

    let query_vector = query_vector(search)?;

    let mut files: HashMap<i64, FileVectorAggregate> = HashMap::new();
    crate::db::stream_vector_chunks_for_profile(
        search.conn,
        search.profile_id,
        search.config.search.batch_scan_size,
        |rows| {
            for row in rows {
                let entry = files
                    .entry(row.file_id)
                    .or_insert_with(|| FileVectorAggregate::new(row.file_id, &row.path));
                entry.add_chunk(&row.text, &row.vector, search.config.search.snippet_chars);
            }
            Ok(())
        },
    )?;

    let mut top = ranking::TopK::new(search.limit);
    for aggregate in files.into_values() {
        let centroid = aggregate.centroid();
        let file_type = file_type(&aggregate.path);
        let vector_score = ranking::cosine_similarity(&query_vector, &centroid).max(0.0);
        let keyword_score = ranking::keyword_score(
            search.query,
            &format!("{}\n{}", aggregate.path, aggregate.sample),
        );
        let metadata_score = ranking::keyword_score(search.query, &aggregate.path);
        let score = weighted_score(
            search.mode,
            search.config,
            vector_score,
            keyword_score,
            metadata_score,
        );
        top.push(RankedResult {
            path: aggregate.path,
            snippet: snippet(&aggregate.sample, search.config.search.snippet_chars),
            score,
            rerank_score: None,
            kind: "file".into(),
            file_type,
            file_id: Some(aggregate.file_id),
            chunk_id: None,
            chunk_index: None,
            start_line: None,
            end_line: None,
            keyword_score,
            vector_score,
            metadata_score,
            profile_hash: search.profile_hash.to_string(),
            mode: mode_label(search.mode).into(),
            level: "file".into(),
        });
    }
    Ok(top.into_sorted_vec())
}

fn query_vector(search: VectorSearch<'_>) -> Result<Vec<f32>> {
    if search.config.embedding.query_cache {
        if let Some(vector) =
            crate::db::query_embedding(search.conn, search.profile_id, search.normalized_query)?
        {
            return Ok(vector);
        }
    }

    if search.cached_query_only {
        return Err(EnfError::QueryEmbeddingNotCached.into());
    }

    let mut provider = crate::providers::build_provider(search.config)?;
    let vector = provider.embed_query(search.query)?;
    if search.config.embedding.query_cache {
        crate::db::upsert_query_embedding(
            search.conn,
            search.profile_id,
            search.normalized_query,
            &vector,
        )?;
    }
    Ok(vector)
}

struct FileVectorAggregate {
    file_id: i64,
    path: String,
    sample: String,
    vector_sum: Vec<f32>,
    count: usize,
}

impl FileVectorAggregate {
    fn new(file_id: i64, path: &str) -> Self {
        Self {
            file_id,
            path: path.to_string(),
            sample: String::new(),
            vector_sum: Vec::new(),
            count: 0,
        }
    }

    fn add_chunk(&mut self, text: &str, vector: &[f32], snippet_chars: usize) {
        if self.vector_sum.is_empty() {
            self.vector_sum.resize(vector.len(), 0.0);
        }
        for (sum, value) in self.vector_sum.iter_mut().zip(vector.iter()) {
            *sum += value;
        }
        self.count += 1;

        if self.sample.chars().count() < snippet_chars {
            if !self.sample.is_empty() {
                self.sample.push('\n');
            }
            self.sample.push_str(&snippet(text, snippet_chars));
            self.sample = snippet(&self.sample, snippet_chars);
        }
    }

    fn centroid(&self) -> Vec<f32> {
        if self.count == 0 {
            return Vec::new();
        }
        self.vector_sum
            .iter()
            .map(|value| value / self.count as f32)
            .collect()
    }
}

fn query_chunks(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
    snippet_chars: usize,
    profile_hash: &str,
) -> Result<Vec<RankedResult>> {
    let Some(fts_query) = fts_query(query) else {
        return Ok(Vec::new());
    };
    let mut stmt = conn.prepare(
        "SELECT f.id, c.id, f.path, c.chunk_index, c.start_line, c.end_line, c.text
         FROM chunks_fts
         JOIN chunks c ON c.id = chunks_fts.rowid
         JOIN files f ON f.id = c.file_id
         WHERE chunks_fts MATCH ?1
         ORDER BY bm25(chunks_fts), f.path, c.chunk_index
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![fts_query, limit.saturating_mul(4).max(limit) as i64],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, String>(6)?,
            ))
        },
    )?;

    let mut results = Vec::new();
    for row in rows {
        let (file_id, chunk_id, path, chunk_index, start_line, end_line, text) = row?;
        let file_type = file_type(&path);
        let keyword_score = ranking::keyword_score(query, &format!("{path}\n{text}"));
        let metadata_score = ranking::keyword_score(query, &path);
        let score = ranking::hybrid_score(
            0.0,
            keyword_score,
            metadata_score,
            ranking::ScoreWeights {
                vector: 0.0,
                keyword: 0.85,
                metadata: 0.15,
            },
        );
        results.push(RankedResult {
            path,
            snippet: snippet(&text, snippet_chars),
            score,
            rerank_score: None,
            kind: "chunk".into(),
            file_type,
            file_id: Some(file_id),
            chunk_id: Some(chunk_id),
            chunk_index: Some(chunk_index),
            start_line,
            end_line,
            keyword_score,
            vector_score: 0.0,
            metadata_score,
            profile_hash: profile_hash.to_string(),
            mode: "keyword".into(),
            level: "chunk".into(),
        });
    }
    Ok(results)
}

fn query_files(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
    snippet_chars: usize,
    profile_hash: &str,
) -> Result<Vec<RankedResult>> {
    let Some(fts_query) = fts_query(query) else {
        return Ok(Vec::new());
    };
    let mut stmt = conn.prepare(
        "WITH matches AS (
           SELECT f.id, f.path, f.content, c.text
           FROM chunks_fts
           JOIN chunks c ON c.id = chunks_fts.rowid
           JOIN files f ON f.id = c.file_id
           WHERE chunks_fts MATCH ?1
         ),
         ranked AS (
           SELECT id, path, content, group_concat(text, char(10)) AS matched_text
           FROM matches
           GROUP BY id, path, content
         )
         SELECT id, path, COALESCE(content, matched_text)
         FROM ranked
         ORDER BY path
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;

    let mut results = Vec::new();
    for row in rows {
        let (file_id, path, content) = row?;
        let file_type = file_type(&path);
        let content = content.unwrap_or_default();
        let keyword_score = ranking::keyword_score(query, &format!("{path}\n{content}"));
        let metadata_score = ranking::keyword_score(query, &path);
        let score = ranking::hybrid_score(
            0.0,
            keyword_score,
            metadata_score,
            ranking::ScoreWeights {
                vector: 0.0,
                keyword: 0.85,
                metadata: 0.15,
            },
        );
        results.push(RankedResult {
            path,
            snippet: snippet(&content, snippet_chars),
            score,
            rerank_score: None,
            kind: "file".into(),
            file_type,
            file_id: Some(file_id),
            chunk_id: None,
            chunk_index: None,
            start_line: None,
            end_line: None,
            keyword_score,
            vector_score: 0.0,
            metadata_score,
            profile_hash: profile_hash.to_string(),
            mode: "keyword".into(),
            level: "file".into(),
        });
    }
    Ok(results)
}

fn query_images_by_path(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
    profile_hash: &str,
) -> Result<Vec<RankedResult>> {
    let mut stmt = conn.prepare(
        "SELECT i.id, f.id, f.path, f.file_type
         FROM images i
         JOIN files f ON f.id = i.file_id
         WHERE lower(f.path) LIKE '%' || lower(?1) || '%'
         ORDER BY f.path
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![query, limit.saturating_mul(8).max(limit) as i64],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        },
    )?;
    let mut top = ranking::TopK::new(limit.saturating_mul(2).max(limit));
    for row in rows {
        let (image_id, file_id, path, file_type) = row?;
        let metadata_score = ranking::keyword_score(query, &path);
        if metadata_score == 0.0 {
            continue;
        }
        top.push(RankedResult {
            path: path.clone(),
            snippet: path.clone(),
            score: metadata_score,
            rerank_score: None,
            kind: "image".into(),
            file_type,
            file_id: Some(file_id),
            chunk_id: Some(image_id),
            chunk_index: None,
            start_line: None,
            end_line: None,
            keyword_score: 0.0,
            vector_score: 0.0,
            metadata_score,
            profile_hash: profile_hash.to_string(),
            mode: "keyword".into(),
            level: "image".into(),
        });
    }
    Ok(top.into_sorted_vec())
}

fn image_vector_results(
    conn: &rusqlite::Connection,
    config: &crate::config::Config,
    query: &str,
    limit: usize,
    mode: &SearchMode,
) -> Result<Vec<RankedResult>> {
    let endpoint = config
        .image
        .embedding
        .endpoint
        .as_deref()
        .filter(|endpoint| !endpoint.trim().is_empty())
        .context("image.embedding.endpoint is required when image embeddings are enabled")?;
    let image_profile_id = crate::db::upsert_image_profile(
        conn,
        &config.image.embedding.model,
        Some(endpoint),
        config.image.embedding.dimensions,
        config.image.embedding.normalize,
    )?;
    let has_vectors =
        !crate::db::vector_images_for_profile(conn, image_profile_id, 1, 0)?.is_empty();
    if !has_vectors {
        return Ok(Vec::new());
    }
    let _ = (query, limit, mode);
    anyhow::bail!(
        "image vector search requires a compatible text-to-image query embedding endpoint, which is not configured"
    )
}

fn snippet(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

struct SearchFilters {
    kind: SearchKindArg,
    filetypes: HashSet<String>,
    paths: GlobSet,
    has_path_filters: bool,
}

impl SearchFilters {
    fn from_args(args: &SearchArgs) -> Result<Self> {
        let filetypes = args
            .filetypes
            .iter()
            .map(normalize_filetype)
            .filter(|filetype| !filetype.is_empty())
            .collect::<HashSet<_>>();
        let mut builder = GlobSetBuilder::new();
        for path in &args.paths {
            builder.add(Glob::new(path).with_context(|| format!("parsing --path glob {path}"))?);
        }
        Ok(Self {
            kind: args.kind.clone(),
            filetypes,
            paths: builder.build()?,
            has_path_filters: !args.paths.is_empty(),
        })
    }

    fn apply(&self, results: Vec<RankedResult>) -> Vec<RankedResult> {
        results
            .into_iter()
            .filter(|result| match self.kind {
                SearchKindArg::All => true,
                SearchKindArg::Text => result.kind != "image",
                SearchKindArg::Image => result.kind == "image",
            })
            .filter(|result| {
                self.filetypes.is_empty()
                    || self
                        .filetypes
                        .contains(&normalize_filetype(&result.file_type))
            })
            .filter(|result| !self.has_path_filters || self.paths.is_match(&result.path))
            .collect()
    }
}

fn maybe_rerank(
    config: &crate::config::Config,
    query: &str,
    mut results: Vec<RankedResult>,
    warnings: &mut Vec<String>,
) -> Result<Vec<RankedResult>> {
    if !config.reranker.enabled || results.is_empty() {
        return Ok(results);
    }
    let candidate_limit = config.reranker.candidate_limit.min(results.len());
    let texts = results
        .iter()
        .take(candidate_limit)
        .map(|result| {
            if result.kind == "image" {
                format!("image path: {}", result.path)
            } else {
                format!("{}\n{}", result.path, result.snippet)
            }
        })
        .collect::<Vec<_>>();
    let provider = crate::providers::RerankerProvider::from_config(config)?;
    let reranked = match provider.rerank(query, texts) {
        Ok(reranked) => reranked,
        Err(err) => {
            warnings.push(format!("reranker disabled for this query: {}", err));
            return Ok(results);
        }
    };
    let mut reordered = Vec::new();
    let mut used = HashSet::new();
    for item in reranked {
        if item.index >= candidate_limit || !used.insert(item.index) {
            continue;
        }
        let mut result = results[item.index].clone();
        result.rerank_score = Some(item.score);
        result.score = item.score;
        reordered.push(result);
    }
    for (idx, result) in results.drain(..).enumerate() {
        if !used.contains(&idx) {
            reordered.push(result);
        }
    }
    Ok(reordered)
}

fn file_type(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(normalize_filetype)
        .unwrap_or_default()
}

fn normalize_filetype(filetype: impl AsRef<str>) -> String {
    filetype
        .as_ref()
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
}

fn fts_query(query: &str) -> Option<String> {
    let terms = query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

fn weighted_score(
    mode: &SearchMode,
    config: &crate::config::Config,
    vector_score: f32,
    keyword_score: f32,
    metadata_score: f32,
) -> f32 {
    match mode {
        SearchMode::Vector => vector_score.clamp(0.0, 1.0),
        SearchMode::Keyword => ranking::hybrid_score(
            0.0,
            keyword_score,
            metadata_score,
            ranking::ScoreWeights {
                vector: 0.0,
                keyword: 0.85,
                metadata: 0.15,
            },
        ),
        SearchMode::Hybrid => ranking::hybrid_score(
            vector_score,
            keyword_score,
            metadata_score,
            ranking::ScoreWeights {
                vector: config.search.vector_weight,
                keyword: config.search.keyword_weight,
                metadata: config.search.metadata_weight,
            },
        ),
    }
}

fn top_results(
    results: Vec<RankedResult>,
    limit: usize,
    max_chunks_per_file: usize,
) -> Vec<RankedResult> {
    let mut deduped: HashMap<(Option<i64>, Option<i64>, String), RankedResult> = HashMap::new();
    for result in results {
        let key = (result.file_id, result.chunk_id, result.level.clone());
        match deduped.get_mut(&key) {
            Some(existing) if result.score > existing.score => {
                *existing = result;
            }
            Some(_) => {}
            None => {
                deduped.insert(key, result);
            }
        }
    }
    let results = deduped.into_values().collect::<Vec<_>>();
    let candidate_limit = limit.saturating_mul(max_chunks_per_file.max(1)).max(limit);
    let candidates = ranking::top_k(results, candidate_limit);
    let mut chunk_counts: HashMap<i64, usize> = HashMap::new();
    let filtered = candidates
        .into_iter()
        .filter(|result| {
            if result.level != "chunk" {
                return true;
            }
            let Some(file_id) = result.file_id else {
                return true;
            };
            let count = chunk_counts.entry(file_id).or_insert(0);
            if *count >= max_chunks_per_file {
                return false;
            }
            *count += 1;
            true
        })
        .collect::<Vec<_>>();
    ranking::top_k(filtered, limit)
}

fn mode_label(mode: &SearchMode) -> &'static str {
    match mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::Vector => "vector",
        SearchMode::Keyword => "keyword",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    fn result(path: &str, score: f32) -> RankedResult {
        RankedResult {
            path: path.into(),
            snippet: format!("{path} snippet"),
            score,
            rerank_score: None,
            kind: "chunk".into(),
            file_type: "md".into(),
            file_id: None,
            chunk_id: None,
            chunk_index: None,
            start_line: None,
            end_line: None,
            keyword_score: score,
            vector_score: 0.0,
            metadata_score: 0.0,
            profile_hash: "profile".into(),
            mode: "keyword".into(),
            level: "chunk".into(),
        }
    }

    fn reranker_config(endpoint: String) -> crate::config::Config {
        let mut config = crate::config::Config::default();
        config.reranker.enabled = true;
        config.reranker.endpoint = Some(endpoint);
        config.reranker.candidate_limit = 3;
        config
    }

    fn serve_once(
        response_status: &str,
        response_body: &'static str,
    ) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/rerank", listener.local_addr().unwrap());
        let response_status = response_status.to_string();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&request);
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .or_else(|| {
                    headers
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                })
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body_start = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
                .unwrap_or(request.len());
            while request.len().saturating_sub(body_start) < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 {response_status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&request[body_start..]).to_string()
        });
        (endpoint, handle)
    }

    #[test]
    fn rerank_reorders_results_and_ignores_duplicate_or_out_of_range_indices() {
        let (endpoint, handle) = serve_once(
            "200 OK",
            r#"[{"index":1,"score":0.9},{"index":1,"score":0.8},{"index":99,"score":1.0}]"#,
        );
        let config = reranker_config(endpoint);
        let mut warnings = Vec::new();
        let results = vec![
            result("first.md", 0.1),
            result("second.md", 0.2),
            result("third.md", 0.3),
        ];

        let reranked = maybe_rerank(&config, "query", results, &mut warnings).unwrap();
        let request_body = handle.join().unwrap();

        assert!(request_body.contains("\"texts\""));
        assert!(!request_body.contains("\"documents\""));
        assert!(warnings.is_empty());
        assert_eq!(reranked[0].path, "second.md");
        assert_eq!(reranked[0].rerank_score, Some(0.9));
        assert_eq!(reranked[1].path, "first.md");
        assert_eq!(reranked[2].path, "third.md");
    }

    #[test]
    fn rerank_errors_warn_and_preserve_original_order() {
        let (endpoint, handle) = serve_once("500 Internal Server Error", r#"{"error":"down"}"#);
        let config = reranker_config(endpoint);
        let mut warnings = Vec::new();
        let results = vec![result("first.md", 0.1), result("second.md", 0.2)];

        let reranked = maybe_rerank(&config, "query", results, &mut warnings).unwrap();
        handle.join().unwrap();

        assert_eq!(reranked[0].path, "first.md");
        assert_eq!(reranked[1].path, "second.md");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("reranker disabled for this query"));
    }
}
