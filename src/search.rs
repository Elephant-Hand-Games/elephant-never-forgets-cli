use std::collections::HashMap;

use crate::{
    cli::SearchArgs,
    config::{Provider, SearchLevel, SearchMode},
    errors::EnfError,
    ranking::{self, RankedResult},
};
use anyhow::Result;

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
        && crate::db::query_embedding(&conn, profile_id, &normalized_query)?.is_none()
    {
        return Err(EnfError::QueryEmbeddingNotCached.into());
    }

    let limit = args.limit.unwrap_or(config.search.limit);
    let mut results = Vec::new();
    let mut warnings = Vec::new();
    if mode != SearchMode::Vector {
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
    if mode != SearchMode::Keyword {
        let has_vectors = active_profile_has_vectors(&conn, profile_id)?;
        if !has_vectors {
            if mode == SearchMode::Vector && config.embedding.provider == Provider::Native {
                return Err(EnfError::NativeModelMissing.into());
            }
            warnings.push(format!(
                "active profile {} has no indexed embeddings; run `enf index --install-models .` to populate vectors",
                profile.profile_hash
            ));
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
            if let (Some(start), Some(end)) = (result.start_line, result.end_line) {
                println!("{}:{}-{}  {:.3}", result.path, start, end, result.score);
            } else {
                println!("{}  {:.3}", result.path, result.score);
            }
        }
    }
    Ok(())
}

fn active_profile_has_vectors(conn: &rusqlite::Connection, profile_id: i64) -> Result<bool> {
    Ok(!crate::db::vector_chunks_for_profile(conn, profile_id, 1, 0)?.is_empty())
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

    let query_vector = if let Some(vector) =
        crate::db::query_embedding(search.conn, search.profile_id, search.normalized_query)?
    {
        vector
    } else {
        if search.cached_query_only {
            return Err(EnfError::QueryEmbeddingNotCached.into());
        }
        let mut provider = crate::providers::build_provider(search.config)?;
        let vector = provider.embed_query(search.query)?;
        crate::db::upsert_query_embedding(
            search.conn,
            search.profile_id,
            search.normalized_query,
            &vector,
        )?;
        vector
    };

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
                    kind: "chunk".into(),
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

    let query_vector = if let Some(vector) =
        crate::db::query_embedding(search.conn, search.profile_id, search.normalized_query)?
    {
        vector
    } else {
        if search.cached_query_only {
            return Err(EnfError::QueryEmbeddingNotCached.into());
        }
        let mut provider = crate::providers::build_provider(search.config)?;
        let vector = provider.embed_query(search.query)?;
        crate::db::upsert_query_embedding(
            search.conn,
            search.profile_id,
            search.normalized_query,
            &vector,
        )?;
        vector
    };

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
            kind: "file".into(),
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
    let like_query = format!("%{}%", query);
    let mut stmt = conn.prepare(
        "SELECT f.id, c.id, f.path, c.chunk_index, c.start_line, c.end_line, c.text
         FROM chunks c
         JOIN files f ON f.id = c.file_id
         WHERE c.text LIKE ?1 OR f.path LIKE ?1
         ORDER BY f.path, c.chunk_index
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![like_query, limit.saturating_mul(4).max(limit) as i64],
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
            kind: "chunk".into(),
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
    let like_query = format!("%{}%", query);
    let mut stmt = conn.prepare(
        "SELECT id, path, content
         FROM files
         WHERE content LIKE ?1 OR path LIKE ?1
         ORDER BY path
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![like_query, limit as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;

    let mut results = Vec::new();
    for row in rows {
        let (file_id, path, content) = row?;
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
            kind: "file".into(),
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

fn snippet(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
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
