use crate::{
    cli::SearchArgs,
    errors::EnfError,
    ranking::{self, RankedResult},
};
use anyhow::Result;

pub fn run(args: SearchArgs, retrieve: bool) -> Result<()> {
    let config = crate::config::load()?;
    let profile = crate::embed::active_profile(&config);
    let cwd = std::env::current_dir()?;
    let conn = crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let profile_id = crate::db::upsert_embedding_profile(&conn, &profile)?;
    let normalized_query = ranking::normalize_query(&args.query);

    if args.cached_query_only
        && crate::db::query_embedding(&conn, profile_id, &normalized_query)?.is_none()
    {
        return Err(EnfError::QueryEmbeddingNotCached.into());
    }

    let limit = args.limit.unwrap_or(config.search.limit);
    let mut results = query_chunks(
        &conn,
        &args.query,
        limit,
        config.search.snippet_chars,
        &profile.profile_hash,
    )?;
    let mut vector_results = vector_results(VectorSearch {
        conn: &conn,
        config: &config,
        profile_id,
        profile_hash: &profile.profile_hash,
        query: &args.query,
        normalized_query: &normalized_query,
        cached_query_only: args.cached_query_only,
        limit,
    })?;
    results.append(&mut vector_results);
    if results.is_empty() {
        results = query_files(
            &conn,
            &args.query,
            limit,
            config.search.snippet_chars,
            &profile.profile_hash,
        )?;
    }
    let results = ranking::top_k(results, limit);

    if args.json || retrieve {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "query": args.query,
                "profile_hash": profile.profile_hash,
                "results": results,
            }))?
        );
    } else {
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

struct VectorSearch<'a> {
    conn: &'a rusqlite::Connection,
    config: &'a crate::config::Config,
    profile_id: i64,
    profile_hash: &'a str,
    query: &'a str,
    normalized_query: &'a str,
    cached_query_only: bool,
    limit: usize,
}

fn vector_results(search: VectorSearch<'_>) -> Result<Vec<RankedResult>> {
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
                let score = ranking::hybrid_score(
                    vector_score,
                    keyword_score,
                    metadata_score,
                    ranking::ScoreWeights {
                        vector: search.config.search.vector_weight,
                        keyword: search.config.search.keyword_weight,
                        metadata: search.config.search.metadata_weight,
                    },
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
                    mode: "hybrid".into(),
                    level: "chunk".into(),
                });
            }
            Ok(())
        },
    )?;
    Ok(top.into_sorted_vec())
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
