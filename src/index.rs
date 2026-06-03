use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result};
use rusqlite::{params, OptionalExtension, Transaction};

use crate::{cli::IndexArgs, config::Config, db, discovery, extract, models, providers};

pub fn run(args: IndexArgs) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    index_path(&cwd, args.path, &config, args.install_models)
}

pub fn index_path(root: &Path, path: PathBuf, config: &Config, install_models: bool) -> Result<()> {
    let db_path = root.join(&config.state.db_path);
    let mut conn = db::open_or_create(&db_path)?;
    let tx = conn.transaction()?;

    let target = root.join(&path);
    let discovered = discovery::discover(root, &target, config)?;
    let discovered_paths: HashSet<String> = discovered
        .iter()
        .map(|file| file.relative_path.clone())
        .collect();
    let scope_prefix = scope_prefix(&path);

    remove_missing_files(&tx, &scope_prefix, &discovered_paths)?;

    let mut indexed = 0usize;
    for file in discovered {
        indexed += sync_file(&tx, &file, config)?;
    }

    tx.commit()?;

    let embedded = maybe_embed_missing_chunks(root, &conn, config, install_models)?;
    println!("Indexed {indexed} files");
    if embedded > 0 {
        println!("Embedded {embedded} chunks");
    }
    Ok(())
}

fn maybe_embed_missing_chunks(
    root: &Path,
    conn: &rusqlite::Connection,
    config: &Config,
    install_models: bool,
) -> Result<usize> {
    let global_cache_dir = dirs::cache_dir();
    if install_models {
        models::install_active_model_in(config, root, global_cache_dir.as_deref())?;
    }
    if !install_models
        && !models::is_active_model_installed_in(config, root, global_cache_dir.as_deref())?
    {
        return Ok(0);
    }

    let mut provider = providers::build_provider(config)?;
    let profile = provider.profile();
    let profile_id = db::upsert_embedding_profile(conn, &profile)?;
    let mut embedded = 0usize;

    loop {
        let chunks = db::chunks_missing_embeddings(conn, profile_id, config.embedding.batch_size)?;
        if chunks.is_empty() {
            break;
        }
        let texts = chunks
            .iter()
            .map(|chunk| chunk.text.clone())
            .collect::<Vec<_>>();
        let vectors = provider.embed_documents(&texts)?;
        if vectors.len() != chunks.len() {
            anyhow::bail!(
                "embedding provider returned {} vectors for {} chunks",
                vectors.len(),
                chunks.len()
            );
        }
        for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
            db::upsert_chunk_embedding(conn, profile_id, chunk.chunk_id, vector)?;
            embedded += 1;
        }
    }

    Ok(embedded)
}

fn sync_file(
    tx: &Transaction<'_>,
    file: &discovery::DiscoveredFile,
    config: &Config,
) -> Result<usize> {
    let extracted = extract::extract_text(&file.absolute_path)
        .with_context(|| format!("extracting {}", file.absolute_path.display()))?;
    let file_hash = blake3::hash(extracted.text.as_bytes()).to_hex().to_string();
    let size_bytes = fs::metadata(&file.absolute_path)
        .with_context(|| format!("reading metadata for {}", file.absolute_path.display()))?
        .len() as i64;
    let content = if config.index.store_full_files {
        Some(extracted.text.clone())
    } else {
        None
    };

    if let Some(existing) = load_file(tx, &file.relative_path)? {
        if existing.hash == file_hash
            && existing.size_bytes == size_bytes
            && existing.content == content
        {
            return Ok(0);
        }
        replace_indexed_file(
            tx,
            existing.id,
            &file.relative_path,
            &file_hash,
            size_bytes,
            content,
        )?;
        replace_chunks(
            tx,
            existing.id,
            &file.relative_path,
            &extracted.text,
            config,
        )?;
        return Ok(1);
    }

    tx.execute(
        "INSERT INTO files(path, hash, size_bytes, indexed_at, content)
         VALUES (?1, ?2, ?3, datetime('now'), ?4)",
        params![file.relative_path, file_hash, size_bytes, content],
    )?;
    let file_id = tx.last_insert_rowid();
    insert_chunks(tx, file_id, &file.relative_path, &extracted.text, config)?;
    Ok(1)
}

fn load_file(tx: &Transaction<'_>, path: &str) -> Result<Option<FileRecord>> {
    Ok(tx
        .query_row(
            "SELECT id, hash, size_bytes, content FROM files WHERE path = ?1",
            [path],
            |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    hash: row.get(1)?,
                    size_bytes: row.get(2)?,
                    content: row.get(3)?,
                })
            },
        )
        .optional()?)
}

fn replace_indexed_file(
    tx: &Transaction<'_>,
    file_id: i64,
    path: &str,
    hash: &str,
    size_bytes: i64,
    content: Option<String>,
) -> Result<()> {
    tx.execute(
        "UPDATE files
         SET path = ?2, hash = ?3, size_bytes = ?4, indexed_at = datetime('now'), content = ?5
         WHERE id = ?1",
        params![file_id, path, hash, size_bytes, content],
    )?;
    Ok(())
}

fn replace_chunks(
    tx: &Transaction<'_>,
    file_id: i64,
    path: &str,
    text: &str,
    config: &Config,
) -> Result<()> {
    delete_chunks(tx, file_id, path)?;
    insert_chunks(tx, file_id, path, text, config)
}

fn delete_chunks(tx: &Transaction<'_>, file_id: i64, path: &str) -> Result<()> {
    let mut stmt =
        tx.prepare("SELECT id, text FROM chunks WHERE file_id = ?1 ORDER BY chunk_index")?;
    let rows = stmt.query_map([file_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut chunk_ids = Vec::new();
    for row in rows {
        chunk_ids.push(row?);
    }
    for (chunk_id, text) in chunk_ids {
        tx.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, path, text) VALUES ('delete', ?1, ?2, ?3)",
            params![chunk_id, path, text],
        )?;
    }
    tx.execute("DELETE FROM chunks WHERE file_id = ?1", [file_id])?;
    Ok(())
}

fn insert_chunks(
    tx: &Transaction<'_>,
    file_id: i64,
    path: &str,
    text: &str,
    config: &Config,
) -> Result<()> {
    if !config.index.store_chunks {
        return Ok(());
    }

    for (chunk_index, chunk) in chunk_text(text, config).into_iter().enumerate() {
        tx.execute(
            "INSERT INTO chunks(file_id, chunk_index, hash, text, start_line, end_line, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                file_id,
                chunk_index as i64,
                chunk.hash,
                chunk.text,
                chunk.start_line as i64,
                chunk.end_line as i64,
                chunk.token_count as i64,
            ],
        )?;
        let chunk_id = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO chunks_fts(rowid, path, text) VALUES (?1, ?2, ?3)",
            params![chunk_id, path, chunk.text],
        )?;
    }
    Ok(())
}

fn remove_missing_files(
    tx: &Transaction<'_>,
    scope_prefix: &str,
    discovered_paths: &HashSet<String>,
) -> Result<()> {
    let mut existing = Vec::new();
    let mut stmt = tx.prepare("SELECT id, path FROM files")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        existing.push(row?);
    }

    for (file_id, path) in existing {
        if !is_within_scope(&path, scope_prefix) {
            continue;
        }
        if discovered_paths.contains(&path) {
            continue;
        }
        delete_indexed_file(tx, file_id)?;
    }
    Ok(())
}

fn delete_indexed_file(tx: &Transaction<'_>, file_id: i64) -> Result<()> {
    let path: String = tx.query_row("SELECT path FROM files WHERE id = ?1", [file_id], |row| {
        row.get(0)
    })?;
    let mut stmt =
        tx.prepare("SELECT id, text FROM chunks WHERE file_id = ?1 ORDER BY chunk_index")?;
    let rows = stmt.query_map([file_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut chunk_ids = Vec::new();
    for row in rows {
        chunk_ids.push(row?);
    }
    for (chunk_id, text) in chunk_ids {
        tx.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, path, text) VALUES ('delete', ?1, ?2, ?3)",
            params![chunk_id, path, text],
        )?;
    }
    tx.execute("DELETE FROM files WHERE id = ?1", [file_id])?;
    Ok(())
}

fn is_within_scope(path: &str, scope_prefix: &str) -> bool {
    if scope_prefix.is_empty() {
        return true;
    }
    path == scope_prefix || path.starts_with(&format!("{scope_prefix}/"))
}

fn scope_prefix(path: &Path) -> String {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    let prefix = normalized.to_string_lossy().replace('\\', "/");
    if prefix == "." {
        String::new()
    } else {
        prefix
    }
}

#[derive(Debug, Clone)]
struct FileRecord {
    id: i64,
    hash: String,
    size_bytes: i64,
    content: Option<String>,
}

#[derive(Debug, Clone)]
struct ChunkRecord {
    text: String,
    hash: String,
    start_line: usize,
    end_line: usize,
    token_count: usize,
}

fn chunk_text(text: &str, config: &Config) -> Vec<ChunkRecord> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let (end, chunk_token_count) = chunk_window(&lines, start, config);
        let chunk_text = lines[start..end].join("\n");
        let start_line = start + 1;
        let end_line = end;
        chunks.push(ChunkRecord {
            hash: blake3::hash(chunk_text.as_bytes()).to_hex().to_string(),
            text: chunk_text,
            start_line,
            end_line,
            token_count: chunk_token_count,
        });

        if end >= lines.len() {
            break;
        }
        start = overlap_start(&lines, start, end, config.index.chunk_overlap_tokens);
    }

    chunks
}

fn chunk_window(lines: &[&str], start: usize, config: &Config) -> (usize, usize) {
    let mut end = start;
    let mut token_count = 0usize;
    let target = config.index.chunk_target_tokens.max(1);
    let max_tokens = config.index.chunk_max_tokens.max(target);
    let min_chars = config.index.min_chunk_chars;

    while end < lines.len() {
        let line_tokens = count_tokens(lines[end]);
        let would_exceed_max = end > start && token_count + line_tokens > max_tokens;
        if would_exceed_max {
            break;
        }
        token_count += line_tokens;
        end += 1;
        let chunk_chars = lines[start..end].join("\n").len();
        if token_count >= target && chunk_chars >= min_chars {
            break;
        }
    }

    if end == start {
        end = (start + 1).min(lines.len());
        token_count = count_tokens(lines[start]);
    }

    (end, token_count)
}

fn overlap_start(lines: &[&str], start: usize, end: usize, overlap_tokens: usize) -> usize {
    if overlap_tokens == 0 {
        return end;
    }

    let mut accumulated = 0usize;
    let mut overlap_start = end;
    while overlap_start > start {
        let candidate = overlap_start - 1;
        accumulated += count_tokens(lines[candidate]);
        overlap_start = candidate;
        if accumulated >= overlap_tokens {
            break;
        }
    }

    if overlap_start <= start {
        (start + 1).min(end)
    } else {
        overlap_start
    }
}

fn count_tokens(line: &str) -> usize {
    line.split_whitespace().count()
}
