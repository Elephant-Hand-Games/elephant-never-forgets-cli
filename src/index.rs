use std::{
    collections::HashSet,
    fs,
    io::{self, Write},
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Transaction};
use serde::Serialize;

use crate::{
    cli::{IndexArgs, RemoveArgs},
    config::{Config, Provider},
    db, discovery, extract, models, providers,
};

pub fn run(args: IndexArgs) -> Result<()> {
    let mut config = crate::config::load()?;
    crate::config::apply_provider_overrides(&mut config, &args.provider);
    crate::config::validate(&config)?;
    let cwd = std::env::current_dir()?;
    if args.dry_run {
        return dry_run_index(&cwd, &args, &config);
    }
    let summary = index_path_with_options(
        &cwd,
        args.path,
        &config,
        EmbedOptions {
            install_models: args.install_models,
            no_embed: args.no_embed,
            reembed: args.reembed,
            changed_only: args.changed_only,
            quiet: args.json,
        },
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }
    Ok(())
}

pub fn run_add(args: IndexArgs) -> Result<()> {
    run(args)
}

pub fn run_remove(args: RemoveArgs) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    if args.dry_run {
        let normalized = normalize_index_path(&cwd, &args.path)?;
        println!("Dry run: would remove {normalized} from index");
        return Ok(());
    }
    remove_indexed_path(&cwd, args.path, &config)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EmbedOptions {
    pub install_models: bool,
    pub no_embed: bool,
    pub reembed: bool,
    pub changed_only: bool,
    pub quiet: bool,
}

impl From<bool> for EmbedOptions {
    fn from(install_models: bool) -> Self {
        Self {
            install_models,
            no_embed: false,
            reembed: false,
            changed_only: false,
            quiet: false,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct IndexSummary {
    pub path: String,
    pub discovered_files: usize,
    pub changed_files: usize,
    pub embedded_chunks: usize,
    pub reembedded: bool,
    pub changed_only: bool,
    pub no_embed: bool,
}

#[derive(Debug, Serialize)]
struct IndexDryRunSummary {
    path: String,
    discovered_files: usize,
    provider: String,
    would_embed: bool,
    would_install_models: bool,
    reembed: bool,
    changed_only: bool,
    no_embed: bool,
}

fn dry_run_index(root: &Path, args: &IndexArgs, config: &Config) -> Result<()> {
    let target = root.join(&args.path);
    let discovered = discovery::discover(root, &target, config)?;
    let summary = IndexDryRunSummary {
        path: display_index_path(&args.path),
        discovered_files: discovered.len(),
        provider: config.embedding.provider.as_str().to_string(),
        would_embed: !args.no_embed,
        would_install_models: args.install_models,
        reembed: args.reembed,
        changed_only: args.changed_only,
        no_embed: args.no_embed,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("Dry run: would index {}", summary.path);
        println!("  discovered files: {}", summary.discovered_files);
        println!("  provider: {}", summary.provider);
        println!("  would embed: {}", summary.would_embed);
        println!("  would install models: {}", summary.would_install_models);
        println!("  reembed: {}", summary.reembed);
        println!("  changed only: {}", summary.changed_only);
    }
    Ok(())
}

pub fn index_path(
    root: &Path,
    path: PathBuf,
    config: &Config,
    embed_options: impl Into<EmbedOptions>,
) -> Result<()> {
    index_path_with_options(root, path, config, embed_options).map(|_| ())
}

pub fn index_path_with_options(
    root: &Path,
    path: PathBuf,
    config: &Config,
    embed_options: impl Into<EmbedOptions>,
) -> Result<IndexSummary> {
    let embed_options = embed_options.into();
    let db_path = root.join(&config.state.db_path);
    let mut conn = db::open_or_create(&db_path)?;

    ensure_embedding_ready(root, config, embed_options)?;

    let tx = conn.transaction()?;

    let target = root.join(&path);
    print_progress(
        embed_options,
        format_args!("==> Discovering files under {}", display_index_path(&path)),
    );
    let discovered = discovery::discover(root, &target, config)?;
    let discovered_count = discovered.len();
    print_progress(
        embed_options,
        format_args!("==> Indexing {discovered_count} files"),
    );
    let discovered_paths: HashSet<String> = discovered
        .iter()
        .map(|file| file.relative_path.clone())
        .collect();
    let scope_prefix = scope_prefix(&path);

    remove_missing_files(&tx, &scope_prefix, &discovered_paths)?;

    let mut changed = 0usize;
    let mut changed_paths = Vec::new();
    for file in &discovered {
        let file_changed = sync_file(&tx, file, config)?;
        changed += file_changed;
        if file_changed > 0 {
            changed_paths.push(file.relative_path.clone());
        }
    }

    tx.commit()?;
    print_progress(
        embed_options,
        format_args!("✓ Indexed {discovered_count} files ({changed} changed)"),
    );

    let reembed_paths = if embed_options.changed_only {
        changed_paths
    } else {
        discovered_paths.iter().cloned().collect::<Vec<_>>()
    };
    let embedded = maybe_embed_missing_chunks(&conn, config, embed_options, &reembed_paths)?;
    if embedded > 0 {
        print_progress(embed_options, format_args!("✓ Embedded {embedded} chunks"));
    }
    Ok(IndexSummary {
        path: display_index_path(&path),
        discovered_files: discovered_count,
        changed_files: changed,
        embedded_chunks: embedded,
        reembedded: embed_options.reembed,
        changed_only: embed_options.changed_only,
        no_embed: embed_options.no_embed,
    })
}

pub fn remove_indexed_path(root: &Path, path: PathBuf, config: &Config) -> Result<()> {
    let db_path = root.join(&config.state.db_path);
    let mut conn = db::open_or_create(&db_path)?;
    let tx = conn.transaction()?;
    let normalized = normalize_index_path(root, &path)?;
    let deleted = delete_indexed_path(&tx, &normalized)?;
    tx.commit()?;
    if deleted {
        println!("✓ Removed {normalized} from index");
    } else {
        println!("warning: {normalized} was not present in the index");
    }
    Ok(())
}

fn maybe_embed_missing_chunks(
    conn: &rusqlite::Connection,
    config: &Config,
    embed_options: EmbedOptions,
    reembed_paths: &[String],
) -> Result<usize> {
    if embed_options.no_embed {
        return Ok(0);
    }

    let mut provider = providers::build_provider(config)?;
    let profile = provider.profile();
    let profile_id = db::upsert_embedding_profile(conn, &profile)?;
    if embed_options.reembed {
        delete_embeddings_for_paths(conn, profile_id, reembed_paths)?;
    }
    let total_missing = missing_embedding_count(conn, profile_id)?;
    if total_missing == 0 {
        return Ok(0);
    }

    if config.embedding.provider == Provider::Native {
        print_progress(
            embed_options,
            format_args!("==> Loading native embedding model"),
        );
        provider.ensure_ready()?;
    }

    let batch_size = effective_embedding_batch_size(config);
    print_progress(
        embed_options,
        format_args!("==> Embedding {total_missing} chunks (batch size {batch_size})"),
    );
    let mut embedded = 0usize;

    loop {
        let chunks = db::chunks_missing_embeddings(conn, profile_id, batch_size)?;
        if chunks.is_empty() {
            break;
        }
        let batch_start = embedded + 1;
        let batch_end = embedded + chunks.len();
        print_progress(
            embed_options,
            format_args!("==> Embedding chunks {batch_start}-{batch_end} of {total_missing}"),
        );
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
        print_progress(
            embed_options,
            format_args!("✓ Embedded {embedded}/{total_missing} chunks"),
        );
    }

    Ok(embedded)
}

fn delete_embeddings_for_paths(
    conn: &rusqlite::Connection,
    profile_id: i64,
    paths: &[String],
) -> Result<()> {
    for path in paths {
        conn.execute(
            "DELETE FROM embeddings
             WHERE profile_id = ?1
               AND chunk_id IN (
                 SELECT c.id
                 FROM chunks c
                 JOIN files f ON f.id = c.file_id
                 WHERE f.path = ?2
               )",
            params![profile_id, path],
        )?;
    }
    Ok(())
}

fn effective_embedding_batch_size(config: &Config) -> usize {
    if config.embedding.provider == Provider::Native {
        config
            .embedding
            .batch_size
            .clamp(1, native_embedding_batch_cap())
    } else {
        config.embedding.batch_size.max(1)
    }
}

fn native_embedding_batch_cap() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !std::is_x86_feature_detected!("avx") {
            return 1;
        }
    }
    8
}

fn missing_embedding_count(conn: &rusqlite::Connection, profile_id: i64) -> Result<usize> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*)
         FROM chunks c
         LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.profile_id = ?1
         WHERE e.id IS NULL",
        [profile_id],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

fn ensure_embedding_ready(root: &Path, config: &Config, embed_options: EmbedOptions) -> Result<()> {
    if embed_options.no_embed || config.embedding.provider != Provider::Native {
        return Ok(());
    }

    let global_cache_dir = dirs::cache_dir();
    if embed_options.install_models {
        models::install_active_model_in(config, root, global_cache_dir.as_deref())?;
        return Ok(());
    }

    if !models::is_active_model_installed_in(config, root, global_cache_dir.as_deref())? {
        anyhow::bail!(
            "native embedding model is not installed.\nRun:\n  enf models install\n\nOr initialize a new native project with:\n  enf init --db"
        );
    }

    Ok(())
}

fn display_index_path(path: &Path) -> String {
    if path.as_os_str().is_empty() {
        ".".into()
    } else {
        path.display().to_string()
    }
}

fn flush_stdout() {
    let _ = io::stdout().flush();
}

fn print_progress(options: EmbedOptions, args: std::fmt::Arguments<'_>) {
    if !options.quiet {
        println!("{args}");
        flush_stdout();
    }
}

fn sync_file(
    tx: &Transaction<'_>,
    file: &discovery::DiscoveredFile,
    config: &Config,
) -> Result<usize> {
    let extracted = extract::extract_text(&file.absolute_path)
        .with_context(|| format!("extracting {}", file.absolute_path.display()))?;
    let file_hash = blake3::hash(extracted.text.as_bytes()).to_hex().to_string();
    let metadata = fs::metadata(&file.absolute_path)
        .with_context(|| format!("reading metadata for {}", file.absolute_path.display()))?;
    let size_bytes = metadata.len() as i64;
    let modified_at = metadata
        .modified()
        .ok()
        .map(|time| DateTime::<Utc>::from(time).to_rfc3339());
    let file_type = file_type(&file.relative_path);
    let content = if config.index.store_full_files {
        Some(extracted.text.clone())
    } else {
        None
    };

    if let Some(existing) = load_file(tx, &file.relative_path)? {
        if existing.hash == file_hash
            && existing.size_bytes == size_bytes
            && existing.file_type == file_type
            && existing.modified_at == modified_at
            && existing.content == content
        {
            if !config.index.store_chunks && file_has_chunks(tx, existing.id)? {
                delete_chunks(tx, existing.id, &file.relative_path)?;
                return Ok(1);
            }
            return Ok(0);
        }
        replace_indexed_file(
            tx,
            existing.id,
            FileUpdate {
                path: &file.relative_path,
                file_type: &file_type,
                hash: &file_hash,
                size_bytes,
                modified_at: modified_at.as_deref(),
                content,
            },
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
        "INSERT INTO files(path, file_type, hash, size_bytes, modified_at, indexed_at, content)
         VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'), ?6)",
        params![
            file.relative_path,
            file_type,
            file_hash,
            size_bytes,
            modified_at.as_deref(),
            content
        ],
    )?;
    let file_id = tx.last_insert_rowid();
    insert_chunks(tx, file_id, &file.relative_path, &extracted.text, config)?;
    Ok(1)
}

fn file_has_chunks(tx: &Transaction<'_>, file_id: i64) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM chunks WHERE file_id = ?1)",
        [file_id],
        |row| row.get(0),
    )?)
}

fn load_file(tx: &Transaction<'_>, path: &str) -> Result<Option<FileRecord>> {
    Ok(tx
        .query_row(
            "SELECT id, hash, size_bytes, content, file_type, modified_at FROM files WHERE path = ?1",
            [path],
            |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    hash: row.get(1)?,
                    size_bytes: row.get(2)?,
                    content: row.get(3)?,
                    file_type: row.get(4)?,
                    modified_at: row.get(5)?,
                })
            },
        )
        .optional()?)
}

fn replace_indexed_file(tx: &Transaction<'_>, file_id: i64, update: FileUpdate<'_>) -> Result<()> {
    tx.execute(
        "UPDATE files
         SET path = ?2, file_type = ?3, hash = ?4, size_bytes = ?5, modified_at = ?6, indexed_at = datetime('now'), content = ?7
         WHERE id = ?1",
        params![
            file_id,
            update.path,
            update.file_type,
            update.hash,
            update.size_bytes,
            update.modified_at,
            update.content
        ],
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
    delete_chunks(tx, file_id, &path)?;
    tx.execute("DELETE FROM files WHERE id = ?1", [file_id])?;
    Ok(())
}

fn delete_indexed_path(tx: &Transaction<'_>, path: &str) -> Result<bool> {
    let file_id = tx
        .query_row("SELECT id FROM files WHERE path = ?1", [path], |row| {
            row.get::<_, i64>(0)
        })
        .optional()?;
    if let Some(file_id) = file_id {
        delete_indexed_file(tx, file_id)?;
        Ok(true)
    } else {
        Ok(false)
    }
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

fn normalize_index_path(root: &Path, path: &Path) -> Result<String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let relative = absolute
        .strip_prefix(root)
        .with_context(|| format!("{} is outside {}", absolute.display(), root.display()))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

#[derive(Debug, Clone)]
struct FileRecord {
    id: i64,
    hash: String,
    size_bytes: i64,
    file_type: String,
    modified_at: Option<String>,
    content: Option<String>,
}

struct FileUpdate<'a> {
    path: &'a str,
    file_type: &'a str,
    hash: &'a str,
    size_bytes: i64,
    modified_at: Option<&'a str>,
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

fn file_type(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_default()
}
