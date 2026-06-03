use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    config::Config,
    embed::{active_profile, deserialize_vector, serialize_vector, EmbeddingProfile},
};

#[derive(Debug, Clone, PartialEq)]
pub struct StoredEmbeddingProfile {
    pub id: i64,
    pub profile: EmbeddingProfile,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct ChunkForEmbedding {
    pub chunk_id: i64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct VectorChunkRow {
    pub chunk_id: i64,
    pub file_id: i64,
    pub path: String,
    pub chunk_index: i64,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub text: String,
    pub vector: Vec<f32>,
}

const MIGRATIONS: &[&str] = &[
    r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  hash TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  modified_at TEXT,
  indexed_at TEXT NOT NULL,
  content TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  hash TEXT NOT NULL,
  text TEXT NOT NULL,
  start_line INTEGER,
  end_line INTEGER,
  token_count INTEGER NOT NULL,
  UNIQUE(file_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embedding_profiles (
  id INTEGER PRIMARY KEY,
  profile_hash TEXT NOT NULL UNIQUE,
  provider TEXT NOT NULL,
  engine TEXT,
  model TEXT NOT NULL,
  variant TEXT,
  endpoint TEXT,
  dimensions INTEGER NOT NULL,
  document_prefix TEXT NOT NULL DEFAULT '',
  query_prefix TEXT NOT NULL DEFAULT '',
  normalizer_version TEXT NOT NULL,
  chunker_version TEXT NOT NULL,
  serialization_version TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY,
  profile_id INTEGER NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
  file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(profile_id, chunk_id),
  UNIQUE(profile_id, file_id)
);

CREATE TABLE IF NOT EXISTS query_embeddings (
  id INTEGER PRIMARY KEY,
  profile_id INTEGER NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
  normalized_query TEXT NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL,
  UNIQUE(profile_id, normalized_query)
);

CREATE TABLE IF NOT EXISTS model_cache (
  id INTEGER PRIMARY KEY,
  provider TEXT NOT NULL,
  engine TEXT,
  model TEXT NOT NULL,
  variant TEXT,
  dimensions INTEGER,
  cache_path TEXT,
  installed_at TEXT,
  last_used_at TEXT,
  status TEXT NOT NULL,
  UNIQUE(provider, engine, model, variant, dimensions, cache_path)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  path,
  text,
  content='',
  tokenize='porter unicode61'
);
"#,
    r#"
ALTER TABLE files ADD COLUMN file_type TEXT NOT NULL DEFAULT '';
"#,
];

pub fn open_or_create(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating database directory {}", parent.display()))?;
    }
    let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    migrate(&conn)?;
    Ok(conn)
}

pub fn migrate(conn: &Connection) -> Result<()> {
    for (idx, migration) in MIGRATIONS.iter().enumerate() {
        let version = (idx + 1) as i64;
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version = ?1)",
                [version],
                |row| row.get(0),
            )
            .unwrap_or(false);
        if exists {
            continue;
        }
        conn.execute_batch(migration)?;
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?1, datetime('now'))",
            [version],
        )?;
    }
    Ok(())
}

pub fn upsert_embedding_profile(conn: &Connection, profile: &EmbeddingProfile) -> Result<i64> {
    conn.execute(
        "INSERT INTO embedding_profiles (
            profile_hash,
            provider,
            engine,
            model,
            variant,
            endpoint,
            dimensions,
            document_prefix,
            query_prefix,
            normalizer_version,
            chunker_version,
            serialization_version,
            created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
        ON CONFLICT(profile_hash) DO UPDATE SET profile_hash = excluded.profile_hash",
        params![
            &profile.profile_hash,
            &profile.provider,
            profile.engine.as_deref(),
            &profile.model,
            profile.variant.as_deref(),
            profile.endpoint.as_deref(),
            profile.dimensions as i64,
            &profile.document_prefix,
            &profile.query_prefix,
            &profile.normalizer_version,
            &profile.chunker_version,
            &profile.serialization_version,
        ],
    )?;
    embedding_profile_id(conn, &profile.profile_hash)
}

pub fn upsert_active_embedding_profile(conn: &Connection, config: &Config) -> Result<i64> {
    let profile = active_profile(config);
    upsert_embedding_profile(conn, &profile)
}

pub fn embedding_profile_id(conn: &Connection, profile_hash: &str) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT id FROM embedding_profiles WHERE profile_hash = ?1",
        [profile_hash],
        |row| row.get(0),
    )?)
}

pub fn get_embedding_profile(
    conn: &Connection,
    profile: &EmbeddingProfile,
) -> Result<Option<StoredEmbeddingProfile>> {
    get_embedding_profile_by_hash(conn, &profile.profile_hash)
}

pub fn get_active_embedding_profile(
    conn: &Connection,
    config: &Config,
) -> Result<Option<StoredEmbeddingProfile>> {
    let profile = active_profile(config);
    get_embedding_profile_by_hash(conn, &profile.profile_hash)
}

pub fn get_embedding_profile_id(
    conn: &Connection,
    profile: &EmbeddingProfile,
) -> Result<Option<i64>> {
    get_embedding_profile(conn, profile).map(|profile| profile.map(|profile| profile.id))
}

pub fn get_active_embedding_profile_id(conn: &Connection, config: &Config) -> Result<Option<i64>> {
    get_active_embedding_profile(conn, config).map(|profile| profile.map(|profile| profile.id))
}

fn get_embedding_profile_by_hash(
    conn: &Connection,
    profile_hash: &str,
) -> Result<Option<StoredEmbeddingProfile>> {
    conn.query_row(
        "SELECT id, profile_hash, provider, engine, model, variant, endpoint, dimensions,
                document_prefix, query_prefix, normalizer_version, chunker_version,
                serialization_version, created_at
         FROM embedding_profiles
         WHERE profile_hash = ?1",
        [profile_hash],
        |row| {
            Ok(StoredEmbeddingProfile {
                id: row.get(0)?,
                profile: EmbeddingProfile {
                    profile_hash: row.get(1)?,
                    provider: row.get(2)?,
                    engine: row.get(3)?,
                    model: row.get(4)?,
                    variant: row.get(5)?,
                    endpoint: row.get(6)?,
                    dimensions: row.get::<_, i64>(7)? as usize,
                    document_prefix: row.get(8)?,
                    query_prefix: row.get(9)?,
                    normalizer_version: row.get(10)?,
                    chunker_version: row.get(11)?,
                    serialization_version: row.get(12)?,
                },
                created_at: row.get(13)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn chunks_missing_embeddings(
    conn: &Connection,
    profile_id: i64,
    limit: usize,
) -> Result<Vec<ChunkForEmbedding>> {
    let mut stmt = conn.prepare(
        "SELECT c.id, c.text
         FROM chunks c
         LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.profile_id = ?1
         WHERE e.id IS NULL
         ORDER BY c.id
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![profile_id, limit as i64], |row| {
        Ok(ChunkForEmbedding {
            chunk_id: row.get(0)?,
            text: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn upsert_chunk_embedding(
    conn: &Connection,
    profile_id: i64,
    chunk_id: i64,
    vector: &[f32],
) -> Result<()> {
    validate_vector_dimensions(conn, profile_id, vector)?;
    conn.execute(
        "INSERT INTO embeddings(profile_id, chunk_id, vector, created_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(profile_id, chunk_id)
         DO UPDATE SET vector = excluded.vector, created_at = excluded.created_at",
        params![profile_id, chunk_id, serialize_vector(vector)],
    )?;
    Ok(())
}

pub fn query_embedding(
    conn: &Connection,
    profile_id: i64,
    normalized_query: &str,
) -> Result<Option<Vec<f32>>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vector FROM query_embeddings WHERE profile_id = ?1 AND normalized_query = ?2",
            params![profile_id, normalized_query],
            |row| row.get(0),
        )
        .optional()?;
    if bytes.is_some() {
        conn.execute(
            "UPDATE query_embeddings SET last_used_at = datetime('now') WHERE profile_id = ?1 AND normalized_query = ?2",
            params![profile_id, normalized_query],
        )?;
    }
    bytes.map(|bytes| deserialize_vector(&bytes)).transpose()
}

pub fn upsert_query_embedding(
    conn: &Connection,
    profile_id: i64,
    normalized_query: &str,
    vector: &[f32],
) -> Result<()> {
    validate_vector_dimensions(conn, profile_id, vector)?;
    conn.execute(
        "INSERT INTO query_embeddings(profile_id, normalized_query, vector, created_at, last_used_at)
         VALUES (?1, ?2, ?3, datetime('now'), datetime('now'))
         ON CONFLICT(profile_id, normalized_query)
         DO UPDATE SET vector = excluded.vector, last_used_at = excluded.last_used_at",
        params![profile_id, normalized_query, serialize_vector(vector)],
    )?;
    Ok(())
}

fn validate_vector_dimensions(conn: &Connection, profile_id: i64, vector: &[f32]) -> Result<()> {
    let expected: i64 = conn.query_row(
        "SELECT dimensions FROM embedding_profiles WHERE id = ?1",
        [profile_id],
        |row| row.get(0),
    )?;
    if vector.len() != expected as usize {
        anyhow::bail!(
            "embedding vector has {} dimensions; expected {} for profile {}",
            vector.len(),
            expected,
            profile_id
        );
    }
    Ok(())
}

pub fn vector_chunks_for_profile(
    conn: &Connection,
    profile_id: i64,
    limit: usize,
    offset: usize,
) -> Result<Vec<VectorChunkRow>> {
    let mut stmt = conn.prepare(
        "SELECT c.id, f.id, f.path, c.chunk_index, c.start_line, c.end_line, c.text, e.vector
         FROM embeddings e
         JOIN chunks c ON c.id = e.chunk_id
         JOIN files f ON f.id = c.file_id
         WHERE e.profile_id = ?1
         ORDER BY c.id
         LIMIT ?2 OFFSET ?3",
    )?;
    let rows = stmt.query_map(params![profile_id, limit as i64, offset as i64], |row| {
        let bytes: Vec<u8> = row.get(7)?;
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, String>(6)?,
            bytes,
        ))
    })?;
    let mut results = Vec::new();
    for row in rows {
        let (chunk_id, file_id, path, chunk_index, start_line, end_line, text, bytes) = row?;
        results.push(VectorChunkRow {
            chunk_id,
            file_id,
            path,
            chunk_index,
            start_line,
            end_line,
            text,
            vector: deserialize_vector(&bytes)?,
        });
    }
    Ok(results)
}

pub fn stream_vector_chunks_for_profile<F>(
    conn: &Connection,
    profile_id: i64,
    batch_size: usize,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(&[VectorChunkRow]) -> Result<()>,
{
    if batch_size == 0 {
        anyhow::bail!("batch_size must be greater than 0");
    }

    let mut offset = 0usize;
    loop {
        let rows = vector_chunks_for_profile(conn, profile_id, batch_size, offset)?;
        if rows.is_empty() {
            break;
        }
        offset += rows.len();
        on_batch(&rows)?;
    }
    Ok(())
}

pub fn stream_vector_chunks_for_active_profile<F>(
    conn: &Connection,
    config: &Config,
    batch_size: usize,
    on_batch: F,
) -> Result<()>
where
    F: FnMut(&[VectorChunkRow]) -> Result<()>,
{
    let profile_id = upsert_active_embedding_profile(conn, config)?;
    stream_vector_chunks_for_profile(conn, profile_id, batch_size, on_batch)
}
