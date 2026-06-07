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

#[derive(Debug, Clone)]
pub struct ImageForEmbedding {
    pub image_id: i64,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct VectorImageRow {
    pub image_id: i64,
    pub file_id: i64,
    pub path: String,
    pub file_type: String,
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
    r#"
CREATE TABLE IF NOT EXISTS image_embedding_profiles (
  id INTEGER PRIMARY KEY,
  profile_hash TEXT NOT NULL UNIQUE,
  model TEXT NOT NULL,
  endpoint TEXT,
  dimensions INTEGER NOT NULL,
  normalize INTEGER NOT NULL,
  serialization_version TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  width INTEGER,
  height INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(file_id)
);

CREATE TABLE IF NOT EXISTS image_embeddings (
  id INTEGER PRIMARY KEY,
  profile_id INTEGER NOT NULL REFERENCES image_embedding_profiles(id) ON DELETE CASCADE,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(profile_id, image_id)
);

CREATE TABLE IF NOT EXISTS image_query_embeddings (
  id INTEGER PRIMARY KEY,
  profile_id INTEGER NOT NULL REFERENCES image_embedding_profiles(id) ON DELETE CASCADE,
  normalized_query TEXT NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL,
  UNIQUE(profile_id, normalized_query)
);
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

pub fn upsert_embedding_profile(conn: &Connection, profile: &mut EmbeddingProfile) -> Result<i64> {
    if let Some(stored_profile) = get_embedding_profile_by_hash(conn, &profile.profile_hash)? {
        touch_embedding_profile(conn, stored_profile.id, profile)?;
        profile.profile_hash = stored_profile.profile.profile_hash;
        return Ok(stored_profile.id);
    }

    if let Some(stored_profile) = find_compatible_embedding_profile(conn, profile)? {
        touch_embedding_profile(conn, stored_profile.id, profile)?;
        profile.profile_hash = stored_profile.profile.profile_hash;
        return Ok(stored_profile.id);
    }

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
        ON CONFLICT(profile_hash) DO UPDATE SET
            provider = excluded.provider,
            engine = excluded.engine,
            model = excluded.model,
            variant = excluded.variant,
            endpoint = excluded.endpoint,
            dimensions = excluded.dimensions,
            document_prefix = excluded.document_prefix,
            query_prefix = excluded.query_prefix,
            normalizer_version = excluded.normalizer_version,
            chunker_version = excluded.chunker_version,
            serialization_version = excluded.serialization_version,
            created_at = datetime('now')",
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
    let mut profile = active_profile(config);
    upsert_embedding_profile(conn, &mut profile)
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

fn find_compatible_embedding_profile(
    conn: &Connection,
    profile: &EmbeddingProfile,
) -> Result<Option<StoredEmbeddingProfile>> {
    let dimensions = profile.dimensions as i64;
    conn.query_row(
        "SELECT id, profile_hash, provider, engine, model, variant, endpoint, dimensions,
                document_prefix, query_prefix, normalizer_version, chunker_version,
                serialization_version, created_at
         FROM embedding_profiles
        WHERE model = ?1
           AND dimensions = ?2
           AND document_prefix = ?3
           AND query_prefix = ?4
           AND normalizer_version = ?5
           AND chunker_version = ?6
           AND serialization_version = ?7
         ORDER BY created_at DESC, id DESC
         LIMIT 1",
        params![
            &profile.model,
            dimensions,
            &profile.document_prefix,
            &profile.query_prefix,
            &profile.normalizer_version,
            &profile.chunker_version,
            &profile.serialization_version,
        ],
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

fn touch_embedding_profile(conn: &Connection, id: i64, profile: &EmbeddingProfile) -> Result<()> {
    conn.execute(
        "UPDATE embedding_profiles
         SET provider = ?1,
             engine = ?2,
             model = ?3,
             variant = ?4,
             endpoint = ?5,
             dimensions = ?6,
             document_prefix = ?7,
             query_prefix = ?8,
             normalizer_version = ?9,
             chunker_version = ?10,
             serialization_version = ?11,
             created_at = datetime('now')
         WHERE id = ?12",
        params![
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
            id,
        ],
    )?;
    Ok(())
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

pub fn upsert_image_profile(
    conn: &Connection,
    model: &str,
    endpoint: Option<&str>,
    dimensions: usize,
    normalize: bool,
) -> Result<i64> {
    let profile_hash = image_profile_hash(model, endpoint, dimensions, normalize);
    conn.execute(
        "INSERT INTO image_embedding_profiles (
            profile_hash, model, endpoint, dimensions, normalize, serialization_version, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
         ON CONFLICT(profile_hash) DO UPDATE SET profile_hash = excluded.profile_hash",
        params![
            profile_hash,
            model,
            endpoint,
            dimensions as i64,
            if normalize { 1 } else { 0 },
            crate::embed::EMBEDDING_SERIALIZATION_VERSION,
        ],
    )?;
    Ok(conn.query_row(
        "SELECT id FROM image_embedding_profiles WHERE profile_hash = ?1",
        [profile_hash],
        |row| row.get(0),
    )?)
}

pub fn image_profile_hash(
    model: &str,
    endpoint: Option<&str>,
    dimensions: usize,
    normalize: bool,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"image-profile\0");
    hasher.update(model.as_bytes());
    hasher.update(&[0]);
    hasher.update(endpoint.unwrap_or("").as_bytes());
    hasher.update(&[0]);
    hasher.update(dimensions.to_string().as_bytes());
    hasher.update(&[0]);
    hasher.update(if normalize { b"1" } else { b"0" });
    hasher.update(&[0]);
    hasher.update(crate::embed::EMBEDDING_SERIALIZATION_VERSION.as_bytes());
    hasher.finalize().to_hex().to_string()
}

pub fn image_profile_hash_for_config(config: &Config) -> String {
    image_profile_hash(
        &config.image.embedding.model,
        config.image.embedding.endpoint.as_deref(),
        config.image.embedding.dimensions,
        config.image.embedding.normalize,
    )
}

pub fn upsert_image_record(conn: &Connection, file_id: i64) -> Result<i64> {
    conn.execute(
        "INSERT INTO images(file_id, created_at)
         VALUES (?1, datetime('now'))
         ON CONFLICT(file_id) DO UPDATE SET file_id = excluded.file_id",
        [file_id],
    )?;
    Ok(conn.query_row(
        "SELECT id FROM images WHERE file_id = ?1",
        [file_id],
        |row| row.get(0),
    )?)
}

pub fn images_missing_embeddings(
    conn: &Connection,
    profile_id: i64,
    limit: usize,
) -> Result<Vec<ImageForEmbedding>> {
    let mut stmt = conn.prepare(
        "SELECT i.id, f.path
         FROM images i
         JOIN files f ON f.id = i.file_id
         LEFT JOIN image_embeddings e ON e.image_id = i.id AND e.profile_id = ?1
         WHERE e.id IS NULL
         ORDER BY i.id
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![profile_id, limit as i64], |row| {
        Ok(ImageForEmbedding {
            image_id: row.get(0)?,
            path: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn upsert_image_embedding(
    conn: &Connection,
    profile_id: i64,
    image_id: i64,
    vector: &[f32],
) -> Result<()> {
    validate_image_vector_dimensions(conn, profile_id, vector)?;
    conn.execute(
        "INSERT INTO image_embeddings(profile_id, image_id, vector, created_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(profile_id, image_id)
         DO UPDATE SET vector = excluded.vector, created_at = excluded.created_at",
        params![profile_id, image_id, serialize_vector(vector)],
    )?;
    Ok(())
}

pub fn image_query_embedding(
    conn: &Connection,
    profile_id: i64,
    normalized_query: &str,
) -> Result<Option<Vec<f32>>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vector FROM image_query_embeddings WHERE profile_id = ?1 AND normalized_query = ?2",
            params![profile_id, normalized_query],
            |row| row.get(0),
        )
        .optional()?;
    if bytes.is_some() {
        conn.execute(
            "UPDATE image_query_embeddings SET last_used_at = datetime('now') WHERE profile_id = ?1 AND normalized_query = ?2",
            params![profile_id, normalized_query],
        )?;
    }
    bytes.map(|bytes| deserialize_vector(&bytes)).transpose()
}

pub fn upsert_image_query_embedding(
    conn: &Connection,
    profile_id: i64,
    normalized_query: &str,
    vector: &[f32],
) -> Result<()> {
    validate_image_vector_dimensions(conn, profile_id, vector)?;
    conn.execute(
        "INSERT INTO image_query_embeddings(profile_id, normalized_query, vector, created_at, last_used_at)
         VALUES (?1, ?2, ?3, datetime('now'), datetime('now'))
         ON CONFLICT(profile_id, normalized_query)
         DO UPDATE SET vector = excluded.vector, last_used_at = excluded.last_used_at",
        params![profile_id, normalized_query, serialize_vector(vector)],
    )?;
    Ok(())
}

pub fn vector_images_for_profile(
    conn: &Connection,
    profile_id: i64,
    limit: usize,
    offset: usize,
) -> Result<Vec<VectorImageRow>> {
    let mut stmt = conn.prepare(
        "SELECT i.id, f.id, f.path, f.file_type, e.vector
         FROM image_embeddings e
         JOIN images i ON i.id = e.image_id
         JOIN files f ON f.id = i.file_id
         WHERE e.profile_id = ?1
         ORDER BY i.id
         LIMIT ?2 OFFSET ?3",
    )?;
    let rows = stmt.query_map(params![profile_id, limit as i64, offset as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Vec<u8>>(4)?,
        ))
    })?;
    let mut results = Vec::new();
    for row in rows {
        let (image_id, file_id, path, file_type, bytes) = row?;
        results.push(VectorImageRow {
            image_id,
            file_id,
            path,
            file_type,
            vector: deserialize_vector(&bytes)?,
        });
    }
    Ok(results)
}

pub fn stream_vector_images_for_profile<F>(
    conn: &Connection,
    profile_id: i64,
    batch_size: usize,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(&[VectorImageRow]) -> Result<()>,
{
    if batch_size == 0 {
        anyhow::bail!("batch_size must be greater than 0");
    }
    let mut offset = 0usize;
    loop {
        let rows = vector_images_for_profile(conn, profile_id, batch_size, offset)?;
        if rows.is_empty() {
            break;
        }
        offset += rows.len();
        on_batch(&rows)?;
    }
    Ok(())
}

fn validate_image_vector_dimensions(
    conn: &Connection,
    profile_id: i64,
    vector: &[f32],
) -> Result<()> {
    let expected: i64 = conn.query_row(
        "SELECT dimensions FROM image_embedding_profiles WHERE id = ?1",
        [profile_id],
        |row| row.get(0),
    )?;
    if vector.len() != expected as usize {
        anyhow::bail!(
            "image embedding vector has {} dimensions; expected {} for profile {}",
            vector.len(),
            expected,
            profile_id
        );
    }
    Ok(())
}
