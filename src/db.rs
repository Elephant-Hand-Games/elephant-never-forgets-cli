use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::cli::CiArgs;

const MIGRATIONS: &[&str] = &[r#"
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
"#];

pub fn open_or_create(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating database directory {}", parent.display()))?;
    }
    let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
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

pub fn ci(args: CiArgs) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    open_or_create(&cwd.join(config.state.db_path))?;
    if args.json {
        println!(
            "{}",
            serde_json::json!({"ok": true, "no_embed": args.no_embed})
        );
    } else {
        println!("ENF CI checks passed");
    }
    Ok(())
}
