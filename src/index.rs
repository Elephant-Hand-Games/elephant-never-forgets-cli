use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

use crate::{cli::IndexArgs, config::Config, db};

pub fn run(args: IndexArgs) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    index_path(&cwd, args.path, &config, args.install_models)
}

pub fn index_path(
    root: &std::path::Path,
    path: PathBuf,
    config: &Config,
    _install_models: bool,
) -> Result<()> {
    let db_path = root.join(&config.state.db_path);
    let conn = db::open_or_create(&db_path)?;
    let target = root.join(path);
    let mut indexed = 0usize;
    for entry in WalkDir::new(&target).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let file_path = entry.path();
        let rel = file_path.strip_prefix(root).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if !is_supported(&rel_str) || is_excluded(&rel_str) {
            continue;
        }
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("reading {}", file_path.display()))?;
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        conn.execute(
            "INSERT INTO files(path, hash, size_bytes, indexed_at, content)
             VALUES (?1, ?2, ?3, datetime('now'), ?4)
             ON CONFLICT(path) DO UPDATE SET hash=excluded.hash, size_bytes=excluded.size_bytes, indexed_at=excluded.indexed_at, content=excluded.content",
            rusqlite::params![rel_str, hash, content.len() as i64, content],
        )?;
        indexed += 1;
    }
    println!("Indexed {indexed} files");
    Ok(())
}

fn is_supported(path: &str) -> bool {
    path == "AGENTS.md"
        || path.starts_with("README")
        || path.starts_with("CHANGELOG")
        || path.starts_with("CONTRIBUTING")
        || path.starts_with("docs/")
        || [".md", ".mdx", ".txt", ".rst", ".adoc", ".org", ".ehmeta"]
            .iter()
            .any(|suffix| path.ends_with(suffix))
}

fn is_excluded(path: &str) -> bool {
    path.starts_with(".git/")
        || path.starts_with(".enf/")
        || path.starts_with("node_modules/")
        || path.starts_with("dist/")
        || path.starts_with("build/")
        || path.starts_with("target/")
        || path.starts_with(".next/")
        || path.starts_with(".venv/")
        || path.starts_with("__pycache__/")
        || path.ends_with(".lock")
}
