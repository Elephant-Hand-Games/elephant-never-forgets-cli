use std::{
    fs,
    path::{Path, PathBuf},
};

use elephant_never_forgets::{config::Config, discovery, index};
use rusqlite::Connection;

fn write_file(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn test_config() -> Config {
    let mut config = Config::default();
    config.index.chunk_target_tokens = 4;
    config.index.chunk_max_tokens = 6;
    config.index.chunk_overlap_tokens = 2;
    config.index.min_chunk_chars = 1;
    config
}

fn db_connection(root: &Path) -> Connection {
    Connection::open(root.join(".enf/index.sqlite")).unwrap()
}

fn chunk_rows(root: &Path, path: &str) -> Vec<(i64, String, i64, i64, String)> {
    let conn = db_connection(root);
    let mut stmt = conn
        .prepare(
            "SELECT c.chunk_index, c.hash, c.start_line, c.end_line, c.text
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE f.path = ?1
             ORDER BY c.chunk_index",
        )
        .unwrap();
    let rows = stmt
        .query_map([path], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        })
        .unwrap();
    rows.map(|row| row.unwrap()).collect()
}

#[test]
fn discovery_honors_include_and_exclude_patterns() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    write_file(root, "docs/guide.md", "# guide");
    write_file(root, "sub/AGENTS.md", "rules");
    write_file(root, "notes/keep.txt", "keep");
    write_file(root, ".git/ignored.md", "ignored");
    write_file(root, "node_modules/pkg/ignored.txt", "ignored");

    let files = discovery::discover(root, root, &test_config()).unwrap();
    let paths: Vec<_> = files.into_iter().map(|file| file.relative_path).collect();

    assert_eq!(
        paths,
        vec!["docs/guide.md", "notes/keep.txt", "sub/AGENTS.md"]
    );
}

#[test]
fn index_keeps_chunk_stability_for_unchanged_files() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let config = test_config();

    write_file(
        root,
        "docs/story.txt",
        "one two\nthree four\nfive six\nseven eight\nnine ten\neleven twelve",
    );
    write_file(root, "notes/other.txt", "alpha");

    index::index_path(root, PathBuf::from("."), &config, false).unwrap();
    let before = chunk_rows(root, "docs/story.txt");

    write_file(root, "notes/other.txt", "alpha beta gamma");
    index::index_path(root, PathBuf::from("."), &config, false).unwrap();
    let after = chunk_rows(root, "docs/story.txt");

    assert_eq!(before, after);
}

#[test]
fn index_removes_deleted_files_when_root_is_reindexed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let config = test_config();

    write_file(root, "docs/kept.txt", "keep me");
    write_file(root, "docs/gone.txt", "delete me");

    index::index_path(root, PathBuf::from("."), &config, false).unwrap();
    fs::remove_file(root.join("docs/gone.txt")).unwrap();
    index::index_path(root, Path::new(".").to_path_buf(), &config, false).unwrap();

    let conn = db_connection(root);
    let mut stmt = conn
        .prepare("SELECT path FROM files ORDER BY path")
        .unwrap();
    let paths: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(|row| row.unwrap())
        .collect();

    assert_eq!(paths, vec!["docs/kept.txt"]);
    assert!(chunk_rows(root, "docs/gone.txt").is_empty());
}

#[test]
fn index_honors_store_full_files_and_store_chunks_flags() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let mut config = test_config();
    config.index.store_full_files = false;
    config.index.store_chunks = false;

    write_file(root, "docs/config.txt", "alpha beta gamma");
    index::index_path(root, PathBuf::from("."), &config, false).unwrap();

    let conn = db_connection(root);
    let content: Option<String> = conn
        .query_row(
            "SELECT content FROM files WHERE path = 'docs/config.txt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();

    assert_eq!(content, None);
    assert_eq!(chunks, 0);
}
