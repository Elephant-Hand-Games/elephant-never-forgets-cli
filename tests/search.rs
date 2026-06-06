use assert_cmd::Command;
use elephant_never_forgets::{db, embed, ranking};
use predicates::prelude::*;
use rusqlite::Connection;
use serde_json::Value;
use std::fs;

fn enf() -> Command {
    let mut cmd = Command::cargo_bin("enf").unwrap();
    cmd.env("ENF_SKIP_NATIVE_MODEL_LOAD", "1");
    cmd
}

fn setup_indexed_project() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join("docs")).unwrap();
    std::fs::write(
        temp.path().join("docs/agents.md"),
        "Agent editing rules\n\nUse apply_patch for manual edits.\nKeep tests current.",
    )
    .unwrap();
    std::fs::write(
        temp.path().join("README.md"),
        "Elephant Never Forgets indexes local documents.",
    )
    .unwrap();

    enf()
        .current_dir(temp.path())
        .args(["init", "--db"])
        .assert()
        .success();
    enf()
        .current_dir(temp.path())
        .args(["index", "--no-embed", "."])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed"));

    temp
}

fn setup_indexed_project_with_image() -> tempfile::TempDir {
    let temp = setup_indexed_project();
    std::fs::create_dir_all(temp.path().join("assets")).unwrap();
    std::fs::write(temp.path().join("assets/logo.png"), b"fake png bytes").unwrap();
    enf()
        .current_dir(temp.path())
        .args(["index", "--no-embed", "."])
        .assert()
        .success();
    temp
}

fn load_config(root: &std::path::Path) -> elephant_never_forgets::config::Config {
    let config_text = fs::read_to_string(root.join(".enf.toml")).unwrap();
    toml::from_str(&config_text).unwrap()
}

fn seed_agent_vector(root: &std::path::Path, query: &str) {
    let mut config = load_config(root);
    config.embedding.dimensions = 3;
    elephant_never_forgets::config::write_config(&root.join(".enf.toml"), &config).unwrap();
    let conn = Connection::open(root.join(".enf/index.sqlite")).unwrap();
    let profile = embed::active_profile(&config);
    let profile_id = db::upsert_embedding_profile(&conn, &profile).unwrap();
    let agent_chunk_id: i64 = conn
        .query_row(
            "SELECT c.id
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE f.path = 'docs/agents.md'
             ORDER BY c.chunk_index
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();

    db::upsert_query_embedding(
        &conn,
        profile_id,
        &ranking::query_cache_key(query),
        &[1.0, 0.0, 0.0],
    )
    .unwrap();
    db::upsert_chunk_embedding(&conn, profile_id, agent_chunk_id, &[1.0, 0.0, 0.0]).unwrap();
}

fn seed_image_vector(root: &std::path::Path, query: &str) {
    let mut config = load_config(root);
    config.image.embedding.enabled = true;
    config.image.embedding.endpoint = Some("http://localhost:41802/v1/images/embeddings".into());
    config.image.embedding.query_endpoint =
        Some("http://localhost:41802/v1/images/query_embeddings".into());
    config.image.embedding.dimensions = 3;
    elephant_never_forgets::config::write_config(&root.join(".enf.toml"), &config).unwrap();

    let conn = Connection::open(root.join(".enf/index.sqlite")).unwrap();
    let profile_id = db::upsert_image_profile(
        &conn,
        &config.image.embedding.model,
        config.image.embedding.endpoint.as_deref(),
        config.image.embedding.dimensions,
        config.image.embedding.normalize,
    )
    .unwrap();
    let image_id: i64 = conn
        .query_row(
            "SELECT i.id
             FROM images i
             JOIN files f ON f.id = i.file_id
             WHERE f.path = 'assets/logo.png'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    db::upsert_image_query_embedding(
        &conn,
        profile_id,
        &ranking::query_cache_key(query),
        &[1.0, 0.0, 0.0],
    )
    .unwrap();
    db::upsert_image_embedding(&conn, profile_id, image_id, &[1.0, 0.0, 0.0]).unwrap();
}

#[test]
fn search_returns_ranked_plain_text_results() {
    let temp = setup_indexed_project();

    enf()
        .current_dir(temp.path())
        .args(["search", "apply_patch"])
        .assert()
        .success()
        .stdout(predicate::str::contains("docs/agents.md"));
}

#[test]
fn retrieve_returns_json_results() {
    let temp = setup_indexed_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["retrieve", "editing rules", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(json["query"], "editing rules");
    assert_eq!(json["results"][0]["path"], "docs/agents.md");
    assert!(json["results"][0]["snippet"]
        .as_str()
        .unwrap()
        .contains("Agent editing rules"));
}

#[test]
fn search_limit_bounds_results() {
    let temp = setup_indexed_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["search", "local", "--json", "--limit", "1"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
}

#[test]
fn keyword_mode_allows_strict_cache_flag_without_query_vector() {
    let temp = setup_indexed_project();

    enf()
        .current_dir(temp.path())
        .args([
            "search",
            "apply_patch",
            "--mode",
            "keyword",
            "--cached-query-only",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("docs/agents.md"));
}

#[test]
fn file_level_search_returns_file_results() {
    let temp = setup_indexed_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["search", "documents", "--level", "file", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["results"][0]["level"], "file");
    assert_eq!(json["results"][0]["kind"], "file");
    assert_eq!(json["results"][0]["path"], "README.md");
}

#[test]
fn provider_overrides_use_a_distinct_profile_for_search() {
    let temp = setup_indexed_project();
    let default_config = load_config(temp.path());
    let default_profile = embed::active_profile(&default_config);

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "local",
            "--provider",
            "ollama",
            "--model",
            "nomic-embed-text",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_ne!(
        json["profile_hash"].as_str().unwrap(),
        default_profile.profile_hash
    );
    assert!(json["warnings"][0]
        .as_str()
        .unwrap()
        .contains("has no indexed embeddings"));
    assert_eq!(json["results"][0]["mode"], "keyword");
}

#[test]
fn vector_mode_without_vectors_reports_missing_index_embeddings() {
    let temp = setup_indexed_project();

    enf()
        .current_dir(temp.path())
        .args(["search", "zebra", "--mode", "vector"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("has no indexed embeddings"))
        .stderr(predicate::str::contains("enf index ."));
}

#[test]
fn cached_query_only_fails_on_cache_miss() {
    let temp = setup_indexed_project();

    enf()
        .current_dir(temp.path())
        .args(["search", "not cached", "--cached-query-only"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "query embedding is not cached and --cached-query-only was set",
        ));
}

#[test]
fn cached_query_only_succeeds_with_seeded_query_cache() {
    let temp = setup_indexed_project();
    seed_agent_vector(temp.path(), "zebra");

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "zebra",
            "--mode",
            "vector",
            "--cached-query-only",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["query"], "zebra");
    assert_eq!(json["results"][0]["path"], "docs/agents.md");
    assert_eq!(json["results"][0]["mode"], "vector");
    assert_eq!(json["results"][0]["level"], "chunk");
    assert_eq!(json["results"][0]["vector_score"], 1.0);
}

#[test]
fn cached_query_only_respects_disabled_query_cache() {
    let temp = setup_indexed_project();
    seed_agent_vector(temp.path(), "zebra");
    let mut config = load_config(temp.path());
    config.embedding.query_cache = false;
    elephant_never_forgets::config::write_config(&temp.path().join(".enf.toml"), &config).unwrap();

    enf()
        .current_dir(temp.path())
        .args(["search", "zebra", "--mode", "vector", "--cached-query-only"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "query embedding is not cached and --cached-query-only was set",
        ));
}

#[test]
fn keyword_search_does_not_treat_percent_or_underscore_as_wildcards() {
    let temp = setup_indexed_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["search", "%_", "--mode", "keyword", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert!(json["results"].as_array().unwrap().is_empty());
}

#[test]
fn vector_file_level_aggregates_seeded_chunk_vectors() {
    let temp = setup_indexed_project();
    seed_agent_vector(temp.path(), "zebra");

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "zebra",
            "--mode",
            "vector",
            "--level",
            "file",
            "--cached-query-only",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["results"][0]["path"], "docs/agents.md");
    assert_eq!(json["results"][0]["mode"], "vector");
    assert_eq!(json["results"][0]["level"], "file");
    assert_eq!(json["results"][0]["vector_score"], 1.0);
}

#[test]
fn search_kind_image_returns_labeled_image_results() {
    let temp = setup_indexed_project_with_image();

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search", "logo", "--kind", "image", "--mode", "keyword", "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["results"][0]["kind"], "image");
    assert_eq!(json["results"][0]["path"], "assets/logo.png");
    assert_eq!(json["results"][0]["file_type"], "png");
}

#[test]
fn search_kind_image_vector_uses_cached_query_embedding() {
    let temp = setup_indexed_project_with_image();
    seed_image_vector(temp.path(), "logo");

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "logo",
            "--kind",
            "image",
            "--mode",
            "vector",
            "--cached-query-only",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["results"][0]["kind"], "image");
    assert_eq!(json["results"][0]["path"], "assets/logo.png");
    assert_eq!(json["results"][0]["mode"], "vector");
    assert_eq!(json["results"][0]["level"], "image");
    assert_eq!(json["results"][0]["vector_score"], 1.0);
}

#[test]
fn search_kind_text_excludes_image_results() {
    let temp = setup_indexed_project_with_image();

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search", "logo", "--kind", "text", "--mode", "keyword", "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert!(json["results"].as_array().unwrap().is_empty());
}

#[test]
fn search_filters_by_filetype_and_path_glob() {
    let temp = setup_indexed_project_with_image();

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "logo",
            "--kind",
            "image",
            "--mode",
            "keyword",
            "--filetype",
            ".png",
            "--path",
            "assets/**",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(json["results"][0]["path"], "assets/logo.png");

    let output = enf()
        .current_dir(temp.path())
        .args([
            "search",
            "logo",
            "--kind",
            "image",
            "--mode",
            "keyword",
            "--filetype",
            "jpg",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    assert!(json["results"].as_array().unwrap().is_empty());
}
