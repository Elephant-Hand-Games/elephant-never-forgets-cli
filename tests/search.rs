use assert_cmd::Command;
use elephant_never_forgets::{db, embed, ranking};
use predicates::prelude::*;
use rusqlite::Connection;
use serde_json::Value;
use std::fs;

fn enf() -> Command {
    Command::cargo_bin("enf").unwrap()
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
        .args(["index", "."])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed"));

    temp
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

    let config_text = fs::read_to_string(temp.path().join(".enf.toml")).unwrap();
    let config: elephant_never_forgets::config::Config = toml::from_str(&config_text).unwrap();
    let conn = Connection::open(temp.path().join(".enf/index.sqlite")).unwrap();
    let profile = embed::active_profile(&config);
    let profile_id = db::upsert_embedding_profile(&conn, &profile).unwrap();

    db::upsert_query_embedding(
        &conn,
        profile_id,
        &ranking::query_cache_key("zebra"),
        &[0.25, -0.5, 1.0],
    )
    .unwrap();

    let output = enf()
        .current_dir(temp.path())
        .args(["search", "zebra", "--cached-query-only", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["query"], "zebra");
    assert!(json["results"].is_array());
}
