use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

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
