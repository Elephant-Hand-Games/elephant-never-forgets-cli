use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn enf() -> Command {
    Command::cargo_bin("enf").unwrap()
}

fn setup_project() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join("docs")).unwrap();
    std::fs::write(
        temp.path().join("docs/guide.md"),
        "Elephant Never Forgets status and doctor checks.",
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
        .success();
    temp
}

#[test]
fn status_reports_index_counts_and_active_profile() {
    let temp = setup_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["status", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["config"], ".enf.toml");
    assert_eq!(json["counts"]["files"], 1);
    assert_eq!(json["counts"]["chunks"], 1);
    assert_eq!(json["counts"]["active_profile_embeddings"], 0);
    assert_eq!(json["counts"]["missing_active_profile_embeddings"], 1);
    assert_eq!(json["active_profile"]["provider"], "native");
}

#[test]
fn doctor_reports_missing_native_model_without_loading_it() {
    let temp = setup_project();

    let output = enf()
        .current_dir(temp.path())
        .args(["doctor", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["ok"], false);
    assert_eq!(json["config"]["ok"], true);
    assert_eq!(json["database"]["ok"], true);
    assert_eq!(json["fts5"]["ok"], true);
    assert_eq!(json["model"]["ok"], false);
    assert_eq!(json["provider"]["remediation"], "enf models install");
}

#[test]
fn ci_no_embed_checks_config_and_database_only() {
    let temp = setup_project();

    enf()
        .current_dir(temp.path())
        .args(["ci", "--no-embed"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ENF CI checks passed"));
}

#[test]
fn ci_fails_when_embeddings_are_required_but_missing() {
    let temp = setup_project();

    enf()
        .current_dir(temp.path())
        .args(["ci"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("ENF CI checks failed"));
}
