use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

#[test]
fn help_exposes_core_commands() {
    let mut cmd = Command::cargo_bin("enf").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("init"))
        .stdout(predicate::str::contains("setup"))
        .stdout(predicate::str::contains("config"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("update"))
        .stdout(predicate::str::contains("search"))
        .stdout(predicate::str::contains("add").not())
        .stdout(predicate::str::contains("ci").not());
}

#[test]
fn init_help_guides_model_and_chunking_choices() {
    Command::cargo_bin("enf")
        .unwrap()
        .args(["init", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Gemma"))
        .stdout(predicate::str::contains("code"))
        .stdout(predicate::str::contains("Nomic"))
        .stdout(predicate::str::contains("larger documents"))
        .stdout(predicate::str::contains("--chunking"))
        .stdout(predicate::str::contains("--interactive"))
        .stdout(predicate::str::contains("--preset"))
        .stdout(predicate::str::contains("--no-input"))
        .stdout(predicate::str::contains("--yes"));
}

#[test]
fn setup_presets_and_config_commands_are_available() {
    Command::cargo_bin("enf")
        .unwrap()
        .args(["setup", "presets"])
        .assert()
        .success()
        .stdout(predicate::str::contains("local"))
        .stdout(predicate::str::contains("openai"))
        .stdout(predicate::str::contains("keyword"));

    Command::cargo_bin("enf")
        .unwrap()
        .args(["setup", "preset", "local"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Preset: local"))
        .stdout(predicate::str::contains("embedding.provider"));
}

#[test]
fn config_explain_and_set_support_safe_keys() {
    let temp = tempfile::tempdir().unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["init", "--yes"])
        .assert()
        .success();

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["config", "explain", "reranker.endpoint"])
        .assert()
        .success()
        .stdout(predicate::str::contains("HTTP endpoint"));

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["config", "set", "search.limit", "20"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Updated config"));

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["config", "show", "search"])
        .assert()
        .success()
        .stdout(predicate::str::contains("limit = 20"));
}

#[test]
fn init_creates_config_and_database() {
    let temp = tempfile::tempdir().unwrap();
    let mut cmd = Command::cargo_bin("enf").unwrap();
    cmd.current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .arg("init")
        .assert()
        .success()
        .stdout(predicate::str::contains("Ready."))
        .stdout(predicate::str::contains("Wrote .enf.toml"));

    assert!(temp.path().join(".enf.toml").exists());
    assert!(temp.path().join(".enf/index.sqlite").exists());
}

#[test]
fn init_dry_run_does_not_create_project_files() {
    let temp = tempfile::tempdir().unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["init", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Plan"))
        .stdout(predicate::str::contains("Create .enf.toml"));

    assert!(!temp.path().join(".enf.toml").exists());
    assert!(!temp.path().join(".enf").exists());
}

#[test]
fn init_defaults_to_native_profile() {
    let temp = tempfile::tempdir().unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .arg("init")
        .assert()
        .success();

    let config = std::fs::read_to_string(temp.path().join(".enf.toml")).unwrap();
    assert!(config.contains("provider = \"native\""));
    assert!(config.contains("model = \"nomic-embed-text-v1.5\""));
    assert!(config.contains("variant = \"quantized\""));
    assert!(config.contains("engine = \"candle\""));
}

#[test]
fn index_json_outputs_machine_readable_summary() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("README.md"), "Local notes").unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["init", "--provider", "ollama"])
        .assert()
        .success();

    let output = Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["index", ".", "--no-embeddings", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["discovered_files"], 1);
    assert_eq!(json["changed_files"], 1);
    assert_eq!(json["embedded_chunks"], 0);
    assert_eq!(json["no_embed"], true);
}

#[test]
fn index_dry_run_reports_discovery_without_creating_database_rows() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("README.md"), "Local notes").unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .arg("init")
        .assert()
        .success();

    let output = Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["index", ".", "--dry-run", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(json["discovered_files"], 1);
    let conn = rusqlite::Connection::open(temp.path().join(".enf/index.sqlite")).unwrap();
    let files: i64 = conn
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(files, 0);
}

#[test]
fn update_dry_run_prints_installer_command_and_env() {
    Command::cargo_bin("enf")
        .unwrap()
        .args([
            "update",
            "--version",
            "v2.0.0",
            "--install-dir",
            "/tmp/enf/bin",
            "--method",
            "binary",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("curl -fsSL"))
        .stdout(predicate::str::contains("scripts/install.sh"))
        .stdout(predicate::str::contains("ENF_VERSION=v2.0.0"))
        .stdout(predicate::str::contains("ENF_INSTALL_DIR=/tmp/enf/bin"))
        .stdout(predicate::str::contains("ENF_INSTALL_METHOD=binary"));
}

#[test]
fn remove_dry_run_reports_path_without_deleting_index_row() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("README.md"), "Local notes").unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .arg("init")
        .assert()
        .success();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["index", ".", "--no-embeddings"])
        .assert()
        .success();

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["remove", "README.md", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Dry run: would remove README.md from index",
        ));

    let conn = rusqlite::Connection::open(temp.path().join(".enf/index.sqlite")).unwrap();
    let files: i64 = conn
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(files, 1);
}

#[test]
fn models_install_dry_run_reports_marker_without_writing_it() {
    let temp = tempfile::tempdir().unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["init", "--provider", "ollama", "--model-cache", "project"])
        .assert()
        .success();

    let output = Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["models", "install", "--dry-run", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    let marker = json["marker_path"].as_str().unwrap();

    assert_eq!(json["dry_run"], true);
    assert!(!std::path::Path::new(marker).exists());
}
