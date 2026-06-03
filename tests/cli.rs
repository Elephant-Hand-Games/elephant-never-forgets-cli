use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_exposes_core_commands() {
    let mut cmd = Command::cargo_bin("enf").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("init"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("search"));
}

#[test]
fn init_creates_config_and_database() {
    let temp = tempfile::tempdir().unwrap();
    let mut cmd = Command::cargo_bin("enf").unwrap();
    cmd.current_dir(temp.path())
        .arg("init")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Initialized Elephant Never Forgets project",
        ));

    assert!(temp.path().join(".enf.toml").exists());
    assert!(temp.path().join(".enf/index.sqlite").exists());
}

#[test]
fn init_defaults_to_native_profile() {
    let temp = tempfile::tempdir().unwrap();
    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .arg("init")
        .assert()
        .success();

    let config = std::fs::read_to_string(temp.path().join(".enf.toml")).unwrap();
    assert!(config.contains("provider = \"native\""));
    assert!(config.contains("model = \"nomic-embed-text-v1.5\""));
    assert!(config.contains("variant = \"quantized\""));
}
