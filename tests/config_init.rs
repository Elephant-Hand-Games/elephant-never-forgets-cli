use std::fs;

use assert_cmd::Command;

use elephant_never_forgets::config::{self, Config, ModelCache, ModelVariant, Provider};

fn read_config(path: &std::path::Path) -> Config {
    let text = fs::read_to_string(path).unwrap();
    toml::from_str(&text).unwrap()
}

fn run_init(temp_dir: &std::path::Path, extra_args: &[&str]) {
    let mut args = vec!["init"];
    args.extend(extra_args.iter().copied());

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp_dir)
        .args(args)
        .assert()
        .success();
}

#[test]
fn init_writes_default_spec_config() {
    let temp = tempfile::tempdir().unwrap();

    run_init(temp.path(), &[]);

    let config = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(config, Config::default());
}

#[test]
fn init_applies_provider_and_model_overrides_coherently() {
    let temp = tempfile::tempdir().unwrap();

    run_init(
        temp.path(),
        &["--provider", "ollama", "--model", "nomic-embed-text"],
    );

    let ollama = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(ollama.embedding.provider, Provider::Ollama);
    assert_eq!(ollama.embedding.engine, None);
    assert_eq!(ollama.embedding.model, "nomic-embed-text");
    assert_eq!(
        ollama.embedding.endpoint.as_deref(),
        Some("http://localhost:11434/api/embed")
    );
    assert_eq!(ollama.embedding.api_key_env, None);
    assert_eq!(ollama.embedding.variant, None);
    assert_eq!(ollama.embedding.dimensions, 768);

    run_init(temp.path(), &["--force", "--provider", "openai"]);

    let openai = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(openai.embedding.provider, Provider::Openai);
    assert_eq!(openai.embedding.engine, None);
    assert_eq!(openai.embedding.model, "text-embedding-3-small");
    assert_eq!(
        openai.embedding.endpoint.as_deref(),
        Some("https://api.openai.com/v1/embeddings")
    );
    assert_eq!(
        openai.embedding.api_key_env.as_deref(),
        Some("OPENAI_API_KEY")
    );
    assert_eq!(openai.embedding.variant, None);
    assert_eq!(openai.embedding.dimensions, 1536);
    assert_eq!(openai.embedding.document_prefix, "");
    assert_eq!(openai.embedding.query_prefix, "");
}

#[test]
fn init_supports_native_aliases_model_and_cache_overrides() {
    for native_flag in ["--native-embed", "--local-embed"] {
        let temp = tempfile::tempdir().unwrap();

        run_init(
            temp.path(),
            &[
                native_flag,
                "--model",
                "nomic-embed-text-v1.5",
                "--variant",
                "full",
                "--model-cache",
                "project",
            ],
        );

        let config = read_config(&temp.path().join(".enf.toml"));
        assert_eq!(config.embedding.provider, Provider::Native);
        assert_eq!(config.embedding.engine.as_deref(), Some("fastembed"));
        assert_eq!(config.embedding.model, "nomic-embed-text-v1.5");
        assert_eq!(config.embedding.variant, Some(ModelVariant::Full));
        assert_eq!(config.state.model_cache, ModelCache::Project);
        assert_eq!(config.embedding.endpoint, None);
        assert_eq!(config.embedding.api_key_env, None);
        assert_eq!(config.embedding.dimensions, 768);
    }
}

#[test]
fn init_refuses_to_overwrite_existing_config_without_force() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join(".enf.toml");
    fs::write(&config_path, "sentinel = true\n").unwrap();

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["init"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("config already exists at"))
        .stderr(predicates::str::contains("Use --force to overwrite it"));

    assert_eq!(
        fs::read_to_string(&config_path).unwrap(),
        "sentinel = true\n"
    );
    assert!(!temp.path().join(".enf/index.sqlite").exists());
}

#[test]
fn init_db_false_requires_existing_database() {
    let temp = tempfile::tempdir().unwrap();

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .args(["init", "--db=false"])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "--db=false requires an existing database",
        ));

    assert!(!temp.path().join(".enf.toml").exists());
}

#[test]
fn validate_reports_actionable_errors() {
    let mut config = Config::default();
    config.embedding.provider = Provider::Openai;
    config.embedding.engine = None;
    config.embedding.endpoint = None;
    config.embedding.api_key_env = None;
    config.embedding.variant = None;

    let err = config::validate(&config).unwrap_err().to_string();
    assert!(err.contains("embedding.endpoint"));
    assert!(err.contains("openai"));

    let mut config = Config::default();
    config.search.vector_weight = -0.25;
    let err = config::validate(&config).unwrap_err().to_string();
    assert!(err.contains("search.vector_weight"));

    let mut config = Config::default();
    config.index.chunk_overlap_tokens = config.index.chunk_max_tokens;
    let err = config::validate(&config).unwrap_err().to_string();
    assert!(err.contains("index.chunk_overlap_tokens"));
}
