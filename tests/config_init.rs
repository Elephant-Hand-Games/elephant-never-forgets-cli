use std::fs;

use assert_cmd::Command;

use elephant_never_forgets::config::{
    self, ChunkingMode, Config, EmbeddingFallbackConfig, ModelCache, ModelVariant, Provider,
};

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
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
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
fn default_config_tracks_code_files_and_uses_smart_chunking() {
    let config = Config::default();

    assert_eq!(config.index.chunking, ChunkingMode::Smart);
    for pattern in [
        "**/*.rs",
        "**/*.js",
        "**/*.ts",
        "**/*.jsx",
        "**/*.tsx",
        "**/*.py",
        "**/*.go",
        "**/*.zig",
        "**/*.gd",
        "**/*.html",
        "**/*.css",
        "**/*.json",
        "**/*.toml",
        "**/*.tmol",
        "**/*.yml",
        "**/*.yaml",
        "**/*.cs",
        "**/*.c",
        "**/*.cc",
        "**/*.cpp",
        "**/*.cxx",
        "**/*.h",
        "**/*.hh",
        "**/*.hpp",
        "**/*.hxx",
        "**/*.java",
        "**/*.kt",
        "**/*.kts",
        "**/*.swift",
        "**/*.sql",
        "**/*.xml",
        "**/*.svg",
        "**/*.lua",
        "**/*.glsl",
        "**/*.vert",
        "**/*.frag",
        "**/*.comp",
        "**/*.hlsl",
        "**/*.wgsl",
        "**/*.sh",
        "**/*.bash",
        "**/*.zsh",
        "**/*.mk",
        "**/Dockerfile",
        "**/Dockerfile.*",
        "**/Makefile",
    ] {
        assert!(
            config
                .text
                .include
                .patterns
                .iter()
                .any(|value| value == pattern),
            "missing default include pattern {pattern}"
        );
    }
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
    assert_eq!(ollama.embedding.model, "nomic-embed-text-v1.5");
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
                "quantized",
                "--model-cache",
                "project",
            ],
        );

        let config = read_config(&temp.path().join(".enf.toml"));
        assert_eq!(config.embedding.provider, Provider::Native);
        assert_eq!(config.embedding.engine.as_deref(), Some("candle"));
        assert_eq!(config.embedding.model, "nomic-embed-text-v1.5");
        assert_eq!(config.embedding.variant, Some(ModelVariant::Quantized));
        assert_eq!(config.state.model_cache, ModelCache::Project);
        assert_eq!(config.embedding.endpoint, None);
        assert_eq!(config.embedding.api_key_env, None);
        assert_eq!(config.embedding.dimensions, 768);
    }
}

#[test]
fn init_normalizes_nomic_and_gemma_aliases_with_model_specific_guidance() {
    let temp = tempfile::tempdir().unwrap();

    run_init(
        temp.path(),
        &[
            "--provider",
            "native",
            "--model",
            "gemma",
            "--chunking",
            "smart",
        ],
    );

    let gemma = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(gemma.embedding.provider, Provider::Native);
    assert_eq!(gemma.embedding.engine.as_deref(), Some("candle"));
    assert_eq!(gemma.embedding.model, "google/embeddinggemma-300m");
    assert_eq!(gemma.embedding.dimensions, 768);
    assert_eq!(gemma.embedding.document_prefix, "title: none | text: ");
    assert_eq!(
        gemma.embedding.query_prefix,
        "task: search result | query: "
    );
    assert_eq!(gemma.index.chunking, ChunkingMode::Smart);

    run_init(
        temp.path(),
        &[
            "--force",
            "--provider",
            "openai-compatible",
            "--model",
            "nomic",
            "--endpoint",
            "https://embed.example.test/v1/embeddings",
            "--chunking",
            "off",
        ],
    );

    let nomic = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(nomic.embedding.provider, Provider::OpenaiCompatible);
    assert_eq!(nomic.embedding.model, "nomic-embed-text-v1.5");
    assert_eq!(nomic.embedding.document_prefix, "search_document: ");
    assert_eq!(nomic.embedding.query_prefix, "search_query: ");
    assert_eq!(nomic.index.chunking, ChunkingMode::Off);
}

#[test]
fn init_records_same_model_fallback_endpoint_or_native_fallback() {
    let temp = tempfile::tempdir().unwrap();

    run_init(
        temp.path(),
        &[
            "--provider",
            "openai-compatible",
            "--model",
            "google/embeddinggemma-300m",
            "--endpoint",
            "https://primary.example.test/v1/embeddings",
            "--fallback-provider",
            "native",
        ],
    );

    let config = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(
        config.embedding.fallback,
        Some(EmbeddingFallbackConfig {
            provider: Provider::Native,
            endpoint: None,
            api_key_env: None,
        })
    );

    run_init(
        temp.path(),
        &[
            "--force",
            "--provider",
            "openai-compatible",
            "--model",
            "gemma",
            "--endpoint",
            "https://primary.example.test/v1/embeddings",
            "--fallback-provider",
            "openai-compatible",
            "--fallback-endpoint",
            "https://fallback.example.test/v1/embeddings",
            "--fallback-api-key-env",
            "FALLBACK_EMBED_KEY",
        ],
    );

    let config = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(
        config.embedding.fallback,
        Some(EmbeddingFallbackConfig {
            provider: Provider::OpenaiCompatible,
            endpoint: Some("https://fallback.example.test/v1/embeddings".into()),
            api_key_env: Some("FALLBACK_EMBED_KEY".into()),
        })
    );
}

#[test]
fn init_accepts_custom_provider_endpoint_and_dimensions() {
    let temp = tempfile::tempdir().unwrap();

    run_init(
        temp.path(),
        &[
            "--provider",
            "http",
            "--model",
            "custom-embedder",
            "--endpoint",
            "https://api.example.com/embeddings",
            "--api-key-env",
            "CUSTOM_EMBEDDING_KEY",
            "--dimensions",
            "1024",
        ],
    );

    let config = read_config(&temp.path().join(".enf.toml"));
    assert_eq!(config.embedding.provider, Provider::Http);
    assert_eq!(config.embedding.engine, None);
    assert_eq!(config.embedding.variant, None);
    assert_eq!(config.embedding.model, "custom-embedder");
    assert_eq!(
        config.embedding.endpoint.as_deref(),
        Some("https://api.example.com/embeddings")
    );
    assert_eq!(
        config.embedding.api_key_env.as_deref(),
        Some("CUSTOM_EMBEDDING_KEY")
    );
    assert_eq!(config.embedding.dimensions, 1024);
}

#[test]
fn init_refuses_to_overwrite_existing_config_without_force() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join(".enf.toml");
    fs::write(&config_path, "sentinel = true\n").unwrap();

    Command::cargo_bin("enf")
        .unwrap()
        .current_dir(temp.path())
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
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
        .env("ENF_SKIP_NATIVE_MODEL_LOAD", "1")
        .args(["init", "--db=false"])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "--no-db requires an existing database",
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
