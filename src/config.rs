use std::io::{self, IsTerminal};
use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use dialoguer::{Confirm, Input, Select};

use crate::{
    cli::{
        ChunkingModeArg, DbArg, InitArgs, ModelCacheArg, ModelVariantArg, ProviderArg,
        ProviderOverrideArgs, SearchLevelArg, SearchModeArg,
    },
    db,
    errors::EnfError,
};

pub const CONFIG_FILE: &str = ".enf.toml";
pub const LOCAL_CONFIG_FILE: &str = ".enf.local.toml";
pub const STATE_DIR: &str = ".enf";
pub const DB_PATH: &str = ".enf/index.sqlite";
pub const CACHE_DIR: &str = ".enf/cache";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub version: u32,
    pub state: StateConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub index: IndexConfig,
    #[serde(default)]
    pub text: TextConfig,
    #[serde(default)]
    pub image: ImageConfig,
    #[serde(default)]
    pub reranker: RerankerConfig,
    #[serde(default, skip_serializing_if = "PatternConfig::is_empty")]
    pub include: PatternConfig,
    #[serde(default, skip_serializing_if = "PatternConfig::is_empty")]
    pub exclude: PatternConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateConfig {
    pub db_path: String,
    pub cache_dir: String,
    pub model_cache: ModelCache,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCache {
    Global,
    Project,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EmbeddingConfig {
    pub provider: Provider,
    pub engine: Option<String>,
    pub model: String,
    pub variant: Option<ModelVariant>,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub dimensions: usize,
    #[serde(default)]
    pub fallback: Option<EmbeddingFallbackConfig>,
    pub batch_size: usize,
    pub query_cache: bool,
    pub document_prefix: String,
    pub query_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EmbeddingFallbackConfig {
    pub provider: Provider,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Provider {
    Native,
    Ollama,
    Openai,
    OpenaiCompatible,
    Http,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    Quantized,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchConfig {
    pub default_mode: SearchMode,
    pub default_level: SearchLevel,
    pub limit: usize,
    pub vector_weight: f32,
    pub keyword_weight: f32,
    pub metadata_weight: f32,
    pub batch_scan_size: usize,
    pub max_chunks_per_file: usize,
    pub snippet_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum SearchLevel {
    Chunk,
    File,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    #[serde(default = "ChunkingMode::default")]
    pub chunking: ChunkingMode,
    pub chunk_target_tokens: usize,
    pub chunk_max_tokens: usize,
    pub chunk_overlap_tokens: usize,
    pub min_chunk_chars: usize,
    pub hash_algorithm: String,
    pub store_full_files: bool,
    pub store_chunks: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ChunkingMode {
    #[default]
    Smart,
    LineWindow,
    Off,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PatternConfig {
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextConfig {
    pub include: PatternConfig,
    pub exclude: PatternConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageConfig {
    pub include: PatternConfig,
    pub exclude: PatternConfig,
    pub embedding: ImageEmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct ImageEmbeddingConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    #[serde(default)]
    pub query_endpoint: Option<String>,
    pub model: String,
    pub dimensions: usize,
    pub batch_size: usize,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct RerankerConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub model: String,
    pub candidate_limit: usize,
    pub timeout_seconds: u64,
    pub raw_scores: bool,
    pub return_text: bool,
    pub truncate: bool,
    pub truncation_direction: String,
}

impl PatternConfig {
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            include: PatternConfig {
                patterns: default_text_include_patterns(),
            },
            exclude: PatternConfig {
                patterns: default_exclude_patterns(),
            },
        }
    }
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            include: PatternConfig {
                patterns: default_image_include_patterns(),
            },
            exclude: PatternConfig {
                patterns: default_exclude_patterns(),
            },
            embedding: ImageEmbeddingConfig::default(),
        }
    }
}

impl Default for ImageEmbeddingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            query_endpoint: None,
            model: "open_clip/ViT-H-14:laion2b_s32b_b79k".into(),
            dimensions: 1024,
            batch_size: 16,
            normalize: true,
        }
    }
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            model: "BAAI/bge-reranker-v2-m3".into(),
            candidate_limit: 50,
            timeout_seconds: 30,
            raw_scores: false,
            return_text: true,
            truncate: true,
            truncation_direction: "right".into(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            state: StateConfig {
                db_path: DB_PATH.into(),
                cache_dir: CACHE_DIR.into(),
                model_cache: ModelCache::Global,
            },
            embedding: EmbeddingConfig {
                provider: Provider::Native,
                engine: Some("candle".into()),
                model: "nomic-embed-text-v1.5".into(),
                variant: Some(ModelVariant::Quantized),
                endpoint: None,
                api_key_env: None,
                dimensions: 768,
                fallback: None,
                batch_size: 64,
                query_cache: true,
                document_prefix: "search_document: ".into(),
                query_prefix: "search_query: ".into(),
            },
            search: SearchConfig {
                default_mode: SearchMode::Hybrid,
                default_level: SearchLevel::Chunk,
                limit: 10,
                vector_weight: 0.65,
                keyword_weight: 0.30,
                metadata_weight: 0.05,
                batch_scan_size: 1024,
                max_chunks_per_file: 3,
                snippet_chars: 700,
            },
            index: IndexConfig {
                chunking: ChunkingMode::Smart,
                chunk_target_tokens: 450,
                chunk_max_tokens: 900,
                chunk_overlap_tokens: 80,
                min_chunk_chars: 80,
                hash_algorithm: "blake3".into(),
                store_full_files: true,
                store_chunks: true,
            },
            text: TextConfig::default(),
            image: ImageConfig::default(),
            reranker: RerankerConfig::default(),
            include: PatternConfig::default(),
            exclude: PatternConfig::default(),
        }
    }
}

fn default_text_include_patterns() -> Vec<String> {
    vec![
        "AGENTS.md",
        "README*",
        "CHANGELOG*",
        "CONTRIBUTING*",
        "docs/**",
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
        "**/*.md",
        "**/*.mdx",
        "**/*.txt",
        "**/*.rst",
        "**/*.adoc",
        "**/*.org",
        "**/*.ehmeta",
        "**/*.docx",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn default_image_include_patterns() -> Vec<String> {
    vec![
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.webp",
        "**/*.gif",
        "**/*.bmp",
        "**/*.tif",
        "**/*.tiff",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn default_exclude_patterns() -> Vec<String> {
    vec![
        ".git/**",
        ".enf/**",
        ".enf.toml",
        ".enf.local.toml",
        "node_modules/**",
        "dist/**",
        "build/**",
        "target/**",
        ".next/**",
        ".venv/**",
        "__pycache__/**",
        "*.lock",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn init(args: InitArgs) -> Result<()> {
    let cwd = std::env::current_dir().context("reading current directory")?;
    let config_path = cwd.join(CONFIG_FILE);
    if config_path.exists() && !args.force {
        return Err(EnfError::ConfigExists {
            path: config_path.display().to_string(),
        }
        .into());
    }

    let interactive = should_prompt_for_init(&args);
    if interactive {
        print_interactive_init_intro(&cwd);
    }

    let config = init_config(&args);
    validate(&config)?;
    let db_enabled = args.db == DbArg::Sqlite && !args.no_db;
    let db_path = cwd.join(&config.state.db_path);

    if !db_enabled && !db_path.exists() {
        anyhow::bail!(
            "--no-db requires an existing database at {}; use `enf init` or `enf init --db=sqlite` to create one",
            config.state.db_path
        );
    }
    if !db_enabled && args.index {
        anyhow::bail!("--index requires database initialization; use `enf init --index` or run `enf index .` later");
    }

    if args.dry_run {
        println!("Plan");
        println!("  Create {CONFIG_FILE}");
        println!("  config: {}", CONFIG_FILE);
        if db_enabled {
            println!("  database: {}", config.state.db_path);
        } else {
            println!("  database: existing database required, creation skipped");
        }
        println!("  provider: {}", config.embedding.provider.as_str());
        if args.install_models || config.embedding.provider == Provider::Native {
            println!("  model action: install active model profile");
        }
        if args.index {
            println!("  index action: index current directory");
        }
        return Ok(());
    }

    fs::create_dir_all(cwd.join(STATE_DIR)).context("creating .enf directory")?;
    fs::create_dir_all(cwd.join(CACHE_DIR)).context("creating .enf/cache directory")?;
    write_config(&config_path, &config)?;
    ensure_gitignore(&cwd.join(".gitignore"))?;
    if db_enabled {
        if db_path.exists() {
            eprintln!(
                "warning: database already exists at {}; leaving existing data in place",
                config.state.db_path
            );
        }
        println!("==> Preparing SQLite index");
        db::open_or_create(&db_path)?;
    }

    if args.install_models || config.embedding.provider == Provider::Native {
        if config.embedding.provider == Provider::Native && !native_candle_available() {
            println!("==> Recording native embedding profile");
            eprintln!(
                "warning: this build does not include native Candle embeddings; use an Ollama/OpenAI/HTTP provider or build from source with native-candle enabled before indexing"
            );
        } else {
            println!("==> Installing native embedding model");
        }
        crate::models::install_active_model(&config)?;
    }

    if args.index {
        crate::index::index_path(
            &cwd,
            PathBuf::from("."),
            &config,
            crate::index::EmbedOptions {
                install_models: args.install_models,
                no_embed: false,
                ..Default::default()
            },
        )?;
    }

    println!("✓ Wrote {CONFIG_FILE}");
    if db_enabled {
        println!("✓ Prepared SQLite index at {}", config.state.db_path);
    } else {
        println!("✓ Database: existing database required, creation skipped");
    }
    println!("✓ Provider: {}", provider_label(&config));
    println!();
    println!("Ready.");
    println!();
    println!("Try:");
    println!("  enf search \"your query\"");
    println!("  enf status");
    Ok(())
}

fn should_prompt_for_init(args: &InitArgs) -> bool {
    (args.interactive || (!args.no_input && !args.yes && args.preset.is_none()))
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
}

fn print_interactive_init_intro(cwd: &std::path::Path) {
    println!("Elephant Never Forgets");
    println!("Set up semantic search for this project.");
    println!();
    println!("Project");
    println!("  Folder: {}", cwd.display());
    println!("  Config: {CONFIG_FILE}");
    println!("  Index:  {DB_PATH}");
    println!();
    println!("Using recommended local preset. Change it later with `enf setup`.");
    println!();
}

fn provider_label(config: &Config) -> String {
    match config.embedding.provider {
        Provider::Native => format!("local packaged model / {}", config.embedding.model),
        Provider::Ollama => format!(
            "Ollama / {}",
            config
                .embedding
                .endpoint
                .as_deref()
                .unwrap_or("http://localhost:11434/api/embed")
        ),
        Provider::Openai => format!(
            "OpenAI / {}",
            config
                .embedding
                .api_key_env
                .as_deref()
                .unwrap_or("OPENAI_API_KEY")
        ),
        Provider::OpenaiCompatible => "OpenAI-compatible HTTP endpoint".into(),
        Provider::Http => "custom HTTP endpoint".into(),
    }
}

fn native_candle_available() -> bool {
    cfg!(feature = "native-candle")
}

pub fn init_config(args: &InitArgs) -> Config {
    let mut config = args
        .preset
        .clone()
        .map(crate::setup::preset_config)
        .unwrap_or_default();
    apply_init_overrides(&mut config, args);
    config
}

pub fn load() -> Result<Config> {
    let path = std::env::current_dir()?.join(CONFIG_FILE);
    if !path.exists() {
        return Err(EnfError::NotInitialized.into());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut config: Config =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    apply_local_overrides(&mut config)?;
    apply_legacy_pattern_compat(&mut config);
    validate(&config)?;
    Ok(config)
}

fn apply_local_overrides(config: &mut Config) -> Result<()> {
    let path = std::env::current_dir()?.join(LOCAL_CONFIG_FILE);
    if !path.exists() {
        return Ok(());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;

    if let Some(embedding) = value.get("embedding").and_then(toml::Value::as_table) {
        if let Some(provider) = table_str(embedding, "provider") {
            config.embedding.provider = parse_provider_value(provider)?;
        }
        if let Some(engine) = table_optional_str(embedding, "engine") {
            config.embedding.engine = engine;
        }
        if let Some(model) = table_str(embedding, "model") {
            config.embedding.model = model.to_string();
        }
        if let Some(variant) = table_optional_str(embedding, "variant") {
            config.embedding.variant = variant
                .as_deref()
                .map(parse_variant_value)
                .transpose()
                .with_context(|| format!("parsing {LOCAL_CONFIG_FILE} embedding.variant"))?;
        }
        if let Some(endpoint) = table_optional_str(embedding, "endpoint") {
            config.embedding.endpoint = endpoint;
        }
        if let Some(api_key_env) = table_optional_str(embedding, "api_key_env") {
            config.embedding.api_key_env = api_key_env;
        }
        if let Some(dimensions) = table_usize(embedding, "dimensions")? {
            config.embedding.dimensions = dimensions;
        }
    }

    if let Some(search) = value.get("search").and_then(toml::Value::as_table) {
        if let Some(mode) = table_str(search, "default_mode") {
            config.search.default_mode = parse_search_mode_value(mode)?;
        }
        if let Some(level) = table_str(search, "default_level") {
            config.search.default_level = parse_search_level_value(level)?;
        }
        if let Some(limit) = table_usize(search, "limit")? {
            config.search.limit = limit;
        }
    }

    if let Some(reranker) = value.get("reranker").and_then(toml::Value::as_table) {
        if let Some(enabled) = table_bool(reranker, "enabled")? {
            config.reranker.enabled = enabled;
        }
        if let Some(endpoint) = table_optional_str(reranker, "endpoint") {
            config.reranker.endpoint = endpoint;
        }
        if let Some(model) = table_str(reranker, "model") {
            config.reranker.model = model.to_string();
        }
        if let Some(candidate_limit) = table_usize(reranker, "candidate_limit")? {
            config.reranker.candidate_limit = candidate_limit;
        }
    }

    if let Some(image) = value
        .get("image")
        .and_then(|image| image.get("embedding"))
        .and_then(toml::Value::as_table)
    {
        if let Some(enabled) = table_bool(image, "enabled")? {
            config.image.embedding.enabled = enabled;
        }
        if let Some(endpoint) = table_optional_str(image, "endpoint") {
            config.image.embedding.endpoint = endpoint;
        }
        if let Some(query_endpoint) = table_optional_str(image, "query_endpoint") {
            config.image.embedding.query_endpoint = query_endpoint;
        }
        if let Some(model) = table_str(image, "model") {
            config.image.embedding.model = model.to_string();
        }
        if let Some(dimensions) = table_usize(image, "dimensions")? {
            config.image.embedding.dimensions = dimensions;
        }
    }

    Ok(())
}

fn table_str<'a>(table: &'a toml::map::Map<String, toml::Value>, key: &str) -> Option<&'a str> {
    table.get(key).and_then(toml::Value::as_str)
}

fn table_optional_str(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
) -> Option<Option<String>> {
    table.get(key).map(|value| {
        value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn table_usize(table: &toml::map::Map<String, toml::Value>, key: &str) -> Result<Option<usize>> {
    table
        .get(key)
        .map(|value| {
            value
                .as_integer()
                .and_then(|value| usize::try_from(value).ok())
                .with_context(|| format!("{key} must be a non-negative integer"))
        })
        .transpose()
}

fn table_bool(table: &toml::map::Map<String, toml::Value>, key: &str) -> Result<Option<bool>> {
    table
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("{key} must be true or false"))
        })
        .transpose()
}

pub fn write_config(path: &std::path::Path, config: &Config) -> Result<()> {
    let text = format_config(config)?;
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn format_config(config: &Config) -> Result<String> {
    let text = toml::to_string_pretty(config).context("serializing config")?;
    Ok(format!(
        "# Elephant Never Forgets project configuration\n\
         #\n\
         # This file controls what ENF indexes, how embeddings are created, and how\n\
         # search results are ranked. It is safe to edit by hand.\n\
         #\n\
         # Common commands:\n\
         #   enf status                 Show whether this setup is ready\n\
         #   enf doctor                 Diagnose problems in this file and the index\n\
         #   enf config explain <key>   Explain an important setting\n\
         #   enf setup                  Change provider/model/features interactively\n\
         #\n\
         # Commit this file if the team should share ENF settings.\n\
         # Do not commit .enf/; it contains the local SQLite index and cache.\n\
         # Put machine-local overrides and secrets in .enf.local.toml.\n\
         #\n\
         # Presets:\n\
         #   enf setup presets\n\
         #   enf setup use local|code|docs|ollama|openai|custom|keyword\n\
         #\n\
         # Provider options:\n\
         #   native              packaged local embeddings, no API key or server\n\
         #   ollama              Ollama /api/embed endpoint\n\
         #   openai              OpenAI embeddings endpoint\n\
         #   openai-compatible   any OpenAI-shaped embeddings endpoint\n\
         #   http                ENF custom embeddings response shape\n\
         #\n\
         # Optional endpoint-backed features:\n\
         #   enf setup reranker  Configure search result reranking\n\
         #   enf setup images    Configure visual image search\n\n\
         {text}"
    ))
}

pub fn validate(config: &Config) -> Result<()> {
    if config.version != 1 {
        anyhow::bail!("unsupported config version {}", config.version);
    }
    validate_state(config)?;
    validate_search(config)?;
    validate_index(config)?;
    validate_embedding(config)?;
    validate_image(config)?;
    validate_reranker(config)?;
    Ok(())
}

fn apply_legacy_pattern_compat(config: &mut Config) {
    if !config.include.patterns.is_empty()
        && config.text.include.patterns == default_text_include_patterns()
    {
        config.text.include = config.include.clone();
    }
    if !config.exclude.patterns.is_empty()
        && config.text.exclude.patterns == default_exclude_patterns()
    {
        config.text.exclude = config.exclude.clone();
    }
}

fn validate_state(config: &Config) -> Result<()> {
    validate_non_empty_path(
        "state.db_path",
        &config.state.db_path,
        "use the default .enf/index.sqlite or another project-relative SQLite path",
    )?;
    validate_non_empty_path(
        "state.cache_dir",
        &config.state.cache_dir,
        "use the default .enf/cache or another project-relative cache directory",
    )?;
    Ok(())
}

fn validate_search(config: &Config) -> Result<()> {
    validate_non_negative_weight("search.vector_weight", config.search.vector_weight)?;
    validate_non_negative_weight("search.keyword_weight", config.search.keyword_weight)?;
    validate_non_negative_weight("search.metadata_weight", config.search.metadata_weight)?;
    let total_weight =
        config.search.vector_weight + config.search.keyword_weight + config.search.metadata_weight;
    if total_weight <= 0.0 {
        anyhow::bail!(
            "search.vector_weight + search.keyword_weight + search.metadata_weight must be greater than 0.0"
        );
    }
    if config.search.limit == 0 {
        anyhow::bail!("search.limit must be greater than 0");
    }
    if config.search.batch_scan_size == 0 {
        anyhow::bail!("search.batch_scan_size must be greater than 0");
    }
    if config.search.max_chunks_per_file == 0 {
        anyhow::bail!("search.max_chunks_per_file must be greater than 0");
    }
    if config.search.snippet_chars == 0 {
        anyhow::bail!("search.snippet_chars must be greater than 0");
    }
    Ok(())
}

fn validate_index(config: &Config) -> Result<()> {
    if config.index.chunk_target_tokens == 0 {
        anyhow::bail!("index.chunk_target_tokens must be greater than 0");
    }
    if config.index.chunk_max_tokens == 0 {
        anyhow::bail!("index.chunk_max_tokens must be greater than 0");
    }
    if config.index.chunk_overlap_tokens >= config.index.chunk_max_tokens {
        anyhow::bail!("index.chunk_overlap_tokens must be less than index.chunk_max_tokens");
    }
    if config.index.min_chunk_chars == 0 {
        anyhow::bail!("index.min_chunk_chars must be greater than 0");
    }
    Ok(())
}

fn validate_embedding(config: &Config) -> Result<()> {
    validate_non_empty_path(
        "embedding.model",
        &config.embedding.model,
        "choose a model name for the selected provider",
    )?;
    if config.embedding.dimensions == 0 {
        anyhow::bail!("embedding.dimensions must be greater than 0");
    }

    match config.embedding.provider {
        Provider::Native => {
            if !matches!(
                config.embedding.engine.as_deref(),
                Some("candle") | Some("fastembed-candle") | Some("fastembed")
            ) {
                anyhow::bail!(
                    "embedding.engine must be \"candle\" when embedding.provider = \"native\""
                );
            }
            if config.embedding.endpoint.is_some() {
                anyhow::bail!(
                    "embedding.endpoint must be omitted when embedding.provider = \"native\""
                );
            }
            if config.embedding.api_key_env.is_some() {
                anyhow::bail!(
                    "embedding.api_key_env must be omitted when embedding.provider = \"native\""
                );
            }
            if config.embedding.variant.is_none() {
                anyhow::bail!(
                    "embedding.variant is required when embedding.provider = \"native\"; use \"quantized\""
                );
            }
            if config.embedding.variant != Some(ModelVariant::Quantized) {
                anyhow::bail!(
                    "embedding.variant must be \"quantized\" for the native Candle model"
                );
            }
        }
        Provider::Ollama => {
            if config.embedding.engine.is_some() {
                anyhow::bail!(
                    "embedding.engine must be omitted when embedding.provider = \"ollama\""
                );
            }
            if config
                .embedding
                .endpoint
                .as_deref()
                .map(str::trim)
                .filter(|endpoint| !endpoint.is_empty())
                .is_none()
            {
                anyhow::bail!(
                    "embedding.endpoint is required when embedding.provider = \"ollama\"; use the Ollama embeddings URL"
                );
            }
            if config.embedding.api_key_env.is_some() {
                anyhow::bail!(
                    "embedding.api_key_env must be omitted when embedding.provider = \"ollama\""
                );
            }
            if config.embedding.variant.is_some() {
                anyhow::bail!(
                    "embedding.variant must be omitted when embedding.provider = \"ollama\""
                );
            }
        }
        Provider::Openai => {
            if config.embedding.engine.is_some() {
                anyhow::bail!(
                    "embedding.engine must be omitted when embedding.provider = \"openai\""
                );
            }
            if config
                .embedding
                .endpoint
                .as_deref()
                .map(str::trim)
                .filter(|endpoint| !endpoint.is_empty())
                .is_none()
            {
                anyhow::bail!(
                    "embedding.endpoint is required when embedding.provider = \"openai\"; use https://api.openai.com/v1/embeddings"
                );
            }
            if config
                .embedding
                .api_key_env
                .as_deref()
                .map(str::trim)
                .filter(|env| !env.is_empty())
                .is_none()
            {
                anyhow::bail!(
                    "embedding.api_key_env is required when embedding.provider = \"openai\"; use OPENAI_API_KEY"
                );
            }
            if config.embedding.variant.is_some() {
                anyhow::bail!(
                    "embedding.variant must be omitted when embedding.provider = \"openai\""
                );
            }
        }
        Provider::OpenaiCompatible | Provider::Http => {}
    }

    if let Some(fallback) = &config.embedding.fallback {
        validate_fallback_embedding(config, fallback)?;
    }

    Ok(())
}

fn validate_image(config: &Config) -> Result<()> {
    if config.image.embedding.dimensions == 0 {
        anyhow::bail!("image.embedding.dimensions must be greater than 0");
    }
    if config.image.embedding.batch_size == 0 {
        anyhow::bail!("image.embedding.batch_size must be greater than 0");
    }
    if config.image.embedding.enabled
        && config
            .image
            .embedding
            .endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .is_none()
    {
        anyhow::bail!("image.embedding.endpoint is required when image.embedding.enabled = true");
    }
    if config
        .image
        .embedding
        .query_endpoint
        .as_deref()
        .is_some_and(|endpoint| endpoint.trim().is_empty())
    {
        anyhow::bail!("image.embedding.query_endpoint must not be empty when set");
    }
    Ok(())
}

fn validate_reranker(config: &Config) -> Result<()> {
    if config.reranker.candidate_limit == 0 {
        anyhow::bail!("reranker.candidate_limit must be greater than 0");
    }
    if config.reranker.timeout_seconds == 0 {
        anyhow::bail!("reranker.timeout_seconds must be greater than 0");
    }
    if config.reranker.enabled
        && config
            .reranker
            .endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .is_none()
    {
        anyhow::bail!("reranker.endpoint is required when reranker.enabled = true");
    }
    Ok(())
}

fn validate_fallback_embedding(config: &Config, fallback: &EmbeddingFallbackConfig) -> Result<()> {
    let mut fallback_config = config.clone();
    fallback_config.embedding.fallback = None;
    fallback_config.embedding.provider = fallback.provider.clone();
    fallback_config.embedding.endpoint = fallback.endpoint.clone();
    fallback_config.embedding.api_key_env = fallback.api_key_env.clone();
    fallback_config.embedding.model = normalize_model_aliases_for_provider(
        &config.embedding.model,
        &fallback_config.embedding.provider,
    );
    normalize_provider_defaults(&mut fallback_config);

    if fallback_config.embedding.provider == Provider::Native {
        if fallback_config.embedding.endpoint.is_some() {
            anyhow::bail!(
                "embedding.fallback.endpoint must be omitted when embedding.fallback.provider = \"native\""
            );
        }
        if fallback_config.embedding.api_key_env.is_some() {
            anyhow::bail!(
                "embedding.fallback.api-key-env must be omitted when embedding.fallback.provider = \"native\""
            );
        }
    }

    validate_embedding(&fallback_config).map_err(|err| {
        err.context("embedding.fallback is invalid. Configure [embedding.fallback] consistently")
    })?;
    Ok(())
}

fn validate_non_empty_path(field: &str, value: &str, hint: &str) -> Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{} must not be empty; {}", field, hint);
    }
    Ok(())
}

fn validate_non_negative_weight(field: &str, value: f32) -> Result<()> {
    if value < 0.0 {
        anyhow::bail!("{} must be greater than or equal to 0.0", field);
    }
    Ok(())
}

fn apply_init_overrides(config: &mut Config, args: &InitArgs) {
    let wants_interactive = should_prompt_for_init(args);
    let interactive = if wants_interactive {
        interactive_init_defaults(config)
    } else {
        Ok(InteractiveInitDefaults::default())
    };

    let interactive = match interactive {
        Ok(values) => values,
        Err(err) => {
            if wants_interactive {
                log_interactive_warning(&err);
                InteractiveInitDefaults::default()
            } else {
                InteractiveInitDefaults::default()
            }
        }
    };

    let provider = args
        .provider
        .clone()
        .or(interactive.provider)
        .or_else(|| (args.native_embed || args.local_embed).then_some(ProviderArg::Native))
        .unwrap_or(match config.embedding.provider {
            Provider::Native => ProviderArg::Native,
            Provider::Ollama => ProviderArg::Ollama,
            Provider::Openai => ProviderArg::Openai,
            Provider::OpenaiCompatible => ProviderArg::OpenaiCompatible,
            Provider::Http => ProviderArg::Http,
        });
    config.embedding.provider = provider.into();

    if let Some(model) = args.model.clone().or(interactive.model) {
        config.embedding.model = model;
    }
    if let Some(variant) = &args.variant {
        config.embedding.variant = Some(variant.clone().into());
    }
    config.embedding.model =
        normalize_model_aliases_for_provider(&config.embedding.model, &config.embedding.provider);
    if let Some(chunking) = args.chunking.clone().or(interactive.chunking) {
        config.index.chunking = chunking.into();
    }
    if let Some(model_cache) = &args.model_cache {
        config.state.model_cache = model_cache.clone().into();
    }
    normalize_provider_defaults(config);
    if let Some(endpoint) = &args.endpoint {
        config.embedding.endpoint = Some(endpoint.clone());
    }
    if let Some(api_key_env) = &args.api_key_env {
        config.embedding.api_key_env = Some(api_key_env.clone());
    }
    if let Some(dimensions) = args.dimensions {
        config.embedding.dimensions = dimensions;
    }
    normalize_provider_defaults(config);
    if args.dimensions.is_none() {
        config.embedding.dimensions = default_model_dimensions(config);
    }
    if let Some(fallback_provider) = args
        .fallback_provider
        .clone()
        .or(interactive.fallback_provider)
    {
        config.embedding.fallback = Some(EmbeddingFallbackConfig {
            provider: fallback_provider.into(),
            endpoint: args
                .fallback_endpoint
                .clone()
                .or(interactive.fallback_endpoint),
            api_key_env: args
                .fallback_api_key_env
                .clone()
                .or(interactive.fallback_api_key_env),
        });
    } else {
        config.embedding.fallback = None;
    }
    if let Some(fallback) = &mut config.embedding.fallback {
        if fallback.provider == Provider::Native {
            fallback.endpoint = None;
            fallback.api_key_env = None;
        }
    }
    if config.embedding.fallback.is_some() {
        config.embedding.fallback = Some(normalize_fallback_config(
            config
                .embedding
                .fallback
                .clone()
                .unwrap_or(EmbeddingFallbackConfig {
                    provider: config.embedding.provider.clone(),
                    endpoint: None,
                    api_key_env: None,
                }),
            config,
        ));
    }
}

pub fn normalize_provider_defaults(config: &mut Config) {
    config.embedding.model =
        normalize_model_aliases_for_provider(&config.embedding.model, &config.embedding.provider);
    match config.embedding.provider {
        Provider::Native => {
            config.embedding.engine = Some("candle".into());
            config.embedding.endpoint = None;
            config.embedding.api_key_env = None;
            if config.embedding.variant.is_none() {
                config.embedding.variant = Some(ModelVariant::Quantized);
            }
        }
        Provider::Ollama => {
            config.embedding.engine = None;
            config.embedding.endpoint = Some("http://localhost:11434/api/embed".into());
            config.embedding.variant = None;
        }
        Provider::Openai => {
            config.embedding.engine = None;
            config.embedding.endpoint = Some("https://api.openai.com/v1/embeddings".into());
            config.embedding.api_key_env = Some("OPENAI_API_KEY".into());
            config.embedding.variant = None;
            if is_nomic_model(&config.embedding.model) {
                config.embedding.model = "text-embedding-3-small".into();
            }
            config.embedding.document_prefix.clear();
            config.embedding.query_prefix.clear();
        }
        Provider::OpenaiCompatible | Provider::Http => {
            config.embedding.engine = None;
            config.embedding.variant = None;
        }
    }
    if config.embedding.provider != Provider::Openai {
        config.embedding.document_prefix = default_document_prefix(&config.embedding.model);
        config.embedding.query_prefix = default_query_prefix(&config.embedding.model);
    }
    config.embedding.dimensions = default_model_dimensions(config);
}

pub fn apply_provider_overrides(config: &mut Config, overrides: &ProviderOverrideArgs) {
    let has_overrides = overrides.provider.is_some()
        || overrides.model.is_some()
        || overrides.variant.is_some()
        || overrides.endpoint.is_some()
        || overrides.api_key_env.is_some()
        || overrides.dimensions.is_some();
    if !has_overrides {
        return;
    }

    if let Some(provider) = &overrides.provider {
        config.embedding.provider = provider.clone().into();
        normalize_provider_defaults(config);
    }
    if let Some(model) = &overrides.model {
        config.embedding.model = model.clone();
        config.embedding.model = normalize_model_aliases_for_provider(
            &config.embedding.model,
            &config.embedding.provider,
        );
        normalize_provider_defaults(config);
    }
    if let Some(variant) = &overrides.variant {
        config.embedding.variant = Some(variant.clone().into());
    }
    if let Some(endpoint) = &overrides.endpoint {
        config.embedding.endpoint = Some(endpoint.clone());
    }
    if let Some(api_key_env) = &overrides.api_key_env {
        config.embedding.api_key_env = Some(api_key_env.clone());
    }
    if let Some(dimensions) = overrides.dimensions {
        config.embedding.dimensions = dimensions;
    } else {
        config.embedding.dimensions = default_model_dimensions(config);
    }
}

pub fn normalize_model_aliases_for_provider(model: &str, provider: &Provider) -> String {
    match model {
        "nomic" | "nomic-embed-text" | "nomic-embed-text-v1.5" | "nomic-embed-text-v2-moe" => {
            "nomic-embed-text-v1.5".into()
        }
        "gemma" | "embeddinggemma-300m" | "google" | "google/embeddinggemma-300m"
            if *provider == Provider::Native =>
        {
            "google/embeddinggemma-300m".into()
        }
        "gemma" | "embeddinggemma-300m" | "embeddinggemma:300m" => "embeddinggemma:300m".into(),
        _ => model.to_string(),
    }
}

fn default_model_dimensions(config: &Config) -> usize {
    if is_gemma_model(&config.embedding.model) || is_nomic_model(&config.embedding.model) {
        768
    } else if config.embedding.provider == Provider::Openai {
        openai_dimensions(&config.embedding.model)
    } else {
        config.embedding.dimensions
    }
}

fn is_gemma_model(model: &str) -> bool {
    matches!(
        model,
        "google/embeddinggemma-300m" | "embeddinggemma-300m" | "embeddinggemma:300m" | "gemma"
    )
}

fn is_nomic_model(model: &str) -> bool {
    matches!(
        model,
        "nomic-embed-text" | "nomic-embed-text-v1.5" | "nomic-embed-text-v2-moe" | "nomic"
    )
}

fn default_document_prefix(model: &str) -> String {
    if is_gemma_model(model) {
        "title: none | text: ".into()
    } else {
        "search_document: ".into()
    }
}

fn default_query_prefix(model: &str) -> String {
    if is_gemma_model(model) {
        "task: search result | query: ".into()
    } else {
        "search_query: ".into()
    }
}

fn interactive_init_defaults(config: &Config) -> Result<InteractiveInitDefaults> {
    run_interactive_init_prompt(config)
}

#[derive(Default)]
struct InteractiveInitDefaults {
    pub provider: Option<ProviderArg>,
    pub model: Option<String>,
    pub chunking: Option<ChunkingModeArg>,
    pub fallback_provider: Option<ProviderArg>,
    pub fallback_endpoint: Option<String>,
    pub fallback_api_key_env: Option<String>,
}

fn run_interactive_init_prompt(config: &Config) -> Result<InteractiveInitDefaults> {
    if !io::stdin().is_terminal() {
        anyhow::bail!("--interactive requires a TTY");
    }
    let mut answers = InteractiveInitDefaults::default();

    let provider_items = [
        "Local packaged model",
        "Ollama",
        "OpenAI",
        "Custom OpenAI-compatible endpoint",
        "Custom ENF HTTP endpoint",
        "Keyword only",
    ];
    let provider = Select::new()
        .with_prompt("Embeddings")
        .items(&provider_items)
        .default(0)
        .interact()?;
    answers.provider = match provider {
        0 | 5 => Some(ProviderArg::Native),
        1 => Some(ProviderArg::Ollama),
        2 => Some(ProviderArg::Openai),
        3 => Some(ProviderArg::OpenaiCompatible),
        4 => Some(ProviderArg::Http),
        _ => None,
    };

    let model_items = ["Nomic Embed Text v1.5", "EmbeddingGemma 300M", "Custom"];
    let model_choice = Select::new()
        .with_prompt("Local model")
        .items(&model_items)
        .default(0)
        .interact()?;
    answers.model = Some(match model_choice {
        0 => "nomic-embed-text-v1.5".into(),
        1 => "google/embeddinggemma-300m".into(),
        _ => Input::new()
            .with_prompt("Model")
            .default(config.embedding.model.clone())
            .interact_text()?,
    });

    let chunking_items = ["Smart", "Line window", "Off"];
    let chunking = Select::new()
        .with_prompt("Chunking")
        .items(&chunking_items)
        .default(0)
        .interact()?;
    answers.chunking = Some(match chunking {
        0 => ChunkingModeArg::Smart,
        1 => ChunkingModeArg::LineWindow,
        _ => ChunkingModeArg::Off,
    });

    if Confirm::new()
        .with_prompt("Configure fallback provider now?")
        .default(false)
        .interact()?
    {
        let fallback_provider: String = Input::new()
            .with_prompt("Fallback provider")
            .default("ollama".into())
            .interact_text()?;
        answers.fallback_provider = Some(parse_provider_arg(&fallback_provider)?);
        let fallback_endpoint: String = Input::new()
            .with_prompt("Fallback endpoint")
            .allow_empty(true)
            .interact_text()?;
        if !fallback_endpoint.trim().is_empty() {
            answers.fallback_endpoint = Some(fallback_endpoint);
        }
        let fallback_api_key_env: String = Input::new()
            .with_prompt("Fallback API key env")
            .allow_empty(true)
            .interact_text()?;
        if !fallback_api_key_env.trim().is_empty() {
            answers.fallback_api_key_env = Some(fallback_api_key_env);
        }
    }

    Ok(answers)
}

fn parse_provider_arg(value: &str) -> Result<ProviderArg> {
    match value.to_lowercase().as_str() {
        "native" => Ok(ProviderArg::Native),
        "ollama" => Ok(ProviderArg::Ollama),
        "openai" => Ok(ProviderArg::Openai),
        "openai-compatible" => Ok(ProviderArg::OpenaiCompatible),
        "http" => Ok(ProviderArg::Http),
        _ => anyhow::bail!("expected one of native, ollama, openai, openai-compatible, http"),
    }
}

fn parse_provider_value(value: &str) -> Result<Provider> {
    Ok(parse_provider_arg(value)?.into())
}

fn parse_variant_value(value: &str) -> Result<ModelVariant> {
    match value {
        "quantized" => Ok(ModelVariant::Quantized),
        "full" => Ok(ModelVariant::Full),
        _ => anyhow::bail!("unknown model variant `{value}`"),
    }
}

fn parse_search_mode_value(value: &str) -> Result<SearchMode> {
    match value {
        "hybrid" => Ok(SearchMode::Hybrid),
        "vector" => Ok(SearchMode::Vector),
        "keyword" => Ok(SearchMode::Keyword),
        _ => anyhow::bail!("unknown search mode `{value}`"),
    }
}

fn parse_search_level_value(value: &str) -> Result<SearchLevel> {
    match value {
        "chunk" => Ok(SearchLevel::Chunk),
        "file" => Ok(SearchLevel::File),
        "both" => Ok(SearchLevel::Both),
        _ => anyhow::bail!("unknown search level `{value}`"),
    }
}

fn normalize_fallback_config(
    fallback: EmbeddingFallbackConfig,
    config: &Config,
) -> EmbeddingFallbackConfig {
    let mut normalized = config.clone();
    normalized.embedding.fallback = None;
    normalized.embedding.provider = fallback.provider.clone();
    normalized.embedding.endpoint = fallback.endpoint.clone();
    normalized.embedding.api_key_env = fallback.api_key_env.clone();
    normalize_provider_defaults(&mut normalized);

    let mut fallback = EmbeddingFallbackConfig {
        provider: normalized.embedding.provider,
        endpoint: normalized.embedding.endpoint,
        api_key_env: normalized.embedding.api_key_env,
    };
    if fallback.provider == Provider::Native {
        fallback.endpoint = None;
        fallback.api_key_env = None;
    }

    fallback
}

impl From<ChunkingModeArg> for ChunkingMode {
    fn from(value: ChunkingModeArg) -> Self {
        match value {
            ChunkingModeArg::Smart => Self::Smart,
            ChunkingModeArg::LineWindow => Self::LineWindow,
            ChunkingModeArg::Off => Self::Off,
        }
    }
}

fn log_interactive_warning(err: &anyhow::Error) {
    eprintln!("warning: interactive input unavailable: {err:#}");
}

fn openai_dimensions(model: &str) -> usize {
    match model {
        "text-embedding-3-large" => 3072,
        _ => 1536,
    }
}

fn ensure_gitignore(path: &std::path::Path) -> Result<()> {
    let mut existing = fs::read_to_string(path).unwrap_or_default();
    let mut changed = false;
    for line in [".enf/", ".enf.local.toml"] {
        if !existing.lines().any(|candidate| candidate.trim() == line) {
            if !existing.ends_with('\n') && !existing.is_empty() {
                existing.push('\n');
            }
            existing.push_str(line);
            existing.push('\n');
            changed = true;
        }
    }
    if changed {
        fs::write(path, existing).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Native => "native",
            Provider::Ollama => "ollama",
            Provider::Openai => "openai",
            Provider::OpenaiCompatible => "openai-compatible",
            Provider::Http => "http",
        }
    }
}

impl From<ProviderArg> for Provider {
    fn from(value: ProviderArg) -> Self {
        match value {
            ProviderArg::Native => Provider::Native,
            ProviderArg::Ollama => Provider::Ollama,
            ProviderArg::Openai => Provider::Openai,
            ProviderArg::OpenaiCompatible => Provider::OpenaiCompatible,
            ProviderArg::Http => Provider::Http,
        }
    }
}

impl From<ModelVariantArg> for ModelVariant {
    fn from(value: ModelVariantArg) -> Self {
        match value {
            ModelVariantArg::Quantized => ModelVariant::Quantized,
            ModelVariantArg::Full => ModelVariant::Full,
        }
    }
}

impl ModelVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelVariant::Quantized => "quantized",
            ModelVariant::Full => "full",
        }
    }
}

impl From<ModelCacheArg> for ModelCache {
    fn from(value: ModelCacheArg) -> Self {
        match value {
            ModelCacheArg::Global => ModelCache::Global,
            ModelCacheArg::Project => ModelCache::Project,
        }
    }
}

impl From<SearchModeArg> for SearchMode {
    fn from(value: SearchModeArg) -> Self {
        match value {
            SearchModeArg::Hybrid => SearchMode::Hybrid,
            SearchModeArg::Vector => SearchMode::Vector,
            SearchModeArg::Keyword => SearchMode::Keyword,
        }
    }
}

impl From<SearchLevelArg> for SearchLevel {
    fn from(value: SearchLevelArg) -> Self {
        match value {
            SearchLevelArg::Chunk => SearchLevel::Chunk,
            SearchLevelArg::File => SearchLevel::File,
            SearchLevelArg::Both => SearchLevel::Both,
        }
    }
}
