use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    cli::{InitArgs, ModelCacheArg, ModelVariantArg, ProviderArg},
    db,
    errors::EnfError,
};

pub const CONFIG_FILE: &str = ".enf.toml";
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
    pub include: PatternConfig,
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
    pub batch_size: usize,
    pub query_cache: bool,
    pub document_prefix: String,
    pub query_prefix: String,
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
    pub chunk_target_tokens: usize,
    pub chunk_max_tokens: usize,
    pub chunk_overlap_tokens: usize,
    pub min_chunk_chars: usize,
    pub hash_algorithm: String,
    pub store_full_files: bool,
    pub store_chunks: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatternConfig {
    pub patterns: Vec<String>,
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
                engine: Some("fastembed".into()),
                model: "nomic-embed-text-v1.5".into(),
                variant: Some(ModelVariant::Quantized),
                endpoint: None,
                api_key_env: None,
                dimensions: 768,
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
                chunk_target_tokens: 450,
                chunk_max_tokens: 900,
                chunk_overlap_tokens: 80,
                min_chunk_chars: 80,
                hash_algorithm: "blake3".into(),
                store_full_files: true,
                store_chunks: true,
            },
            include: PatternConfig {
                patterns: vec![
                    "AGENTS.md",
                    "README*",
                    "CHANGELOG*",
                    "CONTRIBUTING*",
                    "docs/**",
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
                .collect(),
            },
            exclude: PatternConfig {
                patterns: vec![
                    ".git/**",
                    ".enf/**",
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
                .collect(),
            },
        }
    }
}

pub fn init(args: InitArgs) -> Result<()> {
    if !args.db {
        anyhow::bail!("init currently requires --db");
    }

    let cwd = std::env::current_dir().context("reading current directory")?;
    let config_path = cwd.join(CONFIG_FILE);
    if config_path.exists() && !args.force {
        return Err(EnfError::ConfigExists {
            path: config_path.display().to_string(),
        }
        .into());
    }

    let mut config = Config::default();
    apply_init_overrides(&mut config, &args);

    fs::create_dir_all(cwd.join(STATE_DIR)).context("creating .enf directory")?;
    fs::create_dir_all(cwd.join(CACHE_DIR)).context("creating .enf/cache directory")?;
    write_config(&config_path, &config)?;
    ensure_gitignore(&cwd.join(".gitignore"))?;
    db::open_or_create(&cwd.join(&config.state.db_path))?;

    if args.install_models {
        crate::models::install_active_model(&config)?;
    }

    if args.index {
        crate::index::index_path(&cwd, PathBuf::from("."), &config, args.install_models)?;
    }

    println!("Initialized Elephant Never Forgets project");
    println!("  config: {}", CONFIG_FILE);
    println!("  database: {}", config.state.db_path);
    println!("  provider: {}", config.embedding.provider.as_str());
    Ok(())
}

pub fn load() -> Result<Config> {
    let path = std::env::current_dir()?.join(CONFIG_FILE);
    if !path.exists() {
        return Err(EnfError::NotInitialized.into());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let config: Config =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    validate(&config)?;
    Ok(config)
}

pub fn write_config(path: &std::path::Path, config: &Config) -> Result<()> {
    let text = toml::to_string_pretty(config).context("serializing config")?;
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

pub fn validate(config: &Config) -> Result<()> {
    if config.version != 1 {
        anyhow::bail!("unsupported config version {}", config.version);
    }
    if config.embedding.dimensions == 0 {
        anyhow::bail!("embedding.dimensions must be greater than 0");
    }
    let total_weight =
        config.search.vector_weight + config.search.keyword_weight + config.search.metadata_weight;
    if total_weight <= 0.0 {
        anyhow::bail!("search weights must sum to a positive value");
    }
    if config.index.chunk_overlap_tokens >= config.index.chunk_max_tokens {
        anyhow::bail!("index.chunk_overlap_tokens must be less than index.chunk_max_tokens");
    }
    Ok(())
}

fn apply_init_overrides(config: &mut Config, args: &InitArgs) {
    if let Some(provider) = &args.provider {
        config.embedding.provider = provider.clone().into();
    }
    if args.native_embed || args.local_embed {
        config.embedding.provider = Provider::Native;
        config.embedding.engine = Some("fastembed".into());
    }
    if let Some(model) = &args.model {
        config.embedding.model = model.clone();
    }
    if let Some(variant) = &args.variant {
        config.embedding.variant = Some(variant.clone().into());
    }
    if let Some(model_cache) = &args.model_cache {
        config.state.model_cache = model_cache.clone().into();
    }
    normalize_provider_defaults(config);
}

pub fn normalize_provider_defaults(config: &mut Config) {
    match config.embedding.provider {
        Provider::Native => {
            config.embedding.engine = Some("fastembed".into());
            config.embedding.endpoint = None;
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 768;
            if config.embedding.variant.is_none() {
                config.embedding.variant = Some(ModelVariant::Quantized);
            }
            config.embedding.document_prefix = "search_document: ".into();
            config.embedding.query_prefix = "search_query: ".into();
        }
        Provider::Ollama => {
            config.embedding.engine = None;
            config.embedding.endpoint = Some("http://localhost:11434/api/embed".into());
            config.embedding.dimensions = 768;
            config.embedding.variant = None;
            config.embedding.document_prefix = "search_document: ".into();
            config.embedding.query_prefix = "search_query: ".into();
            if config.embedding.model == "nomic-embed-text-v1.5" {
                config.embedding.model = "nomic-embed-text".into();
            }
        }
        Provider::Openai => {
            config.embedding.engine = None;
            config.embedding.endpoint = Some("https://api.openai.com/v1/embeddings".into());
            config.embedding.api_key_env = Some("OPENAI_API_KEY".into());
            config.embedding.variant = None;
            config.embedding.document_prefix.clear();
            config.embedding.query_prefix.clear();
            if config.embedding.model == "nomic-embed-text-v1.5" {
                config.embedding.model = "text-embedding-3-small".into();
            }
            config.embedding.dimensions = match config.embedding.model.as_str() {
                "text-embedding-3-large" => 3072,
                _ => 1536,
            };
        }
        Provider::OpenaiCompatible | Provider::Http => {}
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

impl From<ModelCacheArg> for ModelCache {
    fn from(value: ModelCacheArg) -> Self {
        match value {
            ModelCacheArg::Global => ModelCache::Global,
            ModelCacheArg::Project => ModelCache::Project,
        }
    }
}
