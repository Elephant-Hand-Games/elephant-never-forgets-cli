use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::{
    cli::{
        ConfigArgs, ConfigCommand, ConfigDiffArgs, ConfigExplainArgs, ConfigSetArgs,
        ConfigShowArgs, JsonArgs, PresetArg, SetupArgs, SetupCommand, SetupFallbackArgs,
        SetupImagesArgs, SetupRerankerArgs, SetupSearchArgs, SetupUseArgs,
    },
    config::{
        self, Config, EmbeddingFallbackConfig, ModelVariant, Provider, SearchLevel, SearchMode,
        CONFIG_FILE,
    },
};

#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    name: &'static str,
    purpose: &'static str,
    provider: &'static str,
    external_service: bool,
    best_for: &'static str,
}

pub fn run_setup(args: SetupArgs) -> Result<()> {
    match args.command {
        None => print_setup_overview(),
        Some(SetupCommand::Presets(json)) => list_presets(json),
        Some(SetupCommand::Preset(args)) => show_preset(args.preset, args.json),
        Some(SetupCommand::Use(args)) => use_preset(args),
        Some(SetupCommand::Reranker(args)) => setup_reranker(args),
        Some(SetupCommand::Images(args)) => setup_images(args),
        Some(SetupCommand::Fallback(args)) => setup_fallback(args),
        Some(SetupCommand::Search(args)) => setup_search(args),
    }
}

pub fn run_config(args: ConfigArgs) -> Result<()> {
    match args.command {
        None => config_show(ConfigShowArgs {
            section: None,
            json: false,
        }),
        Some(ConfigCommand::Path(args)) => config_path(args),
        Some(ConfigCommand::Show(args)) => config_show(args),
        Some(ConfigCommand::Explain(args)) => config_explain(args),
        Some(ConfigCommand::Validate(args)) => config_validate(args),
        Some(ConfigCommand::Edit) => config_edit(),
        Some(ConfigCommand::Set(args)) => config_set(args),
        Some(ConfigCommand::Diff(args)) => config_diff(args),
        Some(ConfigCommand::Doctor(args)) => crate::output::doctor(crate::cli::DoctorArgs {
            ci: false,
            fix: false,
            yes: false,
            check: None,
            no_embeddings: false,
            install_models: false,
            dry_run: false,
            json: args.json,
        }),
    }
}

pub fn preset_config(preset: PresetArg) -> Config {
    let mut config = Config::default();
    apply_preset(&mut config, preset);
    config
}

pub fn apply_preset(config: &mut Config, preset: PresetArg) {
    match preset {
        PresetArg::Local | PresetArg::Docs => {
            config.embedding.provider = Provider::Native;
            config.embedding.engine = Some("candle".into());
            config.embedding.model = "nomic-embed-text-v1.5".into();
            config.embedding.variant = Some(ModelVariant::Quantized);
            config.embedding.endpoint = None;
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 768;
            config.index.chunking = crate::config::ChunkingMode::Smart;
        }
        PresetArg::Code => {
            config.embedding.provider = Provider::Native;
            config.embedding.engine = Some("candle".into());
            config.embedding.model = "google/embeddinggemma-300m".into();
            config.embedding.variant = Some(ModelVariant::Quantized);
            config.embedding.endpoint = None;
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 768;
            config.embedding.document_prefix = "title: none | text: ".into();
            config.embedding.query_prefix = "task: search result | query: ".into();
            config.index.chunking = crate::config::ChunkingMode::Smart;
        }
        PresetArg::Ollama => {
            config.embedding.provider = Provider::Ollama;
            config.embedding.engine = None;
            config.embedding.model = "nomic-embed-text-v1.5".into();
            config.embedding.variant = None;
            config.embedding.endpoint = Some("http://localhost:11434/api/embed".into());
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 768;
        }
        PresetArg::Openai => {
            config.embedding.provider = Provider::Openai;
            config.embedding.engine = None;
            config.embedding.model = "text-embedding-3-small".into();
            config.embedding.variant = None;
            config.embedding.endpoint = Some("https://api.openai.com/v1/embeddings".into());
            config.embedding.api_key_env = Some("OPENAI_API_KEY".into());
            config.embedding.dimensions = 1536;
            config.embedding.document_prefix.clear();
            config.embedding.query_prefix.clear();
        }
        PresetArg::Custom => {
            config.embedding.provider = Provider::OpenaiCompatible;
            config.embedding.engine = None;
            config.embedding.model = "custom-embedding-model".into();
            config.embedding.variant = None;
            config.embedding.endpoint = Some("http://localhost:8080/v1/embeddings".into());
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 768;
        }
        PresetArg::Keyword => {
            config.search.default_mode = SearchMode::Keyword;
            config.embedding.provider = Provider::Http;
            config.embedding.engine = None;
            config.embedding.model = "keyword-only".into();
            config.embedding.variant = None;
            config.embedding.endpoint = Some("http://localhost/unused".into());
            config.embedding.api_key_env = None;
            config.embedding.dimensions = 1;
        }
    }
}

fn print_setup_overview() -> Result<()> {
    println!("ENF setup");
    println!();
    println!("Change providers, presets, reranker, images, or search defaults.");
    println!();
    println!("Common commands:");
    println!("  enf setup presets");
    println!("  enf setup preset local");
    println!("  enf setup use local");
    println!("  enf setup reranker --endpoint http://localhost:8080/rerank");
    println!("  enf setup images --endpoint http://localhost:8081/embed-images --query-endpoint http://localhost:8081/embed-query");
    Ok(())
}

fn list_presets(args: JsonArgs) -> Result<()> {
    let presets = presets();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&presets)?);
        return Ok(());
    }
    println!("Available presets\n");
    for preset in presets {
        println!("  {:<10} {}", preset.name, preset.purpose);
    }
    println!("\nShow details:\n  enf setup preset local");
    println!("\nSwitch:\n  enf setup use local");
    Ok(())
}

fn show_preset(preset: PresetArg, json: bool) -> Result<()> {
    let config = preset_config(preset.clone());
    if json {
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }
    let info = preset_info(preset);
    println!("Preset: {}\n", info.name);
    println!("What it does\n  {}", info.purpose);
    println!("\nWrites");
    println!(
        "  embedding.provider = \"{}\"",
        config.embedding.provider.as_str()
    );
    println!("  embedding.model = \"{}\"", config.embedding.model);
    if let Some(variant) = &config.embedding.variant {
        println!("  embedding.variant = \"{}\"", variant.as_str());
    }
    println!("  embedding.dimensions = {}", config.embedding.dimensions);
    println!(
        "\nRequires external service\n  {}",
        if info.external_service { "yes" } else { "no" }
    );
    println!("\nGood for\n  {}", info.best_for);
    println!("\nUse it\n  enf setup use {}", info.name);
    Ok(())
}

fn use_preset(args: SetupUseArgs) -> Result<()> {
    let mut config = config::load()?;
    apply_preset(&mut config, args.preset.clone());
    if let Some(endpoint) = args.endpoint {
        config.embedding.endpoint = Some(endpoint);
    }
    if let Some(api_key_env) = args.api_key_env {
        config.embedding.api_key_env = Some(api_key_env);
    }
    if let Some(model) = args.model {
        config.embedding.model = model;
    }
    if let Some(dimensions) = args.dimensions {
        config.embedding.dimensions = dimensions;
    }
    write_or_preview(config, args.dry_run, args.json, "Updated embedding preset")
}

fn setup_reranker(args: SetupRerankerArgs) -> Result<()> {
    if args.schema {
        println!("Reranker schema");
        println!("  Request:  POST JSON with query, texts, raw_scores, return_text, truncate, truncation_direction");
        println!("  Response: [{{\"index\":0,\"score\":0.92}}] or {{\"results\":[{{\"index\":0,\"relevance_score\":0.92}}]}}");
        return Ok(());
    }
    let mut config = config::load()?;
    if args.off {
        config.reranker.enabled = false;
        config.reranker.endpoint = None;
    } else {
        let Some(endpoint) = args.endpoint else {
            println!("Reranker\n");
            println!("ENF does not ship a reranker server. Provide an endpoint to enable it:");
            println!("  enf setup reranker --endpoint http://localhost:8080/rerank");
            return Ok(());
        };
        config.reranker.enabled = true;
        config.reranker.endpoint = Some(endpoint);
    }
    if let Some(model) = args.model {
        config.reranker.model = model;
    }
    if let Some(candidate_limit) = args.candidate_limit {
        config.reranker.candidate_limit = candidate_limit;
    }
    write_or_preview(config, args.dry_run, args.json, "Updated reranker settings")
}

fn setup_images(args: SetupImagesArgs) -> Result<()> {
    let mut config = config::load()?;
    if args.off {
        config.image.embedding.enabled = false;
        config.image.embedding.endpoint = None;
        config.image.embedding.query_endpoint = None;
    } else {
        let Some(endpoint) = args.endpoint else {
            println!("Image search\n");
            println!("Filename image search works now. Visual search requires image embedding endpoints.");
            println!("  enf setup images --endpoint http://localhost:8081/embed-images --query-endpoint http://localhost:8081/embed-query");
            return Ok(());
        };
        config.image.embedding.enabled = true;
        config.image.embedding.endpoint = Some(endpoint);
        config.image.embedding.query_endpoint = args.query_endpoint;
    }
    if let Some(model) = args.model {
        config.image.embedding.model = model;
    }
    if let Some(dimensions) = args.dimensions {
        config.image.embedding.dimensions = dimensions;
    }
    write_or_preview(
        config,
        args.dry_run,
        args.json,
        "Updated image embedding settings",
    )
}

fn setup_fallback(args: SetupFallbackArgs) -> Result<()> {
    let mut config = config::load()?;
    if args.off {
        config.embedding.fallback = None;
    } else {
        let Some(provider) = args.provider else {
            println!("Fallback provider\n");
            println!("Configure a backup embedding provider with:");
            println!("  enf setup fallback --provider ollama --endpoint http://localhost:11434/api/embed");
            return Ok(());
        };
        config.embedding.fallback = Some(EmbeddingFallbackConfig {
            provider: provider.into(),
            endpoint: args.endpoint,
            api_key_env: args.api_key_env,
        });
    }
    write_or_preview(config, args.dry_run, args.json, "Updated fallback provider")
}

fn setup_search(args: SetupSearchArgs) -> Result<()> {
    let mut config = config::load()?;
    if let Some(mode) = args.mode {
        config.search.default_mode = mode.into();
    }
    if let Some(level) = args.level {
        config.search.default_level = level.into();
    }
    if let Some(limit) = args.limit {
        config.search.limit = limit;
    }
    write_or_preview(config, args.dry_run, args.json, "Updated search defaults")
}

fn config_path(args: JsonArgs) -> Result<()> {
    let path = std::env::current_dir()?.join(CONFIG_FILE);
    if args.json {
        println!("{}", serde_json::json!({ "path": path }));
    } else {
        println!("{}", path.display());
    }
    Ok(())
}

fn config_show(args: ConfigShowArgs) -> Result<()> {
    let config = config::load()?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }
    let text = toml::to_string_pretty(&config)?;
    if let Some(section) = args.section {
        print_section(&text, &section);
    } else {
        print!("{text}");
    }
    Ok(())
}

fn config_explain(args: ConfigExplainArgs) -> Result<()> {
    let key = args.key.unwrap_or_else(|| "precedence".into());
    match key.as_str() {
        "precedence" => {
            println!("Configuration precedence\n");
            println!("1. Command-line flags");
            println!("2. Environment variables");
            println!("3. .enf.local.toml");
            println!("4. .enf.toml");
            println!("5. ENF defaults");
        }
        "reranker.endpoint" => {
            println!("reranker.endpoint\n");
            println!("What it does\n  HTTP endpoint ENF calls to rerank search candidates.");
            println!("\nRequired?\n  Only when reranker.enabled = true.");
            println!("\nConfigure\n  enf setup reranker --endpoint http://localhost:8080/rerank");
        }
        "image.embedding.endpoint" => {
            println!("image.embedding.endpoint\n");
            println!("What it does\n  HTTP endpoint ENF calls to embed image files.");
            println!("\nConfigure\n  enf setup images --endpoint http://localhost:8081/embed-images --query-endpoint http://localhost:8081/embed-query");
        }
        "embedding.provider" => {
            println!("embedding.provider\n");
            println!("Options\n  native, ollama, openai, openai-compatible, http");
            println!("\nRecommended\n  enf setup presets\n  enf setup use local");
        }
        _ => {
            println!("{key}\n");
            println!("No built-in explanation exists for this key yet.");
            println!("Edit the config directly:\n  enf config edit");
        }
    }
    Ok(())
}

fn config_validate(args: JsonArgs) -> Result<()> {
    let config = config::load()?;
    config::validate(&config)?;
    if args.json {
        println!("{}", serde_json::json!({ "ok": true }));
    } else {
        println!("✓ .enf.toml is valid");
    }
    Ok(())
}

fn config_edit() -> Result<()> {
    let path = std::env::current_dir()?.join(CONFIG_FILE);
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".into());
    let status = Command::new(editor)
        .arg(&path)
        .status()
        .with_context(|| format!("opening {}", path.display()))?;
    if !status.success() {
        bail!("editor exited unsuccessfully");
    }
    Ok(())
}

fn config_set(args: ConfigSetArgs) -> Result<()> {
    let mut config = config::load()?;
    let handled = match args.key.as_str() {
        "embedding.provider" => {
            config.embedding.provider = parse_provider(&args.value)?;
            true
        }
        "embedding.model" => {
            config.embedding.model = args.value.clone();
            true
        }
        "embedding.endpoint" => {
            config.embedding.endpoint = optional_string(&args.value);
            true
        }
        "embedding.api_key_env" => {
            config.embedding.api_key_env = optional_string(&args.value);
            true
        }
        "embedding.dimensions" => {
            config.embedding.dimensions = args.value.parse()?;
            true
        }
        "search.default_mode" => {
            config.search.default_mode = parse_search_mode(&args.value)?;
            true
        }
        "search.default_level" => {
            config.search.default_level = parse_search_level(&args.value)?;
            true
        }
        "search.limit" => {
            config.search.limit = args.value.parse()?;
            true
        }
        "reranker.enabled" => {
            config.reranker.enabled = parse_bool(&args.value)?;
            true
        }
        "reranker.endpoint" => {
            config.reranker.endpoint = optional_string(&args.value);
            true
        }
        "image.embedding.enabled" => {
            config.image.embedding.enabled = parse_bool(&args.value)?;
            true
        }
        "image.embedding.endpoint" => {
            config.image.embedding.endpoint = optional_string(&args.value);
            true
        }
        "image.embedding.query_endpoint" => {
            config.image.embedding.query_endpoint = optional_string(&args.value);
            true
        }
        _ => false,
    };
    if !handled {
        let path = std::env::current_dir()?.join(CONFIG_FILE);
        println!("ENF does not safely edit `{}` yet.", args.key);
        println!("Edit the TOML directly:\n  {}", path.display());
        println!("Or run:\n  enf config edit");
        return Ok(());
    }
    write_or_preview(config, args.dry_run, args.json, "Updated config")
}

fn config_diff(args: ConfigDiffArgs) -> Result<()> {
    let current = config::load()?;
    let preset = preset_config(args.preset);
    println!("Preset diff");
    println!(
        "  current provider: {}",
        current.embedding.provider.as_str()
    );
    println!("  preset provider:  {}", preset.embedding.provider.as_str());
    println!("  current model:    {}", current.embedding.model);
    println!("  preset model:     {}", preset.embedding.model);
    println!("  current dims:     {}", current.embedding.dimensions);
    println!("  preset dims:      {}", preset.embedding.dimensions);
    Ok(())
}

fn write_or_preview(config: Config, dry_run: bool, json: bool, message: &str) -> Result<()> {
    config::validate(&config)?;
    let path = std::env::current_dir()?.join(CONFIG_FILE);
    if dry_run {
        if json {
            println!("{}", serde_json::to_string_pretty(&config)?);
        } else {
            println!("Dry run: would write {}", path.display());
            println!("{}", toml::to_string_pretty(&config)?);
        }
        return Ok(());
    }
    config::write_config(&path, &config)?;
    if json {
        println!("{}", serde_json::json!({ "ok": true, "path": path }));
    } else {
        println!("✓ {message}");
        println!("✓ Wrote {}", path.display());
    }
    Ok(())
}

fn presets() -> Vec<Preset> {
    [
        PresetArg::Local,
        PresetArg::Code,
        PresetArg::Docs,
        PresetArg::Ollama,
        PresetArg::Openai,
        PresetArg::Custom,
        PresetArg::Keyword,
    ]
    .into_iter()
    .map(preset_info)
    .collect()
}

fn preset_info(preset: PresetArg) -> Preset {
    match preset {
        PresetArg::Local => Preset {
            name: "local",
            purpose: "Recommended starter. Local packaged embeddings, no API key.",
            provider: "native",
            external_service: false,
            best_for: "first run, private projects, offline workflows",
        },
        PresetArg::Code => Preset {
            name: "code",
            purpose: "Local packaged model tuned for code-heavy repos.",
            provider: "native",
            external_service: false,
            best_for: "source repos, game scripts, agent files",
        },
        PresetArg::Docs => Preset {
            name: "docs",
            purpose: "Local packaged model tuned for long documents and prose.",
            provider: "native",
            external_service: false,
            best_for: "markdown, notes, design docs",
        },
        PresetArg::Ollama => Preset {
            name: "ollama",
            purpose: "Use an Ollama embedding endpoint.",
            provider: "ollama",
            external_service: true,
            best_for: "users already running Ollama",
        },
        PresetArg::Openai => Preset {
            name: "openai",
            purpose: "Use OpenAI embeddings via OPENAI_API_KEY.",
            provider: "openai",
            external_service: true,
            best_for: "team and CI reliability",
        },
        PresetArg::Custom => Preset {
            name: "custom",
            purpose: "Use your own OpenAI-compatible or HTTP embedding endpoint.",
            provider: "openai-compatible",
            external_service: true,
            best_for: "TEI, vLLM, or custom embedding servers",
        },
        PresetArg::Keyword => Preset {
            name: "keyword",
            purpose: "Keyword search only; no embedding model required for search.",
            provider: "none",
            external_service: false,
            best_for: "zero setup, fallback, CI checks",
        },
    }
}

fn print_section(text: &str, section: &str) {
    let header = format!("[{section}]");
    let mut printing = section == "root";
    for line in text.lines() {
        if line.starts_with('[') {
            printing = line == header || line.starts_with(&format!("[{section}."));
        }
        if printing {
            println!("{line}");
        }
    }
}

fn parse_provider(value: &str) -> Result<Provider> {
    match value {
        "native" => Ok(Provider::Native),
        "ollama" => Ok(Provider::Ollama),
        "openai" => Ok(Provider::Openai),
        "openai-compatible" => Ok(Provider::OpenaiCompatible),
        "http" => Ok(Provider::Http),
        _ => bail!("unknown provider `{value}`"),
    }
}

fn parse_search_mode(value: &str) -> Result<SearchMode> {
    match value {
        "hybrid" => Ok(SearchMode::Hybrid),
        "vector" => Ok(SearchMode::Vector),
        "keyword" => Ok(SearchMode::Keyword),
        _ => bail!("unknown search mode `{value}`"),
    }
}

fn parse_search_level(value: &str) -> Result<SearchLevel> {
    match value {
        "chunk" => Ok(SearchLevel::Chunk),
        "file" => Ok(SearchLevel::File),
        "both" => Ok(SearchLevel::Both),
        _ => bail!("unknown search level `{value}`"),
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        _ => bail!("expected true or false"),
    }
}

fn optional_string(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.to_string())
}
