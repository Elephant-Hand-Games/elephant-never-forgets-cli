use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use crate::{
    cli::{ModelsArgs, ModelsCommand},
    config::{Config, ModelCache},
};

#[derive(Debug, Serialize)]
struct ModelStatus<'a> {
    provider: &'a str,
    engine: Option<&'a str>,
    model: &'a str,
    variant: Option<&'a str>,
    dimensions: usize,
    cache_path: String,
    installed: bool,
}

pub fn run(args: ModelsArgs) -> Result<()> {
    let config = crate::config::load().unwrap_or_default();
    match args.command {
        ModelsCommand::List(json) => print_model_status(&config, json.json),
        ModelsCommand::Current(json) => print_model_status(&config, json.json),
        ModelsCommand::Install(install) => {
            let mut install_config = config.clone();
            if let Some(model) = install.model {
                install_config.embedding.model = model;
            }
            if let Some(variant) = install.variant {
                install_config.embedding.variant = Some(variant.into());
            }
            install_active_model(&install_config)?;
            print_model_status(&install_config, install.json)
        }
        ModelsCommand::CachePath(json) => {
            let path = cache_path(&config)?;
            if json.json {
                println!("{}", serde_json::json!({ "cache_path": path }));
            } else {
                println!("{}", path.display());
            }
            Ok(())
        }
        ModelsCommand::Gc(json) => {
            if json.json {
                println!("{}", serde_json::json!({"removed": 0}));
            } else {
                println!("No cached models removed");
            }
            Ok(())
        }
    }
}

pub fn install_active_model(config: &Config) -> Result<()> {
    let path = cache_path(config)?;
    std::fs::create_dir_all(&path)?;
    let marker = path.join(model_marker_name(config));
    std::fs::write(marker, "installed\n")?;
    Ok(())
}

pub fn is_active_model_installed(config: &Config) -> Result<bool> {
    Ok(cache_path(config)?.join(model_marker_name(config)).exists())
}

pub fn cache_path(config: &Config) -> Result<PathBuf> {
    let path = match config.state.model_cache {
        ModelCache::Project => std::env::current_dir()?.join(".enf/models"),
        ModelCache::Global => dirs::cache_dir()
            .unwrap_or(std::env::current_dir()?.join(".cache"))
            .join("enf/models"),
    };
    Ok(path)
}

fn print_model_status(config: &Config, json: bool) -> Result<()> {
    let variant = config
        .embedding
        .variant
        .as_ref()
        .map(|variant| match variant {
            crate::config::ModelVariant::Quantized => "quantized",
            crate::config::ModelVariant::Full => "full",
        });
    let status = ModelStatus {
        provider: config.embedding.provider.as_str(),
        engine: config.embedding.engine.as_deref(),
        model: &config.embedding.model,
        variant,
        dimensions: config.embedding.dimensions,
        cache_path: cache_path(config)?.display().to_string(),
        installed: is_active_model_installed(config)?,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Active embedding profile:");
        println!("  provider: {}", status.provider);
        if let Some(engine) = status.engine {
            println!("  engine:   {}", engine);
        }
        println!("  model:    {}", status.model);
        if let Some(variant) = status.variant {
            println!("  variant:  {}", variant);
        }
        println!("  dims:     {}", status.dimensions);
        println!("  cache:    {}", status.cache_path);
        println!("  installed: {}", status.installed);
    }
    Ok(())
}

fn model_marker_name(config: &Config) -> String {
    let variant = config
        .embedding
        .variant
        .as_ref()
        .map(|variant| match variant {
            crate::config::ModelVariant::Quantized => "quantized",
            crate::config::ModelVariant::Full => "full",
        })
        .unwrap_or("none");
    format!(
        "{}-{}-{}.installed",
        config.embedding.provider.as_str(),
        config.embedding.model,
        variant
    )
}
