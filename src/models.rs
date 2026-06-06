use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

use crate::{
    cli::{ModelsArgs, ModelsCommand},
    config::{Config, ModelCache},
    db,
    embed::{active_profile, EmbeddingProfile},
};

#[derive(Debug, Serialize)]
struct ModelStatus<'a> {
    profile_hash: &'a str,
    status: &'a str,
    provider: &'a str,
    engine: Option<&'a str>,
    model: &'a str,
    variant: Option<&'a str>,
    dimensions: usize,
    cache_path: String,
    marker_path: String,
    installed: bool,
}

#[derive(Debug, Serialize)]
struct InstalledModelMarker<'a> {
    status: &'a str,
    cache_path: String,
    profile: &'a EmbeddingProfile,
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
            if install.dry_run {
                return print_model_install_dry_run(&install_config, install.json);
            }
            install_active_model(&install_config)?;
            print_model_status(&install_config, install.json)
        }
        ModelsCommand::Path(json) | ModelsCommand::CachePath(json) => {
            let path = cache_path(&config)?;
            if json.json {
                println!("{}", serde_json::json!({ "cache_path": path }));
            } else {
                println!("{}", path.display());
            }
            Ok(())
        }
        ModelsCommand::Clean(json) | ModelsCommand::Gc(json) => {
            if json.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "removed": 0,
                        "dry_run": json.dry_run,
                    })
                );
            } else if json.dry_run {
                println!("Dry run: no cached models would be removed");
            } else {
                println!("No cached models removed");
            }
            Ok(())
        }
    }
}

fn print_model_install_dry_run(config: &Config, json: bool) -> Result<()> {
    let profile = active_profile(config);
    let model_cache_path = cache_path(config)?;
    let marker_path = active_model_marker_path(&profile, &model_cache_path);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "dry_run": true,
                "profile_hash": profile.profile_hash,
                "provider": profile.provider,
                "model": profile.model,
                "variant": profile.variant,
                "dimensions": profile.dimensions,
                "cache_path": model_cache_path,
                "marker_path": marker_path,
            }))?
        );
    } else {
        println!("Dry run: would install active embedding profile");
        println!("  profile: {}", profile.profile_hash);
        println!("  provider: {}", profile.provider);
        println!("  model: {}", profile.model);
        if let Some(variant) = profile.variant {
            println!("  variant: {variant}");
        }
        println!("  dims: {}", profile.dimensions);
        println!("  cache: {}", model_cache_path.display());
        println!("  marker: {}", marker_path.display());
    }
    Ok(())
}

pub fn install_active_model(config: &Config) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let global_cache_dir = dirs::cache_dir();
    install_active_model_in(config, &cwd, global_cache_dir.as_deref())
}

pub fn is_active_model_installed(config: &Config) -> Result<bool> {
    let cwd = std::env::current_dir()?;
    let global_cache_dir = dirs::cache_dir();
    is_active_model_installed_in(config, &cwd, global_cache_dir.as_deref())
}

pub fn cache_path(config: &Config) -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let global_cache_dir = dirs::cache_dir();
    Ok(cache_path_for(config, &cwd, global_cache_dir.as_deref()))
}

pub fn cache_path_for(config: &Config, cwd: &Path, global_cache_dir: Option<&Path>) -> PathBuf {
    match config.state.model_cache {
        ModelCache::Project => cwd.join(".enf/models"),
        ModelCache::Global => global_cache_dir
            .map(|root| root.join("enf/models"))
            .unwrap_or_else(|| cwd.join(".cache").join("enf/models")),
    }
}

pub fn install_active_model_in(
    config: &Config,
    cwd: &Path,
    global_cache_dir: Option<&Path>,
) -> Result<()> {
    let profile = active_profile(config);
    let path = cache_path_for(config, cwd, global_cache_dir);
    std::fs::create_dir_all(&path)?;

    if should_load_native_model(config) {
        let mut provider = crate::providers::build_provider(config)?;
        provider.ensure_ready()?;
    }

    let marker = InstalledModelMarker {
        status: "installed",
        cache_path: path.display().to_string(),
        profile: &profile,
    };
    let marker_path = active_model_marker_path(&profile, &path);
    std::fs::write(marker_path, serde_json::to_vec_pretty(&marker)?)?;

    let conn = db::open_or_create(&cwd.join(&config.state.db_path))?;
    upsert_model_cache(&conn, config, &path, "installed")?;
    Ok(())
}

fn should_load_native_model(config: &Config) -> bool {
    config.embedding.provider == crate::config::Provider::Native
        && cfg!(feature = "native-candle")
        && std::env::var("ENF_SKIP_NATIVE_MODEL_LOAD").as_deref() != Ok("1")
}

pub fn is_active_model_installed_in(
    config: &Config,
    cwd: &Path,
    global_cache_dir: Option<&Path>,
) -> Result<bool> {
    let profile = active_profile(config);
    let path = cache_path_for(config, cwd, global_cache_dir);
    let marker_path = active_model_marker_path(&profile, &path);
    if marker_path.exists() {
        return Ok(true);
    }
    let conn = db::open_or_create(&cwd.join(&config.state.db_path))?;
    Ok(model_cache_status(&conn, config, &path)?.as_deref() == Some("installed"))
}

fn print_model_status(config: &Config, json: bool) -> Result<()> {
    let profile = active_profile(config);
    let installed = is_active_model_installed(config)?;
    let model_cache_path = cache_path(config)?;
    let marker_path = active_model_marker_path(&profile, &model_cache_path);
    let status = if installed { "installed" } else { "missing" };
    let status = ModelStatus {
        profile_hash: &profile.profile_hash,
        status,
        provider: profile.provider.as_str(),
        engine: profile.engine.as_deref(),
        model: &profile.model,
        variant: profile.variant.as_deref(),
        dimensions: profile.dimensions,
        cache_path: model_cache_path.display().to_string(),
        marker_path: marker_path.display().to_string(),
        installed,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Active embedding profile:");
        println!("  profile:  {}", status.profile_hash);
        println!("  status:   {}", status.status);
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
        println!("  marker:   {}", status.marker_path);
        println!("  installed: {}", status.installed);
    }
    Ok(())
}

fn active_model_marker_path(profile: &EmbeddingProfile, cache_path: &Path) -> PathBuf {
    cache_path.join(format!("{}.json", profile.profile_hash))
}

fn model_cache_status(
    conn: &Connection,
    config: &Config,
    cache_path: &Path,
) -> Result<Option<String>> {
    let profile = active_profile(config);
    let engine = profile.engine.as_deref().unwrap_or("");
    let variant = profile.variant.as_deref().unwrap_or("");
    let cache_path = cache_path.display().to_string();
    let status = conn
        .query_row(
            r#"
            SELECT status
            FROM model_cache
            WHERE provider = ?1
              AND ifnull(engine, '') = ?2
              AND model = ?3
              AND ifnull(variant, '') = ?4
              AND dimensions = ?5
              AND cache_path = ?6
            LIMIT 1
            "#,
            params![
                profile.provider.as_str(),
                engine,
                profile.model.as_str(),
                variant,
                profile.dimensions as i64,
                cache_path,
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(status)
}

fn upsert_model_cache(
    conn: &Connection,
    config: &Config,
    cache_path: &Path,
    status: &str,
) -> Result<()> {
    let profile = active_profile(config);
    let engine = profile.engine.as_deref().unwrap_or("");
    let variant = profile.variant.as_deref().unwrap_or("");
    let cache_path = cache_path.display().to_string();
    conn.execute(
        r#"
        INSERT INTO model_cache (
            provider,
            engine,
            model,
            variant,
            dimensions,
            cache_path,
            installed_at,
            last_used_at,
            status
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'), datetime('now'), ?7)
        ON CONFLICT(provider, engine, model, variant, dimensions, cache_path)
        DO UPDATE SET
            installed_at = excluded.installed_at,
            last_used_at = excluded.last_used_at,
            status = excluded.status
        "#,
        params![
            profile.provider.as_str(),
            engine,
            profile.model.as_str(),
            variant,
            profile.dimensions as i64,
            cache_path,
            status,
        ],
    )?;
    Ok(())
}
