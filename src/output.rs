use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

use crate::{
    cli::{CiArgs, DoctorArgs, StatusArgs},
    config::{Config, Provider},
    embed::EmbeddingProfile,
};

#[derive(Debug, Serialize)]
struct StatusReport {
    config: String,
    db_path: String,
    active_profile: EmbeddingProfile,
    model_installed: bool,
    native_runtime_available: Option<bool>,
    counts: IndexCounts,
    indexed_profiles: Vec<IndexedProfile>,
}

#[derive(Debug, Serialize)]
struct IndexCounts {
    files: i64,
    chunks: i64,
    active_profile_embeddings: i64,
    missing_active_profile_embeddings: i64,
    query_embeddings: i64,
}

#[derive(Debug, Serialize)]
struct IndexedProfile {
    profile_hash: String,
    provider: String,
    engine: Option<String>,
    model: String,
    variant: Option<String>,
    dimensions: i64,
    embeddings: i64,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    ok: bool,
    config: Check,
    database: Check,
    fts5: Check,
    model: Check,
    provider: Check,
    counts: IndexCounts,
}

#[derive(Debug, Serialize)]
struct Check {
    ok: bool,
    message: String,
    remediation: Option<String>,
}

pub fn status(args: StatusArgs) -> Result<()> {
    let config = crate::config::load()?;
    crate::config::validate(&config)?;
    let cwd = std::env::current_dir()?;
    let conn = crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let report = status_report(&config, &conn)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Elephant Never Forgets status");
        println!("  config: {}", report.config);
        println!("  database: {}", report.db_path);
        println!("  provider: {}", report.active_profile.provider);
        println!("  model: {}", report.active_profile.model);
        println!("  profile: {}", report.active_profile.profile_hash);
        println!("  model installed: {}", report.model_installed);
        if let Some(native_runtime_available) = report.native_runtime_available {
            println!("  native runtime available: {}", native_runtime_available);
        }
        println!("  files: {}", report.counts.files);
        println!("  chunks: {}", report.counts.chunks);
        println!(
            "  active profile embeddings: {}",
            report.counts.active_profile_embeddings
        );
        println!(
            "  missing active profile embeddings: {}",
            report.counts.missing_active_profile_embeddings
        );
        println!("  indexed profiles: {}", report.indexed_profiles.len());
    }
    Ok(())
}

pub fn doctor(args: DoctorArgs) -> Result<()> {
    let config = crate::config::load()?;
    crate::config::validate(&config)?;
    let cwd = std::env::current_dir()?;
    let conn = crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let report = doctor_report(&config, &conn)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("ENF doctor");
        print_check("config", &report.config);
        print_check("database", &report.database);
        print_check("fts5", &report.fts5);
        print_check("model", &report.model);
        print_check("provider", &report.provider);
        println!(
            "  missing active profile embeddings: {}",
            report.counts.missing_active_profile_embeddings
        );
    }
    Ok(())
}

pub fn ci(args: CiArgs) -> Result<()> {
    let config = crate::config::load()?;
    crate::config::validate(&config)?;
    let cwd = std::env::current_dir()?;
    if args.install_models {
        crate::models::install_active_model_in(&config, &cwd, dirs::cache_dir().as_deref())?;
    }
    let conn = crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let report = doctor_report(&config, &conn)?;

    let mut ok = report.config.ok && report.database.ok && report.fts5.ok;
    if !args.no_embed {
        ok = ok
            && report.model.ok
            && report.provider.ok
            && report.counts.missing_active_profile_embeddings == 0;
    }

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": ok,
                "no_embed": args.no_embed,
                "install_models": args.install_models,
                "doctor": report,
            }))?
        );
    } else if ok {
        println!("ENF CI checks passed");
    }

    if !ok {
        anyhow::bail!(
            "ENF CI checks failed. Run `enf doctor` for diagnostics or `enf ci --no-embed` for config/database-only checks."
        );
    }
    Ok(())
}

fn status_report(config: &Config, conn: &Connection) -> Result<StatusReport> {
    let active_profile = crate::embed::active_profile(config);
    Ok(StatusReport {
        config: ".enf.toml".into(),
        db_path: config.state.db_path.clone(),
        active_profile,
        model_installed: crate::models::is_active_model_installed(config)?,
        native_runtime_available: (config.embedding.provider == Provider::Native)
            .then(native_fastembed_available),
        counts: index_counts(config, conn)?,
        indexed_profiles: indexed_profiles(conn)?,
    })
}

fn doctor_report(config: &Config, conn: &Connection) -> Result<DoctorReport> {
    let counts = index_counts(config, conn)?;
    let model_installed = crate::models::is_active_model_installed(config)?;
    let model = if model_installed {
        Check::ok("active model marker is installed")
    } else {
        Check::warn("active model marker is missing", "enf models install")
    };
    let provider = provider_check(config, model_installed);
    let fts5 = match scalar_count(conn, "SELECT COUNT(*) FROM chunks_fts") {
        Ok(_) => Check::ok("FTS5 table is readable"),
        Err(err) => Check::warn(
            format!("FTS5 table is not readable: {err}"),
            "rm .enf/index.sqlite && enf init --db && enf index .",
        ),
    };
    let database = Check::ok("SQLite database is readable");
    let config_check = Check::ok("configuration is valid");
    let ok = config_check.ok && database.ok && fts5.ok && model.ok && provider.ok;
    Ok(DoctorReport {
        ok,
        config: config_check,
        database,
        fts5,
        model,
        provider,
        counts,
    })
}

fn provider_check(config: &Config, model_installed: bool) -> Check {
    match config.embedding.provider {
        Provider::Native => {
            if !native_fastembed_available() {
                return Check::warn(
                    "native fastembed runtime is not included in this portable build",
                    "enf init --provider ollama --model nomic-embed-text --force",
                );
            }
            if model_installed {
                Check::ok("native provider is ready from installed model marker")
            } else {
                Check::warn("native model is not installed", "enf models install")
            }
        }
        _ => match crate::providers::build_provider(config)
            .and_then(|mut provider| provider.ensure_ready())
        {
            Ok(()) => Check::ok("provider configuration is reachable"),
            Err(err) => Check::warn(format!("provider is not ready: {err}"), "enf doctor"),
        },
    }
}

fn native_fastembed_available() -> bool {
    cfg!(feature = "native-fastembed")
}

fn index_counts(config: &Config, conn: &Connection) -> Result<IndexCounts> {
    let profile_id = crate::db::get_active_embedding_profile_id(conn, config)?;
    let chunks = scalar_count(conn, "SELECT COUNT(*) FROM chunks")?;
    let active_profile_embeddings = if let Some(profile_id) = profile_id {
        conn.query_row(
            "SELECT COUNT(*)
             FROM embeddings e
             JOIN chunks c ON c.id = e.chunk_id
             WHERE e.profile_id = ?1",
            [profile_id],
            |row| row.get(0),
        )?
    } else {
        0
    };
    let missing_active_profile_embeddings = chunks - active_profile_embeddings;
    Ok(IndexCounts {
        files: scalar_count(conn, "SELECT COUNT(*) FROM files")?,
        chunks,
        active_profile_embeddings,
        missing_active_profile_embeddings,
        query_embeddings: scalar_count(conn, "SELECT COUNT(*) FROM query_embeddings")?,
    })
}

fn indexed_profiles(conn: &Connection) -> Result<Vec<IndexedProfile>> {
    let mut stmt = conn.prepare(
        "SELECT ep.profile_hash, ep.provider, ep.engine, ep.model, ep.variant, ep.dimensions,
                COUNT(e.id) AS embeddings
         FROM embedding_profiles ep
         LEFT JOIN embeddings e ON e.profile_id = ep.id
         GROUP BY ep.id
         ORDER BY ep.created_at, ep.profile_hash",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(IndexedProfile {
            profile_hash: row.get(0)?,
            provider: row.get(1)?,
            engine: row.get(2)?,
            model: row.get(3)?,
            variant: row.get(4)?,
            dimensions: row.get(5)?,
            embeddings: row.get(6)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn scalar_count(conn: &Connection, sql: &str) -> Result<i64> {
    Ok(conn.query_row(sql, [], |row| row.get(0))?)
}

fn print_check(label: &str, check: &Check) {
    println!(
        "  {label}: {}",
        if check.ok { "ok" } else { "needs attention" }
    );
    println!("    {}", check.message);
    if let Some(remediation) = &check.remediation {
        println!("    run: {remediation}");
    }
}

impl Check {
    fn ok(message: impl Into<String>) -> Self {
        Self {
            ok: true,
            message: message.into(),
            remediation: None,
        }
    }

    fn warn(message: impl Into<String>, remediation: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: message.into(),
            remediation: Some(remediation.into()),
        }
    }
}
