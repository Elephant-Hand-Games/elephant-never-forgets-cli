use anyhow::Result;

use crate::cli::{DoctorArgs, StatusArgs};

pub fn status(args: StatusArgs) -> Result<()> {
    let config = crate::config::load()?;
    let profile = crate::embed::active_profile(&config);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "config": ".enf.toml",
                "db_path": config.state.db_path,
                "active_profile": profile,
            }))?
        );
    } else {
        println!("Elephant Never Forgets status");
        println!("  config: .enf.toml");
        println!("  database: {}", config.state.db_path);
        println!("  provider: {}", profile.provider);
        println!("  model: {}", profile.model);
        println!("  profile: {}", profile.profile_hash);
    }
    Ok(())
}

pub fn doctor(args: DoctorArgs) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    crate::db::open_or_create(&cwd.join(&config.state.db_path))?;
    let model_installed = crate::models::is_active_model_installed(&config)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "db": true,
                "model_installed": model_installed,
            }))?
        );
    } else {
        println!("ENF doctor");
        println!("  config: ok");
        println!("  database: ok");
        println!("  active model installed: {}", model_installed);
    }
    Ok(())
}
