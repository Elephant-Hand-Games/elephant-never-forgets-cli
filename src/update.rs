use std::process::Command;

use anyhow::{Context, Result};

use crate::cli::UpdateArgs;

const DEFAULT_REPO: &str = "Elephant-Hand-Games/elephant-never-forgets-cli";
const INSTALL_SCRIPT_PATH: &str = "scripts/install.sh";

pub fn run(args: UpdateArgs) -> Result<()> {
    let repo = args.repo.as_deref().unwrap_or(DEFAULT_REPO);
    let script_url = install_script_url(repo);
    let command = shell_command(&script_url);
    let envs = update_envs(&args);

    if args.dry_run {
        println!("command: {command}");
        for (key, value) in envs {
            println!("env: {key}={value}");
        }
        return Ok(());
    }

    println!("==> Updating Elephant Never Forgets");
    println!("==> Fetching installer from {script_url}");
    let mut child = Command::new("sh");
    child.arg("-c").arg(&command);
    for (key, value) in envs {
        child.env(key, value);
    }
    let status = child.status().context("running enf update installer")?;
    if !status.success() {
        anyhow::bail!("enf update failed with status {status}");
    }
    Ok(())
}

fn install_script_url(repo: &str) -> String {
    format!("https://raw.githubusercontent.com/{repo}/main/{INSTALL_SCRIPT_PATH}")
}

fn shell_command(script_url: &str) -> String {
    format!("curl -fsSL {script_url} | sh")
}

fn update_envs(args: &UpdateArgs) -> Vec<(&'static str, String)> {
    let mut envs = Vec::new();
    if let Some(version) = &args.version {
        envs.push(("ENF_VERSION", version.clone()));
    }
    if let Some(install_dir) = &args.install_dir {
        envs.push(("ENF_INSTALL_DIR", install_dir.clone()));
    }
    if let Some(method) = &args.method {
        envs.push(("ENF_INSTALL_METHOD", method.clone()));
    }
    if let Some(repo) = &args.repo {
        envs.push(("ENF_REPO", repo.clone()));
    }
    envs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_command_uses_main_installer_url() {
        let url = install_script_url(DEFAULT_REPO);
        assert_eq!(
            url,
            "https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh"
        );
        assert_eq!(
            shell_command(&url),
            "curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | sh"
        );
    }

    #[test]
    fn update_envs_include_explicit_overrides() {
        let envs = update_envs(&UpdateArgs {
            version: Some("v1.2.3".into()),
            install_dir: Some("/tmp/enf/bin".into()),
            method: Some("cargo".into()),
            repo: Some("Example/enf".into()),
            dry_run: false,
        });

        assert_eq!(
            envs,
            vec![
                ("ENF_VERSION", "v1.2.3".into()),
                ("ENF_INSTALL_DIR", "/tmp/enf/bin".into()),
                ("ENF_INSTALL_METHOD", "cargo".into()),
                ("ENF_REPO", "Example/enf".into()),
            ]
        );
    }
}
