use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use walkdir::WalkDir;

use crate::config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub absolute_path: PathBuf,
    pub relative_path: String,
}

pub fn discover(root: &Path, target: &Path, config: &Config) -> Result<Vec<DiscoveredFile>> {
    let include = build_globset(&config.include.patterns)?;
    let exclude = build_globset(&config.exclude.patterns)?;
    let include_all = config.include.patterns.is_empty();
    if !target.exists() {
        anyhow::bail!("index target does not exist: {}", target.display());
    }
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing root {}", root.display()))?;
    let target = target
        .canonicalize()
        .with_context(|| format!("canonicalizing target {}", target.display()))?;
    if !target.starts_with(&root) {
        anyhow::bail!(
            "index target {} is outside project root {}",
            target.display(),
            root.display()
        );
    }

    if target.is_file() {
        let absolute_path = target.to_path_buf();
        let relative_path = normalized_relative_path(&root, &absolute_path)?;
        if exclude.is_match(&relative_path) {
            return Ok(Vec::new());
        }
        return Ok(vec![DiscoveredFile {
            absolute_path,
            relative_path,
        }]);
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(&target) {
        let entry = entry.with_context(|| format!("walking {}", target.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }

        let absolute_path = entry.path().to_path_buf();
        let relative_path = normalized_relative_path(&root, &absolute_path)?;
        if !should_include(&relative_path, &include, &exclude, include_all) {
            continue;
        }

        files.push(DiscoveredFile {
            absolute_path,
            relative_path,
        });
    }

    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn normalized_relative_path(root: &Path, absolute_path: &Path) -> Result<String> {
    let relative = absolute_path.strip_prefix(root).with_context(|| {
        format!(
            "computing relative path for {} from {}",
            absolute_path.display(),
            root.display()
        )
    })?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern).with_context(|| format!("parsing glob pattern {pattern}"))?);
    }
    Ok(builder.build()?)
}

fn should_include(path: &str, include: &GlobSet, exclude: &GlobSet, include_all: bool) -> bool {
    if exclude.is_match(path) {
        return false;
    }
    include_all || include.is_match(path)
}
