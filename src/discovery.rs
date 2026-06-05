use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use walkdir::WalkDir;

use crate::config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub absolute_path: PathBuf,
    pub relative_path: String,
    pub kind: DiscoveredFileKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveredFileKind {
    Text,
    Image,
}

pub fn discover(root: &Path, target: &Path, config: &Config) -> Result<Vec<DiscoveredFile>> {
    let text_include = build_globset(&config.text.include.patterns)?;
    let text_exclude = build_globset(&config.text.exclude.patterns)?;
    let image_include = build_globset(&config.image.include.patterns)?;
    let image_exclude = build_globset(&config.image.exclude.patterns)?;
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
    let enfignore = build_enfignore(&root)?;

    if target.is_file() {
        let absolute_path = target.to_path_buf();
        let relative_path = normalized_relative_path(&root, &absolute_path)?;
        if is_enfignored(&enfignore, &absolute_path, false)
            || is_under_enfignoredir(&root, &absolute_path)
            || is_excluded(&relative_path, &text_exclude, &image_exclude)
        {
            return Ok(Vec::new());
        }
        let Some(kind) = classify_path(
            &relative_path,
            &text_include,
            &text_exclude,
            &image_include,
            &image_exclude,
        ) else {
            return Ok(Vec::new());
        };
        return Ok(vec![DiscoveredFile {
            absolute_path,
            relative_path,
            kind,
        }]);
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(&target).into_iter().filter_entry(|entry| {
        if entry.depth() == 0 {
            return true;
        }
        let path = entry.path();
        if entry.file_type().is_dir() && path.join(".enfignoredir").exists() {
            return false;
        }
        !is_enfignored(&enfignore, path, entry.file_type().is_dir())
    }) {
        let entry = entry.with_context(|| format!("walking {}", target.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }

        let absolute_path = entry.path().to_path_buf();
        let relative_path = normalized_relative_path(&root, &absolute_path)?;
        let Some(kind) = classify_path(
            &relative_path,
            &text_include,
            &text_exclude,
            &image_include,
            &image_exclude,
        ) else {
            continue;
        };

        files.push(DiscoveredFile {
            absolute_path,
            relative_path,
            kind,
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

fn build_enfignore(root: &Path) -> Result<Gitignore> {
    let mut builder = GitignoreBuilder::new(root);
    let ignore_path = root.join(".enfignore");
    if ignore_path.exists() {
        builder.add(ignore_path);
    }
    builder.build().context("building .enfignore matcher")
}

fn is_enfignored(ignore: &Gitignore, path: &Path, is_dir: bool) -> bool {
    ignore.matched(path, is_dir).is_ignore()
}

fn is_under_enfignoredir(root: &Path, path: &Path) -> bool {
    let mut current = path.parent();
    while let Some(dir) = current {
        if !dir.starts_with(root) {
            break;
        }
        if dir.join(".enfignoredir").exists() {
            return true;
        }
        current = dir.parent();
    }
    false
}

fn is_excluded(path: &str, text_exclude: &GlobSet, image_exclude: &GlobSet) -> bool {
    text_exclude.is_match(path) || image_exclude.is_match(path)
}

fn classify_path(
    path: &str,
    text_include: &GlobSet,
    text_exclude: &GlobSet,
    image_include: &GlobSet,
    image_exclude: &GlobSet,
) -> Option<DiscoveredFileKind> {
    if !image_exclude.is_match(path) && image_include.is_match(path) {
        return Some(DiscoveredFileKind::Image);
    }
    if !text_exclude.is_match(path) && text_include.is_match(path) {
        return Some(DiscoveredFileKind::Text);
    }
    None
}
