use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub absolute_path: PathBuf,
    pub relative_path: String,
}

pub fn discover(root: &Path, target: &Path, _config: &Config) -> Result<Vec<DiscoveredFile>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(target)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let absolute_path = entry.path().to_path_buf();
        let relative_path = absolute_path
            .strip_prefix(root)
            .unwrap_or(&absolute_path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push(DiscoveredFile {
            absolute_path,
            relative_path,
        });
    }
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}
