use anyhow::Result;

use crate::{config::Config, embed::EmbeddingProvider};

pub fn build_provider(_config: &Config) -> Result<Box<dyn EmbeddingProvider + Send>> {
    anyhow::bail!("embedding providers are not fully implemented yet")
}
