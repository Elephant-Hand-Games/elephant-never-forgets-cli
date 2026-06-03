use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::{Config, ModelVariant};

pub const NORMALIZER_VERSION: &str = "normalizer-v1";
pub const CHUNKER_VERSION: &str = "chunker-v1";
pub const EMBEDDING_SERIALIZATION_VERSION: &str = "f32-le-v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingProfile {
    pub profile_hash: String,
    pub provider: String,
    pub engine: Option<String>,
    pub model: String,
    pub variant: Option<String>,
    pub endpoint: Option<String>,
    pub dimensions: usize,
    pub document_prefix: String,
    pub query_prefix: String,
    pub normalizer_version: String,
    pub chunker_version: String,
    pub serialization_version: String,
}

#[async_trait]
pub trait EmbeddingProvider {
    fn profile(&self) -> EmbeddingProfile;
    fn ensure_ready(&mut self) -> Result<()>;
    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>>;
}

pub fn active_profile(config: &Config) -> EmbeddingProfile {
    let variant = config
        .embedding
        .variant
        .as_ref()
        .map(|variant| match variant {
            ModelVariant::Quantized => "quantized".to_string(),
            ModelVariant::Full => "full".to_string(),
        });
    let mut profile = EmbeddingProfile {
        profile_hash: String::new(),
        provider: config.embedding.provider.as_str().to_string(),
        engine: config.embedding.engine.clone(),
        model: config.embedding.model.clone(),
        variant,
        endpoint: config.embedding.endpoint.clone(),
        dimensions: config.embedding.dimensions,
        document_prefix: config.embedding.document_prefix.clone(),
        query_prefix: config.embedding.query_prefix.clone(),
        normalizer_version: NORMALIZER_VERSION.into(),
        chunker_version: CHUNKER_VERSION.into(),
        serialization_version: EMBEDDING_SERIALIZATION_VERSION.into(),
    };
    profile.profile_hash = profile_hash(&profile);
    profile
}

pub fn profile_hash(profile: &EmbeddingProfile) -> String {
    let input = serde_json::json!({
        "provider": profile.provider,
        "engine": profile.engine,
        "model": profile.model,
        "variant": profile.variant,
        "dimensions": profile.dimensions,
        "document_prefix": profile.document_prefix,
        "query_prefix": profile.query_prefix,
        "normalizer_version": profile.normalizer_version,
        "chunker_version": profile.chunker_version,
        "serialization_version": profile.serialization_version,
    });
    blake3::hash(input.to_string().as_bytes())
        .to_hex()
        .to_string()
}

pub fn serialize_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

pub fn deserialize_vector(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("invalid vector byte length {}", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
