use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::{Config, EmbeddingFallbackConfig, ModelVariant, Provider};

pub const NORMALIZER_VERSION: &str = "normalizer-v1";
pub const CHUNKER_VERSION: &str = "chunker-v2";
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
    let (engine, variant, endpoint) = match config.embedding.provider {
        Provider::Native => (Some("candle".to_string()), variant, None),
        Provider::Ollama | Provider::Openai | Provider::OpenaiCompatible | Provider::Http => {
            (None, None, config.embedding.endpoint.clone())
        }
    };
    let mut profile = EmbeddingProfile {
        profile_hash: String::new(),
        provider: config.embedding.provider.as_str().to_string(),
        engine,
        model: config.embedding.model.clone(),
        variant,
        endpoint,
        dimensions: config.embedding.dimensions,
        document_prefix: config.embedding.document_prefix.clone(),
        query_prefix: config.embedding.query_prefix.clone(),
        normalizer_version: NORMALIZER_VERSION.into(),
        chunker_version: CHUNKER_VERSION.into(),
        serialization_version: EMBEDDING_SERIALIZATION_VERSION.into(),
    };
    profile.profile_hash = profile_hash_with_fallback(&profile, config.embedding.fallback.as_ref());
    profile
}

pub fn profile_hash(profile: &EmbeddingProfile) -> String {
    profile_hash_with_fallback(profile, None)
}

pub fn profile_hash_with_fallback(
    profile: &EmbeddingProfile,
    fallback: Option<&EmbeddingFallbackConfig>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, "model", &profile.model);
    update_hash_field(&mut hasher, "dimensions", &profile.dimensions.to_string());
    update_hash_field(&mut hasher, "document_prefix", &profile.document_prefix);
    update_hash_field(&mut hasher, "query_prefix", &profile.query_prefix);
    update_hash_field(
        &mut hasher,
        "normalizer_version",
        &profile.normalizer_version,
    );
    update_hash_field(&mut hasher, "chunker_version", &profile.chunker_version);
    update_hash_field(
        &mut hasher,
        "serialization_version",
        &profile.serialization_version,
    );
    if let Some(fallback) = fallback {
        let fallback_provider = fallback.provider.as_str();
        update_hash_field(&mut hasher, "fallback.provider", fallback_provider);
        update_optional_hash_field(
            &mut hasher,
            "fallback.endpoint",
            fallback.endpoint.as_deref(),
        );
        update_optional_hash_field(
            &mut hasher,
            "fallback.api_key_env",
            fallback.api_key_env.as_deref(),
        );
    }
    hasher.finalize().to_hex().to_string()
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

fn update_hash_field(hasher: &mut blake3::Hasher, label: &str, value: &str) {
    hasher.update(label.as_bytes());
    hasher.update(&[0]);
    hasher.update(value.as_bytes());
    hasher.update(&[0]);
}

fn update_optional_hash_field(hasher: &mut blake3::Hasher, label: &str, value: Option<&str>) {
    hasher.update(label.as_bytes());
    hasher.update(&[0]);
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(value.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[0]);
}
