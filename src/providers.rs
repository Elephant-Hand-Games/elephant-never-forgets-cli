use std::{env, path::Path, time::Duration};

#[cfg(feature = "native-candle")]
use crate::native_candle::NomicV15CandleEmbedding;
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::{
    blocking::Client,
    header::{AUTHORIZATION, CONTENT_TYPE},
    StatusCode, Url,
};
use serde::{Deserialize, Serialize};

use crate::{
    config::{Config, Provider},
    embed::{
        profile_hash, EmbeddingProfile, EmbeddingProvider, CHUNKER_VERSION,
        EMBEDDING_SERIALIZATION_VERSION, NORMALIZER_VERSION,
    },
};

pub fn build_provider(config: &Config) -> Result<Box<dyn EmbeddingProvider + Send>> {
    match config.embedding.provider {
        Provider::Native => Ok(Box::new(NativeCandleProvider::from_config(config)?)),
        Provider::Ollama => Ok(Box::new(OllamaProvider::from_config(config)?)),
        Provider::Openai => Ok(Box::new(OpenAiProvider::from_config(config)?)),
        Provider::OpenaiCompatible => Ok(Box::new(OpenAiCompatibleProvider::from_config(config)?)),
        Provider::Http => Ok(Box::new(HttpProvider::from_config(config)?)),
    }
}

#[cfg(feature = "native-candle")]
pub struct NativeCandleProvider {
    model: Option<NomicV15CandleEmbedding>,
    model_name: String,
    variant: Option<String>,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[cfg(not(feature = "native-candle"))]
pub struct NativeCandleProvider {
    profile: EmbeddingProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiEmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageEmbeddingsRequest {
    pub images: Vec<String>,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageQueryEmbeddingsRequest {
    pub query: String,
    pub normalize: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ImageEmbeddingsResponse {
    pub model: String,
    pub dimensions: usize,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankRequest {
    pub query: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<String>>,
    pub raw_scores: bool,
    pub return_text: bool,
    pub truncate: bool,
    pub truncation_direction: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RerankItem {
    pub index: usize,
    pub score: f32,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct RerankResponse {
    results: Vec<RerankLegacyItem>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct RerankLegacyItem {
    index: usize,
    #[serde(rename = "relevance_score")]
    score: f32,
    #[serde(rename = "document", default)]
    text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct OllamaEmbedResponse {
    embeddings: Option<Vec<Vec<f32>>>,
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct OpenAiEmbeddingsResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct OpenAiEmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    client: Client,
    endpoint: String,
    model: String,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: Client,
    endpoint: String,
    api_key_env: String,
    model: String,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    client: Client,
    endpoint: String,
    api_key_env: Option<String>,
    model: String,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[derive(Debug, Clone)]
pub struct HttpProvider {
    client: Client,
    endpoint: String,
    api_key_env: Option<String>,
    model: String,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[derive(Debug, Clone)]
pub struct ImageEmbeddingProvider {
    client: Client,
    endpoint: String,
    query_endpoint: Option<String>,
    normalize: bool,
}

#[derive(Debug, Clone)]
pub struct RerankerProvider {
    client: Client,
    endpoint: String,
    raw_scores: bool,
    return_text: bool,
    truncate: bool,
    truncation_direction: String,
}

impl NativeCandleProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        #[cfg(feature = "native-candle")]
        {
            Self::from_native_config(config)
        }
        #[cfg(not(feature = "native-candle"))]
        {
            Ok(Self {
                profile: native_profile_for_config(config),
            })
        }
    }

    #[cfg(feature = "native-candle")]
    fn from_native_config(config: &Config) -> Result<Self> {
        if config.embedding.model != "nomic-embed-text-v1.5" {
            anyhow::bail!(
                "unsupported native Candle model profile: model={}. \
                 This build ships nomic-embed-text-v1.5 for native embeddings.",
                config.embedding.model
            );
        }
        if !matches!(
            config.embedding.variant,
            Some(crate::config::ModelVariant::Quantized)
        ) {
            anyhow::bail!(
                "embedding.variant must be \"quantized\" for native Candle model {}",
                config.embedding.model
            );
        }
        let variant = config
            .embedding
            .variant
            .as_ref()
            .map(|variant| match variant {
                crate::config::ModelVariant::Quantized => "quantized".to_string(),
                crate::config::ModelVariant::Full => "full".to_string(),
            });
        Ok(Self {
            model: None,
            model_name: config.embedding.model.clone(),
            variant,
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    #[cfg(feature = "native-candle")]
    fn model(&mut self) -> Result<&mut NomicV15CandleEmbedding> {
        if self.model.is_none() {
            self.model = Some(NomicV15CandleEmbedding::from_hf()?);
        }
        Ok(self.model.as_mut().expect("model was initialized"))
    }
}

impl OllamaProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            client: embedding_client()?,
            endpoint: required_endpoint(config.embedding.endpoint.as_deref(), "ollama")?,
            model: config.embedding.model.clone(),
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    pub fn request_for_texts(&self, texts: &[String]) -> OllamaEmbedRequest {
        OllamaEmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        }
    }

    pub fn parse_embeddings(&self, body: &[u8]) -> Result<Vec<Vec<f32>>> {
        parse_ollama_embeddings(body)
    }

    fn execute(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = self.request_for_texts(texts);
        let response = self
            .client
            .post(&self.endpoint)
            .json(&request)
            .send()
            .with_context(|| format!("posting embeddings request to {}", self.endpoint))?
            .error_for_status()
            .with_context(|| format!("embedding request failed for {}", self.endpoint))?
            .bytes()
            .context("reading Ollama embedding response body")?;
        let embeddings = self.parse_embeddings(&response)?;
        if embeddings.len() != texts.len() {
            anyhow::bail!(
                "ollama embedding response returned {} vectors for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }
        Ok(embeddings)
    }
}

impl OpenAiProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            client: embedding_client()?,
            endpoint: required_endpoint(config.embedding.endpoint.as_deref(), "openai")?,
            api_key_env: config
                .embedding
                .api_key_env
                .clone()
                .unwrap_or_else(|| "OPENAI_API_KEY".to_string()),
            model: config.embedding.model.clone(),
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    pub fn request_for_texts(&self, texts: &[String]) -> OpenAiEmbeddingsRequest {
        OpenAiEmbeddingsRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        }
    }

    pub fn parse_embeddings(&self, body: &[u8]) -> Result<Vec<Vec<f32>>> {
        parse_openai_embeddings(body)
    }

    fn execute(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post(&self.endpoint)
            .json(&self.request_for_texts(texts))
            .header(AUTHORIZATION, bearer_token(&self.api_key_env)?)
            .header(CONTENT_TYPE, "application/json")
            .send()
            .with_context(|| format!("posting embeddings request to {}", self.endpoint))?
            .error_for_status()
            .with_context(|| format!("embedding request failed for {}", self.endpoint))?
            .bytes()
            .context("reading OpenAI embedding response body")?;
        let embeddings = self.parse_embeddings(&response)?;
        if embeddings.len() != texts.len() {
            anyhow::bail!(
                "openai embedding response returned {} vectors for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }
        Ok(embeddings)
    }
}

impl OpenAiCompatibleProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            client: embedding_client()?,
            endpoint: required_endpoint(config.embedding.endpoint.as_deref(), "openai-compatible")?,
            api_key_env: config.embedding.api_key_env.clone(),
            model: config.embedding.model.clone(),
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    pub fn request_for_texts(&self, texts: &[String]) -> OpenAiEmbeddingsRequest {
        OpenAiEmbeddingsRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        }
    }

    pub fn parse_embeddings(&self, body: &[u8]) -> Result<Vec<Vec<f32>>> {
        parse_openai_embeddings(body)
    }

    fn execute(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut request = self
            .client
            .post(&self.endpoint)
            .json(&self.request_for_texts(texts))
            .header(CONTENT_TYPE, "application/json");
        if let Some(api_key) = api_key_value(self.api_key_env.as_deref())? {
            request = request.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }
        let response = request
            .send()
            .with_context(|| format!("posting embeddings request to {}", self.endpoint))?
            .error_for_status()
            .with_context(|| format!("embedding request failed for {}", self.endpoint))?
            .bytes()
            .context("reading OpenAI-compatible embedding response body")?;
        let embeddings = self.parse_embeddings(&response)?;
        if embeddings.len() != texts.len() {
            anyhow::bail!(
                "openai-compatible embedding response returned {} vectors for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }
        Ok(embeddings)
    }
}

impl HttpProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            client: embedding_client()?,
            endpoint: required_endpoint(config.embedding.endpoint.as_deref(), "http")?,
            api_key_env: config.embedding.api_key_env.clone(),
            model: config.embedding.model.clone(),
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    pub fn request_for_texts(&self, texts: &[String]) -> OpenAiEmbeddingsRequest {
        OpenAiEmbeddingsRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        }
    }

    pub fn parse_embeddings(&self, body: &[u8]) -> Result<Vec<Vec<f32>>> {
        parse_openai_embeddings(body)
    }

    fn execute(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut request = self
            .client
            .post(&self.endpoint)
            .json(&self.request_for_texts(texts))
            .header(CONTENT_TYPE, "application/json");
        if let Some(api_key) = api_key_value(self.api_key_env.as_deref())? {
            request = request.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }
        let response = request
            .send()
            .with_context(|| format!("posting embeddings request to {}", self.endpoint))?
            .error_for_status()
            .with_context(|| format!("embedding request failed for {}", self.endpoint))?
            .bytes()
            .context("reading custom HTTP embedding response body")?;
        let embeddings = self.parse_embeddings(&response)?;
        if embeddings.len() != texts.len() {
            anyhow::bail!(
                "custom http embedding response returned {} vectors for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }
        Ok(embeddings)
    }
}

impl ImageEmbeddingProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        let endpoint = config
            .image
            .embedding
            .endpoint
            .as_deref()
            .filter(|endpoint| !endpoint.trim().is_empty())
            .context("image embedding endpoint is required")?;
        validate_endpoint(endpoint, "image embedding")?;
        Ok(Self {
            client: embedding_client()?,
            endpoint: endpoint.to_string(),
            query_endpoint: config.image.embedding.query_endpoint.clone(),
            normalize: config.image.embedding.normalize,
        })
    }

    pub fn request_for_images(&self, images: Vec<String>) -> ImageEmbeddingsRequest {
        ImageEmbeddingsRequest {
            images,
            normalize: self.normalize,
        }
    }

    pub fn request_for_query(&self, query: &str) -> ImageQueryEmbeddingsRequest {
        ImageQueryEmbeddingsRequest {
            query: query.to_string(),
            normalize: self.normalize,
        }
    }

    pub fn parse_embeddings(&self, body: &[u8]) -> Result<ImageEmbeddingsResponse> {
        parse_image_embeddings(body)
    }

    pub fn embed_image_files(
        &self,
        root: &Path,
        relative_paths: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let images = relative_paths
            .iter()
            .map(|path| image_base64(&root.join(path)))
            .collect::<Result<Vec<_>>>()?;
        let response = self
            .client
            .post(&self.endpoint)
            .json(&self.request_for_images(images))
            .header(CONTENT_TYPE, "application/json")
            .send()
            .with_context(|| format!("posting image embedding request to {}", self.endpoint))?;
        let status = response.status();
        let response = response
            .bytes()
            .context("reading image embedding response body")?;
        if !status.is_success() {
            anyhow::bail!(
                "image embedding request failed for {} with status {} for files [{}]: {}",
                self.endpoint,
                status,
                relative_paths.join(", "),
                String::from_utf8_lossy(&response)
            );
        }
        let parsed = self.parse_embeddings(&response)?;
        if parsed.embeddings.len() != relative_paths.len() {
            anyhow::bail!(
                "image embedding response returned {} vectors for {} images",
                parsed.embeddings.len(),
                relative_paths.len()
            );
        }
        Ok(parsed.embeddings)
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let endpoint = self
            .query_endpoint
            .as_deref()
            .filter(|endpoint| !endpoint.trim().is_empty())
            .context("image.embedding.query-endpoint is required for image vector search")?;
        validate_endpoint(endpoint, "image query embedding")?;
        let response = self
            .client
            .post(endpoint)
            .json(&self.request_for_query(query))
            .header(CONTENT_TYPE, "application/json")
            .send()
            .with_context(|| format!("posting image query embedding request to {endpoint}"))?;
        let status = response.status();
        let response = response
            .bytes()
            .context("reading image query embedding response body")?;
        if !status.is_success() {
            anyhow::bail!(
                "image query embedding request failed for {} with status {}: {}",
                endpoint,
                status,
                String::from_utf8_lossy(&response)
            );
        }
        let parsed = self.parse_embeddings(&response)?;
        if parsed.embeddings.len() != 1 {
            anyhow::bail!(
                "image query embedding response returned {} vectors for one query",
                parsed.embeddings.len()
            );
        }
        Ok(parsed.embeddings[0].clone())
    }
}

impl RerankerProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        let endpoint = config
            .reranker
            .endpoint
            .as_deref()
            .filter(|endpoint| !endpoint.trim().is_empty())
            .context("reranker endpoint is required")?;
        validate_endpoint(endpoint, "reranker")?;
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(config.reranker.timeout_seconds))
                .build()
                .context("building reranker HTTP client")?,
            endpoint: endpoint.to_string(),
            raw_scores: config.reranker.raw_scores,
            return_text: config.reranker.return_text,
            truncate: config.reranker.truncate,
            truncation_direction: config.reranker.truncation_direction.clone(),
        })
    }

    pub fn request_for_texts(&self, query: &str, texts: Vec<String>) -> RerankRequest {
        RerankRequest {
            query: query.to_string(),
            texts,
            documents: None,
            raw_scores: self.raw_scores,
            return_text: self.return_text,
            truncate: self.truncate,
            truncation_direction: self.truncation_direction.clone(),
        }
    }

    pub fn request_for_documents(&self, query: &str, documents: Vec<String>) -> RerankRequest {
        RerankRequest {
            query: query.to_string(),
            texts: Vec::new(),
            documents: Some(documents),
            raw_scores: self.raw_scores,
            return_text: self.return_text,
            truncate: self.truncate,
            truncation_direction: self.truncation_direction.clone(),
        }
    }

    pub fn parse_rerank(&self, body: &[u8]) -> Result<Vec<RerankItem>> {
        parse_rerank_response(body)
    }

    pub fn rerank(&self, query: &str, texts: Vec<String>) -> Result<Vec<RerankItem>> {
        let response = self
            .client
            .post(&self.endpoint)
            .json(&self.request_for_texts(query, texts.clone()))
            .header(CONTENT_TYPE, "application/json")
            .send()
            .with_context(|| format!("posting rerank request to {}", self.endpoint))?;
        let status = response.status();
        let raw_body = response.bytes().context("reading rerank response body")?;
        if status.is_success() {
            return self.parse_rerank(&raw_body);
        }
        if status == StatusCode::UNPROCESSABLE_ENTITY {
            let body = String::from_utf8_lossy(&raw_body);
            if body.contains("\"texts\"") || body.contains("texts") {
                let response = self
                    .client
                    .post(&self.endpoint)
                    .json(&self.request_for_documents(query, texts))
                    .header(CONTENT_TYPE, "application/json")
                    .send()
                    .with_context(|| format!("posting rerank request to {}", self.endpoint))?
                    .error_for_status()
                    .with_context(|| format!("rerank request failed for {}", self.endpoint))?
                    .bytes()
                    .context("reading rerank response body")?;
                return self.parse_rerank(&response);
            }
        }
        anyhow::bail!(
            "rerank request failed for {}: {}",
            self.endpoint,
            String::from_utf8_lossy(&raw_body)
        )
    }
}

impl EmbeddingProvider for NativeCandleProvider {
    fn profile(&self) -> EmbeddingProfile {
        #[cfg(feature = "native-candle")]
        {
            native_profile(
                &self.model_name,
                self.variant.as_deref(),
                self.dimensions,
                &self.document_prefix,
                &self.query_prefix,
            )
        }
        #[cfg(not(feature = "native-candle"))]
        {
            self.profile.clone()
        }
    }

    fn ensure_ready(&mut self) -> Result<()> {
        #[cfg(feature = "native-candle")]
        {
            let _ = self.model()?;
            Ok(())
        }
        #[cfg(not(feature = "native-candle"))]
        {
            native_candle_unavailable()
        }
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "native-candle")]
        {
            if texts.is_empty() {
                Ok(Vec::new())
            } else {
                let prefixed = texts
                    .iter()
                    .map(|text| format!("{}{}", self.document_prefix, text))
                    .collect::<Vec<_>>();
                self.model()?.embed(&prefixed)
            }
        }
        #[cfg(not(feature = "native-candle"))]
        {
            let _ = texts;
            native_candle_unavailable()
        }
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "native-candle")]
        {
            let prefixed = format!("{}{}", self.query_prefix, query);
            let mut embeddings = self.model()?.embed(&[prefixed])?;
            embeddings
                .pop()
                .context("native Candle returned no query embedding")
        }
        #[cfg(not(feature = "native-candle"))]
        {
            let _ = query;
            native_candle_unavailable()
        }
    }
}

#[cfg(feature = "native-candle")]
fn native_profile(
    model_name: &str,
    variant: Option<&str>,
    dimensions: usize,
    document_prefix: &str,
    query_prefix: &str,
) -> EmbeddingProfile {
    let mut profile = EmbeddingProfile {
        profile_hash: String::new(),
        provider: "native".into(),
        engine: Some("candle".into()),
        model: model_name.to_string(),
        variant: variant.map(str::to_string),
        endpoint: None,
        dimensions,
        document_prefix: document_prefix.to_string(),
        query_prefix: query_prefix.to_string(),
        normalizer_version: NORMALIZER_VERSION.into(),
        chunker_version: CHUNKER_VERSION.into(),
        serialization_version: EMBEDDING_SERIALIZATION_VERSION.into(),
    };
    profile.profile_hash = profile_hash(&profile);
    profile
}

#[cfg(not(feature = "native-candle"))]
fn native_profile_for_config(config: &Config) -> EmbeddingProfile {
    let variant = config
        .embedding
        .variant
        .as_ref()
        .map(|variant| match variant {
            crate::config::ModelVariant::Quantized => "quantized",
            crate::config::ModelVariant::Full => "full",
        });
    let mut profile = EmbeddingProfile {
        profile_hash: String::new(),
        provider: "native".into(),
        engine: config.embedding.engine.clone(),
        model: config.embedding.model.clone(),
        variant: variant.map(str::to_string),
        endpoint: None,
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

#[cfg(not(feature = "native-candle"))]
fn native_candle_unavailable<T>() -> Result<T> {
    anyhow::bail!(
        "native Candle embeddings are not available in this build.\n\
         Use `enf init --provider ollama --model nomic-embed-text` or another remote provider, \
         or build enf from source with native-candle enabled."
    )
}

impl EmbeddingProvider for OllamaProvider {
    fn profile(&self) -> EmbeddingProfile {
        profile_for(
            "ollama",
            None,
            &self.model,
            Some(&self.endpoint),
            self.dimensions,
            &self.document_prefix,
            &self.query_prefix,
        )
    }

    fn ensure_ready(&mut self) -> Result<()> {
        validate_endpoint(&self.endpoint, "ollama")
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.execute(
            &texts
                .iter()
                .map(|text| format!("{}{}", self.document_prefix, text))
                .collect::<Vec<_>>(),
        )
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.execute(&[format!("{}{}", self.query_prefix, query)])?;
        embeddings
            .into_iter()
            .next()
            .context("ollama returned no query embedding")
    }
}

impl EmbeddingProvider for OpenAiProvider {
    fn profile(&self) -> EmbeddingProfile {
        profile_for(
            "openai",
            None,
            &self.model,
            Some(&self.endpoint),
            self.dimensions,
            &self.document_prefix,
            &self.query_prefix,
        )
    }

    fn ensure_ready(&mut self) -> Result<()> {
        validate_endpoint(&self.endpoint, "openai")?;
        ensure_env_present(&self.api_key_env)
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.execute(
            &texts
                .iter()
                .map(|text| format!("{}{}", self.document_prefix, text))
                .collect::<Vec<_>>(),
        )
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.execute(&[format!("{}{}", self.query_prefix, query)])?;
        embeddings
            .into_iter()
            .next()
            .context("openai returned no query embedding")
    }
}

impl EmbeddingProvider for OpenAiCompatibleProvider {
    fn profile(&self) -> EmbeddingProfile {
        profile_for(
            "openai-compatible",
            None,
            &self.model,
            Some(&self.endpoint),
            self.dimensions,
            &self.document_prefix,
            &self.query_prefix,
        )
    }

    fn ensure_ready(&mut self) -> Result<()> {
        validate_endpoint(&self.endpoint, "openai-compatible")
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.execute(
            &texts
                .iter()
                .map(|text| format!("{}{}", self.document_prefix, text))
                .collect::<Vec<_>>(),
        )
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.execute(&[format!("{}{}", self.query_prefix, query)])?;
        embeddings
            .into_iter()
            .next()
            .context("openai-compatible returned no query embedding")
    }
}

impl EmbeddingProvider for HttpProvider {
    fn profile(&self) -> EmbeddingProfile {
        profile_for(
            "http",
            None,
            &self.model,
            Some(&self.endpoint),
            self.dimensions,
            &self.document_prefix,
            &self.query_prefix,
        )
    }

    fn ensure_ready(&mut self) -> Result<()> {
        validate_endpoint(&self.endpoint, "http")
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.execute(
            &texts
                .iter()
                .map(|text| format!("{}{}", self.document_prefix, text))
                .collect::<Vec<_>>(),
        )
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.execute(&[format!("{}{}", self.query_prefix, query)])?;
        embeddings
            .into_iter()
            .next()
            .context("custom http returned no query embedding")
    }
}

pub fn parse_ollama_embeddings(body: &[u8]) -> Result<Vec<Vec<f32>>> {
    let response: OllamaEmbedResponse =
        serde_json::from_slice(body).context("parsing Ollama embedding response")?;
    if let Some(embeddings) = response.embeddings {
        return Ok(embeddings);
    }
    if let Some(embedding) = response.embedding {
        return Ok(vec![embedding]);
    }
    anyhow::bail!("ollama embedding response did not include embeddings");
}

pub fn parse_openai_embeddings(body: &[u8]) -> Result<Vec<Vec<f32>>> {
    let response: OpenAiEmbeddingsResponse =
        serde_json::from_slice(body).context("parsing OpenAI embedding response")?;
    if response.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; response.data.len()];
    for item in response.data {
        if item.index >= embeddings.len() {
            anyhow::bail!(
                "openai embedding response index {} is out of range",
                item.index
            );
        }
        if embeddings[item.index].is_some() {
            anyhow::bail!(
                "openai embedding response contained a duplicate index {}",
                item.index
            );
        }
        embeddings[item.index] = Some(item.embedding);
    }
    embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            embedding.with_context(|| format!("openai embedding response missing index {}", index))
        })
        .collect()
}

pub fn parse_image_embeddings(body: &[u8]) -> Result<ImageEmbeddingsResponse> {
    let response: ImageEmbeddingsResponse =
        serde_json::from_slice(body).context("parsing image embedding response")?;
    if response.dimensions == 0 {
        anyhow::bail!("image embedding response dimensions must be greater than 0");
    }
    Ok(response)
}

pub fn parse_rerank_response(body: &[u8]) -> Result<Vec<RerankItem>> {
    let legacy =
        serde_json::from_slice::<Vec<RerankItem>>(body).context("parsing reranker response");
    if let Ok(items) = legacy {
        return Ok(items);
    }
    let wrapped: RerankResponse =
        serde_json::from_slice(body).context("parsing reranker response")?;
    Ok(wrapped
        .results
        .into_iter()
        .map(|item| RerankItem {
            index: item.index,
            score: item.score,
            text: item.text,
        })
        .collect())
}

fn image_base64(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    Ok(STANDARD.encode(bytes))
}

pub fn openai_request_payload(model: &str, texts: &[String]) -> OpenAiEmbeddingsRequest {
    OpenAiEmbeddingsRequest {
        model: model.to_string(),
        input: texts.to_vec(),
    }
}

pub fn ollama_request_payload(model: &str, texts: &[String]) -> OllamaEmbedRequest {
    OllamaEmbedRequest {
        model: model.to_string(),
        input: texts.to_vec(),
    }
}

fn profile_for(
    provider: &str,
    engine: Option<String>,
    model: &str,
    endpoint: Option<&str>,
    dimensions: usize,
    document_prefix: &str,
    query_prefix: &str,
) -> EmbeddingProfile {
    let mut profile = EmbeddingProfile {
        profile_hash: String::new(),
        provider: provider.to_string(),
        engine,
        model: model.to_string(),
        variant: None,
        endpoint: endpoint.map(str::to_string),
        dimensions,
        document_prefix: document_prefix.to_string(),
        query_prefix: query_prefix.to_string(),
        normalizer_version: NORMALIZER_VERSION.into(),
        chunker_version: CHUNKER_VERSION.into(),
        serialization_version: EMBEDDING_SERIALIZATION_VERSION.into(),
    };
    profile.profile_hash = profile_hash(&profile);
    profile
}

fn required_endpoint(endpoint: Option<&str>, provider: &str) -> Result<String> {
    let endpoint = endpoint
        .filter(|endpoint| !endpoint.trim().is_empty())
        .context(format!("{} endpoint is required", provider))?;
    validate_endpoint(endpoint, provider)?;
    Ok(endpoint.to_string())
}

fn validate_endpoint(endpoint: &str, provider: &str) -> Result<()> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        anyhow::bail!("{} endpoint is required", provider);
    }
    let url = Url::parse(endpoint).with_context(|| format!("parsing {} endpoint", provider))?;
    match url.scheme() {
        "http" | "https" => Ok(()),
        scheme => anyhow::bail!(
            "{} endpoint must use http or https, got {}",
            provider,
            scheme
        ),
    }
}

fn embedding_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("building embedding HTTP client")
}

fn api_key_value(api_key_env: Option<&str>) -> Result<Option<String>> {
    match api_key_env {
        Some(env_name) => {
            Ok(Some(env::var(env_name).with_context(|| {
                format!("reading API key from {}", env_name)
            })?))
        }
        None => Ok(None),
    }
}

fn ensure_env_present(env_name: &str) -> Result<()> {
    let _ = env::var(env_name)
        .with_context(|| format!("missing API key environment variable {}", env_name))?;
    Ok(())
}

fn bearer_token(env_name: &str) -> Result<String> {
    let value = env::var(env_name)
        .with_context(|| format!("missing API key environment variable {}", env_name))?;
    Ok(format!("Bearer {}", value))
}
