use std::env;

use anyhow::{Context, Result};
#[cfg(feature = "native-fastembed")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use reqwest::{
    blocking::Client,
    header::{AUTHORIZATION, CONTENT_TYPE},
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
        Provider::Native => Ok(Box::new(NativeFastEmbedProvider::from_config(config)?)),
        Provider::Ollama => Ok(Box::new(OllamaProvider::from_config(config)?)),
        Provider::Openai => Ok(Box::new(OpenAiProvider::from_config(config)?)),
        Provider::OpenaiCompatible => Ok(Box::new(OpenAiCompatibleProvider::from_config(config)?)),
        Provider::Http => Ok(Box::new(HttpProvider::from_config(config)?)),
    }
}

#[cfg(feature = "native-fastembed")]
pub struct NativeFastEmbedProvider {
    model: Option<TextEmbedding>,
    embedding_model: EmbeddingModel,
    model_name: String,
    variant: Option<String>,
    dimensions: usize,
    document_prefix: String,
    query_prefix: String,
}

#[cfg(not(feature = "native-fastembed"))]
pub struct NativeFastEmbedProvider {
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

impl NativeFastEmbedProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        #[cfg(feature = "native-fastembed")]
        {
            Self::from_native_config(config)
        }
        #[cfg(not(feature = "native-fastembed"))]
        {
            Ok(Self {
                profile: native_profile_for_config(config),
            })
        }
    }

    #[cfg(feature = "native-fastembed")]
    fn from_native_config(config: &Config) -> Result<Self> {
        let variant = config
            .embedding
            .variant
            .as_ref()
            .map(|variant| match variant {
                crate::config::ModelVariant::Quantized => "quantized".to_string(),
                crate::config::ModelVariant::Full => "full".to_string(),
            });
        let embedding_model = native_model_for(&config.embedding.model, variant.as_deref())?;
        Ok(Self {
            model: None,
            embedding_model,
            model_name: config.embedding.model.clone(),
            variant,
            dimensions: config.embedding.dimensions,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        })
    }

    #[cfg(feature = "native-fastembed")]
    fn model(&mut self) -> Result<&mut TextEmbedding> {
        if self.model.is_none() {
            self.model = Some(TextEmbedding::try_new(InitOptions::new(
                self.embedding_model.clone(),
            ))?);
        }
        Ok(self.model.as_mut().expect("model was initialized"))
    }
}

impl OllamaProvider {
    pub fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
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
            client: Client::new(),
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
            client: Client::new(),
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
            client: Client::new(),
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

impl EmbeddingProvider for NativeFastEmbedProvider {
    fn profile(&self) -> EmbeddingProfile {
        #[cfg(feature = "native-fastembed")]
        {
            native_profile(
                &self.model_name,
                self.variant.as_deref(),
                self.dimensions,
                &self.document_prefix,
                &self.query_prefix,
            )
        }
        #[cfg(not(feature = "native-fastembed"))]
        {
            self.profile.clone()
        }
    }

    fn ensure_ready(&mut self) -> Result<()> {
        #[cfg(feature = "native-fastembed")]
        {
            let _ = self.model()?;
            Ok(())
        }
        #[cfg(not(feature = "native-fastembed"))]
        {
            native_fastembed_unavailable()
        }
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "native-fastembed")]
        {
            if texts.is_empty() {
                Ok(Vec::new())
            } else {
                let prefixed = texts
                    .iter()
                    .map(|text| format!("{}{}", self.document_prefix, text))
                    .collect::<Vec<_>>();
                self.model()?.embed(prefixed, None)
            }
        }
        #[cfg(not(feature = "native-fastembed"))]
        {
            let _ = texts;
            native_fastembed_unavailable()
        }
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "native-fastembed")]
        {
            let prefixed = format!("{}{}", self.query_prefix, query);
            let mut embeddings = self.model()?.embed(vec![prefixed], None)?;
            embeddings
                .pop()
                .context("native fastembed returned no query embedding")
        }
        #[cfg(not(feature = "native-fastembed"))]
        {
            let _ = query;
            native_fastembed_unavailable()
        }
    }
}

#[cfg(feature = "native-fastembed")]
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
        engine: Some("fastembed".into()),
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

#[cfg(not(feature = "native-fastembed"))]
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
        engine: Some("fastembed".into()),
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

#[cfg(not(feature = "native-fastembed"))]
fn native_fastembed_unavailable<T>() -> Result<T> {
    anyhow::bail!(
        "native fastembed is not available in this portable build.\n\
         Use `enf init --provider ollama --model nomic-embed-text` or another remote provider, \
         or build enf from source with native fastembed enabled on a supported CPU."
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
    Ok(endpoint.to_string())
}

fn validate_endpoint(endpoint: &str, provider: &str) -> Result<()> {
    if endpoint.trim().is_empty() {
        anyhow::bail!("{} endpoint is required", provider);
    }
    Ok(())
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

#[cfg(feature = "native-fastembed")]
fn native_model_for(model: &str, variant: Option<&str>) -> Result<EmbeddingModel> {
    match (model, variant.unwrap_or("quantized")) {
        ("nomic-embed-text-v1.5", "quantized") => Ok(EmbeddingModel::NomicEmbedTextV15Q),
        ("nomic-embed-text-v1.5", "full") => Ok(EmbeddingModel::NomicEmbedTextV15),
        _ => anyhow::bail!(
            "unsupported native fastembed model profile: model={model}, variant={}",
            variant.unwrap_or("none")
        ),
    }
}
