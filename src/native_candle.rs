use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::nomic_bert::{
    l2_normalize, mean_pooling, Config as NomicBertConfig, NomicBertModel,
};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::native_gemma3::{Config as Gemma3Config, Gemma3TextModel};

const NOMIC_V15_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
const GEMMA_REPO: &str = "google/embeddinggemma-300m";
const NOMIC_MAX_LENGTH: usize = 512;
const GEMMA_MAX_LENGTH: usize = 2048;
const GEMMA_HF_ACCESS_HELP: &str = "Native Gemma downloads require accepting the Google Gemma license on Hugging Face and configuring an HF token. Visit https://huggingface.co/google/embeddinggemma-300m, accept the license, then run `hf auth login` or set HF_TOKEN before retrying `enf models install gemma --provider native`.";

pub enum NativeCandleEmbedding {
    Nomic(NomicV15CandleEmbedding),
    Gemma(EmbeddingGemma300MCandleEmbedding),
}

impl NativeCandleEmbedding {
    pub fn from_hf(model: &str) -> Result<Self> {
        match model {
            "nomic-embed-text-v1.5" => Ok(Self::Nomic(NomicV15CandleEmbedding::from_hf()?)),
            "google/embeddinggemma-300m" => Ok(Self::Gemma(
                EmbeddingGemma300MCandleEmbedding::from_hf()
                    .context("loading native google/embeddinggemma-300m")?,
            )),
            other => anyhow::bail!("unsupported native Candle embedding model: {other}"),
        }
    }

    pub fn embed<S: AsRef<str>>(&mut self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Nomic(model) => model.embed(texts),
            Self::Gemma(model) => model.embed(texts),
        }
    }
}

pub struct NomicV15CandleEmbedding {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl NomicV15CandleEmbedding {
    pub fn from_hf() -> Result<Self> {
        let device = Device::Cpu;
        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("initializing Hugging Face model cache")?;
        let repo = api.model(NOMIC_V15_REPO.to_string());

        let config_path = repo.get("config.json").context("downloading config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("downloading tokenizer.json")?;
        let weights = repo
            .get("model.safetensors")
            .context("downloading model.safetensors")?;

        Self::from_paths(config_path, tokenizer_path, weights, device)
    }

    fn from_paths(
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weights_path: PathBuf,
        device: Device,
    ) -> Result<Self> {
        let config: NomicBertConfig = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("loading Nomic v1.5 safetensors")?
        };
        let model = NomicBertModel::load(vb, &config).context("loading Nomic v1.5 model")?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!(err.to_string()))
            .with_context(|| format!("loading {}", tokenizer_path.display()))?;
        let pad_token = "[PAD]".to_string();
        let pad_id = tokenizer.token_to_id(&pad_token).unwrap_or(0);
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));
        let _ = tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: NOMIC_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(
                texts.iter().map(|text| text.as_ref()).collect::<Vec<_>>(),
                true,
            )
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();
        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().copied());
            attention_mask.extend(encoding.get_attention_mask().iter().copied());
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, seq_len), &self.device)?;
        let token_type_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &self.device)?;

        let hidden =
            self.model
                .forward(&input_ids, Some(&token_type_ids), Some(&attention_mask))?;
        let pooled = mean_pooling(&hidden, &attention_mask)?;
        let normalized = l2_normalize(&pooled)?;
        normalized.to_vec2::<f32>().map_err(Into::into)
    }
}

struct DenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_function: String,
}

impl<'de> serde::Deserialize<'de> for DenseConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawDenseConfig {
            in_features: usize,
            out_features: usize,
            #[serde(default = "default_true")]
            bias: bool,
            #[serde(default)]
            activation_function: String,
        }

        fn default_true() -> bool {
            true
        }

        let raw = RawDenseConfig::deserialize(deserializer)?;
        Ok(Self {
            in_features: raw.in_features,
            out_features: raw.out_features,
            bias: raw.bias,
            activation_function: raw.activation_function,
        })
    }
}

struct DenseLayer {
    linear: Linear,
    activation_function: String,
}

impl DenseLayer {
    fn load(config_path: PathBuf, weights_path: PathBuf, device: &Device) -> Result<Self> {
        let config: DenseConfig = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .context("loading EmbeddingGemma dense safetensors")?
        };
        let linear = if config.bias {
            candle_nn::linear(config.in_features, config.out_features, vb.pp("linear"))?
        } else {
            candle_nn::linear_no_bias(config.in_features, config.out_features, vb.pp("linear"))?
        };
        Ok(Self {
            linear,
            activation_function: config.activation_function,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(xs)?;
        let activation = self.activation_function.to_ascii_lowercase();
        if activation.contains("gelu") {
            xs.gelu().map_err(Into::into)
        } else if activation.contains("relu") {
            xs.relu().map_err(Into::into)
        } else if activation.contains("tanh") {
            xs.tanh().map_err(Into::into)
        } else {
            Ok(xs)
        }
    }
}

pub struct EmbeddingGemma300MCandleEmbedding {
    model: Gemma3TextModel,
    tokenizer: Tokenizer,
    dense_layers: Vec<DenseLayer>,
    device: Device,
}

impl EmbeddingGemma300MCandleEmbedding {
    pub fn from_hf() -> Result<Self> {
        let device = Device::Cpu;
        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("initializing Hugging Face model cache")?;
        let repo = api.model(GEMMA_REPO.to_string());

        let config_path = repo.get("config.json").with_context(|| {
            format!("downloading {GEMMA_REPO}/config.json. {GEMMA_HF_ACCESS_HELP}")
        })?;
        let tokenizer_path = repo.get("tokenizer.json").with_context(|| {
            format!("downloading {GEMMA_REPO}/tokenizer.json. {GEMMA_HF_ACCESS_HELP}")
        })?;
        let weights = repo.get("model.safetensors").with_context(|| {
            format!("downloading {GEMMA_REPO}/model.safetensors. {GEMMA_HF_ACCESS_HELP}")
        })?;

        let dense_specs = [
            (
                repo.get("2_Dense/config.json").with_context(|| {
                    format!("downloading EmbeddingGemma dense projection config. {GEMMA_HF_ACCESS_HELP}")
                })?,
                repo.get("2_Dense/model.safetensors").with_context(|| {
                    format!("downloading EmbeddingGemma dense projection weights. {GEMMA_HF_ACCESS_HELP}")
                })?,
            ),
            (
                repo.get("3_Dense/config.json").with_context(|| {
                    format!("downloading EmbeddingGemma output projection config. {GEMMA_HF_ACCESS_HELP}")
                })?,
                repo.get("3_Dense/model.safetensors").with_context(|| {
                    format!("downloading EmbeddingGemma output projection weights. {GEMMA_HF_ACCESS_HELP}")
                })?,
            ),
        ];

        Self::from_paths(config_path, tokenizer_path, weights, dense_specs, device)
    }

    fn from_paths(
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weights_path: PathBuf,
        dense_specs: [(PathBuf, PathBuf); 2],
        device: Device,
    ) -> Result<Self> {
        let config: Gemma3Config = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("loading EmbeddingGemma safetensors")?
        };
        let model = Gemma3TextModel::new(&config, vb).context("loading EmbeddingGemma model")?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!(err.to_string()))
            .with_context(|| format!("loading {}", tokenizer_path.display()))?;
        let pad_token = "<pad>".to_string();
        let pad_id = tokenizer.token_to_id(&pad_token).unwrap_or(0);
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));
        let _ = tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: GEMMA_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        let dense_layers = dense_specs
            .into_iter()
            .map(|(config, weights)| DenseLayer::load(config, weights, &device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            model,
            tokenizer,
            dense_layers,
            device,
        })
    }

    pub fn embed<S: AsRef<str>>(&mut self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(
                texts.iter().map(|text| text.as_ref()).collect::<Vec<_>>(),
                true,
            )
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();
        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().copied());
            attention_mask.extend(encoding.get_attention_mask().iter().copied());
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, seq_len), &self.device)?;
        let hidden = self.model.forward_hidden(&input_ids)?;
        let mut pooled = mean_pooling(&hidden, &attention_mask)?;
        for dense in &self.dense_layers {
            pooled = dense.forward(&pooled)?;
        }
        let normalized = l2_normalize(&pooled)?;
        normalized.to_vec2::<f32>().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::NomicV15CandleEmbedding;

    #[test]
    #[ignore = "downloads nomic-embed-text-v1.5 from Hugging Face"]
    fn embeds_are_768d_normalized_and_deterministic() {
        let model = NomicV15CandleEmbedding::from_hf().expect("load native model");
        let text = "search_document: Elephant Never Forgets indexes local markdown.";

        let first = model.embed(&[text]).expect("embed first text").remove(0);
        let second = model.embed(&[text]).expect("embed second text").remove(0);

        assert_eq!(first.len(), 768);
        assert_eq!(second.len(), 768);
        assert!(first.iter().all(|value| !value.is_nan()));
        assert!(second.iter().all(|value| !value.is_nan()));

        let norm = first.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected L2-normalized vector, got norm {norm}"
        );
        for (left, right) in first.iter().zip(second.iter()) {
            assert!(
                (left - right).abs() < 1e-6,
                "expected deterministic output, got {left} != {right}"
            );
        }
    }
}
