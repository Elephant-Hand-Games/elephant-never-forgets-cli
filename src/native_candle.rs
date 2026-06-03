use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{
    l2_normalize, mean_pooling, Config as NomicBertConfig, NomicBertModel,
};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const NOMIC_V15_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
const MAX_LENGTH: usize = 512;

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
                max_length: MAX_LENGTH,
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
