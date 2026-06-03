use elephant_never_forgets::{
    config::{self, Config, Provider},
    providers::{
        build_provider, parse_ollama_embeddings, parse_openai_embeddings, HttpProvider,
        OllamaProvider, OpenAiCompatibleProvider, OpenAiProvider,
    },
};
use serde_json::json;

fn base_config(provider: Provider) -> Config {
    let mut config = Config::default();
    config.embedding.provider = provider;
    config.embedding.model = match config.embedding.provider {
        Provider::Ollama => "nomic-embed-text".into(),
        Provider::Openai => "text-embedding-3-small".into(),
        Provider::OpenaiCompatible => "custom-embed-model".into(),
        Provider::Http => "http-embed-model".into(),
        Provider::Native => "nomic-embed-text-v1.5".into(),
    };
    config.embedding.document_prefix = "search_document: ".into();
    config.embedding.query_prefix = "search_query: ".into();
    config.embedding.endpoint = match config.embedding.provider {
        Provider::Ollama => Some("http://localhost:11434/api/embed".into()),
        Provider::Openai => Some("https://api.openai.com/v1/embeddings".into()),
        Provider::OpenaiCompatible => Some("https://example.invalid/v1/embeddings".into()),
        Provider::Http => Some("https://example.invalid/v1/embeddings".into()),
        Provider::Native => None,
    };
    config.embedding.api_key_env = match config.embedding.provider {
        Provider::Openai => Some("OPENAI_API_KEY".into()),
        Provider::OpenaiCompatible => Some("CUSTOM_EMBED_API_KEY".into()),
        Provider::Http => Some("CUSTOM_HTTP_API_KEY".into()),
        Provider::Ollama | Provider::Native => None,
    };
    config.embedding.dimensions = match config.embedding.provider {
        Provider::Openai => 1536,
        _ => 768,
    };
    config::validate(&config).unwrap();
    config
}

#[test]
fn build_provider_dispatches_to_the_expected_profiles() {
    let ollama = build_provider(&base_config(Provider::Ollama)).unwrap();
    assert_eq!(ollama.profile().provider, "ollama");
    assert_eq!(
        ollama.profile().endpoint.as_deref(),
        Some("http://localhost:11434/api/embed")
    );

    let openai = build_provider(&base_config(Provider::Openai)).unwrap();
    assert_eq!(openai.profile().provider, "openai");
    assert_eq!(
        openai.profile().endpoint.as_deref(),
        Some("https://api.openai.com/v1/embeddings")
    );

    let openai_compatible = build_provider(&base_config(Provider::OpenaiCompatible)).unwrap();
    assert_eq!(openai_compatible.profile().provider, "openai-compatible");
    assert_eq!(
        openai_compatible.profile().endpoint.as_deref(),
        Some("https://example.invalid/v1/embeddings")
    );

    let http = build_provider(&base_config(Provider::Http)).unwrap();
    assert_eq!(http.profile().provider, "http");
    assert_eq!(
        http.profile().endpoint.as_deref(),
        Some("https://example.invalid/v1/embeddings")
    );
}

#[test]
fn ollama_request_and_response_helpers_match_the_api_shape() {
    let provider = OllamaProvider::from_config(&base_config(Provider::Ollama)).unwrap();
    let payload = provider.request_for_texts(&[
        "search_document: first".to_string(),
        "search_document: second".to_string(),
    ]);
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({
            "model": "nomic-embed-text",
            "input": ["search_document: first", "search_document: second"],
        })
    );

    let response = br#"{
        "embeddings": [[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]],
        "total_duration": 1000
    }"#;
    assert_eq!(
        provider.parse_embeddings(response).unwrap(),
        vec![vec![0.1, 0.2, 0.3], vec![1.0, 1.1, 1.2]]
    );

    assert_eq!(
        parse_ollama_embeddings(br#"{"embedding":[0.7,0.8,0.9]}"#).unwrap(),
        vec![vec![0.7, 0.8, 0.9]]
    );
}

#[test]
fn openai_request_and_response_helpers_match_the_api_shape() {
    let provider = OpenAiProvider::from_config(&base_config(Provider::Openai)).unwrap();
    let payload = provider.request_for_texts(&[
        "search_query: alpha".to_string(),
        "search_query: beta".to_string(),
    ]);
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({
            "model": "text-embedding-3-small",
            "input": ["search_query: alpha", "search_query: beta"],
        })
    );

    let response = br#"{
        "object": "list",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [1.1, 1.2]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 2, "total_tokens": 2}
    }"#;
    assert_eq!(
        provider.parse_embeddings(response).unwrap(),
        vec![vec![0.1, 0.2], vec![1.1, 1.2]]
    );

    assert_eq!(
        parse_openai_embeddings(br#"{"data":[{"index":0,"embedding":[3.125]}]}"#).unwrap(),
        vec![vec![3.125]]
    );
}

#[test]
fn openai_compatible_and_http_helpers_share_the_same_json_shape() {
    let compatible =
        OpenAiCompatibleProvider::from_config(&base_config(Provider::OpenaiCompatible)).unwrap();
    let http = HttpProvider::from_config(&base_config(Provider::Http)).unwrap();

    let payload = compatible.request_for_texts(&[
        "search_query: one".to_string(),
        "search_query: two".to_string(),
    ]);
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({
            "model": "custom-embed-model",
            "input": ["search_query: one", "search_query: two"],
        })
    );

    let response = br#"{
        "data": [
            {"index": 0, "embedding": [0.01, 0.02]},
            {"index": 1, "embedding": [0.03, 0.04]}
        ]
    }"#;
    assert_eq!(
        compatible.parse_embeddings(response).unwrap(),
        vec![vec![0.01, 0.02], vec![0.03, 0.04]]
    );
    assert_eq!(
        http.parse_embeddings(response).unwrap(),
        vec![vec![0.01, 0.02], vec![0.03, 0.04]]
    );
}
