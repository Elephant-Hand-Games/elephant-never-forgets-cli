use elephant_never_forgets::{
    config::{self, Config, Provider},
    embed,
    providers::{
        build_provider, parse_image_embeddings, parse_ollama_embeddings, parse_openai_embeddings,
        parse_rerank_response, HttpProvider, ImageEmbeddingProvider, OllamaProvider,
        OpenAiCompatibleProvider, OpenAiProvider, RerankerProvider,
    },
};
use serde_json::json;
use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
};

fn base_config(provider: Provider) -> Config {
    let mut config = Config::default();
    config.embedding.provider = provider;
    config::normalize_provider_defaults(&mut config);
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

fn serve_reranker_fallback() -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/rerank", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut bodies = Vec::new();
        for (status, body) in [
            ("422 Unprocessable Entity", r#"{"detail":"texts rejected"}"#),
            ("200 OK", r#"[{"index":0,"score":0.7}]"#),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&request);
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .or_else(|| {
                    headers
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                })
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body_start = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
                .unwrap_or(request.len());
            while request.len().saturating_sub(body_start) < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            bodies.push(String::from_utf8_lossy(&request[body_start..]).to_string());
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
        bodies
    });
    (endpoint, handle)
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

    let native = build_provider(&base_config(Provider::Native)).unwrap();
    assert_eq!(native.profile().provider, "native");
    assert_eq!(native.profile().engine.as_deref(), Some("candle"));
    assert_eq!(native.profile().model, "nomic-embed-text-v1.5");
    assert_eq!(native.profile().variant.as_deref(), Some("quantized"));
}

#[test]
fn active_profile_matches_provider_profile_for_all_providers() {
    for provider in [
        Provider::Native,
        Provider::Ollama,
        Provider::Openai,
        Provider::OpenaiCompatible,
        Provider::Http,
    ] {
        let config = base_config(provider);
        let provider = build_provider(&config).unwrap();
        assert_eq!(embed::active_profile(&config), provider.profile());
    }
}

#[test]
fn endpoint_changes_embedding_profile_identity() {
    let mut first = base_config(Provider::Http);
    first.embedding.endpoint = Some("https://example.invalid/a".into());
    let mut second = base_config(Provider::Http);
    second.embedding.endpoint = Some("https://example.invalid/b".into());

    assert_ne!(
        embed::active_profile(&first).profile_hash,
        embed::active_profile(&second).profile_hash
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

#[test]
fn image_embedding_helpers_match_endpoint_shape() {
    let mut config = Config::default();
    config.image.embedding.enabled = true;
    config.image.embedding.endpoint = Some("http://localhost:41802/embed".into());
    let provider = ImageEmbeddingProvider::from_config(&config).unwrap();

    let payload = provider.request_for_images(vec!["abc".into()]);
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({
            "images": ["abc"],
            "normalize": true,
        })
    );

    let parsed = parse_image_embeddings(
        br#"{"model":"open_clip/ViT-H-14:laion2b_s32b_b79k","dimensions":1024,"embeddings":[[0.1,0.2]]}"#,
    )
    .unwrap();
    assert_eq!(parsed.dimensions, 1024);
    assert_eq!(parsed.embeddings, vec![vec![0.1, 0.2]]);
}

#[test]
fn reranker_helpers_match_endpoint_shape() {
    let mut config = Config::default();
    config.reranker.enabled = true;
    config.reranker.endpoint = Some("http://localhost:41801/rerank".into());
    let provider = RerankerProvider::from_config(&config).unwrap();

    let payload = provider.request_for_texts("deep learning", vec!["Deep learning text".into()]);
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({
            "query": "deep learning",
            "texts": ["Deep learning text"],
            "raw_scores": false,
            "return_text": true,
            "truncate": true,
            "truncation_direction": "right",
        })
    );
    let documents_payload =
        provider.request_for_documents("deep learning", vec!["Deep learning text".into()]);
    assert_eq!(
        serde_json::to_value(documents_payload).unwrap(),
        json!({
            "query": "deep learning",
            "documents": ["Deep learning text"],
            "raw_scores": false,
            "return_text": true,
            "truncate": true,
            "truncation_direction": "right",
        })
    );

    let parsed = parse_rerank_response(
        br#"[{"index":0,"score":0.98,"text":"Deep learning text"},{"index":1,"score":0.01}]"#,
    )
    .unwrap();
    assert_eq!(parsed[0].index, 0);
    assert_eq!(parsed[0].score, 0.98);
    assert_eq!(parsed[1].text, None);
}

#[test]
fn reranker_falls_back_to_documents_only_after_texts_422() {
    let (endpoint, server) = serve_reranker_fallback();
    let mut config = Config::default();
    config.reranker.enabled = true;
    config.reranker.endpoint = Some(endpoint);
    let provider = RerankerProvider::from_config(&config).unwrap();

    let reranked = provider
        .rerank("deep learning", vec!["Deep learning text".into()])
        .unwrap();
    let bodies = server.join().unwrap();

    assert_eq!(reranked[0].index, 0);
    assert_eq!(reranked[0].score, 0.7);
    assert!(bodies[0].contains("\"texts\""));
    assert!(!bodies[0].contains("\"documents\""));
    assert!(bodies[1].contains("\"documents\""));
    assert!(!bodies[1].contains("\"texts\""));
}

#[test]
fn live_image_embedder_smoke_when_configured() {
    let Ok(endpoint) = std::env::var("ENF_LIVE_IMAGE_EMBED_URL") else {
        eprintln!("skipping live image embedder smoke; ENF_LIVE_IMAGE_EMBED_URL is not set");
        return;
    };
    let mut config = Config::default();
    config.image.embedding.enabled = true;
    config.image.embedding.endpoint = Some(endpoint);
    let provider = ImageEmbeddingProvider::from_config(&config).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let image_path = temp.path().join("pixel.png");
    let png = base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=");
    std::fs::File::create(&image_path)
        .unwrap()
        .write_all(&png)
        .unwrap();

    let vectors = provider
        .embed_image_files(temp.path(), &["pixel.png".to_string()])
        .unwrap();
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 1024);
}

fn base64_decode(input: &str) -> Vec<u8> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    STANDARD.decode(input).unwrap()
}
