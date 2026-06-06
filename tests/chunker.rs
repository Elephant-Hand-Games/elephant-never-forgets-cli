use elephant_never_forgets::chunker;

#[test]
fn markdown_chunker_preserves_breadcrumbs_tables_lists_and_fences() {
    let source = r#"# Guide

Intro paragraph.

## Install

| Tool | Version |
| ---- | ------- |
| enf  | 1.3.0   |

- first
  - nested
- second

```rust
fn main() {
    println!("hello");
}
```
"#;

    let chunks = chunker::process_file_for_rag("docs/guide.md", source.as_bytes());

    assert!(!chunks.is_empty());
    let joined = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join("\n---\n");
    assert!(joined.contains("File: docs/guide.md"));
    assert!(joined.contains("Language: markdown"));
    assert!(joined.contains("Section: Guide > Install"));
    assert!(joined.contains("| Tool | Version |"));
    assert!(joined.contains("- first\n  - nested\n- second"));
    assert!(joined.contains("```rust\nfn main()"));
}

#[test]
fn smart_chunker_dispatches_code_and_config_extensions() {
    let rust = r#"pub struct Engine {
    value: usize,
}

impl Engine {
    pub fn start(&self) {}

    pub fn stop(&self) {}
}
"#;
    let rust_chunks = chunker::process_file_for_rag("src/engine.rs", rust.as_bytes());
    assert!(rust_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Language: rust")));
    assert!(rust_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Symbol: Engine") || chunk.text.contains("Parent impl")));

    let tsx = r#"export function SearchPanel() {
  const handleSubmit = () => {};
  return <form onSubmit={handleSubmit} />;
}
"#;
    let tsx_chunks = chunker::process_file_for_rag("src/SearchPanel.tsx", tsx.as_bytes());
    assert!(tsx_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Language: tsx")));
    assert!(tsx_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Symbol: SearchPanel")));

    let yaml = r#"name: CI
on: push
jobs:
  test:
    runs-on: ubuntu-latest
"#;
    let yaml_chunks = chunker::process_file_for_rag(".github/workflows/ci.yml", yaml.as_bytes());
    assert!(yaml_chunks
        .iter()
        .any(|chunk| chunk.text.contains("github_actions_job")));
    assert!(yaml_chunks
        .iter()
        .any(|chunk| chunk.text.contains("jobs.test")));
}

#[test]
fn json_and_toml_chunks_include_paths_and_table_context() {
    let json = r#"{"package":{"name":"enf","version":"1.3.0"},"items":[{"id":"alpha","value":1}]}"#;
    let json_chunks = chunker::process_file_for_rag("package.json", json.as_bytes());
    let json_joined = json_chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(json_joined.contains("JSON path: $"));
    assert!(json_joined.contains("Language: json"));

    let toml = r#"[package]
name = "elephant-never-forgets"

[dependencies]
anyhow = "1"
"#;
    let toml_chunks = chunker::process_file_for_rag("Cargo.toml", toml.as_bytes());
    assert!(toml_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Section: package")));
    assert!(toml_chunks
        .iter()
        .any(|chunk| chunk.text.contains("Section: dependencies")));
}

#[test]
fn smart_chunker_dispatches_additional_code_families() {
    let cases = [
        (
            "src/Game.cs",
            r#"namespace App;
/// <summary>Player loop.</summary>
public class Player {
    public void Awake() {}
    public void Update() {}
}
"#,
            "Language: csharp",
            "Symbol: Player",
        ),
        (
            "src/engine.cpp",
            r#"template <typename T>
class Engine {
public:
  void start();
};

void tick() {}
"#,
            "Language: cpp",
            "Symbol: Engine",
        ),
        (
            "src/App.java",
            r#"package app;
public class App {
  public void run() {}
}
"#,
            "Language: java",
            "Symbol: App",
        ),
        (
            "src/App.kt",
            r#"data class User(val name: String)
fun render() {}
"#,
            "Language: kotlin",
            "Symbol: User",
        ),
        (
            "src/App.swift",
            r#"struct AppView {
  func render() {}
}
"#,
            "Language: swift",
            "Symbol: AppView",
        ),
        (
            "scripts/install.sh",
            r#"# Install
setup() {
  echo ok
}
"#,
            "Language: shell",
            "Symbol: Install",
        ),
        (
            "game/init.lua",
            r#"local function boot()
end
"#,
            "Language: lua",
            "Symbol: boot",
        ),
        (
            "shader/main.wgsl",
            r#"struct VertexOut {}
@vertex
fn vs_main() {}
"#,
            "Language: wgsl",
            "Symbol: VertexOut",
        ),
    ];

    for (path, source, language, symbol) in cases {
        let joined = chunker::process_file_for_rag(path, source.as_bytes())
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        assert!(joined.contains(language), "{path} missing {language}");
        assert!(joined.contains(symbol), "{path} missing {symbol}");
    }
}

#[test]
fn smart_chunker_dispatches_schema_markup_and_build_files() {
    let cases = [
        (
            "db/schema.sql",
            "CREATE TABLE notes (id integer);\nCREATE INDEX notes_id ON notes(id);\n",
            "Language: sql",
            "Symbol: create table notes",
        ),
        (
            "assets/icon.svg",
            r#"<svg><defs id="icons"></defs><g id="mark"><path d="M0 0" /></g></svg>"#,
            "Language: svg",
            "Symbol: svg",
        ),
        (
            "Dockerfile",
            "FROM rust:1 AS builder\nRUN cargo build\nFROM debian:stable\n",
            "Language: dockerfile",
            "Symbol: builder",
        ),
        (
            "Makefile",
            "build:\n\tcargo build\n\ntest:\n\tcargo test\n",
            "Language: makefile",
            "Symbol: build",
        ),
    ];

    for (path, source, language, symbol) in cases {
        let joined = chunker::process_file_for_rag(path, source.as_bytes())
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        assert!(joined.contains(language), "{path} missing {language}");
        assert!(joined.contains(symbol), "{path} missing {symbol}");
    }
}
