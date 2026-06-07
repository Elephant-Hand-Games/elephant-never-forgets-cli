# Elephant Never Forgets 🐘

`enf` is a native-first Rust CLI for semantic and keyword search over local
directories, repositories, documentation sets, Obsidian vaults, and agent
workflow files.

## Overview

- `enf` stores configuration in `.enf.toml` and data in `.enf/index.sqlite`.
- It supports install/init, model management, indexing, search, retrieve, status,
  doctor, and CI verification workflows.
- Native provider defaults are `provider = "native"` + `engine = "candle"` in
  config, using `nomic-embed-text-v1.5` locally for embeddings.

## Installation

### Install from the latest release

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | sh
```

This downloads the newest GitHub Release archive for your platform, installs the
`enf` executable into `$HOME/.enf/bin`, and adds that directory to your shell
profile when it is not already on `PATH`. Restart your shell after install, or
run the `export PATH=...` command printed by the installer.

Install a specific release tag:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_VERSION=v2.0.2 sh
```

Install somewhere else:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_INSTALL_DIR=/usr/local/bin sh
```

Published binary targets:

- `aarch64-apple-darwin` for Apple Silicon Macs
- `x86_64-unknown-linux-gnu` for Linux x64, including Debian x64
- `x86_64-pc-windows-msvc` for Windows x64

The Linux x64 release binary includes the native Candle embedding runtime and is
intended to work on Debian x64 CPUs that do not support AVX.

If no release archive exists for your platform, the installer falls back to
installing from the Git repo with Cargo:

```sh
cargo install --git https://github.com/Elephant-Hand-Games/elephant-never-forgets-cli.git --locked
```

### From a local checkout while developing

```sh
cargo install --path . --locked
```

Homebrew support is practical now that release archives exist. It needs a tap
formula with the release URL and SHA-256 checksum; the release workflow in this
repo produces the archives and checksums that formula would use.

Update an existing install:

```sh
enf update
```

Preview an update without installing anything:

```sh
enf update --dry-run
```

## Command Workflows

### Set Up A Project

```sh
enf init
enf init --yes --preset local --index
enf init --no-input --preset keyword
enf init --dry-run
enf init --preset openai --api-key-env OPENAI_API_KEY --index
```

Plain `enf init` is equivalent to:

```sh
enf init --preset local
```

In a terminal, `enf init` guides setup with plain-English choices. In CI or any
non-TTY environment, it never prompts; use `--yes`, `--no-input`, and explicit
flags. Native projects install the active model profile during initialization.
Use `--no-db` only when `.enf/index.sqlite` already exists and you want init to
write/refresh config without creating the database. The legacy `--db=false`
spelling still works but is hidden from help.

### Change Setup

```sh
enf setup
enf setup presets
enf setup preset local
enf setup use local
enf setup use openai --api-key-env OPENAI_API_KEY
enf setup reranker --endpoint http://localhost:8080/rerank
enf setup images --endpoint http://localhost:8081/embed-images --query-endpoint http://localhost:8081/embed-query
```

Presets are first-class product modes: `local`, `code`, `docs`, `ollama`,
`openai`, `custom`, and `keyword`.

### Model

```sh
enf models list
enf models current --json
enf models install
enf models install --dry-run
enf models path
enf models clean
```

### Index

```sh
enf index .
enf index . --dry-run
enf index ./docs --reembed --json
enf index ./notes --changed
enf index ./docs --no-embeddings
enf remove ./old-note.md
```

`index` embeds chunks for the active profile. For the native provider, indexing
fails if the active model profile is missing; run `enf models install` to repair
an older project. Use `--no-embeddings` only for metadata-only indexing. Indexed
text metadata includes relative path, file type/extension, size, modified time,
hash, full text when enabled, chunks, line ranges, token counts, and embeddings.
Image files are indexed when they match `[image.include]`; image embeddings are
only created when `[image.embedding].enabled = true`. Hidden compatibility
aliases such as `add`, `remove`, and `ci` still work, but the v2 help keeps the
main surface focused. `remove` deletes a file path from the index without
deleting it from disk.

`enf` also reads `.enfignore` at the project root using gitignore-style patterns.
Place `.enfignoredir` inside a directory to skip that directory and all of its
subdirectories during discovery.

### Search

```sh
enf search "where are the agent editing rules?"
enf search "release notes" --mode hybrid --level chunk --limit 20
enf search "logo" --kind image --filetype png --path 'assets/**'
enf search "save bug" --explain
enf retrieve "debugging command output"
enf retrieve "debugging command output" --jsonl
```

Plain text search output is for humans: ranked paths, line ranges, scores,
snippets, and short “why” hints. `retrieve` is for agents and RAG: stable JSON
with `schema_version`, query metadata, active profile, warnings, source URIs,
result text/snippets, and score components. Human warnings go to stderr; JSON
stdout stays machine-readable.

Search filters:

- `--kind all|text|image`
- `--filetype <ext>` repeatable, with or without a leading dot
- `--path <glob>` repeatable
- `--compact`
- `--full`
- `--explain`
- `--jsonl`

### Status / Doctor / CI

```sh
enf status
enf status --json
enf doctor
enf doctor --ci --install-models --json
enf doctor --ci --no-embeddings
```

`status` answers whether the project is ready to search. `doctor` explains what
is broken and prints exact repair commands. The old `enf ci` command remains as
a hidden alias for CI checks.

### Config

```sh
enf config path
enf config show
enf config show embedding
enf config explain reranker.endpoint
enf config set search.limit 20
enf config validate
enf config edit
enf config diff --preset openai
```

`.enf.toml` is the shared project config. `.enf.local.toml` is ignored by
default and can hold machine-local endpoints, API env names, reranker settings,
and image embedding settings.

## Provider Examples

Provider setup can be written during init, then overridden per indexing or
search command when needed.

- Native: `enf setup use local`
- Ollama: `enf setup use ollama --endpoint http://localhost:11434/api/embed`
- OpenAI: `enf setup use openai --api-key-env OPENAI_API_KEY`
- OpenAI-compatible: `enf setup use custom --endpoint https://provider.example.com/v1/embeddings`
- Generic HTTP: `enf config set embedding.provider http`, then set model/endpoint/dimensions

Optional endpoint-backed features are configured in `.enf.toml`:

- `[reranker] enabled = true` with `endpoint = "http://host:41803/rerank"`
  reranks filtered search candidates using a response array containing `index`
  and `score`.
- `[image.embedding] enabled = true` with `endpoint = "http://host:41802/embed"`
  embeds indexed images by sending raw base64 image strings. Add
  `query-endpoint = "http://host:41802/v1/images/query_embeddings"` to embed
  text search queries into the same image vector space for semantic image
  search. The current compatible wrapper reports model
  `open_clip/ViT-H-14:laion2b_s32b_b79k`, `dimensions = 1024`, and an
  `embeddings` vector list.

Per-command overrides (for `index`, `search`, `retrieve`):

```sh
enf search "..." --provider openai --model text-embedding-3-small --json
enf index . --provider native --model nomic-embed-text-v1.5
```

Notes:

- The OpenAI variant auto-maps `nomic-embed-text-v1.5` to `text-embedding-3-small`
  unless you explicitly set a supported model.
- `openai` sets `OPENAI_API_KEY` as the default environment variable key if
  a per-command `--api-key-env` override is not provided.
- `--model-cache project` stores marker state in `.enf/models`, while `--model-cache
global` uses the environment cache directory.

## Naming

- Product: Elephant Never Forgets
- Crate/package: `elephant-never-forgets`
- Executable command: `enf`

## Agent Skill

Agents can install the bundled Codex skill for working with this CLI:

```sh
/skills install https://github.com/Elephant-Hand-Games/elephant-never-forgets-cli/tree/main/skills/elephant-never-forgets --now
```

Use the skill when an agent needs to install, initialize, index, search, debug,
or release-test `enf`.

## Default Stack

- Rust CLI
- Native embedding provider by default
- Native profile default engine name: `candle`
- `nomic-embed-text-v1.5` quantized profile
- SQLite persistent store
- FTS5 keyword index
- Search output includes stored path/snippet payloads with configurable modes/limits
- Query embedding cache for repeated search/retrieve workflows
- Chunk embedding persistence for the active embedding profile
- Chunk-level and file-level indexing
- JSON output for automation and agent workflows

Optional providers include Ollama, OpenAI, OpenAI-compatible services, and
custom HTTP embedding endpoints.

### Native Candle Status

`enf` ships with a native Candle profile path (`provider = "native"` and
`engine = "candle"`).

- `enf models install` currently prepares and records the active model profile,
  then writes/updates the cache marker under the resolved model cache path.
- `enf index <path>` embeds missing chunks for the active profile.
- Published Linux x64 release binaries include the native Candle runtime.
- `enf index --no-embeddings <path>` skips embedding and is intended only for
  metadata-only workflows.
- `enf search` and `enf retrieve` combine keyword matches with stored chunk vectors
  when embeddings exist, cache query embeddings by normalized query, and continue
  to return text-backed results when no chunk vectors are present.

For a complete CLI surface reference, see [`docs/command-reference.md`](./docs/command-reference.md).

## Release Process

- Run the release checks in [`docs/release-checklist.md`](./docs/release-checklist.md).
