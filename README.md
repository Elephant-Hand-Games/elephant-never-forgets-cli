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
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_VERSION=v1.2.5 sh
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

### Install

```sh
enf init
enf init --db=sqlite
enf init --db=false
enf init --dry-run
enf init --db --provider openai --model text-embedding-3-small --index
enf init --db --provider ollama --model nomic-embed-text
enf init --db --force --provider native
```

Plain `enf init` is equivalent to:

```sh
enf init --db=sqlite --provider native --model nomic-embed-text-v1.5 --variant quantized
```

Native projects install the active model profile during initialization. Use
`--db=false` only when `.enf/index.sqlite` already exists and you want init to
write/refresh config without creating the database.

### Model

```sh
enf models list
enf models current --json
enf models install
enf models install --dry-run
enf models cache-path
enf models gc
```

### Index

```sh
enf index .
enf add ./file.ehmeta
enf remove ./old-note.md
enf index . --dry-run
enf index ./docs --reembed --json
enf index ./notes --changed-only
enf index --no-embed ./docs
```

`index` embeds chunks for the active profile. For the native provider, indexing
fails if the active model profile is missing; run `enf models install` to repair
an older project. Use `--no-embed` only for metadata-only indexing. Indexed text
metadata includes relative path, file type/extension, size, modified time, hash,
full text when enabled, chunks, line ranges, token counts, and embeddings.
Image files are indexed when they match `[image.include]`; image embeddings are
only created when `[image.embedding].enabled = true`. `add` indexes an explicit
file or directory path only when it is included by the text or image include
configuration and not excluded. `remove` deletes a file path from the index
without deleting it from disk.

`enf` also reads `.enfignore` at the project root using gitignore-style patterns.
Place `.enfignoredir` inside a directory to skip that directory and all of its
subdirectories during discovery.

### Search

```sh
enf search "where are the agent editing rules?"
enf search "release notes" --mode hybrid --level chunk --limit 20
enf search "logo" --kind image --filetype png --path 'assets/**'
enf retrieve "debugging command output" --json
```

Plain text search output wraps matched file paths in local terminal hyperlinks
when your terminal supports OSC 8 links. Mixed results are labeled as `[text]`
or `[image]`; JSON results include `kind`, `file_type`, and optional
`rerank_score`.

Search filters:

- `--kind all|text|image`
- `--filetype <ext>` repeatable, with or without a leading dot
- `--path <glob>` repeatable

### Status / Doctor / CI

```sh
enf status
enf status --json
enf doctor
enf ci --install-models --json
enf ci --no-embed
```

## Provider Examples

Provider setup can be written during init, then overridden per indexing or
search command when needed.

- Native (Candle profile target):
  `enf init --db --provider native --model nomic-embed-text-v1.5 --variant quantized`
- Ollama:
  `enf init --db --provider ollama --model nomic-embed-text`
- OpenAI:
  `enf init --db --provider openai --model text-embedding-3-small`
- OpenAI-compatible:
  `enf init --db --provider openai-compatible --endpoint https://provider.example.com/v1/embeddings`
- Generic HTTP:
  `enf init --db --provider http --model custom --endpoint https://api.example.com/embeddings --dimensions 1024`

Optional endpoint-backed features are configured in `.enf.toml`:

- `[reranker] enabled = true` with `endpoint = "http://host:41803/rerank"`
  reranks filtered search candidates using a response array containing `index`
  and `score`.
- `[image.embedding] enabled = true` with `endpoint = "http://host:41802/embed"`
  embeds indexed images by sending raw base64 image strings. These vectors are
  stored for future compatible image search, but `search --kind image` currently
  uses path/metadata matching until a text-to-image query embedding endpoint is
  configured. The current compatible wrapper reports model
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
- `enf index --no-embed <path>` skips embedding and is intended only for
  metadata-only workflows.
- `enf search` and `enf retrieve` combine keyword matches with stored chunk vectors
  when embeddings exist, cache query embeddings by normalized query, and continue
  to return text-backed results when no chunk vectors are present.

For a complete CLI surface reference, see [`docs/command-reference.md`](./docs/command-reference.md).

## Release Process

- Run the release checks in [`docs/release-checklist.md`](./docs/release-checklist.md).

## Local Testing Ground

This repo includes a tiny sample project under [`testing-ground/`](./testing-ground/)
with a couple of directories, Markdown files, and one text file.

After you install `enf`, try:

```sh
cd /Users/turnercore/Projects/CLI/elephant-never-forgets-cli/testing-ground
enf init --db --model-cache project
enf status
enf index .
enf search "where are agent rules documented?"
enf search "strict offline cache" --mode keyword --cached-query-only
enf retrieve "provider override warning" --json
enf doctor
enf ci --no-embed
```
