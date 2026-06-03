# Elephant Never Forgets

`enf` is a native-first Rust CLI for semantic and keyword search over local
directories, repositories, documentation sets, Obsidian vaults, and agent
workflow files.

## Overview

- `enf` stores configuration in `.enf.toml` and data in `.enf/index.sqlite`.
- It supports install/init, model management, indexing, search, retrieve, status,
  doctor, and CI verification workflows.
- Native provider defaults are `provider = "native"` + `engine = "fastembed"` in
  config, and native fastembed is an available runtime path for embeddings.

Use `cargo install` or `cargo install --path .` from this repo for your preferred
distribution method.

## Command Workflows

### Install

```sh
enf init --db
enf init --db --provider openai --model text-embedding-3-small --index
enf init --db --force --provider native --install-models
```

`init` must currently include `--db` and can initialize with optional provider/model
overrides.

### Model

```sh
enf models list
enf models current --json
enf models install
enf models cache-path
enf models gc
```

### Index

```sh
enf index .
enf index ./docs --reembed --json
enf index ./notes --changed-only --install-models
```

### Search

```sh
enf search "where are the agent editing rules?"
enf search "release notes" --mode hybrid --level chunk --limit 20
enf retrieve "debugging command output" --json
```

### Status / Doctor / CI

```sh
enf status
enf status --json
enf doctor
enf ci --install-models --json
enf ci --no-embed
```

## Provider Examples

Example provider setup is done at init, or overridden per indexing/searching
command.

- Native (fastembed profile target):
  `enf init --db --provider native --model nomic-embed-text-v1.5 --variant quantized`
- Ollama:
  `enf init --db --provider ollama --model nomic-embed-text`
- OpenAI:
  `enf init --db --provider openai --model text-embedding-3-small`
- OpenAI-compatible:
  `enf init --db --provider openai-compatible --endpoint https://provider.example.com/v1/embeddings`
- Generic HTTP:
  `enf init --db --provider http --model custom --endpoint https://api.example.com/embeddings --dimensions 1024`

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

## Default Stack

- Rust CLI
- Native embedding provider by default (profile target)
- Native profile default engine name: `fastembed`
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

### Native fastembed status

`enf` now ships with a native fastembed profile path (`provider = "native"` and
`engine = "fastembed"`).

- `enf models install` currently prepares and records the active model profile,
  then writes/updates the cache marker under the resolved model cache path.
- `enf index --install-models <path>` records the active profile and embeds missing
  chunks for that profile.
- `enf search` and `enf retrieve` combine keyword matches with stored chunk vectors
  when embeddings exist, cache query embeddings by normalized query, and continue
  to return text-backed results when no chunk vectors are present.

For a complete CLI surface reference, see [`docs/command-reference.md`](./docs/command-reference.md).

## Release Process

- Run the release checks in [`docs/release-checklist.md`](./docs/release-checklist.md).
