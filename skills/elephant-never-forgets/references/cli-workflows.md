# CLI Workflows

## Install And Update

Latest release:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | sh
export PATH="$HOME/.enf/bin:$PATH"
enf --version
```

Specific release:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_VERSION=v1.2.2 sh
```

Custom install directory:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_INSTALL_DIR="$HOME/.local/bin" sh
```

Source fallback:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_INSTALL_METHOD=cargo sh
```

Supported release binary targets:

- `aarch64-apple-darwin`
- `x86_64-unknown-linux-gnu`
- `x86_64-pc-windows-msvc`

## Project Setup

Plain default setup:

```sh
enf init
```

Equivalent default profile:

```sh
enf init --db=sqlite --provider native --model nomic-embed-text-v1.5 --variant quantized
```

Skip database creation only when the DB already exists:

```sh
enf init --db=false
```

Remote provider examples:

```sh
enf init --db --provider ollama --model nomic-embed-text
enf init --db --provider openai --model text-embedding-3-small
enf init --db --provider openai-compatible --endpoint https://provider.example.com/v1/embeddings
enf init --db --provider http --model custom --endpoint https://api.example.com/embeddings --dimensions 1024
```

## Indexing

Index current project with embeddings:

```sh
enf index .
```

Metadata-only indexing:

```sh
enf index --no-embed .
```

Add explicit paths:

```sh
enf add ./docs
enf add ./file.ehmeta
```

Remove an indexed path without deleting the file:

```sh
enf remove ./old-note.md
```

Native Candle indexing behavior:

- The model loads before embedding.
- On older non-AVX x86 Linux CPUs, native batches are reduced to keep progress visible and committed.
- If interrupted, rerun `enf index .`; already committed embeddings are skipped.

## Searching

Plain search:

```sh
enf search "how does doctor ci work?"
```

JSON retrieval:

```sh
enf retrieve "doctor ci" --json
```

Modes and levels:

```sh
enf search "release assets" --mode hybrid --level chunk --limit 20
enf search "release assets" --mode keyword
enf search "release assets" --level file
```

Search output uses local terminal hyperlinks for file paths when supported.

## Status And Troubleshooting

Basic checks:

```sh
enf status
enf doctor
```

JSON checks:

```sh
enf status --json
enf doctor --json
```

Common symptoms:

- `project is not initialized`: run `enf init` from the target directory.
- Native model missing: run `enf models install`, then rerun `enf index .`.
- `active profile embeddings: 0` after metadata exists: run `enf index .` without `--no-embed`.
- Slow indexing on older Linux: verify progress output shows chunk ranges; let it continue or rerun later to resume.
- Interrupted indexing: rerun `enf index .`.

Process checks on Linux:

```sh
ps -eo pid,etime,pcpu,pmem,stat,args | grep "[e]nf index"
enf status
```

## Release Verification

When validating a release:

```sh
curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | ENF_VERSION=vX.Y.Z sh
export PATH="$HOME/.enf/bin:$PATH"
enf --version
tmp="$(mktemp -d)"
cd "$tmp"
mkdir -p docs
printf "Doctor CI checks release assets and validates native embeddings.\n" > docs/doctor-ci.md
enf init
enf index .
enf search "doctor ci release assets"
```

Expected status after successful indexing:

- `native runtime available: true`
- `files` greater than `0`
- `chunks` greater than `0`
- `active profile embeddings` equals `chunks`
- `missing active profile embeddings: 0`
