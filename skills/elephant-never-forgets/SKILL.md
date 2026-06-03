---
name: elephant-never-forgets
description: Use when Codex needs to install, initialize, configure, index, search, troubleshoot, release-test, or document the Elephant Never Forgets `enf` CLI for local semantic search over directories, repos, docs, Obsidian vaults, `.ehmeta` files, or agent workflow files.
---

# Elephant Never Forgets

Use this skill when working with the `enf` CLI or with a project that has `.enf.toml` and `.enf/index.sqlite`.

## Core Workflow

1. Check whether `enf` is installed:
   ```sh
   enf --version
   ```
2. If missing or stale, install/update from the latest release:
   ```sh
   curl -fsSL https://raw.githubusercontent.com/Elephant-Hand-Games/elephant-never-forgets-cli/main/scripts/install.sh | sh
   export PATH="$HOME/.enf/bin:$PATH"
   ```
3. Initialize a project from the directory to be searched:
   ```sh
   enf init
   ```
4. Index content:
   ```sh
   enf index .
   ```
5. Search:
   ```sh
   enf search "query text"
   enf retrieve "query text" --json
   ```
6. Diagnose problems:
   ```sh
   enf status
   enf doctor
   ```

## Defaults To Preserve

- Product name: Elephant Never Forgets.
- Command: `enf`.
- Default project config: `.enf.toml`.
- Default database: `.enf/index.sqlite`.
- Default provider: `native`.
- Default engine: `candle`.
- Default model: `nomic-embed-text-v1.5`.
- Default variant: `quantized`.
- Default dimensions: `768`.
- Default prefixes:
  - documents: `search_document: `
  - queries: `search_query: `
- Native indexing may be slow on older non-AVX x86 Linux CPUs; progress output and committed embeddings should still advance.

## Operational Rules

- Prefer release binary install via `scripts/install.sh` for normal users.
- Use `ENF_VERSION=vX.Y.Z` only when testing or pinning a known release.
- Use `ENF_INSTALL_METHOD=cargo` only as a fallback or for source builds.
- Do not tell users to run `cargo install --path .` unless they are developing from a local checkout.
- If `enf index .` appears slow, check `enf status` from another shell and inspect process CPU usage before assuming it is frozen.
- If indexing was interrupted, rerun `enf index .`; embeddings are profile-scoped and committed incrementally.
- Use `enf add <path>` for explicit files or non-default extensions such as `.ehmeta`.
- Use `enf remove <path>` to remove a path from the index without deleting it from disk.

## References

For exact commands, install options, troubleshooting checks, and release-test notes, read [references/cli-workflows.md](references/cli-workflows.md).
