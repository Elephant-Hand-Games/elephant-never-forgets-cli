# Command Reference

`enf` is organized around a small visible command tree:

```text
enf init [PATH]          Set up ENF in this project
enf setup               Change providers, presets, reranker, images, or search defaults
enf index [PATH]        Add or refresh files in the index
enf search <QUERY>      Human-friendly search with snippets
enf retrieve <QUERY>    RAG-friendly retrieval with stable JSON
enf status              Show whether the project is ready
enf doctor              Diagnose setup problems
enf config              Show, explain, edit, or validate .enf.toml
enf models              Manage local embedding model assets
enf update              Update ENF
```

Hidden compatibility aliases still work: `add`, `remove`, `ci`, `find`,
`models cache-path`, `models gc`, `--changed-only`, `--no-embed`, and
`--db=false`.

## `init`

```text
enf init [--preset <preset>] [--provider <provider>] [--model <model>]
         [--endpoint <url>] [--api-key-env <env-var>] [--dimensions <n>]
         [--chunking <smart|line-window|off>] [--model-cache <global|project>]
         [--index] [--install-models] [--force] [--no-db]
         [--interactive] [--no-input] [--yes] [--dry-run]
```

In a TTY, plain `enf init` guides setup. In non-TTY/CI contexts it never prompts;
use `--yes`, `--no-input`, and explicit flags.

Presets: `local`, `code`, `docs`, `ollama`, `openai`, `custom`, `keyword`.

Examples:

```sh
enf init
enf init --yes --preset local --index
enf init --no-input --preset keyword
enf init --preset openai --api-key-env OPENAI_API_KEY --index
```

## `setup`

```text
enf setup
enf setup presets [--json]
enf setup preset <preset> [--json]
enf setup use <preset> [--endpoint <url>] [--api-key-env <env>] [--model <model>] [--dimensions <n>] [--dry-run] [--json]
enf setup reranker [--endpoint <url>] [--model <model>] [--candidate-limit <n>] [--off] [--schema] [--dry-run] [--json]
enf setup images [--endpoint <url>] [--query-endpoint <url>] [--model <model>] [--dimensions <n>] [--off] [--dry-run] [--json]
enf setup fallback [--provider <provider>] [--endpoint <url>] [--api-key-env <env>] [--off] [--dry-run] [--json]
enf setup search [--mode <hybrid|vector|keyword>] [--level <chunk|file|both>] [--limit <n>] [--dry-run] [--json]
```

Use `setup` when changing provider/model choices or optional endpoint-backed
features. Reranker and visual image search require external HTTP services; ENF
does not ship those servers.

## `config`

```text
enf config path [--json]
enf config show [section] [--json]
enf config explain [key]
enf config validate [--json]
enf config edit
enf config set <key> <value> [--dry-run] [--json]
enf config diff --preset <preset>
enf config doctor [--json]
```

`config set` supports safe provider/search/reranker/image keys. For unsupported
keys, ENF prints the config path and directs you to edit TOML directly.

Precedence:

1. Command-line flags
2. Environment variables
3. `.enf.local.toml`
4. `.enf.toml`
5. ENF defaults

## `models`

```text
enf models list [--json]
enf models current [--json]
enf models install [model] [--provider <provider>] [--model <model>]
                   [--variant <quantized|full>] [--endpoint <url>]
                   [--api-key-env <env>] [--dimensions <n>] [--json] [--dry-run]
enf models path [--json]
enf models clean [--json] [--dry-run]
```

`list` describes available packaged/local model choices. `current` shows the
active project profile. `install` accepts the same provider/model override shape
as `index`, `search`, and `retrieve`; `gemma` resolves to the native Hugging Face
ID when `--provider native` is selected. `path` prints the resolved model cache
path.

## `index`

```text
enf index [path] [--reembed] [--changed] [--install-models]
         [--no-embeddings] [--provider <provider>] [--model <model>]
         [--endpoint <url>] [--api-key-env <env>] [--dimensions <n>]
         [--json] [--dry-run]
```

Examples:

```sh
enf index .
enf index docs --changed
enf index . --reembed
enf index . --no-embeddings
```

`--no-embeddings` indexes metadata/text without embedding vectors. Use it for
keyword-only workflows or quick CI checks.

## `search`

```text
enf search <query> [--mode <hybrid|vector|keyword>] [--level <chunk|file|both>]
                  [--kind <all|text|image>] [--filetype <ext>]... [--path <glob>]...
                  [--limit <n>] [--cached-query-only] [--compact] [--full]
                  [--explain] [--json] [--jsonl]
```

`search` is optimized for humans. Default output includes ranked paths, line
ranges, scores, snippets, and short reason hints. Use `--compact` for paths and
scores only, `--full` for full stored snippets, and `--explain` for score
components.

## `retrieve`

```text
enf retrieve <query> [same filters as search] [--jsonl]
```

`retrieve` is optimized for agents, scripts, and RAG. Default output is pretty
JSON with this stable shape:

```json
{
  "schema_version": "1",
  "query": "where is player inventory saved",
  "mode": "hybrid",
  "level": "chunk",
  "profile": {
    "provider": "native",
    "engine": "candle",
    "model": "nomic-embed-text-v1.5",
    "variant": "quantized",
    "dimensions": 768,
    "hash": "..."
  },
  "warnings": [],
  "results": [
    {
      "rank": 1,
      "path": "src/save.rs",
      "source_uri": "file:///project/src/save.rs#L10",
      "kind": "text",
      "level": "chunk",
      "start_line": 10,
      "end_line": 40,
      "text": "...",
      "snippet": "...",
      "score": 0.82,
      "scores": {
        "vector": 0.8,
        "keyword": 0.5,
        "metadata": 0.1,
        "rerank": null
      },
      "file_type": "rs"
    }
  ]
}
```

## `status` and `doctor`

```text
enf status [--json]
enf doctor [--ci] [--fix] [--yes] [--check <name>] [--no-embeddings]
           [--install-models] [--dry-run] [--json]
```

`status` answers “can I search successfully?” `doctor` answers “what is broken
and what exact command fixes it?” Use `doctor --ci` in automation.

## `update`

```text
enf update [--version <tag>] [--install-dir <path>] [--method <binary|cargo>]
           [--repo <owner/name>] [--dry-run]
```
