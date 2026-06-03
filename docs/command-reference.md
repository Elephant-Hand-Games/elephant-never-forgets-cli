# Command Reference

Generated from `src/cli.rs`.

## Top level

```text
enf <command>
```

Commands:

- `init`
- `models`
- `index`
- `add`
- `remove`
- `search`
- `retrieve`
- `status`
- `doctor`
- `ci`

## `init`

Initialize project state and write `.enf.toml`.

```text
enf init [--db[=<sqlite|false>]] [--native-embed|--local-embed] [--provider <provider>] [--model <model>] [--variant <variant>] [--model-cache <scope>] [--install-models] [--index] [--force]
```

Flags:

- `--db[=<sqlite|false>]`
  - omitted or `--db` means `sqlite`
  - `--db=false` skips DB creation and requires the configured DB to already exist
- `--native-embed` (alias: `--local-embed`)
- `--provider <provider>` where `<provider>` is one of:
  - `native`
  - `ollama`
  - `openai`
  - `openai-compatible`
  - `http`
- `--model <string>`
- `--variant <variant>` where `<variant>` is `quantized` or `full`
- `--model-cache <scope>` where `<scope>` is `global` or `project`
- `--install-models`
- `--index`
- `--force`

Plain `enf init` creates the default native SQLite project:

```text
enf init --db=sqlite --provider native --model nomic-embed-text-v1.5 --variant quantized
```

Native init installs the active model profile.
Published Linux x64 release binaries are portable builds; use
`enf init --db --provider ollama --model nomic-embed-text` or another remote
provider there unless you build from source with native fastembed enabled on a
supported CPU.

## `models`

Manage the active embedding profile.

```text
enf models <subcommand>
```

Subcommands:

- `list`:
  ```text
  enf models list [--json]
  ```
- `current`:
  ```text
  enf models current [--json]
  ```
- `install`:
  ```text
  enf models install [model] [--variant <variant>] [--json]
  ```
  - `model` positional argument is optional
  - Captures the active profile hash and cache marker state for the resolved cache scope.
- `cache-path`:
  ```text
  enf models cache-path [--json]
  ```
- `gc`:
  ```text
  enf models gc [--json]
  ```

## `index`

Index files into the local SQLite database.

```text
enf index [path] [--reembed] [--changed-only] [--install-models] [--no-embed] [--provider <provider>] [--model <model>] [--variant <variant>] [--endpoint <url>] [--api-key-env <env-var>] [--dimensions <usize>] [--json]
```

Flags:

- `path` positional path (default: `.`)
- `--reembed`
- `--changed-only`
- `--install-models`
- `--no-embed`
- `--json`
- Provider overrides:
  - `--provider <provider>`
  - `--model <string>`
  - `--variant <variant>` (`quantized` | `full`)
  - `--endpoint <url>`
  - `--api-key-env <env-var>`
  - `--dimensions <usize>`

Notes:
- Native `init` installs the active model profile by default.
- `index` embeds chunks for the active profile and fails if the native model
  profile is missing.
- `--no-embed` intentionally skips embedding for metadata-only indexing.
- Explicit file paths are indexed even when their extension is not in the
  default include list, unless excluded.

## `add`

Index an explicit file or directory path.

```text
enf add <path> [same options as index]
```

`add` is an alias for explicit indexing. It is useful for custom file types such
as `.ehmeta`.

## `remove`

Remove an indexed file path from SQLite without deleting it from disk.

```text
enf remove <path>
```

## `search`

Run a query with default output (path list) or JSON output.

```text
enf search <query> [--mode <mode>] [--level <level>] [--limit <usize>] [--cached-query-only] [--provider <provider>] [--model <model>] [--variant <variant>] [--endpoint <url>] [--api-key-env <env-var>] [--dimensions <usize>] [--json]
```

`retrieve` uses the same options as `search` and always prints JSON result payload.

Flags:

- `query` (required positional query string)
- `--mode <mode>` where `<mode>` is `hybrid`, `vector`, or `keyword`
- `--level <level>` where `<level>` is `chunk`, `file`, or `both`
- `--limit <usize>`
- `--cached-query-only`
- `--json`
- Provider overrides: same as `index`

Notes:
- When stored chunk embeddings exist for the active profile, search/retrieve use
  hybrid ranking over vector similarity, keyword score, and metadata score.
- Query embeddings are cached by normalized query. `--cached-query-only` fails if
  the query vector is not already cached for the active profile.
- Plain text output wraps file paths in OSC 8 terminal hyperlinks when supported.

## `retrieve`

```text
enf retrieve <query> [same options as search]
```

Notes:

- At runtime `search` and `retrieve` both dispatch to the same internal search command,
  with `retrieve` enabling JSON output.

## `status`

```text
enf status [--json]
```

- `--json`

## `doctor`

```text
enf doctor [--json]
```

- `--json`

## `ci`

```text
enf ci [--no-embed] [--install-models] [--json]
```

- `--no-embed`
- `--install-models`
- `--json`
