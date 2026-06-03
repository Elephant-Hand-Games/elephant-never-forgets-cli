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
- `search`
- `retrieve`
- `status`
- `doctor`
- `ci`

## `init`

Initialize project state and write `.enf.toml`.

```text
enf init [--db] [--native-embed|--local-embed] [--provider <provider>] [--model <model>] [--variant <variant>] [--model-cache <scope>] [--install-models] [--index] [--force]
```

Flags:

- `--db` (required by current implementation)
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
enf index [path] [--reembed] [--changed-only] [--install-models] [--provider <provider>] [--model <model>] [--variant <variant>] [--endpoint <url>] [--api-key-env <env-var>] [--dimensions <usize>] [--json]
```

Flags:

- `path` positional path (default: `.`)
- `--reembed`
- `--changed-only`
- `--install-models`
- `--json`
- Provider overrides:
  - `--provider <provider>`
  - `--model <string>`
  - `--variant <variant>` (`quantized` | `full`)
  - `--endpoint <url>`
  - `--api-key-env <env-var>`
  - `--dimensions <usize>`

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
