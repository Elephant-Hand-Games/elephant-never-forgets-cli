# Release Checklist

Use this checklist for packaging-ready releases.

## Pre-release checks

- [ ] Verify branch clean and release commit contains only intended changes.
- [ ] Confirm `README.md`, `docs/command-reference.md`, and packaging docs reflect the
  shipped command behavior.
- [ ] Bump version and release notes content as required.

## Code quality

- [ ] Formatting:
  - `cargo fmt --all --check`
- [ ] Lints:
  - `cargo clippy --all-targets --all-features -- -D warnings`

## Test matrix

- [ ] Core test suite:
  - `cargo test --all`
- [ ] Provider mock tests:
  - Add/refresh mock coverage for provider selection and override precedence
    (`native`, `ollama`, `openai`, `openai-compatible`, `http`)
  - Validate error messages for unimplemented providers and fallback paths
- [ ] SQLite migration tests:
  - Regression test for `db::migrate` idempotence
  - Migration can re-run against an existing schema without duplicate state
  - New fields/tables are present after migration
- [ ] CLI argument coverage:
  - Ensure docs and tests cover `enf init`, `models`, `index`, `search`,
    `retrieve`, `status`, `doctor`, and `ci` flag combinations

## Packaging

- [ ] Build and package:
  - `cargo package`
- [ ] Validate package artifact:
  - Confirm `Cargo.toml` metadata, readme linkage, and included files are correct
  - Run `cargo package --list` and confirm docs and runtime files are present
- [ ] Final verification:
  - Confirm `enf --help` reflects all documented commands and flags
  - Confirm `enf init --db` + `enf status` end-to-end path still matches docs

## Embedding pipeline release checks

- [ ] Confirm docs state native fastembed as implemented and remove obsolete
  "not implemented" language.
- [ ] Confirm `enf models install` is documented as recording active profile/cache
  marker state for the selected cache scope.
- [ ] Confirm indexing persists chunk embeddings after active model installation.
- [ ] Confirm search/retrieve use cached query embeddings and stored chunk vectors
  for hybrid ranking, with text-backed fallback when vectors are absent.

## Post-release

- [ ] Tag and publish release artifacts
- [ ] Update installation notes and known limitations (including provider runtime state)
