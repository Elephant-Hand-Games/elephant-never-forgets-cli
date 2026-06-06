# Release Checklist

Use this checklist for packaging-ready releases.

## Pre-release checks

- [ ] Verify branch clean and release commit contains only intended changes.
- [ ] Confirm `README.md`, `docs/command-reference.md`, and packaging docs reflect the
  shipped command behavior.
- [ ] Bump `Cargo.toml`, `Cargo.lock`, README install examples, and tests that
  assert installer versions.

## Code quality

- [ ] Formatting:
  - `cargo fmt --all -- --check`
- [ ] Lints:
  - `cargo clippy --locked --all-targets --all-features -- -D warnings`

## Test matrix

- [ ] Core test suite:
  - `cargo test --locked`
- [ ] Portable Linux profile:
  - `cargo check --locked --no-default-features`
- [ ] Provider mock tests:
  - Add/refresh mock coverage for provider selection and override precedence
    (`native`, `ollama`, `openai`, `openai-compatible`, `http`)
  - Validate error messages for unimplemented providers and fallback paths
- [ ] SQLite migration tests:
  - Regression test for `db::migrate` idempotence
  - Migration can re-run against an existing schema without duplicate state
  - New fields/tables are present after migration
- [ ] CLI argument coverage:
  - Ensure docs and tests cover `enf init`, `setup`, `config`, `models`,
    `index`, `search`, `retrieve`, `status`, and `doctor --ci` flag combinations

## Packaging

- [ ] Build and package:
  - `cargo package --locked --allow-dirty --no-verify`
  - `cargo build --locked --release`
- [ ] Build release archives through GitHub Actions tag workflow:
  - `git tag vX.Y.Z && git push origin vX.Y.Z`
  - Confirm assets exist for Apple Silicon macOS, Linux x64, and Windows x64
- [ ] Validate package artifact:
  - Confirm `Cargo.toml` metadata, readme linkage, and included files are correct
  - Run `cargo package --list` and confirm docs and runtime files are present
- [ ] Validate installer:
  - `scripts/install.sh` installs a release archive on supported platforms
  - Cargo fallback works when no release archive is available
- [ ] Optional Homebrew tap:
  - Create/update formula from release URL and SHA-256 checksum
  - Test with `brew install --build-from-source ./Formula/enf.rb`
- [ ] Final verification:
  - Confirm `enf --help` reflects all documented commands and flags
  - Confirm `enf init --db` + `enf status` end-to-end path still matches docs
  - Confirm the Linux x64 release asset includes native Candle embeddings and
    starts on Debian-class x64 machines.

## Embedding pipeline release checks

- [ ] Confirm docs state native Candle embeddings as implemented and remove obsolete
  "not implemented" language.
- [ ] Confirm `enf models install` is documented as recording active profile/cache
  marker state for the selected cache scope.
- [ ] Confirm indexing persists chunk embeddings after active model installation.
- [ ] Confirm search/retrieve use cached query embeddings and stored chunk vectors
  for hybrid ranking, with text-backed fallback when vectors are absent.

## Post-release

- [ ] Tag and publish release artifacts
- [ ] Update installation notes and known limitations (including provider runtime state)
