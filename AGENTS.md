# Agent Instructions

## Release Completion

When a change is intended to ship, do not stop after merging code to `main`.

- Bump the crate release version in `Cargo.toml` and the matching package entry in `Cargo.lock`.
- Update user-facing release/version references that should point at the new version, including README install examples and tests that assert default installer versions.
- Run the local validation gates before tagging:
  - `cargo fmt --all -- --check`
  - `cargo check --locked --no-default-features`
  - `cargo test --locked`
  - `cargo clippy --locked --all-targets --all-features -- -D warnings`
  - `cargo package --locked --allow-dirty --no-verify`
  - `cargo build --locked --release`
- Push the version-bump commit to `main`, then create and push the matching `vX.Y.Z` tag.
- Watch the tag-triggered GitHub release workflow until it completes successfully.
- Verify the release assets were built and published by CI for the supported binaries before calling the update finished.
- For Daisy installs, use the shipped updater after the release is available:
  `ssh daisy 'enf update && enf --version'`. Do not copy source, build a
  temporary binary on Daisy, or manually replace `/home/turnercore/.enf/bin/enf`
  unless the user explicitly asks for a one-off local build.
