# Status Doctor CI

`enf status` should summarize the project state without loading embedding models.

`enf doctor` should validate configuration, SQLite readability, FTS5 availability,
model marker state, provider readiness, and missing embeddings.

`enf ci --no-embed` is intended for fast automation that must not download models.
`enf ci --install-models` is explicit opt-in behavior for environments that are
allowed to prepare native model state.

