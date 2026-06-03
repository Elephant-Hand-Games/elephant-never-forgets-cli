use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnfError {
    #[error("project is not initialized. Run: enf init --db")]
    NotInitialized,

    #[error("config already exists at {path}. Use --force to overwrite it.")]
    ConfigExists { path: String },

    #[error("native embedding model is not installed.\nRun:\n  enf models install")]
    NativeModelMissing,

    #[error("query embedding is not cached and --cached-query-only was set.")]
    QueryEmbeddingNotCached,

    #[error("could not reach provider at {endpoint}.\nUse:\n  enf doctor")]
    ProviderUnavailable { endpoint: String },

    #[error("{0}")]
    Message(String),
}
