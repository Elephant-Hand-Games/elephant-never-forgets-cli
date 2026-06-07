use tempfile::tempdir;

use elephant_never_forgets::{
    config::{Config, ModelCache},
    embed::{
        active_profile, deserialize_vector, profile_hash, serialize_vector,
        EMBEDDING_SERIALIZATION_VERSION,
    },
    models::{cache_path_for, install_active_model_in, is_active_model_installed_in},
};

#[test]
fn profile_hash_is_deterministic_and_uses_profile_fields() {
    let profile = active_profile(&Config::default());
    let baseline = profile_hash(&profile);

    let mut provider_variant = profile.clone();
    provider_variant.provider = "openai-compatible".into();
    provider_variant.endpoint = Some("https://alt.example.invalid/v1/embeddings".into());
    assert_eq!(baseline, profile_hash(&provider_variant));

    let mut prefix_variant = profile.clone();
    prefix_variant.document_prefix = "search_document: alt ".into();
    assert_ne!(baseline, profile_hash(&prefix_variant));

    let mut dimension_variant = profile.clone();
    dimension_variant.dimensions = 1024;
    assert_ne!(baseline, profile_hash(&dimension_variant));

    let mut serialization_variant = profile.clone();
    serialization_variant.serialization_version = "f32-le-v2".into();
    assert_ne!(baseline, profile_hash(&serialization_variant));

    assert_eq!(baseline, profile_hash(&profile));
    assert_eq!(
        profile.serialization_version,
        EMBEDDING_SERIALIZATION_VERSION
    );
}

#[test]
fn cache_path_differs_between_project_and_global_modes() {
    let cwd = tempdir().unwrap();
    let global_cache = tempdir().unwrap();

    let mut config = Config::default();
    config.state.model_cache = ModelCache::Project;
    assert_eq!(
        cache_path_for(&config, cwd.path(), Some(global_cache.path())),
        cwd.path().join(".enf/models")
    );

    config.state.model_cache = ModelCache::Global;
    assert_eq!(
        cache_path_for(&config, cwd.path(), Some(global_cache.path())),
        global_cache.path().join("enf/models")
    );
}

#[test]
fn install_persists_marker_and_sqlite_status_for_active_profile() {
    let workspace = tempdir().unwrap();
    let global_cache = tempdir().unwrap();

    let mut config = Config::default();
    config.state.model_cache = ModelCache::Project;

    std::env::set_var("ENF_SKIP_NATIVE_MODEL_LOAD", "1");
    install_active_model_in(&config, workspace.path(), Some(global_cache.path())).unwrap();

    let profile = active_profile(&config);
    let cache_path = cache_path_for(&config, workspace.path(), Some(global_cache.path()));
    let marker_path = cache_path.join(format!("{}.json", profile.profile_hash));

    assert!(marker_path.exists());
    assert!(
        is_active_model_installed_in(&config, workspace.path(), Some(global_cache.path())).unwrap()
    );

    let marker = std::fs::read_to_string(&marker_path).unwrap();
    assert!(marker.contains(&profile.profile_hash));
    assert!(marker.contains("\"status\": \"installed\""));

    let conn = rusqlite::Connection::open(workspace.path().join(&config.state.db_path)).unwrap();
    let status: String = conn
        .query_row(
            "SELECT status FROM model_cache WHERE provider = ?1 AND ifnull(engine, '') = ?2 AND model = ?3 AND ifnull(variant, '') = ?4 AND dimensions = ?5 AND cache_path = ?6",
            rusqlite::params![
                profile.provider.as_str(),
                profile.engine.as_deref().unwrap_or(""),
                profile.model.as_str(),
                profile.variant.as_deref().unwrap_or(""),
                profile.dimensions as i64,
                cache_path.display().to_string(),
            ],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(status, "installed");
}

#[test]
fn query_vector_round_trips_and_rejects_bad_lengths() {
    let vector = vec![0.0, -0.5, 1.25, 42.125];
    let encoded = serialize_vector(&vector);
    let decoded = deserialize_vector(&encoded).unwrap();
    assert_eq!(decoded, vector);

    let mut shortened = encoded.clone();
    shortened.pop();
    assert!(deserialize_vector(&shortened).is_err());
}
