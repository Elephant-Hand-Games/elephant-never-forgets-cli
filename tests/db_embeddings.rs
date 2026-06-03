use elephant_never_forgets::{config::Config, db, embed};
use rusqlite::params;

#[test]
fn profile_scoped_chunk_embeddings_round_trip() {
    let temp = tempfile::tempdir().unwrap();
    let conn = db::open_or_create(&temp.path().join(".enf/index.sqlite")).unwrap();
    let profile = embed::active_profile(&Config::default());
    let profile_id = db::upsert_embedding_profile(&conn, &profile).unwrap();

    conn.execute(
        "INSERT INTO files(path, hash, size_bytes, indexed_at, content)
         VALUES ('docs/one.md', 'file-hash', 12, datetime('now'), 'alpha beta')",
        [],
    )
    .unwrap();
    let file_id = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, hash, text, start_line, end_line, token_count)
         VALUES (?1, 0, 'chunk-hash', 'alpha beta', 1, 2, 2)",
        [file_id],
    )
    .unwrap();
    let chunk_id = conn.last_insert_rowid();

    let missing = db::chunks_missing_embeddings(&conn, profile_id, 10).unwrap();
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].chunk_id, chunk_id);
    assert_eq!(missing[0].text, "alpha beta");

    let vector = vec![0.25, 0.5, 0.75];
    db::upsert_chunk_embedding(&conn, profile_id, chunk_id, &vector).unwrap();
    assert!(db::chunks_missing_embeddings(&conn, profile_id, 10)
        .unwrap()
        .is_empty());

    let rows = db::vector_chunks_for_profile(&conn, profile_id, 10, 0).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].chunk_id, chunk_id);
    assert_eq!(rows[0].file_id, file_id);
    assert_eq!(rows[0].path, "docs/one.md");
    assert_eq!(rows[0].chunk_index, 0);
    assert_eq!(rows[0].start_line, Some(1));
    assert_eq!(rows[0].end_line, Some(2));
    assert_eq!(rows[0].text, "alpha beta");
    assert_eq!(rows[0].vector, vector);
}

#[test]
fn query_embeddings_are_profile_scoped_and_update_last_used() {
    let temp = tempfile::tempdir().unwrap();
    let conn = db::open_or_create(&temp.path().join(".enf/index.sqlite")).unwrap();
    let mut config = Config::default();
    let first_profile = embed::active_profile(&config);
    config.embedding.document_prefix = "alternate document: ".into();
    let second_profile = embed::active_profile(&config);
    let first_profile_id = db::upsert_embedding_profile(&conn, &first_profile).unwrap();
    let second_profile_id = db::upsert_embedding_profile(&conn, &second_profile).unwrap();

    db::upsert_query_embedding(&conn, first_profile_id, "agent rules", &[1.0, 0.0]).unwrap();
    db::upsert_query_embedding(&conn, second_profile_id, "agent rules", &[0.0, 1.0]).unwrap();
    db::upsert_query_embedding(&conn, first_profile_id, "agent rules", &[0.5, 0.5]).unwrap();

    assert_eq!(
        db::query_embedding(&conn, first_profile_id, "agent rules").unwrap(),
        Some(vec![0.5, 0.5])
    );
    assert_eq!(
        db::query_embedding(&conn, second_profile_id, "agent rules").unwrap(),
        Some(vec![0.0, 1.0])
    );
    assert_eq!(
        db::query_embedding(&conn, first_profile_id, "missing").unwrap(),
        None
    );

    let query_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM query_embeddings WHERE normalized_query = ?1",
            params!["agent rules"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(query_count, 2);
}

#[test]
fn active_profile_streams_only_its_vectors_and_preserves_chunk_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let conn = db::open_or_create(&temp.path().join(".enf/index.sqlite")).unwrap();
    let config = Config::default();
    let active_profile = embed::active_profile(&config);
    let active_profile_id = db::upsert_active_embedding_profile(&conn, &config).unwrap();

    conn.execute(
        "INSERT INTO files(id, path, hash, size_bytes, indexed_at, content)
         VALUES (?1, ?2, ?3, ?4, datetime('now'), ?5)",
        params![
            1,
            "docs/stream.md",
            "file-hash",
            12,
            Some("alpha beta gamma")
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, hash, text, start_line, end_line, token_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![10, 1, 0, "chunk-a", "alpha beta", 1, 2, 2],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, hash, text, start_line, end_line, token_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![11, 1, 1, "chunk-b", "gamma delta", 3, 4, 2],
    )
    .unwrap();

    let mut alternate_profile = active_profile.clone();
    alternate_profile.model = "alternate-embed-model".into();
    alternate_profile.profile_hash = embed::profile_hash(&alternate_profile);
    let alternate_profile_id = db::upsert_embedding_profile(&conn, &alternate_profile).unwrap();

    db::upsert_chunk_embedding(&conn, active_profile_id, 10, &[0.1, 0.2]).unwrap();
    db::upsert_chunk_embedding(&conn, active_profile_id, 11, &[0.3, 0.4]).unwrap();
    db::upsert_chunk_embedding(&conn, alternate_profile_id, 10, &[9.9, 9.8]).unwrap();
    db::upsert_query_embedding(&conn, active_profile_id, "alpha beta", &[1.5, 2.5]).unwrap();

    let mut batches = Vec::new();
    db::stream_vector_chunks_for_active_profile(&conn, &config, 1, |batch| {
        batches.push(batch.to_vec());
        Ok(())
    })
    .unwrap();

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0][0].chunk_id, 10);
    assert_eq!(batches[0][0].file_id, 1);
    assert_eq!(batches[0][0].path, "docs/stream.md");
    assert_eq!(batches[0][0].chunk_index, 0);
    assert_eq!(batches[0][0].start_line, Some(1));
    assert_eq!(batches[0][0].end_line, Some(2));
    assert_eq!(batches[0][0].text, "alpha beta");
    assert_eq!(batches[0][0].vector, vec![0.1, 0.2]);
    assert_eq!(batches[1][0].chunk_id, 11);
    assert_eq!(batches[1][0].vector, vec![0.3, 0.4]);
    assert!(batches
        .iter()
        .flatten()
        .all(|row| row.vector != vec![9.9, 9.8]));

    let cached = db::query_embedding(&conn, active_profile_id, "alpha beta").unwrap();
    assert_eq!(cached, Some(vec![1.5, 2.5]));
}
