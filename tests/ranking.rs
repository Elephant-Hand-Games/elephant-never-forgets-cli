use elephant_never_forgets::ranking::{self, RankedResult, ScoreWeights, VectorScoreMode};

struct RankedFixture {
    path: &'static str,
    kind: &'static str,
    score: f32,
    vector_score: f32,
    keyword_score: f32,
    metadata_score: f32,
    file_id: Option<i64>,
    chunk_id: Option<i64>,
    chunk_index: Option<i64>,
    start_line: Option<i64>,
    end_line: Option<i64>,
}

impl Default for RankedFixture {
    fn default() -> Self {
        Self {
            path: "result.md",
            kind: "chunk",
            score: 0.0,
            vector_score: 0.0,
            keyword_score: 0.0,
            metadata_score: 0.0,
            file_id: None,
            chunk_id: None,
            chunk_index: None,
            start_line: None,
            end_line: None,
        }
    }
}

fn ranked_result(fixture: RankedFixture) -> RankedResult {
    RankedResult {
        path: fixture.path.to_string(),
        snippet: format!("snippet for {}", fixture.path),
        score: fixture.score,
        kind: fixture.kind.to_string(),
        file_id: fixture.file_id,
        chunk_id: fixture.chunk_id,
        chunk_index: fixture.chunk_index,
        start_line: fixture.start_line,
        end_line: fixture.end_line,
        keyword_score: fixture.keyword_score,
        vector_score: fixture.vector_score,
        metadata_score: fixture.metadata_score,
        profile_hash: "profile".into(),
        mode: "hybrid".into(),
        level: "chunk".into(),
    }
}

#[test]
fn top_k_keeps_the_highest_scoring_results_in_order() {
    let results = vec![
        ranked_result(RankedFixture {
            path: "c.md",
            score: 0.7,
            vector_score: 0.7,
            file_id: Some(3),
            chunk_id: Some(13),
            chunk_index: Some(2),
            start_line: Some(21),
            end_line: Some(24),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "a.md",
            score: 0.9,
            vector_score: 0.9,
            file_id: Some(1),
            chunk_id: Some(11),
            chunk_index: Some(0),
            start_line: Some(1),
            end_line: Some(4),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "b.md",
            score: 0.8,
            vector_score: 0.8,
            file_id: Some(2),
            chunk_id: Some(12),
            chunk_index: Some(1),
            start_line: Some(11),
            end_line: Some(14),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "d.md",
            score: 0.1,
            vector_score: 0.1,
            file_id: Some(4),
            chunk_id: Some(14),
            chunk_index: Some(3),
            start_line: Some(31),
            end_line: Some(34),
            ..Default::default()
        }),
    ];

    let ranked = ranking::top_k(results, 2);
    let paths: Vec<_> = ranked.iter().map(|result| result.path.as_str()).collect();
    let scores: Vec<_> = ranked.iter().map(|result| result.score).collect();

    assert_eq!(paths, vec!["a.md", "b.md"]);
    assert_eq!(scores, vec![0.9, 0.8]);
}

#[test]
fn top_k_uses_deterministic_tie_ordering() {
    let results = vec![
        ranked_result(RankedFixture {
            path: "b.md",
            kind: "file",
            score: 0.5,
            vector_score: 0.5,
            keyword_score: 0.5,
            metadata_score: 0.5,
            file_id: Some(2),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "a.md",
            kind: "file",
            score: 0.5,
            vector_score: 0.5,
            keyword_score: 0.5,
            metadata_score: 0.5,
            file_id: Some(2),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "a.md",
            score: 0.5,
            vector_score: 0.5,
            keyword_score: 0.5,
            metadata_score: 0.5,
            file_id: Some(1),
            chunk_id: Some(20),
            chunk_index: Some(1),
            start_line: Some(10),
            end_line: Some(12),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "a.md",
            score: 0.5,
            vector_score: 0.5,
            keyword_score: 0.5,
            metadata_score: 0.5,
            file_id: Some(1),
            chunk_id: Some(10),
            chunk_index: Some(1),
            start_line: Some(10),
            end_line: Some(11),
            ..Default::default()
        }),
        ranked_result(RankedFixture {
            path: "a.md",
            score: 0.5,
            vector_score: 0.5,
            keyword_score: 0.5,
            metadata_score: 0.5,
            file_id: Some(1),
            chunk_id: Some(10),
            chunk_index: Some(0),
            start_line: Some(8),
            end_line: Some(9),
            ..Default::default()
        }),
    ];

    let ranked = ranking::top_k(results, 5);
    let actual: Vec<_> = ranked
        .iter()
        .map(|result| {
            (
                result.path.as_str(),
                result.kind.as_str(),
                result.file_id,
                result.chunk_id,
                result.chunk_index,
                result.start_line,
                result.end_line,
            )
        })
        .collect();

    assert_eq!(
        actual,
        vec![
            (
                "a.md",
                "chunk",
                Some(1),
                Some(10),
                Some(0),
                Some(8),
                Some(9)
            ),
            (
                "a.md",
                "chunk",
                Some(1),
                Some(10),
                Some(1),
                Some(10),
                Some(11)
            ),
            (
                "a.md",
                "chunk",
                Some(1),
                Some(20),
                Some(1),
                Some(10),
                Some(12)
            ),
            ("a.md", "file", Some(2), None, None, None, None),
            ("b.md", "file", Some(2), None, None, None, None),
        ]
    );
}

#[test]
fn vector_scoring_modes_match_expected_math() {
    let left = [1.0, 2.0, 3.0];
    let right = [4.0, 5.0, 6.0];

    assert_eq!(ranking::dot_product(&left, &right), 32.0);
    assert!((ranking::cosine_similarity(&left, &right) - 0.9746318).abs() < 1e-6);
    assert_eq!(
        ranking::vector_score(&left, &right, VectorScoreMode::Dot),
        32.0
    );
    assert!(
        (ranking::vector_score(&left, &right, VectorScoreMode::Cosine) - 0.9746318).abs() < 1e-6
    );

    let zero = [0.0, 0.0, 0.0];
    assert_eq!(ranking::cosine_similarity(&left, &zero), 0.0);
    assert_eq!(
        ranking::vector_score(&left, &zero, VectorScoreMode::Cosine),
        0.0
    );
    assert_eq!(ranking::dot_product(&left, &[1.0, 2.0]), 0.0);
    assert_eq!(ranking::cosine_similarity(&left, &[1.0, 2.0]), 0.0);
}

#[test]
fn hybrid_score_weights_and_clamps_components() {
    let score = ranking::hybrid_score(
        1.0,
        0.5,
        0.25,
        ScoreWeights {
            vector: 2.0,
            keyword: 1.0,
            metadata: 1.0,
        },
    );
    assert!((score - 0.6875).abs() < f32::EPSILON);

    assert_eq!(
        ranking::hybrid_score(
            0.9,
            0.9,
            0.9,
            ScoreWeights {
                vector: 0.0,
                keyword: 0.0,
                metadata: 0.0,
            },
        ),
        0.0
    );

    assert_eq!(
        ranking::hybrid_score(
            2.0,
            2.0,
            2.0,
            ScoreWeights {
                vector: 1.0,
                keyword: 1.0,
                metadata: 1.0,
            },
        ),
        1.0
    );
}
