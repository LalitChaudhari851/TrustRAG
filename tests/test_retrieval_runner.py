"""
Unit tests for the retrieval evaluation runner using mocked search/reranking outputs.
Ensures zero network/API dependencies while testing report generation,
metric aggregation, and pipeline comparisons with hand-calculated expected values.
"""

import pytest
from unittest.mock import MagicMock
from evals.retrieval_dataset_schema import RetrievalEvalRecord
from evals.scripts.run_retrieval_evaluation import (
    evaluate_pipeline_on_query,
    run_experiment_suite,
    generate_markdown_report,
)

UUID_A = "83358e9c-431c-51b4-97e8-baacb2d11344"
UUID_B = "001e31bd-c9c0-5b6c-90ee-0fdd9aab9c3a"
UUID_DIST_1 = "11111111-1111-1111-1111-111111111111"
UUID_DIST_2 = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def sample_records():
    # 1 Answerable Single-Hop
    r1 = RetrievalEvalRecord(
        id="q1",
        query="How do you start Redis?",
        query_type="answerable_single_hop",
        difficulty="easy",
        domain="k8s",
        source_files=["parallel_work_queue.txt"],
        expected_sections=["STEP 1"],
        evidence_quotes=["kubectl apply redis"],
        candidate_chunk_ids=[UUID_A],
        relevance_grades={UUID_A: 2},
        sufficient_evidence_sets=[[UUID_A]],
        verification_status="unverified_synthetic",
        split="dev",
        expected_answer="Apply redis yaml",
    )
    # 1 Answerable Multi-Hop
    r2 = RetrievalEvalRecord(
        id="q2",
        query="How do workers and jobs interact?",
        query_type="answerable_multi_hop",
        difficulty="hard",
        domain="k8s",
        source_files=["parallel_work_queue.txt"],
        expected_sections=["STEP 1", "STEP 2"],
        evidence_quotes=["rediswq", "parallelism 2"],
        candidate_chunk_ids=[UUID_A, UUID_B],
        relevance_grades={UUID_A: 2, UUID_B: 2},
        sufficient_evidence_sets=[[UUID_A, UUID_B]],
        verification_status="unverified_synthetic",
        split="dev",
        expected_answer="Workers read queue and job configures parallelism",
    )
    # 1 Unanswerable Query
    r3 = RetrievalEvalRecord(
        id="q3",
        query="How to configure Kafka?",
        query_type="unanswerable",
        difficulty="medium",
        domain="k8s",
        source_files=[],
        expected_sections=[],
        evidence_quotes=[],
        candidate_chunk_ids=[],
        relevance_grades={},
        sufficient_evidence_sets=[],
        verification_status="unverified_synthetic",
        split="dev",
        unanswerable_subtype="corpus_unanswerable",
        expected_answer="Cannot answer",
    )
    return [r1, r2, r3]


class TestRetrievalRunnerMocked:
    """Tests the runner execution logic with mocked search and reranking."""

    def test_evaluate_pipeline_single_query_answerable(self, sample_records):
        rec = sample_records[0]  # q1, expects UUID_A

        def mock_search(query, limit, hybrid, query_filter=None):
            # Returns [UUID_DIST_1, UUID_A, UUID_DIST_2]
            return [
                {"id": UUID_DIST_1, "score": 0.80, "content": "distractor text"},
                {"id": UUID_A, "score": 0.75, "content": "redis text"},
                {"id": UUID_DIST_2, "score": 0.70, "content": "another distractor"},
            ]

        res = evaluate_pipeline_on_query(
            record=rec,
            mode_name="hybrid_rrf_raw",
            hybrid=True,
            apply_reranker=False,
            query_filter=None,
            search_fn=mock_search,
        )

        assert res["query_id"] == "q1"
        assert res["is_unanswerable"] is False
        assert res["retrieved_count"] == 3
        # In [DIST_1, UUID_A, DIST_2]:
        # UUID_A is rank 2 -> MRR@5 = 1 / 2 = 0.50
        assert res["mrr@5"] == 0.50
        assert res["recall@5"] == 1.0
        # hits = 1 in 3 returned:
        # Precision@5 fixed = 1 / 5 = 0.20
        # Precision@5 actual = 1 / 3 ≈ 0.3333
        assert res["precision@5_fixed"] == 0.20
        assert pytest.approx(res["precision@5_actual"], 1e-4) == 1.0 / 3.0
        # Distractors before first relevant = 1
        assert res["distractor_info@5"]["distractors_before_first_relevant"] == 1

    def test_evaluate_pipeline_single_query_unanswerable(self, sample_records):
        rec_unans = sample_records[2]  # q3, unanswerable

        # Mock reranker returns empty (below threshold) -> correct abstention
        def mock_search(query, limit, hybrid, query_filter=None):
            return [{"id": UUID_DIST_1, "score": 0.05, "content": "noise"}]

        def mock_rerank(query, documents, top_n, min_score, relative_margin):
            return []  # Dropped because top score < min_score

        res = evaluate_pipeline_on_query(
            record=rec_unans,
            mode_name="hybrid_rrf_reranked",
            hybrid=True,
            apply_reranker=True,
            query_filter=None,
            search_fn=mock_search,
            rerank_fn=mock_rerank,
        )

        assert res["query_id"] == "q3"
        assert res["is_unanswerable"] is True
        assert res["retrieval_abstention_decision"] is True
        assert res["abstention_correct"] is True
        # Relevance metrics should not be computed for unanswerables
        assert "recall@5" not in res

    def test_run_experiment_suite_and_report_generation(self, sample_records, tmp_path):
        # Save sample records to temp file
        temp_dataset = str(tmp_path / "temp_eval_dataset.json")
        from evals.retrieval_dataset_schema import save_dataset
        save_dataset(sample_records, temp_dataset)

        # Mock search and rerank
        def mock_search(query, limit, hybrid, query_filter=None):
            return [
                {"id": UUID_A, "score": 0.85, "content": "chunk A"},
                {"id": UUID_B, "score": 0.80, "content": "chunk B"},
            ]

        def mock_rerank(query, documents, top_n, min_score, relative_margin):
            return [dict(d, rerank_score=0.90) for d in documents[:top_n]]

        # Mock Qdrant Client
        mock_client = MagicMock()
        mock_col_info = MagicMock()
        mock_col_info.points_count = 197
        mock_client.get_collection.return_value = mock_col_info
        mock_client.count.return_value.count = 197
        # retrieve returns both UUIDs
        pt_a = MagicMock()
        pt_a.id = UUID_A
        pt_b = MagicMock()
        pt_b.id = UUID_B
        mock_client.retrieve.return_value = [pt_a, pt_b]

        suite = run_experiment_suite(
            dataset_path=temp_dataset,
            include_dense_baseline=True,
            search_fn=mock_search,
            rerank_fn=mock_rerank,
            client=mock_client,
        )

        assert "full_corpus" in suite["experiments"]
        assert "true_only_corpus" in suite["experiments"]
        assert "hybrid_rrf_raw" in suite["experiments"]["full_corpus"]["pipelines"]
        assert "hybrid_rrf_reranked" in suite["experiments"]["full_corpus"]["pipelines"]
        assert "dense_only_raw" in suite["experiments"]["full_corpus"]["pipelines"]

        # Check macro averages: answerable count = 2
        pipe_summary = suite["experiments"]["full_corpus"]["pipelines"]["hybrid_rrf_raw"]["summary"]
        assert pipe_summary["counts"]["answerable_queries"] == 2
        assert pipe_summary["counts"]["unanswerable_queries"] == 1

        # Check Markdown Report Generation
        md_report = generate_markdown_report(suite)
        assert "# TrustRAG Retrieval Evaluation Report" in md_report
        assert "UNVERIFIED SEED DATASET NOTICE" in md_report
        assert "Relevance Macro-Averages" in md_report
        assert "Multi-Hop Conjunction Metrics" in md_report
        assert "Retrieval Abstention on Unanswerable Queries" in md_report
