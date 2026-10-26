"""
Unit tests for TrustRAG retrieval metrics engine and dataset schema validator.
Includes hand-calculated proofs with manual arithmetic for all IR metrics.
"""

import math
import pytest
from evals.retrieval_metrics import (
    recall_at_k,
    precision_at_k_fixed,
    precision_at_k_actual,
    hit_rate_at_k,
    mrr_at_k,
    dcg_at_k,
    ndcg_at_k,
    multi_hop_evidence_coverage,
    all_required_evidence_coverage,
    multi_hop_success,
    distractor_interference_score,
    retrieval_abstention_signal,
    bootstrap_confidence_interval,
    evaluate_single_query,
    aggregate_retrieval_metrics,
)
from evals.retrieval_dataset_schema import (
    RetrievalEvalRecord,
    validate_eval_record,
    validate_dataset_records,
)


class TestHandCalculatedMetrics:
    """
    Hand-calculated unit test proofs for individual IR metrics.
    """

    def test_recall_at_k_hand_calculated(self):
        # Relevant set has 4 documents: {A, B, C, D}
        # Retrieved @ 3: [A, B, X]
        # Hand calc: hits = 2 (A and B). Total relevant = 4. Recall@3 = 2 / 4 = 0.50
        rel = {"A", "B", "C", "D"}
        ret = ["A", "B", "X"]
        assert recall_at_k(ret, rel, k=3) == 0.50

        # At k=2: ret[:2] is ['A', 'B'] -> hits = 2. Recall@2 = 2 / 4 = 0.50
        assert recall_at_k(ret, rel, k=2) == 0.50

        # At k=1: ret[:1] is ['A'] -> hits = 1. Recall@1 = 1 / 4 = 0.25
        assert recall_at_k(ret, rel, k=1) == 0.25

        # With distractor first: ['X', 'A', 'Y'] at k=2 -> hits = 1. Recall@2 = 1 / 4 = 0.25
        assert recall_at_k(["X", "A", "Y"], rel, k=2) == 0.25

        # Empty relevant -> 0.0
        assert recall_at_k(ret, set(), k=3) == 0.0

        # Empty retrieved -> 0.0
        assert recall_at_k([], rel, k=5) == 0.0

    def test_precision_fixed_vs_actual_hand_calculated(self):
        # Relevant: {A, B}
        # FlashRank reranker dropped low-scoring candidates and returned only 2 docs: [A, X]
        # Evaluated at K = 5:
        # Hand calc:
        #   Precision@5_fixed = hits / 5 = 1 / 5 = 0.20
        #   Precision@5_actual = hits / len(retrieved) = 1 / 2 = 0.50
        rel = {"A", "B"}
        ret = ["A", "X"]
        assert precision_at_k_fixed(ret, rel, k=5) == 0.20
        assert precision_at_k_actual(ret, rel, k=5) == 0.50

        # Both returned are relevant: [A, B]
        #   Precision@5_fixed = 2 / 5 = 0.40
        #   Precision@5_actual = 2 / 2 = 1.00
        ret_perfect = ["A", "B"]
        assert precision_at_k_fixed(ret_perfect, rel, k=5) == 0.40
        assert precision_at_k_actual(ret_perfect, rel, k=5) == 1.00

        # Empty retrieved:
        assert precision_at_k_fixed([], rel, k=5) == 0.0
        assert precision_at_k_actual([], rel, k=5) == 0.0

    def test_hit_rate_hand_calculated(self):
        rel = {"A"}
        assert hit_rate_at_k(["X", "Y", "A"], rel, k=3) == 1.0
        assert hit_rate_at_k(["X", "Y", "A"], rel, k=2) == 0.0
        assert hit_rate_at_k([], rel, k=5) == 0.0

    def test_mrr_hand_calculated(self):
        rel = {"TARGET"}
        # Rank 1 -> 1 / 1 = 1.0
        assert mrr_at_k(["TARGET", "X", "Y"], rel, k=3) == 1.0
        # Rank 2 -> 1 / 2 = 0.5
        assert mrr_at_k(["X", "TARGET", "Y"], rel, k=3) == 0.5
        # Rank 3 -> 1 / 3 = 0.3333333333333333
        assert pytest.approx(mrr_at_k(["X", "Y", "TARGET"], rel, k=3), 1e-6) == 1.0 / 3.0
        # Rank 4 (outside K=3) -> 0.0
        assert mrr_at_k(["X", "Y", "Z", "TARGET"], rel, k=3) == 0.0

    def test_duplicate_retrieved_handling(self):
        # Retrieved list contains duplicates: ["A", "A", "B", "A"]
        # Deduplication preserving order yields: ["A", "B"]
        # Relevant: {A, B}
        rel = {"A", "B"}
        ret_dupes = ["A", "A", "B", "A"]
        # Top 2 after dedup has 2 hits
        assert recall_at_k(ret_dupes, rel, k=2) == 1.0
        assert precision_at_k_fixed(ret_dupes, rel, k=2) == 1.0
        assert precision_at_k_actual(ret_dupes, rel, k=2) == 1.0

    def test_graded_ndcg_hand_calculated(self):
        """
        Hand-calculated proof of Graded nDCG@3:
        Gain formula: (2^rel - 1) / log2(i + 1)
        
        Given relevance grades:
          Chunk A: rel = 2 (Primary)   -> gain = 2^2 - 1 = 3
          Chunk B: rel = 1 (Supporting)-> gain = 2^1 - 1 = 1
          Chunk C: rel = 0 (Irrelevant)-> gain = 2^0 - 1 = 0
          
        Ideal ranking at K=3: [A, B, C]
          Rank 1: gain=3 / log2(2) = 3 / 1.0 = 3.0
          Rank 2: gain=1 / log2(3) = 1 / 1.584962500721156 = 0.6309297535714574
          Rank 3: gain=0 / log2(4) = 0.0
          IDCG@3 = 3.0 + 0.6309297535714574 = 3.6309297535714574
        """
        grades = {"A": 2, "B": 1, "C": 0}
        ideal_order = ["A", "B", "C"]
        expected_idcg = 3.0 + (1.0 / math.log2(3))
        
        calc_idcg = dcg_at_k(ideal_order, grades, k=3)
        assert pytest.approx(calc_idcg, 1e-6) == expected_idcg
        assert ndcg_at_k(ideal_order, grades, k=3) == 1.0

        # Sub-optimal ranking: [B, A, C] (supporting chunk placed before primary chunk)
        #   Rank 1: doc B (rel=1) -> gain=1 / log2(2) = 1.0
        #   Rank 2: doc A (rel=2) -> gain=3 / log2(3) = 3 / 1.584962500721156 = 1.8927892607143721
        #   Rank 3: doc C (rel=0) -> 0.0
        #   Actual DCG@3 = 1.0 + 1.8927892607143721 = 2.8927892607143721
        #   Expected nDCG@3 = 2.8927892607143721 / 3.6309297535714574 ≈ 0.796707
        subopt_order = ["B", "A", "C"]
        calc_subopt_dcg = dcg_at_k(subopt_order, grades, k=3)
        expected_subopt_dcg = 1.0 + (3.0 / math.log2(3))
        assert pytest.approx(calc_subopt_dcg, 1e-6) == expected_subopt_dcg

        expected_ndcg = expected_subopt_dcg / expected_idcg
        assert pytest.approx(ndcg_at_k(subopt_order, grades, k=3), 1e-6) == expected_ndcg

        # Worst ranking: [C, B, A]
        #   Rank 1: doc C (rel=0) -> 0.0
        #   Rank 2: doc B (rel=1) -> 1.0 / log2(3) ≈ 0.630930
        #   Rank 3: doc A (rel=2) -> 3.0 / log2(4) = 3.0 / 2.0 = 1.50
        #   DCG@3 = 0.630930 + 1.50 = 2.130930
        worst_order = ["C", "B", "A"]
        expected_worst_dcg = (1.0 / math.log2(3)) + (3.0 / 2.0)
        assert pytest.approx(dcg_at_k(worst_order, grades, k=3), 1e-6) == expected_worst_dcg
        assert pytest.approx(ndcg_at_k(worst_order, grades, k=3), 1e-6) == expected_worst_dcg / expected_idcg


class TestMultiHopAndDistractors:
    """
    Tests for multi-hop evidence conjunction, multiple sufficient sets, and distractor interference.
    """

    def test_multi_hop_multiple_sufficient_sets(self):
        # A query can be answered either by:
        # Set 1: {Chunk_Redis_Setup, Chunk_Redis_Cli}
        # OR Set 2: {Chunk_Redis_Setup, Chunk_Redis_Config}
        sufficient_sets = [
            ["CHUNK_REDIS_SETUP", "CHUNK_REDIS_CLI"],
            ["CHUNK_REDIS_SETUP", "CHUNK_REDIS_CONFIG"],
        ]

        # Case A: Retrieved Set 2 completely [CHUNK_REDIS_SETUP, CHUNK_REDIS_CONFIG, OTHER]
        ret_a = ["CHUNK_REDIS_SETUP", "CHUNK_REDIS_CONFIG", "OTHER"]
        assert multi_hop_evidence_coverage(ret_a, sufficient_sets, k=3) == 1.0
        assert all_required_evidence_coverage(ret_a, sufficient_sets, k=3) == 1.0
        assert multi_hop_success(ret_a, sufficient_sets, k=3) == 1.0

        # Case B: Partial retrieval - only CHUNK_REDIS_SETUP retrieved at K=2
        # Set 1 has 1/2 coverage (0.5), Set 2 has 1/2 coverage (0.5) -> Best coverage = 0.5
        ret_b = ["CHUNK_REDIS_SETUP", "OTHER"]
        assert multi_hop_evidence_coverage(ret_b, sufficient_sets, k=2) == 0.5
        assert all_required_evidence_coverage(ret_b, sufficient_sets, k=2) == 0.0
        assert multi_hop_success(ret_b, sufficient_sets, k=2) == 0.0

        # Case C: Neither required chunk retrieved
        ret_c = ["OTHER_1", "OTHER_2"]
        assert multi_hop_evidence_coverage(ret_c, sufficient_sets, k=2) == 0.0
        assert all_required_evidence_coverage(ret_c, sufficient_sets, k=2) == 0.0
        assert multi_hop_success(ret_c, sufficient_sets, k=2) == 0.0

    def test_distractor_interference_score(self):
        # Relevant: {"REL_1"}
        # Retrieved: ["DIST_1", "DIST_2", "REL_1", "DIST_3"]
        rel = {"REL_1"}
        ret = ["DIST_1", "DIST_2", "REL_1", "DIST_3"]

        # At K=3: ["DIST_1", "DIST_2", "REL_1"]
        # Distractors before first relevant: 2
        # Distractor ratio: 2 / 3 = 0.666667
        # First relevant rank: 3
        d_info = distractor_interference_score(ret, rel, k=3)
        assert d_info["distractors_before_first_relevant"] == 2
        assert pytest.approx(d_info["distractor_ratio_in_top_k"], 1e-4) == 2.0 / 3.0
        assert d_info["first_relevant_rank"] == 3
        assert d_info["total_distractors"] == 2
        assert d_info["total_retrieved"] == 3

        # At K=2: ["DIST_1", "DIST_2"] -> No relevant retrieved: distractors_before is None (undefined)
        d_info_2 = distractor_interference_score(ret, rel, k=2)
        assert d_info_2["first_relevant_rank"] is None
        assert d_info_2["distractors_before_first_relevant"] is None
        assert d_info_2["distractor_ratio_in_top_k"] == 1.0

        # Unanswerable case: rel is empty -> distractors_before is None (undefined)
        d_unans = distractor_interference_score(ret, set(), k=3)
        assert d_unans["first_relevant_rank"] is None
        assert d_unans["distractors_before_first_relevant"] is None
        assert d_unans["distractor_ratio_in_top_k"] == 1.0


class TestAbstentionAndMacroIsolation:
    """
    Tests separation of retrieval abstention signals and unanswerable query isolation.
    """

    def test_retrieval_abstention_signals(self):
        # 1. Empty retrieval -> abstains
        sig_empty = retrieval_abstention_signal([])
        assert sig_empty["should_abstain_at_retrieval"] is True
        assert sig_empty["reason"] == "empty_retrieval"

        # 2. Score below floor (0.08) -> abstains
        below_floor_docs = [{"id": "uuid1", "rerank_score": 0.04}]
        sig_low = retrieval_abstention_signal(below_floor_docs, score_floor=0.08)
        assert sig_low["should_abstain_at_retrieval"] is True
        assert sig_low["reason"] == "below_score_floor"

        # 3. Score above floor -> proceeds
        good_docs = [{"id": "uuid1", "rerank_score": 0.85}]
        sig_good = retrieval_abstention_signal(good_docs, score_floor=0.08)
        assert sig_good["should_abstain_at_retrieval"] is False
        assert sig_good["reason"] == "proceed"

    def test_unanswerable_isolation_in_aggregate(self):
        # 2 answerable queries + 1 unanswerable query
        # Query 1: Perfect retrieval (Recall=1.0, P=1.0)
        q1 = evaluate_single_query(
            query_id="q1",
            query_type="answerable_single_hop",
            retrieved_docs=[{"id": "doc1", "rerank_score": 0.9}],
            candidate_chunk_ids=["doc1"],
            relevance_grades={"doc1": 2},
            sufficient_evidence_sets=[["doc1"]],
        )

        # Query 2: Missed retrieval (Recall=0.0, P=0.0)
        q2 = evaluate_single_query(
            query_id="q2",
            query_type="answerable_single_hop",
            retrieved_docs=[{"id": "doc_noise", "rerank_score": 0.7}],
            candidate_chunk_ids=["doc2"],
            relevance_grades={"doc2": 2},
            sufficient_evidence_sets=[["doc2"]],
        )

        # Query 3: Unanswerable query. Retrieval correctly abstained (empty results)
        q3 = evaluate_single_query(
            query_id="q3",
            query_type="unanswerable",
            retrieved_docs=[],
            candidate_chunk_ids=[],
            relevance_grades={},
            sufficient_evidence_sets=[],
        )

        agg = aggregate_retrieval_metrics([q1, q2, q3], n_resamples=100)

        # Answerable counts: 2 total answerable
        assert agg["counts"]["answerable_queries"] == 2
        assert agg["counts"]["unanswerable_queries"] == 1

        # Macro average for Recall@5 must be over q1 and q2 ONLY: (1.0 + 0.0) / 2 = 0.50
        # If q3 had been included as 0, average would have been skewed to 1/3 = 0.333!
        recall_mean = agg["relevance_macro_averages"]["recall@5"]["mean"]
        assert pytest.approx(recall_mean, 1e-4) == 0.50

        # Unanswerable abstention is reported separately:
        assert agg["unanswerable_abstention"]["unanswerable_count"] == 1
        assert agg["unanswerable_abstention"]["correct_retrieval_abstentions"] == 1
        assert agg["unanswerable_abstention"]["retrieval_abstention_rate"] == 1.0


class TestDatasetSchemaValidation:
    """
    Tests dataset schema validator constraints and error detection.
    """

    def test_valid_answerable_record(self):
        rec = RetrievalEvalRecord(
            id="seed_001",
            query="How do you deploy Redis for a parallel work queue?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1: STARTING REDIS"],
            evidence_quotes=["kubectl apply -f https://k8s.io/examples/application/job/redis/redis-pod.yaml"],
            candidate_chunk_ids=["83358e9c-431c-51b4-97e8-baacb2d11344"],
            relevance_grades={"83358e9c-431c-51b4-97e8-baacb2d11344": 2},
            sufficient_evidence_sets=[["83358e9c-431c-51b4-97e8-baacb2d11344"]],
            verification_status="unverified_synthetic",
            split="dev",
            expected_answer="Apply the redis-pod.yaml and redis-service.yaml manifests.",
        )
        errors = validate_eval_record(rec)
        assert errors == []

    def test_invalid_uuid_detected(self):
        rec = RetrievalEvalRecord(
            id="seed_002",
            query="Valid query text here?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["OVERVIEW"],
            evidence_quotes=["Some quote"],
            candidate_chunk_ids=["not-a-valid-uuid"],
            relevance_grades={"not-a-valid-uuid": 2},
            sufficient_evidence_sets=[["not-a-valid-uuid"]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec)
        assert any("not a valid UUID" in e for e in errors)

    def test_unanswerable_query_constraints(self):
        # Unanswerable query that erroneously contains candidate chunks
        rec_bad = RetrievalEvalRecord(
            id="seed_003",
            query="How to deploy Apache Cassandra on EKS?",
            query_type="unanswerable",
            difficulty="medium",
            domain="kubernetes",
            source_files=[],
            expected_sections=[],
            evidence_quotes=[],
            candidate_chunk_ids=["83358e9c-431c-51b4-97e8-baacb2d11344"],
            relevance_grades={},
            sufficient_evidence_sets=[["83358e9c-431c-51b4-97e8-baacb2d11344"]],
            verification_status="unverified_synthetic",
            split="test",
        )
        errors = validate_eval_record(rec_bad)
        assert any("unanswerable query must have empty 'candidate_chunk_ids'" in e for e in errors)

    def test_conjunction_integrity_violation(self):
        # A chunk in sufficient_evidence_sets that does not exist in candidate_chunk_ids
        rec_bad = RetrievalEvalRecord(
            id="seed_004",
            query="Valid multi hop query?",
            query_type="answerable_multi_hop",
            difficulty="hard",
            domain="kubernetes",
            source_files=["file.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["Quote 1", "Quote 2"],
            candidate_chunk_ids=["83358e9c-431c-51b4-97e8-baacb2d11344"],
            relevance_grades={"83358e9c-431c-51b4-97e8-baacb2d11344": 2},
            sufficient_evidence_sets=[["83358e9c-431c-51b4-97e8-baacb2d11344", "00000000-0000-0000-0000-000000000000"]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec_bad)
        assert any("missing from candidate_chunk_ids" in e for e in errors)

    def test_duplicate_query_detection(self):
        rec1 = {
            "id": "rec_01",
            "query": "What is the replica count?",
            "query_type": "answerable_single_hop",
            "difficulty": "easy",
            "domain": "k8s",
            "source_files": ["f.txt"],
            "expected_sections": ["s"],
            "evidence_quotes": ["q"],
            "candidate_chunk_ids": ["83358e9c-431c-51b4-97e8-baacb2d11344"],
            "relevance_grades": {"83358e9c-431c-51b4-97e8-baacb2d11344": 2},
            "sufficient_evidence_sets": [["83358e9c-431c-51b4-97e8-baacb2d11344"]],
            "verification_status": "unverified_synthetic",
            "split": "dev",
        }
        rec2 = dict(rec1, id="rec_02", query="What is the replica count? ")  # duplicate with whitespace
        val = validate_dataset_records([rec1, rec2])
        assert val["is_valid"] is False
        assert len(val["duplicates"]) == 1
        assert val["duplicates"][0]["duplicate_id"] == "rec_02"
