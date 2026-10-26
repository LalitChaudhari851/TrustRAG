"""
Unit tests for TrustRAG retrieval evaluation dataset schema, integrity rules,
and verification workflows.
"""

import pytest
from evals.retrieval_dataset_schema import (
    RetrievalEvalRecord,
    validate_eval_record,
    validate_dataset_records,
    load_dataset,
)

VALID_UUID_1 = "83358e9c-431c-51b4-97e8-baacb2d11344"
VALID_UUID_2 = "001e31bd-c9c0-5b6c-90ee-0fdd9aab9c3a"


class TestDatasetSchemaIntegrity:
    """Tests schema validation, malformed evidence, and invalid IDs."""

    def test_malformed_evidence_empty_quotes_rejected(self):
        rec = RetrievalEvalRecord(
            id="test_01",
            query="How does Redis work queue handle parallel tasks?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=[],  # Malformed: empty quotes
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec)
        assert any("at least one evidence quote" in e for e in errors)

    def test_malformed_evidence_empty_candidate_chunks_rejected(self):
        rec = RetrievalEvalRecord(
            id="test_02",
            query="How does Redis work queue handle parallel tasks?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[],  # Malformed: empty chunks for answerable query
            relevance_grades={},
            sufficient_evidence_sets=[],
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec)
        assert any("specify at least one relevant chunk ID" in e for e in errors)

    def test_invalid_uuid_format_rejected(self):
        rec = RetrievalEvalRecord(
            id="test_03",
            query="Valid query text here?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=["not-a-valid-uuid-12345"],
            relevance_grades={"not-a-valid-uuid-12345": 2},
            sufficient_evidence_sets=[["not-a-valid-uuid-12345"]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec)
        assert any("not a valid UUID format" in e for e in errors)

    def test_conjunction_integrity_missing_candidate_id(self):
        # sufficient set contains a chunk that is not in candidate_chunk_ids
        rec = RetrievalEvalRecord(
            id="test_04",
            query="Valid query text here?",
            query_type="answerable_multi_hop",
            difficulty="hard",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["quote 1", "quote 2"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1, VALID_UUID_2]],  # VALID_UUID_2 missing from candidates
            verification_status="unverified_synthetic",
            split="dev",
        )
        errors = validate_eval_record(rec)
        assert any("missing from candidate_chunk_ids" in e for e in errors)


class TestDuplicateAndLeakageRules:
    """Tests duplicate question detection and train/test split leakage."""

    def test_duplicate_question_detection(self):
        rec1 = RetrievalEvalRecord(
            id="dup_01",
            query="What is the default backoffLimit in Kubernetes?",
            query_type="answerable_single_hop",
            difficulty="medium",
            domain="kubernetes",
            source_files=["cronjobs.docx"],
            expected_sections=["SEC"],
            evidence_quotes=["backoffLimit: 6"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        rec2 = RetrievalEvalRecord(
            id="dup_02",
            query="  what is the default backofflimit in kubernetes?  ",  # duplicate with whitespace/case
            query_type="answerable_single_hop",
            difficulty="medium",
            domain="kubernetes",
            source_files=["cronjobs.docx"],
            expected_sections=["SEC"],
            evidence_quotes=["backoffLimit: 6"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        val = validate_dataset_records([rec1, rec2])
        assert val["is_valid"] is False
        assert len(val["duplicates"]) == 1
        assert val["duplicates"][0]["duplicate_id"] == "dup_02"

    def test_split_leakage_detection(self):
        # Exact same query in both dev and test
        rec_dev = RetrievalEvalRecord(
            id="leak_dev",
            query="How do you start Redis for a work queue?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        rec_test = RetrievalEvalRecord(
            id="leak_test",
            query="How do you start Redis for a work queue?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_2],
            relevance_grades={VALID_UUID_2: 2},
            sufficient_evidence_sets=[[VALID_UUID_2]],
            verification_status="unverified_synthetic",
            split="test",
        )
        val = validate_dataset_records([rec_dev, rec_test])
        assert val["is_valid"] is False
        assert len(val["split_leakage"]) == 1
        assert val["split_leakage"][0]["type"] == "verbatim_query_leakage"
        assert val["split_leakage"][0]["dev_id"] == "leak_dev"
        assert val["split_leakage"][0]["test_id"] == "leak_test"


class TestVerificationWorkflow:
    """Tests human verification transitions, audit trail recording, and rejection."""

    def test_unverified_cannot_be_marked_verified_without_audit(self):
        # Directly claiming verified_human without reviewer_decision must fail validation
        rec = RetrievalEvalRecord(
            id="audit_01",
            query="Valid query text here?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="verified_human",  # Set to verified_human directly
            split="dev",
            reviewer_decision=None,  # Missing audit record!
        )
        errors = validate_eval_record(rec)
        assert any("cannot be 'verified_human' without a valid 'reviewer_decision'" in e for e in errors)

    def test_promotion_workflow_records_reviewer_and_reason(self):
        rec = RetrievalEvalRecord(
            id="audit_02",
            query="Valid query text here?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        assert rec.verification_status == "unverified_synthetic"

        # Promoting without reviewer name must raise ValueError
        with pytest.raises(ValueError):
            rec.promote_to_verified(reviewer="", reason="Looks good")

        # Promoting without reason must raise ValueError
        with pytest.raises(ValueError):
            rec.promote_to_verified(reviewer="Alice", reason="")

        # Valid promotion
        rec.promote_to_verified(reviewer="Alice", reason="Verified quote against doc chunk 3")
        assert rec.verification_status == "verified_human"
        assert rec.reviewer_decision["reviewer"] == "Alice"
        assert rec.reviewer_decision["reason"] == "Verified quote against doc chunk 3"
        assert "timestamp" in rec.reviewer_decision

        # Now validation passes
        assert validate_eval_record(rec) == []

    def test_rejection_workflow(self):
        rec = RetrievalEvalRecord(
            id="audit_03",
            query="Ambiguous question?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["SEC"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        rec.reject_record(reviewer="Bob", reason="Question is ambiguous and does not match chunk.")
        assert rec.verification_status == "rejected"
        assert rec.reviewer_decision["decision"] == "rejected"
        assert validate_eval_record(rec) == []


class TestGeneratedSeedDatasetIntegrity:
    """Validates the generated evals/data/retrieval_golden_seed_v1.json file."""

    def test_seed_dataset_file_conforms_to_all_invariants(self):
        records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
        assert len(records) == 50

        # Invariant 1: Records are either unverified_synthetic or verified_human with reviewer audit
        for r in records:
            assert r.verification_status in ("unverified_synthetic", "verified_human")
            if r.verification_status == "unverified_synthetic":
                assert r.reviewer_decision is None
            else:
                assert r.reviewer_decision is not None
                assert r.reviewer_decision.get("reviewer")
                assert r.reviewer_decision.get("reason")

        # Invariant 2: Split distribution
        dev_records = [r for r in records if r.split == "dev"]
        test_records = [r for r in records if r.split == "test"]
        assert len(dev_records) == 30
        assert len(test_records) == 20

        # Invariant 3: Question type breakdown
        single_hop = [r for r in records if r.query_type == "answerable_single_hop"]
        multi_hop = [r for r in records if r.query_type == "answerable_multi_hop"]
        unanswerable = [r for r in records if r.query_type == "unanswerable"]
        assert len(single_hop) == 30
        assert len(multi_hop) == 10
        assert len(unanswerable) == 10

        # Invariant 4: Unanswerable subtypes explicitly distinguished
        corpus_unanswerable = [r for r in unanswerable if r.unanswerable_subtype == "corpus_unanswerable"]
        unresolved = [r for r in unanswerable if r.unanswerable_subtype == "unresolved_evidence"]
        assert len(corpus_unanswerable) == 6
        assert len(unresolved) == 4

        # Invariant 5: Multi-hop alternative sufficient sets
        multi_hop_with_alts = [r for r in multi_hop if len(r.sufficient_evidence_sets) > 1]
        assert len(multi_hop_with_alts) >= 4

        # Invariant 6: Zero duplicates and zero split leakage
        val_res = validate_dataset_records(records)
        assert val_res["is_valid"] is True
        assert len(val_res["duplicates"]) == 0
        assert len(val_res["split_leakage"]) == 0
        assert len(val_res["errors"]) == 0


class TestBenchmarkFreezeAndHumanReviewWorkflowV2:
    """Comprehensive offline tests for Golden Dataset v2 human review and benchmark freeze."""

    def test_annotation_event_history_integrity(self):
        rec = RetrievalEvalRecord(
            id="v2_test_01",
            query="How does Redis work queue handle parallel tasks?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        assert len(rec.review_history) == 0

        # Promote to verified
        rec.promote_to_verified(
            reviewer="reviewer_carol@trustrag.io",
            reason="Verified exact quote against parallel_work_queue.txt chunk 3.",
            answerability="answerable",
            evidence_judgments={VALID_UUID_1: {"grade": 2, "support": "supported"}},
            answer_verified=True,
            multi_hop_verified=True,
        )
        assert rec.verification_status == "verified_human"
        assert len(rec.review_history) == 1
        ev = rec.review_history[0]
        assert ev["reviewer"] == "reviewer_carol@trustrag.io"
        assert ev["decision"] == "verified"
        assert ev["answerability"] == "answerable"
        assert ev["evidence_judgments"][VALID_UUID_1]["support"] == "supported"
        assert "timestamp" in ev
        assert "event_id" in ev

    def test_invalid_approval_attempts_missing_fields(self):
        rec = RetrievalEvalRecord(
            id="v2_test_02",
            query="How does Redis work queue handle parallel tasks?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        with pytest.raises(ValueError, match="Reviewer identity must be specified"):
            rec.promote_to_verified(reviewer="", reason="Looks good")

        with pytest.raises(ValueError, match="Review rationale/reason must be provided"):
            rec.promote_to_verified(reviewer="Alice", reason="")

    def test_rejection_requirements_and_event_recording(self):
        rec = RetrievalEvalRecord(
            id="v2_test_03",
            query="Ambiguous query about Redis?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        with pytest.raises(ValueError, match="Reviewer identity must be specified"):
            rec.reject_record(reviewer="", reason="Ambiguous")

        with pytest.raises(ValueError, match="Rejection reason must be provided"):
            rec.reject_record(reviewer="Bob", reason="")

        rec.reject_record(reviewer="Bob", reason="Question is vague and hallucinated.")
        assert rec.verification_status == "rejected"
        assert len(rec.review_history) == 1
        assert rec.review_history[0]["decision"] == "rejected"
        assert rec.review_history[0]["reviewer"] == "Bob"

    def test_verified_only_export_filters_pending_and_rejected(self):
        from evals.retrieval_dataset_schema import validate_benchmark_freeze

        r_verified = RetrievalEvalRecord(
            id="v2_test_04_v",
            query="Valid verified query?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        r_verified.promote_to_verified(reviewer="Alice", reason="Fully audited evidence.")

        r_rejected = RetrievalEvalRecord(
            id="v2_test_04_r",
            query="Rejected query?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )
        r_rejected.reject_record(reviewer="Bob", reason="Irrelevant.")

        r_pending = RetrievalEvalRecord(
            id="v2_test_04_p",
            query="Pending unverified query?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="test",
        )

        # Freeze check on verified-only succeeds
        val_verified = validate_benchmark_freeze([r_verified])
        assert val_verified["is_freeze_ready"] is True
        assert val_verified["total_approved"] == 1
        assert len(val_verified["content_hash"]) == 64

        # Freeze check including rejected or pending fails
        val_mixed = validate_benchmark_freeze([r_verified, r_rejected, r_pending])
        assert val_mixed["is_freeze_ready"] is False
        assert any("Only 'verified_human' permitted" in e for e in val_mixed["errors"])

    def test_empty_export_fails_freeze(self):
        from evals.retrieval_dataset_schema import validate_benchmark_freeze

        val_empty = validate_benchmark_freeze([])
        assert val_empty["is_freeze_ready"] is False
        assert any("0 approved records provided" in e for e in val_empty["errors"])

    def test_deterministic_content_hash(self):
        from evals.retrieval_dataset_schema import compute_records_content_hash

        rec = RetrievalEvalRecord(
            id="v2_test_05",
            query="Deterministic query test?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["some quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="verified_human",
            split="dev",
            reviewer_decision={"reviewer": "Alice", "reason": "OK", "timestamp": "2026-10-26T00:00:00Z"},
        )
        h1 = compute_records_content_hash([rec])
        h2 = compute_records_content_hash([rec])
        assert h1 == h2
        assert len(h1) == 64

        # Mutate query
        rec_mutated = RetrievalEvalRecord.from_dict(rec.to_dict())
        rec_mutated.query = "Different query text entirely?"
        h3 = compute_records_content_hash([rec_mutated])
        assert h1 != h3

    def test_evidence_resolution_local_corpus(self):
        from evals.retrieval_dataset_schema import resolve_chunk_evidence

        # Known UUID from parallel_work_queue.txt chunk 3
        known_uuid = "b9f7708c-ec05-517b-9e33-b33cd86aa109"
        res = resolve_chunk_evidence(known_uuid)
        assert res is not None
        assert res["filename"] == "parallel_work_queue.txt"
        assert res["chunk_index"] == 3
        assert "redis-pod.yaml" in res["text"]

        # Unknown UUID
        unknown_res = resolve_chunk_evidence("00000000-0000-0000-0000-000000000000")
        assert unknown_res is None

    def test_split_integrity_verbatim_leakage_detected(self):
        from evals.retrieval_dataset_schema import validate_benchmark_freeze

        r_dev = RetrievalEvalRecord(
            id="leak_01",
            query="Identical query text across splits?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="verified_human",
            split="dev",
            reviewer_decision={"reviewer": "Alice", "reason": "OK", "timestamp": "2026-10-26T00:00:00Z"},
        )
        r_test = RetrievalEvalRecord(
            id="leak_02",
            query="Identical query text across splits?",  # Verbatim copy!
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1],
            relevance_grades={VALID_UUID_1: 2},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="verified_human",
            split="test",
            reviewer_decision={"reviewer": "Alice", "reason": "OK", "timestamp": "2026-10-26T00:00:00Z"},
        )
        freeze_res = validate_benchmark_freeze([r_dev, r_test])
        assert freeze_res["is_freeze_ready"] is False
        assert any("verbatim queries appear in both dev and test" in e for e in freeze_res["errors"])

    def test_mocked_qdrant_verification_handling(self, monkeypatch):
        from unittest.mock import MagicMock
        from evals.retrieval_dataset_schema import validate_records_against_qdrant

        rec = RetrievalEvalRecord(
            id="mock_01",
            query="Mock query?",
            query_type="answerable_single_hop",
            difficulty="easy",
            domain="kubernetes",
            source_files=["parallel_work_queue.txt"],
            expected_sections=["STEP 1"],
            evidence_quotes=["quote"],
            candidate_chunk_ids=[VALID_UUID_1, VALID_UUID_2],
            relevance_grades={VALID_UUID_1: 2, VALID_UUID_2: 1},
            sufficient_evidence_sets=[[VALID_UUID_1]],
            verification_status="unverified_synthetic",
            split="dev",
        )

        mock_client = MagicMock()
        # Mock retrieve returning only VALID_UUID_1, missing VALID_UUID_2
        mock_pt = MagicMock()
        mock_pt.id = VALID_UUID_1
        mock_client.retrieve.return_value = [mock_pt]

        res = validate_records_against_qdrant([rec], mock_client, "test_collection")
        assert res["total_checked"] == 2
        assert res["found"] == 1
        assert res["missing"] == [VALID_UUID_2]
        assert res["all_verified"] is False

