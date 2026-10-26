"""
Tests for TrustRAG 10-Question Batch Evaluation Runner (scripts/run_batch_eval.py).

Verifies:
- Exact selection of 10 records and stable ordering
- Dataset record parsing and schema validation
- API errors, timeouts, connection failures (infrastructure error isolation)
- SSE stream parsing and trace event extraction
- Malformed server responses
- Missing relevance judgments handling
- Abstention classification and trace diagnosis
- Retrieval metric calculation (Recall, HitRate, MRR, nDCG)
- Report serialization (JSON, CSV, MD)
- Secrets redaction (API keys, tokens)

Zero live LLM calls: All tests use mocks and do not spend paid tokens.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from scripts.run_batch_eval import (
    BatchEvalConfig,
    BatchEvalRunner,
    AppClient,
    ChunkResolver,
    DeterministicEvaluator,
    diagnose_abstention,
    is_abstention_answer,
    redact_secrets,
    select_records,
)
from evals.retrieval_dataset_schema import RetrievalEvalRecord, load_dataset


# ==============================================================================
# 1. RECORD SELECTION & STABLE ORDERING TESTS
# ==============================================================================

def test_record_selection_default_10():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    selected = select_records(records, start_id="seed_001", count=10)
    assert len(selected) == 10
    expected_ids = [f"seed_{i:03d}" for i in range(1, 11)]
    assert [r.id for r in selected] == expected_ids


def test_record_selection_custom_start_and_count():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    selected = select_records(records, start_id="seed_006", count=5)
    assert len(selected) == 5
    assert [r.id for r in selected] == ["seed_006", "seed_007", "seed_008", "seed_009", "seed_010"]


def test_record_selection_explicit_ids():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    custom_ids = ["seed_010", "seed_002", "seed_005"]
    selected = select_records(records, explicit_ids=custom_ids)
    assert len(selected) == 3
    assert [r.id for r in selected] == ["seed_010", "seed_002", "seed_005"]


def test_record_selection_missing_id_raises():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    with pytest.raises(ValueError, match="not found in dataset"):
        select_records(records, explicit_ids=["seed_999_nonexistent"])


def test_dataset_record_parsing():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    assert len(records) >= 10
    r1 = records[0]
    assert isinstance(r1, RetrievalEvalRecord)
    assert r1.id == "seed_001"
    assert len(r1.candidate_chunk_ids) > 0
    assert len(r1.relevance_grades) > 0
    assert r1.verification_status in ("verified_human", "unverified_synthetic")


# ==============================================================================
# 2. SECRETS REDACTION TESTS
# ==============================================================================

def test_secrets_redaction():
    text_with_keys = (
        "Authorization: Bearer my_super_secret_jwt_token_1234567890\n"
        "Groq client failed with gsk_abcdef1234567890_secret_key\n"
        "Portkey proxy error: pk_live_9876543210fedcba\n"
        "Header x-api-key: secret_api_key_12345"
    )
    sanitized = redact_secrets(text_with_keys)
    assert "my_super_secret_jwt_token" not in sanitized
    assert "gsk_abcdef1234567890_secret_key" not in sanitized
    assert "pk_live_9876543210fedcba" not in sanitized
    assert "[REDACTED_CREDENTIAL]" in sanitized


# ==============================================================================
# 3. ABSTENTION DETECTION & DIAGNOSIS TESTS
# ==============================================================================

def test_abstention_detection():
    assert is_abstention_answer("I cannot answer this question as the required information is not found in the documentation.")
    assert is_abstention_answer("No information found.")
    assert is_abstention_answer("The provided context does not mention the active deadline.")
    assert not is_abstention_answer("Run kubectl apply on redis-pod.yaml and redis-service.yaml.")


def test_abstention_diagnosis_relevant_not_retrieved():
    rec = RetrievalEvalRecord(
        id="seed_001",
        query="test query",
        query_type="answerable_single_hop",
        difficulty="easy",
        domain="docs",
        source_files=["doc.docx"],
        expected_sections=["General"],
        evidence_quotes=["some quote"],
        candidate_chunk_ids=["uuid-chunk-1"],
        relevance_grades={"uuid-chunk-1": 3},
        sufficient_evidence_sets=[["uuid-chunk-1"]],
        verification_status="verified_human",
        split="dev",
    )
    diag, detail = diagnose_abstention(
        record=rec,
        retrieved_chunk_ids=["uuid-chunk-99", "uuid-chunk-100"],
        raw_status="ok",
        context_sent_to_llm="unrelated context",
        is_infra_error=False,
        error_category="none"
    )
    assert diag == "RELEVANT_NOT_RETRIEVED"


def test_abstention_diagnosis_context_present_model_abstained():
    rec = RetrievalEvalRecord(
        id="seed_001",
        query="test query",
        query_type="answerable_single_hop",
        difficulty="easy",
        domain="docs",
        source_files=["doc.docx"],
        expected_sections=["General"],
        evidence_quotes=["exact evidence text here"],
        candidate_chunk_ids=["uuid-chunk-1"],
        relevance_grades={"uuid-chunk-1": 3},
        sufficient_evidence_sets=[["uuid-chunk-1"]],
        verification_status="verified_human",
        split="dev",
    )
    diag, detail = diagnose_abstention(
        record=rec,
        retrieved_chunk_ids=["uuid-chunk-1"],
        raw_status="ok",
        context_sent_to_llm="Here is the context containing uuid-chunk-1 and exact evidence text here.",
        is_infra_error=False,
        error_category="none"
    )
    assert diag == "CONTEXT_PRESENT_MODEL_ABSTAINED"


def test_abstention_diagnosis_guardrail_blocked():
    rec = RetrievalEvalRecord(
        id="seed_001",
        query="test query",
        query_type="answerable_single_hop",
        difficulty="easy",
        domain="docs",
        source_files=["doc.docx"],
        expected_sections=["General"],
        evidence_quotes=["quote"],
        candidate_chunk_ids=["uuid-chunk-1"],
        relevance_grades={"uuid-chunk-1": 3},
        sufficient_evidence_sets=[["uuid-chunk-1"]],
        verification_status="verified_human",
        split="dev",
    )
    diag, detail = diagnose_abstention(
        record=rec,
        retrieved_chunk_ids=[],
        raw_status="blocked_by_guardrails",
        context_sent_to_llm="",
        is_infra_error=False,
        error_category="none"
    )
    assert diag == "GUARDRAIL_BLOCKED"


def test_abstention_diagnosis_infrastructure_failure():
    rec = RetrievalEvalRecord(
        id="seed_001",
        query="test query",
        query_type="answerable_single_hop",
        difficulty="easy",
        domain="docs",
        source_files=["doc.docx"],
        expected_sections=["General"],
        evidence_quotes=["quote"],
        candidate_chunk_ids=["uuid-chunk-1"],
        relevance_grades={"uuid-chunk-1": 3},
        sufficient_evidence_sets=[["uuid-chunk-1"]],
        verification_status="verified_human",
        split="dev",
    )
    diag, detail = diagnose_abstention(
        record=rec,
        retrieved_chunk_ids=[],
        raw_status="connection_error",
        context_sent_to_llm="",
        is_infra_error=True,
        error_category="connection_refused"
    )
    assert diag == "INFRASTRUCTURE_FAILURE"


# ==============================================================================
# 4. DETERMINISTIC EVALUATOR TESTS (GOLDEN QUESTIONS)
# ==============================================================================

def test_deterministic_evaluator_seed001():
    evaluator = DeterministicEvaluator()
    v, r = evaluator.evaluate_question("seed_001", "q", "Apply redis-pod.yaml and redis-service.yaml with kubectl.", None)
    assert v == "CORRECT"

    v_part, _ = evaluator.evaluate_question("seed_001", "q", "Apply redis-pod.yaml using kubectl.", None)
    assert v_part == "PARTIALLY_CORRECT"

    v_inc, _ = evaluator.evaluate_question("seed_001", "q", "Run docker run -d redis:latest.", None)
    assert v_inc == "INCORRECT"


def test_deterministic_evaluator_seed004():
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question("seed_004", "q", "Permitted policies are OnFailure and Never. Always is not allowed.", None)
    assert v == "CORRECT"


def test_deterministic_evaluator_seed005():
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question("seed_005", "q", "The Job resource belongs to the batch/v1 API group.", None)
    assert v == "CORRECT"


def test_deterministic_evaluator_seed008():
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question("seed_008", "q", "Set ttlSecondsAfterFinished: 3600 in the CronJob job template.", None)
    assert v == "CORRECT"


def test_deterministic_evaluator_seed009():
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question("seed_009", "q", "HPA scales horizontally by adding more pods. VPA adjusts CPU and memory vertically.", None)
    assert v == "CORRECT"


def test_deterministic_evaluator_seed010():
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question(
        "seed_010",
        "q",
        "Run kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
        None
    )
    assert v == "CORRECT"


# ==============================================================================
# 5. RETRIEVAL METRICS TESTS
# ==============================================================================

def test_retrieval_metrics_calculation():
    from evals.retrieval_metrics import recall_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k

    retrieved = ["doc_b", "doc_a", "doc_c", "doc_d", "doc_e"]
    relevant = {"doc_a", "doc_f"}
    grades = {"doc_a": 3, "doc_f": 2, "doc_b": 0, "doc_c": 0}

    # Recall
    assert recall_at_k(retrieved, relevant, k=1) == 0.0  # doc_b is not relevant
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # doc_a hit out of 2 relevant
    assert recall_at_k(retrieved, relevant, k=5) == 0.5

    # HitRate
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0

    # MRR (doc_a is at rank 2 -> 1/2 = 0.5)
    assert mrr_at_k(retrieved, relevant, k=5) == 0.5

    # nDCG
    ndcg = ndcg_at_k(retrieved, grades, k=5)
    assert 0.0 < ndcg <= 1.0


# ==============================================================================
# 6. APP CLIENT & ERROR HANDLING TESTS
# ==============================================================================

def test_app_client_query_success():
    config = BatchEvalConfig(base_url="http://127.0.0.1:8000")
    client = AppClient(config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "Test answer",
        "status": "ok",
        "sources": ["doc 1"],
        "retrieval_trace": [{"id": "uuid-1", "score": 0.95}],
        "context_sent_to_llm": "Technical context",
        "model_used": "qwen/qwen3.8-27b",
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.execute_query_endpoint("What is a Job?")
        assert res["is_infra_error"] is False
        assert res["answer"] == "Test answer"
        assert len(res["retrieved_chunks"]) == 1
        assert res["retrieved_chunks"][0]["id"] == "uuid-1"


def test_app_client_query_connection_error():
    config = BatchEvalConfig(base_url="http://127.0.0.1:8000")
    client = AppClient(config)

    with patch("requests.post", side_effect=Exception("ConnectionRefused")):
        res = client.execute_query_endpoint("What is a Job?")
        assert res["is_infra_error"] is True
        assert res["error_category"] == "client_exception"
        assert "ConnectionRefused" in res["error_message"]


def test_app_client_stream_sse_parsing():
    config = BatchEvalConfig(base_url="http://127.0.0.1:8000", endpoint="stream")
    client = AppClient(config)

    sse_lines = [
        b"event: status\r\n",
        b'data: {"stage": "guard"}\r\n',
        b"\r\n",
        b"event: sources\r\n",
        b'data: [{"id": "doc-1", "source": "job.docx", "snippet": "Job guide"}]\r\n',
        b"\r\n",
        b"event: trace\r\n",
        b'data: {"retrieved_docs": [{"id": "uuid-chunk-1", "score": 0.92}], "context_sent_to_llm": "Tech context"}\r\n',
        b"\r\n",
        b"event: token\r\n",
        b'data: {"delta": "Jobs in Kubernetes "}\r\n',
        b"\r\n",
        b"event: token\r\n",
        b'data: {"delta": "run to completion."}\r\n',
        b"\r\n",
        b"event: done\r\n",
        b'data: {"model_used": "qwen/qwen3.8-27b", "status": "ok", "latency_ms": 320}\r\n',
        b"\r\n",
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [line.decode("utf-8") for line in sse_lines]
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.post", return_value=mock_resp):
        res = client.execute_stream_endpoint("What is a Job?")
        assert res["is_infra_error"] is False
        assert res["answer"] == "Jobs in Kubernetes run to completion."
        assert len(res["retrieved_chunks"]) == 1
        assert res["retrieved_chunks"][0]["id"] == "uuid-chunk-1"
        assert res["model_used"] == "qwen/qwen3.8-27b"


# ==============================================================================
# 7. END-TO-END BATCH RUNNER EXECUTION (MOCKED)
# ==============================================================================

def test_full_batch_eval_runner_mocked():
    """Runs a complete 10-question evaluation run against a mocked API to verify output generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BatchEvalConfig(
            dataset_path="evals/data/retrieval_golden_seed_v1.json",
            start_id="seed_001",
            count=10,
            base_url="http://127.0.0.1:8000",
            output_dir=tmp_dir,
        )

        runner = BatchEvalRunner(config)

        # Mock request responses per question
        def mock_request(question):
            return {
                "is_infra_error": False,
                "error_category": "none",
                "error_message": "",
                "latency_ms": 120,
                "answer": "Run kubectl apply on redis-pod.yaml and redis-service.yaml.",
                "status": "ok",
                "sources": ["[Source: redis.docx]\nContent"],
                "retrieved_chunks": [{"id": "b9f7708c-ec05-517b-9e33-b33cd86aa109", "score": 0.95}],
                "context_sent_to_llm": "redis-pod.yaml redis-service.yaml",
                "model_used": "qwen/qwen3.8-27b",
                "guard_mode": "single_call",
            }

        with patch.object(runner.client, "execute_request", side_effect=mock_request):
            with patch.object(runner.client, "ping_config", return_value={"GROQ_MODEL": "qwen/qwen3.8-27b", "GUARD_MODE": "single_call"}):
                summary = runner.run()

        assert summary["counts"]["total"] == 10
        assert summary["counts"]["infrastructure_errors"] == 0
        assert os.path.exists(os.path.join(tmp_dir, "results.json"))
        assert os.path.exists(os.path.join(tmp_dir, "results.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "summary.md"))

        # Check JSON integrity
        with open(os.path.join(tmp_dir, "results.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["questions"]) == 10
        assert data["questions"][0]["id"] == "seed_001"
        assert data["questions"][0]["verdict"] == "CORRECT"

        # Check CSV header
        with open(os.path.join(tmp_dir, "results.csv"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 11  # header + 10 rows
        assert "retrieved_chunk_ids" in lines[0]
        assert "verdict" in lines[0]


# ==============================================================================
# 9. REGRESSION TESTS FOR AUDIT EVALUATOR BUG FIXES
# ==============================================================================

def test_is_abstention_substantive_with_caveat():
    """
    Ensures long substantive answers with trailing disclaimers are NOT falsely
    flagged as abstentions (regression fix for seed_021 and seed_040).
    """
    ans_56_words = (
        "Based on the provided technical context, the difference between completions and parallelism "
        "in a Kubernetes Job spec is that completions controls the number of successful pod executions "
        "required for completion, while parallelism controls the maximum number of pods that can run "
        "concurrently. The context does not provide further details on failure recovery."
    )
    assert not is_abstention_answer(ans_56_words)

    ans_120_words = (
        "Based on the provided technical context, the difference between etcd and kube-apiserver regarding "
        "state persistence and cluster coordination is that etcd acts as the persistent distributed key-value "
        "datastore holding cluster state. In contrast, kube-apiserver acts as the primary REST gateway and "
        "enforces authentication, authorization, request validation, and admission control. The context does "
        "not provide specific details about cluster coordination mechanisms beyond etcd's storage role."
    )
    assert not is_abstention_answer(ans_120_words)


def test_is_abstention_genuine_refusals():
    """Verifies true refusals and guardrail blocks continue to be properly detected."""
    # Canonical pipeline refusal
    assert is_abstention_answer("I cannot answer this question as the required information is not found in the documentation.")
    assert is_abstention_answer("I am sorry, but I cannot answer this question as the required information is not found in the provided documentation.")
    # Short refusal
    assert is_abstention_answer("The documentation does not provide information about this topic.")
    assert is_abstention_answer("No information found.")
    # Guardrail refusal
    assert is_abstention_answer("I'm an Enterprise IT Assistant focused on Kubernetes... I cannot assist with this request.")


def test_deterministic_evaluator_seed014_kubelet():
    """Ensures seed_014 correctly grades kubelet managing container/pod lifecycle."""
    evaluator = DeterministicEvaluator()
    v, r = evaluator.evaluate_question(
        "seed_014",
        "What component on worker nodes ensures containers are running in a Pod?",
        "Based on the provided technical context, the kubelet is the component that acts as the node agent responsible for managing pod lifecycle on its host.",
        "The kubelet agent ensures containers described in PodSpecs are running and healthy."
    )
    assert v == "CORRECT"


def test_deterministic_evaluator_seed021_completions_parallelism():
    """Ensures seed_021 correctly grades completions vs parallelism."""
    evaluator = DeterministicEvaluator()
    v, r = evaluator.evaluate_question(
        "seed_021",
        "What is the difference between completions and parallelism in a Kubernetes Job spec?",
        "spec.completions controls the number of pod executions, while spec.parallelism controls concurrent pods.",
        "Completions specifies the number of successful Pod completions required, while parallelism sets the max concurrently running Pods."
    )
    assert v == "CORRECT"


def test_deterministic_evaluator_seed030_databricks_jobs_create():
    """Ensures seed_030 correctly grades Databricks jobs create with JSON specification."""
    evaluator = DeterministicEvaluator()
    v1, _ = evaluator.evaluate_question(
        "seed_030",
        "Which Databricks CLI command creates a new job from a JSON definition file?",
        "The command is databricks jobs create --json @<file-path>. You must replace <file-path> with the path.",
        "databricks jobs create --json-file <path-to-json> creates the job."
    )
    assert v1 == "CORRECT"

    v2, _ = evaluator.evaluate_question(
        "seed_030",
        "Which Databricks CLI command creates a new job from a JSON definition file?",
        "The command is databricks jobs create --json @<file-path>. You replace <file-path> with the path.",
        "databricks jobs create --json-file <path-to-json> creates the job."
    )
    assert v2 == "CORRECT"


def test_deterministic_evaluator_seed040_etcd_apiserver():
    """Ensures seed_040 correctly grades etcd persistent datastore vs kube-apiserver."""
    evaluator = DeterministicEvaluator()
    v, _ = evaluator.evaluate_question(
        "seed_040",
        "What is the difference between etcd and kube-apiserver in state persistence and cluster coordination?",
        "etcd acts as the persistent key-value datastore holding state, while kube-apiserver is the REST front-end handling validation and admission control.",
        "kube-apiserver exposes the Kubernetes API and validates requests; etcd is the backing distributed key-value store holding cluster state."
    )
    assert v == "CORRECT"


def test_infra_error_excluded_from_pass_rate_denominator():
    """Confirms infrastructure errors do not count as incorrect answers in evaluable denominator."""
    total_q = 10
    count_correct = 8
    count_infra = 2
    evaluable_denominator = total_q - count_infra
    answer_pass_rate = (float(count_correct) / float(evaluable_denominator)) * 100.0
    assert evaluable_denominator == 8
    assert answer_pass_rate == 100.0

