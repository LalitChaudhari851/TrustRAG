import os
import uuid
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.processor import (
    ensure_collection_exists,
    process_file,
    compute_point_id,
    PROJECT_UUID_NAMESPACE,
)
from app.services.retrieval.qdrant_service import compute_python_rrf
from app.services.retrieval.ranking_service import rerank_documents
from app.agents.nodes.responder import generate_node


def test_project_uuid5_namespace_is_not_dns():
    """Verify we are using a project-specific namespace, not uuid.NAMESPACE_DNS."""
    assert PROJECT_UUID_NAMESPACE != uuid.NAMESPACE_DNS
    pid1 = compute_point_id("doc1.txt", 0, "sample content")
    pid2 = compute_point_id("doc1.txt", 0, "sample content")
    assert pid1 == pid2
    assert uuid.UUID(pid1).version == 5


def test_ingestion_idempotency_memory_qdrant():
    """
    Test with QdrantClient(':memory:'):
    Ingest a file twice -> point count must remain identical (idempotency).
    """
    client = QdrantClient(":memory:")
    collection_name = "test_idempotency"

    with patch("app.ingestion.processor.get_embedding_dim", return_value=4):
        ensure_collection_exists(client, collection_name, wipe=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Line 1: Kubernetes CronJob management and distributed cluster workload orchestration.\nLine 2: Scheduled task execution in enterprise infrastructure with high availability and fault tolerance.")
        temp_path = f.name

    test_manifest_file = temp_path + ".manifest.json"

    try:
        mock_dense = [[0.1, 0.2, 0.3, 0.4]]
        mock_sparse = [models.SparseVector(indices=[10, 20], values=[0.5, 0.8])]

        with patch("app.ingestion.processor.MANIFEST_FILE", test_manifest_file), \
             patch("app.ingestion.processor.embed_texts", return_value=mock_dense), \
             patch("app.ingestion.processor.embed_sparse_texts", return_value=mock_sparse):
            # First ingestion
            process_file(temp_path, "sample_idempotent.txt", "test", client=client, collection_name=collection_name)
            count1 = client.count(collection_name).count

            # Second ingestion of the exact same file
            process_file(temp_path, "sample_idempotent.txt", "test", client=client, collection_name=collection_name)
            count2 = client.count(collection_name).count

            assert count1 == count2
            assert count1 > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(test_manifest_file):
            os.remove(test_manifest_file)


def test_ingestion_edited_file_cleans_stale_chunks():
    """
    Test with QdrantClient(':memory:'):
    Ingesting an edited file (fewer chunks) cleans out stale chunks via source filtering.
    """
    client = QdrantClient(":memory:")
    collection_name = "test_stale_chunks"

    with patch("app.ingestion.processor.get_embedding_dim", return_value=4):
        ensure_collection_exists(client, collection_name, wipe=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        temp_path = f.name

    test_manifest_file = temp_path + ".manifest_stale.json"

    try:
        # Version 1: 3 paragraphs -> produces 3 chunks
        mock_dense_3 = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
        mock_sparse_3 = [
            models.SparseVector(indices=[1], values=[1.0]),
            models.SparseVector(indices=[2], values=[1.0]),
            models.SparseVector(indices=[3], values=[1.0]),
        ]

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("# Section A\nDetailed content for chunk A with sufficient explanatory details and enterprise operational specifications to exceed minimum tokens.\n\n# Section B\nDetailed content for chunk B with sufficient explanatory details and enterprise operational specifications to exceed minimum tokens.\n\n# Section C\nDetailed content for chunk C with sufficient explanatory details and enterprise operational specifications to exceed minimum tokens.")

        with patch("app.ingestion.processor.MANIFEST_FILE", test_manifest_file), \
             patch("app.ingestion.processor.embed_texts", return_value=mock_dense_3), \
             patch("app.ingestion.processor.embed_sparse_texts", return_value=mock_sparse_3):
            process_file(temp_path, "article.txt", "test", client=client, collection_name=collection_name)
            assert client.count(collection_name).count == 3

        # Version 2: Edited down to 1 chunk
        mock_dense_1 = [[0.9, 0.8, 0.7, 0.6]]
        mock_sparse_1 = [models.SparseVector(indices=[9], values=[1.0])]

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("# Consolidated Section\nOnly single consolidated chunk remaining with comprehensive technical details and architecture guidance that satisfies token threshold limits.")

        with patch("app.ingestion.processor.MANIFEST_FILE", test_manifest_file), \
             patch("app.ingestion.processor.embed_texts", return_value=mock_dense_1), \
             patch("app.ingestion.processor.embed_sparse_texts", return_value=mock_sparse_1):
            process_file(temp_path, "article.txt", "test", client=client, collection_name=collection_name)
            assert client.count(collection_name).count == 1

            # Verify the remaining point belongs to the updated content
            pts, _ = client.scroll(collection_name, limit=10, with_payload=True)
            assert len(pts) == 1
            assert "Consolidated" in pts[0].payload["section"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(test_manifest_file):
            os.remove(test_manifest_file)


def test_plain_python_rrf():
    """Verify plain-Python RRF ordering matches reciprocal rank calculation."""
    dense_ranks = ["docA", "docB", "docC"]
    sparse_ranks = ["docB", "docD", "docA"]
    fused = compute_python_rrf(dense_ranks, sparse_ranks, k=60)
    # docB: 1/62 + 1/61 = ~0.0325
    # docA: 1/61 + 1/63 = ~0.0322
    assert fused[0][0] == "docB"
    assert fused[1][0] == "docA"


def test_reranker_threshold_filtering():
    """Verify reranker drops documents below the min_score threshold."""
    docs = [
        {"content": "High relevance chunk about Kubernetes CronJobs", "source": "cron.docx"},
        {"content": "Irrelevant noise about chocolate cake baking", "source": "cake.txt"}
    ]

    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.992},
        {"id": 1, "score": 0.0000}
    ]

    with patch("app.services.retrieval.ranking_service._get_ranker", return_value=mock_ranker):
        # min_score = 0.30 drops the second document
        filtered = rerank_documents("Kubernetes CronJobs", docs, top_n=5, min_score=0.30)
        assert len(filtered) == 1
        assert filtered[0]["source"] == "cron.docx"
        assert filtered[0]["rerank_score"] == 0.992


def test_responder_refusal_on_empty_context():
    """
    Verify that when the retrieval threshold leaves 0 chunks,
    the responder immediately returns the 'not enough context' refusal
    without executing an ungrounded LLM synthesis call.
    """
    state = {
        "messages": [{"role": "user", "content": "How do I make chocolate chip cookies?"}],
        "current_query": "How do I make chocolate chip cookies?",
        "documents": [],  # All chunks dropped by reranker threshold
        "plan": ["Start", "Intent: Technical", "Context Filtered Out (Below Threshold)"]
    }

    with patch("app.gateway.execute_gateway_chat") as mock_llm:
        result = generate_node(state)
        # LLM synthesis must NOT be called
        mock_llm.assert_not_called()

        assert "enough context" in result["final_answer"].lower()
        assert result["status"] == "Insufficient context."
        assert "Refusal: Insufficient Context" in result["plan"]

