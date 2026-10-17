import os
import sys
import json
import tempfile
import numpy as np
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query
from evals.metrics import compute_retrieval_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
ranker = Ranker(cache_dir=cache_dir)

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden = json.load(f)

golden_queries = [s["question"] for s in golden["rag_samples"]]

k8s_hard_negatives = [
    "How do I configure Istio Envoy sidecar injection with mutual TLS in Kubernetes?",
    "How do I create a Ceph RBD StorageClass for ReadWriteMany persistent volumes?",
    "How do I configure a Calico NetworkPolicy to block all egress traffic except DNS?",
    "How do I deploy virtual machines using KubeVirt on bare-metal worker nodes?",
    "How do I configure Helm pre-install hooks for database schema migrations?",
]

noisy_overlap_negatives = [
    "How does the block-sorting lossless data compression algorithm reduce memory footprint in distributed queues?",
    "How does NUMA memory affinity affect Linux page cache IOPS for multi-core systems?",
    "Explain the mathematical theory of communication and entropy in message queues",
    "How do Patricia tries optimize lock-free concurrent hash tables in operating system kernels?",
    "What are the optimizing compiler transformations for relational database OLAP queries?",
]

def retrieve_and_score(query, collection_name="enterprise_rag", limit=10):
    q_vec = embed_query(query)
    resp = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=limit,
        with_payload=True
    )
    passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(resp.points)]
    if not passages:
        return []
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return reranked

print("=== THRESHOLD CALIBRATION ANALYSIS ===")
print("Evaluating on collection: enterprise_rag (dense baseline points)")

# 1. Answerable Questions
print("\n--- 1. Answerable Golden Questions (15 queries) ---")
ans_top_scores = []
ans_surviving_counts = {0.0: [], 0.1: [], 0.3: [], 0.5: []}

for idx, q in enumerate(golden_queries):
    res = retrieve_and_score(q)
    scores = [r["score"] for r in res]
    top_score = scores[0] if scores else 0.0
    ans_top_scores.append(top_score)
    for th in [0.0, 0.1, 0.3, 0.5]:
        surviving = sum(1 for s in scores if s >= th)
        ans_surviving_counts[th].append(surviving)
    print(f"[Q{idx+1:02d}] Top: {top_score:.4f} | Surviving (0.1/0.3/0.5): {sum(1 for s in scores if s >= 0.1)}/{sum(1 for s in scores if s >= 0.3)}/{sum(1 for s in scores if s >= 0.5)} | {q[:55]}")

# 2. Hard K8s Negatives
print("\n--- 2. Hard Kubernetes Negatives (Out-of-Corpus K8s Topics) ---")
k8s_top_scores = []
k8s_surviving_counts = {0.0: [], 0.1: [], 0.3: [], 0.5: []}

for idx, q in enumerate(k8s_hard_negatives):
    res = retrieve_and_score(q)
    scores = [r["score"] for r in res]
    top_score = scores[0] if scores else 0.0
    k8s_top_scores.append(top_score)
    for th in [0.0, 0.1, 0.3, 0.5]:
        surviving = sum(1 for s in scores if s >= th)
        k8s_surviving_counts[th].append(surviving)
    print(f"[K8s-Neg {idx+1}] Top: {top_score:.4f} | Surviving (0.1/0.3/0.5): {sum(1 for s in scores if s >= 0.1)}/{sum(1 for s in scores if s >= 0.3)}/{sum(1 for s in scores if s >= 0.5)} | {q[:55]}")

# 3. Noisy Overlap Negatives
print("\n--- 3. Noisy-Corpus Overlap Negatives ---")
noisy_top_scores = []
noisy_surviving_counts = {0.0: [], 0.1: [], 0.3: [], 0.5: []}

for idx, q in enumerate(noisy_overlap_negatives):
    res = retrieve_and_score(q)
    scores = [r["score"] for r in res]
    top_score = scores[0] if scores else 0.0
    noisy_top_scores.append(top_score)
    for th in [0.0, 0.1, 0.3, 0.5]:
        surviving = sum(1 for s in scores if s >= th)
        noisy_surviving_counts[th].append(surviving)
    print(f"[Noisy-Neg {idx+1}] Top: {top_score:.4f} | Surviving (0.1/0.3/0.5): {sum(1 for s in scores if s >= 0.1)}/{sum(1 for s in scores if s >= 0.3)}/{sum(1 for s in scores if s >= 0.5)} | {q[:55]}")

print("\n=== SCORE DISTRIBUTION SUMMARY ===")
print(f"Answerable Golden (n=15):     Mean Top Score = {np.mean(ans_top_scores):.4f} (Min: {np.min(ans_top_scores):.4f}, Max: {np.max(ans_top_scores):.4f})")
print(f"Hard K8s Negatives (n=5):     Mean Top Score = {np.mean(k8s_top_scores):.4f} (Min: {np.min(k8s_top_scores):.4f}, Max: {np.max(k8s_top_scores):.4f})")
print(f"Noisy Overlap Negatives (n=5): Mean Top Score = {np.mean(noisy_top_scores):.4f} (Min: {np.min(noisy_top_scores):.4f}, Max: {np.max(noisy_top_scores):.4f})")

print("\n=== AVERAGE SURVIVING CHUNKS PER QUERY (OUT OF 10) ===")
print(f"Threshold 0.0: Answerable = {np.mean(ans_surviving_counts[0.0]):.1f} | Hard K8s Neg = {np.mean(k8s_surviving_counts[0.0]):.1f} | Noisy Overlap = {np.mean(noisy_surviving_counts[0.0]):.1f}")
print(f"Threshold 0.1: Answerable = {np.mean(ans_surviving_counts[0.1]):.1f} | Hard K8s Neg = {np.mean(k8s_surviving_counts[0.1]):.1f} | Noisy Overlap = {np.mean(noisy_surviving_counts[0.1]):.1f}")
print(f"Threshold 0.3: Answerable = {np.mean(ans_surviving_counts[0.3]):.1f} | Hard K8s Neg = {np.mean(k8s_surviving_counts[0.3]):.1f} | Noisy Overlap = {np.mean(noisy_surviving_counts[0.3]):.1f}")
print(f"Threshold 0.5: Answerable = {np.mean(ans_surviving_counts[0.5]):.1f} | Hard K8s Neg = {np.mean(k8s_surviving_counts[0.5]):.1f} | Noisy Overlap = {np.mean(noisy_surviving_counts[0.5]):.1f}")
