import json
import numpy as np
from flashrank import Ranker, RerankRequest
from app.services.retrieval.qdrant_service import search_enterprise_knowledge

ranker = Ranker()

answerable_queries = [
    "How do CronJobs handle pod restarts in Kubernetes?",
    "What is Horizontal Pod Autoscaling in Kubernetes?",
    "How do I monitor the status of a Kubernetes Job?",
    "Explain the parallel work queue in Kubernetes",
    "What are the core components of Kubernetes master?"
]

off_topic_queries = [
    "How do I make chocolate chip cookies?",
    "Who was the first president of the United States?",
    "Tell me a funny joke about cats",
    "What is the capital of France?",
    "Recommend a sci-fi movie"
]

print("=== FLASH RANK SCORE DISTRIBUTION ANALYSIS ===\n")

ans_top_scores = []
for q in answerable_queries:
    raw = search_enterprise_knowledge(q, limit=10)
    passages = [{"id": i, "text": d["content"]} for i, d in enumerate(raw)]
    if passages:
        res = ranker.rerank(RerankRequest(query=q, passages=passages))
        scores = [r["score"] for r in res]
        top = scores[0] if scores else 0.0
        ans_top_scores.append(top)
        print(f"[Answerable] '{q[:45]}' -> Top: {top:.4f} | Min: {scores[-1]:.4f}")

off_top_scores = []
for q in off_topic_queries:
    raw = search_enterprise_knowledge(q, limit=10)
    passages = [{"id": i, "text": d["content"]} for i, d in enumerate(raw)]
    if passages:
        res = ranker.rerank(RerankRequest(query=q, passages=passages))
        scores = [r["score"] for r in res]
        top = scores[0] if scores else 0.0
        off_top_scores.append(top)
        print(f"[Off-Topic]  '{q[:45]}' -> Top: {top:.4f} | Min: {scores[-1]:.4f}")

print("\n--- Score Summary ---")
print(f"Answerable Top Scores: Mean = {np.mean(ans_top_scores):.4f}, Min = {np.min(ans_top_scores):.4f}, Max = {np.max(ans_top_scores):.4f}")
print(f"Off-Topic Top Scores:  Mean = {np.mean(off_top_scores):.4f}, Min = {np.min(off_top_scores):.4f}, Max = {np.max(off_top_scores):.4f}")
