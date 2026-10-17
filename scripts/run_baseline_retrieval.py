import os
import sys
import json
import time
import numpy as np
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query
from evals.metrics import compute_retrieval_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
# Old cache setting or default
ranker = Ranker()

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden = json.load(f)

samples = golden["rag_samples"]
print(f"Running baseline retrieval evaluation on OLD collection 'enterprise_rag' ({len(samples)} queries)...")

evaluated_samples = []
embed_times = []
qdrant_times = []
rerank_times = []
total_times = []

for idx, s in enumerate(samples):
    q = s["question"]
    
    # Stage 1: Embed query (Dense MPNet)
    t0 = time.perf_counter()
    q_vec = embed_query(q)
    t_embed = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Qdrant dense vector search on old collection
    t1 = time.perf_counter()
    # Old collection has unnamed vector
    resp = client.query_points(
        collection_name="enterprise_rag",
        query=q_vec,
        limit=15,
        with_payload=True,
    )
    t_qdrant = (time.perf_counter() - t1) * 1000.0

    # Stage 3: FlashRank rerank (no threshold in old path)
    t2 = time.perf_counter()
    candidates = resp.points
    passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(candidates)]
    if passages:
        reranked = ranker.rerank(RerankRequest(query=q, passages=passages))
        # Map back to points in reranked order
        top_pts = [candidates[r["id"]] for r in reranked[:5]]
    else:
        top_pts = []
    t_rerank = (time.perf_counter() - t2) * 1000.0

    t_total = t_embed + t_qdrant + t_rerank

    embed_times.append(t_embed)
    qdrant_times.append(t_qdrant)
    rerank_times.append(t_rerank)
    total_times.append(t_total)

    actual_contexts = [
        {
            "source": p.payload.get("source", "Unknown"),
            "section": p.payload.get("section", "General"),
            "content": p.payload.get("text", "")
        }
        for p in top_pts
    ]

    evaluated_samples.append({
        "id": s["id"],
        "question": q,
        "expected_source": s.get("expected_source"),
        "expected_section": s.get("expected_section"),
        "actual_contexts": actual_contexts,
        "latency_ms": {
            "embed": round(t_embed, 1),
            "qdrant": round(t_qdrant, 1),
            "rerank": round(t_rerank, 1),
            "total": round(t_total, 1),
        }
    })
    print(f"[{idx+1}/15] Done: {q[:50]}... (Total: {t_total:.1f}ms)")

# Compute metrics
retrieval_metrics = compute_retrieval_metrics(evaluated_samples)

avg_latencies = {
    "embed_ms": round(float(np.mean(embed_times)), 1),
    "qdrant_ms": round(float(np.mean(qdrant_times)), 1),
    "rerank_ms": round(float(np.mean(rerank_times)), 1),
    "total_retrieval_ms": round(float(np.mean(total_times)), 1),
}

output_data = {
    "collection": "enterprise_rag",
    "mode": "baseline_old_chunking_dense",
    "point_count": client.count("enterprise_rag").count,
    "metrics": {
        "hit@5": retrieval_metrics["source_hit@5"],
        "mrr": retrieval_metrics["source_mrr"],
        "section_hit@5": retrieval_metrics["section_hit@5"],
        "section_mrr": retrieval_metrics["section_mrr"],
    },
    "latencies_ms": avg_latencies,
    "details": retrieval_metrics["details"],
}

out_path = "evals/results/baseline_retrieval.json"
os.makedirs("evals/results", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("\n=== BASELINE RETRIEVAL EVALUATION RESULTS ===")
print(f"Collection: enterprise_rag ({output_data['point_count']} points)")
print(f"Hit@5 (Source): {output_data['metrics']['hit@5']}")
print(f"MRR (Source):   {output_data['metrics']['mrr']}")
print(f"Hit@5 (Section): {output_data['metrics']['section_hit@5']}")
print(f"MRR (Section):   {output_data['metrics']['section_mrr']}")
print(f"Latencies (ms): Embed={avg_latencies['embed_ms']}ms, Qdrant={avg_latencies['qdrant_ms']}ms, Rerank={avg_latencies['rerank_ms']}ms, Total={avg_latencies['total_retrieval_ms']}ms")
print(f"Saved to: {out_path}")
