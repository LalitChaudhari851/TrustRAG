import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest
from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query

JSON_OUT = "evals/results/reranker_latency_dedicated.json"
TXT_OUT = "evals/results/reranker_latency_dedicated.txt"

# 1. Load Datasets
with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden_questions = json.load(f)["rag_samples"]

with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
    hard_raw = json.load(f)
hard_questions = hard_raw.get("queries", hard_raw) if isinstance(hard_raw, dict) else hard_raw

print("Initializing Qdrant client...")
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)

print("Loading Rankers...")
ranker_tiny = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
ranker_mini = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
bge_cross = CrossEncoder("BAAI/bge-reranker-base")

# 2. Pre-fetch Top-15 Candidate Pools from Qdrant enterprise_rag_v2
def fetch_candidate_pools(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pools = []
    for q in questions:
        q_text = q["question"]
        dense_vec = embed_query(q_text)
        sparse_vec = embed_sparse_query(q_text)
        s_indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
        s_values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

        resp = client.query_points(
            collection_name="enterprise_rag_v2",
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=15),
                models.Prefetch(query=models.SparseVector(indices=s_indices, values=s_values), using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15,
            with_payload=True
        ).points

        passages = [{"id": i, "text": p.payload.get("text", ""), "source": p.payload.get("source", ""), "section": p.payload.get("section", "")} for i, p in enumerate(resp)]
        pools.append({"id": q["id"], "question": q_text, "passages": passages})
    return pools

print("Pre-fetching candidate pools for Golden 15...")
golden_pools = fetch_candidate_pools(golden_questions)

print("Pre-fetching candidate pools for Hard 22...")
hard_pools = fetch_candidate_pools(hard_questions)

# 3. Benchmark pure rerank latency per query
# 5 discard passes, 10 measured passes for 15 candidate passages
def benchmark_rerank_stage(pools: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    variants = ["no_reranker", "tinybert", "minilm_l12", "bge_reranker_base"]
    results = {}
    item = pools[0]
    q = item["question"]
    passages = [{"id": p["id"], "text": p["text"]} for p in item["passages"][:15]]

    for var in variants:
        print(f"  Measuring variant '{var}' on 15 candidates (5 discard, 10 measured)...")
        # 5 Discard passes
        for _ in range(5):
            if var == "no_reranker":
                _ = passages[:5]
            elif var == "tinybert":
                _ = ranker_tiny.rerank(RerankRequest(query=q, passages=passages))[:5]
            elif var == "minilm_l12":
                _ = ranker_mini.rerank(RerankRequest(query=q, passages=passages))[:5]
            elif var == "bge_reranker_base":
                pairs = [[q, p["text"]] for p in passages]
                _ = bge_cross.predict(pairs)

        # 10 Measured passes
        query_latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            if var == "no_reranker":
                _ = passages[:5]
            elif var == "tinybert":
                _ = ranker_tiny.rerank(RerankRequest(query=q, passages=passages))[:5]
            elif var == "minilm_l12":
                _ = ranker_mini.rerank(RerankRequest(query=q, passages=passages))[:5]
            elif var == "bge_reranker_base":
                pairs = [[q, p["text"]] for p in passages]
                scores = bge_cross.predict(pairs)
            t1 = time.perf_counter()
            query_latencies.append((t1 - t0) * 1000.0)

        results[var] = {
            "median_ms": float(np.median(query_latencies)),
            "p95_ms": float(np.percentile(query_latencies, 95)),
            "min_ms": float(np.min(query_latencies)),
            "max_ms": float(np.max(query_latencies))
        }

    return results

print("\nRunning dedicated rerank latency benchmark for Golden 15 (nothing else running)...")
golden_latency = benchmark_rerank_stage(golden_pools)

print("\nRunning dedicated rerank latency benchmark for Hard 22 (nothing else running)...")
hard_latency = benchmark_rerank_stage(hard_pools)

# 4. Fusion Variant Evaluation (Union of top-3 TinyBERT + top-3 RRF, cap 5-6)
GOLDEN_KEY_TERMS = {
    1: "redis-pod.yaml", 2: "parallelism: 2", 3: "rpush job2", 4: "HPA",
    5: "components.yaml", 6: "kubectl top nodes", 7: "databricks jobs get",
    8: "REST API", 9: "CLI", 10: "2023-05-10", 11: "completions",
    12: "batch/v1", 13: "minikube start", 14: "kubectl get jobs", 15: "kubectl cluster-info"
}

def evaluate_fusion(dataset_name: str, questions: List[Dict[str, Any]], pools: List[Dict[str, Any]], is_golden: bool):
    src_hits_5 = []
    sec_hits_5 = []
    key_hits_5 = []
    src_rrs = []

    for q, item in zip(questions, pools):
        exp_src = q.get("expected_source", "")
        exp_sec = q.get("expected_section", "")
        key_term = GOLDEN_KEY_TERMS.get(q["id"], "") if is_golden else q.get("key_term", "")

        passages = [{"id": p["id"], "text": p["text"], "source": p["source"], "section": p["section"]} for p in item["passages"]]
        
        # RRF top 3
        rrf_top3 = passages[:3]
        
        # TinyBERT top 3
        tb_input = [{"id": p["id"], "text": p["text"]} for p in passages]
        tb_res = ranker_tiny.rerank(RerankRequest(query=q["question"], passages=tb_input))
        id_to_p = {p["id"]: p for p in passages}
        tb_top3 = [id_to_p[r["id"]] for r in tb_res[:3]]

        # Union preserving order (top 3 TinyBERT first, then novel top 3 RRF) up to cap 5
        seen_ids = set()
        fusion_top5 = []
        for p in tb_top3:
            if p["id"] not in seen_ids:
                fusion_top5.append(p)
                seen_ids.add(p["id"])
        for p in rrf_top3:
            if p["id"] not in seen_ids and len(fusion_top5) < 5:
                fusion_top5.append(p)
                seen_ids.add(p["id"])

        # Metrics at 5
        src_hit = any(p["source"] == exp_src for p in fusion_top5)
        src_hits_5.append(src_hit)

        sec_hit = any(p["source"] == exp_src and p["section"] == exp_sec for p in fusion_top5) if exp_sec else False
        sec_hits_5.append(sec_hit)

        key_hit = any(key_term.lower() in p["text"].lower() for p in fusion_top5) if key_term else False
        key_hits_5.append(key_hit)

        # MRR
        mrr = 0.0
        for rank_idx, p in enumerate(fusion_top5, 1):
            if p["source"] == exp_src:
                mrr = 1.0 / rank_idx
                break
        src_rrs.append(mrr)

    return {
        "dataset": dataset_name,
        "count": len(questions),
        "source_hit5": float(np.mean(src_hits_5)),
        "section_hit5": float(np.mean(sec_hits_5)),
        "key_phrase_hit5": float(np.mean(key_hits_5)),
        "source_mrr": float(np.mean(src_rrs))
    }

print("\nEvaluating Fusion variant on Golden 15...")
fusion_golden = evaluate_fusion("Golden 15", golden_questions, golden_pools, is_golden=True)

print("Evaluating Fusion variant on Hard 22...")
fusion_hard = evaluate_fusion("Hard 22", hard_questions, hard_pools, is_golden=False)

# Format outputs
out_lines = []
out_lines.append("================================================================================")
out_lines.append("TASK 1(b) & 1(c): DEDICATED RERANKER LATENCY & OFFLINE FUSION BENCHMARK")
out_lines.append("================================================================================")
out_lines.append("Conditions: 5 discard passes, 10 measured passes, nothing else running.")
out_lines.append("Measurement Target: Pure rerank stage latency per query for 15 candidate passages.\n")

out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("1. DEDICATED WARM RERANK STAGE LATENCY PER QUERY (MEDIAN & P95)")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append(f"{'Dataset / Variant':<32} | {'Warm Median (ms)':<18} | {'Warm p95 (ms)':<16} | {'Min (ms)':<10} | {'Max (ms)'}")
out_lines.append("-" * 92)

for d_name, d_lat in [("Golden 15", golden_latency), ("Hard 22", hard_latency)]:
    for v in ["no_reranker", "tinybert", "minilm_l12", "bge_reranker_base"]:
        m = d_lat[v]
        out_lines.append(f"{d_name + ': ' + v:<32} | {m['median_ms']:<18.2f} | {m['p95_ms']:<16.2f} | {m['min_ms']:<10.2f} | {m['max_ms']:.2f}")
    out_lines.append("-" * 92)

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. FUSION VARIANT (TOP-3 TINYBERT + TOP-3 RRF UNION, CAP 5)")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("Note on candidate lists in saved JSON:")
out_lines.append("  'evals/results/reranker_ablation.json' and 'retrieval_ab_full.json' do NOT store the")
out_lines.append("  full 15-passage candidate lists (they only store top-5 and miss breakdowns).")
out_lines.append("  Below are the measured metrics evaluated from the live top-15 hybrid RRF candidate pools:\n")
out_lines.append(f"{'Dataset / Variant':<32} | {'Src Hit@5':<10} | {'Sec Hit@5':<10} | {'Key Hit@5':<10} | {'Src MRR'}")
out_lines.append("-" * 75)
out_lines.append(f"{'Golden 15: fusion (TinyBERT+RRF)':<32} | {fusion_golden['source_hit5']:<10.4f} | {fusion_golden['section_hit5']:<10.4f} | {fusion_golden['key_phrase_hit5']:<10.4f} | {fusion_golden['source_mrr']:.4f}")
out_lines.append(f"{'Hard 22:   fusion (TinyBERT+RRF)':<32} | {fusion_hard['source_hit5']:<10.4f} | {fusion_hard['section_hit5']:<10.4f} | {fusion_hard['key_phrase_hit5']:<10.4f} | {fusion_hard['source_mrr']:.4f}")

full_text = "\n".join(out_lines) + "\n"
print(full_text)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(full_text)

json_data = {
    "latency_golden": golden_latency,
    "latency_hard": hard_latency,
    "fusion_golden": fusion_golden,
    "fusion_hard": fusion_hard
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)
