import sys
import os
import re
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

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

JSON_OUT = "evals/results/reranker_ablation.json"
TXT_OUT = "evals/results/reranker_ablation.txt"
COLLECTION = "enterprise_rag_v2"

GOLDEN_KEY_TERMS = {
    1: "redis-pod.yaml",
    2: "parallelism: 2",
    3: "rpush job2",
    4: "HPA",
    5: "components.yaml",
    6: "kubectl top nodes",
    7: "databricks jobs get",
    8: "REST API",
    9: "CLI",
    10: "2023-05-10",
    11: "completions",
    12: "batch/v1",
    13: "minikube start",
    14: "kubectl get jobs",
    15: "kubectl cluster-info"
}

def normalize_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

print("Initializing Qdrant client and models...")
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)

print("Loading Ranker 1: ms-marco-TinyBERT-L-2-v2...")
ranker_tiny = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

print("Loading Ranker 2: ms-marco-MiniLM-L-12-v2...")
ranker_minilm = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

bge_available = False
ranker_bge = None
try:
    print("Loading Cross-Encoder 3: BAAI/bge-reranker-base (offline)...")
    ranker_bge = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)
    bge_available = True
    print("Successfully loaded BAAI/bge-reranker-base.")
except Exception as e:
    print(f"Could not load BAAI/bge-reranker-base offline: {e}. Skipping variant 4.")

with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
    golden_raw = json.load(f)
golden_questions = golden_raw.get("rag_samples", golden_raw)

with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
    hard_questions = json.load(f)

variants = ["no_reranker", "tinybert", "minilm_l12"]
if bge_available:
    variants.append("bge_reranker_base")

def rerank_passages(variant: str, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if variant == "no_reranker":
        return list(passages)
    elif variant == "tinybert":
        res = ranker_tiny.rerank(RerankRequest(query=query, passages=passages))
        return res
    elif variant == "minilm_l12":
        res = ranker_minilm.rerank(RerankRequest(query=query, passages=passages))
        return res
    elif variant == "bge_reranker_base":
        pairs = [[query, p.get("text", "")] for p in passages]
        scores = ranker_bge.predict(pairs)
        scored_passages = []
        for p, s in zip(passages, scores):
            cp = dict(p)
            cp["score"] = float(s)
            scored_passages.append(cp)
        scored_passages.sort(key=lambda x: x["score"], reverse=True)
        return scored_passages
    else:
        raise ValueError(f"Unknown variant: {variant}")

def run_evaluation(dataset_name: str, questions: List[Dict[str, Any]], is_golden: bool):
    results = {v: {
        "src_hit5": [], "sec_hit5": [], "key_hit5": [], "src_rrs": [], "sec_rrs": [],
        "key_ranks": []
    } for v in variants}
    key_hit15_list = []
    per_query_details = []

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        exp_src = q.get("expected_source", "")
        exp_sec = q.get("expected_section", "")
        key_term = GOLDEN_KEY_TERMS.get(qid, "") if is_golden else q.get("key_term", "")

        norm_src = normalize_str(exp_src)
        norm_sec = normalize_str(exp_sec)
        norm_key = normalize_str(key_term)

        dense_vec = embed_query(q_text)
        sparse_vec = embed_sparse_query(q_text)
        s_idx = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
        s_val = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=15),
                models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15,
            with_payload=True
        )
        pts = resp.points

        passages = []
        for idx, p in enumerate(pts):
            payload = p.payload or {}
            passages.append({
                "id": idx,
                "text": payload.get("text", ""),
                "meta": {
                    "source": payload.get("source", ""),
                    "section": payload.get("section", ""),
                    "score": getattr(p, "score", 0.0)
                }
            })

        # Key-phrase hit@15 (same for all variants since all see the same top 15 pool)
        q_key15 = False
        for p in passages:
            if norm_key and norm_key in normalize_str(p.get("text", "")):
                q_key15 = True
                break
        key_hit15_list.append(1.0 if q_key15 else 0.0)

        q_info = {
            "id": qid,
            "question": q_text,
            "expected_source": exp_src,
            "expected_section": exp_sec,
            "key_term": key_term,
            "key_hit15": q_key15,
            "variants": {}
        }

        for v in variants:
            ranked = rerank_passages(v, q_text, passages)
            top5 = ranked[:5]

            src_hit = False
            sec_hit = False
            key_hit = False
            src_rr = 0.0
            sec_rr = 0.0
            key_rank = None

            for r_idx, cand in enumerate(ranked, 1):
                c_text = cand.get("text", "")
                if norm_key and norm_key in normalize_str(c_text):
                    if key_rank is None:
                        key_rank = r_idx

            for r_idx, cand in enumerate(top5, 1):
                c_meta = cand.get("meta", {})
                c_src = normalize_str(c_meta.get("source", ""))
                c_sec = normalize_str(c_meta.get("section", ""))
                c_text = cand.get("text", "")

                if not src_hit and (norm_src in c_src or c_src in norm_src):
                    src_hit = True
                    src_rr = 1.0 / r_idx
                if not sec_hit and norm_sec and c_sec and (norm_sec in c_sec or c_sec in norm_sec):
                    sec_hit = True
                    sec_rr = 1.0 / r_idx
                if not key_hit and norm_key and norm_key in normalize_str(c_text):
                    key_hit = True

            results[v]["src_hit5"].append(1.0 if src_hit else 0.0)
            results[v]["sec_hit5"].append(1.0 if sec_hit else 0.0)
            results[v]["key_hit5"].append(1.0 if key_hit else 0.0)
            results[v]["src_rrs"].append(src_rr)
            results[v]["sec_rrs"].append(sec_rr)
            results[v]["key_ranks"].append(key_rank)

            q_info["variants"][v] = {
                "key_hit5": key_hit,
                "key_rank": key_rank,
                "source_hit5": src_hit,
                "section_hit5": sec_hit,
                "top5": [
                    {
                        "rank": r_idx,
                        "source": cand.get("meta", {}).get("source", ""),
                        "section": cand.get("meta", {}).get("section", ""),
                        "score": float(cand.get("score", 0.0)),
                        "preview": cand.get("text", "")[:120].replace("\n", " ").strip()
                    }
                    for r_idx, cand in enumerate(top5, 1)
                ]
            }

        per_query_details.append(q_info)

    summary = {
        "dataset": dataset_name,
        "count": len(questions),
        "key_phrase_hit15": float(np.mean(key_hit15_list)),
        "variants": {}
    }

    for v in variants:
        summary["variants"][v] = {
            "source_hit5": float(np.mean(results[v]["src_hit5"])),
            "section_hit5": float(np.mean(results[v]["sec_hit5"])),
            "key_phrase_hit5": float(np.mean(results[v]["key_hit5"])),
            "key_phrase_hit15": float(np.mean(key_hit15_list)),
            "source_mrr": float(np.mean(results[v]["src_rrs"])),
            "section_mrr": float(np.mean(results[v]["sec_rrs"])),
        }

    return summary, per_query_details

# --- LATENCY MEASUREMENT (5 discard passes, 10 measured in same process) ---
print("\nMeasuring warm latency per variant (5 discard passes, 10 measured)...")
TEST_QUERY = "How do you configure an NGINX Ingress controller with TLS termination in Kubernetes?"
dense_v = embed_query(TEST_QUERY)
sparse_v = embed_sparse_query(TEST_QUERY)
s_idx = sparse_v.indices if isinstance(sparse_v.indices, list) else sparse_v.indices.tolist()
s_val = sparse_v.values if isinstance(sparse_v.values, list) else sparse_v.values.tolist()

resp_test = client.query_points(
    collection_name=COLLECTION,
    prefetch=[
        models.Prefetch(query=dense_v, using="dense", limit=15),
        models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=15,
    with_payload=True
)
test_passages = [{"id": i, "text": p.payload.get("text", "")} for i, p in enumerate(resp_test.points)]

# Discard 5 passes
for _ in range(5):
    for v in variants:
        _ = rerank_passages(v, TEST_QUERY, test_passages)

variant_latencies = {v: [] for v in variants}
for _ in range(10):
    for v in variants:
        t0 = time.perf_counter()
        _ = rerank_passages(v, TEST_QUERY, test_passages)
        dur = (time.perf_counter() - t0) * 1000.0
        variant_latencies[v].append(dur)

latency_summary = {
    v: {
        "warm_median_ms": float(np.median(variant_latencies[v])),
        "warm_p95_ms": float(np.percentile(variant_latencies[v], 95)),
        "min_ms": float(np.min(variant_latencies[v])),
        "max_ms": float(np.max(variant_latencies[v]))
    }
    for v in variants
}

# --- EVALUATE DATASETS ---
print("Evaluating Golden 15 dataset across all 4 variants...")
golden_summary, golden_details = run_evaluation("Golden 15", golden_questions, is_golden=True)

print("Evaluating Hard 22 dataset across all 4 variants...")
hard_summary, hard_details = run_evaluation("Hard 22", hard_questions, is_golden=False)

# Add latency into summary
for v in variants:
    golden_summary["variants"][v]["warm_median_latency_ms"] = latency_summary[v]["warm_median_ms"]
    hard_summary["variants"][v]["warm_median_latency_ms"] = latency_summary[v]["warm_median_ms"]

# Find differences in key_phrase_hit@5
diff_queries = []
for d_type, d_list in [("golden", golden_details), ("hard", hard_details)]:
    for q in d_list:
        scores = [q["variants"][v]["key_hit5"] for v in variants]
        if len(set(scores)) > 1:
            diff_queries.append({
                "dataset": d_type,
                "id": q["id"],
                "question": q["question"],
                "key_term": q["key_term"],
                "ranks": {v: q["variants"][v]["key_rank"] for v in variants},
                "hits": {v: q["variants"][v]["key_hit5"] for v in variants}
            })

# Find hard Q3, Q5 and all misses on key-phrase@5
miss_queries = []
for q in hard_details:
    # Check if any variant missed key-phrase@5
    any_miss = any(not q["variants"][v]["key_hit5"] for v in variants)
    if any_miss or q["id"] in [3, 5]:
        miss_queries.append(q)

out_lines = []
out_lines.append("================================================================================")
out_lines.append("TASK A: RERANKER ABLATION BENCHMARK (FULL ENTERPRISE_RAG_V2 HYBRID CANDIDATE POOL = 15)")
out_lines.append("================================================================================")
out_lines.append(f"Tested Variants: {', '.join(variants)}")
out_lines.append(f"Offline Cross-Encoder Status: {'BAAI/bge-reranker-base loaded successfully offline' if bge_available else 'bge-reranker-base offline load failed; skipped'}")
out_lines.append("")

out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("1. SUMMARY METRICS TABLE (SEPARATE ROWS FOR GOLDEN AND HARD)")
out_lines.append("--------------------------------------------------------------------------------")
header = f"{'Dataset / Variant':<32} | {'Src Hit@5':<10} | {'Sec Hit@5':<10} | {'Key Hit@5':<10} | {'Key Hit@15':<11} | {'Src MRR':<9} | {'Warm Median Lat (ms)'}"
out_lines.append(header)
out_lines.append("-" * len(header))

for v in variants:
    g = golden_summary["variants"][v]
    label = f"Golden 15: {v}"
    out_lines.append(f"{label:<32} | {g['source_hit5']:<10.4f} | {g['section_hit5']:<10.4f} | {g['key_phrase_hit5']:<10.4f} | {g['key_phrase_hit15']:<11.4f} | {g['source_mrr']:<9.4f} | {g['warm_median_latency_ms']:<10.2f}")

out_lines.append("-" * len(header))

for v in variants:
    h = hard_summary["variants"][v]
    label = f"Hard 22: {v}"
    out_lines.append(f"{label:<32} | {h['source_hit5']:<10.4f} | {h['section_hit5']:<10.4f} | {h['key_phrase_hit5']:<10.4f} | {h['key_phrase_hit15']:<11.4f} | {h['source_mrr']:<9.4f} | {h['warm_median_latency_ms']:<10.2f}")

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. QUERIES WHERE VARIANTS DIFFER IN KEY-PHRASE HIT@5")
out_lines.append("--------------------------------------------------------------------------------")
if diff_queries:
    for d in diff_queries:
        out_lines.append(f"Dataset: {d['dataset'].upper()} | Query ID: {d['id']}")
        out_lines.append(f"  Question: '{d['question']}'")
        out_lines.append(f"  Key Term: '{d['key_term']}'")
        for v in variants:
            r = d["ranks"][v]
            h = d["hits"][v]
            r_str = f"Rank {r}" if r is not None else "Not in top 15"
            out_lines.append(f"    - {v:<18}: Key Hit@5 = {h!s:<5} (Chunk Rank: {r_str})")
else:
    out_lines.append("No queries where the tested variants differed in key-phrase hit@5.")

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. DETAILED BREAKDOWN FOR HARD Q3, Q5 AND ALL KEY-PHRASE@5 MISSES")
out_lines.append("--------------------------------------------------------------------------------")
for q in miss_queries:
    out_lines.append(f"\nHard Query Q{q['id']}: '{q['question']}'")
    out_lines.append(f"  Expected Source:  {q['expected_source']}")
    out_lines.append(f"  Expected Section: {q['expected_section']}")
    out_lines.append(f"  Key Phrase:       '{q['key_term']}'")
    out_lines.append(f"  Present in Top-15 Pool: {q['key_hit15']}")
    for v in variants:
        k_rank = q["variants"][v]["key_rank"]
        k_rank_str = f"Rank {k_rank}" if k_rank is not None else "Absent from top 15"
        out_lines.append(f"  >> Variant: {v} (Key Hit@5: {q['variants'][v]['key_hit5']}, Key Chunk Rank: {k_rank_str})")
        out_lines.append("     Top-5 Chunks:")
        for cand in q["variants"][v]["top5"]:
            out_lines.append(f"       Rank {cand['rank']}: [{cand['source']} -> {cand['section']}] (Score: {cand['score']:.4f})")
            out_lines.append(f"         Preview: {cand['preview']}")

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("4. COMPARATIVE CONCLUSION & VERDICT")
out_lines.append("--------------------------------------------------------------------------------")
best_golden_key = max(variants, key=lambda v: golden_summary["variants"][v]["key_phrase_hit5"])
best_hard_key = max(variants, key=lambda v: hard_summary["variants"][v]["key_phrase_hit5"])
out_lines.append(f"- Best variant on Golden 15 Key-Phrase Hit@5: {best_golden_key} ({golden_summary['variants'][best_golden_key]['key_phrase_hit5']:.4f})")
out_lines.append(f"- Best variant on Hard 22 Key-Phrase Hit@5:   {best_hard_key} ({hard_summary['variants'][best_hard_key]['key_phrase_hit5']:.4f})")

no_rerank_hard = hard_summary["variants"]["no_reranker"]["key_phrase_hit5"]
best_hard_val = hard_summary["variants"][best_hard_key]["key_phrase_hit5"]

if best_hard_val > no_rerank_hard:
    out_lines.append(f"- Does {best_hard_key} beat 'no reranker' on Hard 22 key-phrase hit@5? YES ({best_hard_val:.4f} vs {no_rerank_hard:.4f}).")
elif best_hard_val == no_rerank_hard:
    out_lines.append(f"- Does any reranker beat 'no reranker' on Hard 22 key-phrase hit@5? TIED / NO ({best_hard_val:.4f} vs {no_rerank_hard:.4f}). RRF order matches reranker key-phrase hit@5.")
else:
    out_lines.append(f"- Does any reranker beat 'no reranker' on Hard 22 key-phrase hit@5? NO ('no reranker' achieved {no_rerank_hard:.4f} vs reranker {best_hard_val:.4f}). Plainly stated: no reranker beats RRF order.")

full_text = "\n".join(out_lines) + "\n"

# Save text log
os.makedirs(os.path.dirname(TXT_OUT), exist_ok=True)
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(full_text)

# Save JSON
json_payload = {
    "script": "scripts/run_reranker_ablation.py",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "collection": COLLECTION,
    "bge_available": bge_available,
    "latency_benchmark": latency_summary,
    "golden_summary": golden_summary,
    "hard_summary": hard_summary,
    "differences_key_hit5": diff_queries,
    "miss_queries_details": miss_queries
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)

print(full_text)
