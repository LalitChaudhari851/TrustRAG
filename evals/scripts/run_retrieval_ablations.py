"""
TrustRAG Offline Dev Ablation Study.

Evaluates retrieval ablations on the 30-query dev split to test hypotheses:
  1. Current Baseline: limit=15, top_n=5, floor=0.08, margin=0.20
  2. Pure Rerank (No Thresholds): limit=15, top_n=5, floor=0.0, margin=1.0
  3. Wider Margin (0.40): limit=15, top_n=5, floor=0.08, margin=0.40
  4. Wider Margin (0.65): limit=15, top_n=5, floor=0.08, margin=0.65
  5. Expanded Pool (limit=30) + Margin (0.40): limit=30, top_n=5, floor=0.08, margin=0.40
  6. Expanded Pool (limit=30) + Margin (0.65): limit=30, top_n=5, floor=0.08, margin=0.65

Zero live LLM inference calls; strictly offline execution on dev split.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from flashrank import Ranker, RerankRequest
from evals.retrieval_dataset_schema import load_dataset
from evals.retrieval_metrics import (
    evaluate_single_query,
    aggregate_retrieval_metrics,
)

OUTPUT_ABLATION_JSON = "evals/results/retrieval_audit_experiments.json"


def run_ablation_experiments():
    records = load_dataset("evals/data/retrieval_golden_seed_v1.json")
    dev_records = [r for r in records if r.split == "dev"]
    print(f"Loaded {len(dev_records)} dev records for offline ablation study.")

    ranker = Ranker()

    ablation_configs = [
        ("current_production", 15, 5, 0.08, 0.20, "Current Production Path (limit=15, top_n=5, floor=0.08, margin=0.20)"),
        ("no_thresholds", 15, 5, 0.00, 1.00, "FlashRank Without Thresholds (limit=15, top_n=5, floor=0.00, margin=1.00)"),
        ("wider_margin_40", 15, 5, 0.08, 0.40, "Wider Relative Margin 40% (limit=15, top_n=5, floor=0.08, margin=0.40)"),
        ("wider_margin_65", 15, 5, 0.08, 0.65, "Wider Relative Margin 65% (limit=15, top_n=5, floor=0.08, margin=0.65)"),
        ("pool30_margin_40", 30, 5, 0.08, 0.40, "Expanded Candidate Pool (limit=30, top_n=5, floor=0.08, margin=0.40)"),
        ("pool30_margin_65", 30, 5, 0.08, 0.65, "Expanded Pool + Wide Margin (limit=30, top_n=5, floor=0.08, margin=0.65)"),
    ]

    # Pre-fetch raw candidates for limit=15 and limit=30 so search is identical across configurations
    print("Pre-fetching first-stage search results for all 30 dev queries (limit=30)...")
    raw_cache_30: Dict[str, List[Dict[str, Any]]] = {}
    latencies_embed_qdrant: Dict[str, Dict[str, float]] = {}

    for r in dev_records:
        t0 = time.perf_counter()
        res = search_enterprise_knowledge(r.query, limit=30, hybrid=True)
        tot_ms = (time.perf_counter() - t0) * 1000.0
        raw_cache_30[r.id] = res
        lat_stage = res[0].get("latency_stage", {}) if res else {}
        latencies_embed_qdrant[r.id] = {
            "embed_ms": lat_stage.get("embed_ms", 0.0),
            "qdrant_ms": lat_stage.get("qdrant_ms", 0.0),
            "total_ms": tot_ms,
        }

    results: Dict[str, Any] = {}

    for name, limit_k, top_n, floor, margin, desc in ablation_configs:
        print(f"\n--- Running Ablation: {name} ---")
        query_evals = []
        rerank_times = []

        for r in dev_records:
            raw_cands = raw_cache_30[r.id][:limit_k]
            t_r0 = time.perf_counter()

            if not raw_cands:
                final_docs = []
            else:
                passages = [{"id": i, "text": d["content"]} for i, d in enumerate(raw_cands)]
                req = RerankRequest(query=r.query, passages=passages)
                f_res = ranker.rerank(req)

                if not f_res:
                    final_docs = []
                else:
                    top_score = float(f_res[0].get("score", 0.0))
                    if top_score < floor:
                        final_docs = []
                    else:
                        rel_cutoff = max(floor, top_score * (1.0 - margin))
                        final_docs = []
                        for item in f_res:
                            s = float(item.get("score", 0.0))
                            if s < rel_cutoff:
                                continue
                            doc = dict(raw_cands[int(item["id"])])
                            doc["rerank_score"] = s
                            final_docs.append(doc)
                            if len(final_docs) >= top_n:
                                break

            rerank_ms = (time.perf_counter() - t_r0) * 1000.0
            rerank_times.append(rerank_ms)

            q_eval = evaluate_single_query(
                query_id=r.id,
                query_type=r.query_type,
                retrieved_docs=final_docs,
                candidate_chunk_ids=r.candidate_chunk_ids,
                relevance_grades=r.relevance_grades,
                sufficient_evidence_sets=r.sufficient_evidence_sets,
                score_floor=floor if floor > 0 else 0.08,
            )
            q_eval["latencies"] = {
                "rerank_ms": round(rerank_ms, 2),
                "embed_ms": round(latencies_embed_qdrant[r.id]["embed_ms"], 2),
                "qdrant_ms": round(latencies_embed_qdrant[r.id]["qdrant_ms"], 2),
            }
            query_evals.append(q_eval)

        summary = aggregate_retrieval_metrics(query_evals, n_resamples=1000)
        avg_rerank_ms = sum(rerank_times) / len(rerank_times)
        avg_embed_ms = sum(latencies_embed_qdrant[r.id]["embed_ms"] for r in dev_records) / len(dev_records)
        avg_qdrant_ms = sum(latencies_embed_qdrant[r.id]["qdrant_ms"] for r in dev_records) / len(dev_records)

        results[name] = {
            "name": name,
            "description": desc,
            "config": {
                "first_stage_limit": limit_k,
                "top_n": top_n,
                "floor": floor,
                "margin": margin,
            },
            "latencies_avg_ms": {
                "embed_ms": round(avg_embed_ms, 2),
                "qdrant_ms": round(avg_qdrant_ms, 2),
                "rerank_ms": round(avg_rerank_ms, 2),
                "total_pipeline_ms": round(avg_embed_ms + avg_qdrant_ms + avg_rerank_ms, 2),
            },
            "summary": summary,
            "query_evals": query_evals,
        }

    os.makedirs(os.path.dirname(OUTPUT_ABLATION_JSON), exist_ok=True)
    with open(OUTPUT_ABLATION_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dataset": "evals/data/retrieval_golden_seed_v1.json (dev split, N=30)",
                "ablations": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[OK] Wrote ablation experiment results to {OUTPUT_ABLATION_JSON}")
    return results


if __name__ == "__main__":
    run_ablation_experiments()
