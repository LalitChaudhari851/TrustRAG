"""
Detailed stage-by-stage retrieval diagnostic script for seed_008 and seed_009.

Traces candidate presence across:
1. Dense vector search (all-mpnet-base-v2)
2. Sparse lexical search (FastEmbed BM25)
3. Qdrant Native Hybrid RRF
4. FlashRank TinyBERT Cross-Encoder scoring and threshold filtering (floor + margin)
5. Mode-specific document selection (tinybert vs rrf vs fusion)
6. Final LLM prompt context & response
"""

import json
import os
import sys
import time
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query
from app.services.retrieval.qdrant_service import client
from app.services.retrieval.ranking_service import _get_ranker
from flashrank import RerankRequest
from qdrant_client.http import models

TARGET_QUESTIONS = [
    {
        "id": "seed_008",
        "query": "How can you configure a Kubernetes CronJob to automatically clean up completed jobs after 1 hour to minimize costs?",
        "ground_truth_chunks": {
            "8f7910c9-d6a1-5234-950f-ed26139ea9a3": "Primary (Grade 3) — cronjobs.docx Chunk 52 (ttlSecondsAfterFinished: 3600)"
        }
    },
    {
        "id": "seed_009",
        "query": "What is the core difference between HPA and VPA in Kubernetes?",
        "ground_truth_chunks": {
            "ac2984be-e47e-5b11-937a-46344e9fcfed": "Primary (Grade 3) — pods_autoscale.html Chunk 27 (HPA adds pods vs VPA adjusts CPU/mem)",
            "376b124c-3b4c-53e2-81d6-28ae65918ad9": "Supporting (Grade 2) — pods_autoscale.html Chunk 28",
            "c0f1220e-829c-50f0-bf0d-2e48be167355": "Peripheral (Grade 1) — pods_autoscale.html Chunk 0 (Introductory animation)"
        }
    }
]

def diagnose_question(tq: Dict[str, Any]) -> Dict[str, Any]:
    qid = tq["id"]
    query = tq["query"]
    gt_chunks = tq["ground_truth_chunks"]
    collection = settings.QDRANT_COLLECTION

    print(f"\n================================================================================")
    print(f"🔍 DIAGNOSTIC TRACE FOR {qid}")
    print(f"Query: '{query}'")
    print(f"Target Ground Truth: {list(gt_chunks.keys())}")
    print(f"================================================================================", flush=True)

    # 1. Dense Search
    dense_vec = embed_query(query)
    dense_resp = client.query_points(
        collection_name=collection,
        query=dense_vec,
        using="dense",
        limit=15,
        with_payload=True
    )
    dense_candidates = []
    for rank, pt in enumerate(dense_resp.points, start=1):
        cid = str(pt.id)
        dense_candidates.append({
            "rank": rank,
            "id": cid,
            "score": round(float(pt.score), 4),
            "source": (pt.payload or {}).get("source"),
            "section": (pt.payload or {}).get("section"),
            "is_ground_truth": cid in gt_chunks,
            "gt_label": gt_chunks.get(cid)
        })

    # 2. Sparse Search
    sparse_vec = embed_sparse_query(query)
    sparse_resp = client.query_points(
        collection_name=collection,
        query=sparse_vec,
        using="sparse",
        limit=15,
        with_payload=True
    )
    sparse_candidates = []
    for rank, pt in enumerate(sparse_resp.points, start=1):
        cid = str(pt.id)
        sparse_candidates.append({
            "rank": rank,
            "id": cid,
            "score": round(float(pt.score), 4),
            "source": (pt.payload or {}).get("source"),
            "section": (pt.payload or {}).get("section"),
            "is_ground_truth": cid in gt_chunks,
            "gt_label": gt_chunks.get(cid)
        })

    # 3. Hybrid RRF Search
    rrf_resp = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=15),
            models.Prefetch(query=sparse_vec, using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15,
        with_payload=True,
    )
    rrf_candidates = []
    raw_results_for_reranker = []
    for rank, pt in enumerate(rrf_resp.points, start=1):
        cid = str(pt.id)
        payload = pt.payload or {}
        doc_obj = {
            "id": cid,
            "score": float(pt.score),
            "content": payload.get("text", ""),
            "source": payload.get("source", "Unknown"),
            "section": payload.get("section") or payload.get("slide_title") or "General",
            "chunk_index": payload.get("chunk_index"),
        }
        raw_results_for_reranker.append(doc_obj)
        rrf_candidates.append({
            "rank": rank,
            "id": cid,
            "score": round(float(pt.score), 4),
            "source": payload.get("source"),
            "section": payload.get("section"),
            "is_ground_truth": cid in gt_chunks,
            "gt_label": gt_chunks.get(cid)
        })

    # 4. FlashRank Reranker Stage
    ranker = _get_ranker()
    passages = [{"id": i, "text": d["content"]} for i, d in enumerate(raw_results_for_reranker)]
    req = RerankRequest(query=query, passages=passages)
    rerank_results = ranker.rerank(req)

    floor = settings.RERANKER_MIN_SCORE
    margin = settings.RERANKER_RELATIVE_MARGIN
    top_score = float(rerank_results[0].get("score", 0.0)) if rerank_results else 0.0
    relative_cutoff = max(floor, top_score * (1.0 - margin))

    reranker_candidates = []
    reranked_docs = []
    for r_item in rerank_results:
        idx = int(r_item["id"])
        score = float(r_item.get("score", 0.0))
        orig_doc = dict(raw_results_for_reranker[idx])
        orig_doc["rerank_score"] = score
        cid = orig_doc["id"]

        is_kept = score >= relative_cutoff and top_score >= floor
        drop_reason = None
        if top_score < floor:
            drop_reason = f"top_score ({top_score:.4f}) < floor ({floor:.4f})"
        elif score < relative_cutoff:
            drop_reason = f"score ({score:.4f}) < relative_cutoff ({relative_cutoff:.4f})"

        reranker_candidates.append({
            "id": cid,
            "score": round(score, 4),
            "source": orig_doc.get("source"),
            "section": orig_doc.get("section"),
            "is_ground_truth": cid in gt_chunks,
            "gt_label": gt_chunks.get(cid),
            "is_kept": is_kept,
            "drop_reason": drop_reason,
        })
        if is_kept:
            reranked_docs.append(orig_doc)

    # 5. Document Selection across the 3 modes
    # Tinybert
    tinybert_docs = reranked_docs[:5]
    # RRF
    rrf_docs = raw_results_for_reranker[:5]
    # Fusion
    seen = set()
    fused = []
    for d in reranked_docs[:3]:
        k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
        if k not in seen:
            seen.add(k)
            fused.append(d)
    for d in raw_results_for_reranker[:3]:
        k = (d.get("source"), d.get("section"), str(d.get("content", ""))[:120])
        if k not in seen:
            seen.add(k)
            fused.append(d)
    fusion_docs = fused[:6]

    # Stage-by-Stage Ground Truth Presence Matrix
    presence_matrix = {}
    for cid, label in gt_chunks.items():
        dense_rank = next((c["rank"] for c in dense_candidates if c["id"] == cid), None)
        sparse_rank = next((c["rank"] for c in sparse_candidates if c["id"] == cid), None)
        rrf_rank = next((c["rank"] for c in rrf_candidates if c["id"] == cid), None)
        rerank_entry = next((c for c in reranker_candidates if c["id"] == cid), None)
        rerank_score = rerank_entry["score"] if rerank_entry else None
        rerank_kept = rerank_entry["is_kept"] if rerank_entry else False
        in_tinybert = any(d["id"] == cid for d in tinybert_docs)
        in_rrf = any(d["id"] == cid for d in rrf_docs)
        in_fusion = any(d["id"] == cid for d in fusion_docs)

        presence_matrix[cid] = {
            "label": label,
            "dense_rank": dense_rank,
            "sparse_rank": sparse_rank,
            "rrf_rank": rrf_rank,
            "rerank_score": rerank_score,
            "rerank_kept": rerank_kept,
            "in_tinybert": in_tinybert,
            "in_rrf": in_rrf,
            "in_fusion": in_fusion,
        }

    return {
        "id": qid,
        "query": query,
        "dense_candidates": dense_candidates[:5],
        "sparse_candidates": sparse_candidates[:5],
        "rrf_candidates": rrf_candidates[:5],
        "reranker_top_score": round(top_score, 4),
        "relative_cutoff": round(relative_cutoff, 4),
        "reranker_candidates": reranker_candidates[:8],
        "presence_matrix": presence_matrix,
        "tinybert_selected_ids": [d["id"] for d in tinybert_docs],
        "rrf_selected_ids": [d["id"] for d in rrf_docs],
        "fusion_selected_ids": [d["id"] for d in fusion_docs],
    }

def main():
    results = []
    for tq in TARGET_QUESTIONS:
        diag = diagnose_question(tq)
        results.append(diag)
        print(f"\n--- Ground Truth Presence Summary for {diag['id']} ---")
        for cid, pm in diag["presence_matrix"].items():
            print(f"  Chunk: {cid[:8]}... ({pm['label']})")
            print(f"    Dense Rank:    {pm['dense_rank']}")
            print(f"    Sparse Rank:   {pm['sparse_rank']}")
            print(f"    Hybrid RRF:    {pm['rrf_rank']}")
            print(f"    FlashRank:     Score={pm['rerank_score']}, Kept={pm['rerank_kept']}")
            print(f"    In TinyBERT:   {pm['in_tinybert']}")
            print(f"    In RRF:        {pm['in_rrf']}")
            print(f"    In Fusion:     {pm['in_fusion']}")

    out_file = "evals/results/diagnostic_seed008_seed009.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full diagnostic logs to {out_file}")

if __name__ == "__main__":
    main()
