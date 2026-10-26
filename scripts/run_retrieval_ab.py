import os
import sys
import time
import json
import statistics
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query

RESULTS_FILE = "evals/results/retrieval_ab_full.json"

GOLDEN_KEY_TERMS = {
    1: "redis-pod.yaml",
    2: "parallelism: 2",
    3: "rpush job2",
    4: "adjusting the number of running pods",
    5: "components.yaml",
    6: "kubectl top nodes",
    7: "databricks jobs get",
    8: "language is not currently available",
    9: "Databricks CLI, the Databricks SDKs, and the REST API",
    10: "OnFailure",
    11: "completions",
    12: "batch/v1",
    13: "minikube start",
    14: "COMPLETIONS   DURATION   AGE",
    15: "kubectl get pods -n kube-system"
}


def get_ranker():
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
    os.makedirs(cache_dir, exist_ok=True)
    return Ranker(cache_dir=cache_dir)


def normalize_str(s: Optional[str]) -> str:
    if not s:
        return ""
    import re
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def run_single_retrieval(
    client: QdrantClient,
    ranker: Ranker,
    query: str,
    collection_name: str,
    use_hybrid: bool,
    filter_true_only: bool,
    apply_threshold: bool = False,
    floor: float = 0.08,
    margin: float = 0.20,
    top_k: int = 15,
    top_n: int = 5,
    num_passes: int = 3
) -> Dict[str, Any]:
    embed_passes = []
    qdrant_passes = []
    rerank_passes = []
    total_passes = []

    last_retrieved_points = []
    last_reranked = []
    last_top_candidates = []
    last_passed_chunks = []
    last_is_refused = False

    q_filter = None
    if filter_true_only:
        q_filter = models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="true"))]
        )

    for p_idx in range(num_passes):
        t0 = time.perf_counter()
        dense_vec = embed_query(query)
        sparse_vec = embed_sparse_query(query) if use_hybrid else None
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000

        # Qdrant Search
        t2 = time.perf_counter()
        retrieved_points = []
        for attempt in range(3):
            try:
                if use_hybrid:
                    sparse_indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
                    sparse_values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()
                    prefetch = [
                        models.Prefetch(
                            query=dense_vec,
                            using="dense",
                            limit=top_k,
                            filter=q_filter
                        ),
                        models.Prefetch(
                            query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                            using="sparse",
                            limit=top_k,
                            filter=q_filter
                        ),
                    ]
                    points_result = client.query_points(
                        collection_name=collection_name,
                        prefetch=prefetch,
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=top_k,
                        with_payload=True
                    )
                    retrieved_points = points_result.points
                else:
                    if collection_name == "enterprise_rag":
                        res_pts = client.query_points(
                            collection_name=collection_name,
                            query=dense_vec,
                            query_filter=q_filter,
                            limit=top_k,
                            with_payload=True
                        )
                        retrieved_points = res_pts.points
                    else:
                        res_pts = client.query_points(
                            collection_name=collection_name,
                            query=dense_vec,
                            using="dense",
                            query_filter=q_filter,
                            limit=top_k,
                            with_payload=True
                        )
                        retrieved_points = res_pts.points
                break
            except Exception as query_err:
                if attempt == 2:
                    raise query_err
                time.sleep(1.5)
        t3 = time.perf_counter()
        qdrant_ms = (t3 - t2) * 1000

        # FlashRank Rerank
        t4 = time.perf_counter()
        passages = []
        for idx, p in enumerate(retrieved_points):
            payload = p.payload or {}
            passages.append({
                "id": idx,
                "text": payload.get("text", ""),
                "meta": {
                    "source": payload.get("source", ""),
                    "section": payload.get("section", ""),
                    "source_type": payload.get("source_type", ""),
                    "page": payload.get("page"),
                    "slide": payload.get("slide")
                }
            })

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(rerank_req)
        t5 = time.perf_counter()
        rerank_ms = (t5 - t4) * 1000
        total_ms = (t5 - t0) * 1000

        embed_passes.append(embed_ms)
        qdrant_passes.append(qdrant_ms)
        rerank_passes.append(rerank_ms)
        total_passes.append(total_ms)

        if p_idx == num_passes - 1:
            last_retrieved_points = retrieved_points
            last_reranked = reranked

            # Dynamic Threshold
            is_refused = False
            passed_chunks = []
            if apply_threshold:
                if not reranked:
                    is_refused = True
                else:
                    top_score = reranked[0].get("score", 0.0)
                    if top_score < floor:
                        is_refused = True
                    else:
                        cutoff = max(floor, top_score * (1.0 - margin))
                        passed_chunks = [c for c in reranked[:top_n] if c.get("score", 0.0) >= cutoff]
                        if not passed_chunks:
                            is_refused = True
                top_candidates = passed_chunks
            else:
                top_candidates = reranked[:top_n]
                passed_chunks = top_candidates

            last_top_candidates = top_candidates
            last_passed_chunks = passed_chunks
            last_is_refused = is_refused

    pre_rerank_sources = [p.payload.get("source", "") for p in last_retrieved_points if p.payload]

    return {
        "pre_rerank_sources": pre_rerank_sources,
        "reranked_chunks": last_reranked,
        "top_candidates": last_top_candidates,
        "passed_chunks_count": len(last_passed_chunks),
        "is_refused": last_is_refused,
        "latency": {
            "embed_ms": round(statistics.median(embed_passes), 2),
            "qdrant_ms": round(statistics.median(qdrant_passes), 2),
            "rerank_ms": round(statistics.median(rerank_passes), 2),
            "total_ms": round(statistics.median(total_passes), 2)
        }
    }


def evaluate_dataset(
    dataset_name: str,
    questions: List[Dict[str, Any]],
    client: QdrantClient,
    ranker: Ranker,
    modes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"EVALUATING DATASET: {dataset_name} ({len(questions)} queries)")
    print(f"==================================================")

    results_by_mode = {}

    for mode in modes:
        m_name = mode["name"]
        coll = mode["collection"]
        use_hybrid = mode["hybrid"]
        filter_true = mode.get("filter_true", False)
        apply_thresh = mode.get("apply_threshold", False)

        print(f"\nEvaluating Mode: {m_name} (collection={coll}, hybrid={use_hybrid}, filter_true={filter_true}, threshold={apply_thresh})...")

        pre_recalls_15 = []
        source_hits_5 = []
        section_hits_5 = []
        source_rrs = []
        section_rrs = []
        key_hits_5 = []
        key_hits_15 = []

        chunks_passed_counts = []
        refusals_count = 0

        embed_lats = []
        qdrant_lats = []
        rerank_lats = []
        total_lats = []

        query_details = []

        for q_item in questions:
            qid = q_item["id"]
            q_text = q_item["question"]
            exp_src = q_item.get("expected_source", "")
            exp_sec = q_item.get("expected_section", "")

            res = run_single_retrieval(
                client=client,
                ranker=ranker,
                query=q_text,
                collection_name=coll,
                use_hybrid=use_hybrid,
                filter_true_only=filter_true,
                apply_threshold=apply_thresh,
                floor=0.08,
                margin=0.20,
                num_passes=3
            )

            pre_srcs = res["pre_rerank_sources"]
            top_cands = res["top_candidates"]
            lat = res["latency"]

            embed_lats.append(lat["embed_ms"])
            qdrant_lats.append(lat["qdrant_ms"])
            rerank_lats.append(lat["rerank_ms"])
            total_lats.append(lat["total_ms"])

            chunks_passed_counts.append(res["passed_chunks_count"])
            if res["is_refused"]:
                refusals_count += 1

            # 1. Source Hit@15 before rerank (binary match on source document; historically labeled Recall@15)
            norm_exp_src = normalize_str(exp_src)
            src_in_top15 = any(norm_exp_src in normalize_str(s) or normalize_str(s) in norm_exp_src for s in pre_srcs)
            pre_recalls_15.append(1.0 if src_in_top15 else 0.0)

            # 2. Hit@5 & MRR after rerank (source and section)
            src_hit = False
            sec_hit = False
            src_rr = 0.0
            sec_rr = 0.0

            norm_exp_sec = normalize_str(exp_sec)

            for rank_idx, cand in enumerate(top_cands, 1):
                meta = cand.get("meta", {})
                cand_src = normalize_str(meta.get("source", ""))
                cand_sec = normalize_str(meta.get("section", ""))

                # Check source match
                if not src_hit and (norm_exp_src in cand_src or cand_src in norm_exp_src):
                    src_hit = True
                    src_rr = 1.0 / rank_idx

                # Check section match (Empty section strings MUST NOT match)
                if norm_exp_sec and cand_sec and not sec_hit:
                    if (norm_exp_sec in cand_sec) or (cand_sec in norm_exp_sec):
                        sec_hit = True
                        sec_rr = 1.0 / rank_idx

            source_hits_5.append(1.0 if src_hit else 0.0)
            if coll == "enterprise_rag":
                # Legacy collection has no section metadata in payloads -> n/a
                section_hits_5.append(None)
                section_rrs.append(None)
            else:
                section_hits_5.append(1.0 if sec_hit else 0.0)
                section_rrs.append(sec_rr)

            source_rrs.append(src_rr)

            # 3. Key phrase Hit@5 and Hit@15 (ground-truth answer term in chunk text)
            key_term = q_item.get("key_term") or GOLDEN_KEY_TERMS.get(qid, "")
            norm_key = key_term.lower().strip() if key_term else ""

            cand_5_texts = [c.get("text", "").lower() for c in top_cands[:5]]
            cand_15_texts = [c.get("text", "").lower() for c in res["reranked_chunks"][:15]]

            k_hit5 = any(norm_key in t for t in cand_5_texts) if norm_key else False
            k_hit15 = any(norm_key in t for t in cand_15_texts) if norm_key else False

            key_hits_5.append(1.0 if k_hit5 else 0.0)
            key_hits_15.append(1.0 if k_hit15 else 0.0)

            query_details.append({
                "id": qid,
                "question": q_text,
                "expected_source": exp_src,
                "expected_section": exp_sec,
                "key_term": key_term,
                "recall_15_before": src_in_top15,
                "source_hit_5": src_hit,
                "section_hit_5": sec_hit if coll != "enterprise_rag" else "n/a",
                "key_phrase_hit_5": k_hit5,
                "key_phrase_hit_15": k_hit15,
                "source_rr": round(src_rr, 3),
                "section_rr": round(sec_rr, 3) if coll != "enterprise_rag" else "n/a",
                "is_refused": res["is_refused"],
                "chunks_passed": res["passed_chunks_count"],
                "top_5_sources": [c.get("meta", {}).get("source", "") for c in top_cands[:5]],
                "top_5_sections": [c.get("meta", {}).get("section", "") for c in top_cands[:5]],
                "top_score": round(res["reranked_chunks"][0].get("score", 0.0), 4) if res["reranked_chunks"] else 0.0
            })

        mean_rec15 = sum(pre_recalls_15) / len(pre_recalls_15)
        mean_src_hit5 = sum(source_hits_5) / len(source_hits_5)
        mean_src_mrr = sum(source_rrs) / len(source_rrs)

        if coll == "enterprise_rag":
            sec_hit5_metric = "n/a"
            sec_mrr_metric = "n/a"
        else:
            valid_sec_hits = [s for s in section_hits_5 if s is not None]
            valid_sec_rrs = [s for s in section_rrs if s is not None]
            sec_hit5_metric = round(sum(valid_sec_hits) / len(valid_sec_hits), 4) if valid_sec_hits else 0.0
            sec_mrr_metric = round(sum(valid_sec_rrs) / len(valid_sec_rrs), 4) if valid_sec_rrs else 0.0

        mean_key_hit5 = sum(key_hits_5) / len(key_hits_5)
        mean_key_hit15 = sum(key_hits_15) / len(key_hits_15)

        results_by_mode[m_name] = {
            "mode": m_name,
            "collection": coll,
            "hybrid": use_hybrid,
            "filter_true": filter_true,
            "apply_threshold": apply_thresh,
            "queries_count": len(questions),
            "metrics": {
                "source_hit_15_before_rerank": round(mean_rec15, 4),
                "recall_15_before_rerank": round(mean_rec15, 4),  # legacy key alias
                "source_hit_5_after_rerank": round(mean_src_hit5, 4),
                "section_hit_5_after_rerank": sec_hit5_metric,
                "source_mrr_after_rerank": round(mean_src_mrr, 4),
                "section_mrr_after_rerank": sec_mrr_metric,
                "key_phrase_hit_5": round(mean_key_hit5, 4),
                "key_phrase_hit_15": round(mean_key_hit15, 4)
            },
            "threshold_stats": {
                "refusals_count": refusals_count,
                "refusal_rate": round(refusals_count / len(questions), 4),
                "mean_chunks_passed": round(statistics.mean(chunks_passed_counts), 2),
                "median_chunks_passed": statistics.median(chunks_passed_counts)
            },
            "latency": {
                "embed_median_ms": round(statistics.median(embed_lats), 1),
                "qdrant_median_ms": round(statistics.median(qdrant_lats), 1),
                "rerank_median_ms": round(statistics.median(rerank_lats), 1),
                "total_median_ms": round(statistics.median(total_lats), 1)
            },
            "queries": query_details
        }

        print(f"  Results for {m_name}:")
        print(f"    Source Hit@15 (pre): {mean_rec15:.4f}")
        print(f"    Source Hit@5:      {mean_src_hit5:.4f} | Source MRR:  {mean_src_mrr:.4f}")
        print(f"    Section Hit@5:     {sec_hit5_metric} | Section MRR: {sec_mrr_metric}")
        print(f"    Key Phrase Hit@5:  {mean_key_hit5:.4f} | Hit@15: {mean_key_hit15:.4f}")
        print(f"    Refusals:          {refusals_count}/{len(questions)} | Mean Chunks Passed: {statistics.mean(chunks_passed_counts):.2f}")
        print(f"    Latency (Warm Median): Total: {statistics.median(total_lats):.1f}ms (Embed: {statistics.median(embed_lats):.1f}ms, Qdrant: {statistics.median(qdrant_lats):.1f}ms, Rerank: {statistics.median(rerank_lats):.1f}ms)")

    return results_by_mode


def find_dense_hybrid_differences(dense_res: Dict[str, Any], hybrid_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    diffs = []
    dense_queries = {q["id"]: q for q in dense_res["queries"]}
    hybrid_queries = {q["id"]: q for q in hybrid_res["queries"]}

    for qid, dq in dense_queries.items():
        hq = hybrid_queries.get(qid)
        if not hq:
            continue
        d_top_pairs = list(zip(dq["top_5_sources"], dq["top_5_sections"]))
        h_top_pairs = list(zip(hq["top_5_sources"], hq["top_5_sections"]))

        if d_top_pairs != h_top_pairs:
            diffs.append({
                "id": qid,
                "question": dq["question"],
                "expected_source": dq["expected_source"],
                "dense_top_5": d_top_pairs,
                "hybrid_top_5": h_top_pairs,
                "dense_source_hit": dq["source_hit_5"],
                "hybrid_source_hit": hq["source_hit_5"],
                "dense_source_rr": dq["source_rr"],
                "hybrid_source_rr": hq["source_rr"],
                "dense_key_hit_5": dq["key_phrase_hit_5"],
                "hybrid_key_hit_5": hq["key_phrase_hit_5"]
            })
    return diffs


def main():
    print("=== RETRIEVAL A/B BENCHMARK WITH ANSWER METRICS & WARM 3-PASS LATENCY ===")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)
    ranker = get_ranker()

    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_questions = golden_raw.get("rag_samples", golden_raw)

    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_questions = json.load(f)

    controlled_true_modes = [
        {"name": "legacy_true_dense", "collection": "enterprise_rag", "hybrid": False, "filter_true": True, "apply_threshold": False},
        {"name": "v2_true_dense", "collection": "enterprise_rag_v2", "hybrid": False, "filter_true": True, "apply_threshold": False},
        {"name": "v2_true_hybrid", "collection": "enterprise_rag_v2", "hybrid": True, "filter_true": True, "apply_threshold": False},
    ]

    full_v2_modes = [
        {"name": "v2_full_dense", "collection": "enterprise_rag_v2", "hybrid": False, "filter_true": False, "apply_threshold": False},
        {"name": "v2_full_hybrid", "collection": "enterprise_rag_v2", "hybrid": True, "filter_true": False, "apply_threshold": False},
        {"name": "v2_full_hybrid_with_threshold", "collection": "enterprise_rag_v2", "hybrid": True, "filter_true": False, "apply_threshold": True},
    ]

    # Pre-warm all models with multiple warmup passes
    print("Warming up embeddings (dense + sparse) and reranker...")
    for _ in range(5):
        _ = embed_query("warmup sentence query for model initialization")
        _ = embed_sparse_query("warmup sentence query for model initialization")
        _ = ranker.rerank(RerankRequest(query="warmup", passages=[{"text": "warmup text", "id": 1}]))

    print("Warmup complete. Starting benchmark passes...")

    # 1. Controlled True-Only
    golden_controlled = evaluate_dataset("Golden 15 (True-Only Controlled)", golden_questions, client, ranker, controlled_true_modes)

    # 2. Golden 15 (Full Collection)
    golden_full = evaluate_dataset("Golden 15 (Full Collection True+Noisy)", golden_questions, client, ranker, full_v2_modes)

    # 3. Hard 22 (Full Collection)
    hard_full = evaluate_dataset("Hard 22 (Full Collection True+Noisy)", hard_questions, client, ranker, full_v2_modes)

    # 4. Dense vs Hybrid Differences
    golden_diffs = find_dense_hybrid_differences(golden_full["v2_full_dense"], golden_full["v2_full_hybrid"])
    hard_diffs = find_dense_hybrid_differences(hard_full["v2_full_dense"], hard_full["v2_full_hybrid"])

    payload = {
        "script": "scripts/run_retrieval_ab.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_collection": "enterprise_rag_v2",
        "warm_latency_passes": 3,
        "golden_controlled_true_only": golden_controlled,
        "golden_full_v2": golden_full,
        "hard_full_v2": hard_full,
        "differences": {
            "golden_count": len(golden_diffs),
            "golden_diffs": golden_diffs,
            "hard_count": len(hard_diffs),
            "hard_diffs": hard_diffs
        }
    }

    def json_serializer(o):
        if hasattr(o, "__float__"):
            return float(o)
        if hasattr(o, "__int__"):
            return int(o)
        return str(o)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_serializer)

    print(f"\n✅ Retrieval A/B Evaluation complete. Saved full report to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    main()
