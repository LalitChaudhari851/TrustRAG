"""
Controlled retrieval evaluation and threshold calibration sweep.
Produces:
  1. evals/results/controlled_comparison.json
  2. evals/results/threshold_sweep.json
"""
import sys, os, time, json, statistics
sys.path.insert(0, os.path.abspath("."))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_sparse_query

def get_ranker():
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
    os.makedirs(cache_dir, exist_ok=True)
    return Ranker(cache_dir=cache_dir)

def run_controlled_eval():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    ranker = get_ranker()
    ranker_model_name = os.path.basename(str(ranker.model_dir))
    print(f"Ranker model name: {ranker_model_name}")

    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_samples = golden_raw.get("rag_samples", golden_raw)

    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_samples = json.load(f)

    # Hard negatives (adversarial / unanswerable / out-of-domain)
    hard_negatives = [
        {"id": "HN1", "question": "What is the recommended dosage of amoxicillin for adults?"},
        {"id": "HN2", "question": "How do you implement quicksort in Rust with SIMD AVX-512?"},
        {"id": "HN3", "question": "What are the quantum gate fidelity requirements for Shor's algorithm?"},
        {"id": "HN4", "question": "Who was the Roman emperor during the eruption of Mount Vesuvius in 79 AD?"},
        {"id": "HN5", "question": "Explain how to bake a sourdough bread starter from scratch."}
    ]

    # Warm-up models
    print("Warming up models...")
    _ = embed_query("warmup query")
    _ = embed_sparse_query("warmup query")
    _ = ranker.rerank(RerankRequest(query="warmup", passages=[{"text": "warmup text", "id": 1}]))

    # Test Configurations:
    # 1. legacy_true_dense: enterprise_rag (filtered source_type == 'true')
    # 2. v2_dense: enterprise_rag_v2 (source_type == 'true', dense only)
    # 3. v2_hybrid: enterprise_rag_v2 (source_type == 'true', native hybrid)
    
    modes = [
        {"name": "legacy_true_dense", "collection": "enterprise_rag", "hybrid": False, "filter_true": True},
        {"name": "v2_true_dense", "collection": "enterprise_rag_v2", "hybrid": False, "filter_true": True},
        {"name": "v2_true_hybrid", "collection": "enterprise_rag_v2", "hybrid": True, "filter_true": True},
    ]

    mode_results = {}

    for mode in modes:
        m_name = mode["name"]
        coll = mode["collection"]
        use_hybrid = mode["hybrid"]
        print(f"\nRunning mode: {m_name} (collection={coll}, hybrid={use_hybrid})...")

        details = []
        source_hits = 0
        section_hits = 0
        source_rrs = []
        section_rrs = []

        embed_latencies = []
        qdrant_latencies = []
        rerank_latencies = []

        for item in golden_samples:
            q = item["question"]
            exp_src = item.get("expected_source", "")
            exp_sec = item.get("expected_section", "")

            # 1. Embed query
            t0 = time.perf_counter()
            dense_vec = embed_query(q)
            sparse_vec = embed_sparse_query(q) if use_hybrid else None
            t1 = time.perf_counter()
            embed_latencies.append((t1 - t0) * 1000)

            # 2. Search in Qdrant with payload filter
            t2 = time.perf_counter()
            query_filter = models.Filter(
                must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="true"))]
            )

            if use_hybrid:
                sparse_indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
                sparse_values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()
                res = client.query_points(
                    collection_name=coll,
                    prefetch=[
                        models.Prefetch(query=dense_vec, using="dense", limit=20, filter=query_filter),
                        models.Prefetch(
                            query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                            using="sparse",
                            limit=20,
                            filter=query_filter
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=15,
                    with_payload=True
                )
            else:
                # Check vector name in collection
                vector_name = "dense" if coll == "enterprise_rag_v2" else None
                if vector_name:
                    res = client.query_points(
                        collection_name=coll,
                        query=dense_vec,
                        using=vector_name,
                        limit=15,
                        query_filter=query_filter,
                        with_payload=True
                    )
                else:
                    res = client.query_points(
                        collection_name=coll,
                        query=dense_vec,
                        limit=15,
                        query_filter=query_filter,
                        with_payload=True
                    )
            t3 = time.perf_counter()
            qdrant_latencies.append((t3 - t2) * 1000)

            # 3. FlashRank Rerank
            candidates = []
            for pt in res.points:
                txt = pt.payload.get("text", "") or pt.payload.get("content", "")
                candidates.append({
                    "id": str(pt.id),
                    "text": txt,
                    "metadata": pt.payload
                })

            t4 = time.perf_counter()
            if candidates:
                passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
                rerank_req = RerankRequest(query=q, passages=passages)
                ranked_out = ranker.rerank(rerank_req)[:5]
                t5 = time.perf_counter()
                rerank_latencies.append((t5 - t4) * 1000)

                # Map back to metadata
                cand_map = {c["id"]: c["metadata"] for c in candidates}
                ranked_docs = []
                for r in ranked_out:
                    md = cand_map.get(r["id"], {})
                    ranked_docs.append({
                        "id": r["id"],
                        "score": float(r["score"]),
                        "source": md.get("source", ""),
                        "section": md.get("section", ""),
                        "text_snippet": md.get("text", "")[:120].replace("\n", " ")
                    })
            else:
                t5 = time.perf_counter()
                rerank_latencies.append((t5 - t4) * 1000)
                ranked_docs = []

            # Evaluate source and section hit/MRR
            s_rank = None
            sec_rank = None
            for idx, doc in enumerate(ranked_docs):
                d_src = doc["source"].lower()
                d_sec = doc["section"].lower()
                if exp_src.lower() in d_src and s_rank is None:
                    s_rank = idx + 1
                if exp_sec and exp_sec.lower() in d_sec and sec_rank is None:
                    sec_rank = idx + 1

            s_hit = 1.0 if (s_rank is not None and s_rank <= 5) else 0.0
            sec_hit = 1.0 if (sec_rank is not None and sec_rank <= 5) else 0.0
            s_rr = 1.0 / s_rank if s_rank else 0.0
            sec_rr = 1.0 / sec_rank if sec_rank else 0.0

            source_hits += s_hit
            section_hits += sec_hit
            source_rrs.append(s_rr)
            section_rrs.append(sec_rr)

            details.append({
                "question": q,
                "expected_source": exp_src,
                "expected_section": exp_sec,
                "source_rank": s_rank,
                "section_rank": sec_rank,
                "source_hit@5": s_hit,
                "section_hit@5": sec_hit,
                "top_score": ranked_docs[0]["score"] if ranked_docs else 0.0,
                "top_docs": ranked_docs
            })

        n_samples = len(golden_samples)
        src_hit_rate = source_hits / n_samples
        sec_hit_rate = section_hits / n_samples
        src_mrr = sum(source_rrs) / n_samples
        sec_mrr = sum(section_rrs) / n_samples

        med_embed = statistics.median(embed_latencies)
        med_qdrant = statistics.median(qdrant_latencies)
        med_rerank = statistics.median(rerank_latencies)
        med_total = med_embed + med_qdrant + med_rerank

        mode_results[m_name] = {
            "collection": coll,
            "hybrid": use_hybrid,
            "source_hit@5": src_hit_rate,
            "section_hit@5": sec_hit_rate,
            "source_mrr": src_mrr,
            "section_mrr": sec_mrr,
            "median_latencies_ms": {
                "embed_ms": round(med_embed, 2),
                "qdrant_ms": round(med_qdrant, 2),
                "rerank_ms": round(med_rerank, 2),
                "total_retrieval_ms": round(med_total, 2)
            },
            "details": details
        }
        print(f"  Source Hit@5:  {src_hit_rate:.4f} | Source MRR:  {src_mrr:.4f}")
        print(f"  Section Hit@5: {sec_hit_rate:.4f} | Section MRR: {sec_mrr:.4f}")
        print(f"  Median Latencies (ms): embed={med_embed:.1f}, qdrant={med_qdrant:.1f}, rerank={med_rerank:.1f}, total={med_total:.1f}")

    with open("evals/results/controlled_comparison.json", "w", encoding="utf-8") as f:
        json.dump(mode_results, f, indent=2)
    print("\nSaved controlled comparison to evals/results/controlled_comparison.json")

    # ==========================================
    # 2. THRESHOLD SWEEP (on enterprise_rag_v2)
    # ==========================================
    print("\n=== RUNNING DYNAMIC THRESHOLD SWEEP ===")
    floors = [0.05, 0.12, 0.20]
    margins = [0.30, 0.40, 0.50]

    # Precompute top scores for:
    # 1. 15 golden questions
    # 2. 22 hard retrieval questions
    # 3. 5 hard negative questions

    def get_query_scores(query_list):
        scores_list = []
        for item in query_list:
            q = item["question"]
            dense_vec = embed_query(q)
            sparse_vec = embed_sparse_query(q)
            sparse_indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
            sparse_values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

            res = client.query_points(
                collection_name="enterprise_rag_v2",
                prefetch=[
                    models.Prefetch(query=dense_vec, using="dense", limit=20),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=20
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=15,
                with_payload=True
            )
            passages = [{"id": str(pt.id), "text": pt.payload.get("text", "")} for pt in res.points]
            if passages:
                rerank_req = RerankRequest(query=q, passages=passages)
                ranked = ranker.rerank(rerank_req)[:5]
                top_s = float(ranked[0]["score"]) if ranked else 0.0
                chunk_scores = [float(r["score"]) for r in ranked]
            else:
                top_s = 0.0
                chunk_scores = []
            scores_list.append({"id": item.get("id"), "question": q, "top_score": top_s, "chunk_scores": chunk_scores})
        return scores_list

    print("Fetching v2 retrieval scores for 15 golden, 22 hard, and 5 hard negatives...")
    golden_scores = get_query_scores(golden_samples)
    hard_scores = get_query_scores(hard_samples)
    neg_scores = get_query_scores(hard_negatives)

    sweep_results = []

    print(f"\n{'Floor':<8} | {'Margin':<8} | {'Golden False Ref':<18} | {'Hard False Ref':<16} | {'Neg Refusal Rate':<18} | Status")
    print("-" * 85)

    for fl in floors:
        for mg in margins:
            # Policy:
            # Refuse query if top_score < fl.
            # Otherwise, retain chunks where score >= max(fl, top_score * mg).
            # A query is falsely refused if top_score < fl.
            
            golden_false_ref = sum(1 for s in golden_scores if s["top_score"] < fl)
            hard_false_ref = sum(1 for s in hard_scores if s["top_score"] < fl)
            
            # Negatives refused: query refused if top_score < fl
            neg_refused = sum(1 for s in neg_scores if s["top_score"] < fl)
            neg_refusal_rate = neg_refused / len(neg_scores) if neg_scores else 0.0

            status = "RECOMMENDED" if golden_false_ref == 0 and hard_false_ref == 0 and neg_refusal_rate >= 0.8 else "ACCEPTABLE"
            if golden_false_ref > 0 or hard_false_ref > 0:
                status = "FALSE_REFUSAL_RISK"

            row = {
                "floor": fl,
                "margin": mg,
                "golden_false_refusals": golden_false_ref,
                "golden_total": len(golden_scores),
                "hard_false_refusals": hard_false_ref,
                "hard_total": len(hard_scores),
                "negatives_refused": neg_refused,
                "negatives_total": len(neg_scores),
                "negative_refusal_rate": neg_refusal_rate,
                "status": status
            }
            sweep_results.append(row)
            print(f"{fl:<8.2f} | {mg:<8.2f} | {golden_false_ref}/{len(golden_scores):<14} | {hard_false_ref}/{len(hard_scores):<12} | {neg_refusal_rate*100:<15.1f}% | {status}")

    sweep_payload = {
        "script": "scripts/run_controlled_eval.py",
        "ranker_model": ranker_model_name,
        "floors_tested": floors,
        "margins_tested": margins,
        "sweep_grid": sweep_results,
        "query_top_scores": {
            "golden_queries": [{"id": s["id"], "score": s["top_score"], "q": s["question"][:50]} for s in golden_scores],
            "hard_queries": [{"id": s["id"], "score": s["top_score"], "q": s["question"][:50]} for s in hard_scores],
            "negative_queries": [{"id": s["id"], "score": s["top_score"], "q": s["question"][:50]} for s in neg_scores]
        }
    }

    with open("evals/results/threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_payload, f, indent=2)
    print("\nSaved threshold sweep results to evals/results/threshold_sweep.json")

if __name__ == "__main__":
    run_controlled_eval()
