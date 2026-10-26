"""
TrustRAG Retrieval Evaluation Runner.

Executes side-by-side retrieval evaluations across:
  1. Hybrid RRF (First-stage Native Qdrant Fusion)
  2. Hybrid RRF + FlashRank TinyBERT Reranking (Production Path)
  3. Dense-Only (Single vector baseline)

Across two independent corpus modes:
  - Full-Corpus: Tests production noise resistance against the entire 6,068-point index (with distractors).
  - True-Only: Tests isolated retrieval against the 197 chunks from approved documents.

Outputs comprehensive JSON results and a GitHub Flavored Markdown report with
index/model manifests, bootstrap 95% confidence intervals, and explicit unverified seed warnings.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from evals.retrieval_dataset_schema import (
    RetrievalEvalRecord,
    load_dataset,
    validate_records_against_qdrant,
)
from evals.retrieval_metrics import (
    evaluate_single_query,
    aggregate_retrieval_metrics,
)

DEFAULT_DATASET_PATH = "evals/data/retrieval_golden_seed_v1.json"
DEFAULT_RESULTS_JSON = "evals/results/retrieval_eval_results.json"
DEFAULT_REPORT_MD = "evals/results/retrieval_evaluation_report.md"

TRUE_FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="source_type",
            match=models.MatchValue(value="true"),
        )
    ]
)


def get_corpus_manifest(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
    """Retrieves index and corpus configuration metadata."""
    try:
        col_info = client.get_collection(collection_name)
        total_pts = col_info.points_count
        true_cnt = client.count(collection_name=collection_name, count_filter=TRUE_FILTER, exact=True).count
    except Exception as e:
        total_pts = "Unknown"
        true_cnt = "Unknown"

    return {
        "collection_name": collection_name,
        "total_points": total_pts,
        "true_data_points": true_cnt,
        "dense_model": "sentence-transformers/all-mpnet-base-v2 (768-dim, Cosine)",
        "sparse_model": "FastEmbed BM25 (Modifier.IDF)",
        "reranker_model": "ms-marco-TinyBERT-L-2-v2 (FlashRank locally cached)",
        "reranker_thresholds": {
            "floor": settings.RERANKER_MIN_SCORE,
            "margin": settings.RERANKER_RELATIVE_MARGIN,
        },
        "chunking": "Section-aware token splitting (max 512 tokens / chunk)",
    }


def evaluate_pipeline_on_query(
    record: RetrievalEvalRecord,
    mode_name: str,
    hybrid: bool,
    apply_reranker: bool,
    query_filter: Optional[models.Filter],
    first_stage_limit: int = 15,
    rerank_top_n: int = 5,
    search_fn=search_enterprise_knowledge,
    rerank_fn=rerank_documents,
) -> Dict[str, Any]:
    """Executes retrieval and optional reranking for a single query."""
    t0 = time.perf_counter()
    raw_results = search_fn(
        query=record.query,
        limit=first_stage_limit,
        hybrid=hybrid,
        query_filter=query_filter,
    )
    first_stage_latency_ms = (time.perf_counter() - t0) * 1000.0

    rerank_latency_ms = 0.0
    if apply_reranker:
        t_rerank = time.perf_counter()
        final_docs = rerank_fn(
            query=record.query,
            documents=raw_results,
            top_n=rerank_top_n,
            min_score=settings.RERANKER_MIN_SCORE,
            relative_margin=settings.RERANKER_RELATIVE_MARGIN,
        )
        rerank_latency_ms = (time.perf_counter() - t_rerank) * 1000.0
    else:
        final_docs = raw_results[:rerank_top_n]

    # Evaluate using the metric engine
    eval_result = evaluate_single_query(
        query_id=record.id,
        query_type=record.query_type,
        retrieved_docs=final_docs,
        candidate_chunk_ids=record.candidate_chunk_ids,
        relevance_grades=record.relevance_grades,
        sufficient_evidence_sets=record.sufficient_evidence_sets,
        score_floor=settings.RERANKER_MIN_SCORE,
    )

    eval_result["pipeline_mode"] = mode_name
    eval_result["query"] = record.query
    eval_result["split"] = record.split
    eval_result["difficulty"] = record.difficulty
    eval_result["unanswerable_subtype"] = record.unanswerable_subtype
    eval_result["latencies"] = {
        "first_stage_ms": round(first_stage_latency_ms, 2),
        "rerank_ms": round(rerank_latency_ms, 2),
        "total_ms": round(first_stage_latency_ms + rerank_latency_ms, 2),
    }

    return eval_result


def run_experiment_suite(
    dataset_path: str = DEFAULT_DATASET_PATH,
    split_filter: Optional[str] = None,
    include_dense_baseline: bool = True,
    search_fn=search_enterprise_knowledge,
    rerank_fn=rerank_documents,
    client: Optional[QdrantClient] = None,
) -> Dict[str, Any]:
    """
    Executes the matrix of experiments:
      - Corpora: ['full_corpus', 'true_only_corpus']
      - Pipelines: ['hybrid_rrf_raw', 'hybrid_rrf_reranked', 'dense_only_raw' (optional)]
    """
    records = load_dataset(dataset_path)
    if split_filter:
        records = [r for r in records if r.split == split_filter]

    # Check verification status banner
    has_unverified = any(r.verification_status == "unverified_synthetic" for r in records)

    q_client = client or QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    manifest = get_corpus_manifest(q_client, settings.QDRANT_COLLECTION)

    # Validate chunk IDs before running
    qdrant_check = validate_records_against_qdrant(records, q_client, settings.QDRANT_COLLECTION)
    if not qdrant_check["all_verified"]:
        print(f"WARNING: {len(qdrant_check['missing'])} chunk UUIDs missing from Qdrant: {qdrant_check['missing']}")

    corpora_configs = [
        ("full_corpus", None, "Full Corpus (Production Realism with 5,871 Distractor Points)"),
        ("true_only_corpus", TRUE_FILTER, "True-Only Corpus (Isolated 197 True Knowledge Units)"),
    ]

    pipeline_configs = [
        ("hybrid_rrf_raw", True, False, "Hybrid RRF (Dense + Sparse Fusion, Raw Top-K)"),
        ("hybrid_rrf_reranked", True, True, "Hybrid RRF + FlashRank TinyBERT (Production Path)"),
    ]
    if include_dense_baseline:
        pipeline_configs.append(("dense_only_raw", False, False, "Dense-Only (Single MPNet Vector, Raw Top-K)"))

    experiment_results: Dict[str, Dict[str, Any]] = {}

    for corpus_id, q_filter, corpus_desc in corpora_configs:
        experiment_results[corpus_id] = {
            "description": corpus_desc,
            "pipelines": {},
        }
        for pipe_id, use_hybrid, do_rerank, pipe_desc in pipeline_configs:
            print(f"\n--- Running: Corpus='{corpus_id}' | Pipeline='{pipe_id}' ({len(records)} queries) ---")
            query_evals: List[Dict[str, Any]] = []

            for r in records:
                q_res = evaluate_pipeline_on_query(
                    record=r,
                    mode_name=pipe_id,
                    hybrid=use_hybrid,
                    apply_reranker=do_rerank,
                    query_filter=q_filter,
                    search_fn=search_fn,
                    rerank_fn=rerank_fn,
                )
                query_evals.append(q_res)

            # Aggregate
            aggregated = aggregate_retrieval_metrics(query_evals, n_resamples=1000)

            # Latency averages
            avg_first_stage_ms = float(sum(q["latencies"]["first_stage_ms"] for q in query_evals) / len(query_evals))
            avg_rerank_ms = float(sum(q["latencies"]["rerank_ms"] for q in query_evals) / len(query_evals))
            avg_total_ms = float(sum(q["latencies"]["total_ms"] for q in query_evals) / len(query_evals))

            experiment_results[corpus_id]["pipelines"][pipe_id] = {
                "pipeline_id": pipe_id,
                "description": pipe_desc,
                "latency_ms": {
                    "first_stage_avg": round(avg_first_stage_ms, 2),
                    "rerank_avg": round(avg_rerank_ms, 2),
                    "total_avg": round(avg_total_ms, 2),
                },
                "summary": aggregated,
                "per_query_results": query_evals,
            }

    run_meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_split": split_filter or "all (dev + test)",
        "total_queries": len(records),
        "has_unverified_records": has_unverified,
        "verification_banner": (
            "NOTICE: ALL EVALUATED RECORDS ARE UNVERIFIED SYNTHETIC SEED CANDIDATES. "
            "RESULTS REFLECT INITIAL SEED PERFORMANCE AND MUST NOT BE CITED AS A VERIFIED BENCHMARK."
            if has_unverified else "Verified Human Benchmark"
        ),
        "corpus_manifest": manifest,
        "experiments": experiment_results,
    }

    return run_meta


def generate_markdown_report(run_meta: Dict[str, Any]) -> str:
    """Formats experiment suite results into GitHub Flavored Markdown."""
    lines: List[str] = []
    lines.append("# TrustRAG Retrieval Evaluation Report\n")

    if run_meta.get("has_unverified_records"):
        lines.append("> [!WARNING]")
        lines.append("> **UNVERIFIED SEED DATASET NOTICE**")
        lines.append(f"> {run_meta['verification_banner']}\n")

    meta = run_meta["corpus_manifest"]
    lines.append("## 1. Evaluation & Environment Manifest\n")
    lines.append("| Manifest Property | Configuration |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Target Qdrant Collection** | `{meta['collection_name']}` |")
    lines.append(f"| **Total Indexed Points** | `{meta['total_points']}` |")
    lines.append(f"| **True Knowledge Points** | `{meta['true_data_points']}` |")
    lines.append(f"| **Dense Embedding Model** | `{meta['dense_model']}` |")
    lines.append(f"| **Sparse Embedding Model** | `{meta['sparse_model']}` |")
    lines.append(f"| **Reranker Engine** | `{meta['reranker_model']}` |")
    lines.append(f"| **Reranker Floor / Margin** | `floor={meta['reranker_thresholds']['floor']}, margin={meta['reranker_thresholds']['margin']}` |")
    lines.append(f"| **Dataset Version & Split** | `{run_meta['dataset_path']}` (Split: `{run_meta['dataset_split']}`, N={run_meta['total_queries']}) |")
    lines.append(f"| **Execution Timestamp** | `{run_meta['timestamp']}` |\n")

    experiments = run_meta["experiments"]

    for corpus_id, c_data in experiments.items():
        lines.append(f"## 2. Results: {c_data['description']}\n")
        pipelines = c_data["pipelines"]

        # Summary IR Table
        lines.append("### Relevance Macro-Averages (Answerable Queries Only)\n")
        lines.append("| Pipeline | Recall@5 | Recall@10 | Precision@5 (Fixed) | Precision@5 (Actual) | HitRate@5 | MRR@10 | Graded nDCG@10 | Avg Latency |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for pipe_id, p_data in pipelines.items():
            summary = p_data["summary"]["relevance_macro_averages"]
            lat = p_data["latency_ms"]["total_avg"]

            r5 = f"{summary['recall@5']['mean']:.3f} [{summary['recall@5']['ci_lower']:.3f}, {summary['recall@5']['ci_upper']:.3f}]"
            r10 = f"{summary['recall@10']['mean']:.3f}"
            p5_fix = f"{summary['precision@5_fixed']['mean']:.3f}"
            p5_act = f"{summary['precision@5_actual']['mean']:.3f}"
            hr5 = f"{summary['hit_rate@5']['mean']:.3f}"
            mrr10 = f"{summary['mrr@10']['mean']:.3f}"
            ndcg10 = f"{summary['ndcg@10']['mean']:.3f} [{summary['ndcg@10']['ci_lower']:.3f}, {summary['ndcg@10']['ci_upper']:.3f}]"

            lines.append(f"| **{p_data['description']}** | {r5} | {r10} | {p5_fix} | {p5_act} | {hr5} | {mrr10} | {ndcg10} | {lat}ms |")

        ans_count = next(iter(pipelines.values()))["summary"]["counts"]["answerable_queries"]
        lines.append(f"\n*Note: Bracketed ranges denote 95% bootstrap confidence intervals (1,000 resamples). Sample size N={ans_count} answerable queries.*")

        # Multi-Hop Summary Table
        sample_pipe = next(iter(pipelines.values()))
        if "multi_hop_metrics" in sample_pipe["summary"]:
            lines.append("\n### Multi-Hop Conjunction Metrics\n")
            lines.append("| Pipeline | Evidence Coverage@5 | All-Required Coverage@5 | Multi-Hop Success@5 | Evidence Coverage@10 | All-Required Coverage@10 |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

            for pipe_id, p_data in pipelines.items():
                mh = p_data["summary"].get("multi_hop_metrics", {})
                ec5 = f"{mh.get('evidence_coverage@5', {}).get('mean', 0.0):.3f}"
                ar5 = f"{mh.get('all_required_coverage@5', {}).get('mean', 0.0):.3f}"
                ms5 = f"{mh.get('multi_hop_success@5', {}).get('mean', 0.0):.3f}"
                ec10 = f"{mh.get('evidence_coverage@10', {}).get('mean', 0.0):.3f}"
                ar10 = f"{mh.get('all_required_coverage@10', {}).get('mean', 0.0):.3f}"
                lines.append(f"| **{p_data['description']}** | {ec5} | {ar5} | {ms5} | {ec10} | {ar10} |")

        # Distractor Interference Table
        lines.append("\n### Distractor Interference\n")
        lines.append("| Pipeline | Distractors Before First Relevant@5 | Distractor Ratio@5 | Distractor Ratio@10 |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for pipe_id, p_data in pipelines.items():
            dist = p_data["summary"].get("distractor_interference", {})
            d_before = f"{dist.get('distractors_before_first_relevant@5', {}).get('mean', 0.0):.2f}"
            d_ratio5 = f"{dist.get('distractor_ratio@5', {}).get('mean', 0.0):.3f}"
            d_ratio10 = f"{dist.get('distractor_ratio@10', {}).get('mean', 0.0):.3f}"
            lines.append(f"| **{p_data['description']}** | {d_before} | {d_ratio5} | {d_ratio10} |")

        # Abstention Table
        lines.append("\n### Retrieval Abstention on Unanswerable Queries\n")
        lines.append("| Pipeline | Unanswerable Queries | Correct Retrieval Abstentions | Retrieval Abstention Rate |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for pipe_id, p_data in pipelines.items():
            abst = p_data["summary"].get("unanswerable_abstention", {})
            u_cnt = abst.get("unanswerable_count", 0)
            c_abst = abst.get("correct_retrieval_abstentions", 0)
            rate = abst.get("retrieval_abstention_rate", 0.0)
            lines.append(f"| **{p_data['description']}** | {u_cnt} | {c_abst} | {rate * 100.0:.1f}% |")

        lines.append("\n---\n")

    lines.append("## 3. Methodological Notes\n")
    lines.append("1. **Dual Precision Conventions**: Precision@5 (Fixed) divides hits by 5, penalizing pipelines when fewer than 5 items are returned. Precision@5 (Actual) divides hits by the candidate count actually returned by FlashRank after score filtering.")
    lines.append("2. **Unanswerable Isolation**: Unanswerable queries ($N=10$) are isolated from relevance macro-averages to prevent $0/0$ distortion, and are evaluated under Abstention Rate.")
    lines.append("3. **Multi-Hop Conjunction**: Multi-hop queries evaluate whether *all* required evidence chunks from at least one sufficient set are retrieved in top-K, not merely whether any random chunk was hit.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustRAG Retrieval Evaluation Runner")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to evaluation dataset JSON")
    parser.add_argument("--split", choices=["dev", "test"], default=None, help="Evaluate on specific split")
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON, help="Destination for detailed JSON results")
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD, help="Destination for markdown summary report")
    parser.add_argument("--no-dense", action="store_true", help="Omit dense-only baseline")

    args = parser.parse_args()

    print("================================================================")
    print("TrustRAG Retrieval Evaluation Benchmark Suite")
    print(f"Dataset: {args.dataset} | Split: {args.split or 'all'}")
    print("================================================================")

    suite_results = run_experiment_suite(
        dataset_path=args.dataset,
        split_filter=args.split,
        include_dense_baseline=not args.no_dense,
    )

    # Save JSON results
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote full per-query JSON results to {args.results_json}")

    # Generate & Save Markdown Report
    report_md = generate_markdown_report(suite_results)
    os.makedirs(os.path.dirname(args.report_md), exist_ok=True)
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Wrote markdown report to {args.report_md}")


if __name__ == "__main__":
    main()
