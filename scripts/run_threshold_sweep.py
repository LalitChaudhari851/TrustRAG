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

RESULTS_FILE = "evals/results/threshold_sweep_full.json"
COLLECTION_NAME = "enterprise_rag_v2"


def get_ranker():
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
    os.makedirs(cache_dir, exist_ok=True)
    return Ranker(cache_dir=cache_dir)


HARD_NEGATIVES_K8S = [
    {"id": "HN_K8S_1", "question": "How do I configure an NGINX Ingress controller with TLS termination in Kubernetes?"},
    {"id": "HN_K8S_2", "question": "How do RBAC RoleBinding and ClusterRoleBinding resources restrict service account permissions?"},
    {"id": "HN_K8S_3", "question": "How does Istio service mesh manage mutual TLS (mTLS) traffic between envoy sidecars?"},
    {"id": "HN_K8S_4", "question": "How do Helm post-install and pre-delete hooks execute during chart release upgrades?"},
    {"id": "HN_K8S_5", "question": "How do Kubernetes NetworkPolicy ingress CIDR rules restrict pod-to-pod network traffic?"},
    {"id": "HN_K8S_6", "question": "How does PodDisruptionBudget ensure application availability during node drain operations?"},
    {"id": "HN_K8S_7", "question": "How do PersistentVolume dynamic provisioners interact with CSI storage drivers in Kubernetes?"},
    {"id": "HN_K8S_8", "question": "How does kube-scheduler handle node affinity, anti-affinity, and pod topology spread constraints?"},
    {"id": "HN_K8S_9", "question": "How do Kubernetes Custom Resource Definitions (CRDs) register custom controllers with the API server?"}
]

HARD_NEGATIVES_NOISY = [
    {"id": "HN_NOISY_1", "question": "How does the Linux kernel manage 5-level paging and Extended Page Tables (EPT) for virtual memory?"},
    {"id": "HN_NOISY_2", "question": "How does C++ exception stack unwinding work on Windows x64 architecture?"},
    {"id": "HN_NOISY_3", "question": "What are the core techniques in the Burrows-Wheeler block-sorting lossless data compression algorithm?"},
    {"id": "HN_NOISY_4", "question": "What are the compiler error patterns identified in student data across 37 million compilations?"},
    {"id": "HN_NOISY_5", "question": "What are the formal verification rules in the Standard Annotation Language (SAL) for buffer overflows?"},
    {"id": "HN_NOISY_6", "question": "How do 100G Ethernet optical transceivers handle SerDes framing and forward error correction?"},
    {"id": "HN_NOISY_7", "question": "How does transactional memory optimization improve concurrency in HTM-enabled dynamic data structures?"},
    {"id": "HN_NOISY_8", "question": "What are the multidimensional schema design principles for OLAP star and snowflake queries?"},
    {"id": "HN_NOISY_9", "question": "How do feedforward neural networks compute backpropagation error gradients with sigmoid activation?"}
]


def retrieve_and_rerank(client: QdrantClient, ranker: Ranker, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    dense_vec = embed_query(query)
    sparse_vec = embed_sparse_query(query)

    sparse_indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
    sparse_values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense", limit=top_k),
        models.Prefetch(query=models.SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=top_k),
    ]

    points_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True
    )
    retrieved_points = points_result.points

    passages = []
    for idx, p in enumerate(retrieved_points):
        payload = p.payload or {}
        passages.append({
            "id": idx,
            "text": payload.get("text", ""),
            "meta": {
                "source": payload.get("source", ""),
                "section": payload.get("section", ""),
                "source_type": payload.get("source_type", "")
            }
        })

    rerank_req = RerankRequest(query=query, passages=passages)
    return ranker.rerank(rerank_req)


def compute_distribution(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p10": 0, "p90": 0}
    s = sorted(scores)
    n = len(s)
    p10_idx = int(n * 0.10)
    p90_idx = min(int(n * 0.90), n - 1)
    return {
        "min": round(min(s), 4),
        "max": round(max(s), 4),
        "mean": round(statistics.mean(s), 4),
        "median": round(statistics.median(s), 4),
        "p10": round(s[p10_idx], 4),
        "p90": round(s[p90_idx], 4)
    }


def main():
    print(f"=== STEP 3: THRESHOLD CALIBRATION SWEEP ON '{COLLECTION_NAME}' ===")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    ranker = get_ranker()

    ranker_model_name = os.path.basename(str(ranker.model_dir))
    print(f"Ranker model name: {ranker_model_name}")

    # Load datasets
    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_questions = golden_raw.get("rag_samples", golden_raw)

    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_questions = json.load(f)

    # 1. Warm-up
    print("Warming up models...")
    _ = embed_query("warmup")
    _ = embed_sparse_query("warmup")
    _ = ranker.rerank(RerankRequest(query="warmup", passages=[{"text": "warmup", "id": 1}]))

    # Collect top rerank scores for all groups
    groups = {
        "golden_15": {"queries": [q["question"] for q in golden_questions], "scores": [], "details": []},
        "hard_22": {"queries": [q["question"] for q in hard_questions], "scores": [], "details": []},
        "negatives_k8s_9": {"queries": [q["question"] for q in HARD_NEGATIVES_K8S], "scores": [], "details": []},
        "negatives_noisy_9": {"queries": [q["question"] for q in HARD_NEGATIVES_NOISY], "scores": [], "details": []}
    }

    for g_name, g_data in groups.items():
        print(f"\nScoring group: {g_name} ({len(g_data['queries'])} queries)...")
        for q in g_data["queries"]:
            reranked = retrieve_and_rerank(client, ranker, q, top_k=15)
            top_score = reranked[0].get("score", 0.0) if reranked else 0.0
            g_data["scores"].append(top_score)
            
            top_cand = reranked[0] if reranked else {}
            meta = top_cand.get("meta", {})
            g_data["details"].append({
                "question": q,
                "top_score": round(top_score, 4),
                "top_source": meta.get("source", ""),
                "top_section": meta.get("section", ""),
                "top_source_type": meta.get("source_type", ""),
                "all_scores": [round(c.get("score", 0.0), 4) for c in reranked[:5]]
            })

    # 2. Score Distributions
    distributions = {}
    print("\n==================================================")
    print("TOP-1 RERANK SCORE DISTRIBUTIONS PER GROUP")
    print("==================================================")
    for g_name, g_data in groups.items():
        dist = compute_distribution(g_data["scores"])
        distributions[g_name] = dist
        print(f"Group: {g_name:18s} | Count: {len(g_data['scores']):2d} | Min: {dist['min']:.4f} | Median: {dist['median']:.4f} | Mean: {dist['mean']:.4f} | Max: {dist['max']:.4f} | [p10: {dist['p10']:.4f}, p90: {dist['p90']:.4f}]")

    # Positive and Negative pools
    positive_details = groups["golden_15"]["details"] + groups["hard_22"]["details"]
    negative_details = groups["negatives_k8s_9"]["details"] + groups["negatives_noisy_9"]["details"]

    print(f"\nTotal Positives: {len(positive_details)} | Total Negatives: {len(negative_details)}")

    # 3. Grid Sweep over Floor and Margin
    floor_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    margin_candidates = [0.20, 0.30, 0.40, 0.50]

    sweep_results = []
    print("\n==================================================")
    print("GRID SWEEP EVALUATION (Floor x Relative Margin)")
    print("==================================================")

    for floor in floor_candidates:
        for margin in margin_candidates:
            # Evaluate on Positives (False Refusals)
            false_refusals = 0
            chunks_passed_on_accepted = []

            for item in positive_details:
                top_score = item["top_score"]
                if top_score < floor:
                    false_refusals += 1
                else:
                    cutoff = top_score * (1.0 - margin)
                    passed = [s for s in item["all_scores"] if s >= cutoff]
                    if not passed:
                        false_refusals += 1
                    else:
                        chunks_passed_on_accepted.append(len(passed))

            false_refusal_rate = false_refusals / len(positive_details)
            true_acceptance_rate = 1.0 - false_refusal_rate

            # Evaluate on Negatives (True Rejections)
            true_rejections = 0
            for item in negative_details:
                top_score = item["top_score"]
                if top_score < floor:
                    true_rejections += 1
                else:
                    cutoff = top_score * (1.0 - margin)
                    passed = [s for s in item["all_scores"] if s >= cutoff]
                    if not passed:
                        true_rejections += 1

            true_rejection_rate = true_rejections / len(negative_details)

            # Balanced F1-score: harmonic mean of True Acceptance and True Rejection
            if (true_acceptance_rate + true_rejection_rate) > 0:
                f1_balance = 2 * (true_acceptance_rate * true_rejection_rate) / (true_acceptance_rate + true_rejection_rate)
            else:
                f1_balance = 0.0

            avg_chunks = statistics.mean(chunks_passed_on_accepted) if chunks_passed_on_accepted else 0.0

            entry = {
                "floor": floor,
                "margin": margin,
                "positives_count": len(positive_details),
                "false_refusals": false_refusals,
                "false_refusal_rate": round(false_refusal_rate, 4),
                "true_acceptance_rate": round(true_acceptance_rate, 4),
                "negatives_count": len(negative_details),
                "true_rejections": true_rejections,
                "true_rejection_rate": round(true_rejection_rate, 4),
                "balanced_score": round(f1_balance, 4),
                "avg_chunks_passed_when_accepted": round(avg_chunks, 2)
            }
            sweep_results.append(entry)

    # Sort to find Pareto frontier / optimal settings
    # Sort primarily by balanced_score descending, then by false_refusals ascending
    sorted_sweep = sorted(sweep_results, key=lambda x: (x["balanced_score"], -x["false_refusals"]), reverse=True)

    print(f"{'Floor':<7} | {'Margin':<7} | {'False Refusals':<14} | {'True Rejection':<14} | {'Balanced Score':<14} | {'Avg Chunks':<10}")
    print("-" * 75)
    for s in sweep_results:
        print(f"{s['floor']:<7.2f} | {s['margin']:<7.2f} | {s['false_refusals']:2d}/{s['positives_count']} ({s['false_refusal_rate']*100:.1f}%) | {s['true_rejections']:2d}/{s['negatives_count']} ({s['true_rejection_rate']*100:.1f}%) | {s['balanced_score']:<14.4f} | {s['avg_chunks_passed_when_accepted']:<10.2f}")

    best = sorted_sweep[0]
    print(f"\nOptimal Setting Identified:")
    print(f"  Floor:              {best['floor']}")
    print(f"  Margin:             {best['margin']}")
    print(f"  Balanced Score:     {best['balanced_score']}")
    print(f"  False Refusals:     {best['false_refusals']}/{best['positives_count']} ({best['false_refusal_rate']*100:.1f}%)")
    print(f"  True Rejections:    {best['true_rejections']}/{best['negatives_count']} ({best['true_rejection_rate']*100:.1f}%)")
    print(f"  Avg Chunks Passed:  {best['avg_chunks_passed_when_accepted']}")

    payload = {
        "script": "scripts/run_threshold_sweep.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection": COLLECTION_NAME,
        "ranker_model": ranker_model_name,
        "score_distributions": distributions,
        "sweep_grid": sweep_results,
        "optimal_setting": best,
        "per_query_details": {
            "golden_15": groups["golden_15"]["details"],
            "hard_22": groups["hard_22"]["details"],
            "negatives_k8s_9": groups["negatives_k8s_9"]["details"],
            "negatives_noisy_9": groups["negatives_noisy_9"]["details"]
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

    print(f"\nSaved threshold calibration results to '{RESULTS_FILE}'")


if __name__ == "__main__":
    main()
