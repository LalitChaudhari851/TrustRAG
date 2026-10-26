"""
TrustRAG Retrieval Evaluation Metric Engine.

Implements explicit, pure-Python information retrieval (IR) metrics, multi-hop
conjunction metrics, distractor interference metrics, retrieval abstention signals,
and bootstrap confidence intervals.

Design invariants:
1. Zero live LLM calls: All IR computations are deterministic and mathematical.
2. Dual precision reporting: Reports both fixed K (hits / K) and actual count
   (hits / max(1, len(retrieved))) to properly handle reranker threshold drops.
3. Multi-hop conjunction: Supports multiple sufficient evidence sets. Evaluates
   evidence coverage, all-required coverage, and multi-hop success separately
   from ordinary single-chunk recall.
4. Distractor separation: Distinguishes evidence completeness from distractor interference.
5. Abstention separation: Separates retrieval-level abstention signals (empty retrieval or
   all candidates below score floor) from final generation-level abstention.
6. Unanswerable isolation: Unanswerable queries (ground_truth == empty) are excluded from
   relevance macro-averages (Recall, Precision, MRR, nDCG) and evaluated under abstention metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Sequence
import numpy as np


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    """Preserves order while removing duplicate identifiers."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str] | Sequence[str], k: int) -> float:
    """
    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    
    If relevant_ids is empty, returns 0.0 (or should be excluded via unanswerable partitioning).
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    hits = len(set(top_k) & rel_set)
    return float(hits) / float(len(rel_set))


def precision_at_k_fixed(retrieved_ids: Sequence[str], relevant_ids: Set[str] | Sequence[str], k: int) -> float:
    """
    Precision@K (fixed K denominator) = |Retrieved@K ∩ Relevant| / K
    
    Standard IR definition. When the reranker drops candidates below threshold
    and returns fewer than K, this penalizes the system for missing slots.
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    hits = len(set(top_k) & rel_set)
    return float(hits) / float(k)


def precision_at_k_actual(retrieved_ids: Sequence[str], relevant_ids: Set[str] | Sequence[str], k: int) -> float:
    """
    Precision@K (actual returned denominator) = |Retrieved@K ∩ Relevant| / max(1, len(Retrieved@K))
    
    Measures precision among only the candidates actually returned by the reranker.
    If the reranker returned 2 documents and both are relevant, actual precision is 1.0 (2/2),
    even if K=5. If 0 documents are returned, precision is 0.0.
    """
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    rel_set = set(relevant_ids)
    hits = len(set(top_k) & rel_set)
    return float(hits) / float(len(top_k))


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str] | Sequence[str], k: int) -> float:
    """
    HitRate@K = 1.0 if any relevant chunk is in top-K, else 0.0.
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    return 1.0 if bool(set(top_k) & rel_set) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str] | Sequence[str], k: int) -> float:
    """
    MRR@K = 1 / rank of first relevant chunk (1-indexed), or 0.0 if none in top-K.
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(retrieved_ids: Sequence[str], relevance_grades: Dict[str, int], k: int) -> float:
    """
    DCG@K using the standard exponential gain formulation:
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0:
        return 0.0
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = relevance_grades.get(doc_id, 0)
        gain = float((2 ** rel) - 1)
        discount = math.log2(i + 1)
        dcg += gain / discount
    return dcg


def ndcg_at_k(retrieved_ids: Sequence[str], relevance_grades: Dict[str, int], k: int) -> float:
    """
    Graded nDCG@K = DCG@K / IDCG@K
    
    relevance_grades maps doc_id -> integer grade (e.g., 0=irrelevant, 1=supporting, 2=primary).
    If ideal DCG is 0 (no relevant items exist), returns 0.0.
    """
    if k <= 0 or not relevance_grades:
        return 0.0
    
    # Calculate DCG@K for actual retrieved
    actual_dcg = dcg_at_k(retrieved_ids, relevance_grades, k)
    if actual_dcg == 0.0:
        return 0.0

    # Calculate Ideal DCG@K by sorting known grades in descending order
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_grades, start=1):
        if rel <= 0:
            continue
        gain = float((2 ** rel) - 1)
        discount = math.log2(i + 1)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0

    return actual_dcg / idcg


def multi_hop_evidence_coverage(
    retrieved_ids: Sequence[str],
    sufficient_evidence_sets: Sequence[Sequence[str]],
    k: int
) -> float:
    """
    Evidence Coverage = max_{S in sufficient_sets} (|S ∩ Retrieved@K| / |S|)
    
    Measures the highest proportion of required evidence chunks retrieved across
    any sufficient evidence set. Range [0.0, 1.0].
    """
    if not sufficient_evidence_sets or k <= 0:
        return 0.0
    top_k_set = set(deduplicate_preserve_order(retrieved_ids)[:k])
    best_coverage = 0.0
    for s_set in sufficient_evidence_sets:
        s_unique = set(s_set)
        if not s_unique:
            continue
        coverage = len(s_unique & top_k_set) / len(s_unique)
        if coverage > best_coverage:
            best_coverage = coverage
    return float(best_coverage)


def all_required_evidence_coverage(
    retrieved_ids: Sequence[str],
    sufficient_evidence_sets: Sequence[Sequence[str]],
    k: int
) -> float:
    """
    All-Required Evidence Coverage = 1.0 if ANY sufficient evidence set is fully
    contained within Retrieved@K, else 0.0.
    """
    if not sufficient_evidence_sets or k <= 0:
        return 0.0
    top_k_set = set(deduplicate_preserve_order(retrieved_ids)[:k])
    for s_set in sufficient_evidence_sets:
        s_unique = set(s_set)
        if s_unique and s_unique.issubset(top_k_set):
            return 1.0
    return 0.0


def multi_hop_success(
    retrieved_ids: Sequence[str],
    sufficient_evidence_sets: Sequence[Sequence[str]],
    k: int,
    max_rank_span: Optional[int] = None
) -> float:
    """
    Multi-Hop Success = 1.0 if:
    1. At least one sufficient evidence set S is completely present in Retrieved@K.
    2. If max_rank_span is provided, the difference between the lowest rank and
       highest rank of chunks in S is <= max_rank_span (checking that required
       chunks are clustered without being buried under distant distractors).
    """
    if not sufficient_evidence_sets or k <= 0:
        return 0.0
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    top_k_set = set(top_k)

    for s_set in sufficient_evidence_sets:
        s_unique = set(s_set)
        if not s_unique or not s_unique.issubset(top_k_set):
            continue
        if max_rank_span is not None:
            ranks = [top_k.index(chunk_id) for chunk_id in s_unique]
            span = max(ranks) - min(ranks)
            if span > max_rank_span:
                continue
        return 1.0
    return 0.0


def distractor_interference_score(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int
) -> Dict[str, Any]:
    """
    Separates evidence completeness from distractor interference.
    
    Computes:
    - distractors_before_first_relevant: number of non-relevant items ranked before the first relevant hit.
    - distractor_ratio_in_top_k: fraction of returned top-K that are distractors (non-relevant).
    - first_relevant_rank: 1-indexed rank of the first relevant chunk, or None.
    """
    rel_set = set(relevant_ids)
    top_k = deduplicate_preserve_order(retrieved_ids)[:k]
    total_returned = len(top_k)
    
    if not rel_set:
        # Unanswerable query or empty relevance: undefined distractors before first relevant
        return {
            "distractors_before_first_relevant": None,
            "distractor_ratio_in_top_k": 1.0 if total_returned > 0 else 0.0,
            "first_relevant_rank": None,
            "total_distractors": total_returned,
            "total_retrieved": total_returned,
        }

    first_rel_rank = None
    distractors_before = 0
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            first_rel_rank = idx
            break
        distractors_before += 1

    # If no relevant chunk was retrieved in top-K, distractors before first relevant is undefined (None)
    distractors_before_val = distractors_before if first_rel_rank is not None else None

    distractor_count = sum(1 for doc_id in top_k if doc_id not in rel_set)
    distractor_ratio = float(distractor_count) / float(total_returned) if total_returned > 0 else 0.0

    return {
        "distractors_before_first_relevant": distractors_before_val,
        "distractor_ratio_in_top_k": distractor_ratio,
        "first_relevant_rank": first_rel_rank,
        "total_distractors": distractor_count,
        "total_retrieved": total_returned,
    }


def retrieval_abstention_signal(
    retrieved_candidates: Sequence[Dict[str, Any] | str],
    score_floor: float = 0.08,
    min_candidates: int = 1
) -> Dict[str, Any]:
    """
    Evaluates whether the retrieval stage signalled abstention.
    
    Separates retrieval-level abstention signals (empty retrieval or all candidates
    below score floor) from final application/generation abstention correctness.
    
    Returns:
    - should_abstain_at_retrieval: bool
    - reason: 'empty_retrieval' | 'below_score_floor' | 'insufficient_candidates' | 'proceed'
    - top_score: float or None
    """
    if not retrieved_candidates:
        return {
            "should_abstain_at_retrieval": True,
            "reason": "empty_retrieval",
            "top_score": None,
            "candidate_count": 0,
        }

    # Extract score if candidate is a dict
    first = retrieved_candidates[0]
    top_score = None
    if isinstance(first, dict):
        top_score = float(first.get("rerank_score") or first.get("score") or 0.0)

    if len(retrieved_candidates) < min_candidates:
        return {
            "should_abstain_at_retrieval": True,
            "reason": "insufficient_candidates",
            "top_score": top_score,
            "candidate_count": len(retrieved_candidates),
        }

    if top_score is not None and top_score < score_floor:
        return {
            "should_abstain_at_retrieval": True,
            "reason": "below_score_floor",
            "top_score": top_score,
            "candidate_count": len(retrieved_candidates),
        }

    return {
        "should_abstain_at_retrieval": False,
        "reason": "proceed",
        "top_score": top_score,
        "candidate_count": len(retrieved_candidates),
    }


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes percentile bootstrap confidence intervals on metric sequences.
    
    Returns dict:
    - mean: float
    - ci_lower: float
    - ci_upper: float
    - sample_size: int
    - note: cautionary note on sample size
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "sample_size": 0, "note": "empty"}

    mean_val = float(np.mean(arr))
    if n == 1 or np.all(arr == arr[0]):
        return {
            "mean": round(mean_val, 4),
            "ci_lower": round(mean_val, 4),
            "ci_upper": round(mean_val, 4),
            "sample_size": n,
            "note": "single value or zero variance"
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    note = "Sample size N < 100: CIs are approximate and reflect seed dataset uncertainty." if n < 100 else "Standard bootstrap CI."

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "sample_size": n,
        "note": note,
    }


def evaluate_single_query(
    query_id: str,
    query_type: str,
    retrieved_docs: Sequence[Dict[str, Any]],
    candidate_chunk_ids: Sequence[str],
    relevance_grades: Dict[str, int],
    sufficient_evidence_sets: Sequence[Sequence[str]],
    score_floor: float = 0.08
) -> Dict[str, Any]:
    """
    Evaluates retrieval for a single query across standard K cutoffs (K=5, K=10).
    """
    retrieved_ids = [str(d.get("id")) for d in retrieved_docs if d.get("id") is not None]
    is_unanswerable = (query_type == "unanswerable") or (len(candidate_chunk_ids) == 0)

    # Retrieval abstention signal
    abstention_eval = retrieval_abstention_signal(retrieved_docs, score_floor=score_floor)

    res: Dict[str, Any] = {
        "query_id": query_id,
        "query_type": query_type,
        "is_unanswerable": is_unanswerable,
        "retrieved_count": len(retrieved_ids),
        "top_score": abstention_eval["top_score"],
        "retrieval_abstention_decision": abstention_eval["should_abstain_at_retrieval"],
        "retrieval_abstention_reason": abstention_eval["reason"],
    }

    if is_unanswerable:
        # Correct retrieval abstention if retrieval decided to abstain (empty or below floor)
        # Or if 0 candidates returned.
        res["abstention_correct"] = bool(abstention_eval["should_abstain_at_retrieval"])
        return res

    # Answerable query metrics
    relevant_set = set(candidate_chunk_ids)

    # Calculate at K=5
    res["recall@5"] = recall_at_k(retrieved_ids, relevant_set, k=5)
    res["precision@5_fixed"] = precision_at_k_fixed(retrieved_ids, relevant_set, k=5)
    res["precision@5_actual"] = precision_at_k_actual(retrieved_ids, relevant_set, k=5)
    res["hit_rate@5"] = hit_rate_at_k(retrieved_ids, relevant_set, k=5)
    res["mrr@5"] = mrr_at_k(retrieved_ids, relevant_set, k=5)
    res["ndcg@5"] = ndcg_at_k(retrieved_ids, relevance_grades, k=5)

    # Calculate at K=10
    res["recall@10"] = recall_at_k(retrieved_ids, relevant_set, k=10)
    res["precision@10_fixed"] = precision_at_k_fixed(retrieved_ids, relevant_set, k=10)
    res["precision@10_actual"] = precision_at_k_actual(retrieved_ids, relevant_set, k=10)
    res["hit_rate@10"] = hit_rate_at_k(retrieved_ids, relevant_set, k=10)
    res["mrr@10"] = mrr_at_k(retrieved_ids, relevant_set, k=10)
    res["ndcg@10"] = ndcg_at_k(retrieved_ids, relevance_grades, k=10)

    # Distractor interference at K=5 and K=10
    res["distractor_info@5"] = distractor_interference_score(retrieved_ids, relevant_set, k=5)
    res["distractor_info@10"] = distractor_interference_score(retrieved_ids, relevant_set, k=10)

    # Multi-hop specific metrics
    if query_type == "answerable_multi_hop" or len(sufficient_evidence_sets) > 0:
        res["evidence_coverage@5"] = multi_hop_evidence_coverage(retrieved_ids, sufficient_evidence_sets, k=5)
        res["all_required_coverage@5"] = all_required_evidence_coverage(retrieved_ids, sufficient_evidence_sets, k=5)
        res["multi_hop_success@5"] = multi_hop_success(retrieved_ids, sufficient_evidence_sets, k=5)

        res["evidence_coverage@10"] = multi_hop_evidence_coverage(retrieved_ids, sufficient_evidence_sets, k=10)
        res["all_required_coverage@10"] = all_required_evidence_coverage(retrieved_ids, sufficient_evidence_sets, k=10)
        res["multi_hop_success@10"] = multi_hop_success(retrieved_ids, sufficient_evidence_sets, k=10)

    return res


def aggregate_retrieval_metrics(
    eval_records: Sequence[Dict[str, Any]],
    n_resamples: int = 1000
) -> Dict[str, Any]:
    """
    Aggregates per-query evaluation records into a macro report with bootstrap CIs.
    
    CRITICAL: Excludes unanswerable queries from ordinary relevance macro-averages
    (Recall, Precision, MRR, nDCG) and reports abstention separately.
    """
    answerable_records = [r for r in eval_records if not r.get("is_unanswerable")]
    unanswerable_records = [r for r in eval_records if r.get("is_unanswerable")]
    multi_hop_records = [r for r in answerable_records if r.get("query_type") == "answerable_multi_hop"]

    summary: Dict[str, Any] = {
        "counts": {
            "total_queries": len(eval_records),
            "answerable_queries": len(answerable_records),
            "unanswerable_queries": len(unanswerable_records),
            "multi_hop_queries": len(multi_hop_records),
        }
    }

    # 1. Relevance macro-averages on answerable queries
    metric_keys = [
        "recall@5", "precision@5_fixed", "precision@5_actual", "hit_rate@5", "mrr@5", "ndcg@5",
        "recall@10", "precision@10_fixed", "precision@10_actual", "hit_rate@10", "mrr@10", "ndcg@10",
    ]

    relevance_macro: Dict[str, Any] = {}
    for key in metric_keys:
        vals = [r[key] for r in answerable_records if key in r]
        relevance_macro[key] = bootstrap_confidence_interval(vals, n_resamples=n_resamples)

    summary["relevance_macro_averages"] = relevance_macro

    # 2. Multi-hop conjunction metrics
    if multi_hop_records:
        mh_summary: Dict[str, Any] = {}
        for mh_key in [
            "evidence_coverage@5", "all_required_coverage@5", "multi_hop_success@5",
            "evidence_coverage@10", "all_required_coverage@10", "multi_hop_success@10"
        ]:
            mh_vals = [r[mh_key] for r in multi_hop_records if mh_key in r]
            mh_summary[mh_key] = bootstrap_confidence_interval(mh_vals, n_resamples=n_resamples)
        summary["multi_hop_metrics"] = mh_summary

    # 3. Distractor interference summary
    if answerable_records:
        dist_before_5 = [
            r["distractor_info@5"]["distractors_before_first_relevant"]
            for r in answerable_records
            if "distractor_info@5" in r and r["distractor_info@5"].get("distractors_before_first_relevant") is not None
        ]
        dist_ratio_5 = [r["distractor_info@5"]["distractor_ratio_in_top_k"] for r in answerable_records if "distractor_info@5" in r]
        dist_ratio_10 = [r["distractor_info@10"]["distractor_ratio_in_top_k"] for r in answerable_records if "distractor_info@10" in r]
        summary["distractor_interference"] = {
            "distractors_before_first_relevant@5": bootstrap_confidence_interval(dist_before_5, n_resamples=n_resamples),
            "distractor_ratio@5": bootstrap_confidence_interval(dist_ratio_5, n_resamples=n_resamples),
            "distractor_ratio@10": bootstrap_confidence_interval(dist_ratio_10, n_resamples=n_resamples),
        }

    # 4. Abstention & unanswerable metrics
    if unanswerable_records:
        correct_abstentions = sum(1 for r in unanswerable_records if r.get("abstention_correct"))
        abstention_rate = float(correct_abstentions) / float(len(unanswerable_records))
        summary["unanswerable_abstention"] = {
            "unanswerable_count": len(unanswerable_records),
            "correct_retrieval_abstentions": correct_abstentions,
            "retrieval_abstention_rate": round(abstention_rate, 4),
            "interpretation": "Fraction of unanswerable queries where retrieval signalled abstention (empty results or top score below floor)."
        }

    return summary
