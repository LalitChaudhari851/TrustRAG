#!/usr/bin/env python3
"""
TrustRAG Paired Retrieval Experiment: K=15 vs K=25 Candidate Pool Evaluation.

Executes a paired sequential evaluation of all 50 golden dataset records (seed_001 to seed_050)
against the running TrustRAG FastAPI application, capturing end-to-end production traces,
comparing retrieval effectiveness, answer correctness, abstention behavior, context token load,
and latency between:
  A. Baseline Candidate Limit: K=15 (reranker_margin=0.20, min_score=0.08)
  B. Candidate Pool Expansion: K=25 (reranker_margin=0.20, min_score=0.08)

Outputs structured artifacts in evals/results/experiment_k15_vs_k25_<run_id>/:
  - paired_results.json
  - paired_results.csv
  - summary.md
  - reproducibility_config.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from evals.retrieval_dataset_schema import RetrievalEvalRecord, load_dataset
from evals.retrieval_metrics import (
    recall_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k_actual,
    retrieval_abstention_signal,
    bootstrap_confidence_interval,
)

from scripts.run_batch_eval import (
    SECRET_PATTERNS,
    redact_secrets,
    REFUSAL_PHRASES,
    is_abstention_answer,
    ChunkResolver,
    DeterministicEvaluator,
    diagnose_abstention,
)


class PairedKAppClient:
    """Client for executing requests against the running TrustRAG FastAPI server with retry."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 35.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ping_config(self) -> Dict[str, Any]:
        url = f"{self.base_url}/config"
        try:
            r = requests.get(url, timeout=5.0)
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"error": f"Could not connect to {url}: {e}"}

    def execute_query(
        self,
        question: str,
        candidate_limit: int,
        margin: float = 0.20,
        min_score: float = 0.08,
        retries: int = 2
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/query"
        payload = {
            "q": question,
            "thread_id": f"eval_{uuid.uuid4().hex[:8]}",
            "return_trace": True,
            "reranker_margin": margin,
            "reranker_min_score": min_score,
            "candidate_limit": candidate_limit,
        }
        headers = {
            "Content-Type": "application/json",
            "X-User-ID": "paired_k_eval_runner",
            "X-Eval-Trace": "true",
            "X-Reranker-Margin": str(margin),
            "X-Reranker-Min-Score": str(min_score),
            "X-Candidate-Limit": str(candidate_limit),
        }

        last_error = ""
        last_error_category = "none"

        for attempt in range(retries + 1):
            start_time = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                latency_ms = int((time.perf_counter() - start_time) * 1000)

                if resp.status_code != 200:
                    last_error_category = f"http_{resp.status_code}"
                    last_error = resp.text[:300]
                    if resp.status_code in (502, 503, 504, 429) and attempt < retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    return {
                        "is_infra_error": True,
                        "error_category": last_error_category,
                        "error_message": last_error,
                        "latency_ms": latency_ms,
                        "answer": "",
                        "status": "http_error",
                        "sources": [],
                        "retrieved_chunks": [],
                        "context_sent_to_llm": "",
                        "model_used": None,
                        "effective_margin": margin,
                        "effective_min_score": min_score,
                        "candidate_pool_limit": candidate_limit,
                        "raw_search_count": 0,
                    }

                data = resp.json()
                retrieved_docs = []
                raw_trace = data.get("retrieval_trace") or []
                for idx, doc in enumerate(raw_trace, start=1):
                    retrieved_docs.append({
                        "id": str(doc.get("id")),
                        "rank": idx,
                        "score": doc.get("score"),
                        "rerank_score": doc.get("rerank_score"),
                        "source": doc.get("source"),
                        "section": doc.get("section"),
                        "chunk_index": doc.get("chunk_index"),
                        "content": doc.get("content", ""),
                    })

                eff_margin = data.get("effective_reranker_margin", margin)
                eff_floor = data.get("effective_reranker_min_score", min_score)
                cand_limit = data.get("candidate_pool_limit", candidate_limit)
                raw_count = data.get("raw_search_count", len(retrieved_docs))

                return {
                    "is_infra_error": False,
                    "error_category": "none",
                    "error_message": "",
                    "latency_ms": latency_ms,
                    "answer": data.get("answer", ""),
                    "status": data.get("status", "ok"),
                    "sources": data.get("sources", []),
                    "retrieved_chunks": retrieved_docs,
                    "context_sent_to_llm": data.get("context_sent_to_llm", ""),
                    "model_used": data.get("model_used"),
                    "effective_margin": eff_margin,
                    "effective_min_score": eff_floor,
                    "candidate_pool_limit": cand_limit,
                    "raw_search_count": raw_count,
                }

            except requests.exceptions.Timeout:
                last_error_category = "timeout"
                last_error = f"Request timed out after {self.timeout}s"
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                if attempt < retries:
                    time.sleep(1.0)
                    continue
            except requests.exceptions.RequestException as e:
                last_error_category = "network_error"
                last_error = str(e)[:300]
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                if attempt < retries:
                    time.sleep(1.0)
                    continue

        return {
            "is_infra_error": True,
            "error_category": last_error_category,
            "error_message": last_error,
            "latency_ms": latency_ms,
            "answer": "",
            "status": "error",
            "sources": [],
            "retrieved_chunks": [],
            "context_sent_to_llm": "",
            "model_used": None,
            "effective_margin": margin,
            "effective_min_score": min_score,
            "candidate_pool_limit": candidate_limit,
            "raw_search_count": 0,
        }


class PairedKEvaluationRunner:
    """Orchestrates paired sequential execution of 50 questions across K=15 and K=25."""

    def __init__(
        self,
        dataset_path: str = "evals/data/retrieval_golden_seed_v1.json",
        output_dir: Optional[str] = None,
        base_url: str = "http://127.0.0.1:8000",
        margin: float = 0.20,
        min_score: float = 0.08,
    ):
        self.dataset_path = dataset_path
        self.base_url = base_url
        self.margin = margin
        self.min_score = min_score
        self.run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        self.output_dir = output_dir or os.path.join("evals", "results", f"experiment_k15_vs_k25_{self.run_id}")
        self.client = PairedKAppClient(base_url=self.base_url, timeout=35.0)
        self.evaluator = DeterministicEvaluator()
        self.resolver: Optional[ChunkResolver] = None
        self.records: List[RetrievalEvalRecord] = []
        self.dataset_hash = ""

    def initialize(self) -> None:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        with open(self.dataset_path, "rb") as f:
            self.dataset_hash = hashlib.sha256(f.read()).hexdigest()

        all_records = load_dataset(self.dataset_path)
        record_map = {r.id: r for r in all_records}
        self.records = []
        for i in range(1, 51):
            qid = f"seed_{i:03d}"
            if qid in record_map:
                self.records.append(record_map[qid])
            else:
                raise ValueError(f"Required record ID {qid} missing from dataset {self.dataset_path}")

        self.resolver = ChunkResolver(self.records)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_response(
        self,
        rec: RetrievalEvalRecord,
        resp: Dict[str, Any],
        candidate_limit: int,
    ) -> Dict[str, Any]:
        """Evaluates retrieval metrics, answer verdict, abstention, and evidence retention."""
        latency_ms = resp.get("latency_ms", 0)
        is_infra_error = resp.get("is_infra_error", False)
        error_category = resp.get("error_category", "none")
        app_answer = resp.get("answer", "")
        raw_status = resp.get("status", "")
        context_sent = resp.get("context_sent_to_llm", "")

        retrieved_chunk_info = self.resolver.extract_retrieved_chunk_ids(
            resp.get("retrieved_chunks", []),
            resp.get("sources", []),
            rec.candidate_chunk_ids
        )
        retrieved_ids = [c["id"] for c in retrieved_chunk_info]
        retrieved_ranks = [str(c["rank"]) for c in retrieved_chunk_info]
        retrieved_scores = [str(c["score"] or "") for c in retrieved_chunk_info]
        rerank_scores = [c.get("rerank_score") for c in retrieved_chunk_info]

        relevant_candidates = set(rec.candidate_chunk_ids)
        relevant_retrieved = bool(relevant_candidates & set(retrieved_ids))
        context_evidence_present = bool(rec.candidate_chunk_ids and any(cid in context_sent for cid in rec.candidate_chunk_ids))

        rec_metrics = {
            "recall@1": recall_at_k(retrieved_ids, relevant_candidates, k=1),
            "recall@5": recall_at_k(retrieved_ids, relevant_candidates, k=5),
            "recall@10": recall_at_k(retrieved_ids, relevant_candidates, k=10),
            "hit_rate@1": hit_rate_at_k(retrieved_ids, relevant_candidates, k=1),
            "hit_rate@5": hit_rate_at_k(retrieved_ids, relevant_candidates, k=5),
            "hit_rate@10": hit_rate_at_k(retrieved_ids, relevant_candidates, k=10),
            "mrr@10": mrr_at_k(retrieved_ids, relevant_candidates, k=10),
            "ndcg@5": ndcg_at_k(retrieved_ids, rec.relevance_grades, k=5),
            "ndcg@10": ndcg_at_k(retrieved_ids, rec.relevance_grades, k=10),
        }

        # Multi-hop evidence calculation
        multihop_success = None
        if rec.query_type == "multi_hop":
            if len(rec.candidate_chunk_ids) > 0:
                recalled_count = len(relevant_candidates & set(retrieved_ids))
                multihop_success = (recalled_count == len(relevant_candidates))
            else:
                multihop_success = relevant_retrieved

        is_abstained = is_abstention_answer(app_answer) or raw_status == "insufficient_context"
        abstention_status = "none"
        verdict = "INCORRECT"
        verdict_reason = ""
        diagnosis = "none"
        diagnosis_detail = ""

        if is_infra_error:
            verdict = "INFRASTRUCTURE_ERROR"
            verdict_reason = f"Request failed: {resp.get('error_message')}"
            diagnosis, diagnosis_detail = diagnose_abstention(rec, retrieved_ids, raw_status, context_sent, True, error_category)
        elif raw_status == "blocked_by_guardrails":
            if rec.query_type == "unanswerable":
                verdict = "CORRECT"
                verdict_reason = "Correctly abstained/blocked on unanswerable query."
            else:
                verdict = "ABSTAINED"
                abstention_status = "guardrail_blocked"
                verdict_reason = "Blocked by safety guardrails."
            diagnosis, diagnosis_detail = diagnose_abstention(rec, retrieved_ids, raw_status, context_sent, False, "guardrail_blocked")
        elif is_abstained:
            abstention_status = "refused_insufficient_context"
            if rec.query_type == "unanswerable":
                verdict = "CORRECT"
                verdict_reason = "Correctly abstained on unanswerable query."
            else:
                verdict = "ABSTAINED"
                verdict_reason = "Model abstained ('Not Found in Documentation') for answerable question."
            diagnosis, diagnosis_detail = diagnose_abstention(rec, retrieved_ids, raw_status, context_sent, False, "abstained")
        else:
            if rec.query_type == "unanswerable":
                verdict = "INCORRECT"
                verdict_reason = "Fabricated answer or failed to abstain on unanswerable query."
            else:
                v, r = self.evaluator.evaluate_question(rec.id, rec.query, app_answer, rec.expected_answer)
                verdict = v
                verdict_reason = r

        cohort = "seed_011_050_unverified_synthetic"
        if rec.id in [f"seed_{i:03d}" for i in range(1, 6)]:
            cohort = "seed_001_005_disputed_mapping"
        elif rec.id in [f"seed_{i:03d}" for i in range(6, 11)]:
            cohort = "seed_006_010_batch2_edits_unverified"

        return {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "app_answer": app_answer,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_ranks": retrieved_ranks,
            "retrieved_scores": retrieved_scores,
            "rerank_scores": rerank_scores,
            "chunks_retained": len(retrieved_ids),
            "relevant_evidence_retrieved": relevant_retrieved,
            "context_evidence_present": context_evidence_present,
            "multihop_success": multihop_success,
            "abstention_status": abstention_status,
            "latency_ms": latency_ms,
            "is_infra_error": is_infra_error,
            "error_category": error_category,
            "error_message": resp.get("error_message", ""),
            "diagnosis": diagnosis,
            "diagnosis_detail": diagnosis_detail,
            "retrieval_metrics": rec_metrics,
            "effective_margin": resp.get("effective_margin", self.margin),
            "effective_min_score": resp.get("effective_min_score", self.min_score),
            "candidate_pool_limit": resp.get("candidate_pool_limit", candidate_limit),
            "raw_search_count": resp.get("raw_search_count", 0),
            "model_used": resp.get("model_used"),
            "context_word_count": len(context_sent.split()) if context_sent else 0,
            "cohort": cohort,
        }

    def run(self) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"TrustRAG Paired Retrieval Experiment: K=15 vs K=25")
        print(f"Output Directory: {self.output_dir}")
        print(f"Dataset SHA256:   {self.dataset_hash}")
        print(f"=======================================================\n")

        srv_cfg = self.client.ping_config()
        print(f"Connected to FastAPI Server at {self.base_url}")
        print(f"Server Config: {json.dumps(srv_cfg, indent=2)}\n")

        paired_rows = []
        k15_results = []
        k25_results = []

        total_questions = len(self.records)

        for idx, rec in enumerate(self.records, start=1):
            qid = rec.id
            q_text = rec.query
            print(f"[{idx:02d}/{total_questions:02d}] Evaluating {qid}: {q_text[:50]}...")

            # Run K=15
            resp_k15 = self.client.execute_query(
                question=q_text,
                candidate_limit=15,
                margin=self.margin,
                min_score=self.min_score,
            )
            eval_k15 = self.evaluate_response(rec, resp_k15, candidate_limit=15)
            time.sleep(0.3)

            # Run K=25
            resp_k25 = self.client.execute_query(
                question=q_text,
                candidate_limit=25,
                margin=self.margin,
                min_score=self.min_score,
            )
            eval_k25 = self.evaluate_response(rec, resp_k25, candidate_limit=25)
            time.sleep(0.3)

            k15_results.append(eval_k15)
            k25_results.append(eval_k25)

            # Paired comparison
            v_k15 = eval_k15["verdict"]
            v_k25 = eval_k25["verdict"]

            verdict_comp = "TIE"
            if v_k15 != v_k25:
                if v_k15 == "CORRECT" and v_k25 != "CORRECT":
                    verdict_comp = "K15_WIN"
                elif v_k25 == "CORRECT" and v_k15 != "CORRECT":
                    verdict_comp = "K25_WIN"
                else:
                    verdict_comp = "DISAGREEMENT"

            paired_rows.append({
                "question_id": qid,
                "cohort": eval_k15["cohort"],
                "query": q_text,
                "query_type": rec.query_type,
                "expected_answer": rec.expected_answer,
                "ground_truth_chunks": "|".join(rec.candidate_chunk_ids),
                "k15_verdict": v_k15,
                "k25_verdict": v_k25,
                "verdict_comparison": verdict_comp,
                "k15_latency_ms": eval_k15["latency_ms"],
                "k25_latency_ms": eval_k25["latency_ms"],
                "k15_raw_search_count": eval_k15["raw_search_count"],
                "k25_raw_search_count": eval_k25["raw_search_count"],
                "k15_chunks_retained": eval_k15["chunks_retained"],
                "k25_chunks_retained": eval_k25["chunks_retained"],
                "k15_recall@5": eval_k15["retrieval_metrics"]["recall@5"],
                "k25_recall@5": eval_k25["retrieval_metrics"]["recall@5"],
                "k15_hit_rate@5": eval_k15["retrieval_metrics"]["hit_rate@5"],
                "k25_hit_rate@5": eval_k25["retrieval_metrics"]["hit_rate@5"],
                "k15_mrr@10": eval_k15["retrieval_metrics"]["mrr@10"],
                "k25_mrr@10": eval_k25["retrieval_metrics"]["mrr@10"],
                "k15_ndcg@5": eval_k15["retrieval_metrics"]["ndcg@5"],
                "k25_ndcg@5": eval_k25["retrieval_metrics"]["ndcg@5"],
                "k15_multihop_success": eval_k15["multihop_success"],
                "k25_multihop_success": eval_k25["multihop_success"],
                "k15_abstention_status": eval_k15["abstention_status"],
                "k25_abstention_status": eval_k25["abstention_status"],
                "k15_diagnosis": eval_k15["diagnosis"],
                "k25_diagnosis": eval_k25["diagnosis"],
                "k15_retrieved_chunk_ids": "|".join(eval_k15["retrieved_chunk_ids"]),
                "k25_retrieved_chunk_ids": "|".join(eval_k25["retrieved_chunk_ids"]),
                "k15_app_answer": eval_k15["app_answer"],
                "k25_app_answer": eval_k25["app_answer"],
                "k15_verdict_reason": eval_k15["verdict_reason"],
                "k25_verdict_reason": eval_k25["verdict_reason"],
                "k15_infra_error": eval_k15["is_infra_error"],
                "k25_infra_error": eval_k25["is_infra_error"],
            })

            print(f"      K=15: {v_k15:<10} (ret={eval_k15['chunks_retained']}, lat={eval_k15['latency_ms']}ms) | "
                  f"K=25: {v_k25:<10} (ret={eval_k25['chunks_retained']}, lat={eval_k25['latency_ms']}ms) -> {verdict_comp}")

        # Compute aggregate metrics
        return self._generate_artifacts(paired_rows, k15_results, k25_results, srv_cfg)

    def _generate_artifacts(
        self,
        paired_rows: List[Dict[str, Any]],
        k15_results: List[Dict[str, Any]],
        k25_results: List[Dict[str, Any]],
        srv_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1. Write CSV
        csv_path = os.path.join(self.output_dir, "paired_results.csv")
        fieldnames = list(paired_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in paired_rows:
                writer.writerow(row)

        # 2. Compute Cohort Statistics
        cohorts = {
            "all_50": lambda r: True,
            "seed_001_005_disputed": lambda r: r["cohort"] == "seed_001_005_disputed_mapping",
            "seed_006_010_corpus_edits": lambda r: r["cohort"] == "seed_006_010_batch2_edits_unverified",
            "seed_011_050_remainder": lambda r: r["cohort"] == "seed_011_050_unverified_synthetic",
            "eligible_clean_dev_cohort_2_3": lambda r: r["cohort"] != "seed_001_005_disputed_mapping",
        }

        stats = {}
        for cname, cfilter in cohorts.items():
            c_rows = [r for r in paired_rows if cfilter(r)]
            total = len(c_rows)
            if total == 0:
                continue

            # Infra errors
            k15_infra = sum(1 for r in c_rows if r["k15_infra_error"])
            k25_infra = sum(1 for r in c_rows if r["k25_infra_error"])

            # Completed (excluding infra error)
            k15_denom = total - k15_infra
            k25_denom = total - k25_infra

            k15_correct = sum(1 for r in c_rows if r["k15_verdict"] == "CORRECT")
            k25_correct = sum(1 for r in c_rows if r["k25_verdict"] == "CORRECT")

            k15_pass_rate = (k15_correct / k15_denom) if k15_denom > 0 else 0.0
            k25_pass_rate = (k25_correct / k25_denom) if k25_denom > 0 else 0.0

            # Answerable records for retrieval metrics
            ans_rows = [r for r in c_rows if r["query_type"] != "unanswerable"]
            ans_total = len(ans_rows)

            def avg_m(metric_key):
                vals = [r[metric_key] for r in ans_rows if r[metric_key] is not None]
                return float(np.mean(vals)) if vals else 0.0

            # Multi-hop records
            mh_rows = [r for r in c_rows if r["query_type"] == "multi_hop"]
            mh_total = len(mh_rows)
            k15_mh_succ = sum(1 for r in mh_rows if r["k15_multihop_success"]) if mh_total > 0 else 0
            k25_mh_succ = sum(1 for r in mh_rows if r["k25_multihop_success"]) if mh_total > 0 else 0

            # Latency
            k15_lats = [r["k15_latency_ms"] for r in c_rows]
            k25_lats = [r["k25_latency_ms"] for r in c_rows]

            stats[cname] = {
                "total_queries": total,
                "answerable_queries": ans_total,
                "k15_completed": k15_denom,
                "k25_completed": k25_denom,
                "k15_infra_errors": k15_infra,
                "k25_infra_errors": k25_infra,
                "k15_correct": k15_correct,
                "k25_correct": k25_correct,
                "k15_pass_rate": k15_pass_rate,
                "k25_pass_rate": k25_pass_rate,
                "k15_recall@5": avg_m("k15_recall@5"),
                "k25_recall@5": avg_m("k25_recall@5"),
                "k15_hit_rate@5": avg_m("k15_hit_rate@5"),
                "k25_hit_rate@5": avg_m("k25_hit_rate@5"),
                "k15_mrr@10": avg_m("k15_mrr@10"),
                "k25_mrr@10": avg_m("k25_mrr@10"),
                "k15_ndcg@5": avg_m("k15_ndcg@5"),
                "k25_ndcg@5": avg_m("k25_ndcg@5"),
                "multihop_total": mh_total,
                "k15_multihop_success": k15_mh_succ,
                "k25_multihop_success": k25_mh_succ,
                "k15_mean_latency_ms": float(np.mean(k15_lats)),
                "k25_mean_latency_ms": float(np.mean(k25_lats)),
                "k15_p95_latency_ms": float(np.percentile(k15_lats, 95)),
                "k25_p95_latency_ms": float(np.percentile(k25_lats, 95)),
            }

        # Paired outcome summary
        wins_k15 = sum(1 for r in paired_rows if r["verdict_comparison"] == "K15_WIN")
        wins_k25 = sum(1 for r in paired_rows if r["verdict_comparison"] == "K25_WIN")
        ties = sum(1 for r in paired_rows if r["verdict_comparison"] == "TIE")
        disagreements = sum(1 for r in paired_rows if r["verdict_comparison"] == "DISAGREEMENT")

        # 3. Write JSON
        json_output = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_sha256": self.dataset_hash,
            "dataset_path": self.dataset_path,
            "server_config": srv_cfg,
            "experiment_parameters": {
                "variant_A": {"candidate_limit": 15, "margin": self.margin, "min_score": self.min_score},
                "variant_B": {"candidate_limit": 25, "margin": self.margin, "min_score": self.min_score},
            },
            "paired_verdicts": {
                "k15_wins": wins_k15,
                "k25_wins": wins_k25,
                "ties": ties,
                "disagreements": disagreements,
            },
            "cohort_statistics": stats,
            "detailed_records": paired_rows,
        }
        json_path = os.path.join(self.output_dir, "paired_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)

        # 4. Write Reproducibility Config
        repro_path = os.path.join(self.output_dir, "reproducibility_config.json")
        with open(repro_path, "w", encoding="utf-8") as f:
            json.dump({
                "command": f".venv\\Scripts\\python.exe scripts/run_paired_k_validation.py",
                "run_id": self.run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "dataset_sha256": self.dataset_hash,
                "server_config": srv_cfg,
                "candidate_limits": [15, 25],
                "reranker_margin": self.margin,
                "reranker_min_score": self.min_score,
            }, f, indent=2)

        # 5. Write Summary Markdown
        summary_md_path = os.path.join(self.output_dir, "summary.md")
        self._write_summary_md(summary_md_path, stats, wins_k15, wins_k25, ties, disagreements, paired_rows)

        print(f"\n=======================================================")
        print(f"Experiment Complete! Artifacts written to: {self.output_dir}")
        print(f"Paired Verdicts: K=15 Wins={wins_k15}, K=25 Wins={wins_k25}, Ties={ties}")
        print(f"Pass Rates (All 50): K=15: {stats['all_50']['k15_correct']}/{stats['all_50']['k15_completed']} ({stats['all_50']['k15_pass_rate']:.1%}) vs K=25: {stats['all_50']['k25_correct']}/{stats['all_50']['k25_completed']} ({stats['all_50']['k25_pass_rate']:.1%})")
        print(f"Pass Rates (Clean Dev Cohort 2+3): K=15: {stats['eligible_clean_dev_cohort_2_3']['k15_correct']}/{stats['eligible_clean_dev_cohort_2_3']['k15_completed']} ({stats['eligible_clean_dev_cohort_2_3']['k15_pass_rate']:.1%}) vs K=25: {stats['eligible_clean_dev_cohort_2_3']['k25_correct']}/{stats['eligible_clean_dev_cohort_2_3']['k25_completed']} ({stats['eligible_clean_dev_cohort_2_3']['k25_pass_rate']:.1%})")
        print(f"=======================================================\n")

        return json_output

    def _write_summary_md(
        self,
        filepath: str,
        stats: Dict[str, Any],
        wins_k15: int,
        wins_k25: int,
        ties: int,
        disagreements: int,
        paired_rows: List[Dict[str, Any]],
    ) -> None:
        all_s = stats["all_50"]
        clean_s = stats["eligible_clean_dev_cohort_2_3"]

        lines = [
            "# TrustRAG Paired Retrieval Experiment: K=15 vs K=25",
            "",
            f"**Run ID**: `{self.run_id}`  ",
            f"**Timestamp**: `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  ",
            f"**Dataset SHA256**: `{self.dataset_hash}`  ",
            f"**Parameters**: `reranker_margin=0.20`, `reranker_min_score=0.08`, `top_n=5`  ",
            "",
            "## 1. Executive Summary",
            "",
            "| Metric | K=15 Baseline | K=25 Candidate Pool | Delta |",
            "|---|---|---|---|",
            f"| **Pass Rate (All 50)** | {all_s['k15_correct']}/{all_s['k15_completed']} ({all_s['k15_pass_rate']:.1%}) | {all_s['k25_correct']}/{all_s['k25_completed']} ({all_s['k25_pass_rate']:.1%}) | {all_s['k25_pass_rate'] - all_s['k15_pass_rate']:+.1%} |",
            f"| **Pass Rate (Clean Dev Cohorts 2+3, n=45)** | {clean_s['k15_correct']}/{clean_s['k15_completed']} ({clean_s['k15_pass_rate']:.1%}) | {clean_s['k25_correct']}/{clean_s['k25_completed']} ({clean_s['k25_pass_rate']:.1%}) | {clean_s['k25_pass_rate'] - clean_s['k15_pass_rate']:+.1%} |",
            f"| **Recall@5** (answerable) | {all_s['k15_recall@5']:.4f} | {all_s['k25_recall@5']:.4f} | {all_s['k25_recall@5'] - all_s['k15_recall@5']:+.4f} |",
            f"| **HitRate@5** (answerable) | {all_s['k15_hit_rate@5']:.4f} | {all_s['k25_hit_rate@5']:.4f} | {all_s['k25_hit_rate@5'] - all_s['k15_hit_rate@5']:+.4f} |",
            f"| **MRR@10** (answerable) | {all_s['k15_mrr@10']:.4f} | {all_s['k25_mrr@10']:.4f} | {all_s['k25_mrr@10'] - all_s['k15_mrr@10']:+.4f} |",
            f"| **nDCG@5** (answerable) | {all_s['k15_ndcg@5']:.4f} | {all_s['k25_ndcg@5']:.4f} | {all_s['k25_ndcg@5'] - all_s['k15_ndcg@5']:+.4f} |",
            f"| **Multi-hop Coverage** | {all_s['k15_multihop_success']}/{all_s['multihop_total']} ({all_s['k15_multihop_success']/max(1, all_s['multihop_total']):.1%}) | {all_s['k25_multihop_success']}/{all_s['multihop_total']} ({all_s['k25_multihop_success']/max(1, all_s['multihop_total']):.1%}) | {all_s['k25_multihop_success'] - all_s['k15_multihop_success']:+d} |",
            f"| **Mean Latency (ms)** | {all_s['k15_mean_latency_ms']:.1f} ms | {all_s['k25_mean_latency_ms']:.1f} ms | {all_s['k25_mean_latency_ms'] - all_s['k15_mean_latency_ms']:+.1f} ms |",
            f"| **p95 Latency (ms)** | {all_s['k15_p95_latency_ms']:.1f} ms | {all_s['k25_p95_latency_ms']:.1f} ms | {all_s['k25_p95_latency_ms'] - all_s['k15_p95_latency_ms']:+.1f} ms |",
            f"| **Infra Errors / Timeouts** | {all_s['k15_infra_errors']} | {all_s['k25_infra_errors']} | 0 |",
            "",
            "## 2. Paired Verdict Analysis",
            "",
            f"- **K=15 Wins**: {wins_k15}",
            f"- **K=25 Wins**: {wins_k25}",
            f"- **Ties**: {ties}",
            f"- **Disagreements**: {disagreements}",
            "",
            "### Paired Discordance Details",
            "",
        ]

        discordant = [r for r in paired_rows if r["verdict_comparison"] != "TIE"]
        if not discordant:
            lines.append("No paired discordance observed across all 50 questions.")
        else:
            lines.append("| Question ID | Cohort | K=15 Verdict | K=25 Verdict | Outcome | Detail |")
            lines.append("|---|---|---|---|---|---|")
            for r in discordant:
                lines.append(f"| `{r['question_id']}` | {r['cohort']} | {r['k15_verdict']} | {r['k25_verdict']} | {r['verdict_comparison']} | {r['k25_verdict_reason'][:60]} |")

        lines.extend([
            "",
            "## 3. Stratified Cohort Breakdown",
            "",
            "| Cohort | N | K=15 Pass Rate | K=25 Pass Rate | K=15 Recall@5 | K=25 Recall@5 | K=15 HitRate@5 | K=25 HitRate@5 |",
            "|---|---|---|---|---|---|---|---|",
        ])

        for cname, s in stats.items():
            lines.append(
                f"| `{cname}` | {s['total_queries']} | "
                f"{s['k15_correct']}/{s['k15_completed']} ({s['k15_pass_rate']:.1%}) | "
                f"{s['k25_correct']}/{s['k25_completed']} ({s['k25_pass_rate']:.1%}) | "
                f"{s['k15_recall@5']:.3f} | {s['k25_recall@5']:.3f} | "
                f"{s['k15_hit_rate@5']:.3f} | {s['k25_hit_rate@5']:.3f} |"
            )

        lines.extend([
            "",
            "## 4. Engineering Findings & Recommendation",
            "",
            "- **Candidate Pool Impact**: Expanding Qdrant initial candidate pool from 15 to 25 feeds 25 candidates to FlashRank reranking.",
            "- **Latency Tradeoff**: Measured average latency difference between scoring 15 vs 25 candidates.",
            "- **Retrieval & Grounding Effect**: Documented whether any missing documents from smaller candidate pools are recovered.",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="TrustRAG Paired K=15 vs K=25 Retrieval Evaluation")
    parser.add_argument("--dataset", default="evals/data/retrieval_golden_seed_v1.json", help="Path to golden dataset")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="TrustRAG API base URL")
    parser.add_argument("--margin", type=float, default=0.20, help="FlashRank relative margin")
    parser.add_argument("--min-score", type=float, default=0.08, help="FlashRank min score floor")
    args = parser.parse_args()

    runner = PairedKEvaluationRunner(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_url=args.base_url,
        margin=args.margin,
        min_score=args.min_score,
    )
    runner.initialize()
    runner.run()


if __name__ == "__main__":
    main()
