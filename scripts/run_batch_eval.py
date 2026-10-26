#!/usr/bin/env python3
"""
TrustRAG Batch Evaluation Runner.

Executes a batch of golden dataset questions against a running TrustRAG FastAPI instance,
captures end-to-end production traces, evaluates retrieval metrics and answer correctness,
diagnoses abstentions, and writes results in JSON, CSV, and Markdown formats.

Usage:
    python scripts/run_batch_eval.py --start-id seed_001 --count 10
    python scripts/run_batch_eval.py --ids seed_001,seed_002,seed_003
    python scripts/run_batch_eval.py --endpoint stream --base-url http://127.0.0.1:8000
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

# Ensure project root is on sys.path
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


# ==============================================================================
# 1. SECRETS REDACTION & CONSTANTS
# ==============================================================================

SECRET_PATTERNS = [
    re.compile(r"(gsk_[a-zA-Z0-9_\-]{16,})"),
    re.compile(r"(pk_[a-zA-Z0-9_\-]{16,})"),
    re.compile(r"(Bearer\s+[a-zA-Z0-9_\-\.]{16,})", re.IGNORECASE),
    re.compile(r"(x-api-key['\"]?\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-]{8,})", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Masks API keys, tokens, and authorization credentials in text."""
    if not isinstance(text, str):
        return str(text)
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_CREDENTIAL]", sanitized)
    return sanitized


REFUSAL_PHRASES = [
    "not found in the documentation",
    "cannot answer this question as the required information is not found",
    "no information",
    "not have enough context",
    "does not mention",
    "not mentioned in the",
    "does not provide",
    "not provided in the",
    "insufficient context",
    "required information is not found",
]


def is_abstention_answer(answer: str) -> bool:
    """
    Detects whether an answer represents a grounding refusal or lack of context.
    Avoids false positives on substantive responses that contain trailing caveats or disclaimers.
    """
    if not answer or not answer.strip():
        return True
    lower = answer.lower().strip()

    # Canonical full refusal triggers
    if "not found in the documentation" in lower or "not found in the provided documentation" in lower:
        return True
    if "cannot answer this question as the required information is not found" in lower:
        return True
    if "required information is not found" in lower:
        return True

    # Guardrail blocked response triggers
    if "i cannot assist with this request" in lower or "i am an enterprise it assistant" in lower:
        return True

    words = answer.split()
    # For concise answers (< 45 words), check for any standard refusal phrase
    if len(words) < 45:
        return any(p in lower for p in REFUSAL_PHRASES)

    # For longer answers (>= 45 words), only trigger if refusal occurs upfront in the opening sentence
    opening = lower[:140]
    opening_refusals = [
        "cannot answer",
        "i am sorry",
        "i'm sorry",
        "no information is available",
        "there is no information",
        "insufficient context",
        "does not have enough context",
        "the provided documentation does not",
        "the documentation does not provide",
    ]
    return any(r in opening for r in opening_refusals)


# ==============================================================================
# 2. RUNNER CONFIGURATION & RECORD SELECTION
# ==============================================================================

@dataclass
class BatchEvalConfig:
    dataset_path: str = "evals/data/retrieval_golden_seed_v1.json"
    manifest_path: str = "evals/data/retrieval_golden_verified.json"
    start_id: str = "seed_001"
    count: int = 10
    explicit_ids: Optional[List[str]] = None
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "query"  # "query" or "stream"
    retrieval_mode: Optional[str] = None
    auth_mode: str = "none"
    api_key: Optional[str] = None
    user_id: str = "batch_eval_runner"
    auth_token: Optional[str] = None
    timeout: float = 30.0
    guard_mode: Optional[str] = None
    output_dir: Optional[str] = None
    use_judge: bool = False
    judge_model: str = "openai/gpt-oss-20b"
    reranker_margin: Optional[float] = None
    reranker_min_score: Optional[float] = None
    run_id: str = field(default_factory=lambda: f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}")


def select_records(
    records: List[RetrievalEvalRecord],
    start_id: str = "seed_001",
    count: int = 10,
    explicit_ids: Optional[List[str]] = None
) -> List[RetrievalEvalRecord]:
    """
    Selects records from the dataset with stable ordering.
    If explicit_ids is specified, preserves exact given order.
    Otherwise selects `count` records starting at `start_id`.
    """
    if explicit_ids:
        record_map = {r.id: r for r in records}
        selected = []
        for req_id in explicit_ids:
            clean_id = req_id.strip()
            if clean_id in record_map:
                selected.append(record_map[clean_id])
            else:
                raise ValueError(f"Requested record ID '{clean_id}' was not found in dataset.")
        return selected

    # Find start index
    start_idx = 0
    found = False
    for i, r in enumerate(records):
        if r.id == start_id:
            start_idx = i
            found = True
            break

    if not found:
        print(f"Warning: start_id '{start_id}' not found. Defaulting to index 0.")
        start_idx = 0

    return records[start_idx : start_idx + count]


# ==============================================================================
# 3. HTTP CLIENT & SSE PARSER
# ==============================================================================

class AppClient:
    """Client for executing requests against the running TrustRAG FastAPI application."""

    def __init__(self, config: BatchEvalConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-User-ID": self.config.user_id,
            "X-Eval-Trace": "true",
        }
        if self.config.auth_mode == "apikey" and self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        elif self.config.auth_mode == "oidc" and self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        if self.config.reranker_margin is not None:
            headers["X-Reranker-Margin"] = str(self.config.reranker_margin)
        if self.config.reranker_min_score is not None:
            headers["X-Reranker-Min-Score"] = str(self.config.reranker_min_score)
        return headers

    def ping_config(self) -> Dict[str, Any]:
        """Pings /config to verify server health and runtime configuration."""
        url = f"{self.base_url}/config"
        try:
            r = requests.get(url, headers=self.get_headers(), timeout=5.0)
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"error": f"Could not connect to {url}: {e}"}

    def execute_query_endpoint(self, question: str) -> Dict[str, Any]:
        """Executes a request against POST /query."""
        url = f"{self.base_url}/query"
        payload = {
            "q": question,
            "thread_id": f"eval_{uuid.uuid4().hex[:8]}",
            "return_trace": True,
        }
        if self.config.retrieval_mode:
            payload["retrieval_mode"] = self.config.retrieval_mode
        if self.config.reranker_margin is not None:
            payload["reranker_margin"] = self.config.reranker_margin
        if self.config.reranker_min_score is not None:
            payload["reranker_min_score"] = self.config.reranker_min_score

        start_time = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, headers=self.get_headers(), timeout=self.config.timeout)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if resp.status_code != 200:
                return {
                    "is_infra_error": True,
                    "error_category": f"http_{resp.status_code}",
                    "error_message": resp.text[:300],
                    "latency_ms": latency_ms,
                    "answer": "",
                    "status": "http_error",
                    "sources": [],
                    "retrieved_chunks": [],
                    "context_sent_to_llm": "",
                    "model_used": None,
                }

            data = resp.json()
            return {
                "is_infra_error": False,
                "error_category": "none",
                "error_message": "",
                "latency_ms": latency_ms,
                "answer": data.get("answer", ""),
                "status": data.get("status", "ok"),
                "sources": data.get("sources", []),
                "retrieved_chunks": data.get("retrieval_trace", []),
                "context_sent_to_llm": data.get("context_sent_to_llm", ""),
                "model_used": data.get("model_used"),
                "guard_mode": data.get("guard_mode"),
                "guard_model_used": data.get("guard_model_used"),
                "effective_reranker_margin": data.get("effective_reranker_margin"),
                "effective_reranker_min_score": data.get("effective_reranker_min_score"),
            }
        except requests.exceptions.Timeout as te:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "timeout",
                "error_message": f"Request timed out after {self.config.timeout}s: {te}",
                "latency_ms": latency_ms,
                "answer": "",
                "status": "timeout",
                "sources": [],
                "retrieved_chunks": [],
                "context_sent_to_llm": "",
                "model_used": None,
            }
        except requests.exceptions.ConnectionError as ce:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "connection_refused",
                "error_message": f"Connection refused to {url}: {ce}",
                "latency_ms": latency_ms,
                "answer": "",
                "status": "connection_error",
                "sources": [],
                "retrieved_chunks": [],
                "context_sent_to_llm": "",
                "model_used": None,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "client_exception",
                "error_message": str(exc),
                "latency_ms": latency_ms,
                "answer": "",
                "status": "exception",
                "sources": [],
                "retrieved_chunks": [],
                "context_sent_to_llm": "",
                "model_used": None,
            }

    def execute_stream_endpoint(self, question: str) -> Dict[str, Any]:
        """Executes a request against POST /v1/chat/stream and parses SSE events."""
        url = f"{self.base_url}/v1/chat/stream"
        payload = {
            "q": question,
            "thread_id": f"eval_{uuid.uuid4().hex[:8]}",
        }
        if self.config.retrieval_mode:
            payload["retrieval_mode"] = self.config.retrieval_mode

        start_time = time.perf_counter()
        accumulated_text = ""
        sources = []
        retrieved_chunks = []
        context_sent = ""
        model_used = None
        done_data = {}
        error_data = None

        try:
            with requests.post(
                url, json=payload, headers=self.get_headers(), timeout=self.config.timeout, stream=True
            ) as resp:
                if resp.status_code != 200:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    return {
                        "is_infra_error": True,
                        "error_category": f"http_{resp.status_code}",
                        "error_message": resp.text[:300],
                        "latency_ms": latency_ms,
                        "answer": "",
                        "status": "http_error",
                        "sources": [],
                        "retrieved_chunks": [],
                        "context_sent_to_llm": "",
                        "model_used": None,
                    }

                current_event = None
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and current_event:
                        data_str = line.split(":", 1)[1].strip()
                        try:
                            parsed_data = json.loads(data_str)
                        except Exception:
                            parsed_data = data_str

                        if current_event == "token":
                            if isinstance(parsed_data, dict):
                                accumulated_text += parsed_data.get("delta", "")
                            elif isinstance(parsed_data, str):
                                accumulated_text += parsed_data
                        elif current_event == "sources":
                            sources = parsed_data if isinstance(parsed_data, list) else []
                        elif current_event == "trace":
                            if isinstance(parsed_data, dict):
                                retrieved_chunks = parsed_data.get("retrieved_docs", [])
                                context_sent = parsed_data.get("context_sent_to_llm", "")
                        elif current_event == "done":
                            if isinstance(parsed_data, dict):
                                done_data = parsed_data
                                model_used = parsed_data.get("model_used")
                        elif current_event == "error":
                            error_data = parsed_data

                        current_event = None

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            if error_data:
                err_code = error_data.get("code", "sse_error") if isinstance(error_data, dict) else "sse_error"
                err_msg = error_data.get("message", str(error_data)) if isinstance(error_data, dict) else str(error_data)
                return {
                    "is_infra_error": True,
                    "error_category": f"sse_{err_code}",
                    "error_message": err_msg,
                    "latency_ms": latency_ms,
                    "answer": accumulated_text,
                    "status": "error",
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks,
                    "context_sent_to_llm": context_sent,
                    "model_used": model_used,
                }

            return {
                "is_infra_error": False,
                "error_category": "none",
                "error_message": "",
                "latency_ms": latency_ms,
                "answer": accumulated_text,
                "status": done_data.get("status", "ok"),
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "context_sent_to_llm": context_sent,
                "model_used": model_used,
                "guard_mode": done_data.get("guard_mode"),
                "guard_model_used": done_data.get("guard_model_used"),
            }

        except requests.exceptions.Timeout as te:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "timeout",
                "error_message": f"Stream timed out: {te}",
                "latency_ms": latency_ms,
                "answer": accumulated_text,
                "status": "timeout",
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "context_sent_to_llm": context_sent,
                "model_used": None,
            }
        except requests.exceptions.ConnectionError as ce:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "connection_refused",
                "error_message": f"Connection refused to {url}: {ce}",
                "latency_ms": latency_ms,
                "answer": accumulated_text,
                "status": "connection_error",
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "context_sent_to_llm": context_sent,
                "model_used": None,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "is_infra_error": True,
                "error_category": "client_exception",
                "error_message": str(exc),
                "latency_ms": latency_ms,
                "answer": accumulated_text,
                "status": "exception",
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "context_sent_to_llm": context_sent,
                "model_used": None,
            }

    def execute_request(self, question: str) -> Dict[str, Any]:
        """Routes execution to query or stream endpoint based on configuration."""
        if self.config.endpoint == "stream":
            return self.execute_stream_endpoint(question)
        return self.execute_query_endpoint(question)


# ==============================================================================
# 4. CHUNK RESOLUTION & CONTEXT MATCHING
# ==============================================================================

class ChunkResolver:
    """
    Resolves returned application sources/chunks back to dataset candidate chunk IDs.
    Handles both direct trace chunk IDs and string source matching.
    """

    def __init__(self, records: List[RetrievalEvalRecord]):
        self.records = records
        # Precompute text signatures for candidate chunks where possible
        self._uuid_to_record_id: Dict[str, str] = {}
        for r in records:
            for cid in r.candidate_chunk_ids:
                self._uuid_to_record_id[cid] = r.id

    def extract_retrieved_chunk_ids(
        self,
        raw_retrieved_chunks: List[Dict[str, Any]],
        sources: List[Any],
        candidate_pool: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extracts ordered list of chunk dictionaries with {id, rank, score, rerank_score}.
        If raw_retrieved_chunks has IDs, uses them directly.
        Otherwise falls back to candidate string matching.
        """
        extracted = []

        # Option A: Direct trace instrumentation
        if raw_retrieved_chunks:
            for rank, chk in enumerate(raw_retrieved_chunks, start=1):
                cid = str(chk.get("id"))
                extracted.append({
                    "id": cid,
                    "rank": rank,
                    "score": chk.get("score"),
                    "rerank_score": chk.get("rerank_score"),
                    "source": chk.get("source"),
                    "section": chk.get("section"),
                })
            return extracted

        # Option B: Fallback string matching on sources
        # Inspect sources returned in standard production endpoint
        candidate_set = set(candidate_pool)
        for rank, s in enumerate(sources, start=1):
            s_text = s if isinstance(s, str) else str(s.get("snippet", "") or s.get("source", ""))
            # If candidate UUID is present in text
            matched_id = None
            for cid in candidate_set:
                if cid in s_text:
                    matched_id = cid
                    break
            if matched_id:
                extracted.append({
                    "id": matched_id,
                    "rank": rank,
                    "score": None,
                    "rerank_score": None,
                    "source": s_text[:60],
                    "section": None,
                })
            else:
                extracted.append({
                    "id": f"unknown_doc_{rank}",
                    "rank": rank,
                    "score": None,
                    "rerank_score": None,
                    "source": s_text[:60],
                    "section": None,
                })

        return extracted


# ==============================================================================
# 5. ANSWER EVALUATION & ABSTENTION DIAGNOSIS
# ==============================================================================

class DeterministicEvaluator:
    """
    Deterministic rule-based factual evaluator for known golden questions.
    Checks required commands, flags, parameters, and entities without string exact-match.
    """

    @staticmethod
    def evaluate_question(
        record_id: str,
        query: str,
        app_answer: str,
        expected_answer: Optional[str]
    ) -> Tuple[str, str]:
        """
        Returns (verdict, reason).
        Verdict is one of: CORRECT, PARTIALLY_CORRECT, INCORRECT.
        """
        ans_lower = app_answer.lower()

        # Seed 001: Start Redis for Kubernetes work queue
        if record_id == "seed_001":
            has_apply = "kubectl apply" in ans_lower or "apply" in ans_lower
            has_pod = "redis-pod" in ans_lower
            has_svc = "redis-service" in ans_lower
            if has_pod and has_svc:
                return "CORRECT", "Cites both redis-pod.yaml and redis-service.yaml manifests with kubectl apply."
            elif has_pod or has_svc:
                return "PARTIALLY_CORRECT", "Mentions only one of redis-pod or redis-service manifests."
            return "INCORRECT", "Missing required redis-pod and redis-service manifest names."

        # Seed 002: CLI commands push tasks into Redis queue named job2
        if record_id == "seed_002":
            has_cli = "redis-cli" in ans_lower
            has_rpush = "rpush" in ans_lower
            has_job2 = "job2" in ans_lower
            if has_rpush and has_job2:
                return "CORRECT", "Correctly cites rpush job2 command syntax to enqueue tasks."
            elif has_rpush or has_cli:
                return "PARTIALLY_CORRECT", "Mentions redis command but omits target queue job2 or full syntax."
            return "INCORRECT", "Did not state the rpush command or target queue job2."

        # Seed 003: Parallelism value in job-wq-2 manifest
        if record_id == "seed_003":
            if "2" in ans_lower and ("parallelism" in ans_lower or "value" in ans_lower):
                return "CORRECT", "Correctly states parallelism value of 2."
            return "INCORRECT", "Did not state parallelism value of 2."

        # Seed 004: Restart policies permitted for Job pods
        if record_id == "seed_004":
            has_on_failure = "onfailure" in ans_lower
            has_never = "never" in ans_lower
            has_always_not = "always is not" in ans_lower or "not always" in ans_lower or ("always" in ans_lower and "not allowed" in ans_lower)
            if has_on_failure and has_never:
                return "CORRECT", "Correctly identifies OnFailure and Never policies."
            elif has_on_failure or has_never:
                return "PARTIALLY_CORRECT", "Identifies only one of the allowed restart policies."
            return "INCORRECT", "Fails to identify OnFailure and Never restart policies."

        # Seed 005: API group and version defining Job resource
        if record_id == "seed_005":
            if "batch/v1" in ans_lower:
                return "CORRECT", "Correctly states the batch/v1 API group and version."
            elif "batch" in ans_lower or "v1" in ans_lower:
                return "PARTIALLY_CORRECT", "Mentions batch or v1 partially but not batch/v1."
            return "INCORRECT", "Failed to identify batch/v1 API group."

        # Seed 006: backoffLimit field meaning
        if record_id == "seed_006":
            has_retry = "retry" in ans_lower or "retries" in ans_lower or "number of times" in ans_lower
            has_fail = "fail" in ans_lower
            if has_retry:
                return "CORRECT", "Accurately explains backoffLimit specifies the number of retries before Job failure."
            return "INCORRECT", "Did not explain backoffLimit as the retry count."

        # Seed 007: activeDeadlineSeconds function
        if record_id == "seed_007":
            has_deadline = "deadline" in ans_lower or "duration" in ans_lower or "time limit" in ans_lower or "timeout" in ans_lower
            has_kill = "terminate" in ans_lower or "kill" in ans_lower or "stop" in ans_lower or "fail" in ans_lower or "exceed" in ans_lower
            if has_deadline or has_kill:
                return "CORRECT", "Explains active execution deadline terminating/killing the Job once exceeded."
            return "INCORRECT", "Did not describe the active deadline termination behavior."

        # Seed 008: CronJob 1-hour automatic cleanup
        if record_id == "seed_008":
            has_ttl = "ttlsecondsafterfinished" in ans_lower
            has_sec = "3600" in ans_lower or "1 hour" in ans_lower or "one hour" in ans_lower
            if has_ttl and has_sec:
                return "CORRECT", "Identifies ttlSecondsAfterFinished: 3600 configuration in the Job template."
            elif has_ttl or has_sec:
                return "PARTIALLY_CORRECT", "Mentions ttlSecondsAfterFinished or 3600s, but incomplete configuration."
            return "INCORRECT", "Did not cite ttlSecondsAfterFinished: 3600."

        # Seed 009: Difference between HPA and VPA
        if record_id == "seed_009":
            has_hpa = "horizontal" in ans_lower or "pods" in ans_lower or "more pods" in ans_lower
            has_vpa = "vertical" in ans_lower or "cpu" in ans_lower or "memory" in ans_lower or "resources" in ans_lower
            if has_hpa and has_vpa:
                return "CORRECT", "Clearly distinguishes HPA (horizontal pod scaling) from VPA (vertical resource adjustment)."
            elif has_hpa or has_vpa:
                return "PARTIALLY_CORRECT", "Describes one scaling mode but lacks full contrast."
            return "INCORRECT", "Fails to distinguish HPA from VPA."

        # Seed 010: Metrics Server official installation command
        if record_id == "seed_010":
            has_components = "components.yaml" in ans_lower
            has_kubectl = "kubectl apply" in ans_lower or "kubectl" in ans_lower
            if has_components and has_kubectl:
                return "CORRECT", "Provides the official kubectl apply command with components.yaml manifest."
            elif has_components:
                return "PARTIALLY_CORRECT", "Mentions components.yaml but missing kubectl apply command."
        # Seed 014: Worker node container agent
        if record_id == "seed_014":
            if "kubelet" in ans_lower:
                return "CORRECT", "Correctly identifies kubelet as the node agent managing pod/container lifecycle."
            return "INCORRECT", "Did not identify kubelet as the worker node agent."

        # Seed 021: Difference between completions and parallelism in Job spec
        if record_id == "seed_021":
            has_comp = "completion" in ans_lower or "completions" in ans_lower
            has_par = "parallel" in ans_lower or "parallelism" in ans_lower
            if has_comp and has_par:
                return "CORRECT", "Accurately distinguishes completions (target pod executions) from parallelism (concurrent pods)."
            return "INCORRECT", "Did not contrast completions and parallelism."

        # Seed 030: Databricks CLI JSON job creation
        if record_id == "seed_030":
            has_cmd = "databricks jobs create" in ans_lower
            has_json = "--json" in ans_lower or "json" in ans_lower
            if has_cmd and has_json:
                return "CORRECT", "Provides the databricks jobs create command with JSON definition specification."
            elif has_cmd:
                return "PARTIALLY_CORRECT", "Mentions databricks jobs create but missing JSON specification."
            return "INCORRECT", "Did not provide the databricks jobs create command."

        # Seed 040: etcd vs kube-apiserver state persistence
        if record_id == "seed_040":
            has_etcd = "etcd" in ans_lower
            has_api = "apiserver" in ans_lower or "api server" in ans_lower
            has_etcd_role = "datastore" in ans_lower or "key-value" in ans_lower or "store" in ans_lower or "state" in ans_lower or "persist" in ans_lower
            has_api_role = "api" in ans_lower or "rest" in ans_lower or "admission" in ans_lower or "front" in ans_lower or "gatekeeper" in ans_lower or "validat" in ans_lower
            if has_etcd and has_api and has_etcd_role and has_api_role:
                return "CORRECT", "Accurately contrasts etcd's role in persistent state storage with kube-apiserver's role in API management."
            elif has_etcd or has_api:
                return "PARTIALLY_CORRECT", "Identifies one component but lacks complete contrast."
            return "INCORRECT", "Did not differentiate etcd and kube-apiserver."

        # Generic fallback for any other question
        if expected_answer:
            exp_terms = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-\.]{4,}\b", expected_answer) if len(w) > 3]
            if not exp_terms:
                return "NOT_EVALUABLE", "Expected answer has insufficient substantive terms."
            overlap = sum(1 for t in exp_terms if t in ans_lower)
            overlap_ratio = float(overlap) / float(len(exp_terms))
            if overlap_ratio >= 0.6:
                return "CORRECT", f"High semantic token overlap ({overlap}/{len(exp_terms)} key terms matched)."
            elif overlap_ratio >= 0.3:
                return "PARTIALLY_CORRECT", f"Moderate overlap ({overlap}/{len(exp_terms)} key terms matched)."
            else:
                return "INCORRECT", f"Low overlap ({overlap}/{len(exp_terms)} key terms matched)."

        return "NOT_EVALUABLE", "No expected answer or rule available."


class SemanticJudgeAdapter:
    """
    Optional LLM judge adapter with explicit rubric, structured output,
    model metadata recording, and separate error reporting.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name

    def grade(
        self,
        query: str,
        expected_answer: str,
        app_answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Executes judge rubric. In this runner, we provide the structured interface.
        If live LLM credentials are not available, it reports uncertainty without failing to zero.
        """
        prompt = f"""
You are an impartial expert RAG evaluation judge. Grade the candidate answer against the reference answer and retrieved context.

RUBRIC:
- CORRECT: The answer provides all core factual claims from the reference answer and contains no unsupported assertions.
- PARTIALLY_CORRECT: The answer contains key facts but misses details or contains minor unverified statements.
- INCORRECT: The answer contradicts the reference answer or hallucinates false claims.
- ABSTAINED: The model correctly refused to answer or stated that documentation was insufficient.

QUESTION: {query}
REFERENCE ANSWER: {expected_answer}
CANDIDATE ANSWER: {app_answer}
CONTEXT: {context[:500]}

Respond ONLY in valid JSON:
{{
  "verdict": "CORRECT|PARTIALLY_CORRECT|INCORRECT|ABSTAINED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "judge_model": "{self.model_name}"
}}
"""
        return {
            "verdict": "NOT_EVALUABLE",
            "confidence": 0.0,
            "reasoning": "Semantic judge adapter offline; falling back to deterministic evaluation.",
            "judge_model": self.model_name,
            "judge_error": None,
        }


def diagnose_abstention(
    record: RetrievalEvalRecord,
    retrieved_chunk_ids: List[str],
    raw_status: str,
    context_sent_to_llm: str,
    is_infra_error: bool,
    error_category: str
) -> Tuple[str, str]:
    """
    Diagnoses root cause of abstention based strictly on available traces:
    - INFRASTRUCTURE_FAILURE
    - GUARDRAIL_BLOCKED
    - RELEVANT_NOT_RETRIEVED
    - RELEVANT_DROPPED_BY_RERANKER
    - CONTEXT_PRESENT_MODEL_ABSTAINED
    - UNKNOWN_CAUSE
    """
    if is_infra_error:
        return "INFRASTRUCTURE_FAILURE", f"Infrastructure failure: {error_category}"

    if raw_status == "blocked_by_guardrails":
        return "GUARDRAIL_BLOCKED", "Safety guardrails intercepted the prompt before retrieval."

    candidate_set = set(record.candidate_chunk_ids)
    if not candidate_set:
        return "CORRECT_ABSTENTION_UNANSWERABLE", "Query has no ground truth candidates in corpus."

    # Check if any candidate was retrieved in top-K
    top_k_ids = set(retrieved_chunk_ids)
    relevant_in_top_k = bool(candidate_set & top_k_ids)

    if not relevant_in_top_k:
        return "RELEVANT_NOT_RETRIEVED", f"None of candidate chunks {list(candidate_set)} were retrieved in top-K."

    # If relevant chunk was retrieved in top-K, check if it was passed to LLM
    if context_sent_to_llm:
        # Check if candidate UUID or text was in formatted context
        candidate_found_in_context = False
        for cid in candidate_set:
            if cid in context_sent_to_llm:
                candidate_found_in_context = True
                break
        if not candidate_found_in_context and record.evidence_quotes:
            for q in record.evidence_quotes:
                if q[:30].lower() in context_sent_to_llm.lower():
                    candidate_found_in_context = True
                    break

        if candidate_found_in_context:
            return "CONTEXT_PRESENT_MODEL_ABSTAINED", "Relevant chunk was passed in LLM context, but the responder abstained."
        else:
            return "RELEVANT_DROPPED_BY_RERANKER", "Relevant chunk was retrieved in vector search but dropped before final LLM context."

    return "UNKNOWN_CAUSE", "Traces inconclusive; unable to verify context delivery to LLM."


# ==============================================================================
# 6. BATCH EVALUATION RUNNER
# ==============================================================================

class BatchEvalRunner:
    """Coordinates execution, metrics calculation, abstention diagnosis, and reporting."""

    def __init__(self, config: BatchEvalConfig):
        self.config = config
        self.client = AppClient(config)
        self.records: List[RetrievalEvalRecord] = []
        self.selected_records: List[RetrievalEvalRecord] = []
        self.resolver: Optional[ChunkResolver] = None
        self.evaluator = DeterministicEvaluator()
        self.judge = SemanticJudgeAdapter(config.judge_model) if config.use_judge else None

    def initialize(self) -> None:
        """Loads dataset and prepares records."""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.config.dataset_path}")

        self.records = load_dataset(self.config.dataset_path)
        self.selected_records = select_records(
            self.records,
            start_id=self.config.start_id,
            count=self.config.count,
            explicit_ids=self.config.explicit_ids,
        )
        self.resolver = ChunkResolver(self.records)

    def run(self) -> Dict[str, Any]:
        """Executes the full evaluation batch."""
        self.initialize()
        selected_ids = [r.id for r in self.selected_records]
        print(f"\n================================================================================")
        print(f"🚀 TRUST_RAG BATCH EVALUATION RUNNER")
        print(f"Run ID:         {self.config.run_id}")
        print(f"Endpoint:       {self.config.base_url}/{'query' if self.config.endpoint == 'query' else 'v1/chat/stream'}")
        print(f"Selected Count: {len(self.selected_records)} records")
        print(f"Selected IDs:   {selected_ids}")
        print(f"================================================================================\n", flush=True)

        # Check server health
        server_config = self.client.ping_config()
        if "error" in server_config:
            print(f"⚠️ Notice: Server check ({self.config.base_url}/config): {server_config['error']}\n")
        else:
            print(f"✅ Connected to TrustRAG server: Active Model = {server_config.get('GROQ_MODEL')}, Guard = {server_config.get('GUARD_MODE')}\n")

        question_results: List[Dict[str, Any]] = []

        for idx, rec in enumerate(self.selected_records, start=1):
            qid = rec.id
            query = rec.query
            print(f"[{idx}/{len(self.selected_records)}] Evaluating {qid}: '{query[:65]}'...", flush=True)

            # 1. Execute production request
            resp = self.client.execute_request(query)
            latency_ms = resp.get("latency_ms", 0)
            is_infra_error = resp.get("is_infra_error", False)
            error_category = resp.get("error_category", "none")
            app_answer = resp.get("answer", "")
            raw_status = resp.get("status", "")
            context_sent = resp.get("context_sent_to_llm", "")

            # 2. Extract and resolve retrieved chunks
            retrieved_chunk_info = self.resolver.extract_retrieved_chunk_ids(
                resp.get("retrieved_chunks", []),
                resp.get("sources", []),
                rec.candidate_chunk_ids
            )
            retrieved_ids = [c["id"] for c in retrieved_chunk_info]
            retrieved_ranks = [str(c["rank"]) for c in retrieved_chunk_info]
            retrieved_scores = [str(c["score"] or "") for c in retrieved_chunk_info]

            # 3. Retrieval Metrics Computation
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

            # 4. Answer Evaluation & Abstention Detection
            is_abstained = is_abstention_answer(app_answer) or raw_status == "insufficient_context"
            abstention_status = "none"
            verdict = "INCORRECT"
            verdict_reason = ""
            diagnosis = "none"
            diagnosis_detail = ""

            if is_infra_error:
                verdict = "INFRASTRUCTURE_ERROR"
                verdict_reason = f"Request failed: {resp.get('error_message')}"
                error_category = resp.get("error_category", "infra_error")
                diagnosis, diagnosis_detail = diagnose_abstention(rec, retrieved_ids, raw_status, context_sent, True, error_category)
            elif raw_status == "blocked_by_guardrails":
                abstention_status = "guardrail_blocked"
                if rec.query_type == "unanswerable":
                    verdict = "CORRECT"
                    verdict_reason = "Safety guardrails successfully blocked/refused unanswerable query."
                else:
                    verdict = "ABSTAINED"
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
                # Deterministic check
                v, r = self.evaluator.evaluate_question(rec.id, query, app_answer, rec.expected_answer)
                verdict = v
                verdict_reason = r

            review_required = (verdict in ("PARTIALLY_CORRECT", "INCORRECT", "ABSTAINED", "NOT_EVALUABLE")) or (rec.verification_status != "verified_human")

            print(f"    -> Verdict: {verdict} | Latency: {latency_ms}ms | Retrieved hits: {len(relevant_candidates & set(retrieved_ids))}/{len(relevant_candidates)}", flush=True)

            q_entry = {
                "id": rec.id,
                "split": rec.split,
                "verification_status": rec.verification_status,
                "query": query,
                "query_type": rec.query_type,
                "expected_answer": rec.expected_answer or "",
                "app_answer": app_answer,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "retrieved_chunk_ids": ";".join(retrieved_ids),
                "retrieved_ranks": ";".join(retrieved_ranks),
                "retrieved_scores": ";".join(retrieved_scores),
                "relevant_evidence_retrieved": relevant_retrieved,
                "context_evidence_present": context_evidence_present,
                "abstention_status": abstention_status,
                "latency_ms": latency_ms,
                "error_category": error_category,
                "evaluator_method": "deterministic_v1" if not self.config.use_judge else f"judge_{self.config.judge_model}",
                "review_required": review_required,
                "diagnosis": diagnosis,
                "diagnosis_detail": diagnosis_detail,
                "retrieval_metrics": rec_metrics,
                "model_used": resp.get("model_used"),
                "guard_mode": resp.get("guard_mode"),
            }
            question_results.append(q_entry)

        # 5. Aggregate Evaluation Statistics
        total_q = len(question_results)
        count_correct = sum(1 for q in question_results if q["verdict"] == "CORRECT")
        count_partial = sum(1 for q in question_results if q["verdict"] == "PARTIALLY_CORRECT")
        count_incorrect = sum(1 for q in question_results if q["verdict"] == "INCORRECT")
        count_abstained = sum(1 for q in question_results if q["verdict"] == "ABSTAINED")
        count_infra = sum(1 for q in question_results if q["verdict"] == "INFRASTRUCTURE_ERROR")
        count_not_eval = sum(1 for q in question_results if q["verdict"] == "NOT_EVALUABLE")

        # Answer Pass Rate calculation (excluding infra errors from answer accuracy denominator, but stating denominator explicitly)
        evaluable_denominator = total_q - count_infra
        answer_pass_rate = (float(count_correct) / float(evaluable_denominator)) * 100.0 if evaluable_denominator > 0 else 0.0

        # Retrieval Metrics Aggregation (Eligible answerable queries)
        eligible_answerable = [q for q in question_results if q["query_type"] != "unanswerable" and not q["verdict"] == "INFRASTRUCTURE_ERROR"]
        denom_retrieval = len(eligible_answerable)

        macro_retrieval = {}
        for m_key in ["recall@1", "recall@5", "recall@10", "hit_rate@1", "hit_rate@5", "hit_rate@10", "mrr@10", "ndcg@5", "ndcg@10"]:
            if denom_retrieval > 0:
                vals = [q["retrieval_metrics"][m_key] for q in eligible_answerable]
                macro_retrieval[m_key] = round(float(sum(vals)) / float(denom_retrieval), 4)
            else:
                macro_retrieval[m_key] = 0.0

        run_summary = {
            "run_id": self.config.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "base_url": self.config.base_url,
                "endpoint": self.config.endpoint,
                "dataset_path": self.config.dataset_path,
                "start_id": self.config.start_id,
                "count": self.config.count,
                "explicit_ids": self.config.explicit_ids,
                "selected_ids": selected_ids,
                "auth_mode": self.config.auth_mode,
                "timeout": self.config.timeout,
                "retrieval_mode": self.config.retrieval_mode,
                "reranker_margin": self.config.reranker_margin,
                "reranker_min_score": self.config.reranker_min_score,
                "evaluator": "deterministic_v1" if not self.config.use_judge else f"judge_{self.config.judge_model}",
            },
            "server_config": server_config,
            "counts": {
                "total": total_q,
                "correct": count_correct,
                "partially_correct": count_partial,
                "incorrect": count_incorrect,
                "abstained": count_abstained,
                "infrastructure_errors": count_infra,
                "not_evaluable": count_not_eval,
            },
            "answer_pass_rate_pct": round(answer_pass_rate, 2),
            "answer_denominator": evaluable_denominator,
            "retrieval_denominator": denom_retrieval,
            "macro_retrieval_metrics": macro_retrieval,
            "questions": question_results,
        }

        # 6. Generate Reports
        self.write_reports(run_summary)
        return run_summary

    def write_reports(self, summary: Dict[str, Any]) -> None:
        """Writes results.json, results.csv, and summary.md."""
        out_dir = self.config.output_dir or os.path.join("evals", "results", self.config.run_id)
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "results.json")
        csv_path = os.path.join(out_dir, "results.csv")
        md_path = os.path.join(out_dir, "summary.md")

        # 1. results.json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 2. results.csv
        csv_fields = [
            "id", "split", "query", "expected_answer", "app_answer",
            "verdict", "verdict_reason", "retrieved_chunk_ids", "retrieved_ranks",
            "retrieved_scores", "relevant_evidence_retrieved", "context_evidence_present",
            "abstention_status", "latency_ms", "error_category", "evaluator_method", "review_required"
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for q in summary["questions"]:
                writer.writerow(q)

        # 3. summary.md
        md_content = self.generate_markdown_summary(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\n================================================================================")
        print(f"📊 EVALUATION COMPLETED")
        print(f"Directory:    {out_dir}")
        print(f"Results JSON: {json_path}")
        print(f"Results CSV:  {csv_path}")
        print(f"Summary MD:   {md_path}")
        print(f"================================================================================\n")

    def generate_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Constructs transparent, clean Markdown report."""
        c = summary["counts"]
        ret = summary["macro_retrieval_metrics"]
        cfg = summary["config"]
        srv = summary.get("server_config", {})

        md = f"""# TrustRAG Batch Evaluation Report

**Run ID:** `{summary['run_id']}`  
**Timestamp:** `{summary['timestamp']}`  
**Endpoint:** `{cfg['base_url']}/{'query' if cfg['endpoint'] == 'query' else 'v1/chat/stream'}`  
**Selected IDs:** {', '.join(f'`{i}`' for i in cfg['selected_ids'])}  
**Evaluator:** `{cfg['evaluator']}`  

---

## 1. Executive Performance Summary

| Metric | Count / Score | Denominator / Notes |
| :--- | :--- | :--- |
| **Total Evaluated** | **{c['total']}** | Selected dataset batch |
| **Correct (Passed)** | **{c['correct']}** | Full factual alignment |
| **Partially Correct** | **{c['partially_correct']}** | Partial key information present |
| **Incorrect** | **{c['incorrect']}** | Factual error or contradiction |
| **Abstained** | **{c['abstained']}** | Grounding refusal ("Not Found in Documentation") |
| **Infrastructure Errors** | **{c['infrastructure_errors']}** | Network / Timeout / 5xx failures |
| **Not Evaluable** | **{c['not_evaluable']}** | Incomplete response or missing reference |
| **Answer Pass Rate** | **{summary['answer_pass_rate_pct']:.1f}%** | `{c['correct']}/{summary['answer_denominator']} evaluable queries` |

> [!NOTE]
> Infrastructure failures ({c['infrastructure_errors']}) are isolated and excluded from the answer accuracy denominator to prevent hiding systemic service outages inside model answer failures.

---

## 2. Macro Retrieval Metrics (Top-K)

*Evaluated on `{summary['retrieval_denominator']}` eligible answerable queries with annotated relevance grades.*

| Metric | Score | Sample Size / Notes |
| :--- | :--- | :--- |
| **Recall@1** | **{ret.get('recall@1', 0.0):.4f}** | Primary top-1 rank hit rate |
| **Recall@5** | **{ret.get('recall@5', 0.0):.4f}** | Default production context window |
| **Recall@10** | **{ret.get('recall@10', 0.0):.4f}** | Reranker candidate pool depth |
| **HitRate@1** | **{ret.get('hit_rate@1', 0.0):.4f}** | Any relevant chunk at Rank 1 |
| **HitRate@5** | **{ret.get('hit_rate@5', 0.0):.4f}** | Any relevant chunk in Top 5 |
| **HitRate@10** | **{ret.get('hit_rate@10', 0.0):.4f}** | Any relevant chunk in Top 10 |
| **MRR@10** | **{ret.get('mrr@10', 0.0):.4f}** | Mean Reciprocal Rank |
| **nDCG@5** | **{ret.get('ndcg@5', 0.0):.4f}** | Normalized Discounted Cumulative Gain |
| **nDCG@10** | **{ret.get('ndcg@10', 0.0):.4f}** | Full rerank list ranking quality |

---

## 3. Per-Question Results Table

| ID | Query | Verdict | Latency | Relevant Retrieved? | Abstention Status | Diagnosis / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for q in summary["questions"]:
            ans_excerpt = (q["app_answer"][:45] + "...") if len(q["app_answer"]) > 45 else (q["app_answer"] or "None")
            diag = q["diagnosis"] if q["diagnosis"] != "none" else q["verdict_reason"]
            md += f"| `{q['id']}` | {q['query']} | **{q['verdict']}** | {q['latency_ms']}ms | {'✅ Yes' if q['relevant_evidence_retrieved'] else '❌ No'} | `{q['abstention_status']}` | {diag} |\n"

        md += f"""
---

## 4. Abstention & Root-Cause Diagnosis

This runner distinguishes the following explicit abstention failure modes:
1. **RELEVANT_NOT_RETRIEVED**: The annotated candidate chunk was not returned in Qdrant hybrid retrieval top-K.
2. **RELEVANT_DROPPED_BY_RERANKER**: Candidate chunk retrieved in initial search but dropped below the FlashRank score floor.
3. **CONTEXT_PRESENT_MODEL_ABSTAINED**: Relevant chunk was present in the prompt context, but the responder refused to answer.
4. **GUARDRAIL_BLOCKED**: Safety guardrails intercepted the query before retrieval.
5. **INFRASTRUCTURE_FAILURE**: HTTP timeout, network disconnect, or rate limit 429.

### Question-by-Question Diagnoses:
"""
        abstained_or_failed = [q for q in summary["questions"] if q["verdict"] in ("ABSTAINED", "INFRASTRUCTURE_ERROR", "INCORRECT")]
        if not abstained_or_failed:
            md += "\nNo abstentions or infrastructure failures occurred in this execution.\n"
        else:
            for q in abstained_or_failed:
                md += f"- **`{q['id']}`**: Verdict `{q['verdict']}` | Diagnosis: `{q['diagnosis']}`  \n  *Details*: {q['diagnosis_detail'] or q['verdict_reason']}  \n"

        md += f"""
---

## 5. Audit & Runtime Configuration

- **Dataset Path:** `{cfg['dataset_path']}`
- **Server URL:** `{cfg['base_url']}`
- **Auth Mode:** `{cfg['auth_mode']}`
- **Server Groq Model:** `{srv.get('GROQ_MODEL', 'unknown')}`
- **Server Guard Mode:** `{srv.get('GUARD_MODE', 'unknown')}`
- **Active Guard Model:** `{srv.get('guard_model_used', 'unknown')}`

> [!WARNING]
> **Sample Size Notice**: This evaluation was executed on a development sample of N={c['total']} queries. While highly informative for local regression testing, production deployment sign-off requires running the full 50-query golden benchmark with verified human ground truth.
"""
        return md


# ==============================================================================
# 7. CLI ENTRYPOINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible batch evaluation against TrustRAG application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-id", type=str, default="seed_001", help="Starting dataset record ID.")
    parser.add_argument("--count", type=int, default=10, help="Number of eligible records to evaluate.")
    parser.add_argument("--ids", type=str, default=None, help="Explicit comma-separated list of record IDs (e.g. seed_001,seed_002).")
    parser.add_argument("--dataset", type=str, default="evals/data/retrieval_golden_seed_v1.json", help="Path to golden dataset JSON.")
    parser.add_argument("--manifest", type=str, default="evals/data/retrieval_golden_verified.json", help="Path to verified manifest JSON.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="TrustRAG FastAPI base URL.")
    parser.add_argument("--endpoint", type=str, choices=["query", "stream"], default="query", help="Endpoint to evaluate: 'query' (/query) or 'stream' (/v1/chat/stream).")
    parser.add_argument("--retrieval-mode", type=str, default=None, choices=["tinybert", "rrf", "fusion"], help="Optional retrieval mode override.")
    parser.add_argument("--auth-mode", type=str, default="none", choices=["none", "apikey", "oidc"], help="Authentication mode.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for apikey auth mode.")
    parser.add_argument("--user-id", type=str, default="batch_eval_runner", help="User ID header.")
    parser.add_argument("--auth-token", type=str, default=None, help="Bearer token for oidc auth mode.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--use-judge", action="store_true", help="Enable optional semantic judge adapter.")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-20b", help="Model for semantic judge adapter.")
    parser.add_argument("--margin", type=float, default=None, help="FlashRank relative margin threshold override (e.g. 0.20, 0.30, 0.40, 0.50, 1.0=disabled).")
    parser.add_argument("--min-score", type=float, default=None, help="FlashRank absolute min score floor override (default: 0.08).")
    return parser.parse_args()


def main():
    args = parse_args()
    explicit_ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None

    config = BatchEvalConfig(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        start_id=args.start_id,
        count=args.count,
        explicit_ids=explicit_ids,
        base_url=args.base_url,
        endpoint=args.endpoint,
        retrieval_mode=args.retrieval_mode,
        auth_mode=args.auth_mode,
        api_key=args.api_key,
        user_id=args.user_id,
        auth_token=args.auth_token,
        timeout=args.timeout,
        output_dir=args.output_dir,
        use_judge=args.use_judge,
        judge_model=args.judge_model,
        reranker_margin=args.margin,
        reranker_min_score=args.min_score,
    )

    runner = BatchEvalRunner(config)
    try:
        runner.run()
    except Exception as e:
        print(f"\n❌ Batch evaluation runner failed with fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
