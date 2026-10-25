"""
TrustRAG — Guardrails Orchestrator (Defense in Depth)

Coordinates sequential execution of:
1. Stage 1: Single-call structured JSON input guard
2. Stage 2: NeMo Guardrails flow & policy adapter

Enforces strict AND semantics:
- Both guards must explicitly return ALLOW.
- If Stage 1 blocks or errors, Stage 2 is short-circuited and the request is rejected.
- If Stage 2 blocks, errors, or returns an unrecognized intent, the request is rejected.
- Emits structured Logfire audit telemetry.
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple
import time
import uuid
import logfire

from app.config import settings
from app.guardrails.single_call import (
    evaluate_single_call_guard,
    guard_single_call,
    GuardResult,
    DEFAULT_BLOCK_MESSAGE,
    INFRA_UNAVAILABLE_MESSAGE,
)


class GuardDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    INVALID_DECISION = "INVALID_DECISION"


def evaluate_combined_guard(
    message: str,
    request_id: Optional[str] = None,
) -> GuardResult:
    """
    Evaluates both single_call and NeMo guardrails sequentially with strict AND semantics.
    1. Runs single_call guard first. If blocked or infra failure, stops immediately (Stage 2 skipped).
    2. Runs NeMo guard only if Stage 1 explicitly approves.
    3. Continues to retrieval/downstream ONLY if NeMo also explicitly approves.
    4. Any block, error, timeout, or unrecognized intent halts the pipeline (Fail Closed).
    """
    from app.guardrails.rails import evaluate_nemo_guard

    req_id = request_id or str(uuid.uuid4())
    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # STAGE 1: Single-Call Guard
    # -------------------------------------------------------------------------
    s1_start = time.perf_counter()
    s1_res = evaluate_single_call_guard(message)
    s1_latency = int((time.perf_counter() - s1_start) * 1000)

    # Map Stage 1 decision
    s1_raw_decision = str(s1_res.get("decision", "")).strip().lower()
    if s1_raw_decision == "allow":
        s1_decision = GuardDecision.ALLOW
    elif s1_raw_decision == "block":
        s1_decision = GuardDecision.BLOCK
    elif s1_raw_decision == "infra_unavailable":
        s1_decision = GuardDecision.ERROR
    else:
        s1_decision = GuardDecision.INVALID_DECISION

    # Stage 1 Gating: if not ALLOW, short-circuit immediately
    if s1_decision != GuardDecision.ALLOW:
        total_latency = int((time.perf_counter() - start_time) * 1000)
        logfire.info(
            "🛡️ [Combined Guard] Stage 1 (single_call) rejected request",
            request_id=req_id,
            stage1_decision=s1_decision.value,
            stage1_reason=s1_res.get("reason"),
            stage1_latency_ms=s1_latency,
            stage2_status="SKIPPED",
            total_latency_ms=total_latency,
        )
        resp_text = s1_res.get("response_text") or (
            DEFAULT_BLOCK_MESSAGE if s1_decision == GuardDecision.BLOCK else INFRA_UNAVAILABLE_MESSAGE
        )
        reason_code = f"stage1_{s1_res.get('reason', s1_decision.value.lower())}"
        return GuardResult(
            True,
            resp_text,
            s1_res.get("model_used") or settings.GUARDRAIL_MODEL,
            reason_code,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: NeMo Guardrails (Defense in Depth)
    # -------------------------------------------------------------------------
    s2_start = time.perf_counter()
    s2_res = evaluate_nemo_guard(message)
    s2_latency = int((time.perf_counter() - s2_start) * 1000)

    # Map Stage 2 decision
    s2_raw_decision = str(s2_res.get("decision", "")).strip().lower()
    if s2_raw_decision == "allow":
        s2_decision = GuardDecision.ALLOW
    elif s2_raw_decision == "block":
        s2_decision = GuardDecision.BLOCK
    elif s2_raw_decision == "infra_unavailable":
        s2_decision = GuardDecision.ERROR
    else:
        s2_decision = GuardDecision.INVALID_DECISION

    total_latency = int((time.perf_counter() - start_time) * 1000)

    # Stage 2 Gating: must explicitly ALLOW
    if s2_decision != GuardDecision.ALLOW:
        logfire.info(
            "🛡️ [Combined Guard] Stage 2 (NeMo) rejected request",
            request_id=req_id,
            stage1_decision=s1_decision.value,
            stage2_decision=s2_decision.value,
            stage2_reason=s2_res.get("reason"),
            stage2_latency_ms=s2_latency,
            total_latency_ms=total_latency,
        )
        resp_text = s2_res.get("response_text") or (
            DEFAULT_BLOCK_MESSAGE if s2_decision in (GuardDecision.BLOCK, GuardDecision.INVALID_DECISION) else INFRA_UNAVAILABLE_MESSAGE
        )
        reason_code = f"stage2_{s2_res.get('reason', s2_decision.value.lower())}"
        return GuardResult(
            True,
            resp_text,
            s2_res.get("model_used") or settings.GUARDRAIL_MODEL,
            reason_code,
        )

    # -------------------------------------------------------------------------
    # BOTH GUARDS APPROVED
    # -------------------------------------------------------------------------
    logfire.info(
        "✅ [Combined Guard] Both guards explicitly approved request",
        request_id=req_id,
        stage1_decision="ALLOW",
        stage2_decision="ALLOW",
        stage1_latency_ms=s1_latency,
        stage2_latency_ms=s2_latency,
        total_latency_ms=total_latency,
    )
    return GuardResult(
        False,
        None,
        settings.GUARDRAIL_MODEL,
        "none",
    )


def guard(message: str) -> GuardResult:
    """
    Centralized dispatcher for guardrail enforcement.
    Resolves GUARD_MODE from settings (single_call | nemo | combined).
    """
    from app.guardrails.rails import guard_nemo

    mode = getattr(settings, "GUARD_MODE", "single_call").lower()

    if mode == "combined":
        return evaluate_combined_guard(message)
    elif mode == "nemo":
        return guard_nemo(message)
    else:
        # Default single_call
        return guard_single_call(message)
