import re
import logfire
from typing import Tuple, Optional
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
    BLOCKED_MARKER,
    ALLOWED_MARKER,
    ALLOWED_FLOWS,
    BLOCKED_FLOWS,
)
from app.gateway.client import is_retryable_exception, tenacity_wait_with_retry_after, log_retry_attempt

_rails: LLMRails | None = None

# Distinct user-facing responses
INFRA_UNAVAILABLE_MESSAGE = (
    "Safety check temporarily unavailable, please retry in a moment."
)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses settings.GUARDRAIL_MODEL with graceful fallback to settings.GROQ_MODEL
    if the configured guard model is not available (404) on the active Groq key.
    """
    global _rails

    mode = getattr(settings, "GUARD_MODE", "single_call").lower()
    if mode == "single_call":
        logfire.info("🛡️ single_call Guardrails mode active — NeMo startup bypassed.")
        return

    active_guard_model = settings.GUARDRAIL_MODEL

    try:
        guard_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=active_guard_model,
            temperature=0,
            max_tokens=64,
            max_retries=0  # Retries controlled by tenacity
        )
        # Test model availability
        guard_llm.invoke("probe")
    except Exception as e:
        logfire.error(f"❌ Failed to initialise NeMo Guardrails with GUARDRAIL_MODEL '{active_guard_model}': {e}")
        raise

    yaml_content = YAML_CONTENT.replace(settings.GUARDRAIL_MODEL, active_guard_model)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=yaml_content
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({active_guard_model}).")



@retry(
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
    wait=tenacity_wait_with_retry_after,
    retry=retry_if_exception(is_retryable_exception),
    before_sleep=log_retry_attempt,
    reraise=True
)
def _generate_rails_response(message: str) -> Tuple[str, list]:
    """Execute NeMo generate with retry and return (response_text, activated_rails)."""
    if _rails is None:
        raise RuntimeError("Guardrails not initialized")

    options = GenerationOptions(output_vars=True, log={"activated_rails": True})
    result = _rails.generate(messages=[{"role": "user", "content": message}], options=options)

    # Extract text content
    if hasattr(result, "response") and result.response:
        content = result.response[0].get("content", "") if isinstance(result.response, list) else str(result.response)
    elif isinstance(result, dict):
        content = result.get("content", "")
    else:
        content = str(result)

    # Extract activated rails from NeMo log
    activated = []
    if hasattr(result, "log") and result.log:
        activated = getattr(result.log, "activated_rails", []) or []

    return content, activated


def _clean_user_message(content: str) -> str:
    """Strip any internal markers (e.g. [GUARDRAIL_BLOCKED: reason]) before showing to user."""
    cleaned = re.sub(r"\[GUARDRAIL_BLOCKED:[^\]]+\]\s*", "", content)
    cleaned = re.sub(r"\[GUARDRAIL_ALLOWED:[^\]]+\]\s*", "", cleaned)
    return cleaned.strip()


def evaluate_nemo_guard(message: str) -> dict:
    """
    Evaluates NeMo Guardrails against an explicit positive allow contract.
    Returns structured dict:
      - status: 'PASSED' | 'BLOCKED' | 'INFRA_UNAVAILABLE' | 'INVALID_DECISION'
      - decision: 'allow' | 'block' | 'infra_unavailable' | 'invalid_decision'
      - reason: 'none' | 'off_topic' | 'jailbreak' | 'nemo_refusal' | 'nemo_rail' | 'unrecognized_intent' | 'infra_error'
      - response_text: user-facing refusal text or None
      - model_used: str
      - activated_rails: list[str]
    """
    from app.guardrails.single_call import DEFAULT_BLOCK_MESSAGE

    if _rails is None:
        logfire.warning("⚠️ NeMo Guardrails not initialised — failing closed for safety.")
        return {
            "status": "INFRA_UNAVAILABLE",
            "decision": "infra_unavailable",
            "reason": "infra_uninitialized",
            "response_text": INFRA_UNAVAILABLE_MESSAGE,
            "model_used": settings.GUARDRAIL_MODEL,
            "activated_rails": [],
        }

    with logfire.span("🛡️ NeMo Guardrails Check"):
        try:
            content, activated = _generate_rails_response(message)
        except Exception as e:
            logfire.error(f"❌ NeMo evaluation infra failure after retries: {e}")
            return {
                "status": "INFRA_UNAVAILABLE",
                "decision": "infra_unavailable",
                "reason": "infra_error",
                "response_text": INFRA_UNAVAILABLE_MESSAGE,
                "model_used": settings.GUARDRAIL_MODEL,
                "activated_rails": [],
            }

        rail_names = [getattr(r, "name", str(r)).lower() for r in activated]
        raw_content = (content or "").strip()
        logfire.info(f"🛡️ NeMo Raw Output: '{raw_content}' | activated_rails={rail_names}")

        # 1. Primary Signal: Blocking flow or blocked marker injected
        has_blocked_rail = any(
            r in BLOCKED_FLOWS or "refuse" in r or "jailbreak" in r or "off topic" in r
            for r in rail_names
        )
        has_blocked_marker = BLOCKED_MARKER in raw_content

        if has_blocked_rail or has_blocked_marker:
            reason = "jailbreak" if ("jailbreak" in raw_content or any("jailbreak" in r for r in rail_names)) else "off_topic"
            cleaned_reply = _clean_user_message(raw_content) or DEFAULT_BLOCK_MESSAGE
            logfire.info(f"🛡️ NeMo explicitly blocked request | reason={reason}")
            return {
                "status": "BLOCKED",
                "decision": "block",
                "reason": reason,
                "response_text": cleaned_reply,
                "model_used": settings.GUARDRAIL_MODEL,
                "activated_rails": rail_names,
            }

        # 2. Refusal Detection Signal: Model conversational refusal / indicators
        is_nemo_fallback = raw_content.lower() == "i'm sorry, i can't respond to that."
        has_refusal_indicator = any(ind.lower() in raw_content.lower() for ind in RAIL_INDICATORS)

        if is_nemo_fallback or has_refusal_indicator:
            logfire.info(f"🛡️ NeMo model conversational refusal detected | indicators={has_refusal_indicator}")
            return {
                "status": "BLOCKED",
                "decision": "block",
                "reason": "nemo_refusal",
                "response_text": DEFAULT_BLOCK_MESSAGE,
                "model_used": settings.GUARDRAIL_MODEL,
                "activated_rails": rail_names,
            }

        # 3. Authoritative Explicit Allow Contract:
        # Requires positive confirmation of an allowed technical or dialog flow OR explicit allow marker
        has_allowed_marker = ALLOWED_MARKER in raw_content
        has_allowed_rail = any(
            r in ALLOWED_FLOWS or "permit" in r or "technical" in r
            for r in rail_names
        )

        if has_allowed_marker or has_allowed_rail:
            logfire.info("✅ NeMo explicitly allowed request.")
            return {
                "status": "PASSED",
                "decision": "allow",
                "reason": "none",
                "response_text": None,
                "model_used": settings.GUARDRAIL_MODEL,
                "activated_rails": rail_names,
            }

        # 4. Fail-Closed on Unrecognized / Ambiguous Output:
        # If NeMo neither matched an allowed flow/marker nor cleanly blocked, it cannot be trusted to allow.
        logfire.warning(f"⚠️ NeMo returned unrecognized output without explicit allow/block. Failing closed: '{raw_content[:80]}'")
        return {
            "status": "BLOCKED",
            "decision": "invalid_decision",
            "reason": "unrecognized_intent",
            "response_text": DEFAULT_BLOCK_MESSAGE,
            "model_used": settings.GUARDRAIL_MODEL,
            "activated_rails": rail_names,
        }


def guard_nemo(message: str):
    """Drop-in GuardResult wrapper for NeMo Guardrails."""
    from app.guardrails.single_call import GuardResult, DEFAULT_BLOCK_MESSAGE
    res = evaluate_nemo_guard(message)
    if res["decision"] == "allow":
        return GuardResult(False, None, res.get("model_used"), "none")
    return GuardResult(
        True,
        res.get("response_text") or DEFAULT_BLOCK_MESSAGE,
        res.get("model_used"),
        res.get("reason"),
    )


def guard(message: str) -> tuple[bool, str | None]:
    """
    Centralized dispatcher: routes to orchestrator.guard(message).
    Preserves 100% backward compatibility for all callers and tests.
    """
    from app.guardrails.orchestrator import guard as orchestrator_guard
    return orchestrator_guard(message)

