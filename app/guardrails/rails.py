import re
import logfire
from typing import Tuple, Optional
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS, BLOCKED_MARKER
from app.gateway.client import is_retryable_exception, tenacity_wait_with_retry_after, log_retry_attempt

_rails: LLMRails | None = None

# Distinct user-facing responses
INFRA_UNAVAILABLE_MESSAGE = (
    "Safety check temporarily unavailable, please retry in a moment."
)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Dynamically aligns with settings.GROQ_MODEL.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
        max_tokens=64,
        max_retries=0  # Retries controlled by tenacity
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_MODEL}).")


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
    return cleaned.strip()


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired or infra error; return this response immediately,
                                skip the RAG pipeline entirely (Fail Closed).
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — failing closed for safety.")
        return True, INFRA_UNAVAILABLE_MESSAGE

    with logfire.span("🛡️ Guardrails Check"):
        try:
            content, activated = _generate_rails_response(message)
        except Exception as e:
            # Case B: Infrastructure / rate limit failure after retries
            logfire.error(f"❌ Guardrail evaluation infra failure after retries: {e}")
            # Fail closed with distinct message (do NOT pass through, do NOT fake policy refusal)
            return True, INFRA_UNAVAILABLE_MESSAGE

        # Case A: Check primary signal — activated rails log
        rail_names = [getattr(r, "name", str(r)).lower() for r in activated]
        has_blocked_rail = any("refuse" in r or "handle off topic" in r or "jailbreak" in r for r in rail_names)

        # Check secondary signal — structured blocked marker
        has_marker = BLOCKED_MARKER in content

        # Check tertiary signal — indicators
        has_indicator = any(indicator in content for indicator in RAIL_INDICATORS)

        if has_blocked_rail or has_marker or has_indicator:
            cleaned_reply = _clean_user_message(content)
            logfire.info(f"🛡️ Guardrails fired (blocked) | query='{message[:80]}'")
            return True, cleaned_reply

        logfire.info("✅ Guardrails passed.")
        return False, None
