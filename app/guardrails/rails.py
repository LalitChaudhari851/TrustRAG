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
    Uses settings.GUARDRAIL_MODEL with graceful fallback to settings.GROQ_MODEL
    if the configured guard model is not available (404) on the active Groq key.
    """
    global _rails

    if getattr(settings, "GUARD_MODE", "nemo").lower() == "single_call":
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
    return cleaned.strip()


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the guardrails gate.
    Routes to single_call or NeMo according to settings.GUARD_MODE.

    Returns:
        (True,  rail_response) — a rail fired or infra error; return this response immediately,
                                skip the RAG pipeline entirely (Fail Closed).
        (False, None)          — message is clean; proceed to LangGraph.
    """
    from app.guardrails.single_call import GuardResult

    if getattr(settings, "GUARD_MODE", "single_call").lower() == "single_call":
        from app.guardrails.single_call import guard_single_call
        return guard_single_call(message)

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — failing closed for safety.")
        return GuardResult(True, INFRA_UNAVAILABLE_MESSAGE, settings.GUARDRAIL_MODEL, "infra_uninitialized")

    with logfire.span("🛡️ Guardrails Check"):
        try:
            content, activated = _generate_rails_response(message)
        except Exception as e:
            # Case B: Infrastructure / rate limit failure after retries
            logfire.error(f"❌ Guardrail evaluation infra failure after retries: {e}")
            # Fail closed with distinct message (do NOT pass through, do NOT fake policy refusal)
            return GuardResult(True, INFRA_UNAVAILABLE_MESSAGE, settings.GUARDRAIL_MODEL, "infra_error")

        # Log activated_rails and raw rail output for auditing
        rail_names = [getattr(r, "name", str(r)).lower() for r in activated]
        logfire.info(f"🛡️ NeMo Raw Output: '{content}' | activated_rails={rail_names}")

        # Primary Signal: Check activated rails log for blocking flow
        # In Colang, blocking flows are named 'handle off topic' or 'jailbreak protection',
        # and their bot actions are 'refuse off topic' or 'refuse jailbreak'.
        has_blocked_rail = any(
            r in ("handle off topic", "jailbreak protection", "refuse off topic", "refuse jailbreak")
            or "refuse" in r
            for r in rail_names
        )

        # Secondary Signal: Structured marker prefix explicitly injected by our bot refusal actions
        has_marker = BLOCKED_MARKER in content

        # Tertiary Signal: NeMo built-in safety fallback refusal
        # When NeMo's LLM detects an off-topic or safety violation that doesn't cleanly match
        # a Colang flow, it produces its built-in fallback: "I'm sorry, I can't respond to that."
        # This is an EXACT match check, NOT a broad substring match, preventing false positives on legitimate queries.
        is_nemo_fallback = content.strip().lower() == "i'm sorry, i can't respond to that."

        if has_blocked_rail or has_marker or is_nemo_fallback:
            cleaned_reply = _clean_user_message(content)
            if not cleaned_reply or is_nemo_fallback:
                cleaned_reply = "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I cannot assist with this request."
            logfire.info(f"🛡️ Guardrails fired (blocked) | query='{message[:80]}'")
            return GuardResult(True, cleaned_reply, settings.GUARDRAIL_MODEL, "nemo_rail")

        logfire.info("✅ Guardrails passed.")
        return GuardResult(False, None, settings.GUARDRAIL_MODEL, "none")

