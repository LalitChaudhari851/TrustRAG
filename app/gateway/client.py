import time
import re
import logfire
from typing import Any, Dict, Optional, Tuple
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
    RetryCallState,
)
import logging

from app.config import settings

logger = logging.getLogger("app.gateway.client")

# Portkey saved config ID: check if user configured a Portkey saved slug (e.g. pc-...)
# NOTE ON RETRIES: We set max_retries=0 on the underlying Portkey client and pass no inline retries
# to ensure Tenacity is the single authoritative retry controller and prevent retry multiplication.
PORTKEY_CONFIG = settings.PORTKEY_CONFIG_ID if getattr(settings, "PORTKEY_CONFIG_ID", None) else None

# Underlying client initialized with max_retries=0 so tenacity alone controls retry policy
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG,
    max_retries=0
)


def is_daily_quota_exhaustion(exc: Exception) -> bool:
    """Detect if 429 error is due to daily token/request exhaustion (TPD / 200k daily limit)."""
    msg = str(exc).lower()
    return any(term in msg for term in [
        "limit 200000",
        "tokens per day",
        "daily limit",
        "requests per day",
        "tpd",
        "rpd"
    ])


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """
    Extract Retry-After header or message in seconds from 429 response if present.
    Strictly anchored on 'try again in|retry after|wait' to avoid matching arbitrary numbers like 'Limit 200000'.
    Caps sleep at 20.0s.
    Returns None if daily quota is exhausted.
    """
    if is_daily_quota_exhaustion(exc):
        return None

    msg = str(exc)
    # Check for direct headers attribute if available on HTTP response
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                val = float(retry_after)
                return min(val, 20.0)
            except (ValueError, TypeError):
                pass
                
    # Search message strictly anchored on 'try again in', 'retry after', or 'wait'
    match = re.search(r"(?:try again in|retry after|wait)\s+([\d\.]+)\s*(ms|s|seconds)?", msg, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            unit = (match.group(2) or "s").lower()
            if unit == "ms":
                seconds = val / 1000.0
            else:
                seconds = val
            return min(seconds, 20.0)
        except (ValueError, TypeError):
            pass
    return None


def is_retryable_exception(exc: Exception) -> bool:
    """
    Retry only transient infra/rate issues:
    - HTTP 429 (Transient burst Rate limit)
    - HTTP 503 (Service unavailable / server overloaded)
    - Network / Connection timeouts / Connection errors
    
    NEVER retry client errors (HTTP 400, 401, 403, 404).
    NEVER retry when daily token/request quota (TPD) is exhausted (fail fast).
    """
    if is_daily_quota_exhaustion(exc):
        logfire.error("🛑 Daily quota (TPD/RPD) exhausted on LLM provider. Failing fast without retrying.")
        return False

    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    
    # Check explicit non-retryable status codes
    if status_code in (400, 401, 403, 404):
        return False
    if any(code in msg for code in ["400", "401", "403", "404", "invalid_api_key", "model_not_found"]):
        return False
        
    # Check retryable status codes
    if status_code in (429, 502, 503, 504):
        return True
    if any(term in msg for term in ["429", "502", "503", "504", "rate limit", "rate_limit", "timeout", "timed out", "connection error"]):
        return True
        
    return False


def tenacity_wait_with_retry_after(retry_state: RetryCallState) -> float:
    """Custom tenacity wait function: honors Retry-After if available, otherwise exponential jitter."""
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        if exc is not None:
            retry_after = _extract_retry_after(exc)
            if retry_after is not None and retry_after > 0:
                # Add a 0.5s safety buffer to the provider's suggested wait
                wait_time = min(retry_after + 0.5, settings.LLM_RETRY_MAX_WAIT)
                logfire.info(f"⏳ Honoring provider Retry-After header: waiting {wait_time:.1f}s")
                return wait_time

    # Default exponential backoff with random jitter between MIN and MAX
    wait_strategy = wait_random_exponential(
        min=settings.LLM_RETRY_MIN_WAIT,
        max=settings.LLM_RETRY_MAX_WAIT
    )
    return wait_strategy(retry_state)


def log_retry_attempt(retry_state: RetryCallState):
    """Log structured retry span to Logfire with attempt number and reason."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logfire.warning(
        f"⚠️ Transient LLM failure on attempt {attempt}/{settings.LLM_MAX_RETRIES} | Retrying... | Error: {exc}"
    )


@retry(
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
    wait=tenacity_wait_with_retry_after,
    retry=retry_if_exception(is_retryable_exception),
    before_sleep=log_retry_attempt,
    reraise=True
)
def execute_gateway_chat(**create_params) -> Any:
    """
    Centralized, resilient completion function via Portkey Gateway.
    Applies Tenacity retry with exponential backoff + jitter for 429/503 and timeouts.
    Applies to request/response calls (not mid-stream tokens).
    """
    if "model" not in create_params and not PORTKEY_CONFIG:
        create_params["model"] = f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"
        
    return portkey_client.chat.completions.create(**create_params)


def get_langchain_llm(feature: str = "rag", max_tokens: int = 50) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI for LangChain nodes.
    Configured with max_retries=0 so retry resilience is handled consistently.
    """
    header_params = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        }
    }
    if PORTKEY_CONFIG:
        header_params["config"] = PORTKEY_CONFIG

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        max_tokens=max_tokens,
        max_retries=0, # Controlled via unified gateway policy
        default_headers=createHeaders(**header_params)
    )


def extract_cache_status(response) -> str:
    """Pull x-portkey-cache-status from Portkey response headers."""
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"