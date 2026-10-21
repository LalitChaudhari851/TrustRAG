import re
import json
from typing import Tuple, Dict, Any

ERROR_CODES = {
    "guard_blocked",
    "guard_unavailable",
    "rate_limited",
    "quota_exhausted",
    "no_context",
    "internal"
}


def sanitize_message(msg: str) -> str:
    """Removes sensitive keys, URLs with tokens, and raw path traces from error text."""
    if not msg:
        return "An internal error occurred."
    cleaned = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED_GROQ_KEY]", msg)
    cleaned = re.sub(r"pk_[A-Za-z0-9_-]+", "[REDACTED_PORTKEY_KEY]", cleaned)
    cleaned = re.sub(r"https?://[^:\s]+:[^@\s]+@[^\s]+", "[REDACTED_URL]", cleaned)
    # Strip deep filesystem trace lines
    lines = [l for l in cleaned.splitlines() if not l.strip().startswith("File \"") and "traceback" not in l.lower()]
    return "\n".join(lines).strip() or "An internal error occurred."


def map_exception_to_error_event(exc: Exception) -> Dict[str, Any]:
    """
    Maps an arbitrary internal exception to a standardized error payload.
    Codes: guard_blocked, guard_unavailable, rate_limited, quota_exhausted, no_context, internal
    """
    raw_msg = str(exc)
    low = raw_msg.lower()

    if "tokens per day" in low or "(tpd)" in low or "quota exhausted" in low:
        return {
            "code": "quota_exhausted",
            "message": "Daily provider inference quota has been exhausted. Please retry tomorrow or upgrade tier.",
            "retryable": False
        }

    if "429" in low or "rate limit" in low:
        return {
            "code": "rate_limited",
            "message": "Model inference rate limit reached. Please wait a few moments and retry.",
            "retryable": True
        }

    if "guardrail" in low and ("timeout" in low or "unavailable" in low or "connection" in low):
        return {
            "code": "guard_unavailable",
            "message": "Safety inspection service is temporarily unavailable. Please retry.",
            "retryable": True
        }

    if "guard_blocked" in low or "guardrail blocked" in low:
        return {
            "code": "guard_blocked",
            "message": sanitize_message(raw_msg),
            "retryable": False
        }

    if "no_context" in low or "insufficient context" in low or "not found in documentation" in low:
        return {
            "code": "no_context",
            "message": "The required information is not found in the technical documentation.",
            "retryable": False
        }

    return {
        "code": "internal",
        "message": "An internal error occurred while processing your request. Please try again shortly.",
        "retryable": False
    }


def format_sse_event(event_name: str, data: Any) -> str:
    """Formats an event name and JSON data dictionary into standard SSE text."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n"
