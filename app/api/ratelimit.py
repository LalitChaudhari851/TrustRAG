from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from app.config import settings

def rate_limit_key_func(request: Request) -> str:
    """Derives rate-limiting key from authenticated/cookie user_id or client IP address."""
    user_id = request.headers.get("x-user-id") or request.cookies.get("trustrag_user_id")
    if user_id and user_id.strip():
        return f"user:{user_id.strip()}"
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(
    key_func=rate_limit_key_func,
    default_limits=[getattr(settings, "RATE_LIMIT_PER_MINUTE", "60/minute")]
)
