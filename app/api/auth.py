import uuid
import hashlib
import logfire
from typing import Optional
from fastapi import Request, Response, HTTPException, status
from app.config import settings

COOKIE_NAME = "trustrag_user_id"


def get_current_user(request: Request, response: Optional[Response] = None) -> str:
    """
    Derives the authenticated or anonymous user_id based on AUTH_MODE.
    Supports:
    - none: Issues an anonymous user ID cookie if absent. Accepts X-User-ID for testing.
    - apikey: Validates X-API-Key against settings.API_KEYS.
    - oidc: Documented stub verifying Authorization Bearer JWT.
    """
    mode = getattr(settings, "AUTH_MODE", "none").lower()

    if mode == "apikey":
        api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        valid_keys = getattr(settings, "API_KEYS", [])
        if valid_keys and api_key not in valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key.",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        # Derive stable user_id from key
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        return f"apikey-{key_hash}"

    elif mode == "oidc":
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization Bearer token.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        token = auth_header.split(" ", 1)[1].strip()
        # Documented OIDC Stub: in production, verify signature against JWKS endpoint
        # For evaluation/testing stub, we accept tokens formatted as 'sub:user_id' or hash the token
        if token.startswith("sub:"):
            return token.split("sub:", 1)[1]
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        return f"oidc-{token_hash}"

    else:
        # Default: AUTH_MODE="none" (Anonymous cookie mode)
        # Allow X-User-ID override for automated test isolation
        header_user_id = request.headers.get("x-user-id") or request.headers.get("X-User-ID")
        if header_user_id and header_user_id.strip():
            return header_user_id.strip()

        cookie_user_id = request.cookies.get(COOKIE_NAME)
        if cookie_user_id and cookie_user_id.strip():
            return cookie_user_id.strip()

        # Generate fresh anonymous user ID
        new_user_id = f"anon-{uuid.uuid4().hex[:12]}"
        if response is not None:
            response.set_cookie(
                key=COOKIE_NAME,
                value=new_user_id,
                httponly=True,
                samesite="lax",
                max_age=365 * 24 * 3600
            )
        return new_user_id
