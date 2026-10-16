"""
Unit tests for Unified LLM Gateway with Tenacity Resilience.
Tests:
- 429 then success
- 429 repeated until exhausted (friendly error)
- Retry-After header parsing and respect
- Non-retryable errors (400, 401, 403, 404) fail immediately
- No retry multiplication: exactly LLM_MAX_RETRIES calls on repeated 429
All tests run offline using mocks.
"""

from unittest.mock import MagicMock, patch
import pytest
from tenacity import RetryError

from app.gateway.client import (
    execute_gateway_chat,
    is_retryable_exception,
    _extract_retry_after,
)
from app.gateway import client as gateway_module
from app.config import settings


def test_exception_retryability_classification():
    """Verify transient status codes are retryable and client errors fail immediately."""
    assert is_retryable_exception(Exception("Error code: 429 - rate_limit_exceeded")) is True
    assert is_retryable_exception(Exception("HTTP 503: Service Unavailable")) is True
    assert is_retryable_exception(Exception("Request timed out after 30s")) is True
    assert is_retryable_exception(Exception("Connection error")) is True

    # Non-retryable client errors
    assert is_retryable_exception(Exception("Error code: 400 - Bad Request")) is False
    assert is_retryable_exception(Exception("Error code: 401 - invalid_api_key")) is False
    assert is_retryable_exception(Exception("Error code: 403 - Forbidden")) is False
    assert is_retryable_exception(Exception("Error code: 404 - model_not_found")) is False


def test_retry_after_extraction():
    """Verify Retry-After is correctly parsed from headers or error message strings."""
    # From string format
    exc1 = Exception("Rate limit reached. Please try again in 3.5s.")
    assert _extract_retry_after(exc1) == 3.5

    exc2 = Exception("Rate limit reached. Please retry after 12 seconds.")
    assert _extract_retry_after(exc2) == 12.0

    # From mock HTTP response headers
    mock_resp = MagicMock()
    mock_resp.headers = {"retry-after": "5"}
    exc3 = Exception("429 Too Many Requests")
    exc3.response = mock_resp
    assert _extract_retry_after(exc3) == 5.0


def test_gateway_retry_on_429_then_success():
    """Verify tenacity retries on 429 and succeeds on subsequent attempt."""
    mock_success_response = MagicMock()
    mock_success_response.choices = [MagicMock(message=MagicMock(content="Success response"))]

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")
        return mock_success_response

    with patch.object(gateway_module.portkey_client.chat.completions, "create", side_effect=mock_create):
        # Override wait times to keep test instantaneous
        with patch.object(settings, "LLM_RETRY_MIN_WAIT", 0.01), patch.object(settings, "LLM_RETRY_MAX_WAIT", 0.05):
            res = execute_gateway_chat(messages=[{"role": "user", "content": "hello"}])
            assert res == mock_success_response
            assert call_count == 2


def test_gateway_exhausted_retry_count_and_no_multiplication():
    """
    Verify that when 429 persists, execute_gateway_chat retries exactly LLM_MAX_RETRIES times
    and raises RetryError/RuntimeError, confirming no hidden internal retry multiplication.
    """
    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    with patch.object(gateway_module.portkey_client.chat.completions, "create", side_effect=mock_create):
        with patch.object(settings, "LLM_MAX_RETRIES", 3), \
             patch.object(settings, "LLM_RETRY_MIN_WAIT", 0.01), \
             patch.object(settings, "LLM_RETRY_MAX_WAIT", 0.05):
            with pytest.raises(Exception) as excinfo:
                execute_gateway_chat(messages=[{"role": "user", "content": "hello"}])

            # Asserts exactly 3 calls occurred (no underlying portkey retry multiplication)
            assert call_count == 3
            assert "429" in str(excinfo.value)


def test_non_retryable_error_fails_immediately():
    """Verify HTTP 401 fails immediately without retrying (call count == 1)."""
    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Error code: 401 - invalid_api_key")

    with patch.object(gateway_module.portkey_client.chat.completions, "create", side_effect=mock_create):
        with pytest.raises(Exception):
            execute_gateway_chat(messages=[{"role": "user", "content": "hello"}])

        assert call_count == 1
