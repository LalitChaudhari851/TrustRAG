import sys, os
sys.path.insert(0, os.path.abspath("."))

import pytest
from app.gateway.client import _extract_retry_after, is_retryable_exception, is_daily_quota_exhaustion


class MockResponse:
    def __init__(self, headers):
        self.headers = headers


class MockException(Exception):
    def __init__(self, message, headers=None, status_code=429):
        super().__init__(message)
        self.status_code = status_code
        if headers is not None:
            self.response = MockResponse(headers)


def test_extract_retry_after_ms():
    exc = Exception("Rate limit reached. Please try again in 797.14ms. Visit https://...")
    res = _extract_retry_after(exc)
    assert res is not None
    assert pytest.approx(res, rel=1e-3) == 0.79714


def test_extract_retry_after_seconds():
    exc = Exception("Rate limit exceeded. Please try again in 2.5s.")
    res = _extract_retry_after(exc)
    assert res == 2.5


def test_extract_retry_after_header_within_cap():
    exc = MockException("Rate limit", headers={"retry-after": "12.0"})
    res = _extract_retry_after(exc)
    assert res == 12.0


def test_extract_retry_after_header_capped_at_20s():
    exc = MockException("Rate limit", headers={"retry-after": "45.0"})
    res = _extract_retry_after(exc)
    assert res == 20.0


def test_extract_retry_after_daily_limit_ignored():
    # A message containing 'Limit 200000' or TPD must not parse 200000 as sleep
    exc = Exception("Rate limit reached for model `qwen/qwen3.8-27b` on tokens per day (TPD): Limit 200000, Used 200050.")
    res = _extract_retry_after(exc)
    assert res is None


def test_daily_limit_fails_fast_not_retryable():
    exc = MockException(
        "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 200050. Please try again tomorrow.",
        status_code=429
    )
    assert is_daily_quota_exhaustion(exc) is True
    assert is_retryable_exception(exc) is False


def test_transient_rate_limit_is_retryable():
    exc = MockException(
        "Rate limit reached. Please try again in 500ms.",
        status_code=429
    )
    assert is_daily_quota_exhaustion(exc) is False
    assert is_retryable_exception(exc) is True
