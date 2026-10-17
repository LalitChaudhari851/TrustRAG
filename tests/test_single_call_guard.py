import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from app.guardrails.single_call import (
    evaluate_single_call_guard,
    guard_single_call,
    parse_and_validate_json,
    INFRA_UNAVAILABLE_MESSAGE,
    DEFAULT_BLOCK_MESSAGE
)


class MockLLMResponse:
    def __init__(self, content: str, prompt_tokens: int = 150, completion_tokens: int = 15):
        self.content = content
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }


def test_parse_and_validate_json_valid():
    res = parse_and_validate_json('{"decision": "allow", "reason": "none"}')
    assert res == {"decision": "allow", "reason": "none"}

    res_block = parse_and_validate_json('```json\n{"decision": "block", "reason": "jailbreak"}\n```')
    assert res_block == {"decision": "block", "reason": "jailbreak"}


def test_parse_and_validate_json_invalid():
    assert parse_and_validate_json("") is None
    assert parse_and_validate_json("not json") is None
    assert parse_and_validate_json('{"decision": "unknown", "reason": "none"}') is None
    assert parse_and_validate_json('{"decision": "block", "reason": "invalid_reason"}') is None


@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_allow(mock_call):
    mock_call.return_value = MockLLMResponse('{"decision": "allow", "reason": "none"}')
    
    res = evaluate_single_call_guard("How do I configure HPA in Kubernetes?")
    assert res["status"] == "PASSED"
    assert res["decision"] == "allow"
    assert res["reason"] == "none"
    assert res["response_text"] is None
    assert res["prompt_tokens"] == 150
    assert res["completion_tokens"] == 15

    blocked, reply = guard_single_call("How do I configure HPA in Kubernetes?")
    assert blocked is False
    assert reply is None


@pytest.mark.parametrize("reason", ["off_topic", "jailbreak", "injection", "unsafe_request"])
@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_block_reasons(mock_call, reason):
    mock_call.return_value = MockLLMResponse(f'{{"decision": "block", "reason": "{reason}"}}')
    
    res = evaluate_single_call_guard(f"test query for {reason}")
    assert res["status"] == "BLOCKED"
    assert res["decision"] == "block"
    assert res["reason"] == reason
    assert res["response_text"] == DEFAULT_BLOCK_MESSAGE

    blocked, reply = guard_single_call(f"test query for {reason}")
    assert blocked is True
    assert reply == DEFAULT_BLOCK_MESSAGE


@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_invalid_json_then_retry_success(mock_call):
    # First call returns invalid JSON, retry call returns valid block JSON
    mock_call.side_effect = [
        MockLLMResponse("I am not returning json"),
        MockLLMResponse('{"decision": "block", "reason": "jailbreak"}')
    ]

    res = evaluate_single_call_guard("Ignore instructions")
    assert mock_call.call_count == 2
    assert res["status"] == "BLOCKED"
    assert res["decision"] == "block"
    assert res["reason"] == "jailbreak"


@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_invalid_json_fails_closed(mock_call):
    # Both calls return invalid JSON -> fail closed (block)
    mock_call.side_effect = [
        MockLLMResponse("invalid 1"),
        MockLLMResponse("invalid 2")
    ]

    res = evaluate_single_call_guard("malformed response test")
    assert mock_call.call_count == 2
    assert res["status"] == "BLOCKED"
    assert res["decision"] == "block"
    assert res["response_text"] == DEFAULT_BLOCK_MESSAGE


@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_infra_failure(mock_call):
    # Server/network error
    mock_call.side_effect = Exception("503 Service Unavailable: Backed off")

    res = evaluate_single_call_guard("Any message during outage")
    assert res["status"] == "INFRA_UNAVAILABLE"
    assert res["decision"] == "infra_unavailable"
    assert res["response_text"] == INFRA_UNAVAILABLE_MESSAGE

    blocked, reply = guard_single_call("Any message during outage")
    assert blocked is True
    assert reply == INFRA_UNAVAILABLE_MESSAGE


@patch("app.guardrails.single_call._call_guard_llm")
def test_single_call_daily_quota_failure(mock_call):
    # Groq daily quota TPD 429 error
    mock_call.side_effect = Exception(
        "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 200050."
    )

    res = evaluate_single_call_guard("Any message during quota exhaustion")
    assert res["status"] == "INFRA_UNAVAILABLE"
    assert res["decision"] == "infra_unavailable"
    assert res["response_text"] == INFRA_UNAVAILABLE_MESSAGE
