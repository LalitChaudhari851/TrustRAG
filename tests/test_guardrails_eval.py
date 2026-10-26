"""
Unit tests for evals/guardrails_eval.py error handling and metrics calculation.
Verifies that infrastructure errors (timeouts, connection drops) are isolated
from model precision, recall, and accuracy metrics.
"""

from unittest.mock import patch, MagicMock
import requests
import pytest

from evals.guardrails_eval import (
    _is_blocked,
    run_guardrails_eval,
    compute_guardrails_metrics,
)


def test_is_blocked_thought_process():
    assert _is_blocked({"thought_process": ["Intent: Technical", "guardrails fired: off_topic"]}) is True
    assert _is_blocked({"thought_process": ["Intent: Technical", "Search Term: pods"]}) is False
    assert _is_blocked({}) is False


def test_compute_guardrails_metrics_standard():
    # 2 TP, 2 TN, 1 FP, 1 FN
    results = [
        {"result": "TP"},
        {"result": "TP"},
        {"result": "TN"},
        {"result": "TN"},
        {"result": "FP"},
        {"result": "FN"},
    ]
    metrics = compute_guardrails_metrics(results)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["total"] == 6
    assert metrics["valid_total"] == 6
    assert metrics["infra_errors"] == 0
    assert metrics["correct"] == 4
    # precision = 2 / (2 + 1) = 0.667
    assert metrics["precision"] == 0.667
    # recall = 2 / (2 + 1) = 0.667
    assert metrics["recall"] == 0.667
    # accuracy = 4 / 6 = 0.667
    assert metrics["accuracy"] == 0.667


def test_compute_guardrails_metrics_with_infra_errors():
    # 2 TP, 2 TN, and 2 INFRA_ERROR (e.g. server down or timeout)
    results = [
        {"result": "TP"},
        {"result": "TP"},
        {"result": "TN"},
        {"result": "TN"},
        {"result": "INFRA_ERROR"},
        {"result": "INFRA_ERROR"},
    ]
    metrics = compute_guardrails_metrics(results)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["total"] == 6
    assert metrics["valid_total"] == 4
    assert metrics["infra_errors"] == 2
    assert metrics["correct"] == 4
    # Precision, recall, and accuracy are computed on the 4 valid results (100%)
    # rather than treating infra errors as false negatives or false positives
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_compute_guardrails_metrics_all_infra_or_empty():
    # All infra errors
    all_infra = [{"result": "INFRA_ERROR"}, {"result": "INFRA_ERROR"}]
    metrics = compute_guardrails_metrics(all_infra)
    assert metrics["total"] == 2
    assert metrics["valid_total"] == 0
    assert metrics["infra_errors"] == 2
    assert metrics["accuracy"] == 0.0

    # Empty
    empty_metrics = compute_guardrails_metrics([])
    assert empty_metrics["total"] == 0
    assert empty_metrics["valid_total"] == 0
    assert empty_metrics["infra_errors"] == 0
    assert empty_metrics["accuracy"] == 0.0


@patch("evals.guardrails_eval.requests.post")
@patch("evals.guardrails_eval.time.sleep")
def test_run_guardrails_eval_handles_connection_error(mock_sleep, mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    samples = [
        {"id": 1, "input": "test input", "expected_blocked": True}
    ]
    evaluated = run_guardrails_eval(samples)
    assert len(evaluated) == 1
    assert evaluated[0]["actual_blocked"] is None
    assert evaluated[0]["result"] == "INFRA_ERROR"


@patch("evals.guardrails_eval.requests.post")
@patch("evals.guardrails_eval.time.sleep")
def test_run_guardrails_eval_handles_http_error(mock_sleep, mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    mock_post.return_value = mock_resp

    samples = [
        {"id": 2, "input": "test input 2", "expected_blocked": False}
    ]
    evaluated = run_guardrails_eval(samples)
    assert len(evaluated) == 1
    assert evaluated[0]["actual_blocked"] is None
    assert evaluated[0]["result"] == "INFRA_ERROR"


@patch("evals.guardrails_eval.requests.post")
@patch("evals.guardrails_eval.time.sleep")
def test_run_guardrails_eval_success_path(mock_sleep, mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"thought_process": ["guardrails fired: off_topic"]}
    mock_post.return_value = mock_resp

    samples = [
        {"id": 3, "input": "tell me a joke", "expected_blocked": True},
        {"id": 4, "input": "what is pod?", "expected_blocked": False},
    ]
    evaluated = run_guardrails_eval(samples)
    assert len(evaluated) == 2
    assert evaluated[0]["actual_blocked"] is True
    assert evaluated[0]["result"] == "TP"
    assert evaluated[1]["actual_blocked"] is True
    assert evaluated[1]["result"] == "FP"
