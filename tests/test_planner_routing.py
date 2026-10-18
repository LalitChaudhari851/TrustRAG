import json
import pytest
from unittest.mock import MagicMock, patch

from app.agents.nodes.planner import planner_node, is_pure_chitchat, extract_json_route
from app.agents.nodes.responder import generate_node


def test_is_pure_chitchat():
    # True positives: pure chitchat / meta questions
    assert is_pure_chitchat("hi") is True
    assert is_pure_chitchat("hello") is True
    assert is_pure_chitchat("Hello there!") is True
    assert is_pure_chitchat("thanks") is True
    assert is_pure_chitchat("thank you so much") is True
    assert is_pure_chitchat("who are you") is True
    assert is_pure_chitchat("what can you do") is True
    assert is_pure_chitchat("ok") is True
    assert is_pure_chitchat("OKAY") is True
    assert is_pure_chitchat("good morning") is True

    # True negatives: technical / work queue / k8s queries
    assert is_pure_chitchat("How do you fill the Redis work queue with tasks using the CLI?") is False
    assert is_pure_chitchat("What is Horizontal Pod Autoscaling in Kubernetes?") is False
    assert is_pure_chitchat("What is a Kubernetes Job?") is False
    assert is_pure_chitchat("rpush job2 apple") is False
    assert is_pure_chitchat("databricks jobs get") is False
    assert is_pure_chitchat("kubectl get pods -n kube-system") is False
    assert is_pure_chitchat("hi, how do I configure a cronjob?") is False


def test_greeting_routes_to_skip():
    for greeting in ["hi", "thanks", "who are you", "what can you do", "ok"]:
        state = {
            "messages": [{"role": "user", "content": greeting}],
            "documents": [],
            "plan": [],
            "status": ""
        }
        out = planner_node(state)
        assert out["route"] == "chitchat"
        assert out["current_query"] == "CONVERSATIONAL"
        assert "Retrieval: Skipped" in out["plan"][1]


def test_standalone_technical_question_skips_llm():
    """Standalone question with no history skips LLM planner entirely and uses raw message."""
    state = {
        "messages": [{"role": "user", "content": "How do you fill the Redis work queue with tasks using the CLI?"}],
        "documents": [],
        "plan": [],
        "status": ""
    }
    mock_llm = MagicMock()
    with patch("app.agents.nodes.planner.llm", mock_llm):
        out = planner_node(state)

    assert mock_llm.invoke.call_count == 0  # LLM planner skipped for standalone query
    assert out["route"] == "retrieve"
    assert out["current_query"] == "How do you fill the Redis work queue with tasks using the CLI?"
    assert "Planner LLM: Skipped (raw query)" in out["plan"][-1]


def test_followup_with_history_invokes_llm_and_concatenates():
    """Short follow-up with pronouns in history triggers LLM rewrite and concatenates with original query."""
    state = {
        "messages": [
            {"role": "user", "content": "Tell me about Kubernetes Job spec."},
            {"role": "assistant", "content": "A Job creates one or more Pods and will continue to retry execution."},
            {"role": "user", "content": "how do I run it?"}  # short follow-up (<6 words, has pronoun 'it')
        ],
        "documents": [],
        "plan": [],
        "status": ""
    }
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"route": "retrieve", "query": "kubectl apply Kubernetes Job manifest execution"}'
    mock_llm.invoke.return_value = mock_resp

    with patch("app.agents.nodes.planner.llm", mock_llm):
        out = planner_node(state)

    assert mock_llm.invoke.call_count == 1
    assert out["route"] == "retrieve"
    # Concatenated: original + rewritten
    assert out["current_query"] == "how do I run it? kubectl apply Kubernetes Job manifest execution"
    assert "Intent: Technical" in out["plan"][0]


def test_query_rewrite_config_modes():
    """Test QUERY_REWRITE=off and QUERY_REWRITE=always configs."""
    from app.config import settings
    state = {
        "messages": [
            {"role": "user", "content": "What is Redis?"},
            {"role": "assistant", "content": "Redis is an in-memory data store."},
            {"role": "user", "content": "how does it work?"}
        ],
        "documents": [],
        "plan": [],
        "status": ""
    }
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"route": "retrieve", "query": "Redis architecture in-memory engine"}'
    mock_llm.invoke.return_value = mock_resp

    # Mode: off
    with patch.object(settings, "QUERY_REWRITE", "off"):
        with patch("app.agents.nodes.planner.llm", mock_llm):
            out = planner_node(state)
            assert mock_llm.invoke.call_count == 0
            assert out["current_query"] == "how does it work?"

    # Mode: always on standalone
    standalone_state = {
        "messages": [{"role": "user", "content": "What is Redis?"}],
        "documents": [],
        "plan": [],
        "status": ""
    }
    with patch.object(settings, "QUERY_REWRITE", "always"):
        with patch("app.agents.nodes.planner.llm", mock_llm):
            out = planner_node(standalone_state)
            assert mock_llm.invoke.call_count == 1
            assert out["current_query"] == "What is Redis? Redis architecture in-memory engine"


def test_planner_exception_defaults_to_retrieve():
    """Follow-up query rewrite failure safely falls back to raw message and retrieve."""
    from app.config import settings
    state = {
        "messages": [
            {"role": "user", "content": "Tell me about Horizontal Pod Autoscaling."},
            {"role": "assistant", "content": "HPA scales pods based on CPU/memory."},
            {"role": "user", "content": "how does it scale?"}
        ],
        "documents": [],
        "plan": [],
        "status": ""
    }
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Groq 429 TPD rate limit exhausted")

    with patch.object(settings, "QUERY_REWRITE", "followup_only"):
        with patch("app.agents.nodes.planner.llm", mock_llm):
            out = planner_node(state)

    assert out["route"] == "retrieve"
    assert out["current_query"] == "how does it scale?"
    assert "Intent: Technical" in out["plan"][0]


def test_invalid_json_retries_and_then_retrieves():
    """Follow-up query with invalid JSON retries once and falls back cleanly."""
    from app.config import settings
    state = {
        "messages": [
            {"role": "user", "content": "Tell me about Kubernetes Job."},
            {"role": "assistant", "content": "Jobs manage batch tasks."},
            {"role": "user", "content": "what is it?"}
        ],
        "documents": [],
        "plan": [],
        "status": ""
    }
    mock_llm = MagicMock()
    bad_resp = MagicMock()
    bad_resp.content = "I think you should look up Kubernetes jobs."
    mock_llm.invoke.return_value = bad_resp

    with patch.object(settings, "QUERY_REWRITE", "followup_only"):
        with patch("app.agents.nodes.planner.llm", mock_llm):
            out = planner_node(state)
            assert mock_llm.invoke.call_count == 2  # initial + 1 retry

    assert out["route"] == "retrieve"
    assert out["current_query"] == "what is it?"
    assert "Intent: Technical" in out["plan"][0]


def test_empty_context_causes_grounded_refusal():
    state = {
        "messages": [{"role": "user", "content": "What is the secret internal cluster password?"}],
        "current_query": "secret internal cluster password",
        "route": "retrieve",
        "documents": [],  # Zero docs passed threshold
        "plan": ["Intent: Technical"],
        "status": ""
    }
    out = generate_node(state)

    assert out["final_answer"] == "I do not have enough context in the documentation to answer this question accurately."
    assert "Refusal: Insufficient Context" in out["plan"][-1]


def test_all_37_questions_adversarial_mock_chitchat_kept_on_retrieve():
    """
    Test all 15 golden questions + 22 hard questions with an adversarial mocked LLM
    that maliciously or erroneously attempts to return {"route": "chitchat"}.
    The pre-router MUST override it and ensure retrieval stays ON (37/37 retrieve).
    """
    with open("evals/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_questions = [s["question"] for s in golden_raw.get("rag_samples", golden_raw)]

    with open("evals/data/hard_retrieval_22.json", "r", encoding="utf-8") as f:
        hard_raw = json.load(f)
    hard_questions = [s["question"] for s in hard_raw]

    all_37 = golden_questions + hard_questions
    assert len(all_37) == 37

    # Adversarial mock returning chitchat for everything
    mock_llm = MagicMock()
    adversarial_mock = MagicMock()
    adversarial_mock.content = '{"route": "chitchat"}'
    mock_llm.invoke.return_value = adversarial_mock

    with patch("app.agents.nodes.planner.llm", mock_llm):
        for idx, q in enumerate(all_37, start=1):
            state = {
                "messages": [{"role": "user", "content": q}],
                "documents": [],
                "plan": [],
                "status": ""
            }
            out = planner_node(state)
            assert out["route"] == "retrieve", f"Question #{idx} ('{q}') erroneously skipped retrieval!"
            assert out["current_query"] == q, f"Question #{idx} query was not preserved on override"
