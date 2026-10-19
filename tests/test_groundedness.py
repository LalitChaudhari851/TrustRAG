import pytest
from app.agents.nodes.responder import generate_node
from app.agents.state import AgentState


def test_empty_context_refusal():
    """Verify that when zero documents pass retrieval, responder returns grounded refusal."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "What is the secret quantum compiler setting?"}],
        "current_query": "What is the secret quantum compiler setting?",
        "documents": [],
        "plan": ["Start"],
        "status": "No relevant context found."
    }

    result = generate_node(state)
    assert "not have enough context" in result["final_answer"].lower()
    assert result["status"] == "Insufficient context."


def test_unsupported_context_refusal(monkeypatch):
    """
    Verify that when context does not contain the answer,
    the tightened responder prompt instructs the model to refuse or report not found.
    """
    class MockChoice:
        class MockMessage:
            content = "I cannot answer this question as the required information is not found in the documentation."
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]
        response_metadata = {}

    def mock_chat(**kwargs):
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        assert "CRITICAL GROUNDING RULES" in prompt
        assert "not found in the documentation" in prompt
        return MockResponse()

    monkeypatch.setattr("app.gateway.execute_gateway_chat", mock_chat)

    state: AgentState = {
        "messages": [{"role": "user", "content": "What is the secret quantum compiler setting in Kubernetes?"}],
        "current_query": "What is the secret quantum compiler setting in Kubernetes?",
        "documents": [
            "[Source: cronjobs.docx | Section: General]\nA Kubernetes Job is a resource designed to run finite workloads."
        ],
        "plan": ["Start", "Context Retrieved"],
        "status": "Context retrieved."
    }

    result = generate_node(state)
    assert "not found in the documentation" in result["final_answer"].lower()
