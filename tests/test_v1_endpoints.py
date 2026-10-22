import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models.feedback_db import init_feedback_db, get_connection

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Initializes a temporary isolated SQLite database for feedback and threads."""
    test_db = str(tmp_path / "test_feedback.db")
    monkeypatch.setattr("app.config.settings.FEEDBACK_DB_PATH", test_db)
    init_feedback_db()


def parse_sse_events(text: str):
    """Helper to parse raw SSE text stream into a list of (event_name, data_dict) tuples."""
    events = []
    current_event = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and current_event:
            data_str = line.split(":", 1)[1].strip()
            try:
                data = json.loads(data_str)
            except Exception:
                data = data_str
            events.append((current_event, data))
            current_event = None
    return events


# ==============================================================================
# 1. SSE EVENT ORDER & STREAMING TESTS
# ==============================================================================

def test_sse_event_order_technical_query():
    """Verifies strict SSE event sequence: status(guard) -> status(retrieval) -> sources -> status(generating) -> token -> done."""
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    mock_chunk_1.model = "test-model"

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="world!"))]
    mock_chunk_2.model = "test-model"

    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.search_enterprise_knowledge", return_value=[
            {"source": "test.md", "section": "Intro", "content": "Sample content about Kubernetes", "page": 1}
        ]):
            with patch("app.api.routes.chat.rerank_documents", return_value=[
                {"source": "test.md", "section": "Intro", "content": "Sample content about Kubernetes", "page": 1}
            ]):
                with patch("app.api.routes.chat.execute_gateway_chat", return_value=[mock_chunk_1, mock_chunk_2]):
                    response = client.post(
                        "/v1/chat/stream",
                        json={"q": "Explain Kubernetes pods", "retrieval_mode": "tinybert"},
                        headers={"X-User-ID": "user-test-1"}
                    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "X-Request-ID" in response.headers

    events = parse_sse_events(response.text)
    event_names = [e[0] for e in events]

    # Verify event order
    assert event_names[0] == "status"
    assert events[0][1]["stage"] == "guard"

    assert event_names[1] == "status"
    assert events[1][1]["stage"] == "retrieval"

    assert event_names[2] == "sources"
    assert len(events[2][1]) == 1
    assert events[2][1][0]["source"] == "test.md"

    assert event_names[3] == "status"
    assert events[3][1]["stage"] == "generating"

    tokens = [e[1]["delta"] for e in events if e[0] == "token"]
    assert "".join(tokens) == "Hello world!"

    assert event_names[-1] == "done"
    done_payload = events[-1][1]
    assert done_payload["model_used"] == "test-model"
    assert done_payload["retrieval_mode"] == "tinybert"
    assert "latency_ms" in done_payload


def test_sse_chit_chat_flow():
    """Verifies conversational chit-chat bypasses retrieval and emits done event."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="Hi there!"))]
    mock_chunk.model = "test-model"

    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.execute_gateway_chat", return_value=[mock_chunk]):
            response = client.post(
                "/v1/chat/stream",
                json={"q": "hello"},
                headers={"X-User-ID": "user-test-1"}
            )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    event_names = [e[0] for e in events]

    assert "retrieval" not in [e[1].get("stage") for e in events if e[0] == "status"]
    assert "sources" not in event_names
    assert event_names[-1] == "done"


# ==============================================================================
# 2. STANDARDIZED ERROR CODE MAPPING TESTS
# ==============================================================================

def test_error_code_guard_blocked():
    """Verifies guardrail block yields token refusal and done event with model_used=null."""
    with patch("app.api.routes.chat.guard", return_value=(True, "I cannot assist with offensive queries.")):
        response = client.post(
            "/v1/chat/stream",
            json={"q": "jailbreak instruction"},
            headers={"X-User-ID": "user-test-1"}
        )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    event_names = [e[0] for e in events]

    assert "token" in event_names
    assert events[-1][0] == "done"
    assert events[-1][1]["model_used"] is None


def test_error_code_rate_limited():
    """Verifies HTTP 429 exception maps to code 'rate_limited' with retryable=True."""
    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.search_enterprise_knowledge", side_effect=Exception("Error code: 429 - rate limit exceeded")):
            response = client.post(
                "/v1/chat/stream",
                json={"q": "What is an ingress?"},
                headers={"X-User-ID": "user-test-1"}
            )

    events = parse_sse_events(response.text)
    error_event = next(e for e in events if e[0] == "error")
    assert error_event[1]["code"] == "rate_limited"
    assert error_event[1]["retryable"] is True


def test_error_code_quota_exhausted():
    """Verifies daily quota exception maps to code 'quota_exhausted' with retryable=False."""
    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.search_enterprise_knowledge", side_effect=Exception("Tokens per day (TPD) limit exhausted")):
            response = client.post(
                "/v1/chat/stream",
                json={"q": "What is an ingress?"},
                headers={"X-User-ID": "user-test-1"}
            )

    events = parse_sse_events(response.text)
    error_event = next(e for e in events if e[0] == "error")
    assert error_event[1]["code"] == "quota_exhausted"
    assert error_event[1]["retryable"] is False


def test_error_code_guard_unavailable():
    """Verifies guardrail timeout maps to code 'guard_unavailable' with retryable=True."""
    with patch("app.api.routes.chat.guard", side_effect=Exception("Guardrail connection timeout")):
        response = client.post(
            "/v1/chat/stream",
            json={"q": "How to scale?"},
            headers={"X-User-ID": "user-test-1"}
        )

    events = parse_sse_events(response.text)
    error_event = next(e for e in events if e[0] == "error")
    assert error_event[1]["code"] == "guard_unavailable"
    assert error_event[1]["retryable"] is True


def test_error_code_no_context_refusal():
    """Verifies 0 documents passing threshold returns refusal and done with model_used=null."""
    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.search_enterprise_knowledge", return_value=[]):
            with patch("app.api.routes.chat.rerank_documents", return_value=[]):
                response = client.post(
                    "/v1/chat/stream",
                    json={"q": "Obscure question not in corpus"},
                    headers={"X-User-ID": "user-test-1"}
                )

    events = parse_sse_events(response.text)
    token_text = "".join(e[1]["delta"] for e in events if e[0] == "token")
    assert "not found in the documentation" in token_text
    assert events[-1][0] == "done"
    assert events[-1][1]["model_used"] is None


def test_internal_error_sanitizes_keys():
    """Verifies internal errors never leak secret API keys or raw stack traces."""
    leak_string = "Exception with secret gsk_1234567890abcdef and pk_secret_portkey_key at File '/home/app.py'"
    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.search_enterprise_knowledge", side_effect=Exception(leak_string)):
            response = client.post(
                "/v1/chat/stream",
                json={"q": "trigger crash"},
                headers={"X-User-ID": "user-test-1"}
            )

    events = parse_sse_events(response.text)
    error_event = next(e for e in events if e[0] == "error")
    assert error_event[1]["code"] == "internal"
    assert "gsk_" not in error_event[1]["message"]
    assert "pk_" not in error_event[1]["message"]


# ==============================================================================
# 3. THREAD ISOLATION TESTS
# ==============================================================================

def test_thread_isolation_between_users():
    """Verifies User A's thread cannot be read, updated, or deleted by User B."""
    # 1. User A creates Thread A
    user_a_headers = {"X-User-ID": "alice-123"}
    user_b_headers = {"X-User-ID": "bob-456"}

    with patch("app.api.routes.chat.guard", return_value=(False, "")):
        with patch("app.api.routes.chat.execute_gateway_chat", return_value=[]):
            res_a = client.post(
                "/v1/chat/stream",
                json={"q": "Alice's secret plans", "thread_id": "thread-alice-secret"},
                headers=user_a_headers
            )
            assert res_a.status_code == 200

    # 2. User A can list and read the thread
    list_a = client.get("/v1/threads", headers=user_a_headers).json()
    assert any(t["id"] == "thread-alice-secret" for t in list_a)

    detail_a = client.get("/v1/threads/thread-alice-secret", headers=user_a_headers)
    assert detail_a.status_code == 200

    # 3. User B cannot see Thread A in their thread list
    list_b = client.get("/v1/threads", headers=user_b_headers).json()
    assert not any(t["id"] == "thread-alice-secret" for t in list_b)

    # 4. User B guessing thread_id is blocked with HTTP 403 Forbidden
    detail_b = client.get("/v1/threads/thread-alice-secret", headers=user_b_headers)
    assert detail_b.status_code == 403
    assert "Access denied" in detail_b.json()["detail"]

    # 5. User B cannot delete User A's thread
    del_b = client.delete("/v1/threads/thread-alice-secret", headers=user_b_headers)
    assert del_b.status_code == 403

    # 6. User B cannot post a message into User A's thread
    stream_b = client.post(
        "/v1/chat/stream",
        json={"q": "Bob injecting text", "thread_id": "thread-alice-secret"},
        headers=user_b_headers
    )
    assert stream_b.status_code == 403

    # 7. User A can successfully delete their own thread
    del_a = client.delete("/v1/threads/thread-alice-secret", headers=user_a_headers)
    assert del_a.status_code == 200

    # 8. Thread is now gone
    detail_a_after = client.get("/v1/threads/thread-alice-secret", headers=user_a_headers)
    assert detail_a_after.status_code == 404


def test_delete_thread_purges_checkpointer_and_verifies_threads_list(tmp_path, monkeypatch):
    """Verifies DELETE /v1/threads/{id} deletes thread, removes it from GET /v1/threads, and purges checkpointer state."""
    import sqlite3
    from app.models.feedback_db import upsert_thread
    from app.agents.persistence import close_checkpointer

    test_cp_db = str(tmp_path / "test_cp.db")
    monkeypatch.setattr("app.config.settings.SQLITE_DB_PATH", test_cp_db)
    close_checkpointer()

    # Create dummy checkpointer table with entry
    conn = sqlite3.connect(test_cp_db)
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint TEXT)")
    conn.execute("CREATE TABLE writes (thread_id TEXT, task_id TEXT)")
    conn.execute("INSERT INTO checkpoints VALUES ('thread-purge-test', 'data')")
    conn.execute("INSERT INTO writes VALUES ('thread-purge-test', 'write_data')")
    conn.commit()
    conn.close()

    # Create thread in feedback DB
    headers = {"X-User-ID": "user-purge-99"}
    upsert_thread("thread-purge-test", "user-purge-99", title="Thread To Purge")

    # Confirm thread is listed
    threads_before = client.get("/v1/threads", headers=headers).json()
    assert any(t["id"] == "thread-purge-test" for t in threads_before)

    # Delete thread
    res_del = client.delete("/v1/threads/thread-purge-test", headers=headers)
    assert res_del.status_code == 200

    # Verify GET /v1/threads no longer lists the old one
    threads_after = client.get("/v1/threads", headers=headers).json()
    assert not any(t["id"] == "thread-purge-test" for t in threads_after)

    # Verify checkpointer table has no remaining records
    conn = sqlite3.connect(test_cp_db)
    cp_rows = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'thread-purge-test'").fetchone()[0]
    wr_rows = conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = 'thread-purge-test'").fetchone()[0]
    conn.close()
    assert cp_rows == 0
    assert wr_rows == 0


# ==============================================================================
# 4. FEEDBACK STORAGE TESTS
# ==============================================================================

def test_feedback_storage():
    """Verifies thumbs up/down feedback is saved into SQLite feedback table."""
    payload = {
        "message_id": "msg-alpha-99",
        "rating": "up",
        "comment": "Accurate response with exact commands."
    }
    res = client.post("/v1/feedback", json=payload, headers={"X-User-ID": "user-charlie"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "feedback_id" in data

    # Verify directly in SQLite
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM feedback WHERE message_id = ?", ("msg-alpha-99",)).fetchone()
        assert row is not None
        assert row["rating"] == "up"
        assert row["user_id"] == "user-charlie"
        assert row["comment"] == "Accurate response with exact commands."


def test_anonymous_cookie_issuance():
    """Verifies an anonymous session cookie is set when AUTH_MODE=none and no identity header is passed."""
    res = client.get("/v1/threads")
    assert res.status_code == 200
    assert "trustrag_user_id" in res.cookies
    assert res.cookies["trustrag_user_id"].startswith("anon-")
