"""
Unit tests for Persistent Checkpointer (SQLite WAL mode + Postgres interface).
Runs offline with no real LLM/Qdrant calls.
"""

import os
import sqlite3
import pytest
from app.agents.persistence import get_checkpointer, close_checkpointer
from app.config import settings
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState

TEST_DB_PATH = "test_checkpoints.db"


@pytest.fixture(autouse=True)
def cleanup_test_db():
    settings.SQLITE_DB_PATH = TEST_DB_PATH
    settings.DATABASE_URL = None
    close_checkpointer()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    yield
    close_checkpointer()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


def _dummy_node(state: AgentState):
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    return {
        "messages": [{"role": "assistant", "content": f"Echo: {user_msg}"}],
        "status": "done"
    }


def _build_test_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _dummy_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)


def test_sqlite_wal_mode_enabled():
    """Verify SQLite connection is created with WAL mode."""
    checkpointer = get_checkpointer()
    assert checkpointer is not None

    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


def test_thread_state_survives_process_restart():
    """Verify thread state persists across simulated process shutdown and restart."""
    # Process 1: Initialize checkpointer, invoke graph for user_a
    checkpointer1 = get_checkpointer()
    app1 = _build_test_graph(checkpointer1)

    config_a = {"configurable": {"thread_id": "thread_alice"}}
    app1.invoke(
        {"messages": [{"role": "user", "content": "I am Alice"}], "status": "init"},
        config=config_a
    )

    # Simulate process shutdown: close connection and reset singleton
    close_checkpointer()

    # Process 2: Re-open checkpointer from same SQLite file
    checkpointer2 = get_checkpointer()
    app2 = _build_test_graph(checkpointer2)

    # Verify conversation history for thread_alice is preserved
    state = app2.get_state(config_a)
    assert state is not None
    messages = state.values.get("messages", [])
    assert len(messages) >= 2
    assert messages[0]["content"] == "I am Alice"
    assert messages[1]["content"] == "Echo: I am Alice"


def test_thread_isolation():
    """Verify two distinct thread_ids remain strictly isolated in storage."""
    checkpointer = get_checkpointer()
    app = _build_test_graph(checkpointer)

    config_user1 = {"configurable": {"thread_id": "user_101"}}
    config_user2 = {"configurable": {"thread_id": "user_202"}}

    app.invoke(
        {"messages": [{"role": "user", "content": "Secret 101"}], "status": "init"},
        config=config_user1
    )
    app.invoke(
        {"messages": [{"role": "user", "content": "Secret 202"}], "status": "init"},
        config=config_user2
    )

    state1 = app.get_state(config_user1)
    state2 = app.get_state(config_user2)

    msg1_contents = [m["content"] for m in state1.values["messages"]]
    msg2_contents = [m["content"] for m in state2.values["messages"]]

    assert "Secret 101" in msg1_contents
    assert "Secret 202" not in msg1_contents

    assert "Secret 202" in msg2_contents
    assert "Secret 101" not in msg2_contents
