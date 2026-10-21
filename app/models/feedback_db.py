import sqlite3
import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from app.config import settings

def _get_db_path() -> str:
    path = getattr(settings, "FEEDBACK_DB_PATH", "feedback.db")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_feedback_db():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON thread_messages(thread_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_msg ON feedback(message_id);")
        conn.commit()


def upsert_thread(thread_id: str, user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    default_title = (title or "New Conversation")[:60]
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row:
            if row["user_id"] != user_id:
                raise PermissionError("Access denied: thread owned by another user.")
            if title and title != row["title"]:
                conn.execute("UPDATE threads SET title = ?, updated_at = ? WHERE id = ?", (title, now, thread_id))
            else:
                conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
            conn.commit()
            return {"id": thread_id, "user_id": user_id, "title": title or row["title"], "updated_at": now}
        else:
            conn.execute(
                "INSERT INTO threads (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, default_title, now, now)
            )
            conn.commit()
            return {"id": thread_id, "user_id": user_id, "title": default_title, "created_at": now, "updated_at": now}


def add_thread_message(thread_id: str, user_id: str, role: str, content: str, sources: Optional[List[dict]] = None) -> str:
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources) if sources else None

    with get_connection() as conn:
        row = conn.execute("SELECT user_id, title FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            # Create thread on first message
            title = content[:40].replace("\n", " ").strip() or "New Conversation"
            conn.execute(
                "INSERT INTO threads (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, title, now, now)
            )
        elif row["user_id"] != user_id:
            raise PermissionError("Access denied: thread owned by another user.")
        else:
            conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))

        conn.execute(
            "INSERT INTO thread_messages (id, thread_id, user_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, thread_id, user_id, role, content, sources_json, now)
        )
        conn.commit()
    return msg_id


def get_user_threads(user_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM threads WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_thread_detail(thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        thread = conn.execute("SELECT id, user_id, title, created_at, updated_at FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not thread:
            return None
        if thread["user_id"] != user_id:
            raise PermissionError("Access denied: thread owned by another user.")

        msgs = conn.execute(
            "SELECT id, role, content, sources, created_at FROM thread_messages WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,)
        ).fetchall()

        parsed_msgs = []
        for m in msgs:
            d = dict(m)
            d["sources"] = json.loads(d["sources"]) if d.get("sources") else None
            parsed_msgs.append(d)

        return {
            "id": thread["id"],
            "title": thread["title"],
            "created_at": thread["created_at"],
            "updated_at": thread["updated_at"],
            "messages": parsed_msgs
        }


def delete_user_thread(thread_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        thread = conn.execute("SELECT user_id FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not thread:
            return False
        if thread["user_id"] != user_id:
            raise PermissionError("Access denied: thread owned by another user.")

        conn.execute("DELETE FROM thread_messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        conn.commit()
        return True


def save_feedback(message_id: str, user_id: str, rating: str, comment: Optional[str] = None) -> str:
    fb_id = f"fb-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (id, message_id, user_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (fb_id, message_id, user_id, rating, comment, now)
        )
        conn.commit()
    return fb_id
