import sqlite3
import os
import logfire
from typing import Optional, Any
from app.config import settings

_checkpointer = None
_sqlite_conn = None
_postgres_pool = None

def get_checkpointer():
    """
    Returns the active LangGraph checkpointer singleton.
    
    1. If DATABASE_URL is configured:
       Initializes PostgresSaver with a ConnectionPool (autocommit=True, row_factory=dict_row)
       and executes setup() once.
    2. Otherwise (Default):
       Initializes SqliteSaver with a local SQLite connection configured with WAL mode
       (PRAGMA journal_mode=WAL) for concurrent read access in single-worker / dev mode.
    """
    global _checkpointer, _sqlite_conn, _postgres_pool
    if _checkpointer is not None:
        return _checkpointer

    database_url = getattr(settings, "DATABASE_URL", None)
    if database_url and database_url.strip():
        try:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres import PostgresSaver

            logfire.info("🐘 Initializing Postgres checkpointer pool for LangGraph...")
            _postgres_pool = ConnectionPool(
                conninfo=database_url,
                max_size=20,
                kwargs={"autocommit": True, "row_factory": dict_row}
            )
            checkpointer = PostgresSaver(_postgres_pool)
            checkpointer.setup()
            _checkpointer = checkpointer
            logfire.info("✅ Postgres checkpointer setup complete.")
            return _checkpointer
        except Exception as e:
            logfire.error(f"❌ Failed to initialize Postgres checkpointer: {e}. Falling back to SQLite.")

    # SQLite Fallback (Default for local single-worker development)
    db_path = getattr(settings, "SQLITE_DB_PATH", "checkpoints.db")
    logfire.info(f"💾 Initializing SQLite checkpointer at: {db_path} (WAL mode enabled)")
    
    # Ensure parent dir exists if a subfolder is specified
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    _sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
    # Enable Write-Ahead Logging (WAL) mode for concurrency & resilience
    _sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    _sqlite_conn.commit()

    from langgraph.checkpoint.sqlite import SqliteSaver
    _checkpointer = SqliteSaver(_sqlite_conn)
    _checkpointer.setup()
    logfire.info("✅ SQLite checkpointer ready.")
    return _checkpointer


def close_checkpointer():
    """Cleanly closes database connections upon application shutdown."""
    global _checkpointer, _sqlite_conn, _postgres_pool
    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
            logfire.info("Closed SQLite checkpointer connection.")
        except Exception:
            pass
        _sqlite_conn = None

    if _postgres_pool is not None:
        try:
            _postgres_pool.close()
            logfire.info("Closed Postgres checkpointer pool.")
        except Exception:
            pass
        _postgres_pool = None

    _checkpointer = None
