"""
Maintenance script to prune stale LangGraph checkpointer threads from SQLite/Postgres.
Usage:
    python scripts/prune_checkpoints.py --days 7
"""

import sys
import os
import argparse
import sqlite3
import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import settings

def prune_sqlite(db_path: str, days: int):
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return

    cutoff_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    print(f"Pruning SQLite checkpoints older than {days} days (cutoff: {cutoff_date})...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check available tables in SQLite checkpointer
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]

        if "checkpoints" in tables:
            # Delete checkpoints older than cutoff
            # In langgraph-checkpoint-sqlite, checkpoint records store checkpoint_id / metadata
            cursor.execute("DELETE FROM checkpoints WHERE checkpoint_id < ?", (cutoff_date,))
            deleted = cursor.rowcount
            conn.commit()
            print(f"Successfully pruned {deleted} stale checkpoint records from {db_path}.")
            
            # Execute VACUUM to reclaim disk space
            conn.execute("VACUUM;")
            print("SQLite VACUUM complete.")
        else:
            print("No checkpoints table found yet in database.")
    except Exception as e:
        print(f"Error during pruning: {e}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Prune stale conversation checkpoints.")
    parser.add_argument("--days", type=int, default=14, help="Delete threads older than N days (default: 14)")
    args = parser.parse_args()

    db_path = getattr(settings, "SQLITE_DB_PATH", "checkpoints.db")
    prune_sqlite(db_path, args.days)

if __name__ == "__main__":
    main()
