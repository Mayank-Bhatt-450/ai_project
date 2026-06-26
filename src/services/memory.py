import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from  config import Settings
config=Settings()
SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);

CREATE TABLE IF NOT EXISTS preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_user ON token_usage(user_id, created_at);
"""


class Memory:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(config.memory_db_path)
        with self._contextmanager() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _contextmanager(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        with self._contextmanager() as conn:
            conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, session_id, role, content, time.time()),
            )

    def get_recent_history(self, user_id: str, limit: int = None):

        limit = limit or config.recent_history_turns
        with self._contextmanager() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]


    def set_preference(self, user_id: str, key: str, value: str):
        with self._contextmanager() as conn:
            conn.execute(
                "INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (user_id, key, value, time.time()),
            )

    def get_preferences(self, user_id: str) :
        with self._contextmanager() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {k: v for k, v in rows}

    def list_sessions(self, user_id: str) -> list[dict]:
        """Return all sessions for a user, ordered by most recent activity."""
        with self._contextmanager() as conn:
            rows = conn.execute(
                "SELECT session_id, COUNT(*) as message_count, "
                "MIN(created_at) as started_at, MAX(created_at) as last_active_at "
                "FROM messages WHERE user_id = ? "
                "GROUP BY session_id ORDER BY last_active_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "session_id": session_id,
                "message_count": count,
                "started_at": started_at,
                "last_active_at": last_active_at,
            }
            for session_id, count, started_at, last_active_at in rows
        ]

    def get_chat_history(
        self,
        user_id: str,
        session_id: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        if session_id:
            with self._contextmanager() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at FROM messages "
                    "WHERE user_id = ? AND session_id = ? "
                    "ORDER BY created_at ASC LIMIT ? OFFSET ?",
                    (user_id, session_id, limit, offset),
                ).fetchall()
        else:
            with self._contextmanager() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at FROM messages "
                    "WHERE user_id = ? "
                    "ORDER BY created_at ASC LIMIT ? OFFSET ?",
                    (user_id, limit, offset),
                ).fetchall()

        messages = [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
        return {
            "user_id": user_id,
            "session_id": session_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "messages": messages,
        }

    def forget_preference(self, user_id: str, key: str) :
        with self._contextmanager() as conn:
            conn.execute(
                "DELETE FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
            )

    def record_token_usage(
        self,
        user_id: str,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record token usage for a single LLM call."""
        with self._contextmanager() as conn:
            conn.execute(
                "INSERT INTO token_usage "
                "(user_id, session_id, model, input_tokens, output_tokens, total_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    session_id,
                    model,
                    input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    time.time(),
                ),
            )

    def get_token_usage(self, user_id: str) -> dict:
        """Return lifetime and per-model token totals for a user."""
        with self._contextmanager() as conn:
            rows = conn.execute(
                "SELECT model, "
                "SUM(input_tokens), SUM(output_tokens), SUM(total_tokens), COUNT(*) "
                "FROM token_usage WHERE user_id = ? "
                "GROUP BY model",
                (user_id,),
            ).fetchall()
        breakdown = {
            model: {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": tot,
                "calls": calls,
            }
            for model, inp, out, tot, calls in rows
        }
        totals = {
            "input_tokens": sum(v["input_tokens"] for v in breakdown.values()),
            "output_tokens": sum(v["output_tokens"] for v in breakdown.values()),
            "total_tokens": sum(v["total_tokens"] for v in breakdown.values()),
            "calls": sum(v["calls"] for v in breakdown.values()),
        }
        return {"user_id": user_id, "totals": totals, "by_model": breakdown}