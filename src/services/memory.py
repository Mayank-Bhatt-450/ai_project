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

    def forget_preference(self, user_id: str, key: str) :
        with self._contextmanager() as conn:
            conn.execute(
                "DELETE FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
            )
