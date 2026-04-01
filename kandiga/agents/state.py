"""SQLite state store — queryable sessions, messages, FTS search, token tracking.

Replaces file-based memory with a proper database.
Every conversation turn is stored, searchable, and queryable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


DB_PATH = os.path.expanduser("~/.kandiga/state.db")


class StateStore:
    """Persistent state store backed by SQLite with WAL + FTS5."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                title TEXT DEFAULT '',
                model TEXT DEFAULT '',
                turn_count INTEGER DEFAULT 0,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                total_duration_ms INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                route TEXT DEFAULT '',
                confidence REAL DEFAULT 0,
                verified INTEGER DEFAULT 0,
                tools_used TEXT DEFAULT '[]',
                duration_ms INTEGER DEFAULT 0,
                tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id INTEGER,
                tool TEXT NOT NULL,
                args TEXT DEFAULT '{}',
                output TEXT DEFAULT '',
                success INTEGER DEFAULT 1,
                error TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_tools_session ON tool_executions(session_id);
        """)

        # FTS5 for full-text search
        try:
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(content, session_id, content='messages', content_rowid='id')
            """)
        except sqlite3.OperationalError:
            pass  # FTS5 might not be available

        self.db.commit()

    # --- Sessions ---

    def create_session(self, model: str = "") -> str:
        import uuid
        sid = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self.db.execute(
            "INSERT INTO sessions (id, started_at, model) VALUES (?, ?, ?)",
            (sid, now, model),
        )
        self.db.commit()
        return sid

    def end_session(self, session_id: str):
        self.db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (datetime.now().isoformat(), session_id),
        )
        self.db.commit()

    def get_session(self, session_id: str) -> Optional[Dict]:
        row = self.db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Messages ---

    def add_message(
        self,
        session_id: str,
        turn: int,
        role: str,
        content: str,
        route: str = "",
        confidence: float = 0,
        verified: bool = False,
        tools_used: Optional[List[str]] = None,
        duration_ms: int = 0,
        tokens: int = 0,
    ) -> int:
        now = datetime.now().isoformat()
        cursor = self.db.execute(
            """INSERT INTO messages
               (session_id, turn, role, content, route, confidence, verified,
                tools_used, duration_ms, tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, turn, role, content, route, confidence,
             1 if verified else 0, json.dumps(tools_used or []),
             duration_ms, tokens, now),
        )
        msg_id = cursor.lastrowid

        # Update session stats
        self.db.execute(
            "UPDATE sessions SET turn_count = turn_count + 1, tokens_out = tokens_out + ? WHERE id = ?",
            (tokens, session_id),
        )

        # Update FTS
        try:
            self.db.execute(
                "INSERT INTO messages_fts (rowid, content, session_id) VALUES (?, ?, ?)",
                (msg_id, content, session_id),
            )
        except sqlite3.OperationalError:
            pass

        self.db.commit()
        return msg_id

    def add_tool_execution(
        self,
        session_id: str,
        message_id: int,
        tool: str,
        args: Dict,
        output: str,
        success: bool,
        error: str = "",
        duration_ms: int = 0,
    ):
        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO tool_executions
               (session_id, message_id, tool, args, output, success, error, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, message_id, tool, json.dumps(args),
             output[:2000], 1 if success else 0, error, duration_ms, now),
        )
        self.db.commit()

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        rows = self.db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY turn ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT * FROM messages ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Search ---

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search across all messages."""
        try:
            rows = self.db.execute(
                """SELECT m.* FROM messages m
                   JOIN messages_fts f ON m.id = f.rowid
                   WHERE messages_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS not available, fallback to LIKE
            rows = self.db.execute(
                "SELECT * FROM messages WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Stats ---

    def stats(self) -> Dict[str, Any]:
        sessions = self.db.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"]
        messages = self.db.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        tools = self.db.execute("SELECT COUNT(*) c FROM tool_executions").fetchone()["c"]
        tokens = self.db.execute("SELECT COALESCE(SUM(tokens), 0) t FROM messages").fetchone()["t"]
        return {
            "sessions": sessions,
            "messages": messages,
            "tool_executions": tools,
            "total_tokens": tokens,
        }
