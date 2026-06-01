"""SQLite-backed chat history for persistent conversational RAG sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Persist chat messages by session while exposing LangChain's history API."""

    def __init__(self, session_id: str, db_path: Path, max_messages: int | None = None) -> None:
        self.session_id = session_id
        self.db_path = db_path
        self.max_messages = max_messages
        self._ensure_table()

    @property
    def messages(self) -> list[BaseMessage]:
        limit_clause = ""
        params: tuple[str, int] | tuple[str]
        params = (self.session_id,)
        if self.max_messages is not None:
            limit_clause = "LIMIT ?"
            params = (self.session_id, self.max_messages)

        query = f"""
            SELECT message_json
            FROM (
                SELECT id, message_json
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                {limit_clause}
            )
            ORDER BY id ASC
        """
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return messages_from_dict([json.loads(row[0]) for row in rows])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        serialized_messages = messages_to_dict(list(messages))
        if not serialized_messages:
            return

        rows = [
            (self.session_id, json.dumps(message, ensure_ascii=False))
            for message in serialized_messages
        ]
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO chat_messages (session_id, message_json) VALUES (?, ?)",
                rows,
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (self.session_id,))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=TRUNCATE")
        return connection

    def _ensure_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id
                ON chat_messages (session_id, id)
                """
            )


def get_session_history(
    session_id: str,
    db_path: Path,
    max_messages: int | None = 12,
) -> SQLiteChatMessageHistory:
    return SQLiteChatMessageHistory(
        session_id=session_id,
        db_path=db_path,
        max_messages=max_messages,
    )


def clear_session_history(session_id: str, db_path: Path) -> None:
    get_session_history(session_id=session_id, db_path=db_path, max_messages=None).clear()
