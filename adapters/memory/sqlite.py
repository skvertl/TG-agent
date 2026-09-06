import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Generator, Optional
from core.ports import BaseMemoryStore
from core.models import ChatMessage


def get_default_memory_db_path() -> str:
    env_path = os.environ.get("SQLITE_MEMORY_DB_PATH")
    if env_path:
        return env_path
    local_dir = Path("data")
    local_dir.mkdir(parents=True, exist_ok=True)
    return str(local_dir / "chat_history.db")


class SqliteMemoryStore(BaseMemoryStore):
    """
    Персистентное хранилище истории диалогов на базе SQLite.
    Поддерживает сохранение сообщений между перезапусками, изоляцию сессий и очистку.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_default_memory_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
                """
            )

    async def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        with self._get_connection() as conn:
            if limit is not None and limit > 0:
                cursor = conn.execute(
                    """
                    SELECT role, content FROM (
                        SELECT id, role, content FROM chat_messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (session_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT role, content FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
            rows = cursor.fetchall()
            return [
                ChatMessage(role=row["role"], content=row["content"])
                for row in rows
            ]

    async def save_message(self, session_id: str, message: ChatMessage) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, message.role, message.content),
            )

    async def clear(self, session_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?",
                (session_id,),
            )

    def archive_session(self, session_id: str) -> str:
        """
        Создает новый уникальный session_id для чистого контекста.
        """
        return f"{session_id}_session_{uuid.uuid4().hex[:8]}"
