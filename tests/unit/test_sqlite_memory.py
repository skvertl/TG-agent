import os
import pytest
from core.ports import BaseMemoryStore
from core.models import ChatMessage
from adapters.memory.sqlite import SqliteMemoryStore


class TestSqliteMemoryStore:
    def test_inherits_base_memory_store(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        assert isinstance(store, BaseMemoryStore)

    @pytest.mark.asyncio
    async def test_get_history_empty_initially(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        history = await store.get_history("session-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_save_and_retrieve_messages(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)

        msg1 = ChatMessage(role="user", content="Hello, agent!")
        msg2 = ChatMessage(role="assistant", content="Hello! How can I help?")

        await store.save_message("session-1", msg1)
        await store.save_message("session-1", msg2)

        history = await store.get_history("session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello, agent!"
        assert history[1].role == "assistant"
        assert history[1].content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_session_isolation(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)

        await store.save_message("user-1", ChatMessage(role="user", content="Secret from user 1"))
        await store.save_message("user-2", ChatMessage(role="user", content="Secret from user 2"))

        h1 = await store.get_history("user-1")
        h2 = await store.get_history("user-2")

        assert len(h1) == 1
        assert h1[0].content == "Secret from user 1"
        assert len(h2) == 1
        assert h2[0].content == "Secret from user 2"

    @pytest.mark.asyncio
    async def test_clear_session(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)

        await store.save_message("session-1", ChatMessage(role="user", content="Message 1"))
        await store.save_message("session-2", ChatMessage(role="user", content="Message 2"))

        await store.clear("session-1")

        assert await store.get_history("session-1") == []
        assert len(await store.get_history("session-2")) == 1

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store1 = SqliteMemoryStore(db_path=db_path)
        await store1.save_message("session-p", ChatMessage(role="user", content="Persistent message"))

        # Create new store pointing to same database
        store2 = SqliteMemoryStore(db_path=db_path)
        history = await store2.get_history("session-p")
        assert len(history) == 1
        assert history[0].content == "Persistent message"

    @pytest.mark.asyncio
    async def test_get_history_with_limit_none_returns_all(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        for i in range(5):
            await store.save_message("s-none", ChatMessage(role="user", content=f"msg-{i}"))

        history = await store.get_history("s-none", limit=None)
        assert len(history) == 5
        assert [m.content for m in history] == [f"msg-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_get_history_with_limit_returns_latest_chronological(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        for i in range(5):
            await store.save_message("s-limit", ChatMessage(role="user", content=f"msg-{i}"))

        history = await store.get_history("s-limit", limit=3)
        assert len(history) == 3
        # Must return the latest 3 messages in chronological (ascending) order
        assert [m.content for m in history] == ["msg-2", "msg-3", "msg-4"]

    @pytest.mark.asyncio
    async def test_get_history_with_limit_larger_than_total(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        for i in range(3):
            await store.save_message("s-large", ChatMessage(role="user", content=f"msg-{i}"))

        history = await store.get_history("s-large", limit=10)
        assert len(history) == 3
        assert [m.content for m in history] == ["msg-0", "msg-1", "msg-2"]

    @pytest.mark.asyncio
    async def test_get_history_with_limit_session_isolation(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        store = SqliteMemoryStore(db_path=db_path)
        for i in range(4):
            await store.save_message("s-iso-1", ChatMessage(role="user", content=f"s1-{i}"))
        for i in range(4):
            await store.save_message("s-iso-2", ChatMessage(role="user", content=f"s2-{i}"))

        h1 = await store.get_history("s-iso-1", limit=2)
        h2 = await store.get_history("s-iso-2", limit=2)

        assert len(h1) == 2
        assert [m.content for m in h1] == ["s1-2", "s1-3"]
        assert len(h2) == 2
        assert [m.content for m in h2] == ["s2-2", "s2-3"]
