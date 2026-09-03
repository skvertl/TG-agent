import pytest

from core.ports import BaseMemoryStore
from core.models import ChatMessage
from adapters.memory import StatelessMemoryStore


class TestStatelessMemoryStore:
    def test_inherits_base_memory_store(self):
        store = StatelessMemoryStore()
        assert isinstance(store, BaseMemoryStore)

    @pytest.mark.asyncio
    async def test_get_history_always_returns_empty_list(self):
        store = StatelessMemoryStore()
        history = await store.get_history("session-123")
        assert history == []

    @pytest.mark.asyncio
    async def test_save_message_is_noop_and_history_remains_empty(self):
        store = StatelessMemoryStore()
        message = ChatMessage(role="user", content="Hello, world!")

        # Should complete without error
        await store.save_message("session-123", message)

        # History must still be empty
        history = await store.get_history("session-123")
        assert history == []

    @pytest.mark.asyncio
    async def test_clear_is_noop(self):
        store = StatelessMemoryStore()
        # Should complete without error
        await store.clear("session-123")

        history = await store.get_history("session-123")
        assert history == []

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        store = StatelessMemoryStore()
        msg_user = ChatMessage(role="user", content="User question")
        msg_bot = ChatMessage(role="assistant", content="Assistant answer")

        await store.save_message("session-1", msg_user)
        await store.save_message("session-2", msg_bot)

        assert await store.get_history("session-1") == []
        assert await store.get_history("session-2") == []
