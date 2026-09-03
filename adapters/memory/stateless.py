"""Stateless memory store implementation."""
from core.ports.memory_store import BaseMemoryStore
from core.models import ChatMessage


class StatelessMemoryStore(BaseMemoryStore):
    """Stateless memory store returning empty history for isolated single-turn requests."""

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Always return empty list for stateless execution."""
        return []

    async def save_message(self, session_id: str, message: ChatMessage) -> None:
        """No-op: messages are discarded in stateless mode."""
        pass

    async def clear(self, session_id: str) -> None:
        """No-op: no state to clear in stateless mode."""
        pass
