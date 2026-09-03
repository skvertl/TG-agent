"""Port interface for memory stores."""
from abc import ABC, abstractmethod

from core.models import ChatMessage


class BaseMemoryStore(ABC):
    """Abstract port for dialogue context storage."""

    @abstractmethod
    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve message history for specified session."""
        pass

    @abstractmethod
    async def save_message(self, session_id: str, message: ChatMessage) -> None:
        """Save message to session history."""
        pass

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear dialogue history for session."""
        pass
