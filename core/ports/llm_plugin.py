"""Port interface for LLM plugins."""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from core.models import PromptPayload, AgentResult


class BaseLLMPlugin(ABC):
    """Abstract port for LLM inference plugins."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize plugin resources (connection pool, model availability check)."""
        pass

    @abstractmethod
    async def generate(self, payload: PromptPayload) -> AgentResult:
        """Synchronous (batch) generation of response for prompt."""
        pass

    @abstractmethod
    async def generate_stream(self, payload: PromptPayload) -> AsyncIterator[str]:
        """Streaming token generation."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully close network connections and release resources."""
        pass
