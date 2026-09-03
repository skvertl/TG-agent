"""Domain ports (interfaces)."""
from core.ports.llm_plugin import BaseLLMPlugin
from core.ports.memory_store import BaseMemoryStore

__all__ = ["BaseLLMPlugin", "BaseMemoryStore"]
