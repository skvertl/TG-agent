"""Embedding provider port interface."""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract port for generating dense text vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality (e.g. 384)."""
        pass

    @abstractmethod
    async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generates a dense vector embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generates dense vector embeddings for a batch of texts."""
        pass
