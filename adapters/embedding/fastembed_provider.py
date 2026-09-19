"""FastEmbed local ONNX embedding provider."""
import asyncio
from typing import List, Optional, Any

from core.ports.embedding import BaseEmbeddingProvider


class FastEmbedProvider(BaseEmbeddingProvider):
    """
    Local embedding provider based on FastEmbed (ONNX Runtime)
    using the multilingual-e5-small model (384 dimensions).
    """

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIMENSION = 384
    PREFIX_PASSAGE = "passage: "
    PREFIX_QUERY = "query: "

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.cache_dir = cache_dir
        self._model: Optional[Any] = None

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self.DIMENSION

    def _get_model(self) -> Any:
        """Lazy loader for FastEmbed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed is required for FastEmbedProvider. Please install with: pip install fastembed"
                )
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding inference using ONNX."""
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in embeddings
        ]

    def _format_text(self, text: str, is_query: bool) -> str:
        """Appends appropriate prefix if not already present."""
        prefix = self.PREFIX_QUERY if is_query else self.PREFIX_PASSAGE
        if not text.startswith((self.PREFIX_PASSAGE, self.PREFIX_QUERY)):
            return f"{prefix}{text}"
        return text

    async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embeds a single string.
        Uses 'passage: ' prefix for document passages and 'query: ' for queries.
        """
        formatted = self._format_text(text, is_query=is_query)
        results = await asyncio.to_thread(self._embed_sync, [formatted])
        return results[0]

    async def embed_query(self, query: str) -> List[float]:
        """Convenience method to embed a search query with 'query: ' prefix."""
        return await self.embed_text(query, is_query=True)

    async def embed_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """
        Embeds a batch of strings asynchronously.
        """
        if not texts:
            return []
        formatted_texts = [self._format_text(t, is_query=is_query) for t in texts]
        return await asyncio.to_thread(self._embed_sync, formatted_texts)
