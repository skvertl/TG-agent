"""RAG store port interface."""
from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import DocumentMetadata, DocumentChunk, SearchResult


class BaseRAGStore(ABC):
    """Abstract port for vector and document persistence."""

    @abstractmethod
    async def add_document(
        self,
        metadata: DocumentMetadata,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Atomically persist document record, text chunks, and their vector embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 4,
        score_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """
        Search documents strictly scoped to user_id.
        Supports dense vector similarity and optional hybrid FTS5 search.
        """
        pass

    @abstractmethod
    async def list_documents(self, user_id: str) -> List[DocumentMetadata]:
        """Return all documents owned by user_id, sorted by created_at DESC."""
        pass

    @abstractmethod
    async def get_document(self, user_id: str, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve single document metadata if owned by user_id."""
        pass

    @abstractmethod
    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Permanently delete document, all associated chunks, and vector index rows.
        Returns True if deleted, False if document not found or access denied.
        """
        pass
