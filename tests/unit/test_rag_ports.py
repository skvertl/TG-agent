"""Unit tests for RAG abstract port interfaces."""
import pytest
from typing import List, Optional

from core.ports import BaseEmbeddingProvider, BaseRAGStore
from core.models import DocumentMetadata, DocumentChunk, SearchResult


class TestBaseEmbeddingProvider:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseEmbeddingProvider()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        abstract_methods = BaseEmbeddingProvider.__abstractmethods__
        expected_methods = {"dimension", "embed_text", "embed_batch"}
        assert expected_methods.issubset(abstract_methods)

    @pytest.mark.asyncio
    async def test_concrete_implementation(self):
        class MockEmbeddingProvider(BaseEmbeddingProvider):
            @property
            def dimension(self) -> int:
                return 384

            async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
                return [0.1] * self.dimension

            async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
                return [[0.1] * self.dimension for _ in texts]

        provider = MockEmbeddingProvider()
        assert isinstance(provider, BaseEmbeddingProvider)
        assert provider.dimension == 384
        vec = await provider.embed_text("test", is_query=True)
        assert len(vec) == 384
        batch_vecs = await provider.embed_batch(["test1", "test2"])
        assert len(batch_vecs) == 2


class TestBaseRAGStore:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseRAGStore()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        abstract_methods = BaseRAGStore.__abstractmethods__
        expected_methods = {
            "add_document",
            "search",
            "list_documents",
            "get_document",
            "delete_document",
        }
        assert expected_methods.issubset(abstract_methods)

    @pytest.mark.asyncio
    async def test_concrete_implementation(self):
        class MockRAGStore(BaseRAGStore):
            async def add_document(
                self,
                metadata: DocumentMetadata,
                chunks: List[DocumentChunk],
                embeddings: List[List[float]],
            ) -> None:
                pass

            async def search(
                self,
                user_id: str,
                query_embedding: List[float],
                query_text: Optional[str] = None,
                top_k: int = 4,
                score_threshold: float = 0.5,
            ) -> List[SearchResult]:
                return []

            async def list_documents(self, user_id: str) -> List[DocumentMetadata]:
                return []

            async def get_document(self, user_id: str, document_id: str) -> Optional[DocumentMetadata]:
                return None

            async def delete_document(self, user_id: str, document_id: str) -> bool:
                return True

        store = MockRAGStore()
        assert isinstance(store, BaseRAGStore)
        meta = DocumentMetadata(
            user_id="u1",
            filename="f.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=0,
        )
        await store.add_document(meta, [], [])
        results = await store.search("u1", [0.1] * 384, "test")
        assert results == []
        docs = await store.list_documents("u1")
        assert docs == []
        d = await store.get_document("u1", "doc-1")
        assert d is None
        deleted = await store.delete_document("u1", "doc-1")
        assert deleted is True
