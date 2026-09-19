"""Unit tests for SQLite + sqlite-vec RAG store."""
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from adapters.rag.sqlite_vec_store import SqliteVecStore
from core.models import DocumentMetadata, DocumentChunk
from core.ports.rag_store import BaseRAGStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_rag.db"
        store = SqliteVecStore(db_path=str(db_path))
        yield store


class TestSqliteVecStore:
    def test_implements_interface(self, temp_store):
        assert isinstance(temp_store, BaseRAGStore)

    @pytest.mark.asyncio
    async def test_add_and_get_document(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="guide.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            page_count=2,
            chunk_count=2,
        )
        chunks = [
            DocumentChunk(
                document_id=meta.id,
                user_id="user_1",
                content="Chunk 1 content",
                page_number=1,
                chunk_index=0,
                token_count=10,
            ),
            DocumentChunk(
                document_id=meta.id,
                user_id="user_1",
                content="Chunk 2 content",
                page_number=2,
                chunk_index=1,
                token_count=12,
            ),
        ]
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
        ]

        # S-1 & SPEC-2: add_document accepts metadata, chunks, embeddings explicitly
        await temp_store.add_document(
            metadata=meta,
            chunks=chunks,
            embeddings=embeddings,
        )

        fetched = await temp_store.get_document("user_1", meta.id)
        assert fetched is not None
        assert fetched.id == meta.id
        assert fetched.filename == "guide.pdf"
        assert fetched.chunk_count == 2
        assert fetched.page_count == 2
        assert isinstance(fetched.created_at, datetime)

    @pytest.mark.asyncio
    async def test_add_document_length_mismatch_raises_value_error(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="test.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=1,
        )
        chunks = [
            DocumentChunk(
                document_id=meta.id,
                user_id="user_1",
                content="Chunk content",
                chunk_index=0,
            )
        ]
        with pytest.raises(ValueError, match="must match"):
            await temp_store.add_document(
                metadata=meta,
                chunks=chunks,
                embeddings=[],  # Mismatched length
            )

    @pytest.mark.asyncio
    async def test_list_documents(self, temp_store):
        doc1 = DocumentMetadata(
            user_id="user_1",
            filename="doc1.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=0,
        )
        doc2 = DocumentMetadata(
            user_id="user_1",
            filename="doc2.md",
            file_type="md",
            file_size_bytes=200,
            page_count=1,
            chunk_count=0,
        )
        await temp_store.add_document(doc1, [], [])
        await temp_store.add_document(doc2, [], [])

        docs = await temp_store.list_documents("user_1")
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert filenames == {"doc1.txt", "doc2.md"}

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation_get_and_list(self, temp_store):
        doc_a = DocumentMetadata(
            user_id="user_A",
            filename="secret_a.txt",
            file_type="txt",
            file_size_bytes=500,
            page_count=1,
            chunk_count=0,
        )
        await temp_store.add_document(doc_a, [], [])

        # User B listing documents
        b_docs = await temp_store.list_documents("user_B")
        assert b_docs == []

        # User B attempting to get User A's document
        b_fetched = await temp_store.get_document("user_B", doc_a.id)
        assert b_fetched is None

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation_search(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_A",
            filename="finances.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            page_count=1,
            chunk_count=1,
        )
        chunks = [
            DocumentChunk(
                document_id=meta.id,
                user_id="user_A",
                content="Confidential financial report of user A.",
                page_number=1,
                chunk_index=0,
                token_count=15,
            )
        ]
        embeddings = [[0.5] * 384]

        await temp_store.add_document(
            metadata=meta,
            chunks=chunks,
            embeddings=embeddings,
        )

        # User B searches with the exact matching vector and text
        b_results = await temp_store.search(
            user_id="user_B",
            query_embedding=[0.5] * 384,
            query_text="financial report",
            top_k=4,
            score_threshold=0.5,
        )
        assert len(b_results) == 0

        # User A searches and finds the chunk
        a_results = await temp_store.search(
            user_id="user_A",
            query_embedding=[0.5] * 384,
            query_text="financial report",
            top_k=4,
            score_threshold=0.5,
        )
        assert len(a_results) == 1
        assert a_results[0].filename == "finances.pdf"
        assert a_results[0].content == "Confidential financial report of user A."
        assert a_results[0].page_number == 1
        assert a_results[0].score > 0
        assert isinstance(a_results[0].chunk_id, str)
        assert isinstance(a_results[0].document_id, str)

    @pytest.mark.asyncio
    async def test_search_score_threshold_filters_irrelevant(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_A",
            filename="sample.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=1,
        )
        chunks = [
            DocumentChunk(
                document_id=meta.id,
                user_id="user_A",
                content="Some content",
                chunk_index=0,
            )
        ]
        # Vector pointing in direction [1.0, 0, 0, ...]
        vec = [0.0] * 384
        vec[0] = 1.0

        await temp_store.add_document(meta, chunks, [vec])

        # Query vector pointing in opposite direction [-1.0, 0, 0, ...]
        opp_vec = [0.0] * 384
        opp_vec[0] = -1.0

        results = await temp_store.search(
            user_id="user_A",
            query_embedding=opp_vec,
            score_threshold=0.5,
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_delete_document_by_id_and_filename(self, temp_store):
        doc1 = DocumentMetadata(
            user_id="user_1",
            filename="file1.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=1,
        )
        chunk1 = DocumentChunk(
            document_id=doc1.id,
            user_id="user_1",
            content="Hello world",
            chunk_index=0,
        )
        doc2 = DocumentMetadata(
            user_id="user_1",
            filename="file2.txt",
            file_type="txt",
            file_size_bytes=200,
            page_count=1,
            chunk_count=0,
        )

        await temp_store.add_document(doc1, [chunk1], [[0.1] * 384])
        await temp_store.add_document(doc2, [], [])

        # S-3: Delete by document UUID string using batched statements
        res1 = await temp_store.delete_document("user_1", doc1.id)
        assert res1 is True
        assert await temp_store.get_document("user_1", doc1.id) is None

        # Verify search no longer finds deleted document's chunks
        search_after = await temp_store.search(
            user_id="user_1",
            query_embedding=[0.1] * 384,
            query_text="Hello world",
            score_threshold=0.0,
        )
        assert len(search_after) == 0

        # Delete by filename
        res2 = await temp_store.delete_document("user_1", "file2.txt")
        assert res2 is True
        assert await temp_store.get_document("user_1", doc2.id) is None

        # Nonexistent document
        res3 = await temp_store.delete_document("user_1", "nonexistent.txt")
        assert res3 is False

    @pytest.mark.asyncio
    async def test_delete_multi_tenant_isolation(self, temp_store):
        doc_a = DocumentMetadata(
            user_id="user_A",
            filename="doc_a.txt",
            file_type="txt",
            file_size_bytes=100,
            page_count=1,
            chunk_count=0,
        )
        await temp_store.add_document(doc_a, [], [])

        # User B attempts to delete User A's document
        deleted = await temp_store.delete_document("user_B", doc_a.id)
        assert deleted is False

        # User A's document is still present
        assert await temp_store.get_document("user_A", doc_a.id) is not None
