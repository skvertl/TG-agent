"""Unit tests for Hybrid Search with Reciprocal Rank Fusion (RRF) in SqliteVecStore."""
import pytest
import tempfile
from pathlib import Path

from adapters.rag.sqlite_vec_store import SqliteVecStore, _sanitize_fts_query
from core.models import DocumentMetadata, DocumentChunk


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_hybrid_rag.db"
        store = SqliteVecStore(db_path=str(db_path))
        yield store


class TestFtsQuerySanitization:
    def test_empty_or_whitespace_returns_none(self):
        assert _sanitize_fts_query("") is None
        assert _sanitize_fts_query("   ") is None
        assert _sanitize_fts_query("!@#$%^&*()") is None

    def test_alphanumeric_words_extracted_and_quoted(self):
        sanitized = _sanitize_fts_query("annual leave policy 2026")
        assert sanitized is not None
        assert '"annual"*' in sanitized
        assert '"leave"*' in sanitized
        assert '"policy"*' in sanitized
        assert " OR " in sanitized

    def test_cyrillic_tokens_supported(self):
        sanitized = _sanitize_fts_query("Сколько дней отпуска (в год)?")
        assert sanitized is not None
        assert '"Сколько"*' in sanitized
        assert '"отпуска"*' in sanitized
        assert '"дней"*' in sanitized
        # Single or short letter "в" quoted exactly
        assert '"в"' in sanitized


class TestHybridSearchRRF:
    @pytest.mark.asyncio
    async def test_fallback_to_pure_vector_when_query_text_is_none_or_empty(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="doc.txt",
            file_type="txt",
            file_size_bytes=100,
        )
        chunk = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Sample text content for vector testing",
            chunk_index=0,
        )
        vec = [0.1] * 384
        await temp_store.add_document(meta, [chunk], [vec])

        # 1. query_text=None
        res_none = await temp_store.search(
            user_id="user_1",
            query_embedding=[0.1] * 384,
            query_text=None,
            top_k=4,
            score_threshold=0.5,
        )
        assert len(res_none) == 1
        assert res_none[0].source == "vector"
        assert res_none[0].score >= 0.5

        # 2. query_text="   "
        res_empty = await temp_store.search(
            user_id="user_1",
            query_embedding=[0.1] * 384,
            query_text="   ",
            top_k=4,
            score_threshold=0.5,
        )
        assert len(res_empty) == 1
        assert res_empty[0].source == "vector"

    @pytest.mark.asyncio
    async def test_hybrid_source_and_rrf_scoring(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="policy.pdf",
            file_type="pdf",
            file_size_bytes=500,
            page_count=3,
        )
        # Chunk 1: High semantic similarity AND exact keyword match
        chunk1 = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Employees receive 28 calendar days of annual paid vacation.",
            page_number=1,
            chunk_index=0,
        )
        # Chunk 2: High semantic similarity, no exact keywords
        chunk2 = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Staff members are entitled to rest periods during the summer period.",
            page_number=2,
            chunk_index=1,
        )
        # Chunk 3: Keyword match only (e.g. 'vacation'), orthogonal vector
        chunk3 = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Unused vacation days cannot be carried over without approval.",
            page_number=3,
            chunk_index=2,
        )

        vec1 = [0.2] * 384
        vec2 = [0.19] * 384
        vec3 = [-0.2] * 384  # Negative similarity to query

        await temp_store.add_document(
            metadata=meta,
            chunks=[chunk1, chunk2, chunk3],
            embeddings=[vec1, vec2, vec3],
        )

        # Search with vector close to vec1/vec2 and keyword matching chunk1/chunk3
        query_vec = [0.2] * 384
        query_text = "annual paid vacation"

        results = await temp_store.search(
            user_id="user_1",
            query_embedding=query_vec,
            query_text=query_text,
            top_k=3,
            score_threshold=0.0,
        )

        assert len(results) >= 2
        top_result = results[0]

        # Chunk 1 matched both vector (rank 1) and FTS (rank 1) -> hybrid
        assert top_result.chunk_id == chunk1.id
        assert top_result.source == "hybrid"
        # RRF = 1 / (60 + 1) + 1 / (60 + 1) = 2 / 61 ≈ 0.0328
        expected_rrf = round(1.0 / 61.0 + 1.0 / 61.0, 4)
        assert abs(top_result.score - expected_rrf) <= 0.001

    @pytest.mark.asyncio
    async def test_score_threshold_filtering_on_rrf(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="handbook.md",
            file_type="md",
            file_size_bytes=200,
        )
        chunk1 = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Important security rules regarding password strength.",
            chunk_index=0,
        )
        chunk2 = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="General office cafeteria opening times and menu options.",
            chunk_index=1,
        )

        vec1 = [0.5] * 384
        vec2 = [-0.5] * 384

        await temp_store.add_document(meta, [chunk1, chunk2], [vec1, vec2])

        # Filter with RRF-scale threshold = 0.02 (only hybrid rank 1 with 0.0328 should pass)
        results = await temp_store.search(
            user_id="user_1",
            query_embedding=[0.5] * 384,
            query_text="security password rules",
            top_k=4,
            score_threshold=0.02,
        )

        assert len(results) == 1
        assert results[0].chunk_id == chunk1.id

    @pytest.mark.asyncio
    async def test_multi_tenant_fts_isolation(self, temp_store):
        meta_a = DocumentMetadata(
            user_id="user_A",
            filename="secret_a.txt",
            file_type="txt",
            file_size_bytes=100,
        )
        chunk_a = DocumentChunk(
            document_id=meta_a.id,
            user_id="user_A",
            content="Top secret project codename Prometheus.",
            chunk_index=0,
        )
        meta_b = DocumentMetadata(
            user_id="user_B",
            filename="secret_b.txt",
            file_type="txt",
            file_size_bytes=100,
        )
        chunk_b = DocumentChunk(
            document_id=meta_b.id,
            user_id="user_B",
            content="Unrelated public notes about Greek mythology.",
            chunk_index=0,
        )

        await temp_store.add_document(meta_a, [chunk_a], [[0.1] * 384])
        await temp_store.add_document(meta_b, [chunk_b], [[0.1] * 384])

        # User B searches for "Prometheus"
        b_res = await temp_store.search(
            user_id="user_B",
            query_embedding=[0.1] * 384,
            query_text="Prometheus",
            top_k=4,
            score_threshold=0.0,
        )
        # Should NOT find User A's Prometheus chunk
        for r in b_res:
            assert "Prometheus" not in r.content
            assert r.document_id != meta_a.id

    @pytest.mark.asyncio
    async def test_special_characters_in_query_do_not_crash(self, temp_store):
        meta = DocumentMetadata(
            user_id="user_1",
            filename="code.txt",
            file_type="txt",
            file_size_bytes=100,
        )
        chunk = DocumentChunk(
            document_id=meta.id,
            user_id="user_1",
            content="Special syntax: test_func() -> Dict[str, Any] * 42",
            chunk_index=0,
        )
        await temp_store.add_document(meta, [chunk], [[0.2] * 384])

        # Query with characters that typically break raw FTS5 queries: () [] -> * " :
        dirty_query = 'test_func() -> Dict[str, Any] * 42 "AND" OR NOT'
        res = await temp_store.search(
            user_id="user_1",
            query_embedding=[0.2] * 384,
            query_text=dirty_query,
            top_k=4,
        )
        assert len(res) == 1
        assert res[0].chunk_id == chunk.id
