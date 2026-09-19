"""Unit tests for RAG domain models."""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from core.models import DocumentMetadata, DocumentChunk, SearchResult


class TestDocumentMetadata:
    def test_valid_document_metadata(self):
        created = datetime(2026, 9, 19, 10, 0, 0, tzinfo=timezone.utc)
        doc = DocumentMetadata(
            id="doc-123",
            user_id="user-123",
            filename="report.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            page_count=5,
            chunk_count=12,
            created_at=created,
        )
        assert doc.id == "doc-123"
        assert doc.user_id == "user-123"
        assert doc.filename == "report.pdf"
        assert doc.file_type == "pdf"
        assert doc.file_size_bytes == 1024
        assert doc.page_count == 5
        assert doc.chunk_count == 12
        assert doc.created_at == created

    def test_default_id_and_created_at(self):
        doc = DocumentMetadata(
            user_id="user-123",
            filename="notes.txt",
            file_type="txt",
            file_size_bytes=256,
            page_count=1,
            chunk_count=1,
        )
        assert isinstance(doc.id, str)
        assert len(doc.id) > 0
        assert isinstance(doc.created_at, datetime)

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(user_id="user-123")


class TestDocumentChunk:
    def test_valid_document_chunk(self):
        chunk = DocumentChunk(
            id="chunk-10",
            document_id="doc-1",
            user_id="user-123",
            content="This is chunk content.",
            page_number=2,
            chunk_index=0,
            token_count=5,
            metadata={"header": "Section 1"},
        )
        assert chunk.id == "chunk-10"
        assert chunk.document_id == "doc-1"
        assert chunk.user_id == "user-123"
        assert chunk.content == "This is chunk content."
        assert chunk.page_number == 2
        assert chunk.chunk_index == 0
        assert chunk.token_count == 5
        assert chunk.metadata == {"header": "Section 1"}

    def test_defaults(self):
        chunk = DocumentChunk(
            document_id="doc-1",
            user_id="user-123",
            content="Sample text",
            chunk_index=3,
        )
        assert isinstance(chunk.id, str)
        assert len(chunk.id) > 0
        assert chunk.document_id == "doc-1"
        assert chunk.page_number is None
        assert chunk.token_count is None
        assert chunk.metadata == {}

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DocumentChunk(user_id="user-123")  # Missing document_id, content, chunk_index


class TestSearchResult:
    def test_valid_search_result(self):
        result = SearchResult(
            chunk_id="chunk-10",
            document_id="doc-1",
            filename="guide.docx",
            content="Relevant passage",
            page_number=3,
            score=0.92,
            source="vector",
        )
        assert result.chunk_id == "chunk-10"
        assert result.document_id == "doc-1"
        assert result.filename == "guide.docx"
        assert result.content == "Relevant passage"
        assert result.page_number == 3
        assert result.score == 0.92
        assert result.source == "vector"

    def test_valid_sources(self):
        r1 = SearchResult(
            chunk_id="c1",
            document_id="d1",
            filename="f.txt",
            content="text",
            score=0.8,
            source="fts",
        )
        assert r1.source == "fts"

        r2 = SearchResult(
            chunk_id="c2",
            document_id="d2",
            filename="f.txt",
            content="text",
            score=0.8,
            source="hybrid",
        )
        assert r2.source == "hybrid"

    def test_invalid_source_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="c1",
                document_id="d1",
                filename="f.txt",
                content="text",
                score=0.8,
                source="invalid",  # type: ignore[arg-type]
            )
