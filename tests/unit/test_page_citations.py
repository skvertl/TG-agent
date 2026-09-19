"""Unit tests for Page Number Citations Verification across RAG Pipeline."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from adapters.rag.parsers import RecursiveCharacterChunker
from adapters.rag.sqlite_vec_store import SqliteVecStore
from core.models import DocumentMetadata, SearchResult
from core.ports import BaseRAGStore, BaseEmbeddingProvider, BaseLLMPlugin, BaseMemoryStore
from core.tools.rag import create_rag_search_tool
from core.tools.registry import ToolRegistry
from core.runner import AgentRunner


class TestPageNumberCitations:
    def test_page_number_propagation_from_pages_to_chunks(self):
        pages = [
            (1, "Page 1: Overview of corporate governance and leadership principles."),
            (2, "Page 2: Financial reporting standards and internal audit policies."),
            (3, "Page 3: Employee benefits, annual leave, and compensation structure."),
        ]
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_pages(pages, document_id="doc_abc", user_id="user_1")

        assert len(chunks) >= 3
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == {1, 2, 3}

        # Verify chunks have correct source page
        for c in chunks:
            if "leadership" in c.content:
                assert c.page_number == 1
            elif "audit" in c.content:
                assert c.page_number == 2
            elif "compensation" in c.content:
                assert c.page_number == 3

    @pytest.mark.asyncio
    async def test_page_number_persisted_and_returned_in_search_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_citations.db"
            store = SqliteVecStore(db_path=str(db_path))

            meta = DocumentMetadata(
                user_id="user_citation",
                filename="handbook.pdf",
                file_type="pdf",
                file_size_bytes=1000,
                page_count=3,
            )
            pages = [
                (1, "Page 1 intro"),
                (2, "Page 2 detailed instructions on vacation scheduling."),
            ]
            chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
            chunks = chunker.split_pages(pages, document_id=meta.id, user_id=meta.user_id)

            embeddings = [[0.1 * i] * 384 for i in range(1, len(chunks) + 1)]
            await store.add_document(meta, chunks, embeddings)

            # Search with query_text matching page 2
            results = await store.search(
                user_id="user_citation",
                query_embedding=[0.1] * 384,
                query_text="vacation scheduling",
                top_k=4,
                score_threshold=0.0,
            )

            assert len(results) > 0
            page_2_chunk = next((r for r in results if "vacation scheduling" in r.content), None)
            assert page_2_chunk is not None
            assert page_2_chunk.filename == "handbook.pdf"
            assert page_2_chunk.page_number == 2

    def test_search_documents_tool_formats_page_citation_correctly(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        mock_provider.embed_text.return_value = [0.1] * 384

        mock_store.search.return_value = [
            SearchResult(
                chunk_id="c_page",
                document_id="d1",
                filename="safety_rules.pdf",
                content="Always wear protective equipment on site.",
                page_number=7,
                score=0.0328,
                source="hybrid",
            ),
            SearchResult(
                chunk_id="c_nopage",
                document_id="d2",
                filename="readme.md",
                content="Project readme without page information.",
                page_number=None,
                score=0.0164,
                source="fts",
            ),
        ]

        tool = create_rag_search_tool(mock_store, mock_provider)
        output = tool.execute(query="safety rules", session_id="user_123")

        # Result 1 must include "(Page 7, Score: ...)"
        assert "File: safety_rules.pdf (Page 7, Score: 0.03)" in output
        assert '"Always wear protective equipment on site."' in output

        # Result 2 without page must NOT show "Page None"
        assert "File: readme.md (Score: 0.02)" in output
        assert "Page None" not in output

    def test_agent_system_prompt_enforces_page_citations(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        tool = create_rag_search_tool(mock_store, mock_provider)

        registry = ToolRegistry(engine=MagicMock())
        registry.register(tool)

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            tool_registry=registry,
        )

        prompt = runner._build_system_prompt_with_tools()
        # Must instruct citation format [filename, p. X]
        assert "[filename, p. X]" in prompt
        assert "search_documents" in prompt
        assert "Я не нашёл этой информации в загруженных документах." in prompt
