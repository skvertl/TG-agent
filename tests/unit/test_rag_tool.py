"""Unit tests for search_documents ReAct tool."""
from unittest.mock import AsyncMock
from core.models import SearchResult
from core.ports import BaseRAGStore, BaseEmbeddingProvider
from core.tools.rag import create_rag_search_tool


class TestRagSearchTool:
    def test_tool_metadata(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        tool = create_rag_search_tool(mock_store, mock_provider)

        assert tool.name == "search_documents"
        assert "search" in tool.description.lower()
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    def test_execute_missing_user_context(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        tool = create_rag_search_tool(mock_store, mock_provider)

        result = tool.execute(query="salary policy")
        assert "Error: user context is required to access documents." in result

    def test_execute_empty_query(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        tool = create_rag_search_tool(mock_store, mock_provider)

        result = tool.execute(query="   ", session_id="user-123")
        assert "Error: query cannot be empty." in result

    def test_execute_success_formatting(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        mock_provider.embed_text.return_value = [0.1] * 384

        mock_store.search.return_value = [
            SearchResult(
                chunk_id="c1",
                document_id="d1",
                filename="contract.pdf",
                content="The employee shall receive 30 days of annual leave.",
                page_number=4,
                score=0.89,
                source="vector",
            ),
            SearchResult(
                chunk_id="c2",
                document_id="d2",
                filename="handbook.docx",
                content="Sick leave policy requires a doctor certificate.",
                page_number=None,
                score=0.74,
                source="vector",
            ),
        ]

        tool = create_rag_search_tool(mock_store, mock_provider)
        result = tool.execute(query="annual leave", session_id="user-42")

        assert "Found 2 relevant document excerpts:" in result
        assert "[1] File: contract.pdf (Page 4, Score: 0.89)" in result
        assert '"The employee shall receive 30 days of annual leave."' in result
        assert "[2] File: handbook.docx (Score: 0.74)" in result
        assert '"Sick leave policy requires a doctor certificate."' in result

        mock_provider.embed_text.assert_called_once_with("annual leave", is_query=True)
        mock_store.search.assert_called_once_with(
            user_id="user-42",
            query_embedding=[0.1] * 384,
            query_text="annual leave",
            top_k=4,
            score_threshold=0.5,
        )

    def test_execute_no_results(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        mock_provider.embed_text.return_value = [0.1] * 384
        mock_store.search.return_value = []

        tool = create_rag_search_tool(mock_store, mock_provider)
        result = tool.execute(query="secret rocket launch", session_id="user-42")

        assert "No relevant information found in your documents for query: 'secret rocket launch'." in result

    def test_user_id_preferred_over_session_id(self):
        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_provider = AsyncMock(spec=BaseEmbeddingProvider)
        mock_provider.embed_text.return_value = [0.1] * 384
        mock_store.search.return_value = []

        tool = create_rag_search_tool(mock_store, mock_provider)
        tool.execute(query="test", user_id="real-user-id", session_id="session-id-123")

        mock_store.search.assert_called_once_with(
            user_id="real-user-id",
            query_embedding=[0.1] * 384,
            query_text="test",
            top_k=4,
            score_threshold=0.5,
        )
