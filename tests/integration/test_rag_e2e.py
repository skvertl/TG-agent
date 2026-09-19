"""End-to-end integration tests for Autonomous RAG Agent Pipeline."""
import pytest
import tempfile
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from core.models import DocumentMetadata, DocumentChunk, PromptPayload, AgentResult
from core.ports import BaseLLMPlugin, BaseEmbeddingProvider
from core.runner import AgentRunner
from core.tools.registry import ToolRegistry
from core.tools.rag import create_rag_search_tool
from adapters.memory.stateless import StatelessMemoryStore
from adapters.rag.sqlite_vec_store import SqliteVecStore


class MockReActLLMPlugin(BaseLLMPlugin):
    """
    Mock LLM plugin simulating ReAct reasoning loop for RAG queries:
    - Calls search_documents on user question.
    - Synthesizes factual answer with citation if excerpt found.
    - Replies with honest fallback if no excerpts found.
    """

    def __init__(self, model_name: str = "mock-react-llm"):
        self.model_name = model_name
        self.call_history: list[PromptPayload] = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate_stream(self, payload: PromptPayload) -> AsyncIterator[str]:
        yield ""

    async def generate(self, payload: PromptPayload) -> AgentResult:
        self.call_history.append(payload)

        # Inspect turn history from conversation messages
        messages = payload.messages
        # Look for observation in the latest assistant/user sequence
        observation_content = None
        for m in reversed(messages):
            # In runner._prepare_turn_messages, the observation is formatted into messages
            if "Observation: " in m.content or "No relevant information found" in m.content or "relevant document excerpts" in m.content:
                observation_content = m.content
                break

        if observation_content is not None:
            if "28 календарных дней" in observation_content:
                content = (
                    "Final Answer: В компании 'Рога и Копыта' предусмотрено 28 календарных дней "
                    "ежегодного оплачиваемого отпуска.\n\n[policy.md, p. 1]"
                )
            else:
                content = "Final Answer: Я не нашёл этой информации в загруженных документах."
            return AgentResult(
                content=content,
                model=self.model_name,
                prompt_tokens=25,
                completion_tokens=20,
            )

        # Initial turn: model chooses to search documents
        tool_call_text = (
            "Thought: I should check the user's private documents for vacation policy.\n"
            "Action: search_documents\n"
            'Action Input: {"query": "оплачиваемый отпуск"}'
        )
        return AgentResult(
            content=tool_call_text,
            model=self.model_name,
            prompt_tokens=20,
            completion_tokens=15,
        )


class MockEmbeddingProvider(BaseEmbeddingProvider):
    DIMENSION = 384

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    async def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return [0.1] * self.DIMENSION

    async def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return [[0.1] * self.DIMENSION for _ in texts]


class TestRagEndToEndIntegration:
    @pytest.mark.asyncio
    async def test_full_rag_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "e2e_rag.db"
            rag_store = SqliteVecStore(db_path=str(db_path))
            embedding_provider = MockEmbeddingProvider()

            # 1. Ingest sample document for user_e2e
            doc_id = "doc_policy_001"
            meta = DocumentMetadata(
                id=doc_id,
                user_id="user_e2e",
                filename="policy.md",
                file_type="md",
                file_size_bytes=512,
                page_count=1,
                chunk_count=1,
            )
            chunk = DocumentChunk(
                document_id=doc_id,
                user_id="user_e2e",
                content=(
                    "В компании 'Рога и Копыта' предусмотрено 28 календарных дней "
                    "ежегодного оплачиваемого отпуска. Источник: policy.md, стр. 1"
                ),
                page_number=1,
                chunk_index=0,
            )
            embeddings = [[0.1] * 384]
            await rag_store.add_document(meta, [chunk], embeddings)

            # 2. Wire domain components
            search_tool = create_rag_search_tool(rag_store, embedding_provider)
            registry = ToolRegistry(engine=MagicMock())
            registry.register(search_tool)

            llm_plugin = MockReActLLMPlugin()
            memory_store = StatelessMemoryStore()

            runner = AgentRunner(
                llm_plugin=llm_plugin,
                memory_store=memory_store,
                tool_registry=registry,
                max_turns=5,
            )

            # 3. Query via AgentRunner
            user_question = "Сколько дней оплачиваемого отпуска положено сотруднику?"
            result = await runner.run(session_id="user_e2e", user_prompt=user_question)

            # Assert search_documents was invoked and answered accurately with citations
            assert "28 календарных дней" in result.content
            assert "policy.md" in result.content
            assert len(llm_plugin.call_history) == 2

            # 4. Multi-tenant isolation verification
            # Stranger user cannot see or search user_e2e's document
            stranger_results = await rag_store.search(
                user_id="user_stranger",
                query_embedding=[0.1] * 384,
                query_text="оплачиваемый отпуск",
                score_threshold=0.0,
            )
            assert len(stranger_results) == 0

            stranger_docs = await rag_store.list_documents("user_stranger")
            assert len(stranger_docs) == 0

            stranger_doc = await rag_store.get_document("user_stranger", doc_id)
            assert stranger_doc is None

            # 5. Delete document and verify subsequent queries return honest fallback
            deleted = await rag_store.delete_document("user_e2e", doc_id)
            assert deleted is True

            # Verify document is gone from store
            assert await rag_store.get_document("user_e2e", doc_id) is None
            empty_search = await rag_store.search(
                user_id="user_e2e",
                query_embedding=[0.1] * 384,
                query_text="оплачиваемый отпуск",
                score_threshold=0.0,
            )
            assert len(empty_search) == 0

            # Execute question again after deletion
            llm_plugin.call_history.clear()
            result_after_delete = await runner.run(
                session_id="user_e2e",
                user_prompt=user_question,
            )
            assert "Я не нашёл этой информации в загруженных документах." in result_after_delete.content
