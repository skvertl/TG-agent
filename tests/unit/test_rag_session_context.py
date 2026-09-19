"""Unit tests for session context propagation and RAG user isolation."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.runner import AgentRunner
from core.models import AgentResult
from core.ports import BaseLLMPlugin, BaseMemoryStore, BaseRAGStore, BaseEmbeddingProvider
from core.tools.registry import ToolRegistry
from core.tools.base import Tool
from core.tools.rag import create_rag_search_tool
from core.observability.engine import ObservabilityEngine


class TestSessionContextPropagation:
    def test_tool_registry_forwards_session_id(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        received_context = {}

        def sample_tool(query: str, session_id: str = None):
            received_context["query"] = query
            received_context["session_id"] = session_id
            return f"Processed {query} for {session_id}"

        registry.register(
            Tool(
                name="context_tool",
                description="Context testing tool",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=sample_tool,
            )
        )

        out = registry.execute(
            tool_name="context_tool",
            task_id="task-1",
            turn_number=1,
            arguments={"query": "test query"},
            session_id="session-user-123",
        )

        assert "session-user-123" in out
        assert received_context["session_id"] == "session-user-123"
        assert received_context["query"] == "test query"

    def test_tool_registry_backward_compatibility_with_unaware_tools(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        def strict_tool(path: str):
            return f"Path is {path}"

        registry.register(
            Tool(
                name="strict_tool",
                description="Strict tool without session_id",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                func=strict_tool,
            )
        )

        out = registry.execute(
            tool_name="strict_tool",
            task_id="task-1",
            turn_number=1,
            arguments={"path": "main.py"},
            session_id="ignored-session",
        )

        assert out == "Path is main.py"

    @pytest.mark.asyncio
    async def test_agent_runner_passes_session_id_to_registry_execute(self):
        mock_llm = AsyncMock(spec=BaseLLMPlugin)
        mock_llm.generate.side_effect = [
            AgentResult(
                content='Action: test_action\nAction Input: {"param": "value"}',
                model="test-model",
            ),
            AgentResult(
                content="Final Answer: Finished",
                model="test-model",
            ),
        ]

        mock_memory = AsyncMock(spec=BaseMemoryStore)
        mock_memory.get_history.return_value = []

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_definitions.return_value = []
        mock_registry.execute.return_value = "Tool observation output"

        runner = AgentRunner(
            llm_plugin=mock_llm,
            memory_store=mock_memory,
            tool_registry=mock_registry,
        )

        await runner.run(
            session_id="telegram_user_456",
            user_prompt="Run the action",
        )

        mock_registry.execute.assert_called_once_with(
            tool_name="test_action",
            task_id=mock_registry.execute.call_args[1]["task_id"],
            turn_number=1,
            arguments={"param": "value"},
            session_id="telegram_user_456",
        )

    @pytest.mark.asyncio
    async def test_search_documents_end_to_end_user_isolation(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        mock_store = AsyncMock(spec=BaseRAGStore)
        mock_store.search.return_value = []
        mock_emb = AsyncMock(spec=BaseEmbeddingProvider)
        mock_emb.embed_text.return_value = [0.1] * 384

        rag_tool = create_rag_search_tool(mock_store, mock_emb)
        registry.register(rag_tool)

        mock_memory = AsyncMock(spec=BaseMemoryStore)
        mock_memory.get_history.return_value = []

        # User A session
        mock_llm_a = AsyncMock(spec=BaseLLMPlugin)
        mock_llm_a.generate.side_effect = [
            AgentResult(
                content='Action: search_documents\nAction Input: {"query": "User A secret"}',
                model="test-model",
            ),
            AgentResult(content="Final Answer: Done", model="test-model"),
        ]

        runner_a = AgentRunner(
            llm_plugin=mock_llm_a,
            memory_store=mock_memory,
            tool_registry=registry,
        )

        await runner_a.run(session_id="user_A_id", user_prompt="Find my secret")

        mock_store.search.assert_called_with(
            user_id="user_A_id",
            query_embedding=[0.1] * 384,
            query_text="User A secret",
            top_k=4,
            score_threshold=0.5,
        )

        # User B session
        mock_llm_b = AsyncMock(spec=BaseLLMPlugin)
        mock_llm_b.generate.side_effect = [
            AgentResult(
                content='Action: search_documents\nAction Input: {"query": "User B secret"}',
                model="test-model",
            ),
            AgentResult(content="Final Answer: Done", model="test-model"),
        ]

        runner_b = AgentRunner(
            llm_plugin=mock_llm_b,
            memory_store=mock_memory,
            tool_registry=registry,
        )

        await runner_b.run(session_id="user_B_id", user_prompt="Find my secret")

        mock_store.search.assert_called_with(
            user_id="user_B_id",
            query_embedding=[0.1] * 384,
            query_text="User B secret",
            top_k=4,
            score_threshold=0.5,
        )
