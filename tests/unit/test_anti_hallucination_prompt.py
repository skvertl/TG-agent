"""Unit tests for anti-hallucination and citation directives in AgentRunner system prompt."""
from unittest.mock import AsyncMock

from core.runner import AgentRunner
from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.tools.registry import ToolRegistry
from core.tools.base import Tool
from core.observability.engine import ObservabilityEngine


class TestAntiHallucinationPrompt:
    def test_prompt_includes_document_search_and_citation_rules(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))
        registry.register(
            Tool(
                name="search_documents",
                description="Search uploaded documents",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=lambda query: "results",
            )
        )

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            tool_registry=registry,
        )

        prompt = runner._build_system_prompt_with_tools()

        # Directive: Use search_documents for private files/documents/policies
        assert "search_documents" in prompt
        assert "documents" in prompt.lower()

        # Directive: Citation format [filename, p. X]
        assert "[filename, p. X]" in prompt

        # Directive: Anti-hallucination fallback phrase
        assert "Я не нашёл этой информации в загруженных документах." in prompt
        assert "NEVER invent" in prompt or "never invent" in prompt.lower()

    def test_prompt_preserves_react_structure_and_russian_mandate(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            tool_registry=registry,
        )

        prompt = runner._build_system_prompt_with_tools()

        assert "Action:" in prompt
        assert "Action Input:" in prompt
        assert "Final Answer:" in prompt
        assert "Russian (Cyrillic)" in prompt
