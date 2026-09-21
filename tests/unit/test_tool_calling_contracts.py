"""Unit tests for tool calling contracts, schema validation, JSON repair, and ReAct self-correction."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.models import AgentResult, PromptPayload
from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.runner import AgentRunner
from core.tools.base import Tool
from core.tools.registry import ToolRegistry


class TestToolRegistryContracts:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        mock_span = MagicMock()
        engine.track_tool.return_value.__enter__.return_value = mock_span
        return engine

    def test_execute_nonexistent_tool_returns_available_tools(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        # Clear default tools to test with known set
        registry._tools.clear()
        registry.register(
            Tool(
                name="tool_b",
                description="Tool B",
                parameters={"type": "object", "properties": {}},
                func=lambda: "b",
            )
        )
        registry.register(
            Tool(
                name="tool_a",
                description="Tool A",
                parameters={"type": "object", "properties": {}},
                func=lambda: "a",
            )
        )

        res = registry.execute(
            tool_name="nonexistent_tool",
            task_id="task-1",
            turn_number=1,
            arguments={},
        )
        assert res == "Error: Tool 'nonexistent_tool' not found. Available tools: tool_a, tool_b."

    def test_execute_nonexistent_tool_empty_registry(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        registry._tools.clear()

        res = registry.execute(
            tool_name="some_tool",
            task_id="task-1",
            turn_number=1,
            arguments={},
        )
        assert res == "Error: Tool 'some_tool' not found. Available tools: none."

    def test_execute_missing_required_argument(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        tool_params = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "limit": {"type": "integer", "description": "Max lines"},
            },
            "required": ["path"],
        }
        registry.register(
            Tool(
                name="read_file",
                description="Reads file",
                parameters=tool_params,
                func=lambda path, limit=None: f"content of {path}",
            )
        )

        res = registry.execute(
            tool_name="read_file",
            task_id="task-1",
            turn_number=1,
            arguments={"limit": 10},
        )
        expected = f"Error: Missing required argument 'path' for tool 'read_file'. Parameters schema: {tool_params}"
        assert res == expected

    def test_execute_none_required_argument(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        tool_params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        registry.register(
            Tool(
                name="search",
                description="Search tool",
                parameters=tool_params,
                func=lambda query: f"results for {query}",
            )
        )

        res = registry.execute(
            tool_name="search",
            task_id="task-1",
            turn_number=1,
            arguments={"query": None},
        )
        expected = f"Error: Missing required argument 'query' for tool 'search'. Parameters schema: {tool_params}"
        assert res == expected

    def test_execute_missing_argument_sets_span_output(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        tool_params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        registry.register(
            Tool(
                name="search",
                description="Search tool",
                parameters=tool_params,
                func=lambda query: f"results for {query}",
            )
        )

        res = registry.execute(
            tool_name="search",
            task_id="task-1",
            turn_number=1,
            arguments={},
        )
        mock_span = mock_engine.track_tool.return_value.__enter__.return_value
        mock_span.set_output.assert_called_once_with(res)

    def test_execute_valid_required_arguments_succeeds(self, mock_engine):
        registry = ToolRegistry(engine=mock_engine)
        registry.register(
            Tool(
                name="ping",
                description="Ping tool",
                parameters={"type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]},
                func=lambda host: f"pong to {host}",
            )
        )

        res = registry.execute(
            tool_name="ping",
            task_id="task-1",
            turn_number=1,
            arguments={"host": "127.0.0.1"},
        )
        assert res == "pong to 127.0.0.1"


class TestRunnerParseAction:
    @pytest.fixture
    def runner(self):
        llm = AsyncMock(spec=BaseLLMPlugin)
        mem = AsyncMock(spec=BaseMemoryStore)
        reg = ToolRegistry(engine=MagicMock())
        return AgentRunner(llm_plugin=llm, memory_store=mem, tool_registry=reg)

    def test_parse_valid_json(self, runner):
        text = 'Action: read_file\nAction Input: {"path": "main.py"}'
        parsed = runner._parse_action(text)
        assert parsed == ("read_file", {"path": "main.py"})

    def test_parse_trailing_commas(self, runner):
        text = 'Action: read_file\nAction Input: {"path": "main.py",}'
        parsed = runner._parse_action(text)
        assert parsed == ("read_file", {"path": "main.py"})

    def test_parse_single_quotes(self, runner):
        text = "Action: read_file\nAction Input: {'path': 'main.py'}"
        parsed = runner._parse_action(text)
        assert parsed == ("read_file", {"path": "main.py"})

    def test_parse_unquoted_keys(self, runner):
        text = 'Action: read_file\nAction Input: {path: "main.py"}'
        parsed = runner._parse_action(text)
        assert parsed == ("read_file", {"path": "main.py"})

    def test_parse_unquoted_keys_with_single_quotes_and_trailing_commas(self, runner):
        text = "Action: read_file\nAction Input: {path: 'main.py', offset: 10,}"
        parsed = runner._parse_action(text)
        assert parsed == ("read_file", {"path": "main.py", "offset": 10})

    def test_parse_multiline_malformed_json(self, runner):
        text = """Action: search_documents
Action Input: {
    query: 'financial report',
    limit: 5,
}"""
        parsed = runner._parse_action(text)
        assert parsed == ("search_documents", {"query": "financial report", "limit": 5})

    def test_parse_markdown_code_block_json(self, runner):
        text = """Action: run_shell
Action Input: ```json
{"command": "pytest"}
```"""
        parsed = runner._parse_action(text)
        assert parsed == ("run_shell", {"command": "pytest"})

    def test_parse_unrecoverable_malformed_json_returns_error(self, runner):
        text = "Action: read_file\nAction Input: {invalid json without braces"
        parsed = runner._parse_action(text)
        assert parsed is not None
        tool_name, tool_args = parsed
        assert tool_name == "read_file"
        assert isinstance(tool_args, dict)
        assert "__error__" in tool_args

    def test_parse_plain_string_arg(self, runner):
        text = "Action: search_documents\nAction Input: machine learning"
        parsed = runner._parse_action(text)
        assert parsed == ("search_documents", {"input": "machine learning"})


class TestAgentRunnerReActSelfCorrection:
    @pytest.mark.asyncio
    async def test_malformed_json_triggers_feedback_and_self_correction(self):
        llm = AsyncMock(spec=BaseLLMPlugin)
        mem = AsyncMock(spec=BaseMemoryStore)
        mem.get_history.return_value = []
        engine = MagicMock()
        registry = ToolRegistry(engine=engine)
        registry.register(
            Tool(
                name="echo",
                description="Echo tool",
                parameters={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                func=lambda msg: f"Echo: {msg}",
            )
        )

        # Turn 1: Model outputs broken JSON
        # Turn 2: Model receives error feedback and outputs corrected JSON
        # Turn 3: Model outputs Final Answer
        llm.generate.side_effect = [
            AgentResult(
                content="Action: echo\nAction Input: {msg: 'hello', invalid json",
                model="test-llm",
                prompt_tokens=10,
                completion_tokens=10,
            ),
            AgentResult(
                content='Action: echo\nAction Input: {"msg": "hello"}',
                model="test-llm",
                prompt_tokens=20,
                completion_tokens=15,
            ),
            AgentResult(
                content="Final Answer: Echo answered hello",
                model="test-llm",
                prompt_tokens=30,
                completion_tokens=10,
            ),
        ]

        runner = AgentRunner(
            llm_plugin=llm,
            memory_store=mem,
            tool_registry=registry,
        )

        result = await runner.run(session_id="user-123", user_prompt="Please echo hello")

        assert result.content == "Echo answered hello"
        assert llm.generate.call_count == 3

        # Verify that Turn 2 received the Invalid JSON error observation in messages
        turn_2_call = llm.generate.call_args_list[1]
        turn_2_payload: PromptPayload = turn_2_call[0][0]
        obs_messages = [m for m in turn_2_payload.messages if m.role == "user" and "Observation:" in m.content]
        assert any(
            "Error: Invalid JSON in Action Input. Please format arguments as valid JSON: {\"key\": \"value\"}" in m.content
            for m in obs_messages
        )

    @pytest.mark.asyncio
    async def test_missing_required_argument_feedback_and_self_correction(self):
        llm = AsyncMock(spec=BaseLLMPlugin)
        mem = AsyncMock(spec=BaseMemoryStore)
        mem.get_history.return_value = []
        engine = MagicMock()
        registry = ToolRegistry(engine=engine)
        registry.register(
            Tool(
                name="read_file",
                description="Reads file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                func=lambda path: f"Contents of {path}",
            )
        )

        # Turn 1: Missing required argument 'path'
        # Turn 2: Corrected action
        # Turn 3: Final Answer
        llm.generate.side_effect = [
            AgentResult(
                content='Action: read_file\nAction Input: {}',
                model="test-llm",
                prompt_tokens=10,
                completion_tokens=10,
            ),
            AgentResult(
                content='Action: read_file\nAction Input: {"path": "app.log"}',
                model="test-llm",
                prompt_tokens=20,
                completion_tokens=15,
            ),
            AgentResult(
                content="Final Answer: Log content checked",
                model="test-llm",
                prompt_tokens=30,
                completion_tokens=10,
            ),
        ]

        runner = AgentRunner(
            llm_plugin=llm,
            memory_store=mem,
            tool_registry=registry,
        )

        result = await runner.run(session_id="user-123", user_prompt="Check log")

        assert result.content == "Log content checked"
        assert llm.generate.call_count == 3

        # Verify that Turn 2 received missing argument error
        turn_2_call = llm.generate.call_args_list[1]
        turn_2_payload: PromptPayload = turn_2_call[0][0]
        obs_messages = [m for m in turn_2_payload.messages if m.role == "user" and "Observation:" in m.content]
        assert any(
            "Error: Missing required argument 'path' for tool 'read_file'" in m.content
            for m in obs_messages
        )
