from unittest.mock import AsyncMock
import pytest

from core.models import ChatMessage, PromptPayload, AgentResult
from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.exceptions import LLMTimeoutError
from core.runner import AgentRunner


@pytest.fixture
def mock_plugin():
    plugin = AsyncMock(spec=BaseLLMPlugin)
    plugin.generate.return_value = AgentResult(
        content="Assistant response",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=20,
    )
    return plugin


@pytest.fixture
def mock_memory():
    memory = AsyncMock(spec=BaseMemoryStore)
    memory.get_history.return_value = []
    return memory


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_run_basic_flow_with_defaults(self, mock_plugin, mock_memory):
        runner = AgentRunner(llm_plugin=mock_plugin, memory_store=mock_memory)

        result = await runner.run(session_id="session-1", user_prompt="Hello!")

        assert result.content == "Assistant response"
        assert result.model == "test-model"

        # Check history was requested
        mock_memory.get_history.assert_awaited_once_with("session-1", limit=10)

        # Check plugin was called with expected payload
        mock_plugin.generate.assert_awaited_once()
        payload: PromptPayload = mock_plugin.generate.call_args[0][0]
        assert payload.temperature == 0.3
        assert len(payload.messages) == 2
        assert payload.messages[0] == ChatMessage(
            role="system", content="You are a helpful AI assistant."
        )
        assert payload.messages[1] == ChatMessage(role="user", content="Hello!")

        # Check memory saved user message and assistant response
        assert mock_memory.save_message.await_count == 2
        mock_memory.save_message.assert_has_awaits(
            [
                pytest.helpers.call("session-1", ChatMessage(role="user", content="Hello!"))
                if hasattr(pytest, "helpers")
                else ("session-1", ChatMessage(role="user", content="Hello!"))
            ],
            any_order=False,
        ) if False else None

        calls = mock_memory.save_message.await_args_list
        assert calls[0].args == ("session-1", ChatMessage(role="user", content="Hello!"))
        assert calls[1].args == (
            "session-1",
            ChatMessage(role="assistant", content="Assistant response"),
        )

    @pytest.mark.asyncio
    async def test_run_with_custom_prompt_and_history(self, mock_plugin, mock_memory):
        existing_history = [
            ChatMessage(role="user", content="Earlier question"),
            ChatMessage(role="assistant", content="Earlier answer"),
        ]
        mock_memory.get_history.return_value = existing_history

        runner = AgentRunner(
            llm_plugin=mock_plugin,
            memory_store=mock_memory,
            system_prompt="Custom System Prompt",
            default_temperature=0.3,
        )

        result = await runner.run(session_id="session-2", user_prompt="New question")

        payload: PromptPayload = mock_plugin.generate.call_args[0][0]
        assert payload.temperature == 0.3
        assert len(payload.messages) == 4
        assert payload.messages[0] == ChatMessage(role="system", content="Custom System Prompt")
        assert payload.messages[1] == existing_history[0]
        assert payload.messages[2] == existing_history[1]
        assert payload.messages[3] == ChatMessage(role="user", content="New question")

    @pytest.mark.asyncio
    async def test_run_exception_propagation(self, mock_plugin, mock_memory):
        mock_plugin.generate.side_effect = LLMTimeoutError("Inference timed out")

        runner = AgentRunner(llm_plugin=mock_plugin, memory_store=mock_memory)

        with pytest.raises(LLMTimeoutError):
            await runner.run(session_id="session-err", user_prompt="Crash me")

        # User message should not be saved or assistant message should not be saved if generation fails
        assert mock_memory.save_message.await_count == 0

    @pytest.mark.asyncio
    async def test_run_records_telemetry(self, mock_plugin, mock_memory, tmp_path):
        from core.observability.engine import ObservabilityEngine
        from core.observability.storage import TelemetryStorage

        storage = TelemetryStorage(db_path=str(tmp_path / "runner_obs.db"))
        engine = ObservabilityEngine(project_name="test-runner", storage=storage)

        runner = AgentRunner(
            llm_plugin=mock_plugin,
            memory_store=mock_memory,
            observability_engine=engine,
        )

        result = await runner.run(session_id="s-obs", user_prompt="Check telemetry")
        assert result.task_id is not None

        # Verify run summary in telemetry storage
        recent = storage.get_recent_runs()
        assert len(recent) == 1
        assert recent[0].task_id == result.task_id
        assert recent[0].total_input_tokens == 10
        assert recent[0].total_output_tokens == 20

    @pytest.mark.asyncio
    async def test_run_with_react_tool_execution(self, mock_memory, tmp_path):
        from core.observability.engine import ObservabilityEngine
        from core.observability.storage import TelemetryStorage
        from core.tools.registry import ToolRegistry
        from core.tools.base import Tool

        storage = TelemetryStorage(db_path=str(tmp_path / "runner_tools.db"))
        engine = ObservabilityEngine(project_name="test-react", storage=storage)
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        # Register custom test tool
        tool_called = False
        def my_tool(param: str = "") -> str:
            nonlocal tool_called
            tool_called = True
            return f"Tool result for {param}"

        registry.register(
            Tool(
                name="my_tool",
                description="Sample tool",
                parameters={"type": "object", "properties": {"param": {"type": "string"}}},
                func=my_tool,
            )
        )

        # Mock LLM to return Action on Turn 1 and Final Answer on Turn 2
        plugin = AsyncMock(spec=BaseLLMPlugin)
        plugin.generate.side_effect = [
            AgentResult(
                content='Action: my_tool\nAction Input: {"param": "foo"}',
                model="test-model",
                prompt_tokens=50,
                completion_tokens=20,
            ),
            AgentResult(
                content="Final Answer: Task complete with tool!",
                model="test-model",
                prompt_tokens=100,
                completion_tokens=25,
            ),
        ]

        runner = AgentRunner(
            llm_plugin=plugin,
            memory_store=mock_memory,
            observability_engine=engine,
            tool_registry=registry,
        )

        result = await runner.run(session_id="s-react", user_prompt="Use tool")

        assert tool_called is True
        assert result.content == "Task complete with tool!"
        assert plugin.generate.await_count == 2

        # Check telemetry
        timeline = storage.get_run_timeline(result.task_id)
        assert timeline.summary.turns_count == 2
        assert timeline.summary.tool_calls_count == 1

    def test_sanitize_output_filters_cjk_hallucinations(self):
        from core.runner import sanitize_output
        raw = "💡 Дайджест: Сегодня в Москве комфортная погода. Держите手机无法输入，请稍后尝试其他操作"
        cleaned = sanitize_output(raw)
        assert "手机无法输入" not in cleaned
        assert cleaned == "💡 Дайджест: Сегодня в Москве комфортная погода."

    @pytest.mark.asyncio
    async def test_run_sanitizes_cjk_in_final_answer(self, mock_memory):
        from core.runner import AgentRunner
        plugin = AsyncMock(spec=BaseLLMPlugin)
        plugin.generate.return_value = AgentResult(
            content="Final Answer: Сводка готова. Проверьте данные. Ожидайте稍后尝试",
            model="qwen2.5:7b",
        )
        runner = AgentRunner(llm_plugin=plugin, memory_store=mock_memory)
        res = await runner.run(session_id="cjk-test", user_prompt="Привет")
        assert "稍后尝试" not in res.content
        assert res.content == "Сводка готова. Проверьте данные."

