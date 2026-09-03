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
        mock_memory.get_history.assert_awaited_once_with("session-1")

        # Check plugin was called with expected payload
        mock_plugin.generate.assert_awaited_once()
        payload: PromptPayload = mock_plugin.generate.call_args[0][0]
        assert payload.temperature == 0.7
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
