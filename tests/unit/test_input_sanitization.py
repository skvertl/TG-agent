"""Unit tests for user input sanitization and guardrails in AgentRunner."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.models import PromptPayload, AgentResult
from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.runner import AgentRunner, sanitize_input
from core.tools.registry import ToolRegistry


class TestSanitizeInputFunction:
    def test_empty_string(self):
        assert sanitize_input("") == ""

    def test_whitespace_only(self):
        assert sanitize_input("   \n\t  \r  ") == ""

    def test_strips_null_bytes(self):
        raw = "Hello\x00World!\x00"
        assert sanitize_input(raw) == "HelloWorld!"

    def test_strips_non_printable_control_characters(self):
        # Control characters \x01, \x08, \x0b, \x0c, \x1f, \x7f
        raw = "\x01Hello\x08 \x0bWorld\x0c!\x1f\x7f"
        assert sanitize_input(raw) == "Hello World!"

    def test_preserves_tabs_newlines_carriage_returns(self):
        raw = "Line 1\r\nLine 2\tIndented\nLine 3"
        assert sanitize_input(raw) == "Line 1\r\nLine 2\tIndented\nLine 3"

    def test_preserves_markdown_and_cyrillic(self):
        raw = (
            "**Важный заголовок**\n\n"
            "* Пункт 1: [ссылка](https://example.com)\n"
            "* Пункт 2: `inline_code()`\n\n"
            "```python\ndef test():\n    return 'Привет, мир!'\n```"
        )
        assert sanitize_input(raw) == raw

    def test_truncates_exceeding_4096_characters(self):
        long_text = "A" * 5000
        sanitized = sanitize_input(long_text, max_chars=4096)
        # Should start with exactly 4096 'A's
        assert sanitized.startswith("A" * 4096)
        # Should append warning notification
        assert "[Предупреждение: сообщение превысило 4096 символов и было автоматически сокращено]" in sanitized
        # Base portion before warning must be 4096 chars
        base_part = sanitized.split("\n\n[Предупреждение:")[0]
        assert len(base_part) == 4096

    def test_custom_max_chars(self):
        text = "1234567890"
        sanitized = sanitize_input(text, max_chars=5)
        assert sanitized.startswith("12345")
        assert "5 символов" in sanitized


class TestAgentRunnerInputGuard:
    @pytest.fixture
    def mock_llm_plugin(self):
        mock = AsyncMock(spec=BaseLLMPlugin)
        mock.generate.return_value = AgentResult(
            content="Answer from LLM",
            model="mock-llm",
            prompt_tokens=10,
            completion_tokens=10,
        )
        return mock

    @pytest.fixture
    def mock_memory_store(self):
        mock = AsyncMock(spec=BaseMemoryStore)
        mock.get_history.return_value = []
        return mock

    @pytest.fixture
    def runner(self, mock_llm_plugin, mock_memory_store):
        registry = ToolRegistry(engine=MagicMock())
        return AgentRunner(
            llm_plugin=mock_llm_plugin,
            memory_store=mock_memory_store,
            tool_registry=registry,
        )

    @pytest.mark.asyncio
    async def test_empty_string_short_circuits(self, runner, mock_llm_plugin, mock_memory_store):
        result = await runner.run(session_id="test_user", user_prompt="")

        assert "Пожалуйста, отправьте текстовый вопрос или команду." in result.content
        assert result.model == "input_guard"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

        # LLM must NOT be called
        mock_llm_plugin.generate.assert_not_called()
        # Memory store must NOT save empty message
        mock_memory_store.save_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_short_circuits(self, runner, mock_llm_plugin, mock_memory_store):
        result = await runner.run(session_id="test_user", user_prompt="   \n\t  \r  ")

        assert "Пожалуйста, отправьте текстовый вопрос или команду." in result.content
        assert result.model == "input_guard"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

        mock_llm_plugin.generate.assert_not_called()
        mock_memory_store.save_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_null_bytes_only_short_circuits(self, runner, mock_llm_plugin, mock_memory_store):
        result = await runner.run(session_id="test_user", user_prompt="\x00\x00\x00")

        assert "Пожалуйста, отправьте текстовый вопрос или команду." in result.content
        mock_llm_plugin.generate.assert_not_called()
        mock_memory_store.save_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_prompt_with_null_bytes_is_sanitized_and_processed(
        self, runner, mock_llm_plugin, mock_memory_store
    ):
        result = await runner.run(session_id="test_user", user_prompt="Hello\x00 World!\x01")

        assert result.content == "Answer from LLM"
        mock_llm_plugin.generate.assert_called_once()
        payload: PromptPayload = mock_llm_plugin.generate.call_args[0][0]
        # Verify sanitized prompt was passed into ChatMessage
        user_msg = next(m for m in payload.messages if m.role == "user")
        assert user_msg.content == "Hello World!"

    @pytest.mark.asyncio
    async def test_long_prompt_is_truncated_with_warning(
        self, runner, mock_llm_plugin, mock_memory_store
    ):
        long_prompt = "Z" * 5000
        result = await runner.run(session_id="test_user", user_prompt=long_prompt)

        assert result.content == "Answer from LLM"
        mock_llm_plugin.generate.assert_called_once()
        payload: PromptPayload = mock_llm_plugin.generate.call_args[0][0]
        user_msg = next(m for m in payload.messages if m.role == "user")
        assert user_msg.content.startswith("Z" * 4096)
        assert "[Предупреждение: сообщение превысило 4096 символов и было автоматически сокращено]" in user_msg.content
