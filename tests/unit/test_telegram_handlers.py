import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User, Chat
from core.models import AgentResult
from core.exceptions import LLMTimeoutError, LLMConnectionError, LLMResponseError
from adapters.telegram.handlers import start_handler, help_handler, message_handler

@pytest.fixture
def mock_message():
    msg = MagicMock(spec=Message)
    msg.message_id = 1
    msg.text = "Hello, AI!"
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = 999
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = 999
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.answer = AsyncMock()
    return msg

@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run = AsyncMock()
    return runner

class TestTelegramHandlers:
    @pytest.mark.asyncio
    async def test_start_command(self, mock_message):
        await start_handler(mock_message)
        mock_message.answer.assert_called_once()
        args, kwargs = mock_message.answer.call_args
        assert "Привет" in args[0] or "Welcome" in args[0] or "Ollama" in args[0]

    @pytest.mark.asyncio
    async def test_help_command(self, mock_message):
        await help_handler(mock_message)
        mock_message.answer.assert_called_once()
        args, kwargs = mock_message.answer.call_args
        assert "Помощь" in args[0] or "Команды" in args[0] or "Ollama" in args[0]

    @pytest.mark.asyncio
    async def test_message_success_single_chunk(self, mock_message, mock_runner):
        mock_runner.run.return_value = AgentResult(
            content="Hello human, I am Ollama.",
            model="qwen2.5:1.5b"
        )

        await message_handler(mock_message, mock_runner)

        mock_runner.run.assert_called_once_with(
            session_id="999",
            user_prompt="Hello, AI!"
        )
        mock_message.answer.assert_called_once()
        args, _ = mock_message.answer.call_args
        assert args[0] == "Hello human, I am Ollama."

    @pytest.mark.asyncio
    async def test_message_success_multi_chunk(self, mock_message, mock_runner):
        long_content = ("Paragraph one.\n\n" * 200) + ("Paragraph two.\n\n" * 200)
        mock_runner.run.return_value = AgentResult(
            content=long_content,
            model="qwen2.5:1.5b"
        )

        await message_handler(mock_message, mock_runner)

        assert mock_message.answer.call_count >= 2

    @pytest.mark.asyncio
    async def test_message_timeout_error_handling(self, mock_message, mock_runner):
        mock_runner.run.side_effect = LLMTimeoutError("Inference timeout")

        await message_handler(mock_message, mock_runner)

        mock_message.answer.assert_called_once()
        args, _ = mock_message.answer.call_args
        assert "таймаут" in args[0].lower() or "ожидания" in args[0].lower()

    @pytest.mark.asyncio
    async def test_message_connection_error_handling(self, mock_message, mock_runner):
        mock_runner.run.side_effect = LLMConnectionError("Connection refused")

        await message_handler(mock_message, mock_runner)

        mock_message.answer.assert_called_once()
        args, _ = mock_message.answer.call_args
        assert "подключения" in args[0].lower() or "недоступен" in args[0].lower()

    @pytest.mark.asyncio
    async def test_message_response_error_handling(self, mock_message, mock_runner):
        mock_runner.run.side_effect = LLMResponseError("Internal Error", status_code=500)

        await message_handler(mock_message, mock_runner)

        mock_message.answer.assert_called_once()
        args, _ = mock_message.answer.call_args
        assert "500" in args[0] or "ошибка" in args[0].lower()
