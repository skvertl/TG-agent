import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User, Chat
from adapters.telegram.handlers import new_chat_handler


class TestNewCommandHandler:
    @pytest.mark.asyncio
    async def test_new_command_clears_memory_and_replies(self):
        msg = MagicMock(spec=Message)
        msg.from_user = MagicMock(spec=User)
        msg.from_user.id = 12345
        msg.chat = MagicMock(spec=Chat)
        msg.chat.id = 12345
        msg.answer = AsyncMock()

        runner = MagicMock()
        runner.memory_store = MagicMock()
        runner.memory_store.clear = AsyncMock()

        await new_chat_handler(msg, runner)

        runner.memory_store.clear.assert_called_once_with("12345")
        msg.answer.assert_called_once()
        reply_text = msg.answer.call_args[0][0]
        assert "Контекст очищен" in reply_text or "новый диалог" in reply_text
