import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from adapters.telegram.typing import keep_typing

class TestKeepTyping:
    @pytest.mark.asyncio
    async def test_keep_typing_sends_action_and_cancels_cleanly(self):
        bot = MagicMock()
        bot.send_chat_action = AsyncMock()
        
        async with keep_typing(bot, chat_id=12345, interval=0.05):
            await asyncio.sleep(0.12)
            
        assert bot.send_chat_action.call_count >= 2
        bot.send_chat_action.assert_called_with(chat_id=12345, action="typing")

    @pytest.mark.asyncio
    async def test_keep_typing_cancels_on_exception_inside_block(self):
        bot = MagicMock()
        bot.send_chat_action = AsyncMock()

        with pytest.raises(ValueError, match="Something failed"):
            async with keep_typing(bot, chat_id=12345, interval=0.05):
                await asyncio.sleep(0.01)
                raise ValueError("Something failed")

        # Verify no unhandled task leaks or errors

    @pytest.mark.asyncio
    async def test_keep_typing_ignores_bot_action_errors(self):
        bot = MagicMock()
        bot.send_chat_action = AsyncMock(side_effect=Exception("Telegram Network Glitch"))

        # Should not raise exception out of context manager
        async with keep_typing(bot, chat_id=12345, interval=0.05):
            await asyncio.sleep(0.06)
