import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User
from adapters.telegram.middlewares import AuthMiddleware


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_no_whitelist_allows_all(self):
        middleware = AuthMiddleware(allowed_user_ids=[])
        handler = AsyncMock(return_value="success")

        msg = MagicMock(spec=Message)
        msg.from_user = MagicMock(spec=User)
        msg.from_user.id = 12345
        msg.answer = AsyncMock()

        res = await middleware(handler, msg, {})
        assert res == "success"
        handler.assert_called_once_with(msg, {})
        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_user_proceeds(self):
        middleware = AuthMiddleware(allowed_user_ids=[100, 200])
        handler = AsyncMock(return_value="handled")

        msg = MagicMock(spec=Message)
        msg.from_user = MagicMock(spec=User)
        msg.from_user.id = 100
        msg.answer = AsyncMock()

        res = await middleware(handler, msg, {})
        assert res == "handled"
        handler.assert_called_once_with(msg, {})
        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_disallowed_user_is_blocked(self):
        middleware = AuthMiddleware(allowed_user_ids=[100, 200])
        handler = AsyncMock()

        msg = MagicMock(spec=Message)
        msg.from_user = MagicMock(spec=User)
        msg.from_user.id = 999
        msg.answer = AsyncMock()

        res = await middleware(handler, msg, {})
        assert res is None
        handler.assert_not_called()
        msg.answer.assert_called_once()
        assert "Доступ ограничен" in msg.answer.call_args[0][0]
