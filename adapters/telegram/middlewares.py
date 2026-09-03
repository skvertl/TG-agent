import logging
from typing import Callable, Dict, Any, Awaitable, Set, Collection
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Message

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseMiddleware):
    """
    Middleware для ограничения доступа к боту по белому списку Telegram ID.
    Если белый список пуст, доступ разрешен всем.
    """

    def __init__(self, allowed_user_ids: Collection[int] | None = None):
        super().__init__()
        self.allowed_user_ids: Set[int] = set(allowed_user_ids or [])

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        if not self.allowed_user_ids:
            return await handler(event, data)

        from_user = getattr(event, "from_user", None)
        if from_user is not None:
            if from_user.id not in self.allowed_user_ids:
                logger.warning(
                    "Unauthorized access attempt by user_id=%s (@%s)",
                    from_user.id,
                    getattr(from_user, "username", "unknown"),
                )
                if isinstance(event, Message):
                    await event.answer(
                        f"⛔ **Доступ ограничен.** Ваш ID (`{from_user.id}`) не находится в списке разрешенных пользователей.",
                        parse_mode="Markdown",
                    )
                return None

        return await handler(event, data)
