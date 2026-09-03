from typing import Collection, Optional
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from core.runner import AgentRunner
from adapters.telegram.handlers import router
from adapters.telegram.middlewares import AuthMiddleware


def create_bot(token: str) -> Bot:
    """Создает и конфигурирует инстанс Bot aiogram."""
    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )


def create_dispatcher(
    runner: AgentRunner,
    allowed_user_ids: Optional[Collection[int]] = None
) -> Dispatcher:
    """Создает Dispatcher, подключает роутеры, middleware и внедряет зависимости."""
    dp = Dispatcher()
    if allowed_user_ids:
        dp.message.middleware(AuthMiddleware(allowed_user_ids=allowed_user_ids))
    dp.include_router(router)
    dp["runner"] = runner
    return dp
