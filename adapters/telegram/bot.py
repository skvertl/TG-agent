from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from core.runner import AgentRunner
from adapters.telegram.handlers import router


def create_bot(token: str) -> Bot:
    """Создает и конфигурирует инстанс Bot aiogram."""
    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )


def create_dispatcher(runner: AgentRunner) -> Dispatcher:
    """Создает Dispatcher, подключает роутеры и внедряет зависимости."""
    dp = Dispatcher()
    dp.include_router(router)
    dp["runner"] = runner
    return dp
