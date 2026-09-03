import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

logger = logging.getLogger(__name__)


@asynccontextmanager
async def keep_typing(
    bot: Any, chat_id: int | str, interval: float = 4.5
) -> AsyncIterator[None]:
    """
    Асинхронный контекстный менеджер для периодической отправки
    действия 'typing' (печатает...) во время выполнения длительных операций (инференса).
    Гарантированно отменяет фоновую задачу при завершении или исключении в блоке.
    """
    stop_event = asyncio.Event()

    async def _typing_loop() -> None:
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Error while sending typing chat action: %s", e)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

    task = asyncio.create_task(_typing_loop())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
