import logging
from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from core.runner import AgentRunner
from core.exceptions import LLMTimeoutError, LLMConnectionError, LLMResponseError
from adapters.telegram.splitter import split_message
from adapters.telegram.typing import keep_typing

logger = logging.getLogger(__name__)

router = Router(name="main_router")


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    """Обработчик команды /start."""
    text = (
        "👋 **Привет!** Я Telegram-шлюз к локальной нейросети через **Ollama**.\n\n"
        "⚡ **Режим работы:** `Stateless` (одноразовый запрос без сохранения истории диалога).\n"
        "💬 Просто отправьте мне любой вопрос или задачу, и я сгенерирую ответ!"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    """Обработчик команды /help."""
    text = (
        "ℹ️ **Справка и помощь:**\n\n"
        "• Отправьте любое текстовое сообщение для запроса к локальной LLM.\n"
        "• Ответы длиннее 4000 символов автоматически делятся на части без повреждения блоков кода.\n"
        "• Допустимый таймаут генерации: 60 секунд.\n"
        "• Команды:\n"
        "  /start — перезапуск и приветствие\n"
        "  /help — данная справка"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(F.text)
async def message_handler(message: Message, runner: AgentRunner) -> None:
    """Основной обработчик текстовых сообщений с вызовом инференса."""
    if not message.text:
        return

    session_id = str(message.from_user.id if message.from_user else message.chat.id)

    async with keep_typing(message.bot, message.chat.id):
        try:
            result = await runner.run(session_id=session_id, user_prompt=message.text)
            chunks = split_message(result.content, max_chunk_size=4000)

            for chunk in chunks:
                try:
                    await message.answer(chunk, parse_mode="Markdown")
                except Exception:
                    # Fallback на отправку без parse_mode, если в сгенерированном ответе поврежден синтаксис Markdown
                    await message.answer(chunk)

        except LLMTimeoutError as e:
            logger.warning("LLM Timeout for user %s: %s", session_id, e)
            await message.answer(
                "⚠️ **Превышено время ожидания ответа (таймаут 60.0s).** "
                "Модель не успела ответить вовремя. Попробуйте повторить запрос позже.",
                parse_mode="Markdown",
            )
        except LLMConnectionError as e:
            logger.error("LLM Connection error for user %s: %s", session_id, e)
            await message.answer(
                "🔌 **Ошибка подключения к сервису Ollama.** "
                "Сервис инференса временно недоступен.",
                parse_mode="Markdown",
            )
        except LLMResponseError as e:
            logger.error("LLM Response error for user %s: %s (status=%s)", session_id, e, e.status_code)
            await message.answer(
                f"❌ **Ошибка сервиса модели ({e.status_code or 'N/A'}):** {e}",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.exception("Unexpected error during inference: %s", e)
            await message.answer(
                "⚠️ **Произошла непредвиденная ошибка** при обработке вашего запроса.",
                parse_mode="Markdown",
            )
