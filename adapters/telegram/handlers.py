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
        "📊 **Мониторинг:** используйте команду /token_report для просмотра статистики токенов.\n"
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
        "  /help — данная справка\n"
        "  /token_report — глобальная сводка по токенам, затратам и таймлайн последнего запуска\n"
        "  /token_report <task_id> — детальный таймлайн конкретной задачи"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("token_report"))
async def token_report_handler(message: Message, runner: AgentRunner) -> None:
    """Обработчик команды /token_report: глобальная сводка, история и таймлайн."""
    from core.observability.presenter import format_telegram_report, format_timeline_markdown
    from core.observability.storage import TelemetryStorage

    if getattr(runner, "observability_engine", None):
        storage = runner.observability_engine.storage
    else:
        storage = TelemetryStorage()

    args = message.text.split(maxsplit=1) if message.text else []
    if len(args) > 1 and args[1].strip():
        task_id = args[1].strip()
        timeline = storage.get_run_timeline(task_id)
        if not timeline:
            await message.answer(f"❌ Задача с ID `{task_id}` не найдена в базе телеметрии.", parse_mode="Markdown")
            return
        report_text = format_timeline_markdown(timeline)
        await message.answer(report_text, parse_mode="Markdown")
        return

    stats = storage.get_global_stats()
    recent = storage.get_recent_runs(limit=5)
    last_timeline = storage.get_run_timeline(recent[0].task_id) if recent else None

    report_text = format_telegram_report(
        stats=stats,
        recent_runs=recent,
        last_timeline=last_timeline,
    )
    await message.answer(report_text, parse_mode="Markdown")


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
