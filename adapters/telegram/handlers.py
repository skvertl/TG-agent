import logging
from typing import Optional
from aiogram import Router, F
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.types import Message
from core.runner import AgentRunner
from core.ports import BaseRAGStore
from core.exceptions import LLMTimeoutError, LLMConnectionError, LLMResponseError
from adapters.telegram.splitter import split_message
from adapters.telegram.typing import keep_typing

logger = logging.getLogger(__name__)

router = Router(name="main_router")


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    """Обработчик команды /start."""
    text = (
        "👋 **Привет!** Я автономный AI-агент на базе локальной модели **Ollama**.\n\n"
        "🧠 **Режим работы:** `Stateful Agent` (непрерывный контекст диалога с памятью).\n"
        "🧹 **Новый чат:** используйте команду /new для сброса контекста.\n"
        "🛠 **Возможности:** выполнение консольных команд (`exec`), чтение файлов и выполнение сценариев (`skills`).\n"
        "📊 **Мониторинг:** команда /token\\_report для просмотра статистики и затрат токенов.\n\n"
        "💬 Отправьте мне вопрос, задачу или команду!"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    """Обработчик команды /help."""
    text = (
        "ℹ️ **Справка и помощь:**\n\n"
        "• Отправьте любое текстовое сообщение для запроса к автономному агенту.\n"
        "• Агент сохраняет контекст диалога. Чтобы начать с чистого листа, отправьте /new.\n"
        "• Ответы длиннее 4000 символов автоматически делятся на части.\n"
        "• Команды:\n"
        "  /start — приветствие и статус агента\n"
        "  /help — данная справка\n"
        "  /new — начать новый диалог (очистить контекст)\n"
        "  /documents — список загруженных документов\n"
        "  /delete <ID или имя файла> — удалить документ из базы\n"
        "  /morning\\_briefing [город] — утренняя сводка (по умолчанию Москва, например /morning\\_briefing Минск)\n"
        "  /weather [город] — прогноз погоды (например /weather London)\n"
        "  /system\\_health — запустить диагностику контейнера и системы\n"
        "  /skills — список доступных сценариев (скиллов)\n"
        "  /token\\_report — сводка по токенам, затратам и таймлайн последнего запуска\n"
        "  /token\\_report <task\\_id> — детальный таймлайн конкретной задачи"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("skills", "skill"))
async def skills_list_handler(message: Message, runner: AgentRunner) -> None:
    """Обработчик команды /skills: список доступных скиллов."""
    summary = runner._get_skills_summary()
    if not summary:
        text = "⚙️ В каталоге `skills/` пока нет зарегистрированных сценариев."
    else:
        text = f"🛠 **Доступные сценарии (Skills):**\n{summary}\n\nВы можете запустить их командами /morning\\_briefing [город], /system\\_health или обычным сообщением в чат."
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("documents", "docs", "mydocs"))
async def documents_list_handler(
    message: Message,
    rag_store: Optional[BaseRAGStore] = None,
) -> None:
    """Обработчик команды /documents: список проиндексированных документов пользователя."""
    if not rag_store:
        await message.answer("⚙️ RAG-хранилище документов не подключено.", parse_mode="Markdown")
        return

    user_id = str(message.from_user.id if message.from_user else message.chat.id)
    docs = await rag_store.list_documents(user_id)

    if not docs:
        await message.answer(
            "📚 У вас пока нет загруженных документов.\n\n"
            "Отправьте файл (.pdf, .docx, .txt, .md) в чат для индексации.",
            parse_mode="Markdown",
        )
        return

    lines = ["📚 **Ваши документы:**\n"]
    for i, d in enumerate(docs, start=1):
        size_str = (
            f"{d.file_size_bytes / (1024 * 1024):.1f} MB"
            if d.file_size_bytes >= 1024 * 1024
            else f"{d.file_size_bytes / 1024:.1f} KB"
        )
        date_str = (
            d.created_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(d.created_at, "strftime")
            else str(d.created_at)[:16]
        )
        doc_id_short = d.id[:8]
        lines.append(
            f"{i}. 📄 **{d.filename}**\n"
            f"   • ID: `{doc_id_short}`\n"
            f"   • Страниц: {d.page_count} | Чанков: {d.chunk_count} | {size_str}\n"
            f"   • Дата: {date_str}"
        )

    lines.append("\nДля удаления документа отправьте: /delete <ID или имя файла>")
    await message.answer("\n".join(lines), parse_mode="Markdown")


@router.message(Command("delete", "del", "remove"))
async def document_delete_handler(
    message: Message,
    command: Optional[CommandObject] = None,
    rag_store: Optional[BaseRAGStore] = None,
) -> None:
    """Обработчик команды /delete: удаление документа по ID или имени файла."""
    doc_ref = ""
    if command and command.args:
        doc_ref = command.args.strip()
    elif message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) > 1:
            doc_ref = parts[1].strip()

    if not doc_ref:
        await message.answer(
            "ℹ️ Использование: /delete <ID документа или имя файла>",
            parse_mode="Markdown",
        )
        return

    if not rag_store:
        await message.answer("⚙️ RAG-хранилище документов не подключено.", parse_mode="Markdown")
        return

    user_id = str(message.from_user.id if message.from_user else message.chat.id)
    success = await rag_store.delete_document(user_id, doc_ref)

    if success:
        await message.answer(
            f"🗑 Документ '{doc_ref}' успешно удален из базы и векторного индекса.",
            parse_mode="Markdown",
        )
    else:
        await message.answer(
            f"❌ Документ '{doc_ref}' не найден в вашем списке документов.",
            parse_mode="Markdown",
        )



@router.message(Command("morning_briefing", "morningbriefing", "morning-briefing", "weather"))
async def morning_briefing_cmd_handler(message: Message, runner: AgentRunner) -> None:
    """Прямой запуск скилла утренней сводки с поддержкой выбора города."""
    session_id = str(message.from_user.id if message.from_user else message.chat.id)
    args = message.text.split(maxsplit=1) if message.text else []
    city = args[1].strip() if len(args) > 1 and args[1].strip() else None

    if city:
        user_prompt = (
            f"Выполни утреннюю сводку по скиллу morning-briefing для города {city}: "
            f"узнай погоду в городе {city}, проверь дату и сформируй дайджест."
        )
    else:
        user_prompt = (
            "Выполни утреннюю сводку по скиллу morning-briefing (город по умолчанию Москва): "
            "узнай погоду, проверь дату и сформируй дайджест."
        )

    async with keep_typing(message.bot, message.chat.id):
        try:
            result = await runner.run(
                session_id=session_id,
                user_prompt=user_prompt,
            )
            chunks = split_message(result.content, max_chunk_size=4000)
            for chunk in chunks:
                try:
                    await message.answer(chunk, parse_mode="Markdown")
                except Exception:
                    await message.answer(chunk)
        except Exception as e:
            logger.exception("Error executing morning_briefing command: %s", e)
            await message.answer(f"⚠️ Ошибка при выполнении утренней сводки: {e}")


@router.message(Command("system_health", "systemhealth", "system-health", "health"))
async def system_health_cmd_handler(message: Message, runner: AgentRunner) -> None:
    """Прямой запуск скилла диагностики системы."""
    session_id = str(message.from_user.id if message.from_user else message.chat.id)
    async with keep_typing(message.bot, message.chat.id):
        try:
            result = await runner.run(
                session_id=session_id,
                user_prompt="Выполни диагностику системы по скиллу system-health: проверь ресурсы, диск, ОС и доступность Ollama API.",
            )
            chunks = split_message(result.content, max_chunk_size=4000)
            for chunk in chunks:
                try:
                    await message.answer(chunk, parse_mode="Markdown")
                except Exception:
                    await message.answer(chunk)
        except Exception as e:
            logger.exception("Error executing system_health command: %s", e)
            await message.answer(f"⚠️ Ошибка при выполнении диагностики системы: {e}")


@router.message(Command("new", "clear", "reset"))
async def new_chat_handler(message: Message, runner: AgentRunner) -> None:
    """Обработчик команды /new: сброс и создание нового контекста диалога."""
    session_id = str(message.from_user.id if message.from_user else message.chat.id)
    if hasattr(runner, "memory_store") and runner.memory_store:
        await runner.memory_store.clear(session_id)
    text = (
        "🧹 **Контекст очищен!**\n\n"
        "Начат новый чистый диалог. Предыдущая история больше не отправляется модели."
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("token_report", "tokenreport", "tokens", "finops"))
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
