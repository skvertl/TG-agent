"""Telegram document upload handler with interactive progress editing."""
import io
import logging
import uuid
from pathlib import Path

from aiogram import Router, F
from aiogram.types import Message

from core.models import DocumentMetadata
from core.ports.rag_store import BaseRAGStore
from core.ports.embedding import BaseEmbeddingProvider
from adapters.rag.parsers import extract_text_with_pages, RecursiveCharacterChunker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


async def handle_document_upload(
    message: Message,
    rag_store: BaseRAGStore,
    embedding_provider: BaseEmbeddingProvider,
) -> None:
    """
    Processes an uploaded document, providing step-by-step progress updates,
    extracting text with pagination, computing embeddings, and storing in RAG store.
    """
    if not message.document:
        return

    doc = message.document
    filename = doc.file_name or "document.txt"
    file_ext = Path(filename).suffix.lower()
    user_id = str(message.from_user.id if message.from_user else message.chat.id)

    logger.info(
        "Document upload received: filename='%s', size=%s bytes, user_id=%s",
        filename,
        doc.file_size,
        user_id,
    )

    # 1. Validate file extension
    if file_ext not in SUPPORTED_EXTENSIONS:
        await message.answer(
            f"❌ Неподдерживаемый формат файла '{file_ext}'.\n"
            f"Поддерживаются: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            parse_mode=None,
        )
        return

    # 2. Validate file size (Telegram Bot API download limit is 20 MB)
    if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
        size_mb = doc.file_size / (1024 * 1024)
        await message.answer(
            f"❌ Файл '{filename}' слишком большой ({size_mb:.1f} MB).\n"
            "Максимально допустимый размер для скачивания Telegram-ботом: 20 MB (ограничение Telegram Bot API).\n\n"
            "💡 Пожалуйста, сожмите файл, разделите его на части или сохраните в текстовом формате (.txt/.md).",
            parse_mode=None,
        )
        return

    # 3. Interactive Progress Pipeline
    try:
        status_msg = await message.answer(
            f"📄 Документ получен: {filename}\n⏳ Начинаю обработку...",
            parse_mode=None,
        )
    except Exception as e:
        logger.error("Failed to send initial status message: %s", e)
        status_msg = await message.answer(f"📄 Документ получен: {filename}")

    try:
        # Step 1: Download
        await status_msg.edit_text("📥 Скачивание и чтение файла...", parse_mode=None)
        buffer = io.BytesIO()
        await message.bot.download(doc, destination=buffer)
        content_bytes = buffer.getvalue()

        # Step 2: Extraction & Chunking
        await status_msg.edit_text("✂️ Извлечение текста и нарезка на фрагменты...", parse_mode=None)
        pages = extract_text_with_pages(content_bytes, filename)

        doc_id = uuid.uuid4().hex
        chunker = RecursiveCharacterChunker(chunk_size=600, overlap=100)
        chunks = chunker.chunk_document(user_id=user_id, pages=pages, document_id=doc_id)

        if not chunks or all(not c.content.strip() for c in chunks):
            await status_msg.edit_text(f"❌ Документ '{filename}' пуст или не содержит текста.", parse_mode=None)
            return

        # Step 3: Vectorization (Embedding)
        await status_msg.edit_text(
            f"🧠 Векторизация {len(chunks)} фрагментов ({len(pages)} стр.) через ONNX...",
            parse_mode=None,
        )
        texts_to_embed = [c.content for c in chunks]
        embeddings = await embedding_provider.embed_batch(texts_to_embed, is_query=False)

        # Step 4: Persist in RAG Store
        await status_msg.edit_text("💾 Сохранение в векторное хранилище...", parse_mode=None)
        metadata = DocumentMetadata(
            id=doc_id,
            user_id=user_id,
            filename=filename,
            file_type=file_ext.lstrip("."),
            file_size_bytes=doc.file_size or len(content_bytes),
            page_count=len(pages),
            chunk_count=len(chunks),
        )
        await rag_store.add_document(metadata, chunks, embeddings)

        # Step 5: Completion Summary
        size_kb = (doc.file_size or len(content_bytes)) / 1024.0
        success_text = (
            f"✅ Документ {filename} успешно проиндексирован!\n"
            f"📊 Фрагментов: {len(chunks)} | Страниц: {len(pages)} | Размер: {size_kb:.1f} KB\n\n"
            "Теперь вы можете задавать вопросы по документу."
        )
        try:
            await status_msg.edit_text(success_text, parse_mode=None)
        except Exception:
            await message.answer(success_text, parse_mode=None)

        logger.info(
            "Document '%s' (user: %s, chunks: %d, pages: %d) successfully indexed.",
            filename,
            user_id,
            len(chunks),
            len(pages),
        )

    except Exception as exc:
        logger.exception("Error processing document '%s': %s", filename, exc)
        err_text = f"❌ Ошибка при обработке документа '{filename}': {str(exc)}"
        try:
            await status_msg.edit_text(err_text, parse_mode=None)
        except Exception:
            await message.answer(err_text, parse_mode=None)


def create_document_router(
    rag_store: BaseRAGStore,
    embedding_provider: BaseEmbeddingProvider,
) -> Router:
    """Factory creating an aiogram router bound to document upload events."""
    router = Router(name="document_router")

    @router.message(F.document)
    async def document_upload_handler(message: Message) -> None:
        await handle_document_upload(
            message=message,
            rag_store=rag_store,
            embedding_provider=embedding_provider,
        )

    return router
