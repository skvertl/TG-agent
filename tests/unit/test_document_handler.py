"""Unit tests for Telegram document upload handler with interactive progress."""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, Document, User, Chat

from core.ports import BaseRAGStore, BaseEmbeddingProvider
from adapters.telegram.document_handler import (
    create_document_router,
    handle_document_upload,
)


@pytest.fixture
def mock_dependencies():
    rag_store = AsyncMock(spec=BaseRAGStore)
    embedding_provider = AsyncMock(spec=BaseEmbeddingProvider)
    embedding_provider.embed_batch.return_value = [[0.1] * 384]
    return rag_store, embedding_provider


@pytest.fixture
def mock_doc_message():
    msg = MagicMock(spec=Message)
    msg.message_id = 100
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = 12345
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = 12345

    doc = MagicMock(spec=Document)
    doc.file_name = "test_doc.txt"
    doc.file_size = 1024
    msg.document = doc

    msg.bot = MagicMock()
    # bot.download writes bytes into the provided destination (BytesIO)
    async def fake_download(file, destination):
        destination.write(b"This is the content of the test document.")
        destination.seek(0)
    msg.bot.download = AsyncMock(side_effect=fake_download)

    # Status message returned by message.answer
    status_msg = MagicMock(spec=Message)
    status_msg.message_id = 101
    status_msg.edit_text = AsyncMock()
    msg.answer = AsyncMock(return_value=status_msg)

    return msg, status_msg


class TestDocumentHandler:
    def test_router_creation(self, mock_dependencies):
        rag_store, embedding_provider = mock_dependencies
        router = create_document_router(rag_store, embedding_provider)
        assert router is not None

    @pytest.mark.asyncio
    async def test_unsupported_format(self, mock_dependencies, mock_doc_message):
        rag_store, embedding_provider = mock_dependencies
        msg, status_msg = mock_doc_message
        msg.document.file_name = "malicious.exe"

        await handle_document_upload(msg, rag_store, embedding_provider)

        msg.answer.assert_called_once()
        args, _ = msg.answer.call_args
        assert "Неподдерживаемый формат" in args[0]
        rag_store.add_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_too_large(self, mock_dependencies, mock_doc_message):
        rag_store, embedding_provider = mock_dependencies
        msg, status_msg = mock_doc_message
        msg.document.file_size = 25 * 1024 * 1024  # 25 MB

        await handle_document_upload(msg, rag_store, embedding_provider)

        msg.answer.assert_called_once()
        args, _ = msg.answer.call_args
        assert "20 MB" in args[0] or "слишком большой" in args[0].lower()
        rag_store.add_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_document(self, mock_dependencies, mock_doc_message):
        rag_store, embedding_provider = mock_dependencies
        msg, status_msg = mock_doc_message

        async def empty_download(file, destination):
            destination.write(b"   ")
            destination.seek(0)
        msg.bot.download = AsyncMock(side_effect=empty_download)

        await handle_document_upload(msg, rag_store, embedding_provider)

        status_msg.edit_text.assert_called()
        last_edit = status_msg.edit_text.call_args[0][0]
        assert "пуст" in last_edit.lower() or "не содержит текста" in last_edit.lower()
        rag_store.add_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_indexing_pipeline(self, mock_dependencies, mock_doc_message):
        rag_store, embedding_provider = mock_dependencies
        msg, status_msg = mock_doc_message
        msg.document.file_name = "policy.pdf"
        msg.document.file_size = 2048

        # Mock PDF download
        async def pdf_download(file, destination):
            destination.write(b"Important security policy content for all employees.")
            destination.seek(0)
        msg.bot.download = AsyncMock(side_effect=pdf_download)

        # Mock extract_text_with_pages inside test
        from unittest.mock import patch
        with patch(
            "adapters.telegram.document_handler.extract_text_with_pages",
            return_value=[(1, "Important security policy content for all employees.")],
        ):
            await handle_document_upload(msg, rag_store, embedding_provider)

        # Check initial answer
        msg.answer.assert_called_once()
        init_text = msg.answer.call_args[0][0]
        assert "policy.pdf" in init_text

        # Check that edit_text was called through progress steps
        assert status_msg.edit_text.call_count >= 4

        # Final edit should be success summary
        final_call_text = status_msg.edit_text.call_args[0][0]
        assert "успешно проиндексирован" in final_call_text
        assert "policy.pdf" in final_call_text
        assert "Фрагментов:" in final_call_text

        # Verify rag_store.add_document was called
        rag_store.add_document.assert_called_once()
        call_meta = rag_store.add_document.call_args[0][0]
        call_chunks = rag_store.add_document.call_args[0][1]
        call_embs = rag_store.add_document.call_args[0][2]

        assert call_meta.filename == "policy.pdf"
        assert call_meta.user_id == "12345"
        assert len(call_chunks) >= 1
        assert len(call_embs) == len(call_chunks)

    @pytest.mark.asyncio
    async def test_error_handling_edits_status_message(self, mock_dependencies, mock_doc_message):
        rag_store, embedding_provider = mock_dependencies
        msg, status_msg = mock_doc_message

        # Force download failure
        msg.bot.download = AsyncMock(side_effect=RuntimeError("Connection reset by peer"))

        await handle_document_upload(msg, rag_store, embedding_provider)

        status_msg.edit_text.assert_called()
        last_edit = status_msg.edit_text.call_args[0][0]
        assert "Ошибка" in last_edit
        assert "Connection reset by peer" in last_edit
