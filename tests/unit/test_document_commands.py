"""Unit tests for Telegram /documents and /delete commands."""
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User, Chat
from aiogram.filters import CommandObject

from core.models import DocumentMetadata
from core.ports import BaseRAGStore
from adapters.telegram.handlers import (
    documents_list_handler,
    document_delete_handler,
)


@pytest.fixture
def mock_rag_store():
    return AsyncMock(spec=BaseRAGStore)


@pytest.fixture
def mock_message():
    msg = MagicMock(spec=Message)
    msg.message_id = 1
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = 555
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = 555
    msg.answer = AsyncMock()
    return msg


class TestDocumentCommands:
    @pytest.mark.asyncio
    async def test_documents_empty(self, mock_message, mock_rag_store):
        mock_rag_store.list_documents.return_value = []

        await documents_list_handler(mock_message, rag_store=mock_rag_store)

        mock_rag_store.list_documents.assert_called_once_with("555")
        mock_message.answer.assert_called_once()
        args, _ = mock_message.answer.call_args
        assert "У вас пока нет загруженных документов" in args[0]
        assert ".pdf, .docx, .txt, .md" in args[0]

    @pytest.mark.asyncio
    async def test_documents_with_items(self, mock_message, mock_rag_store):
        doc1 = DocumentMetadata(
            id="3fa85f6412345678",
            user_id="555",
            filename="company_policy.pdf",
            file_type="pdf",
            file_size_bytes=1024 * 1024 + 400 * 1024,
            page_count=12,
            chunk_count=48,
            created_at=datetime(2026, 9, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        mock_rag_store.list_documents.return_value = [doc1]

        await documents_list_handler(mock_message, rag_store=mock_rag_store)

        mock_rag_store.list_documents.assert_called_once_with("555")
        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "company_policy.pdf" in text
        assert "3fa85f64" in text
        assert "Страниц: 12" in text
        assert "Чанков: 48" in text
        assert "/delete" in text

    @pytest.mark.asyncio
    async def test_delete_missing_arg(self, mock_message, mock_rag_store):
        cmd = MagicMock(spec=CommandObject)
        cmd.args = None
        mock_message.text = "/delete"

        await document_delete_handler(mock_message, command=cmd, rag_store=mock_rag_store)

        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "Использование: /delete" in text
        mock_rag_store.delete_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_message, mock_rag_store):
        cmd = MagicMock(spec=CommandObject)
        cmd.args = "company_policy.pdf"
        mock_message.text = "/delete company_policy.pdf"
        mock_rag_store.delete_document.return_value = True

        await document_delete_handler(mock_message, command=cmd, rag_store=mock_rag_store)

        mock_rag_store.delete_document.assert_called_once_with("555", "company_policy.pdf")
        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "успешно удален" in text
        assert "company_policy.pdf" in text

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_message, mock_rag_store):
        cmd = MagicMock(spec=CommandObject)
        cmd.args = "nonexistent.docx"
        mock_message.text = "/delete nonexistent.docx"
        mock_rag_store.delete_document.return_value = False

        await document_delete_handler(mock_message, command=cmd, rag_store=mock_rag_store)

        mock_rag_store.delete_document.assert_called_once_with("555", "nonexistent.docx")
        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "не найден" in text
        assert "nonexistent.docx" in text

    @pytest.mark.asyncio
    async def test_documents_with_special_characters_fallback(self, mock_message, mock_rag_store):
        doc = DocumentMetadata(
            id="abc12345",
            user_id="555",
            filename="konspekt_3_uroka_nemeckogo_jazyka-poliglotsd_pet.pdf",
            file_type="pdf",
            file_size_bytes=50000,
            page_count=7,
            chunk_count=16,
            created_at=datetime(2026, 9, 21, 18, 37, 0, tzinfo=timezone.utc),
        )
        mock_rag_store.list_documents.return_value = [doc]

        call_count = 0
        async def mock_answer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("parse_mode") == "Markdown":
                raise Exception("TelegramBadRequest")
            return MagicMock()

        mock_message.answer.side_effect = mock_answer

        await documents_list_handler(mock_message, rag_store=mock_rag_store)
        assert call_count == 2
