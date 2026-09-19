from typing import Collection, Optional
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from core.runner import AgentRunner
from core.ports import BaseRAGStore, BaseEmbeddingProvider
from adapters.telegram.handlers import router
from adapters.telegram.middlewares import AuthMiddleware
from adapters.telegram.document_handler import create_document_router


def create_bot(token: str) -> Bot:
    """Создает и конфигурирует инстанс Bot aiogram."""
    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )


def create_dispatcher(
    runner: AgentRunner,
    allowed_user_ids: Optional[Collection[int]] = None,
    rag_store: Optional[BaseRAGStore] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
) -> Dispatcher:
    """Создает Dispatcher, подключает роутеры, middleware и внедряет зависимости."""
    dp = Dispatcher()
    if allowed_user_ids:
        dp.message.middleware(AuthMiddleware(allowed_user_ids=allowed_user_ids))
    if rag_store and embedding_provider:
        dp.include_router(create_document_router(rag_store=rag_store, embedding_provider=embedding_provider))
    dp.include_router(router)
    dp["runner"] = runner
    dp["rag_store"] = rag_store
    dp["embedding_provider"] = embedding_provider
    return dp
