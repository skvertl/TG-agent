"""RAG ReAct search_documents tool."""
import asyncio
import concurrent.futures
from typing import Optional, Any

from core.ports.rag_store import BaseRAGStore
from core.ports.embedding import BaseEmbeddingProvider
from core.tools.base import Tool


def _run_async_safely(coro):
    """
    Executes a coroutine safely, whether called from inside an active event loop
    or from a synchronous context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def create_rag_search_tool(
    rag_store: BaseRAGStore,
    embedding_provider: BaseEmbeddingProvider,
) -> Tool:
    """
    Creates the 'search_documents' ReAct tool.
    Securely isolates search queries using the caller's session_id or user_id.
    """

    def _execute_search(
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        effective_user_id = user_id or session_id
        if not effective_user_id:
            return "Error: user context is required to access documents."

        if not query or not query.strip():
            return "Error: query cannot be empty."

        clean_query = query.strip()

        async def _perform_search():
            vector = await embedding_provider.embed_text(clean_query, is_query=True)
            return await rag_store.search(
                user_id=effective_user_id,
                query_embedding=vector,
                query_text=clean_query,
                top_k=4,
                score_threshold=0.5,
            )

        results = _run_async_safely(_perform_search())

        if not results:
            return f"No relevant information found in your documents for query: '{clean_query}'."

        snippets = []
        for i, res in enumerate(results, start=1):
            page_info = f"Page {res.page_number}, " if res.page_number is not None else ""
            snippets.append(
                f"[{i}] File: {res.filename} ({page_info}Score: {res.score:.2f})\n\"{res.content}\""
            )

        return f"Found {len(results)} relevant document excerpts:\n\n" + "\n\n".join(snippets)

    return Tool(
        name="search_documents",
        description=(
            "Search across the user's private uploaded documents (PDF, DOCX, TXT, MD) "
            "using semantic vector search. Returns relevant text excerpts with source filenames and page numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic query to search across the user's private documents.",
                }
            },
            "required": ["query"],
        },
        func=_execute_search,
    )
