# Tasks: Autonomous RAG Agent Pipeline (`rag-agent-pipeline`)

## Epoch 1: Core Domain, Ports, Parsers, Chunker, FastEmbed & SQLite-Vec Store

- [x] **Task 1.1: Domain Models for RAG (`core/models.py`)**
  - [x] Add `DocumentMetadata` model (id, user_id, filename, file_type, file_size_bytes, page_count, chunk_count, created_at).
  - [x] Add `DocumentChunk` model (id, document_id, user_id, content, page_number, chunk_index, token_count, metadata).
  - [x] Add `SearchResult` model (chunk_id, document_id, filename, content, page_number, score, source).
  - [x] Write unit tests in `tests/unit/test_rag_models.py` verifying model validation, defaults, and immutability.
  - [x] Run `pytest tests/unit/test_rag_models.py`.

- [x] **Task 1.2: Abstract Port Interfaces (`core/ports/`)**
  - [x] Define `BaseEmbeddingProvider` in `core/ports/embedding.py` (`dimension`, `embed_text`, `embed_batch`).
  - [x] Define `BaseRAGStore` in `core/ports/rag_store.py` (`add_document`, `search`, `list_documents`, `get_document`, `delete_document`).
  - [x] Update `core/ports/__init__.py` to export new ports.
  - [x] Write unit tests in `tests/unit/test_rag_ports.py` checking interface compliance with mock implementations.
  - [x] Run `pytest tests/unit/test_rag_ports.py`.

- [x] **Task 1.3: Document Parsers & Page-Aware Chunker (`adapters/rag/parsers.py`)**
  - [x] Implement format extractors for `.txt`, `.md`, `.docx`, and `.pdf` returning page-annotated tuples `[(page_num, text), ...]`.
  - [x] Implement `RecursiveCharacterChunker` preserving `page_number`, `chunk_index`, and overlap.
  - [x] Write unit tests in `tests/unit/test_rag_parsers.py` with synthetic multi-page PDF, DOCX, MD, and TXT fixtures.
  - [x] Run `pytest tests/unit/test_rag_parsers.py`.

- [x] **Task 1.4: FastEmbed Local Embedding Provider (`adapters/embedding/fastembed_provider.py`)**
  - [x] Implement `FastEmbedProvider(BaseEmbeddingProvider)` using `fastembed.TextEmbedding("intfloat/multilingual-e5-small")`.
  - [x] Implement automatic prefixing (`passage: ` for chunks, `query: ` for queries).
  - [x] Write unit tests in `tests/unit/test_fastembed_provider.py` (including dimension check = 384, prefix handling, and mocked ONNX inference).
  - [x] Run `pytest tests/unit/test_fastembed_provider.py`.

- [x] **Task 1.5: SQLite + sqlite-vec Vector Store (`adapters/rag/sqlite_vec_store.py`)**
  - [x] Implement `SqliteVecStore(BaseRAGStore)` with `documents`, `document_chunks`, `vec_chunks` (vec0), and `chunks_fts` (FTS5) tables.
  - [x] Enforce foreign key constraints with `ON DELETE CASCADE`.
  - [x] Enforce strict multi-tenant isolation via SQL JOINs filtering on `documents.user_id = ?`.
  - [x] Write unit tests in `tests/unit/test_sqlite_vec_store.py`:
    - CRUD operations (add document, list, get, delete).
    - Verification that User B queries NEVER return User A's chunks.
    - Verification that deleting User A's document cascades to chunks, FTS, and vec0 rows.
  - [x] Run `pytest tests/unit/test_sqlite_vec_store.py`.

---

## Epoch 2: ReAct Tool `search_documents`, Session Context & Anti-Hallucination

- [x] **Task 2.1: Domain ReAct Tool `search_documents` (`core/tools/rag.py`)**
  - [x] Implement `create_rag_search_tool(rag_store, embedding_provider) -> Tool`.
  - [x] Define JSON Schema with required `query` parameter.
  - [x] Format retrieved snippets with filename, page numbers, and similarity scores.
  - [x] Return clean fallback message when no matching documents are found.
  - [x] Write unit tests in `tests/unit/test_rag_tool.py`.
  - [x] Run `pytest tests/unit/test_rag_tool.py`.

- [x] **Task 2.2: Session Context Passing in `ToolRegistry` & `AgentRunner`**
  - [x] Update `ToolRegistry.execute` to accept `session_id: Optional[str] = None`.
  - [x] Update `AgentRunner.run` to forward `session_id` into `self.tool_registry.execute`.
  - [x] Ensure `search_documents` securely extracts `user_id` from the trusted session context rather than LLM input.
  - [x] Write unit tests in `tests/unit/test_rag_session_context.py` asserting user context propagation and isolation.
  - [x] Run `pytest tests/unit/test_rag_session_context.py`.

- [x] **Task 2.3: Anti-Hallucination Prompt Directives (`core/runner.py`)**
  - [x] Update `_build_system_prompt_with_tools` with strict RAG instructions:
    - Invoke `search_documents` for user documents or private files.
    - Explicitly cite `[filename, p. X]` for all retrieved facts.
    - Forbid hallucination when search returns no answers.
  - [x] Write unit tests in `tests/unit/test_anti_hallucination_prompt.py`.
  - [x] Run `pytest tests/unit/test_anti_hallucination_prompt.py`.

---

## Epoch 3: Telegram Document Upload & Management Handlers

- [x] **Task 3.1: Document Upload Handler with Interactive Progress (`adapters/telegram/document_handler.py`)**
  - [x] Create router filtering on `F.document`.
  - [x] Validate file extension (`.txt`, `.md`, `.docx`, `.pdf`) and size limit (<= 20 MB).
  - [x] Implement interactive message progression:
    - 📥 Downloading and reading file.
    - ✂️ Splitting into chunks with page numbers.
    - 🧠 Generating local ONNX embeddings.
    - 💾 Indexing in vector store.
    - ✅ Completion summary (filename, chunks, pages).
  - [x] Write unit tests in `tests/unit/test_document_handler.py` with mocked Telegram bot and message objects.
  - [x] Run `pytest tests/unit/test_document_handler.py`.

- [x] **Task 3.2: Telegram Commands `/documents` and `/delete <id>` (`adapters/telegram/handlers.py`)**
  - [x] Implement `/documents` command listing active user's indexed documents with ID, size, chunk count, and date.
  - [x] Implement `/delete <doc_id>` command purging specified document with user feedback.
  - [x] Write unit tests in `tests/unit/test_document_commands.py`.
  - [x] Run `pytest tests/unit/test_document_commands.py`.

- [x] **Task 3.3: Composition Root Wiring (`adapters/telegram/bot.py` & `main.py`)**
  - [x] Instantiate `FastEmbedProvider` and `SqliteVecStore` in `main.py`.
  - [x] Register `search_documents` tool in `ToolRegistry`.
  - [x] Register document router in Telegram bot dispatcher.
  - [x] Run `pytest tests/unit/test_telegram_handlers.py`.

---

## Epoch 4: Bonus Features & Evaluation Harness

- [x] **Task 4.1: Hybrid Search with Reciprocal Rank Fusion (`adapters/rag/sqlite_vec_store.py`)**
  - [x] Implement FTS5 keyword retrieval in `SqliteVecStore`.
  - [x] Implement Reciprocal Rank Fusion algorithm: $RRF(d) = \sum \frac{1}{60 + rank(d)}$.
  - [x] Combine dense vector results and sparse BM25 results into a unified ranked list.
  - [x] Write unit tests in `tests/unit/test_hybrid_search_rrf.py` checking RRF score calculations and tie-breaking.
  - [x] Run `pytest tests/unit/test_hybrid_search_rrf.py`.

- [x] **Task 4.2: Page Number Citations Verification**
  - [x] Verify page numbers are correctly propagated from parsers to chunks and into `SearchResult`.
  - [x] Verify agent prompts include source file and page numbers.
  - [x] Write unit tests in `tests/unit/test_page_citations.py`.
  - [x] Run `pytest tests/unit/test_page_citations.py`.

- [x] **Task 4.3: Golden Evaluation Dataset (`evaluation/rag_dataset.json`)**
  - [x] Create `evaluation/rag_dataset.json` with >= 5 diverse multilingual test questions and ground-truth answers.
  - [x] Include target document references, page numbers, and key excerpts.

- [x] **Task 4.4: Automated RAG Evaluation Script (`scripts/evaluate_rag.py`)**
  - [x] Implement evaluation runner script measuring Hit Rate @ K and Mean Reciprocal Rank (MRR).
  - [x] Verify answer faithfulness against ground truth.
  - [x] Execute `python scripts/evaluate_rag.py` and document results in `evaluation/EVALUATION_REPORT.md`.

---

## Epoch 5: E2E Integration Test, Zero Documentation Drift & Docker Container

- [x] **Task 5.1: End-to-End Integration Test (`tests/integration/test_rag_e2e.py`)**
  - [x] Ingest sample document (`sample_policy.pdf` or `.md`).
  - [x] Execute question query via `AgentRunner`.
  - [x] Assert that `search_documents` is invoked.
  - [x] Assert response includes correct answer and `[filename, p. X]` citation.
  - [x] Delete document and verify subsequent queries confirm absence.
  - [x] Run `pytest tests/integration/test_rag_e2e.py`.

- [x] **Task 5.2: Dependency & Dockerfile Updates**
  - [x] Add `fastembed>=0.3.0`, `sqlite-vec>=0.1.0`, `pypdf>=4.0`, and `python-docx>=1.1.0` to `requirements.txt`.
  - [x] Update `Dockerfile` to ensure ONNX runtime and SQLite extension dependencies build cleanly.
  - [x] Verify Docker container builds and starts cleanly.

- [x] **Task 5.3: Zero Documentation Drift (`README.md`)**
  - [x] Update `README.md` with:
    - RAG Architecture diagram and component descriptions.
    - Supported document formats and chunking specifications.
    - Telegram commands `/documents`, `/delete <id>`, and file drag-and-drop usage.
    - Configuration environment variables (`RAG_DB_PATH`, `EMBEDDING_MODEL`).
    - Evaluation metrics and benchmark instructions.
  - [x] Run full test suite: `pytest` (assert 100% pass rate).
  - [x] Mark all completed checkboxes across Epochs in `tasks.md`.
