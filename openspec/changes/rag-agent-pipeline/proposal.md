# Proposal: Autonomous RAG Agent Pipeline (Local Vector Store, Ingestion & Hybrid Search)

## 1. Overview & Architectural Intent

This proposal introduces a private, fully offline, local **Retrieval-Augmented Generation (RAG)** pipeline into the autonomous Telegram agent. It transforms the agent from a general assistant into a personalized knowledge expert capable of ingesting personal documents, indexing them with local embeddings, storing them in an embedded SQLite vector store, and autonomously querying them during ReAct reasoning.

### 1.1. Hexagonal Architecture (Ports & Adapters) Alignment
To preserve the modularity, portability, and zero-coupling principles of the codebase:
- **Core Domain (`core/`)**: Remains completely agnostic of specific vector databases, embedding model libraries, and Telegram interfaces. It defines pure abstract ports (`BaseRAGStore`, `BaseEmbeddingProvider`), immutable data models (`DocumentMetadata`, `DocumentChunk`, `SearchResult`), and domain tools (`search_documents`).
- **Driven Adapters (`adapters/rag/`, `adapters/embedding/`)**: Encapsulate external dependencies:
  - `SqliteVecStore`: Interacts with SQLite using the `sqlite-vec` extension and `FTS5`.
  - `FastEmbedProvider`: Wraps local ONNX Runtime inference using `fastembed`.
  - `DocumentParser`: Parses raw file formats into clean page-annotated text.
- **Driving Adapters (`adapters/telegram/`)**: Handle document uploads, interactive progress feedback, and user commands (`/documents`, `/delete`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DRIVING ADAPTERS                                 │
│  - Telegram Document Handler (Upload .pdf, .docx, .md, .txt with progress)  │
│  - Telegram Commands (/documents, /delete <id>)                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN CORE                                    │
│  - AgentRunner (ReAct orchestrator with user_id session context)            │
│  - ToolRegistry & 'search_documents' ReAct Tool                             │
│  - Anti-Hallucination & Citation Directives                                 │
│  - Domain Models: DocumentMetadata, DocumentChunk, SearchResult             │
│  - Abstract Ports: BaseRAGStore, BaseEmbeddingProvider                      │
└───────────────────────┬─────────────────────────────┬───────────────────────┘
                        │                             │
                        ▼                             ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│     DRIVEN ADAPTER: EMBEDDINGS       │  │     DRIVEN ADAPTER: RAG STORE     │
│  - FastEmbedProvider                 │  │  - SqliteVecStore                 │
│  - Model: multilingual-e5-small      │  │    * vec0 cosine vector index     │
│  - Local ONNX (384-dim, CPU-fast)    │  │    * FTS5 keyword index           │
│                                      │  │    * RRF Hybrid Search            │
│                                      │  │    * Strict User Isolation (JOIN) │
└──────────────────────────────────────┘  └───────────────────────────────────┘
```

---

## 2. Core Capabilities

### 2.1. Multi-Format Document Ingestion & Page-Aware Parsing
- Supported formats: Plain Text (`.txt`), Markdown (`.md`), Microsoft Word (`.docx`), and Adobe Portable Document Format (`.pdf`).
- Parsers extract text with explicit **page boundary tracking**:
  - For PDF (`pypdf`): Each page maintains its `page_number` (1-indexed).
  - For DOCX (`python-docx`): Page/section breaks identify logical page segments.
  - For TXT / MD: Normalized as single or section-bounded documents.
- Size limit: 20 MB per file, with MIME type and extension validation to prevent corrupted or malicious uploads.

### 2.2. Recursive Semantic Chunking
- Recursive character splitting with hierarchical separators: `["\n\n", "\n", ". ", " ", ""]`.
- Default chunk size: 500 characters with 80 character overlap to retain semantic context across boundaries.
- **Page Attribution**: Every individual chunk preserves its source `page_number` and `chunk_index`, enabling pinpoint source citations.

### 2.3. Local Lightweight Embeddings (`fastembed`)
- Model: `intfloat/multilingual-e5-small` (384 dimensions).
- Runtime: Fast, quantized ONNX Runtime running entirely on the CPU with zero external API calls or token billing.
- Multilingual: Out-of-the-box native support for Russian, English, and 90+ other languages.
- Query / Passage formatting: Automatically applies model prefixes (`passage: ` for document chunks, `query: ` for search queries) to maximize retrieval accuracy.

### 2.4. Embedded SQLite Vector Store (`sqlite-vec`)
- Single-file storage in `data/rag_store.db` with WAL mode (`PRAGMA journal_mode=WAL;`).
- Native vector indexing using SQLite's official vector extension (`sqlite-vec` / `vec0`).
- Relational schema linking `documents` $\rightarrow$ `document_chunks` $\rightarrow$ `vec_chunks` with `ON DELETE CASCADE`.

### 2.5. Strict Multi-Tenant User Isolation
- Every document and chunk is strictly associated with the user's unique Telegram ID (`user_id`).
- All vector and keyword queries execute strict inner joins against `documents.user_id = ?`.
- No user can access, search, retrieve, or delete documents belonging to another user.

### 2.6. Autonomous ReAct Tool: `search_documents`
- Exposed to the agent in `ToolRegistry` as:
  ```json
  {
    "name": "search_documents",
    "description": "Search personal documents and knowledge base for answers.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query or question."}
      },
      "required": ["query"]
    }
  }
  ```
- The `AgentRunner` passes the active session `user_id` to ensure isolated searches.
- The model autonomously determines when to query personal documents, reads the retrieved passages, and cites sources.

### 2.7. Interactive Telegram UX & Document Management
- Direct file upload handling: users simply drag & drop files into the Telegram chat.
- Real-time interactive progress indication: the bot updates a single message through stages:
  1. `📥 Скачивание и валидация файла...`
  2. `✂️ Нарезка на фрагменты (чанкование)...`
  3. `🧠 Векторизация через ONNX (multilingual-e5)...`
  4. `💾 Сохранение в векторное хранилище...`
  5. `✅ Готово! Документ '<name>' проиндексирован (X фрагментов, Y страниц).`
- Commands:
  - `/documents` — Lists all indexed documents for the current user with chunk count, size, and date.
  - `/delete <doc_id>` — Completely purges a document, its text chunks, and its vector embeddings.

### 2.8. Bonus Feature: Hybrid Search via Reciprocal Rank Fusion (RRF)
- Combines:
  1. **Dense Vector Search**: Cosine distance via `vec0` (semantic meaning).
  2. **Sparse Keyword Search**: BM25 via SQLite `FTS5` (exact terms, acronyms, code identifiers).
- Merges ranked lists using standard Reciprocal Rank Fusion:
  $$RRF(d) = \frac{1}{60 + rank_{vec}(d)} + \frac{1}{60 + rank_{fts}(d)}$$
- Yields significantly higher retrieval recall on technical terms, names, and precise numbers.

### 2.9. Evaluation Dataset & Benchmark Harness
- Golden evaluation dataset: `evaluation/rag_dataset.json` containing $\ge 5$ representative multi-lingual questions across diverse documents.
- Automated evaluation script: `scripts/evaluate_rag.py` computing:
  - Retrieval Hit Rate @ $K$.
  - Mean Reciprocal Rank (MRR).
  - Context Relevance and Answer Faithfulness (Anti-Hallucination verification).

---

## 3. Non-Functional Requirements (NFR)

1. **User Isolation Security**:
   - Zero cross-user data leakage.
   - Isolation verified via unit and integration tests asserting that User B's queries never return User A's chunks.
2. **Anti-Hallucination Guarantee**:
   - System prompt instructions mandate that the agent must cite the source filename and page number (`[filename, p. X]`).
   - If retrieved similarity scores fall below threshold or search returns no results, the agent must explicitly state that the answer is not present in the indexed documents rather than inventing facts.
3. **Performance & Latency**:
   - Local embedding inference on CPU $\le 150\text{ ms}$ for standard queries.
   - Vector similarity search $\le 30\text{ ms}$ for collections up to 10,000 chunks.
4. **Resilience & Graceful Degradation**:
   - Corrupted or password-protected files fail cleanly with friendly user error messages without crashing the bot.
   - Database operations use foreign keys and WAL mode for concurrency and atomic rollbacks.
