# Specification: Autonomous RAG Agent Pipeline (`rag_spec.md`)

## 1. Overview

This document provides the formal technical specification for the RAG feature in the autonomous agent, specifying domain models, port interfaces, storage adapters, parsing and chunking components, embedding provider, ReAct tool integration, Telegram upload handling, hybrid search with Reciprocal Rank Fusion (RRF), and the evaluation harness.

---

## 2. Core Domain Layer (`core/`)

### 2.1. Domain Data Models (`core/models.py`)
Extend `core/models.py` with the following immutable Pydantic models:

```python
from datetime import datetime
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata for an ingested user document."""
    id: str = Field(..., description="Unique document UUID")
    user_id: str = Field(..., description="Owner's Telegram user/chat ID")
    filename: str = Field(..., description="Original filename with extension")
    file_type: str = Field(..., description="MIME type or file extension (txt, md, docx, pdf)")
    file_size_bytes: int = Field(..., ge=0, description="Size in bytes")
    page_count: int = Field(default=1, ge=1, description="Total number of pages")
    chunk_count: int = Field(default=0, ge=0, description="Total number of chunks produced")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of ingestion")

class DocumentChunk(BaseModel):
    """An individual text segment of a document."""
    id: str = Field(..., description="Unique chunk UUID")
    document_id: str = Field(..., description="Foreign key to DocumentMetadata.id")
    user_id: str = Field(..., description="Owner's Telegram user ID for isolation")
    content: str = Field(..., min_length=1, description="Chunk text content")
    page_number: Optional[int] = Field(default=None, description="Source page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within document")
    token_count: Optional[int] = Field(default=None, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context/headers")

class SearchResult(BaseModel):
    """Ranked search result returned to the agent."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    page_number: Optional[int] = None
    score: float = Field(..., description="Relevance score (cosine similarity or RRF score)")
    source: Literal["vector", "fts", "hybrid"] = "vector"
```

### 2.2. Port: Embedding Provider (`core/ports/embedding.py`)
Define the abstract interface for generating dense text vector representations:

```python
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingProvider(ABC):
    """Abstract port for text embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector embedding dimension (e.g. 384 for multilingual-e5-small)."""
        pass

    @abstractmethod
    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embed a single text string.
        If is_query is True, applies query-specific prompt/prefix if required by model.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Embed a batch of text strings efficiently.
        """
        pass
```

### 2.3. Port: RAG Store (`core/ports/rag_store.py`)
Define the abstract storage contract for documents, chunks, and hybrid retrieval:

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from core.models import DocumentMetadata, DocumentChunk, SearchResult

class BaseRAGStore(ABC):
    """Abstract port for vector and document persistence."""

    @abstractmethod
    async def add_document(
        self,
        metadata: DocumentMetadata,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Atomically persist document record, text chunks, and their vector embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 4,
        score_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """
        Search documents strictly scoped to user_id.
        Supports dense vector similarity and optional hybrid FTS5 search.
        """
        pass

    @abstractmethod
    async def list_documents(self, user_id: str) -> List[DocumentMetadata]:
        """Return all documents owned by user_id, sorted by created_at DESC."""
        pass

    @abstractmethod
    async def get_document(self, user_id: str, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve single document metadata if owned by user_id."""
        pass

    @abstractmethod
    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Permanently delete document, all associated chunks, and vector index rows.
        Returns True if deleted, False if document not found or access denied.
        """
        pass
```

---

## 3. Driven Adapters (`adapters/`)

### 3.1. Document Parsers & Chunker (`adapters/rag/parsers.py`)
Encapsulates extraction of text and pagination across file formats:

1. **Format Parsers**:
   - `parse_txt(content: bytes) -> List[Tuple[int, str]]`: Decodes UTF-8/CP1251 text; returns `[(1, text)]`.
   - `parse_md(content: bytes) -> List[Tuple[int, str]]`: Cleans Markdown syntax; returns `[(1, text)]`.
   - `parse_docx(content: bytes) -> List[Tuple[int, str]]`: Reads paragraphs via `python-docx`; splits on explicit page breaks or returns `[(1, text)]`.
   - `parse_pdf(content: bytes) -> List[Tuple[int, str]]`: Iterates pages using `pypdf.PdfReader`; returns `[(page_num, page_text), ...]`.

2. **Recursive Character Chunker**:
   ```python
   class RecursiveCharacterChunker:
       def __init__(
           self,
           chunk_size: int = 500,
           chunk_overlap: int = 80,
           separators: Optional[List[str]] = None,
       ):
           self.chunk_size = chunk_size
           self.chunk_overlap = chunk_overlap
           self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

       def split_pages(
           self,
           pages: List[Tuple[int, str]],
           document_id: str,
           user_id: str,
       ) -> List[DocumentChunk]:
           """Splits page-annotated text, preserving page_number and continuity."""
   ```

### 3.2. Local Embedding Provider (`adapters/embedding/fastembed_provider.py`)
Implements `BaseEmbeddingProvider` using `fastembed.TextEmbedding`:
- Model: `intfloat/multilingual-e5-small`.
- Vector dimension: 384.
- Prefix handling:
  - Document chunks: Prepend `"passage: "`
  - Search queries (`is_query=True`): Prepend `"query: "`
- Performance: Generates embeddings in batches using local CPU ONNX runtime.

### 3.3. SQLite Vector Store (`adapters/rag/sqlite_vec_store.py`)
Implements `BaseRAGStore` using standard SQLite with the `sqlite-vec` C-extension and `FTS5`:

1. **Database Path**: `data/rag_store.db` (configurable via `RAG_DB_PATH`).
2. **Relational & Vector Schema**:
   ```sql
   PRAGMA foreign_keys = ON;
   PRAGMA journal_mode = WAL;

   -- 1. Document table
   CREATE TABLE IF NOT EXISTS documents (
       id TEXT PRIMARY KEY,
       user_id TEXT NOT NULL,
       filename TEXT NOT NULL,
       file_type TEXT NOT NULL,
       file_size_bytes INTEGER NOT NULL,
       page_count INTEGER NOT NULL DEFAULT 1,
       chunk_count INTEGER NOT NULL DEFAULT 0,
       created_at DATETIME DEFAULT CURRENT_TIMESTAMP
   );
   CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

   -- 2. Chunks table
   CREATE TABLE IF NOT EXISTS document_chunks (
       id TEXT PRIMARY KEY,
       document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
       user_id TEXT NOT NULL,
       content TEXT NOT NULL,
       page_number INTEGER,
       chunk_index INTEGER NOT NULL
   );
   CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id);
   CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);

   -- 3. Vector table (sqlite-vec)
   CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
       embedding float[384] distance_metric=cosine
   );

   -- 4. Full-Text Search table (FTS5)
   CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
       chunk_id UNINDEXED,
       user_id UNINDEXED,
       content
   );
   ```

3. **Storage Mapping**:
   - `document_chunks.rowid` maps 1:1 to `vec_chunks.rowid`.
   - When inserting chunks:
     - Insert chunk into `document_chunks`.
     - Insert embedding into `vec_chunks(rowid, embedding) VALUES (chunk_rowid, ?)`.
     - Insert text into `chunks_fts(chunk_id, user_id, content) VALUES (?, ?, ?)`.
   - When deleting document:
     - Deleting from `documents WHERE id = ? AND user_id = ?` cascades to `document_chunks`.
     - Triggers or explicit delete statements purge corresponding `vec_chunks` and `chunks_fts` rows.

4. **Strict User Isolation**:
   ```sql
   SELECT
       c.id AS chunk_id,
       c.document_id,
       d.filename,
       c.content,
       c.page_number,
       v.distance
   FROM vec_chunks v
   JOIN document_chunks c ON c.rowid = v.rowid
   JOIN documents d ON d.id = c.document_id
   WHERE d.user_id = :user_id
     AND v.embedding MATCH :query_vector
     AND k = :top_k
   ORDER BY v.distance ASC;
   ```

5. **Bonus: Hybrid Search & Reciprocal Rank Fusion (RRF)**:
   - When `query_text` is provided:
     - Step 1: Fetch top $20$ dense results from `vec_chunks` (filtered by `user_id`).
     - Step 2: Fetch top $20$ sparse results from `chunks_fts` (filtered by `user_id`).
     - Step 3: Compute RRF score for each unique chunk $d$:
       $$RRF(d) = \frac{1}{60 + rank_{vec}(d)} + \frac{1}{60 + rank_{fts}(d)}$$
     - Step 4: Return top $K$ chunks sorted by $RRF(d)$ descending.

---

## 4. Domain Tooling & Agent Harness (`core/tools/`)

### 4.1. `search_documents` Tool (`core/tools/rag.py`)
```python
def create_rag_search_tool(rag_store: BaseRAGStore, embedding_provider: BaseEmbeddingProvider) -> Tool:
    """
    Creates the 'search_documents' tool.
    Extracts user_id dynamically from the trusted execution context.
    """
```
- Parameters schema:
  ```json
  {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Semantic query to search across the user's private documents."
      }
    },
    "required": ["query"]
  }
  ```
- Result formatting:
  ```
  Found 2 relevant document excerpts:

  [1] File: contract.pdf (Page 4, Score: 0.89)
  "... The termination notice shall be delivered in writing at least 30 days prior..."

  [2] File: contract.pdf (Page 5, Score: 0.76)
  "... In the event of force majeure, either party may terminate immediately..."
  ```
- If no results: `"No relevant information found in your documents for query: '{query}'."`

### 4.2. Context Injection in `core/runner.py`
- Pass `session_id` (representing authenticated `user_id`) to `ToolRegistry.execute(..., session_id=session_id)`.
- Tools requiring user identity retrieve `user_id` directly from `session_id`, making cross-user spoofing structurally impossible.

### 4.3. Anti-Hallucination System Directives
Update `AgentRunner._build_system_prompt_with_tools()` with explicit anti-hallucination rules:
1. `When the user asks about personal documents, contracts, notes, or uploaded files, invoke 'search_documents'.`
2. `Always explicitly cite the document name and page number when using retrieved facts, e.g. [contract.pdf, p. 4].`
3. `If the search result does not contain the answer, reply honestly that the information was not found in the uploaded documents. NEVER invent or extrapolate unverified facts.`

---

## 5. Driving Adapter: Telegram UX (`adapters/telegram/`)

### 5.1. Document Upload Handler (`adapters/telegram/document_handler.py`)
- Listens on `F.document` in an isolated router.
- Validation:
  - Allowed extensions: `.txt`, `.md`, `.docx`, `.pdf`.
  - Maximum file size: $20\text{ MB}$.
- Interactive Editing Pipeline:
  1. Send message: `⏳ Обработка документа: filename...`
  2. Edit message: `📥 Скачивание и чтение...`
  3. Edit message: `✂️ Нарезка на фрагменты...`
  4. Edit message: `🧠 Векторизация через ONNX (multilingual-e5)...`
  5. Edit message: `💾 Сохранение в векторное хранилище...`
  6. Final edit: `✅ Документ 'filename' успешно проиндексирован!\n📊 Фрагментов: X | Страниц: Y | Размер: Z KB.`

### 5.2. Document Commands (`adapters/telegram/handlers.py`)
- `/documents`:
  - Queries `rag_store.list_documents(user_id)`.
  - Outputs formatted list:
    ```
    📚 Ваши документы:
    1. 📄 annual_report.pdf
       • ID: `3fa85f64`
       • Страниц: 12 | Чанков: 48 | 1.4 MB
       • Дата: 2026-09-19 14:00

    Для удаления документа отправьте: /delete <ID>
    ```
- `/delete <document_id>`:
  - Calls `rag_store.delete_document(user_id, document_id)`.
  - Confirms deletion or informs if document not found.

---

## 6. Evaluation Harness (`evaluation/` & `scripts/`)

### 6.1. Evaluation Dataset (`evaluation/rag_dataset.json`)
Structure containing $\ge 5$ representative queries:
```json
[
  {
    "id": "rag-q1",
    "document": "hr_policy.pdf",
    "question": "Сколько дней отпуска положено сотруднику после года работы?",
    "ground_truth_answer": "Сотруднику положено 28 календарных дней отпуска.",
    "ground_truth_page": 3,
    "ground_truth_chunk_contains": "28 календарных дней"
  }
]
```

### 6.2. Evaluation Script (`scripts/evaluate_rag.py`)
Executes automated quality assessment:
- **Hit Rate @ K**: Proportion of queries where ground truth chunk appears in top $K$ results.
- **Mean Reciprocal Rank (MRR)**: $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$.
- **Answer Faithfulness**: Evaluates whether the agent's synthesized response cites the correct document, includes the page number, and refrains from hallucination.
