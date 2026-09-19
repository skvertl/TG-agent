"""SQLite + sqlite-vec RAG store implementation."""
import json
import math
import os
import re
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Generator, Tuple, Any, Dict

from core.models import DocumentMetadata, DocumentChunk, SearchResult
from core.ports.rag_store import BaseRAGStore


def get_default_rag_db_path() -> str:
    env_path = os.environ.get("SQLITE_RAG_DB_PATH") or os.environ.get("RAG_DB_PATH")
    if env_path:
        return env_path
    local_dir = Path("data")
    local_dir.mkdir(parents=True, exist_ok=True)
    return str(local_dir / "rag_store.db")


def _serialize_vec(vector: List[float]) -> bytes:
    """Serializes a float vector into float32 binary format for sqlite-vec."""
    try:
        import sqlite_vec
        if hasattr(sqlite_vec, "serialize_float32"):
            return sqlite_vec.serialize_float32(vector)
    except Exception:
        pass
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_vec(blob: bytes) -> List[float]:
    """Deserializes float32 binary blob into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors in range [-1, 1]."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_datetime(val: Any) -> datetime:
    """Parses datetime object or ISO string to UTC datetime."""
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _sanitize_fts_query(query: Optional[str]) -> Optional[str]:
    """
    Sanitizes user search query into valid FTS5 MATCH syntax.
    Extracts alphanumeric tokens (supports Latin, Cyrillic, digits).
    Applies prefix wildcard for terms > 2 chars, exact match for short terms.
    Combines terms with 'OR' for ranked BM25 matching.
    """
    if not query:
        return None
    tokens = re.findall(r"\w+", query)
    clean_tokens = [t for t in tokens if t]
    if not clean_tokens:
        return None

    parts = []
    for tok in clean_tokens:
        if len(tok) <= 2:
            parts.append(f'"{tok}"')
        else:
            parts.append(f'"{tok}"*')
    return " OR ".join(parts)


class SqliteVecStore(BaseRAGStore):
    """
    RAG persistence store leveraging SQLite and sqlite-vec (vec0 virtual table).
    Includes FTS5 full-text search index and batched cascading deletions.
    Enforces strict multi-tenant isolation on all operations.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or get_default_rag_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._vec_enabled = False
        self._fts_enabled = False
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        self._load_sqlite_vec_extension(conn)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _load_sqlite_vec_extension(self, conn: sqlite3.Connection) -> None:
        """Loads sqlite_vec native extension if available."""
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass

    def _init_db(self) -> None:
        """Creates documents, chunks, vec0, and FTS5 tables."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")

            # 1. Base documents table
            conn.execute(
                """
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
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);")

            # 2. Document chunks table (stores UUID id and references document_id)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER,
                    metadata TEXT DEFAULT '{}'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);")

            # 3. Vector chunks table (sqlite-vec vec0 with distance_metric=cosine or fallback)
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                        embedding float[384] distance_metric=cosine
                    );
                    """
                )
                self._vec_enabled = True
            except Exception:
                self._vec_enabled = False
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks_vec_fallback (
                        chunk_rowid INTEGER PRIMARY KEY,
                        embedding BLOB NOT NULL
                    );
                    """
                )

            # 4. Full-text search table (FTS5 with chunk_id and user_id unindexed)
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        user_id UNINDEXED,
                        content
                    );
                    """
                )
                self._fts_enabled = True
            except Exception:
                self._fts_enabled = False

    async def add_document(
        self,
        metadata: DocumentMetadata,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Atomically persists document record, text chunks, and their vector embeddings.
        Takes metadata, chunks, and embeddings as explicit parameters (S-1, SPEC-2).
        Does not mutate callers' chunk domain models in-place (S-2).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        # Remove existing document with the same ID or filename for this user
        await self.delete_document(metadata.user_id, metadata.id)
        await self.delete_document(metadata.user_id, metadata.filename)

        with self._get_connection() as conn:
            # 1. Insert document metadata
            created_str = (
                metadata.created_at.isoformat()
                if isinstance(metadata.created_at, datetime)
                else str(metadata.created_at)
            )
            conn.execute(
                """
                INSERT INTO documents (
                    id, user_id, filename, file_type, file_size_bytes, page_count, chunk_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    metadata.id,
                    metadata.user_id,
                    metadata.filename,
                    metadata.file_type,
                    metadata.file_size_bytes,
                    metadata.page_count,
                    len(chunks),
                    created_str,
                ),
            )

            # 2. Insert chunks and embeddings
            for chunk, emb in zip(chunks, embeddings):
                cursor = conn.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, user_id, content, page_number, chunk_index, token_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.id,
                        metadata.id,
                        metadata.user_id,
                        chunk.content,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.token_count,
                        json.dumps(chunk.metadata or {}),
                    ),
                )
                chunk_rowid = cursor.lastrowid

                raw_vec = _serialize_vec(emb)
                if self._vec_enabled:
                    try:
                        conn.execute(
                            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?);",
                            (chunk_rowid, raw_vec),
                        )
                    except Exception:
                        pass
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO document_chunks_vec_fallback (chunk_rowid, embedding) VALUES (?, ?);",
                        (chunk_rowid, raw_vec),
                    )

                if self._fts_enabled:
                    try:
                        conn.execute(
                            "INSERT INTO chunks_fts (chunk_id, user_id, content) VALUES (?, ?, ?);",
                            (chunk.id, metadata.user_id, chunk.content),
                        )
                    except Exception:
                        pass

    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 4,
        score_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """
        Searches documents strictly scoped to user_id.
        Supports dense vector similarity and hybrid FTS5 search with RRF.
        """
        has_query_text = bool(query_text and query_text.strip())

        with self._get_connection() as conn:
            # Case 1: Pure vector search (when query_text is absent or empty)
            if not has_query_text:
                results: List[SearchResult] = []
                if self._vec_enabled and query_embedding:
                    try:
                        raw_query = _serialize_vec(query_embedding)
                        rows = conn.execute(
                            """
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
                            WHERE d.user_id = ?
                              AND v.embedding MATCH ?
                              AND k = ?
                            ORDER BY v.distance ASC;
                            """,
                            (user_id, raw_query, top_k * 2),
                        ).fetchall()

                        for r in rows:
                            dist = float(r["distance"]) if r["distance"] is not None else 0.0
                            # distance_metric=cosine: distance in [0, 2], similarity is 1.0 - dist
                            score = round(max(0.0, 1.0 - dist), 4)
                            if score >= score_threshold:
                                results.append(
                                    SearchResult(
                                        chunk_id=r["chunk_id"],
                                        document_id=r["document_id"],
                                        filename=r["filename"],
                                        content=r["content"],
                                        page_number=r["page_number"],
                                        score=score,
                                        source="vector",
                                    )
                                )
                        return results[:top_k]
                    except Exception:
                        pass

                # Fallback in-memory vector similarity
                if not self._vec_enabled and query_embedding:
                    rows = conn.execute(
                        """
                        SELECT c.id, c.document_id, d.filename, c.content, c.page_number, f.embedding
                        FROM document_chunks c
                        JOIN documents d ON d.id = c.document_id
                        LEFT JOIN document_chunks_vec_fallback f ON f.chunk_rowid = c.rowid
                        WHERE d.user_id = ? AND f.embedding IS NOT NULL;
                        """,
                        (user_id,),
                    ).fetchall()

                    scored: List[Tuple[float, Any]] = []
                    for r in rows:
                        chunk_vec = _deserialize_vec(r["embedding"])
                        sim = _cosine_similarity(query_embedding, chunk_vec)
                        if sim >= score_threshold:
                            scored.append((sim, r))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    for sim, r in scored[:top_k]:
                        results.append(
                            SearchResult(
                                chunk_id=r["id"],
                                document_id=r["document_id"],
                                filename=r["filename"],
                                content=r["content"],
                                page_number=r["page_number"],
                                score=round(sim, 4),
                                source="vector",
                            )
                        )
                return results

            # Case 2: Hybrid Search with Reciprocal Rank Fusion (RRF)
            # Step 1: Fetch dense vector candidates (up to 20) with ranks
            vec_candidates: Dict[str, Tuple[int, Dict[str, Any]]] = {}
            if self._vec_enabled and query_embedding:
                try:
                    raw_query = _serialize_vec(query_embedding)
                    rows = conn.execute(
                        """
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
                        WHERE d.user_id = ?
                          AND v.embedding MATCH ?
                          AND k = 20
                        ORDER BY v.distance ASC;
                        """,
                        (user_id, raw_query),
                    ).fetchall()
                    for rank, r in enumerate(rows, start=1):
                        cid = r["chunk_id"]
                        vec_candidates[cid] = (
                            rank,
                            {
                                "chunk_id": cid,
                                "document_id": r["document_id"],
                                "filename": r["filename"],
                                "content": r["content"],
                                "page_number": r["page_number"],
                            },
                        )
                except Exception:
                    pass

            # Fallback vector candidates if sqlite-vec was not enabled
            if not self._vec_enabled and query_embedding:
                rows = conn.execute(
                    """
                    SELECT c.id, c.document_id, d.filename, c.content, c.page_number, f.embedding
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    LEFT JOIN document_chunks_vec_fallback f ON f.chunk_rowid = c.rowid
                    WHERE d.user_id = ? AND f.embedding IS NOT NULL;
                    """,
                    (user_id,),
                ).fetchall()

                scored_fallback: List[Tuple[float, Any]] = []
                for r in rows:
                    chunk_vec = _deserialize_vec(r["embedding"])
                    sim = _cosine_similarity(query_embedding, chunk_vec)
                    scored_fallback.append((sim, r))

                scored_fallback.sort(key=lambda x: x[0], reverse=True)
                for rank, (sim, r) in enumerate(scored_fallback[:20], start=1):
                    cid = r["id"]
                    vec_candidates[cid] = (
                        rank,
                        {
                            "chunk_id": cid,
                            "document_id": r["document_id"],
                            "filename": r["filename"],
                            "content": r["content"],
                            "page_number": r["page_number"],
                        },
                    )

            # Step 2: Fetch sparse full-text search candidates (up to 20) with ranks
            fts_candidates: Dict[str, Tuple[int, Dict[str, Any]]] = {}
            if self._fts_enabled:
                sanitized = _sanitize_fts_query(query_text)
                if sanitized:
                    try:
                        rows = conn.execute(
                            """
                            SELECT c.id, c.document_id, d.filename, c.content, c.page_number, rank
                            FROM chunks_fts f
                            JOIN document_chunks c ON c.id = f.chunk_id
                            JOIN documents d ON d.id = c.document_id
                            WHERE f.user_id = ? AND chunks_fts MATCH ?
                            ORDER BY rank
                            LIMIT 20;
                            """,
                            (user_id, sanitized),
                        ).fetchall()
                        for rank, r in enumerate(rows, start=1):
                            cid = r["id"]
                            fts_candidates[cid] = (
                                rank,
                                {
                                    "chunk_id": cid,
                                    "document_id": r["document_id"],
                                    "filename": r["filename"],
                                    "content": r["content"],
                                    "page_number": r["page_number"],
                                },
                            )
                    except Exception:
                        pass

            # Step 3: Compute Reciprocal Rank Fusion (RRF) for each candidate chunk d:
            # RRF(d) = sum_{m in {vec, fts}} 1 / (60 + rank_m(d))
            all_cids = set(vec_candidates.keys()) | set(fts_candidates.keys())
            candidates: List[SearchResult] = []

            for cid in all_cids:
                in_vec = cid in vec_candidates
                in_fts = cid in fts_candidates

                rrf = 0.0
                chunk_data: Optional[Dict[str, Any]] = None

                if in_vec:
                    rank_vec, chunk_data = vec_candidates[cid]
                    rrf += 1.0 / (60.0 + rank_vec)

                if in_fts:
                    rank_fts, fts_data = fts_candidates[cid]
                    rrf += 1.0 / (60.0 + rank_fts)
                    if chunk_data is None:
                        chunk_data = fts_data

                if in_vec and in_fts:
                    source = "hybrid"
                elif in_vec:
                    source = "vector"
                else:
                    source = "fts"

                if chunk_data:
                    candidates.append(
                        SearchResult(
                            chunk_id=cid,
                            document_id=chunk_data["document_id"],
                            filename=chunk_data["filename"],
                            content=chunk_data["content"],
                            page_number=chunk_data["page_number"],
                            score=round(rrf, 4),
                            source=source,
                        )
                    )

            # Step 4: Sort by RRF score descending and apply threshold if appropriate
            candidates.sort(key=lambda x: x.score, reverse=True)

            # Apply score_threshold when defined in RRF scale (<= 0.1)
            if 0.0 < score_threshold <= 0.1:
                candidates = [c for c in candidates if c.score >= score_threshold]

            return candidates[:top_k]

    async def list_documents(self, user_id: str) -> List[DocumentMetadata]:
        """Returns all documents owned by user_id, sorted by created_at DESC (SPEC-3)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, filename, file_type, file_size_bytes, page_count, chunk_count, created_at
                FROM documents
                WHERE user_id = ?
                ORDER BY created_at DESC;
                """,
                (user_id,),
            ).fetchall()

            return [
                DocumentMetadata(
                    id=r["id"],
                    user_id=r["user_id"],
                    filename=r["filename"],
                    file_type=r["file_type"],
                    file_size_bytes=r["file_size_bytes"],
                    page_count=r["page_count"],
                    chunk_count=r["chunk_count"],
                    created_at=_parse_datetime(r["created_at"]),
                )
                for r in rows
            ]

    async def get_document(self, user_id: str, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieves single document metadata if owned by user_id (SPEC-3)."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, filename, file_type, file_size_bytes, page_count, chunk_count, created_at
                FROM documents
                WHERE user_id = ? AND id = ?;
                """,
                (user_id, document_id),
            ).fetchone()

            if not row:
                return None

            return DocumentMetadata(
                id=row["id"],
                user_id=row["user_id"],
                filename=row["filename"],
                file_type=row["file_type"],
                file_size_bytes=row["file_size_bytes"],
                page_count=row["page_count"],
                chunk_count=row["chunk_count"],
                created_at=_parse_datetime(row["created_at"]),
            )

    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Permanently deletes document, all associated chunks, and vector index rows (SPEC-3).
        Supports document ID or filename.
        Uses batched subquery statements to prevent N+1 queries (S-3).
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM documents
                WHERE user_id = ? AND (id = ? OR filename = ?);
                """,
                (user_id, document_id, document_id),
            ).fetchone()

            if not row:
                return False

            doc_id = row["id"]

            # S-3: Batched deletions using subqueries
            if self._vec_enabled:
                try:
                    conn.execute(
                        "DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM document_chunks WHERE document_id = ?);",
                        (doc_id,),
                    )
                except Exception:
                    pass
            else:
                conn.execute(
                    "DELETE FROM document_chunks_vec_fallback WHERE chunk_rowid IN (SELECT rowid FROM document_chunks WHERE document_id = ?);",
                    (doc_id,),
                )

            if self._fts_enabled:
                try:
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?);",
                        (doc_id,),
                    )
                except Exception:
                    pass

            conn.execute("DELETE FROM document_chunks WHERE document_id = ?;", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?;", (doc_id, user_id))

            return True
