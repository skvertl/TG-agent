"""Core domain data models."""
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

RoleType = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Single message in a dialog."""
    role: RoleType
    content: str = Field(..., min_length=1)


class PromptPayload(BaseModel):
    """Payload passed to the LLM port."""
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    options: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Inference result produced by the agent or plugin."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    task_id: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Metadata for an ingested user document."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique document UUID")
    user_id: str = Field(..., description="Owner's Telegram user/chat ID")
    filename: str = Field(..., description="Original filename with extension")
    file_type: str = Field(..., description="MIME type or file extension (txt, md, docx, pdf)")
    file_size_bytes: int = Field(..., ge=0, description="Size in bytes")
    page_count: int = Field(default=1, ge=1, description="Total number of pages")
    chunk_count: int = Field(default=0, ge=0, description="Total number of chunks produced")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of ingestion",
    )


class DocumentChunk(BaseModel):
    """An individual text segment of a document."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique chunk UUID")
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



