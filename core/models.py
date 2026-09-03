"""Core domain data models."""
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
