import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


def _get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCallSpan(BaseModel):
    """Телеметрия единичного обращения к языковой модели."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_get_utc_now)
    project_name: str = Field(default="tg-agent")
    agent_id: str = Field(default="default-agent")
    task_id: str = Field(...)
    session_id: str = Field(default="default-session")
    model: str = Field(...)
    turn_number: int = Field(default=1, ge=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    estimated_cost: float = Field(default=0.0, ge=0.0)


class ToolCallSpan(BaseModel):
    """Телеметрия вызова инструмента агентом."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_get_utc_now)
    project_name: str = Field(default="tg-agent")
    task_id: str = Field(...)
    turn_number: int = Field(default=1, ge=1)
    tool_name: str = Field(...)
    input_size: int = Field(default=0, ge=0)
    output_size: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)


class RunSummary(BaseModel):
    """Итоговая сводка по выполнению задачи / диалога."""

    task_id: str = Field(...)
    project_name: str = Field(default="tg-agent")
    session_id: str = Field(default="default-session")
    timestamp: str = Field(default_factory=_get_utc_now)
    model: str = Field(...)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_cached_tokens: int = Field(default=0, ge=0)
    repeated_tokens: int = Field(default=0, ge=0)
    turns_count: int = Field(default=1, ge=1)
    tool_calls_count: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    success: bool = Field(default=True)


class TurnTimeline(BaseModel):
    """Шаг таймлайна задачи."""

    turn_number: int
    llm_span: Optional[LLMCallSpan] = None
    tool_spans: List[ToolCallSpan] = Field(default_factory=list)


class RunTimeline(BaseModel):
    """Полный пошаговый таймлайн выполнения запуска."""

    summary: RunSummary
    turns: List[TurnTimeline] = Field(default_factory=list)


class GlobalStats(BaseModel):
    """Глобальная сводка по всем запускам."""

    total_tasks: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    avg_tokens_per_task: float = 0.0
    avg_turns_per_task: float = 0.0
    avg_tools_per_task: float = 0.0
    cache_hit_rate: float = 0.0
    repeated_tokens_pct: float = 0.0
    tool_usage_pct: Dict[str, float] = Field(default_factory=dict)
    active_model: Optional[str] = None
    models_usage: Dict[str, int] = Field(default_factory=dict)

