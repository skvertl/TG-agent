import time
from typing import Optional, List, Dict, Any
from core.observability.models import (
    LLMCallSpan,
    ToolCallSpan,
    RunSummary,
)
from core.observability.storage import TelemetryStorage
from core.observability.pricing import calculate_cost
from core.observability.overlap import estimate_tokens, calculate_context_overlap


class LLMSpanContext:
    """Контекст отслеживания единичного LLM-вызова."""

    def __init__(
        self,
        engine: "ObservabilityEngine",
        task_id: str,
        model: str,
        turn_number: int,
        session_id: str = "default",
        agent_id: str = "default-agent",
        messages: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.span = LLMCallSpan(
            project_name=engine.project_name,
            agent_id=agent_id,
            task_id=task_id,
            session_id=session_id,
            model=model,
            turn_number=turn_number,
        )
        self.messages = messages or []
        self._start_time = 0.0

    def __getattr__(self, item: str) -> Any:
        return getattr(self.span, item)

    def set_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        self.span.input_tokens = input_tokens
        self.span.output_tokens = output_tokens
        self.span.cached_tokens = cached_tokens
        self.span.reasoning_tokens = reasoning_tokens

    async def __aenter__(self) -> "LLMSpanContext":
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.span.latency_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        self.span.estimated_cost = calculate_cost(
            model=self.span.model,
            input_tokens=self.span.input_tokens,
            output_tokens=self.span.output_tokens,
            cached_tokens=self.span.cached_tokens,
        )

        # Регистрация спана и подсчет повторов в контексте
        self.engine._record_llm_span(self.span, self.messages)


class ToolSpanContext:
    """Контекст отслеживания единичного вызова инструмента."""

    def __init__(
        self,
        engine: "ObservabilityEngine",
        task_id: str,
        tool_name: str,
        turn_number: int,
        input_data: Any = None,
    ):
        self.engine = engine
        input_str = str(input_data) if input_data is not None else ""
        self.span = ToolCallSpan(
            project_name=engine.project_name,
            task_id=task_id,
            turn_number=turn_number,
            tool_name=tool_name,
            input_size=len(input_str.encode("utf-8")),
        )
        self._start_time = 0.0

    def __getattr__(self, item: str) -> Any:
        return getattr(self.span, item)

    def set_output(self, output: Any) -> None:
        out_str = str(output) if output is not None else ""
        self.span.output_size = len(out_str.encode("utf-8"))
        self.span.output_tokens = estimate_tokens(out_str)

    def __enter__(self) -> "ToolSpanContext":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.span.duration_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        self.engine._record_tool_span(self.span)


class ObservabilityEngine:
    """
    Автономный и переносимый движок мониторинга LLM-агентов.
    Может быть перенесен в любой другой проект без зависимостей от Telegram.
    """

    def __init__(
        self,
        project_name: str = "tg-agent",
        storage: Optional[TelemetryStorage] = None,
    ):
        self.project_name = project_name
        self.storage = storage or TelemetryStorage()
        # Активное состояние по задачам в памяти
        self._active_tasks: Dict[str, Dict[str, Any]] = {}

    def _ensure_task(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._active_tasks:
            self._active_tasks[task_id] = {
                "start_time": time.perf_counter(),
                "llm_spans": [],
                "tool_spans": [],
                "previous_messages": [],
                "total_repeated_tokens": 0,
            }
        return self._active_tasks[task_id]

    def track_llm(
        self,
        task_id: str,
        model: str,
        turn_number: int = 1,
        session_id: str = "default",
        agent_id: str = "default-agent",
        messages: Optional[List[str]] = None,
    ) -> LLMSpanContext:
        """Создает контекстный менеджер для замера и трейсинга LLM-вызова."""
        self._ensure_task(task_id)
        return LLMSpanContext(
            engine=self,
            task_id=task_id,
            model=model,
            turn_number=turn_number,
            session_id=session_id,
            agent_id=agent_id,
            messages=messages,
        )

    def track_tool(
        self,
        task_id: str,
        tool_name: str,
        turn_number: int = 1,
        input_data: Any = None,
    ) -> ToolSpanContext:
        """Создает контекстный менеджер для замера и трейсинга вызова инструмента."""
        self._ensure_task(task_id)
        return ToolSpanContext(
            engine=self,
            task_id=task_id,
            tool_name=tool_name,
            turn_number=turn_number,
            input_data=input_data,
        )

    def _record_llm_span(self, span: LLMCallSpan, messages: List[str]) -> None:
        task_data = self._ensure_task(span.task_id)
        task_data["llm_spans"].append(span)

        # Вычисляем повторный контекст, если есть история сообщений
        prev = task_data["previous_messages"]
        if prev and messages:
            repeated, _ = calculate_context_overlap(prev, messages)
            task_data["total_repeated_tokens"] += repeated

        if messages:
            task_data["previous_messages"] = list(messages)

        # Персистим в хранилище
        self.storage.save_llm_span(span)

    def _record_tool_span(self, span: ToolCallSpan) -> None:
        task_data = self._ensure_task(span.task_id)
        task_data["tool_spans"].append(span)
        # Персистим в хранилище
        self.storage.save_tool_span(span)

    def finish_run(
        self,
        task_id: str,
        model: str,
        session_id: str = "default",
        success: bool = True,
    ) -> RunSummary:
        """Финализирует задачу, подсчитывает агрегаты и сохраняет RunSummary в базу."""
        task_data = self._active_tasks.pop(task_id, {
            "start_time": time.perf_counter(),
            "llm_spans": [],
            "tool_spans": [],
            "previous_messages": [],
            "total_repeated_tokens": 0,
        })

        llm_spans: List[LLMCallSpan] = task_data["llm_spans"]
        tool_spans: List[ToolCallSpan] = task_data["tool_spans"]

        total_input = sum(s.input_tokens for s in llm_spans)
        total_output = sum(s.output_tokens for s in llm_spans)
        total_cached = sum(s.cached_tokens for s in llm_spans)
        total_cost = round(sum(s.estimated_cost for s in llm_spans), 6)

        turns_count = max([s.turn_number for s in llm_spans] + [1])
        duration_ms = round((time.perf_counter() - task_data["start_time"]) * 1000.0, 2)

        summary = RunSummary(
            task_id=task_id,
            project_name=self.project_name,
            session_id=session_id,
            model=model,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cached_tokens=total_cached,
            repeated_tokens=task_data["total_repeated_tokens"],
            turns_count=turns_count,
            tool_calls_count=len(tool_spans),
            total_cost=total_cost,
            duration_ms=duration_ms,
            success=success,
        )

        self.storage.save_run_summary(summary)
        return summary
