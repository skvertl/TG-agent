import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Generator
from core.observability.models import (
    LLMCallSpan,
    ToolCallSpan,
    RunSummary,
    GlobalStats,
    TurnTimeline,
    RunTimeline,
)


def get_default_db_path() -> str:
    """Определяет путь по умолчанию к базе данных телеметрии."""
    env_path = os.environ.get("TELEMETRY_DB_PATH")
    if env_path:
        return env_path

    # Локальная директория data/ в проекте, либо домашний каталог ~/.agent_telemetry
    local_dir = Path("data")
    local_dir.mkdir(parents=True, exist_ok=True)
    return str(local_dir / "telemetry.db")


class TelemetryStorage:
    """Персистентное SQLite хранилище для телеметрии и запусков агента."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_spans (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_cost REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_spans (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_size INTEGER NOT NULL,
                    output_size INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    duration_ms REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_summaries (
                    task_id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_input_tokens INTEGER NOT NULL,
                    total_output_tokens INTEGER NOT NULL,
                    total_cached_tokens INTEGER NOT NULL,
                    repeated_tokens INTEGER NOT NULL,
                    turns_count INTEGER NOT NULL,
                    tool_calls_count INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    duration_ms REAL NOT NULL,
                    success INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_llm_task_id ON llm_spans(task_id);
                CREATE INDEX IF NOT EXISTS idx_tool_task_id ON tool_spans(task_id);
                CREATE INDEX IF NOT EXISTS idx_run_timestamp ON run_summaries(timestamp);
                """
            )

    def save_llm_span(self, span: LLMCallSpan) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_spans (
                    id, timestamp, project_name, agent_id, task_id, session_id,
                    model, turn_number, input_tokens, output_tokens, cached_tokens,
                    reasoning_tokens, latency_ms, estimated_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.id,
                    span.timestamp,
                    span.project_name,
                    span.agent_id,
                    span.task_id,
                    span.session_id,
                    span.model,
                    span.turn_number,
                    span.input_tokens,
                    span.output_tokens,
                    span.cached_tokens,
                    span.reasoning_tokens,
                    span.latency_ms,
                    span.estimated_cost,
                ),
            )

    def save_tool_span(self, span: ToolCallSpan) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_spans (
                    id, timestamp, project_name, task_id, turn_number,
                    tool_name, input_size, output_size, output_tokens, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.id,
                    span.timestamp,
                    span.project_name,
                    span.task_id,
                    span.turn_number,
                    span.tool_name,
                    span.input_size,
                    span.output_size,
                    span.output_tokens,
                    span.duration_ms,
                ),
            )

    def save_run_summary(self, summary: RunSummary) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_summaries (
                    task_id, project_name, session_id, timestamp, model,
                    total_input_tokens, total_output_tokens, total_cached_tokens,
                    repeated_tokens, turns_count, tool_calls_count, total_cost,
                    duration_ms, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.task_id,
                    summary.project_name,
                    summary.session_id,
                    summary.timestamp,
                    summary.model,
                    summary.total_input_tokens,
                    summary.total_output_tokens,
                    summary.total_cached_tokens,
                    summary.repeated_tokens,
                    summary.turns_count,
                    summary.tool_calls_count,
                    summary.total_cost,
                    summary.duration_ms,
                    1 if summary.success else 0,
                ),
            )

    def get_recent_runs(
        self,
        limit: int = 10,
        project_name: Optional[str] = None,
    ) -> List[RunSummary]:
        with self._get_connection() as conn:
            if project_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM run_summaries
                    WHERE project_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (project_name, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM run_summaries
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()
            return [
                RunSummary(
                    task_id=row["task_id"],
                    project_name=row["project_name"],
                    session_id=row["session_id"],
                    timestamp=row["timestamp"],
                    model=row["model"],
                    total_input_tokens=row["total_input_tokens"],
                    total_output_tokens=row["total_output_tokens"],
                    total_cached_tokens=row["total_cached_tokens"],
                    repeated_tokens=row["repeated_tokens"],
                    turns_count=row["turns_count"],
                    tool_calls_count=row["tool_calls_count"],
                    total_cost=row["total_cost"],
                    duration_ms=row["duration_ms"],
                    success=bool(row["success"]),
                )
                for row in rows
            ]

    def get_global_stats(self, project_name: Optional[str] = None) -> GlobalStats:
        with self._get_connection() as conn:
            where_clause = "WHERE project_name = ?" if project_name else ""
            params = (project_name,) if project_name else ()

            # Агрегация по run_summaries
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_tasks,
                    COALESCE(SUM(total_input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(total_output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(total_cached_tokens), 0) as total_cached_tokens,
                    COALESCE(SUM(repeated_tokens), 0) as total_repeated_tokens,
                    COALESCE(SUM(total_cost), 0.0) as total_cost,
                    COALESCE(AVG(total_input_tokens + total_output_tokens), 0.0) as avg_tokens_per_task,
                    COALESCE(AVG(turns_count), 0.0) as avg_turns_per_task,
                    COALESCE(AVG(tool_calls_count), 0.0) as avg_tools_per_task
                FROM run_summaries
                {where_clause}
                """,
                params,
            )
            row = cursor.fetchone()
            total_tasks = row["total_tasks"]
            total_input = row["total_input_tokens"]
            total_output = row["total_output_tokens"]
            total_cached = row["total_cached_tokens"]
            total_repeated = row["total_repeated_tokens"]
            total_cost = round(row["total_cost"], 4)
            avg_tokens = round(row["avg_tokens_per_task"], 1)
            avg_turns = round(row["avg_turns_per_task"], 1)
            avg_tools = round(row["avg_tools_per_task"], 1)

            cache_hit_rate = (
                round((total_cached / total_input) * 100.0, 1) if total_input > 0 else 0.0
            )
            repeated_pct = (
                round((total_repeated / total_input) * 100.0, 1) if total_input > 0 else 0.0
            )

            # Распределение по инструментам
            tool_cursor = conn.execute(
                f"""
                SELECT tool_name, COALESCE(SUM(output_tokens), 0) as total_tokens
                FROM tool_spans
                {where_clause}
                GROUP BY tool_name
                ORDER BY total_tokens DESC
                """,
                params,
            )
            tool_rows = tool_cursor.fetchall()
            all_tool_tokens = sum(r["total_tokens"] for r in tool_rows)
            tool_usage_pct: Dict[str, float] = {}
            if all_tool_tokens > 0:
                for r in tool_rows:
                    tool_usage_pct[r["tool_name"]] = round(
                        (r["total_tokens"] / all_tool_tokens) * 100.0, 1
                    )

            # Распределение по моделям
            model_cursor = conn.execute(
                f"""
                SELECT model, COUNT(*) as count
                FROM run_summaries
                {where_clause}
                GROUP BY model
                ORDER BY count DESC
                """,
                params,
            )
            model_rows = model_cursor.fetchall()
            models_usage = {r["model"]: r["count"] for r in model_rows}
            active_model = model_rows[0]["model"] if model_rows else None

            return GlobalStats(
                total_tasks=total_tasks,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_cached_tokens=total_cached,
                total_cost=total_cost,
                avg_tokens_per_task=avg_tokens,
                avg_turns_per_task=avg_turns,
                avg_tools_per_task=avg_tools,
                cache_hit_rate=cache_hit_rate,
                repeated_tokens_pct=repeated_pct,
                tool_usage_pct=tool_usage_pct,
                active_model=active_model,
                models_usage=models_usage,
            )

    def get_run_timeline(self, task_id: str) -> Optional[RunTimeline]:
        with self._get_connection() as conn:
            # Получаем summary
            cursor = conn.execute(
                "SELECT * FROM run_summaries WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            summary = RunSummary(
                task_id=row["task_id"],
                project_name=row["project_name"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                model=row["model"],
                total_input_tokens=row["total_input_tokens"],
                total_output_tokens=row["total_output_tokens"],
                total_cached_tokens=row["total_cached_tokens"],
                repeated_tokens=row["repeated_tokens"],
                turns_count=row["turns_count"],
                tool_calls_count=row["tool_calls_count"],
                total_cost=row["total_cost"],
                duration_ms=row["duration_ms"],
                success=bool(row["success"]),
            )

            # Получаем все LLM spans
            llm_cursor = conn.execute(
                "SELECT * FROM llm_spans WHERE task_id = ? ORDER BY turn_number ASC",
                (task_id,),
            )
            llm_by_turn: Dict[int, LLMCallSpan] = {}
            for l_row in llm_cursor.fetchall():
                llm_by_turn[l_row["turn_number"]] = LLMCallSpan(
                    id=l_row["id"],
                    timestamp=l_row["timestamp"],
                    project_name=l_row["project_name"],
                    agent_id=l_row["agent_id"],
                    task_id=l_row["task_id"],
                    session_id=l_row["session_id"],
                    model=l_row["model"],
                    turn_number=l_row["turn_number"],
                    input_tokens=l_row["input_tokens"],
                    output_tokens=l_row["output_tokens"],
                    cached_tokens=l_row["cached_tokens"],
                    reasoning_tokens=l_row["reasoning_tokens"],
                    latency_ms=l_row["latency_ms"],
                    estimated_cost=l_row["estimated_cost"],
                )

            # Получаем все Tool spans
            tool_cursor = conn.execute(
                "SELECT * FROM tool_spans WHERE task_id = ? ORDER BY turn_number ASC, timestamp ASC",
                (task_id,),
            )
            tools_by_turn: Dict[int, List[ToolCallSpan]] = {}
            for t_row in tool_cursor.fetchall():
                turn = t_row["turn_number"]
                if turn not in tools_by_turn:
                    tools_by_turn[turn] = []
                tools_by_turn[turn].append(
                    ToolCallSpan(
                        id=t_row["id"],
                        timestamp=t_row["timestamp"],
                        project_name=t_row["project_name"],
                        task_id=t_row["task_id"],
                        turn_number=t_row["turn_number"],
                        tool_name=t_row["tool_name"],
                        input_size=t_row["input_size"],
                        output_size=t_row["output_size"],
                        output_tokens=t_row["output_tokens"],
                        duration_ms=t_row["duration_ms"],
                    )
                )

            all_turns = sorted(set(list(llm_by_turn.keys()) + list(tools_by_turn.keys())))
            turns: List[TurnTimeline] = []
            for t in all_turns:
                turns.append(
                    TurnTimeline(
                        turn_number=t,
                        llm_span=llm_by_turn.get(t),
                        tool_spans=tools_by_turn.get(t, []),
                    )
                )

            return RunTimeline(summary=summary, turns=turns)
