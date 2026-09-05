import pytest
from core.observability.models import (
    LLMCallSpan,
    ToolCallSpan,
    RunSummary,
    GlobalStats,
    TurnTimeline,
    RunTimeline,
)


class TestObservabilityModels:
    def test_llm_call_span_defaults_and_validation(self):
        span = LLMCallSpan(
            task_id="task-101",
            model="qwen2.5:7b",
            turn_number=1,
            input_tokens=150,
            output_tokens=50,
            cached_tokens=20,
            latency_ms=450.2,
            estimated_cost=0.0001,
        )
        assert span.task_id == "task-101"
        assert span.model == "qwen2.5:7b"
        assert span.turn_number == 1
        assert span.input_tokens == 150
        assert span.output_tokens == 50
        assert span.cached_tokens == 20
        assert span.reasoning_tokens == 0
        assert span.project_name == "tg-agent"
        assert span.id is not None
        assert span.timestamp is not None

    def test_tool_call_span(self):
        span = ToolCallSpan(
            task_id="task-101",
            turn_number=1,
            tool_name="read_file",
            input_size=120,
            output_size=4500,
            output_tokens=1100,
            duration_ms=25.4,
        )
        assert span.tool_name == "read_file"
        assert span.output_tokens == 1100
        assert span.duration_ms == 25.4

    def test_run_summary(self):
        summary = RunSummary(
            task_id="task-101",
            model="qwen2.5:7b",
            total_input_tokens=5000,
            total_output_tokens=1000,
            total_cached_tokens=2000,
            repeated_tokens=3500,
            turns_count=3,
            tool_calls_count=4,
            total_cost=0.0035,
            duration_ms=3200.0,
            success=True,
        )
        assert summary.total_input_tokens == 5000
        assert summary.repeated_tokens == 3500
        assert summary.success is True

    def test_global_stats(self):
        stats = GlobalStats(
            total_tasks=10,
            total_input_tokens=50000,
            total_output_tokens=10000,
            total_cached_tokens=25000,
            total_cost=0.045,
            avg_tokens_per_task=6000.0,
            avg_turns_per_task=3.5,
            avg_tools_per_task=4.2,
            cache_hit_rate=33.3,
            repeated_tokens_pct=65.4,
            tool_usage_pct={"read_file": 50.0, "search": 50.0},
        )
        assert stats.total_tasks == 10
        assert stats.cache_hit_rate == 33.3
