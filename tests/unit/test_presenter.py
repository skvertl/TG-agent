import pytest
from core.observability.models import (
    GlobalStats,
    RunSummary,
    RunTimeline,
    TurnTimeline,
    LLMCallSpan,
    ToolCallSpan,
)
from core.observability.presenter import (
    format_token_count,
    format_telegram_report,
    format_timeline_markdown,
)


class TestPresenter:
    def test_format_token_count(self):
        assert format_token_count(500) == "500"
        assert format_token_count(1500) == "1.5k"
        assert format_token_count(39400) == "39.4k"
        assert format_token_count(4200000) == "4.2M"

    def test_format_telegram_report(self):
        stats = GlobalStats(
            total_tasks=127,
            total_input_tokens=4200000,
            total_output_tokens=800000,
            total_cached_tokens=2700000,
            total_cost=18.42,
            avg_tokens_per_task=39400.0,
            avg_turns_per_task=8.7,
            avg_tools_per_task=16.2,
            cache_hit_rate=64.0,
            repeated_tokens_pct=73.0,
            tool_usage_pct={"read_file": 31.0, "run_tests": 22.0, "search": 14.0},
        )
        recent = [
            RunSummary(
                task_id="task-184",
                model="qwen2.5:7b",
                total_input_tokens=24100,
                total_output_tokens=3200,
                total_cached_tokens=15000,
                repeated_tokens=17000,
                turns_count=5,
                tool_calls_count=4,
                total_cost=0.035,
                duration_ms=4500.0,
                success=True,
            )
        ]
        timeline = RunTimeline(
            summary=recent[0],
            turns=[
                TurnTimeline(
                    turn_number=1,
                    llm_span=LLMCallSpan(task_id="task-184", turn_number=1, model="qwen2.5:7b", input_tokens=8200, output_tokens=150),
                    tool_spans=[ToolCallSpan(task_id="task-184", turn_number=1, tool_name="search", output_tokens=3100)],
                )
            ],
        )

        report = format_telegram_report(
            stats=stats,
            recent_runs=recent,
            last_timeline=timeline,
        )

        assert "AI AGENT OBSERVABILITY" in report
        assert "127" in report
        assert "4.2M" in report
        assert "$18.42" in report
        assert "read_file" in report
        assert "task-184" in report
        assert "search" in report
        assert "qwen2.5:7b" in report
        assert "Формула" in report


    def test_format_timeline_markdown(self):
        summary = RunSummary(
            task_id="task-999",
            model="llama3.2",
            total_input_tokens=10000,
            total_output_tokens=1000,
            total_cached_tokens=2000,
            repeated_tokens=5000,
            turns_count=2,
            tool_calls_count=1,
            total_cost=0.002,
            duration_ms=1000.0,
            success=True,
        )
        timeline = RunTimeline(
            summary=summary,
            turns=[
                TurnTimeline(
                    turn_number=1,
                    llm_span=LLMCallSpan(task_id="task-999", turn_number=1, model="llama3.2", input_tokens=5000, output_tokens=200),
                    tool_spans=[ToolCallSpan(task_id="task-999", turn_number=1, tool_name="read_file", output_tokens=1200)],
                )
            ],
        )
        text = format_timeline_markdown(timeline)
        assert "task-999" in text
        assert "Turn 1" in text
        assert "read_file" in text
