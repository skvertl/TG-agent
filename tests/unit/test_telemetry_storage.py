import pytest
from core.observability.models import (
    LLMCallSpan,
    ToolCallSpan,
    RunSummary,
)
from core.observability.storage import TelemetryStorage


class TestTelemetryStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        db_path = tmp_path / "test_telemetry.db"
        return TelemetryStorage(db_path=str(db_path))

    def test_save_and_retrieve_spans(self, storage):
        llm_span = LLMCallSpan(
            task_id="task-1",
            turn_number=1,
            model="qwen2.5:7b",
            input_tokens=1000,
            output_tokens=200,
            cached_tokens=100,
            latency_ms=350.0,
            estimated_cost=0.0005,
        )
        tool_span = ToolCallSpan(
            task_id="task-1",
            turn_number=1,
            tool_name="read_file",
            input_size=50,
            output_size=2000,
            output_tokens=500,
            duration_ms=12.0,
        )
        storage.save_llm_span(llm_span)
        storage.save_tool_span(tool_span)

        summary = RunSummary(
            task_id="task-1",
            model="qwen2.5:7b",
            total_input_tokens=1000,
            total_output_tokens=200,
            total_cached_tokens=100,
            repeated_tokens=300,
            turns_count=1,
            tool_calls_count=1,
            total_cost=0.0005,
            duration_ms=362.0,
            success=True,
        )
        storage.save_run_summary(summary)

        recent = storage.get_recent_runs(limit=10)
        assert len(recent) == 1
        assert recent[0].task_id == "task-1"
        assert recent[0].total_input_tokens == 1000

    def test_get_global_stats(self, storage):
        # Run 1
        storage.save_tool_span(ToolCallSpan(task_id="t1", turn_number=1, tool_name="read_file", output_tokens=300))
        storage.save_tool_span(ToolCallSpan(task_id="t1", turn_number=2, tool_name="search", output_tokens=100))
        storage.save_run_summary(
            RunSummary(
                task_id="t1",
                model="qwen2.5:7b",
                total_input_tokens=4000,
                total_output_tokens=1000,
                total_cached_tokens=2000,
                repeated_tokens=2500,
                turns_count=2,
                tool_calls_count=2,
                total_cost=0.002,
                duration_ms=1500.0,
                success=True,
            )
        )
        # Run 2
        storage.save_tool_span(ToolCallSpan(task_id="t2", turn_number=1, tool_name="read_file", output_tokens=200))
        storage.save_run_summary(
            RunSummary(
                task_id="t2",
                model="qwen2.5:7b",
                total_input_tokens=6000,
                total_output_tokens=1000,
                total_cached_tokens=1000,
                repeated_tokens=3500,
                turns_count=3,
                tool_calls_count=1,
                total_cost=0.003,
                duration_ms=2000.0,
                success=True,
            )
        )

        stats = storage.get_global_stats()
        assert stats.total_tasks == 2
        assert stats.total_input_tokens == 10000
        assert stats.total_output_tokens == 2000
        assert stats.total_cached_tokens == 3000
        assert stats.total_cost == pytest.approx(0.005, rel=1e-3)
        assert stats.avg_turns_per_task == 2.5
        assert stats.avg_tools_per_task == 1.5
        # Cache hit rate = cached / input = 3000 / 10000 = 30%
        assert stats.cache_hit_rate == pytest.approx(30.0, rel=1e-1)
        # Tool usage: read_file (300+200=500), search (100) -> total 600
        # read_file = 500/600 = 83.3%
        assert "read_file" in stats.tool_usage_pct
        assert stats.tool_usage_pct["read_file"] > 80.0

    def test_get_run_timeline(self, storage):
        task_id = "task-timeline"
        # Turn 1: LLM + search
        storage.save_llm_span(LLMCallSpan(task_id=task_id, turn_number=1, model="qwen2.5:7b", input_tokens=1000, output_tokens=100))
        storage.save_tool_span(ToolCallSpan(task_id=task_id, turn_number=1, tool_name="search", output_tokens=300))

        # Turn 2: LLM + read_file
        storage.save_llm_span(LLMCallSpan(task_id=task_id, turn_number=2, model="qwen2.5:7b", input_tokens=2000, output_tokens=150))
        storage.save_tool_span(ToolCallSpan(task_id=task_id, turn_number=2, tool_name="read_file", output_tokens=800))

        summary = RunSummary(
            task_id=task_id,
            model="qwen2.5:7b",
            total_input_tokens=3000,
            total_output_tokens=250,
            total_cached_tokens=500,
            repeated_tokens=1000,
            turns_count=2,
            tool_calls_count=2,
            total_cost=0.001,
            duration_ms=1200.0,
            success=True,
        )
        storage.save_run_summary(summary)

        timeline = storage.get_run_timeline(task_id)
        assert timeline is not None
        assert timeline.summary.task_id == task_id
        assert len(timeline.turns) == 2
        assert timeline.turns[0].turn_number == 1
        assert timeline.turns[0].llm_span.input_tokens == 1000
        assert len(timeline.turns[0].tool_spans) == 1
        assert timeline.turns[0].tool_spans[0].tool_name == "search"
        assert timeline.turns[1].turn_number == 2
        assert timeline.turns[1].tool_spans[0].tool_name == "read_file"
