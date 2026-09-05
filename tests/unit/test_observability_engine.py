import pytest
import asyncio
from core.observability.engine import ObservabilityEngine
from core.observability.storage import TelemetryStorage


class TestObservabilityEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        db = tmp_path / "engine_test.db"
        storage = TelemetryStorage(db_path=str(db))
        return ObservabilityEngine(project_name="test-project", storage=storage)

    @pytest.mark.asyncio
    async def test_track_llm_call(self, engine):
        task_id = "task-llm-1"
        async with engine.track_llm(
            task_id=task_id,
            model="qwen2.5:7b",
            turn_number=1,
            messages=["System prompt", "User query"],
        ) as span:
            await asyncio.sleep(0.01)
            span.set_tokens(input_tokens=1000, output_tokens=200, cached_tokens=100)

        assert span.latency_ms > 0
        assert span.estimated_cost > 0

        # Check storage
        timeline = engine.storage.get_run_timeline(task_id)
        # Note: run not finished yet, but span was saved
        recent = engine.storage.get_recent_runs(limit=5)
        assert len(recent) == 0  # not finished yet

    def test_track_tool_call(self, engine):
        task_id = "task-tool-1"
        with engine.track_tool(
            task_id=task_id,
            tool_name="read_file",
            turn_number=1,
            input_data="test_file.py",
        ) as span:
            span.set_output("def hello():\n    return 'world'\n" * 10)

        assert span.output_size > 0
        assert span.output_tokens > 0
        assert span.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_finish_run_aggregates_totals(self, engine):
        task_id = "task-full-flow"
        # Turn 1: LLM
        async with engine.track_llm(
            task_id=task_id,
            model="qwen2.5:7b",
            turn_number=1,
            messages=["Turn 1 prompt"],
        ) as s1:
            s1.set_tokens(input_tokens=500, output_tokens=100)

        # Turn 1: Tool
        with engine.track_tool(
            task_id=task_id,
            tool_name="search",
            turn_number=1,
            input_data="query",
        ) as t1:
            t1.set_output("Found 3 matches")

        # Turn 2: LLM (with repeat)
        async with engine.track_llm(
            task_id=task_id,
            model="qwen2.5:7b",
            turn_number=2,
            messages=["Turn 1 prompt", "Found 3 matches", "Next step"],
        ) as s2:
            s2.set_tokens(input_tokens=800, output_tokens=150)

        summary = engine.finish_run(task_id=task_id, model="qwen2.5:7b", success=True)

        assert summary.task_id == task_id
        assert summary.turns_count == 2
        assert summary.tool_calls_count == 1
        assert summary.total_input_tokens == 1300
        assert summary.total_output_tokens == 250
        assert summary.repeated_tokens > 0
        assert summary.total_cost > 0

        # Check that it appears in recent runs
        recent = engine.storage.get_recent_runs()
        assert len(recent) == 1
        assert recent[0].task_id == task_id
