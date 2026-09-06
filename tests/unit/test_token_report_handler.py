from unittest.mock import AsyncMock
import pytest
from aiogram.types import Message, Chat, User
from adapters.telegram.handlers import token_report_handler
from core.runner import AgentRunner
from core.observability.engine import ObservabilityEngine
from core.observability.storage import TelemetryStorage
from core.observability.models import LLMCallSpan, RunSummary


class TestTokenReportHandler:
    @pytest.fixture
    def mock_message(self):
        message = AsyncMock(spec=Message)
        message.chat = Chat(id=123, type="private")
        message.from_user = User(id=123, is_bot=False, first_name="TestUser")
        message.text = "/token_report"
        message.answer = AsyncMock()
        return message

    @pytest.mark.asyncio
    async def test_token_report_empty(self, mock_message, tmp_path):
        storage = TelemetryStorage(db_path=str(tmp_path / "tg_test.db"))
        engine = ObservabilityEngine(storage=storage)
        runner = AsyncMock(spec=AgentRunner)
        runner.observability_engine = engine

        await token_report_handler(mock_message, runner)

        mock_message.answer.assert_awaited_once()
        text = mock_message.answer.call_args[0][0]
        assert "AI AGENT OBSERVABILITY" in text
        assert "Всего задач:" in text

    @pytest.mark.asyncio
    async def test_token_report_with_data(self, mock_message, tmp_path):
        storage = TelemetryStorage(db_path=str(tmp_path / "tg_test_data.db"))
        engine = ObservabilityEngine(storage=storage)
        runner = AsyncMock(spec=AgentRunner)
        runner.observability_engine = engine

        # Seed data
        storage.save_llm_span(
            LLMCallSpan(task_id="task-seed-1", turn_number=1, model="qwen2.5:7b", input_tokens=5000, output_tokens=1000)
        )
        storage.save_run_summary(
            RunSummary(
                task_id="task-seed-1",
                model="qwen2.5:7b",
                total_input_tokens=5000,
                total_output_tokens=1000,
                total_cached_tokens=1500,
                repeated_tokens=2000,
                turns_count=1,
                tool_calls_count=0,
                total_cost=0.0027,
                duration_ms=800.0,
                success=True,
            )
        )

        await token_report_handler(mock_message, runner)

        mock_message.answer.assert_awaited_once()
        text = mock_message.answer.call_args[0][0]
        assert "task-seed-1" in text
        assert "$0.0027" in text

    @pytest.mark.asyncio
    async def test_tokenreport_alias_without_underscore(self, mock_message, tmp_path):
        mock_message.text = "/tokenreport"
        storage = TelemetryStorage(db_path=str(tmp_path / "tg_test_alias.db"))
        engine = ObservabilityEngine(storage=storage)
        runner = AsyncMock(spec=AgentRunner)
        runner.observability_engine = engine

        await token_report_handler(mock_message, runner)
        mock_message.answer.assert_awaited_once()
        text = mock_message.answer.call_args[0][0]
        assert "AI AGENT OBSERVABILITY" in text
