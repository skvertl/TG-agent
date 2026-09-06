import pytest
from unittest.mock import AsyncMock, MagicMock
from core.runner import AgentRunner
from core.tools.registry import ToolRegistry
from adapters.telegram.handlers import (
    skills_list_handler,
    morning_briefing_cmd_handler,
    system_health_cmd_handler,
)
from aiogram.types import Message, Chat, User
from core.models import AgentResult


def test_parse_action_robust_to_multi_action():
    runner = AgentRunner(llm_plugin=MagicMock(), memory_store=MagicMock())
    text = (
        'Action: exec\n'
        'Action Input: {"command": "curl -s \\"https://wttr.in/Minsk?format=3\\""}\n\n'
        'Action: exec\n'
        'Action Input: {"command": "date"}\n'
    )
    parsed = runner._parse_action(text)
    assert parsed is not None
    tool_name, args = parsed
    assert tool_name == "exec"
    assert args == {"command": 'curl -s "https://wttr.in/Minsk?format=3"'}


def test_exec_tool_kwargs_resilience():
    import sys
    obs = MagicMock()
    registry = ToolRegistry(engine=obs)
    # Call with 'input' keyword
    output = registry.execute("exec", "t1", 1, {"input": f'"{sys.executable}" -c "print(\'hello\')"' })
    assert "hello" in output


def test_read_skill_kwargs_resilience():
    obs = MagicMock()
    registry = ToolRegistry(engine=obs)
    output = registry.execute("read_skill", "t1", 1, {"input": "morning-briefing"})
    assert "Morning Briefing" in output


@pytest.mark.asyncio
async def test_morning_briefing_cmd_handler():
    message = MagicMock(spec=Message)
    message.text = "/morning_briefing"
    message.chat = MagicMock(spec=Chat, id=12345)
    message.from_user = MagicMock(spec=User, id=12345)
    message.bot = MagicMock()
    message.answer = AsyncMock()

    runner = MagicMock(spec=AgentRunner)
    runner.run = AsyncMock(return_value=AgentResult(
        content="☀️ Morning digest ready!",
        model="qwen2.5:7b",
        task_id="t1",
    ))

    await morning_briefing_cmd_handler(message, runner)
    runner.run.assert_called_once()
    assert "Москва" in runner.run.call_args[1]["user_prompt"]
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_morning_briefing_cmd_handler_with_city():
    message = MagicMock(spec=Message)
    message.text = "/morning_briefing Минск"
    message.chat = MagicMock(spec=Chat, id=12345)
    message.from_user = MagicMock(spec=User, id=12345)
    message.bot = MagicMock()
    message.answer = AsyncMock()

    runner = MagicMock(spec=AgentRunner)
    runner.run = AsyncMock(return_value=AgentResult(
        content="☀️ Morning digest for Minsk ready!",
        model="qwen2.5:7b",
        task_id="t1",
    ))

    await morning_briefing_cmd_handler(message, runner)
    runner.run.assert_called_once()
    assert "Минск" in runner.run.call_args[1]["user_prompt"]
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_weather_cmd_handler_with_english_city():
    message = MagicMock(spec=Message)
    message.text = "/weather London"
    message.chat = MagicMock(spec=Chat, id=12345)
    message.from_user = MagicMock(spec=User, id=12345)
    message.bot = MagicMock()
    message.answer = AsyncMock()

    runner = MagicMock(spec=AgentRunner)
    runner.run = AsyncMock(return_value=AgentResult(
        content="☀️ London weather ready!",
        model="qwen2.5:7b",
        task_id="t1",
    ))

    await morning_briefing_cmd_handler(message, runner)
    runner.run.assert_called_once()
    assert "London" in runner.run.call_args[1]["user_prompt"]
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_system_health_cmd_handler():
    message = MagicMock(spec=Message)
    message.text = "/system_health"
    message.chat = MagicMock(spec=Chat, id=12345)
    message.from_user = MagicMock(spec=User, id=12345)
    message.bot = MagicMock()
    message.answer = AsyncMock()

    runner = MagicMock(spec=AgentRunner)
    runner.run = AsyncMock(return_value=AgentResult(
        content="🟢 System is healthy!",
        model="qwen2.5:7b",
        task_id="t2",
    ))

    await system_health_cmd_handler(message, runner)
    runner.run.assert_called_once()
    assert "system-health" in runner.run.call_args[1]["user_prompt"]
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_skills_list_handler():
    message = MagicMock(spec=Message)
    message.text = "/skills"
    message.chat = MagicMock(spec=Chat, id=12345)
    message.answer = AsyncMock()

    runner = MagicMock(spec=AgentRunner)
    runner._get_skills_summary = MagicMock(return_value="- morning-briefing\n- system-health")

    await skills_list_handler(message, runner)
    message.answer.assert_called_once()
    assert "morning-briefing" in message.answer.call_args[0][0]
