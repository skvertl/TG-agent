import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from core.models import AgentResult, ChatMessage, PromptPayload
from core.ports.llm_plugin import BaseLLMPlugin
from core.ports.memory_store import BaseMemoryStore
from core.runner import AgentRunner
from core.tools.registry import ToolRegistry
from core.observability.engine import ObservabilityEngine


class FakeLLM(BaseLLMPlugin):
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.received_payloads = []

    async def initialize(self) -> None:
        pass

    async def generate(self, payload: PromptPayload) -> AgentResult:
        self.received_payloads.append(payload)
        resp = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return AgentResult(content=resp, model="fake-model", prompt_tokens=10, completion_tokens=10)

    async def generate_stream(self, payload: PromptPayload):
        yield ""

    async def shutdown(self) -> None:
        pass


class FakeMemory(BaseMemoryStore):
    def __init__(self):
        self.messages = []

    async def get_history(self, session_id: str, limit: Optional[int] = None):
        if limit is not None and limit > 0:
            return list(self.messages[-limit:])
        return list(self.messages)

    async def save_message(self, session_id: str, message: ChatMessage):
        self.messages.append(message)

    async def clear(self, session_id: str):
        self.messages.clear()


class TestAgentHarness:
    @pytest.mark.asyncio
    async def test_default_max_turns_is_8(self):
        runner = AgentRunner(llm_plugin=FakeLLM(["Final Answer: Done"]), memory_store=FakeMemory())
        assert runner.max_turns == 8

    @pytest.mark.asyncio
    async def test_system_prompt_includes_skills_directory(self, tmp_path):
        skills_dir = tmp_path / "skills" / "morning-briefing"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("---\nname: morning-briefing\ndescription: Morning routine\n---\n# Routine", encoding="utf-8")

        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        runner = AgentRunner(
            llm_plugin=FakeLLM(["Final Answer: Done"]),
            memory_store=FakeMemory(),
            tool_registry=registry,
            skills_dir=str(tmp_path / "skills"),
        )

        sys_prompt = runner._build_system_prompt_with_tools()
        assert "morning-briefing" in sys_prompt
        assert "read_skill" in sys_prompt or "skills" in sys_prompt

    @pytest.mark.asyncio
    async def test_anti_looping_triggers_synthesis_at_max_turns(self, tmp_path):
        # A loop where model keeps calling tool until synthesis prompt
        looping_responses = [
            'Action: exec\nAction Input: {"command": "echo 1"}',
            'Action: exec\nAction Input: {"command": "echo 2"}',
            'Action: exec\nAction Input: {"command": "echo 3"}',
            'Final Answer: Loop stopped and synthesized result'
        ]
        llm = FakeLLM(looping_responses)
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        runner = AgentRunner(
            llm_plugin=llm,
            memory_store=FakeMemory(),
            tool_registry=registry,
            max_turns=3,  # Strict low limit for test
        )

        result = await runner.run(session_id="s1", user_prompt="Run loop")
        # Should stop at max_turns + 1 synthesis call
        assert result.content == "Loop stopped and synthesized result"
        # Check that the last payload contained a system nudge about reaching step limit
        last_payload = llm.received_payloads[-1]
        last_msg = last_payload.messages[-1]
        assert "step limit" in last_msg.content.lower() or "final answer" in last_msg.content.lower()

    @pytest.mark.asyncio
    async def test_anti_looping_fallback_when_model_fails_synthesis(self, tmp_path):
        # Even on synthesis call, model still returns Action
        looping_responses = [
            'Action: exec\nAction Input: {"command": "echo 1"}',
            'Action: exec\nAction Input: {"command": "echo 2"}',
        ]
        llm = FakeLLM(looping_responses)
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        runner = AgentRunner(
            llm_plugin=llm,
            memory_store=FakeMemory(),
            tool_registry=registry,
            max_turns=1,
        )

        result = await runner.run(session_id="s1", user_prompt="Run loop")
        assert "maximum step limit" in result.content

    @pytest.mark.asyncio
    async def test_runner_forwards_history_limit_to_memory_store(self):
        llm = FakeLLM(["Final Answer: Done"])
        mock_memory = AsyncMock(spec=BaseMemoryStore)
        mock_memory.get_history.return_value = []

        runner = AgentRunner(
            llm_plugin=llm,
            memory_store=mock_memory,
            history_limit=5,
        )

        await runner.run(session_id="session-limit-test", user_prompt="Hi")

        mock_memory.get_history.assert_awaited_once_with("session-limit-test", limit=5)
