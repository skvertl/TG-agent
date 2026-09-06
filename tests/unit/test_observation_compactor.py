"""Unit tests for observation compactor and multi-turn ReAct message preparation."""
from unittest.mock import AsyncMock
import pytest

from core.models import ChatMessage, PromptPayload, AgentResult
from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.runner import AgentRunner, compact_observation
from core.tools.registry import ToolRegistry
from core.tools.base import Tool
from core.observability.engine import ObservabilityEngine
from core.observability.storage import TelemetryStorage


class TestObservationCompactor:
    def test_compact_observation_truncates_long_string(self):
        """Test compact_observation truncates strings exceeding threshold and appends marker."""
        long_str = "x" * 300
        result = compact_observation(long_str, max_chars=250)
        expected_prefix = "x" * 250
        expected_marker = "\n... [Observation compacted: 300 -> 250 chars]"
        assert result == expected_prefix + expected_marker

    def test_compact_observation_preserves_short_string(self):
        """Test compact_observation preserves strings shorter than or equal to threshold."""
        short_str = "Status: OK. Service is healthy."
        assert compact_observation(short_str, max_chars=250) == short_str

    def test_compact_observation_exact_threshold(self):
        """Test compact_observation preserves strings exactly equal to threshold."""
        exact_str = "y" * 250
        assert compact_observation(exact_str, max_chars=250) == exact_str

    def test_compact_observation_preserves_observation_prefix(self):
        """Test compact_observation preserves leading 'Observation: ' prefix if present."""
        long_obs = "Observation: " + "z" * 300
        result = compact_observation(long_obs, max_chars=250)
        assert result.startswith("Observation: " + "z" * 250)
        assert result.endswith("\n... [Observation compacted: 300 -> 250 chars]")

    def test_compact_observation_supports_parameter_names(self):
        """Test compact_observation supports both 'content' and 'observation' parameter names."""
        raw_text = "content_param " * 25  # ~350 chars
        # Keyword 'content'
        res_content = compact_observation(content=raw_text, max_chars=250)
        assert "... [Observation compacted:" in res_content

        # Keyword 'observation'
        res_obs = compact_observation(observation=raw_text, max_chars=250)
        assert "... [Observation compacted:" in res_obs

        # Positional
        res_pos = compact_observation(raw_text, 250)
        assert "... [Observation compacted:" in res_pos

    def test_compact_observation_boundary_trimming(self):
        """Test compact_observation trims at word or newline boundary when feasible (> 80% threshold)."""
        # Word boundary: 50 words of 5 chars each + space = 300 chars
        # Threshold: 250 chars. 80% threshold is 200 chars.
        # body[:250] falls inside the 42nd word: "alpha " * 41 = 246 chars, then "alph"
        wordy_text = "alpha " * 50
        result_word = compact_observation(wordy_text, max_chars=250)
        body_part = result_word.split("\n... [Observation compacted:")[0]
        # Should trim to the end of the last complete word (245 chars) instead of breaking mid-word
        assert body_part.endswith("alpha")
        assert not body_part.endswith("alph")

        # Newline boundary: lines of 20 chars
        lines_text = "Status check line.\n" * 16  # 304 chars
        result_lines = compact_observation(lines_text, max_chars=250)
        body_lines = result_lines.split("\n... [Observation compacted:")[0]
        assert body_lines.endswith("Status check line.")

    def test_multi_turn_react_compaction_progression(self):
        """
        Test multi-turn ReAct compaction:
        - Turn 1: no compaction (empty turn_history).
        - Turn 2: keeps observation 1 in full.
        - Turn 3: compacts observation 1 while keeping observation 2 in full.
        - Turn 4: compacts observations 1 and 2 while keeping observation 3 in full.
        """
        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            max_observation_chars=250,
        )

        obs1 = "Obs1: " + "a" * 300
        obs2 = "Obs2: " + "b" * 300
        obs3 = "Obs3: " + "c" * 300

        # Turn 1: 0 completed turns -> no tool observations
        messages_turn_1 = runner._prepare_turn_messages(
            system_prompt="System Prompt",
            user_prompt="User Query",
            turn_history=[],
        )
        assert len(messages_turn_1) == 2
        assert messages_turn_1[0] == ChatMessage(role="system", content="System Prompt")
        assert messages_turn_1[1] == ChatMessage(role="user", content="User Query")

        # Turn 2: 1 completed turn -> observation 1 in full
        turn_history_2 = [
            {"action": "Action: tool_1\nAction Input: {}", "observation": obs1}
        ]
        messages_turn_2 = runner._prepare_turn_messages(
            system_prompt="System Prompt",
            user_prompt="User Query",
            turn_history=turn_history_2,
        )
        assert len(messages_turn_2) == 4
        assert messages_turn_2[2] == ChatMessage(role="assistant", content="Action: tool_1\nAction Input: {}")
        assert messages_turn_2[3] == ChatMessage(role="user", content=f"Observation: {obs1}")

        # Turn 3: 2 completed turns -> observation 1 compacted, observation 2 in full
        turn_history_3 = [
            {"action": "Action: tool_1\nAction Input: {}", "observation": obs1},
            {"action": "Action: tool_2\nAction Input: {}", "observation": obs2},
        ]
        messages_turn_3 = runner._prepare_turn_messages(
            system_prompt="System Prompt",
            user_prompt="User Query",
            turn_history=turn_history_3,
        )
        assert len(messages_turn_3) == 6
        expected_compacted_obs1 = compact_observation(obs1, max_chars=250)
        assert messages_turn_3[3] == ChatMessage(role="user", content=f"Observation: {expected_compacted_obs1}")
        assert "... [Observation compacted: 306 -> 250 chars]" in messages_turn_3[3].content
        assert messages_turn_3[5] == ChatMessage(role="user", content=f"Observation: {obs2}")

        # Turn 4: 3 completed turns -> observations 1 & 2 compacted, observation 3 in full
        turn_history_4 = [
            {"action": "Action: tool_1\nAction Input: {}", "observation": obs1},
            {"action": "Action: tool_2\nAction Input: {}", "observation": obs2},
            {"action": "Action: tool_3\nAction Input: {}", "observation": obs3},
        ]
        messages_turn_4 = runner._prepare_turn_messages(
            system_prompt="System Prompt",
            user_prompt="User Query",
            turn_history=turn_history_4,
        )
        assert len(messages_turn_4) == 8
        expected_compacted_obs2 = compact_observation(obs2, max_chars=250)
        assert messages_turn_4[3] == ChatMessage(role="user", content=f"Observation: {expected_compacted_obs1}")
        assert messages_turn_4[5] == ChatMessage(role="user", content=f"Observation: {expected_compacted_obs2}")
        assert "... [Observation compacted: 306 -> 250 chars]" in messages_turn_4[5].content
        assert messages_turn_4[7] == ChatMessage(role="user", content=f"Observation: {obs3}")

    @pytest.mark.asyncio
    async def test_runner_run_passes_compacted_messages_to_llm_and_observability(self, tmp_path):
        """
        Verify that AgentRunner.run passes compacted messages to both llm_plugin.generate
        and observability_engine.track_llm across multiple ReAct turns.
        """
        storage = TelemetryStorage(db_path=str(tmp_path / "compaction_obs.db"))
        engine = ObservabilityEngine(project_name="test-compaction", storage=storage)
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))

        long_output_1 = "DATA_OUTPUT_1: " + ("x" * 400)
        long_output_2 = "DATA_OUTPUT_2: " + ("y" * 400)

        call_count = 0
        def multi_tool(step: str = "1") -> str:
            nonlocal call_count
            call_count += 1
            if step == "1":
                return long_output_1
            return long_output_2

        registry.register(
            Tool(
                name="multi_tool",
                description="Test tool returning long output",
                parameters={"type": "object", "properties": {"step": {"type": "string"}}},
                func=multi_tool,
            )
        )

        mock_plugin = AsyncMock(spec=BaseLLMPlugin)
        mock_memory = AsyncMock(spec=BaseMemoryStore)
        mock_memory.get_history.return_value = []

        # Turn 1: calls tool step 1
        # Turn 2: calls tool step 2
        # Turn 3: outputs Final Answer
        mock_plugin.generate.side_effect = [
            AgentResult(
                content='Action: multi_tool\nAction Input: {"step": "1"}',
                model="test-llm",
                prompt_tokens=40,
                completion_tokens=15,
            ),
            AgentResult(
                content='Action: multi_tool\nAction Input: {"step": "2"}',
                model="test-llm",
                prompt_tokens=100,
                completion_tokens=15,
            ),
            AgentResult(
                content="Final Answer: Finished analyzing multi-step data.",
                model="test-llm",
                prompt_tokens=150,
                completion_tokens=20,
            ),
        ]

        # Spy on engine.track_llm to capture messages passed to observability
        track_llm_calls = []
        original_track_llm = engine.track_llm

        def track_llm_spy(*args, **kwargs):
            track_llm_calls.append(kwargs)
            return original_track_llm(*args, **kwargs)

        engine.track_llm = track_llm_spy

        runner = AgentRunner(
            llm_plugin=mock_plugin,
            memory_store=mock_memory,
            observability_engine=engine,
            tool_registry=registry,
            max_observation_chars=250,
        )

        result = await runner.run(session_id="session-compaction", user_prompt="Analyze step data")

        assert result.content == "Finished analyzing multi-step data."
        assert mock_plugin.generate.await_count == 3
        assert call_count == 2

        # Check Turn 3 payload passed to llm_plugin.generate
        turn_3_payload: PromptPayload = mock_plugin.generate.call_args_list[2][0][0]
        turn_3_messages = turn_3_payload.messages

        # Find observations in Turn 3 messages
        obs_messages = [m for m in turn_3_messages if m.role == "user" and m.content.startswith("Observation:")]
        assert len(obs_messages) == 2

        # Observation 1 MUST be compacted
        assert "... [Observation compacted:" in obs_messages[0].content
        assert len(obs_messages[0].content) < len(long_output_1)

        # Observation 2 (immediate previous turn) MUST NOT be compacted
        assert long_output_2 in obs_messages[1].content
        assert "... [Observation compacted:" not in obs_messages[1].content

        # Verify observability track_llm received the exact compacted message content on Turn 3
        assert len(track_llm_calls) == 3
        turn_3_obs_messages = track_llm_calls[2].get("messages", [])
        turn_3_obs_content = " ".join(turn_3_obs_messages)
        assert "... [Observation compacted:" in turn_3_obs_content
        assert long_output_2 in turn_3_obs_content

        # Verify telemetry run timeline summary
        timeline = storage.get_run_timeline(result.task_id)
        assert timeline.summary.turns_count == 3
        assert timeline.summary.tool_calls_count == 2
