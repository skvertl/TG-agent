"""Unit tests for tool schema and system prompt minification."""
import json
import pytest
from unittest.mock import AsyncMock

from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.runner import AgentRunner
from core.tools.registry import ToolRegistry
from core.tools.base import Tool
from core.observability.engine import ObservabilityEngine


class TestPromptMinification:
    def test_tool_schema_minified_json(self, tmp_path):
        """Verify tool parameter schemas use compact JSON representation without whitespace."""
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))
        registry.register(
            Tool(
                name="test_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "First parameter"},
                        "param2": {"type": "integer", "description": "Second parameter"},
                    },
                    "required": ["param1"],
                },
                func=lambda param1, param2=0: "result",
            )
        )

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            tool_registry=registry,
        )

        sys_prompt = runner._build_system_prompt_with_tools()

        # In minified JSON with separators=(',', ':'), there are no spaces after ':' or ','
        test_tool_def = [d for d in registry.get_definitions() if d["name"] == "test_tool"][0]
        expected_compact_json = json.dumps(
            test_tool_def["parameters"],
            ensure_ascii=False,
            separators=(',', ':'),
        )
        assert expected_compact_json in sys_prompt
        assert f"Parameters: {expected_compact_json}" in sys_prompt
        # Confirm no spaced keys or commas in tool parameter schemas
        tool_line = [line for line in sys_prompt.splitlines() if "- test_tool:" in line][0]
        assert '": "' not in tool_line
        assert '", "' not in tool_line

    def test_skills_summary_densely_formatted(self, tmp_path):
        """Verify _get_skills_summary renders skills concisely without empty padding lines."""
        skills_dir = tmp_path / "skills"
        s1 = skills_dir / "system-health"
        s1.mkdir(parents=True, exist_ok=True)
        (s1 / "SKILL.md").write_text("description: Diagnostics routine\n# Steps", encoding="utf-8")

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            skills_dir=str(skills_dir),
        )

        summary = runner._get_skills_summary()
        assert "Skills:" in summary
        assert "- system-health: Diagnostics routine" in summary
        assert "\n\n\n" not in summary
        assert "Available specialized skills:" not in summary

    def test_system_prompt_token_reduction(self, tmp_path):
        """Verify minified system prompt character count is reduced by >= 25-30% compared to verbose baseline."""
        engine = ObservabilityEngine(project_name="test")
        registry = ToolRegistry(engine=engine, workspace_root=str(tmp_path))
        registry.register(
            Tool(
                name="read_skill",
                description="Read skill instructions",
                parameters={"type": "object", "properties": {"skill_name": {"type": "string"}}},
                func=lambda skill_name: "instructions",
            )
        )
        registry.register(
            Tool(
                name="exec",
                description="Run shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
                func=lambda command: "output",
            )
        )

        skills_dir = tmp_path / "skills"
        s1 = skills_dir / "morning-briefing"
        s1.mkdir(parents=True, exist_ok=True)
        (s1 / "SKILL.md").write_text("description: Daily morning brief", encoding="utf-8")

        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
            tool_registry=registry,
            skills_dir=str(skills_dir),
        )

        minified_prompt = runner._build_system_prompt_with_tools()

        # Baseline verbose system prompt format (with indented JSON schemas and verbose instructions)
        verbose_tools = [
            f"- {d['name']}: {d['description']}. Parameters: {json.dumps(d['parameters'], indent=2)}"
            for d in registry.get_definitions()
        ]
        verbose_prompt = (
            runner.system_prompt
            + "\n\nAvailable specialized skills:\n- morning-briefing: Daily morning brief\n"
            + "To execute a skill, use tool 'read_skill' with 'skill_name' to inspect its step-by-step instructions.\n"
            + "\n\nYou have access to the following tools to accomplish your task:\n"
            + "\n".join(verbose_tools)
            + "\n\nCRITICAL INSTRUCTIONS FOR USING TOOLS:\n"
            + "1. When a user requests a specialized skill routine (e.g. morning briefing, system health) or uses a command like /morning-briefing or /system-health, your VERY FIRST step MUST be:\n"
            + '   Action: read_skill\n'
            + '   Action Input: {"skill_name": "<skill_name>"}\n'
            + "2. ALWAYS invoke ONLY ONE action at a time. Never produce multiple Action calls in a single message.\n"
            + "3. Format each tool call strictly as:\n"
            + "Action: <tool_name>\n"
            + "Action Input: <json_arguments>\n\n"
            + "4. After outputting an Action, wait for the system to reply with 'Observation: ...' before taking your next step.\n"
            + "5. When all steps from the skill are complete, formulate your answer and output:\n"
            + "Final Answer: <your final formatted response to the user>\n\n"
            + "6. LANGUAGE MANDATE: You must formulate all output STRICTLY and EXCLUSIVELY in Russian (Cyrillic). Never output any Chinese (CJK) characters under any circumstances."
        )

        reduction = (len(verbose_prompt) - len(minified_prompt)) / len(verbose_prompt)
        assert reduction >= 0.25, f"Expected >= 25% reduction, got {reduction:.1%}"

    def test_parse_action_compatibility_with_minified_prompts(self):
        """Verify _parse_action robustly handles compact JSON inputs."""
        runner = AgentRunner(
            llm_plugin=AsyncMock(spec=BaseLLMPlugin),
            memory_store=AsyncMock(spec=BaseMemoryStore),
        )

        compact_output = 'Action: exec\nAction Input: {"command":"ls -la","timeout":30}'
        parsed = runner._parse_action(compact_output)
        assert parsed is not None
        tool_name, args = parsed
        assert tool_name == "exec"
        assert args == {"command": "ls -la", "timeout": 30}

        compact_output_2 = 'Action: read_skill\nAction Input: {"skill_name":"morning-briefing"}'
        parsed_2 = runner._parse_action(compact_output_2)
        assert parsed_2 is not None
        tool_name_2, args_2 = parsed_2
        assert tool_name_2 == "read_skill"
        assert args_2 == {"skill_name": "morning-briefing"}
