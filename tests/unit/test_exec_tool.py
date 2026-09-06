import sys
import pytest
from pathlib import Path
from core.observability.engine import ObservabilityEngine
from core.tools.registry import ToolRegistry

PYTHON_EXE = f'"{sys.executable}"'


class TestExecAndSkillTools:
    @pytest.fixture
    def registry(self, tmp_path):
        engine = ObservabilityEngine(project_name="test")
        return ToolRegistry(engine=engine, workspace_root=str(tmp_path))

    def test_exec_echo_command(self, registry):
        # Universal exec should run command and capture output
        res = registry.execute(
            tool_name="exec",
            task_id="task-1",
            turn_number=1,
            arguments={"command": f"{PYTHON_EXE} -c \"print('Hello Exec')\""}
        )
        assert "Hello Exec" in res

    def test_exec_handles_non_zero_exit_code(self, registry):
        res = registry.execute(
            tool_name="exec",
            task_id="task-1",
            turn_number=1,
            arguments={"command": f"{PYTHON_EXE} -c \"import sys; sys.exit(42)\""}
        )
        assert "exited with code 42" in res or "42" in res

    def test_exec_timeout_handling(self, registry, monkeypatch):
        import subprocess

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="sleep 100", timeout=30)

        monkeypatch.setattr(subprocess, "run", mock_run)

        res = registry.execute(
            tool_name="exec",
            task_id="task-1",
            turn_number=1,
            arguments={"command": "sleep 100"}
        )
        assert "timed out after 30 seconds" in res

    def test_exec_truncates_large_output(self, registry):
        # Generate 10,000 characters
        cmd = f"{PYTHON_EXE} -c \"print('A' * 10000)\""
        res = registry.execute(
            tool_name="exec",
            task_id="task-1",
            turn_number=1,
            arguments={"command": cmd}
        )
        assert len(res) < 5000
        assert "[Output truncated" in res

    def test_read_skill_success(self, registry, tmp_path):
        skills_dir = tmp_path / "skills" / "test-routine"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("# Test Routine\nStep 1: do this", encoding="utf-8")

        res = registry.execute(
            tool_name="read_skill",
            task_id="task-1",
            turn_number=1,
            arguments={"skill_name": "test-routine"}
        )
        assert "# Test Routine" in res
        assert "Step 1: do this" in res

    def test_read_skill_not_found(self, registry):
        res = registry.execute(
            tool_name="read_skill",
            task_id="task-1",
            turn_number=1,
            arguments={"skill_name": "unknown-skill"}
        )
        assert "Error: Skill 'unknown-skill' not found" in res
