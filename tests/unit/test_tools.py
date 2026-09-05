import pytest
from core.tools.registry import ToolRegistry
from core.observability.engine import ObservabilityEngine


class TestToolRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        engine = ObservabilityEngine(project_name="test-tools")
        reg = ToolRegistry(engine=engine, workspace_root=str(tmp_path))
        # Create a sample file
        sample_file = tmp_path / "hello.py"
        sample_file.write_text("def test_func():\n    return 42\n")
        return reg

    def test_read_file(self, registry):
        result = registry.execute(
            tool_name="read_file",
            task_id="t-1",
            turn_number=1,
            arguments={"path": "hello.py"},
        )
        assert "def test_func" in result
        assert "return 42" in result

    def test_read_file_not_found(self, registry):
        result = registry.execute(
            tool_name="read_file",
            task_id="t-1",
            turn_number=1,
            arguments={"path": "non_existent.py"},
        )
        assert "Error" in result or "not found" in result.lower()

    def test_search_file(self, registry):
        result = registry.execute(
            tool_name="search",
            task_id="t-1",
            turn_number=1,
            arguments={"query": "test_func"},
        )
        assert "hello.py" in result

    def test_get_tool_definitions(self, registry):
        defs = registry.get_definitions()
        assert len(defs) >= 3
        names = [d["name"] for d in defs]
        assert "read_file" in names
        assert "search" in names
        assert "run_tests" in names
