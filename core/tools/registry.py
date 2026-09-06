import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.tools.base import Tool, ToolDefinition
from core.observability.engine import ObservabilityEngine


class ToolRegistry:
    """Реестр инструментов с автоматическим мониторингом через ObservabilityEngine."""

    def __init__(
        self,
        engine: ObservabilityEngine,
        workspace_root: Optional[str] = None,
    ):
        self.engine = engine
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def execute(
        self,
        tool_name: str,
        task_id: str,
        turn_number: int,
        arguments: Dict[str, Any],
    ) -> str:
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        # Автоматический перехват и сбор метрик через ObservabilityEngine
        with self.engine.track_tool(
            task_id=task_id,
            tool_name=tool_name,
            turn_number=turn_number,
            input_data=arguments,
        ) as span:
            try:
                output = tool.execute(**arguments)
            except Exception as e:
                output = f"Error executing '{tool_name}': {str(e)}"
            span.set_output(output)
            return output

    def _resolve_safe_path(self, rel_path: str) -> Path:
        """Ограничивает доступ только пределами рабочей директории (Path Traversal Protection)."""
        target = (self.workspace_root / rel_path).resolve()
        try:
            target.relative_to(self.workspace_root)
        except ValueError:
            raise PermissionError(f"Access denied: path '{rel_path}' resolves outside workspace root.")
        return target

    def _register_default_tools(self) -> None:
        # 1. read_file
        def _read_file(path: str, offset: int = 0, limit: Optional[int] = None) -> str:
            try:
                full_path = self._resolve_safe_path(path)
            except PermissionError as e:
                return f"Error: {str(e)}"

            if not full_path.is_file():
                return f"Error: File '{path}' not found."
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if offset > 0 or limit is not None:
                    end = offset + limit if limit is not None else len(lines)
                    selected = lines[offset:end]
                    return f"[Lines {offset+1}-{min(end, len(lines))} of {len(lines)}]:\n" + "".join(selected)
                return "".join(lines)
            except Exception as e:
                return f"Error reading file '{path}': {str(e)}"

        self.register(
            Tool(
                name="read_file",
                description="Reads content of a file. Supports optional offset and limit for line ranges.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file"},
                        "offset": {"type": "integer", "description": "Start line (0-indexed)"},
                        "limit": {"type": "integer", "description": "Max lines to read"},
                    },
                    "required": ["path"],
                },
                func=_read_file,
            )
        )

        # 2. search
        def _search(query: str, path: str = ".") -> str:
            try:
                root = self._resolve_safe_path(path)
            except PermissionError as e:
                return f"Error: {str(e)}"

            if not root.exists():
                return f"Error: Path '{path}' not found."

            matches = []
            for root_dir, _, files in os.walk(root):
                if ".git" in root_dir or ".venv" in root_dir or "__pycache__" in root_dir:
                    continue
                for file in files:
                    file_p = Path(root_dir) / file
                    try:
                        with open(file_p, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, 1):
                                if query in line:
                                    rel = file_p.relative_to(self.workspace_root)
                                    matches.append(f"{rel}:{idx}: {line.strip()[:100]}")
                    except Exception:
                        continue
            if not matches:
                return f"No matches found for query '{query}'."
            return "\n".join(matches[:50])

        self.register(
            Tool(
                name="search",
                description="Search for exact text query inside workspace files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search string"},
                        "path": {"type": "string", "description": "Subdirectory to search in"},
                    },
                    "required": ["query"],
                },
                func=_search,
            )
        )

        # 3. run_tests
        def _run_tests(command: str = "pytest") -> str:
            import shlex
            args = shlex.split(command.strip())
            if not args or args[0] not in ["pytest", "python", "pytest.exe"]:
                return "Error: Only 'pytest' or 'python -m pytest' test commands are permitted."
            if args[0] == "python" and (len(args) < 3 or args[1] != "-m" or args[2] != "pytest"):
                return "Error: Only 'python -m pytest' commands are permitted."

            try:
                proc = subprocess.run(
                    args,
                    cwd=str(self.workspace_root),
                    shell=False,  # Отключаем shell для предотвращения Command Injection
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                return output.strip() or f"Process exited with code {proc.returncode}"
            except subprocess.TimeoutExpired:
                return "Error: Test command timed out after 30 seconds."
            except Exception as e:
                return f"Error executing tests: {str(e)}"

        self.register(
            Tool(
                name="run_tests",
                description="Runs test suite (e.g. pytest) and returns output.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Test command (default: pytest)"},
                    },
                },
                func=_run_tests,
            )
        )

        # 4. git_diff
        def _git_diff() -> str:
            try:
                proc = subprocess.run(
                    ["git", "diff"],
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return proc.stdout.strip() or "No git changes."
            except Exception as e:
                return f"Error running git diff: {str(e)}"

        self.register(
            Tool(
                name="git_diff",
                description="Shows uncommitted git changes in the workspace.",
                parameters={"type": "object", "properties": {}},
                func=_git_diff,
            )
        )

        # 5. exec
        def _exec(command: str = "", **kwargs) -> str:
            cmd = command or kwargs.get("input") or kwargs.get("cmd") or ""
            if isinstance(cmd, dict):
                cmd = cmd.get("command") or cmd.get("input") or str(cmd)
            cmd = str(cmd).strip()
            if not cmd:
                return "Error: Command cannot be empty."
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.workspace_root),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = proc.stdout
                if proc.stderr:
                    output += ("\n" if output else "") + proc.stderr
                output = output.strip()
                if not output:
                    output = f"Command finished with exit code {proc.returncode} (no output)."
                elif proc.returncode != 0:
                    output += f"\n(Process exited with code {proc.returncode})"

                # Truncate output to 4096 characters to protect LLM context
                max_chars = 4096
                if len(output) > max_chars:
                    truncated = output[:max_chars]
                    output = f"{truncated}\n... [Output truncated: total length was {len(output)} characters]"

                return output
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 30 seconds."
            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.register(
            Tool(
                name="exec",
                description="Executes a shell/CLI/terminal command in the system (e.g. cURL, git, python, date) and returns output.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command line string to execute."},
                    },
                    "required": ["command"],
                },
                func=_exec,
            )
        )

        # 6. read_skill
        def _read_skill(skill_name: str = "", **kwargs) -> str:
            name = skill_name or kwargs.get("name") or kwargs.get("input") or kwargs.get("path") or ""
            if isinstance(name, dict):
                name = name.get("skill_name") or name.get("name") or str(name)
            clean_name = str(name).strip().lower()
            if not clean_name:
                return "Error: skill_name cannot be empty."
            possible_paths = [
                self.workspace_root / "skills" / clean_name / "SKILL.md",
                self.workspace_root / "skills" / f"{clean_name}.md",
                self.workspace_root / "skills" / clean_name / "skill.md",
            ]
            for p in possible_paths:
                if p.is_file():
                    try:
                        return p.read_text(encoding="utf-8")
                    except Exception as e:
                        return f"Error reading skill '{clean_name}': {str(e)}"

            available = []
            skills_base = self.workspace_root / "skills"
            if skills_base.is_dir():
                for item in skills_base.iterdir():
                    if item.is_dir() and ((item / "SKILL.md").exists() or (item / "skill.md").exists()):
                        available.append(item.name)
                    elif item.is_file() and item.suffix == ".md":
                        available.append(item.stem)

            avail_str = ", ".join(available) if available else "none"
            return f"Error: Skill '{clean_name}' not found. Available skills: {avail_str}."

        self.register(
            Tool(
                name="read_skill",
                description="Reads specialized step-by-step instructions (routines) for an agent skill from the skills directory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Name of the skill to read (e.g. 'morning-briefing', 'system-health')."},
                    },
                    "required": ["skill_name"],
                },
                func=_read_skill,
            )
        )
