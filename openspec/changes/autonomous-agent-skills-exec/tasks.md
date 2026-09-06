# Tasks: Autonomous Agent with Skills, Exec Tool & Long Context

- [x] **Task 1: Persistent SQLite Memory Layer (`SqliteMemoryStore`)**
  - [x] Test: `tests/unit/test_sqlite_memory.py` (save, get_history, clear, session isolation).
  - [x] Implementation: `adapters/memory/sqlite.py`.

- [x] **Task 2: Universal `exec` & `read_skill` Tools**
  - [x] Test: `tests/unit/test_exec_tool.py` (exec echo, timeout handling, 4KB truncation, read_skill).
  - [x] Implementation: `core/tools/registry.py`.

- [x] **Task 3: Skill Definitions (`skills/`)**
  - [x] Implementation: `skills/morning-briefing/SKILL.md`.
  - [x] Implementation: `skills/system-health/SKILL.md`.

- [x] **Task 4: Agent Harness & Anti-Looping Loop**
  - [x] Test: `tests/unit/test_agent_harness.py` (max 8 turns limit, anti-looping synthesis nudge, skill prompt injection).
  - [x] Implementation: `core/runner.py`.

- [x] **Task 5: Telegram `/new` Command & Integration**
  - [x] Test: `tests/unit/test_new_command_handler.py`.
  - [x] Implementation: `adapters/telegram/handlers.py` and `adapters/telegram/bot.py`.

- [x] **Task 6: Infrastructure & Composition Root**
  - [x] Implementation: `main.py` wired with `SqliteMemoryStore`.
  - [x] Implementation: `Dockerfile` with `curl` and `ca-certificates`.

- [x] **Task 7: Verification & Zero Documentation Drift**
  - [x] Run full test suite: `pytest` (126 passed).
  - [x] Update `README.md` and mark tasks complete in `tasks.md`.
