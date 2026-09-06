# Tasks: Phase 2 — Token Cost & Context Optimization

- [x] **Task 1: TDD for Observation Compactor in `core/runner.py`**
  - [x] Write unit tests in `tests/unit/test_observation_compactor.py`:
    - Test `compact_observation` truncates strings exceeding threshold (e.g. 250 chars) and appends truncation marker.
    - Test `compact_observation` preserves strings shorter than or equal to threshold.
    - Test multi-turn ReAct compaction: turn 1 has no compaction; turn 2 keeps observation 1 in full; turn 3 compacts observation 1 while keeping observation 2 in full; turn 4 compacts observations 1 and 2 while keeping observation 3 in full.
    - Test that `AgentRunner._prepare_turn_messages` correctly passes compacted messages to `llm_plugin.generate` and `observability_engine.track_llm`.
  - [x] Implement `compact_observation` and `_prepare_turn_messages` in `core/runner.py`.
  - [x] Add `max_observation_chars: int = 250` parameter to `AgentRunner.__init__`.
  - [x] Run `pytest tests/unit/test_observation_compactor.py` and verify all tests pass.

- [x] **Task 2: TDD for Dialogue History Sliding Window in `core/runner.py` and `adapters/memory/sqlite.py`**
  - [x] Write unit tests in `tests/unit/test_sqlite_memory.py`:
    - Test `get_history(session_id, limit=None)` preserves all messages (backward compatibility).
    - Test `get_history(session_id, limit=N)` returns at most the latest N messages in chronological (ascending) order.
    - Test `get_history` when message count is less than `limit`.
    - Test session isolation with `limit`.
  - [x] Update `core/ports/memory_store.py` interface: `get_history(session_id: str, limit: Optional[int] = None) -> list[ChatMessage]`.
  - [x] Update `adapters/memory/sqlite.py` implementing subquery-based reverse limit (`ORDER BY id DESC LIMIT ?` outer `ORDER BY id ASC`).
  - [x] Update `adapters/memory/stateless.py` signature with `limit: Optional[int] = None`.
  - [x] Add `history_limit: Optional[int] = 10` parameter to `AgentRunner.__init__` and pass `limit=self.history_limit` in `AgentRunner.run`.
  - [x] Add unit test in `tests/unit/test_agent_harness.py` verifying `history_limit` is forwarded to memory store.
  - [x] Run `pytest tests/unit/test_sqlite_memory.py tests/unit/test_agent_harness.py` and verify all tests pass.

- [x] **Task 3: TDD for Tool Schema & System Prompt Minification in `core/runner.py`**
  - [x] Write unit tests in `tests/unit/test_prompt_minification.py`:
    - Test that `_build_system_prompt_with_tools` outputs compact JSON parameter schemas (`separators=(',', ':')`).
    - Test that skill summaries in `_get_skills_summary` are densely formatted without redundant whitespace.
    - Test that directive instructions are concise and reduce prompt size by >= 30% compared to baseline.
    - Test that `_parse_action` correctly parses tool calls from model outputs using minified format.
  - [x] Implement minification in `_build_system_prompt_with_tools` and `_get_skills_summary` in `core/runner.py`.
  - [x] Run `pytest tests/unit/test_prompt_minification.py` and verify all tests pass.

- [x] **Task 4: Re-run 20 benchmark tasks with `scripts/run_benchmarks.py`, verify >= 30% token reduction (<= 66,770 tokens) and 100% success rate, update `data/benchmark_optimized.json` and `BENCHMARK_REPORT.md`**
  - [x] Update `scripts/run_benchmarks.py` to support saving output to `data/benchmark_optimized.json` and generating comparative metrics.
  - [x] Execute the 20 benchmark tasks using `scripts/run_benchmarks.py`.
  - [x] Verify acceptance criteria:
    - Total token volume <= 66,770 tokens (>= 30% savings relative to 95,386 baseline: achieved 59,129 tokens, -38.0%).
    - Success rate = 100% (20/20 tasks passing without errors).
    - Total cost <= $0.02340 USD (achieved $0.02059).
  - [x] Update `data/benchmark_optimized.json` with execution telemetry.
  - [x] Update `BENCHMARK_REPORT.md` with side-by-side comparative table (Baseline vs. Phase 2 Optimized).

- [x] **Task 5: Zero Documentation Drift: synchronize `README.md` and rebuild Docker container**
  - [x] Update `README.md` to document Phase 2 optimizations:
    - Observation Compactor architecture.
    - Dialogue History Sliding Window and configuration (`history_limit`).
    - Minified tool schema format.
    - Updated FinOps token metrics and benchmark results.
  - [x] Run complete test suite: `pytest` (ensure 100% pass rate across unit and integration tests: 154 passed).
  - [x] Rebuild Docker container image to verify packaging integrity.
  - [x] Mark all task checkboxes in `openspec/changes/token-optimization-compaction/tasks.md`.
