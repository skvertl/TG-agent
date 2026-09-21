# Tasks: Automated Testing Suite, Red Teaming & LLM Evaluation

- [x] **Task 1.1: TDD for Input Sanitization (`tests/unit/test_input_sanitization.py`)**
  - [x] Create `tests/unit/test_input_sanitization.py` with unit tests for:
    - Empty string `""` input handling without Pydantic validation failure.
    - Whitespace-only `"   \n\t  "` input short-circuiting to friendly guidance message.
    - Length exceeding 4096 characters truncation with warning notice.
    - Null bytes (`\x00`) and dangerous non-printable control characters stripping.
    - Preserving normal newlines, tabs, and unescaped markdown.
  - [x] Implement `sanitize_input(user_prompt: str, max_chars: int = 4096) -> str` in `core/runner.py`.
  - [x] Integrate input validation in `AgentRunner.run()` to return immediate `AgentResult` for empty queries.
  - [x] Run `pytest tests/unit/test_input_sanitization.py` and verify all tests pass.

- [x] **Task 1.2: TDD for Tool Calling Contracts (`tests/unit/test_tool_calling_contracts.py`)**
  - [x] Create `tests/unit/test_tool_calling_contracts.py` with unit tests for:
    - Schema validation: verifying that missing required arguments trigger structured error messages.
    - Graceful recovery from malformed JSON in `Action Input:` (unquoted keys, single quotes, trailing commas).
    - Feedback to the ReAct loop when JSON decoding fails, allowing the agent to self-correct on the next turn.
    - Nonexistent tool call handling: returns available tools without raising unhandled exceptions.
  - [x] Implement argument schema validation in `ToolRegistry.execute()` in `core/tools/registry.py`.
  - [x] Implement JSON repair and error feedback handling in `AgentRunner._parse_action()` in `core/runner.py`.
  - [x] Run `pytest tests/unit/test_tool_calling_contracts.py` and verify all tests pass.

- [x] **Task 2.1: Red Teaming & Behavioral Dataset (`evaluation/test_dataset.json`)**
  - [x] Create `evaluation/test_dataset.json` containing 15 structured test cases:
    - 3 Jailbreaks: `rt-01` (system prompt extraction), `rt-02` (DAN mode), `rt-03` (sudo privilege escalation).
    - 2 Prompt Injections: `rt-04` (base64 encoded instruction), `rt-05` (simulated tool observation).
    - 2 Persona Breakout: `rt-06` (foreign language forcing), `rt-07` (rogue roleplay).
    - 4 Hallucination Refusals: `rt-08` (nonexistent policy), `rt-09` (private credentials), `rt-10` (future event), `rt-11` (nonexistent tool).
    - 4 Multi-Turn Memory: `rt-12` (name retention), `rt-13` (context dependency), `rt-14` (session wipe via `/new`), `rt-15` (multi-tenant session isolation).
  - [x] Validate JSON syntax and structure against specification schema.

- [x] **Task 2.2: TDD for Behavioral & Red Teaming Tests**
  - [x] Create `tests/behavioral/test_red_teaming.py`:
    - Tests for jailbreaks (`rt-01` to `rt-03`), prompt injections (`rt-04`, `rt-05`), and persona breakout (`rt-06`, `rt-07`).
    - Parameterize tests for both `qwen2.5:7b` and `llama3.2:latest`.
  - [x] Create `tests/behavioral/test_hallucinations.py`:
    - Tests for hallucination refusal (`rt-08` to `rt-11`), asserting refusal language or honest lack of knowledge.
  - [x] Create `tests/behavioral/test_multi_turn_memory.py`:
    - Tests for multi-turn retention (`rt-12`, `rt-13`), session wipe on `/new` or `clear` (`rt-14`), and session isolation (`rt-15`).
  - [x] Run behavioral test suite: `pytest tests/behavioral/`.

- [x] **Task 3.1: TDD for LLM-as-a-Judge (`evaluation/llm_judge.py` and `tests/evaluation/test_llm_judge.py`)**
  - [x] Implement `LLMJudge` and `EvaluationScore` models in `evaluation/llm_judge.py`:
    - Scoring dimensions: Politeness (weight 0.2), Accuracy (weight 0.5), Conciseness (weight 0.3).
    - Composite formula: $0.2 \times P + 0.5 \times A + 0.3 \times C$.
    - Pass criteria check: $\ge 0.8$.
    - Robust JSON extraction from model judge responses.
  - [x] Implement unit tests in `tests/evaluation/test_llm_judge.py` testing score calculation, prompt generation, and edge cases.
  - [x] Run `pytest tests/evaluation/test_llm_judge.py` and verify all tests pass.

- [x] **Task 3.2: Latency SLA Benchmark & Model Comparison Script (`scripts/run_model_comparison.py`)**
  - [x] Implement `scripts/run_model_comparison.py`:
    - Runs identical 15 test cases against `qwen2.5:7b` and `llama3.2:latest`.
    - Measures Time-To-First-Token (TTFT < 1.5s SLA) and total end-to-end latency (< 4.0s SLA).
    - Evaluates outputs using `LLMJudge`.
    - Aggregates metrics (pass rate, latency percentiles, judge scores, token throughput).
    - Outputs `evaluation/MODEL_COMPARISON_REPORT.md` with comparative Markdown tables and graphs.
  - [x] Execute `python scripts/run_model_comparison.py` and verify report generation.

- [x] **Task 4: Zero Documentation Drift: Update `README.md` & Full Suite Verification**
  - [x] Update `README.md` to document:
    - Level 1-3 testing architecture and methodology.
    - Red teaming dataset and safety guardrails.
    - LLM-as-a-Judge evaluation rubric and SLA criteria.
    - Instructions to execute `pytest tests/behavioral/` and `python scripts/run_model_comparison.py`.
  - [x] Run full test suite: `pytest` across all test directories (assert 100% pass rate).
  - [x] Mark all task checkboxes complete in `openspec/changes/automated-testing-and-red-teaming/tasks.md`.

