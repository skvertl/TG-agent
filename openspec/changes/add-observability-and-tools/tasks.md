# Tasks: Implementation Checklist for Observability, FinOps & Tools

- [x] **Task 1: Observability Core Models & Contracts**
  - Реализация `LLMCallSpan`, `ToolCallSpan`, `RunSummary`, `GlobalStats`, `TurnTimeline`, `RunTimeline` в `core/observability/models.py`.
  - Тесты: `tests/unit/test_observability_models.py`.

- [x] **Task 2: Pricing & Context Overlap Estimator**
  - Реализация формул тарификации `pricing.py` ($0.30 in / $1.20 out / $0.075 cache).
  - Реализация оценки токенов и пересечения контекста `overlap.py`.
  - Тесты: `tests/unit/test_pricing_and_overlap.py`.

- [x] **Task 3: Persistent SQLite Storage with WAL Mode**
  - Создание таблиц `llm_spans`, `tool_spans`, `run_summaries` с индексами.
  - Управление жизненным циклом соединений через `@contextmanager` с гарантированным `conn.close()`.
  - Включение `PRAGMA journal_mode=WAL;`.
  - Тесты: `tests/unit/test_telemetry_storage.py`.

- [x] **Task 4: Observability Engine Interceptor**
  - Реализация `ObservabilityEngine` с контекстными менеджерами `track_llm` и `track_tool`.
  - Тесты: `tests/unit/test_observability_engine.py`.

- [x] **Task 5: ReAct Tools & Security Sandbox**
  - Реализация `ToolRegistry`, `read_file`, `search`, `run_tests`, `git_diff`.
  - Внедрение защиты от Path Traversal (`_resolve_safe_path`) и Command Injection (`shlex.split`, whitelist).
  - Тесты: `tests/unit/test_tools.py`.

- [x] **Task 6: Multi-turn Runner ReAct Integration**
  - Обновление `AgentRunner` для поддержки парсинга `Action` / `Action Input` и перехвата телеметрии.
  - Тесты: `tests/unit/test_runner.py`.

- [x] **Task 7: Telegram `/token_report` Command**
  - Добавление форматирования глобального отчета и таймлайнов задач в `presenter.py`.
  - Подключение хендлера `/token_report` и `/token_report <task_id>` в `adapters/telegram/handlers.py`.
  - Тесты: `tests/unit/test_token_report_handler.py`.

- [x] **Task 8: Rich CLI Dashboard**
  - Реализация `core/observability/cli.py` для терминального мониторинга.
  - Тесты: `tests/unit/test_presenter.py`.

- [x] **Task 9: Docker & Infrastructure Hardening**
  - Монтирование volume `./data:/app/data` для сохранения телеметрии в контейнере.
  - Rootless `appuser`, `cap_drop: ALL`, `no-new-privileges: true`.

- [x] **Task 10: Zero Documentation Drift Sync**
  - Создание постоянных инструкций агента в `AGENTS.md`.
  - Актуализация архитектурной схемы и описания в `README.md`.
  - Фиксация спецификаций и задач в `openspec/changes/add-observability-and-tools/`.
