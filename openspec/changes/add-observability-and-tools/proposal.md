# Proposal: Observability, FinOps Telemetry & ReAct Tooling

## 1. Overview & Objectives
Внедрение подсистемы мониторинга и аудита расхода токенов (FinOps Observability Layer), а также реестра ReAct-инструментов с автоматическим трейсингом вызовов.
Подсистема решает проблему "чёрного ящика" при работе локальных и облачных LLM-агентов:
1. Замеряет `input_tokens`, `output_tokens`, `cached_tokens`, `latency_ms`, `repeated_tokens` и финансовую стоимость выполнения задачи.
2. Позволяет инспектировать структуру расходов как через Telegram-команду `/token_report`, так и через автономный консольный TUI-дашборд (`rich`).
3. Предоставляет агенту безопасные инструменты (`read_file`, `search`, `run_tests`, `git_diff`) с защитой от Path Traversal и Command Injection.
4. Является на 100% переносимой (Zero Coupling с Telegram): пакет `core/observability/` может быть скопирован и переиспользован в любом другом Python-проекте.

---

## 2. Архитектурные границы

```
┌────────────────────────────────────────────────────────┐
│               Observability & Tools Layer              │
├───────────────────────────┬────────────────────────────┤
│ core/observability/       │ core/tools/                │
│ - models.py (DTOs/Spans)  │ - base.py (Tool/ToolDef)   │
│ - pricing.py (Tariffs/Math│ - registry.py (Security &  │
│ - overlap.py (Repetition) │   Sandbox Execution)       │
│ - storage.py (SQLite WAL) │                            │
│ - engine.py (Interceptors)│                            │
│ - presenter.py (TG & Rich)│                            │
│ - cli.py (Terminal UI)    │                            │
└───────────────────────────┴────────────────────────────┘
```

### 2.1. Изоляция и переносимость
* **Модуль `core/observability/`**:
  * Не импортирует `adapters/` или сторонние фреймворки интерфейса.
  * Работает через стандартные контекстные менеджеры: `track_llm(...)` и `track_tool(...)`.
  * Сохраняет данные в SQLite с автоматическим включением WAL-режима для устранения блокировок БД.
* **Модуль `core/tools/`**:
  * Реализует паттерн ReAct (Action / Action Input / Observation).
  * Вызовы инструментов автоматически перехватываются и регистрируются через `engine.track_tool(...)`.

---

## 3. Нефункциональные требования (NFR)
1. **Накладные расходы (Overhead)**:
   * Запись спанов в SQLite не должна увеличивать задержку вызова LLM более чем на 2-5 мс.
   * Контекст вычислений overlap (`estimate_tokens`) оптимизирован и не блокирует event loop.
2. **Безопасность песочницы**:
   * Инструменты чтения файлов изолированы в пределах `workspace_root` (Path Traversal Guard).
   * Выполнение тестов ограничено `pytest` без `shell=True` (Command Injection Guard).
3. **Отказоустойчивость**:
   * Ошибки внутри трейсинга не должны приводить к падению основного диалога пользователя с ботом.
