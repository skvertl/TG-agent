# Specification: Observability & FinOps Engine (`observability_spec.md`)

## 1. Data Models & Spans (`core/observability/models.py`)

### 1.1. `LLMCallSpan`
Фиксирует метрики отдельного обращения к языковой модели:
* `id`: уникальный UUID спана.
* `timestamp`: время вызова (ISO 8601).
* `project_name`: имя проекта для мультитенантности.
* `task_id`: идентификатор родительской задачи.
* `model`: название модели (например, `qwen2.5:7b`).
* `turn_number`: номер шага в ReAct-цикле.
* `input_tokens`, `output_tokens`, `cached_tokens`, `reasoning_tokens`.
* `latency_ms`: длительность вызова в миллисекундах.
* `estimated_cost`: рассчитанная стоимость обращения в USD.

### 1.2. `ToolCallSpan`
Фиксирует метрики выполнения инструмента:
* `id`, `timestamp`, `task_id`, `turn_number`.
* `tool_name`: имя вызванного инструмента.
* `input_size`: размер входных аргументов в байтах.
* `output_size`: размер результата инструмента в байтах.
* `output_tokens`: примерный эквивалент вывода инструмента в токенах.
* `duration_ms`: время выполнения инструмента.

### 1.3. `RunSummary` & `GlobalStats`
* Агрегированная сводка по всей задаче: общее число токенов, повторные токены (`repeated_tokens`), число шагов, статус завершения и общая стоимость.
* Глобальная аналитика: Cache Hit Rate %, процент повторного контекста, распределение долей использования инструментов.

---

## 2. Формула тарификации и ценообразование (`core/observability/pricing.py`)

Тарифы рассчитываются на базе индустриальных ставок за 1 миллион токенов:
* **`qwen2.5:7b`**: Input: \$0.30 / 1M | Output: \$1.20 / 1M | Cache: \$0.075 / 1M.
* **Формула**:
  $$\text{Cost} = \frac{\text{Input} \times 0.30 + \text{Output} \times 1.20 + \text{Cache} \times 0.075}{1\,000\,000}$$

---

## 3. Анализ повторного контекста (`core/observability/overlap.py`)

* Оценка токенов: `estimate_tokens(text)` на базе эвристики 1 токен $\approx$ 3.5 символа для мультиязычного/русского и кодового текста.
* Расчёт повторов: `calculate_context_overlap(prev_messages, current_messages)` выявляет токены предыдущих шагов диалога, повторно переданные модели в промпте.

---

## 4. Персистентное хранилище (`core/observability/storage.py`)

* СУБД: **SQLite** с файлом по умолчанию `data/telemetry.db`.
* Режим журнала: **WAL (`PRAGMA journal_mode=WAL;`)** для поддержки параллельного чтения и записи.
* Управление соединениями: контекстный менеджер `@contextmanager` с гарантированным `conn.close()` в блоке `finally`.
* Индексы: `idx_llm_task_id`, `idx_tool_task_id`, `idx_run_timestamp`.

---

## 5. Презентация и CLI (`core/observability/presenter.py`, `cli.py`)

* **Telegram Presenter**:
  * Форматирование компактного Markdown-отчета для Telegram.
  * Визуализация распределения инструментов символами прогресс-бара (`█████`).
  * Детальный таймлайн выполнения по шагам: `Turn 1: LLM Call ... | Tool: read_file ...`.
* **CLI Terminal UI**:
  * Использование библиотеки `rich` для рендеринга таблиц в терминале.
  * Команды: `python -m core.observability.cli --limit <N> --task-id <id>`.
