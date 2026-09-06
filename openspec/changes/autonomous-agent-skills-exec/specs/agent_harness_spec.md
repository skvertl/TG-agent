# Specification: Agent Harness, Memory & Exec Tool (`agent_harness_spec.md`)

## 1. Memory Layer: `SqliteMemoryStore` (`adapters/memory/sqlite.py`)

Реализация порта `BaseMemoryStore` для персистентного хранения истории диалога:
* База данных: `data/chat_history.db` (или настраиваемый путь через `SQLITE_MEMORY_DB_PATH`).
* Схема таблицы:
  ```sql
  CREATE TABLE IF NOT EXISTS chat_messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
  ```
* Методы:
  * `get_history(session_id: str) -> list[ChatMessage]`
  * `save_message(session_id: str, message: ChatMessage) -> None`
  * `clear(session_id: str) -> None`
  * `archive_session(session_id: str) -> str`: возвращает новый уникальный `session_id` для последующих сообщений пользователя.

---

## 2. Universal `exec` & Skill Tools (`core/tools/registry.py`)

### 2.1. `exec(command: str) -> str`
* Сигнатура: `command: str`
* Выполнение: `subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)`
* Обработка таймаута: `subprocess.TimeoutExpired` возвращает `Error: Command timed out after 30 seconds.`
* Ограничение вывода: если длина `stdout + stderr` превышает 4096 символов, результат усекается:
  `{truncated_text}\n... [Output truncated: total length was {total_len} characters]`

### 2.2. `read_skill(skill_name: str) -> str`
* Сигнатура: `skill_name: str`
* Считывает файл `skills/{skill_name}/SKILL.md` (или `skills/{skill_name}.md`).
* Возвращает текст сценария/инструкции. Если скилл не найден, возвращает перечень доступных скиллов.

---

## 3. ReAct Harness & Anti-Looping Loop (`core/runner.py`)

* По умолчанию `max_turns = 8`.
* Индексация скиллов: при инициализации `AgentRunner` сканирует каталог `skills/` и добавляет блок в системный промпт:
  ```
  Available specialized skills:
  - morning-briefing: Morning weather, schedule check, and daily briefing routine.
  - system-health: Host/container diagnostics, memory, disk, and Ollama status.
  Use 'read_skill' tool to read a skill's step-by-step instructions.
  ```
* **Anti-Looping Fallback**:
  Если на шаге `max_turns` модель всё ещё возвращает вызов `Action: ...`, цикл выполняет финальный вызов генерации с добавлением сообщения:
  `System: You have reached the maximum step limit (8 turns). Synthesize all gathered observations and provide your Final Answer now.`

---

## 4. Telegram `/new` Command (`adapters/telegram/handlers.py`)

* Команда `/new`:
  * Вызывает `memory_store.clear(session_id)` или переключает сессию.
  * Отправляет пользователю сообщение: `🧹 Начат новый диалог. Контекст очищен!`
