# Specification: ReAct Tools & Security Sandbox (`tools_spec.md`)

## 1. Tool Contracts & Base Classes (`core/tools/base.py`)

* `ToolDefinition`: Pydantic-схема описания инструмента (имя, описание, JSON-схема параметров).
* `Tool`: обёртка над исполняемой функцией с сигнатурой `execute(**kwargs) -> str`.

---

## 2. Tool Registry & Interception (`core/tools/registry.py`)

* `ToolRegistry` хранит зарегистрированные инструменты и автоматически инжектирует их описания в системный промпт модели.
* Автоматический перехват: каждый вызов `registry.execute(...)` оборачивается в контекстный менеджер `engine.track_tool(...)`, фиксируя входной размер, выходной размер, токенизацию и длительность исполнения.

---

## 3. Встроенные инструменты и безопасность

### 3.1. `read_file`
* Параметры: `path: str`, `offset: int = 0`, `limit: Optional[int] = None`.
* **Path Traversal Guard**:
  * Путь передается в `_resolve_safe_path(rel_path)`.
  * Выполняется проверка: `target.relative_to(self.workspace_root)`.
  * Если путь пытается выйти за пределы рабочей директории (например, `../../`), генерируется `PermissionError`.

### 3.2. `search`
* Параметры: `query: str`, `path: str = "."`.
* Поиск текстовых совпадений в файлах рабочей директории.
* Автоматический пропуск служебных каталогов: `.git`, `.venv`, `__pycache__`.
* Защита от утечки абсолютных путей хоста: пути нормализуются относительно `workspace_root`.

### 3.3. `run_tests`
* Параметры: `command: str = "pytest"`.
* **Command Injection Guard**:
  * Исключено использование `shell=True`.
  * Команда парсится через `shlex.split()`.
  * Разрешены только вызовы `pytest` или `python -m pytest`.
  * Таймаут исполнения: 30 секунд.

### 3.4. `git_diff`
* Параметры: отсутствуют.
* Возвращает вывод `git diff` для инспекции локальных незакоммиченных изменений.
