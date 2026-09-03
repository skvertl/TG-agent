# Tasks: Implementation Checklist for TG-Ollama Gateway

- [x] **Task 1: Project Scaffolding & Dependencies**
  - Создать структуру каталогов: `core/`, `core/ports/`, `adapters/`, `adapters/llm/`, `adapters/memory/`, `adapters/telegram/`, `tests/`.
  - Оформить `pyproject.toml` (или `requirements.txt`) с зависимостями: `aiogram>=3.4`, `httpx>=0.27`, `pydantic>=2.6`, `pydantic-settings>=2.2`, `pytest>=8.0`, `pytest-asyncio>=0.23`, `pytest-mock>=3.12`.
  - Настроить конфигурацию `pytest` (asyncio_mode = auto).

- [x] **Task 2: Core DTOs & Domain Exceptions**
  - Тест: `tests/unit/test_models.py` (валидация `ChatMessage`, `PromptPayload`, `AgentResult`).
  - Реализация: `core/models.py`.
  - Реализация: `core/exceptions.py` (`DomainError`, `LLMPluginError`, `LLMTimeoutError`, `LLMConnectionError`, `LLMResponseError`).

- [x] **Task 3: Core Ports & Abstract Interfaces**
  - Реализация: `core/ports/llm_plugin.py` (`BaseLLMPlugin` с методами `initialize`, `generate`, `generate_stream`, `shutdown`).
  - Реализация: `core/ports/memory_store.py` (`BaseMemoryStore` с методами `get_history`, `save_message`, `clear`).
  - Тест: `tests/unit/test_ports.py` (проверка контрактов и невозможности инстанцирования без имплементации).

- [x] **Task 4: Stateless Memory Store**
  - Тест: `tests/unit/test_stateless_memory.py` (проверка возврата пустого контекста, изоляции запросов, no-op сохранения).
  - Реализация: `adapters/memory/stateless.py` (`StatelessMemoryStore`).

- [x] **Task 5: Core Domain Orchestrator (`AgentRunner`)**
  - Тест: `tests/unit/test_runner.py` (тестирование пайплайна с mock-плагином: инъекция системного промпта, вызов генерации, изоляция слоев).
  - Реализация: `core/runner.py` (`AgentRunner`).

- [x] **Task 6: Driven Adapter — OllamaPlugin**
  - Тест: `tests/unit/test_ollama_plugin.py` (mock httpx: успешный ответ `/api/chat`, парсинг токенов, стриминг, обработка `httpx.TimeoutException` -> `LLMTimeoutError`, `httpx.ConnectError` -> `LLMConnectionError`, статус 500 -> `LLMResponseError`).
  - Реализация: `adapters/llm/ollama_plugin.py` (`OllamaPlugin`).

- [ ] **Task 7: Driving Adapter Utilities (Splitter & Typing Indicator)**
  - Тест: `tests/unit/test_splitter.py` (разбиение строк > 4000 символов по `\n\n`, `\n`, пробелам; сохранение Markdown-блоков кода ` ``` `).
  - Реализация: `adapters/telegram/splitter.py`.
  - Тест: `tests/unit/test_typing.py` (проверка интервалов `ChatAction.TYPING` и гарантированной отмены фоновой задачи в `finally`).
  - Реализация: `adapters/telegram/typing.py`.

- [ ] **Task 8: Driving Adapter — Telegram Bot & Handlers**
  - Тест: `tests/unit/test_telegram_handlers.py` (обработка `/start`, `/help`, текстовых сообщений, трансляция доменных ошибок `LLMTimeoutError`/`LLMConnectionError` в понятный UX).
  - Реализация: `adapters/telegram/handlers.py` и сборка диспетчера в `adapters/telegram/bot.py`.

- [ ] **Task 9: Application Config & Graceful Entrypoint**
  - Реализация: `config.py` (Pydantic Settings: `TELEGRAM_BOT_TOKEN`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `SYSTEM_PROMPT`).
  - Реализация: `main.py` (сборка контейнера зависимостей, запуск бота с lifespan, перехват `SIGINT`/`SIGTERM` для graceful shutdown).

- [ ] **Task 10: Infrastructure & Docker Topology**
  - Реализация: `Dockerfile` (multi-stage: builder + runner с непривилегированным пользователем `appuser:appuser`).
  - Реализация: `.dockerignore`.
  - Реализация: `docker-compose.yml` (сервис `ollama` с healthcheck и резервацией GPU, сервис `bot` с `depends_on: ollama: condition: service_healthy`, bridge-сеть `ai-network`, том `ollama_data`).
  - Реализация: `.env.example`.

- [ ] **Task 11: End-to-End Verification & Test Suite Execution**
  - Запуск полного набора unit и integration тестов через `pytest`.
  - Проверка соответствия спецификации `bot_core_spec.md`.
