# Extensible Async Telegram-Ollama Gateway (Stateless)

Асинхронный Telegram-шлюз на Python 3.11+ с плагинным инференсом через локальные модели Ollama (`qwen2.5:1.5b`, `tinyllama` и др.), функционирующий в строго **Stateless-режиме** (одноразовый контекст без сохранения истории между сессиями).

---

## 🏛 Архитектура: Hexagonal (Ports & Adapters)

Система строго разделена на три слоя с направлением зависимостей снаружи внутрь:

```
[ Telegram (aiogram 3.x) ]  -->  (Driving Port: AgentRunner)
                                          │
                                          ▼
                                   [ Domain Core ]
                           (DTOs, Rules, Domain Exceptions)
                                          │
                                          ▼
[ Ollama API (httpx Async) ] <-- (Driven Port: BaseLLMPlugin)
[ Stateless Memory Store   ] <-- (Driven Port: BaseMemoryStore)
```

1. **Доменное ядро (`core/`)**:
   - Полная независимость от фреймворков и сторонних SDK.
   - Pydantic DTO: `ChatMessage`, `PromptPayload`, `AgentResult`.
   - Абстрактные порты: `BaseLLMPlugin`, `BaseMemoryStore`.
   - Изоляция исключений: `DomainError`, `LLMPluginError`, `LLMTimeoutError`, `LLMConnectionError`, `LLMResponseError`.
   - Оркестратор: `AgentRunner`.

2. **Driven-адаптеры (`adapters/`)**:
   - `OllamaPlugin` (`adapters/llm/ollama_plugin.py`): асинхронный HTTP-клиент к Ollama API (`/api/chat`), транслирующий любые сетевые и протокольные сбои в доменные исключения.
   - `StatelessMemoryStore` (`adapters/memory/stateless.py`): гарантирует изоляцию запросов (история не сохраняется между вызовами).

3. **Driving-адаптер (`adapters/telegram/`)**:
   - `TelegramBotAdapter` на базе `aiogram 3.x`.
   - `AuthMiddleware` (`middlewares.py`): приватный белый список доступа по `ALLOWED_USER_ID` / `ALLOWED_USER_IDS` (неавторизованные пользователи блокируются).
   - Безопасный сплиттер (`splitter.py`) длинных сообщений (> 4000 символов) с сохранением блоков кода Markdown (` ``` `).
   - Менеджер периодического статуса «печатает...» (`keep_typing`) с гарантированной отменой фоновой корутины.

## 🔐 Безопасность и Docker Secrets (Zero-Knowledge)

Все секреты передаются **исключительно через Docker Secrets** в виртуальную память (`tmpfs`) по пути `/run/secrets/`.
- ❌ **Секреты НЕ передаются через переменные окружения** (их невозможно увидеть через `docker inspect` или `/proc/1/environ`).
- ❌ **Секреты НЕ попадут в git** (`secrets/*.txt` добавлены в `.gitignore`).

### 1. Подготовка секретов
Создайте файлы секретов из примеров:
```bash
cp secrets/telegram_bot_token.txt.example secrets/telegram_bot_token.txt
cp secrets/allowed_user_ids.txt.example secrets/allowed_user_ids.txt
```
1. В `secrets/telegram_bot_token.txt` вставьте токен вашего бота (от `@BotFather`).
2. В `secrets/allowed_user_ids.txt` укажите ваш Telegram User ID.

*(Для локальной разработки без Docker также поддерживается стандартный `.env` файл из `.env.example`).*

### 2. Запуск контейнеров
```bash
docker compose up -d --build
```
Compose автоматически:
1. Запустит сервис `ollama` с изолированной bridge-сетью `ai-network` и пробросом GPU (NVIDIA Container Toolkit).
2. Выполнит healthcheck доступности Ollama.
3. Соберет multi-stage образ бота под непривилегированным пользователем `appuser` (UID 10001) и запустит шлюз.

### 3. Загрузка модели в Ollama
Если модель еще не загружена в локальный volume `ollama_data`:
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```

---

## 🛠 Локальная разработка и тестирование

### Установка окружения (uv / pip)
```bash
uv venv --python 3.11
source .venv/bin/activate  # на Linux/macOS
# или: .venv\Scripts\activate  # на Windows

uv pip install -r requirements-dev.txt
```

### Запуск тестов
Проект разработан строго по методологии TDD:
```bash
pytest -v
```
Все 72 теста (юнит-тесты ядра, адаптеров и сквозные интеграционные тесты) проверяют контракты, таймауты, чанкинг и изоляцию сессий.
