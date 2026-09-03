# Specification: Bot Core, Contracts & Ports (`bot_core_spec.md`)

## 1. Domain Models & DTOs (`core/models.py`)

Все DTO реализуются как иммутабельные (или строго валидируемые) модели Pydantic V2.

```python
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

RoleType = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    """Единица сообщения в диалоге."""
    role: RoleType
    content: str = Field(..., min_length=1)

class PromptPayload(BaseModel):
    """Полезная нагрузка, передаваемая в порт LLM."""
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    options: Dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    """Результат работы агента / плагина инференса."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
```

---

## 2. Domain Exceptions (`core/exceptions.py`)

Иерархия исключений, предотвращающая утечку деталей сторонних SDK в домен:

```python
class DomainError(Exception):
    """Базовое доменное исключение."""
    pass

class LLMPluginError(DomainError):
    """Ошибка провайдера LLM инференса."""
    pass

class LLMConnectionError(LLMPluginError):
    """Сбой соединения с бэкендом инференса (Ollama)."""
    pass

class LLMTimeoutError(LLMPluginError):
    """Превышение таймаута ожидания генерации ответа."""
    pass

class LLMResponseError(LLMPluginError):
    """Некорректный ответ или ошибка статуса от LLM."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
```

---

## 3. Driven Ports (Выходные интерфейсы)

### 3.1. `BaseLLMPlugin` (`core/ports/llm_plugin.py`)

Абстрактный контракт для любого бэкенда инференса (Ollama, vLLM, Mock, OpenAI API):

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from core.models import PromptPayload, AgentResult

class BaseLLMPlugin(ABC):
    """Абстрактный порт плагина LLM-инференса."""

    @abstractmethod
    async def initialize(self) -> None:
        """Инициализация ресурсов плагина (пул соединений, проверка доступности модели)."""
        pass

    @abstractmethod
    async def generate(self, payload: PromptPayload) -> AgentResult:
        """Синхронная (пакетная) генерация ответа на промпт."""
        pass

    @abstractmethod
    async def generate_stream(self, payload: PromptPayload) -> AsyncIterator[str]:
        """Потоковая генерация токенов (стриминг)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Корректное закрытие сетевых соединений и освобождение ресурсов."""
        pass
```

### 3.2. `BaseMemoryStore` (`core/ports/memory_store.py`)

Абстрактный контракт для управления контекстом и памятью диалога:

```python
from abc import ABC, abstractmethod
from core.models import ChatMessage

class BaseMemoryStore(ABC):
    """Абстрактный порт хранилища контекста диалогов."""

    @abstractmethod
    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Получить историю сообщений для указанной сессии."""
        pass

    @abstractmethod
    async def save_message(self, session_id: str, message: ChatMessage) -> None:
        """Сохранить сообщение в историю сессии."""
        pass

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Очистить историю диалога сессии."""
        pass
```

---

## 4. Domain Service: `AgentRunner` (`core/runner.py`)

Оркестратор взаимодействия домена:

```python
from core.ports.llm_plugin import BaseLLMPlugin
from core.ports.memory_store import BaseMemoryStore
from core.models import ChatMessage, PromptPayload, AgentResult

class AgentRunner:
    """Оркестратор бизнес-логики обработки запросов пользователя."""

    def __init__(
        self,
        llm_plugin: BaseLLMPlugin,
        memory_store: BaseMemoryStore,
        system_prompt: str | None = None,
        default_temperature: float = 0.7,
    ):
        self.llm_plugin = llm_plugin
        self.memory_store = memory_store
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.default_temperature = default_temperature

    async def run(self, session_id: str, user_prompt: str) -> AgentResult:
        """
        1. Получает историю из memory_store.
        2. Формирует список сообщений (System + History + New User Message).
        3. Вызывает llm_plugin.generate.
        4. Сохраняет user и assistant сообщения в memory_store.
        5. Возвращает AgentResult.
        """
        history = await self.memory_store.get_history(session_id)
        
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        messages.extend(history)
        
        user_message = ChatMessage(role="user", content=user_prompt)
        messages.append(user_message)
        
        payload = PromptPayload(
            messages=messages,
            temperature=self.default_temperature
        )
        
        result = await self.llm_plugin.generate(payload)
        
        await self.memory_store.save_message(session_id, user_message)
        await self.memory_store.save_message(
            session_id, ChatMessage(role="assistant", content=result.content)
        )
        
        return result
```

---

## 5. Driven Implementations

### 5.1. `StatelessMemoryStore` (`adapters/memory/stateless.py`)
- Наследует `BaseMemoryStore`.
- `get_history(session_id)` всегда возвращает пустой список `[]`.
- `save_message(session_id, message)` выполняет no-op (ничего не сохраняет в постоянное хранилище).
- `clear(session_id)` выполняет no-op.
- Гарантирует изоляцию каждого запроса (Stateless One-Shot).

### 5.2. `OllamaPlugin` (`adapters/llm/ollama_plugin.py`)
- Наследует `BaseLLMPlugin`.
- В конструктор принимает `base_url: str`, `model_name: str`, `timeout: float = 60.0`.
- Использует `httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))`.
- Принимает `PromptPayload`, сериализует в формат Ollama Chat API (`/api/chat`):
  ```json
  {
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "stream": false,
    "options": {"temperature": 0.7}
  }
  ```
- **Трансляция исключений:**
  - `httpx.TimeoutException` -> `LLMTimeoutError`
  - `httpx.ConnectError` / `httpx.NetworkError` -> `LLMConnectionError`
  - `httpx.HTTPStatusError` -> `LLMResponseError`

---

## 6. Driving Utilities & Telegram Adapter

### 6.1. Safe Message Splitter (`adapters/telegram/splitter.py`)
- Функция `split_message(text: str, max_chunk_size: int = 4000) -> list[str]`.
- Разделение с приоритетом: `\n\n` -> `\n` -> пробел ` ` -> жесткий срез.
- Парсинг незакрытых блоков кода (тройные кавычки ` ``` `): закрывает блок в конце чанка и открывает в начале следующего.

### 6.2. Typing Action Manager (`adapters/telegram/typing.py`)
- Асинхронный контекстный менеджер `async with keep_typing(bot, chat_id, interval=4.5): ...`.
- Запускает фоновую задачу `asyncio.create_task` с периодическим `bot.send_chat_action(chat_id, ChatAction.TYPING)`.
- В блоке `finally` вызывает `task.cancel()` и дожидается отмены через `asyncio.gather(..., return_exceptions=True)`.
