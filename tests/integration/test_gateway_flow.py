import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User, Chat
from core.models import AgentResult
from core.runner import AgentRunner
from adapters.memory.stateless import StatelessMemoryStore
from adapters.llm.ollama_plugin import OllamaPlugin
from adapters.telegram.handlers import message_handler


class TestGatewayEndToEndFlow:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self):
        # 1. Driven HTTP Mock Transport simulating Ollama /api/chat
        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat"
            body = json.loads(request.content.decode("utf-8"))
            assert body["model"] == "qwen2.5:1.5b"
            # Verify system prompt injected
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"
            assert body["messages"][1]["content"] == "Write a haiku about Python."

            resp_data = {
                "model": "qwen2.5:1.5b",
                "message": {
                    "role": "assistant",
                    "content": "Code flows like clear streams,\nIndented lines of beauty,\nBugs melt into dawn."
                },
                "prompt_eval_count": 12,
                "eval_count": 18,
                "done": True
            }
            return httpx.Response(status_code=200, json=resp_data)

        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
            # 2. Wire Driven Adapters
            llm_plugin = OllamaPlugin(
                base_url="http://mock-ollama:11434",
                model_name="qwen2.5:1.5b",
                timeout=5.0,
                client=client
            )
            memory_store = StatelessMemoryStore()

            # 3. Core Domain Runner
            runner = AgentRunner(
                llm_plugin=llm_plugin,
                memory_store=memory_store,
                system_prompt="You are a poetic assistant."
            )

            # 4. Driving Telegram Message Simulation
            msg = MagicMock(spec=Message)
            msg.text = "Write a haiku about Python."
            msg.from_user = MagicMock(spec=User)
            msg.from_user.id = 42
            msg.chat = MagicMock(spec=Chat)
            msg.chat.id = 42
            msg.bot = MagicMock()
            msg.bot.send_chat_action = AsyncMock()
            msg.answer = AsyncMock()

            # 5. Execute Handler
            await message_handler(msg, runner)

            # 6. Assertions
            msg.answer.assert_called_once()
            called_text = msg.answer.call_args[0][0]
            assert "Code flows like clear streams" in called_text

            # 7. Verify Stateless Guarantee: memory_store must still have 0 history
            history = await memory_store.get_history("42")
            assert history == []

    @pytest.mark.asyncio
    async def test_full_pipeline_ollama_timeout(self):
        def mock_timeout_handler(request: httpx.Request):
            raise httpx.ReadTimeout("Timeout waiting for Ollama", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_timeout_handler)) as client:
            llm_plugin = OllamaPlugin(
                base_url="http://mock-ollama:11434",
                model_name="qwen2.5:1.5b",
                timeout=1.0,
                client=client
            )
            memory_store = StatelessMemoryStore()
            runner = AgentRunner(llm_plugin=llm_plugin, memory_store=memory_store)

            msg = MagicMock(spec=Message)
            msg.text = "Tell me a story"
            msg.from_user = MagicMock(spec=User)
            msg.from_user.id = 42
            msg.chat = MagicMock(spec=Chat)
            msg.chat.id = 42
            msg.bot = MagicMock()
            msg.bot.send_chat_action = AsyncMock()
            msg.answer = AsyncMock()

            await message_handler(msg, runner)

            msg.answer.assert_called_once()
            called_text = msg.answer.call_args[0][0]
            assert "Превышено время ожидания" in called_text
