import json
import httpx
import pytest

from core.ports import BaseLLMPlugin
from core.models import PromptPayload, ChatMessage, AgentResult
from core.exceptions import LLMTimeoutError, LLMConnectionError, LLMResponseError
from adapters.llm.ollama_plugin import OllamaPlugin


class TestOllamaPlugin:
    def test_inherits_base_llm_plugin(self):
        plugin = OllamaPlugin()
        assert isinstance(plugin, BaseLLMPlugin)

    @pytest.mark.asyncio
    async def test_generate_success(self):
        request_captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            request_captured["method"] = request.method
            request_captured["url"] = str(request.url)
            request_captured["body"] = json.loads(request.read())
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:1.5b",
                    "message": {"role": "assistant", "content": "Test answer"},
                    "prompt_eval_count": 15,
                    "eval_count": 25,
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(
            base_url="http://mock-ollama:11434",
            model_name="qwen2.5:1.5b",
            client=client,
        )

        payload = PromptPayload(
            messages=[
                ChatMessage(role="system", content="System prompt"),
                ChatMessage(role="user", content="User question"),
            ],
            temperature=0.8,
            max_tokens=100,
            options={"top_k": 40},
        )

        result = await plugin.generate(payload)

        assert isinstance(result, AgentResult)
        assert result.content == "Test answer"
        assert result.model == "qwen2.5:1.5b"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25

        assert request_captured["method"] == "POST"
        assert request_captured["url"] == "http://mock-ollama:11434/api/chat"
        body = request_captured["body"]
        assert body["model"] == "qwen2.5:1.5b"
        assert body["stream"] is False
        assert len(body["messages"]) == 2
        assert body["messages"][0] == {"role": "system", "content": "System prompt"}
        assert body["messages"][1] == {"role": "user", "content": "User question"}
        assert body["options"]["temperature"] == 0.8
        assert body["options"]["num_predict"] == 100
        assert body["options"]["top_k"] == 40

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                json.dumps({"message": {"content": "Hello"}}) + "\n",
                json.dumps({"message": {"content": ", "}}) + "\n",
                json.dumps({"message": {"content": "world!"}}) + "\n",
            ]
            return httpx.Response(
                200,
                content="".join(lines).encode("utf-8"),
                headers={"Content-Type": "application/x-ndjson"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(
            base_url="http://mock-ollama:11434",
            model_name="qwen2.5:1.5b",
            client=client,
        )

        payload = PromptPayload(
            messages=[ChatMessage(role="user", content="Hi")],
        )

        chunks = []
        async for chunk in plugin.generate_stream(payload):
            chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world!"]

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(client=client)

        payload = PromptPayload(messages=[ChatMessage(role="user", content="Hi")])

        with pytest.raises(LLMTimeoutError):
            await plugin.generate(payload)

    @pytest.mark.asyncio
    async def test_generate_connect_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(client=client)

        payload = PromptPayload(messages=[ChatMessage(role="user", content="Hi")])

        with pytest.raises(LLMConnectionError):
            await plugin.generate(payload)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [404, 500, 503])
    async def test_generate_http_status_error(self, status_code):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text="Internal Server Error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(client=client)

        payload = PromptPayload(messages=[ChatMessage(role="user", content="Hi")])

        with pytest.raises(LLMResponseError) as exc_info:
            await plugin.generate(payload)

        assert exc_info.value.status_code == status_code

    @pytest.mark.asyncio
    async def test_generate_stream_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Stream timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(client=client)

        payload = PromptPayload(messages=[ChatMessage(role="user", content="Hi")])

        with pytest.raises(LLMTimeoutError):
            async for _ in plugin.generate_stream(payload):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_connect_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Failed to connect")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        plugin = OllamaPlugin(client=client)

        payload = PromptPayload(messages=[ChatMessage(role="user", content="Hi")])

        with pytest.raises(LLMConnectionError):
            async for _ in plugin.generate_stream(payload):
                pass

    @pytest.mark.asyncio
    async def test_lifecycle_initialize_and_shutdown(self):
        plugin = OllamaPlugin(base_url="http://localhost:11434", timeout=30.0)
        await plugin.initialize()
        assert plugin.client is not None
        assert not plugin.client.is_closed

        await plugin.shutdown()
        assert plugin.client is None or plugin.client.is_closed
