"""Ollama LLM plugin driven adapter."""
import json
from typing import AsyncIterator, Any
import httpx

from core.ports.llm_plugin import BaseLLMPlugin
from core.models import PromptPayload, AgentResult
from core.exceptions import LLMTimeoutError, LLMConnectionError, LLMResponseError


class OllamaPlugin(BaseLLMPlugin):
    """LLM plugin adapter interacting with Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:1.5b",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = client
        self._own_client = client is None

    async def initialize(self) -> None:
        """Initialize HTTP client and connection pool."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0)
            )

    async def shutdown(self) -> None:
        """Close HTTP client connection pool gracefully."""
        if self.client is not None:
            await self.client.aclose()
            if self._own_client:
                self.client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self.client is None:
            await self.initialize()
        assert self.client is not None
        return self.client

    def _build_request_body(self, payload: PromptPayload, stream: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": payload.temperature}
        if payload.max_tokens is not None:
            options["num_predict"] = payload.max_tokens
        options.update(payload.options)

        return {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in payload.messages],
            "stream": stream,
            "options": options,
        }

    async def generate(self, payload: PromptPayload) -> AgentResult:
        """Synchronous chat inference via Ollama /api/chat."""
        client = await self._ensure_client()
        request_body = self._build_request_body(payload, stream=False)

        try:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama request timed out: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise LLMConnectionError(f"Failed to connect to Ollama: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(
                f"Ollama HTTP error: {exc}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Ollama network error: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content", "")
        model = data.get("model", self.model_name)
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        return AgentResult(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def generate_stream(self, payload: PromptPayload) -> AsyncIterator[str]:
        """Streaming chat tokens via Ollama /api/chat."""
        client = await self._ensure_client()
        request_body = self._build_request_body(payload, stream=True)

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=request_body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk_data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama stream timed out: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise LLMConnectionError(f"Failed to connect to Ollama stream: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(
                f"Ollama HTTP stream error: {exc}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Ollama network stream error: {exc}") from exc
