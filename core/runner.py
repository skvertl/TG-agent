"""Domain orchestrator service."""
from core.ports.llm_plugin import BaseLLMPlugin
from core.ports.memory_store import BaseMemoryStore
from core.models import ChatMessage, PromptPayload, AgentResult


class AgentRunner:
    """Orchestrates business logic for user requests and model interaction."""

    def __init__(
        self,
        llm_plugin: BaseLLMPlugin,
        memory_store: BaseMemoryStore,
        system_prompt: str | None = None,
        default_temperature: float = 0.7,
    ) -> None:
        self.llm_plugin = llm_plugin
        self.memory_store = memory_store
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.default_temperature = default_temperature

    async def run(self, session_id: str, user_prompt: str) -> AgentResult:
        """Execute single-turn or multi-turn agent pipeline.

        1. Fetches history from memory_store.
        2. Assembles message list (system + history + new user message).
        3. Invokes llm_plugin.generate.
        4. Saves user and assistant messages into memory_store.
        5. Returns AgentResult.
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
            temperature=self.default_temperature,
        )

        result = await self.llm_plugin.generate(payload)

        await self.memory_store.save_message(session_id, user_message)
        await self.memory_store.save_message(
            session_id, ChatMessage(role="assistant", content=result.content)
        )

        return result
