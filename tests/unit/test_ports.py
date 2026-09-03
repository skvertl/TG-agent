import inspect
import pytest

from core.ports import BaseLLMPlugin, BaseMemoryStore
from core.models import PromptPayload, AgentResult, ChatMessage


class TestBaseLLMPlugin:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLLMPlugin()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        abstract_methods = BaseLLMPlugin.__abstractmethods__
        expected_methods = {"initialize", "generate", "generate_stream", "shutdown"}
        assert expected_methods.issubset(abstract_methods)

    def test_concrete_implementation(self):
        class DummyPlugin(BaseLLMPlugin):
            async def initialize(self) -> None:
                pass

            async def generate(self, payload: PromptPayload) -> AgentResult:
                return AgentResult(content="ok", model="dummy")

            async def generate_stream(self, payload: PromptPayload):
                yield "ok"

            async def shutdown(self) -> None:
                pass

        plugin = DummyPlugin()
        assert isinstance(plugin, BaseLLMPlugin)


class TestBaseMemoryStore:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseMemoryStore()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        abstract_methods = BaseMemoryStore.__abstractmethods__
        expected_methods = {"get_history", "save_message", "clear"}
        assert expected_methods.issubset(abstract_methods)

    def test_concrete_implementation(self):
        class DummyMemory(BaseMemoryStore):
            async def get_history(self, session_id: str) -> list[ChatMessage]:
                return []

            async def save_message(self, session_id: str, message: ChatMessage) -> None:
                pass

            async def clear(self, session_id: str) -> None:
                pass

        memory = DummyMemory()
        assert isinstance(memory, BaseMemoryStore)
