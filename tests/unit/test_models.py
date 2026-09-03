import pytest
from pydantic import ValidationError

from core.models import ChatMessage, PromptPayload, AgentResult


class TestChatMessage:
    def test_valid_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_valid_roles(self, role):
        msg = ChatMessage(role=role, content="valid content")
        assert msg.role == role

    def test_invalid_role_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="Hello")

    def test_empty_content_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_missing_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ChatMessage()  # type: ignore[call-arg]


class TestPromptPayload:
    def test_valid_prompt_payload_defaults(self):
        msg = ChatMessage(role="user", content="Hi")
        payload = PromptPayload(messages=[msg])

        assert payload.messages == [msg]
        assert payload.temperature == 0.7
        assert payload.max_tokens is None
        assert payload.options == {}

    def test_valid_custom_values(self):
        msg = ChatMessage(role="user", content="Hi")
        payload = PromptPayload(
            messages=[msg],
            temperature=1.2,
            max_tokens=256,
            options={"top_p": 0.9},
        )
        assert payload.temperature == 1.2
        assert payload.max_tokens == 256
        assert payload.options == {"top_p": 0.9}

    @pytest.mark.parametrize("temp", [0.0, 1.0, 2.0])
    def test_temperature_boundary_values(self, temp):
        payload = PromptPayload(
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=temp,
        )
        assert payload.temperature == temp

    @pytest.mark.parametrize("invalid_temp", [-0.1, 2.1, 10.0])
    def test_temperature_out_of_range(self, invalid_temp):
        with pytest.raises(ValidationError):
            PromptPayload(
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=invalid_temp,
            )

    @pytest.mark.parametrize("invalid_max_tokens", [0, -1, -100])
    def test_max_tokens_must_be_positive(self, invalid_max_tokens):
        with pytest.raises(ValidationError):
            PromptPayload(
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=invalid_max_tokens,
            )


class TestAgentResult:
    def test_valid_agent_result_required_fields(self):
        result = AgentResult(content="Response text", model="qwen2.5:1.5b")
        assert result.content == "Response text"
        assert result.model == "qwen2.5:1.5b"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None

    def test_valid_agent_result_all_fields(self):
        result = AgentResult(
            content="Response text",
            model="qwen2.5:1.5b",
            prompt_tokens=15,
            completion_tokens=42,
        )
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 42

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentResult(content="Only content")  # type: ignore[call-arg]
