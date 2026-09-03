import pytest
from pydantic import ValidationError
from config import Settings

class TestConfig:
    def test_settings_load_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "tinyllama")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "45.0")
        monkeypatch.setenv("SYSTEM_PROMPT", "Custom prompt")

        settings = Settings()
        assert settings.telegram_bot_token == "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        assert settings.ollama_base_url == "http://ollama:11434"
        assert settings.ollama_model == "tinyllama"
        assert settings.ollama_timeout == 45.0
        assert settings.system_prompt == "Custom prompt"

    def test_settings_defaults(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
        monkeypatch.delenv("SYSTEM_PROMPT", raising=False)

        settings = Settings()
        assert settings.telegram_bot_token == "test-token"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "qwen2.5:1.5b"
        assert settings.ollama_timeout == 60.0
        assert "helpful" in settings.system_prompt.lower()

    def test_missing_token_raises_validation_error(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with pytest.raises(ValidationError):
            Settings()

    def test_allowed_user_ids_parsing(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "111, 222, 333")
        settings = Settings()
        assert settings.allowed_user_ids == [111, 222, 333]

    def test_allowed_single_user_id(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_ID", "999888")
        settings = Settings()
        assert 999888 in settings.allowed_user_ids

    def test_allowed_user_ids_default_empty(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
        monkeypatch.delenv("ALLOWED_USER_ID", raising=False)
        settings = Settings()
        assert settings.allowed_user_ids == []
