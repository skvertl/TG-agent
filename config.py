from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Настройки конфигурации приложения через переменные окружения и .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:1.5b", alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=60.0, alias="OLLAMA_TIMEOUT")
    system_prompt: str = Field(
        default="You are a helpful and concise AI assistant.",
        alias="SYSTEM_PROMPT"
    )
