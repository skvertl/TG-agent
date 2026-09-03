from typing import List, Union, Optional
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
    allowed_user_ids: Union[str, List[int], int] = Field(default="", alias="ALLOWED_USER_IDS")
    allowed_user_id: Optional[int] = Field(default=None, alias="ALLOWED_USER_ID")

    def model_post_init(self, __context) -> None:
        parsed: List[int] = []
        val = self.allowed_user_ids

        if isinstance(val, int):
            parsed = [val]
        elif isinstance(val, str) and val.strip():
            cleaned = val.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            parsed = [int(x.strip()) for x in cleaned.split(",") if x.strip().isdigit()]
        elif isinstance(val, (list, tuple, set)):
            parsed = [int(x) for x in val]

        if self.allowed_user_id is not None and self.allowed_user_id not in parsed:
            parsed.append(self.allowed_user_id)

        object.__setattr__(self, "allowed_user_ids", parsed)
