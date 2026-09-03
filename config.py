import os
from pathlib import Path
from typing import List, Union, Optional, Tuple, Type
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    SecretsSettingsSource,
)
from pydantic import Field, model_validator


def _resolve_secrets_dir() -> Optional[Path]:
    """Определяет директорию Docker Secrets (через SECRETS_DIR, /run/secrets или ./secrets)."""
    env_dir = os.environ.get("SECRETS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
    for default_path in ["/run/secrets", "./secrets"]:
        p = Path(default_path)
        if p.is_dir():
            return p
    return None


def _read_file_content(file_path: Optional[Union[str, Path]]) -> Optional[str]:
    """Безопасно читает содержимое файла секрета, игнорируя комментарии и пустые строки."""
    if not file_path:
        return None
    p = Path(file_path)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
                return "\n".join(lines) if lines else None
        except OSError:
            return None
    return None


class Settings(BaseSettings):
    """Настройки конфигурации приложения с поддержкой Docker Secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_bot_token_file: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN_FILE")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:1.5b", alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=60.0, alias="OLLAMA_TIMEOUT")
    system_prompt: str = Field(
        default="You are a helpful and concise AI assistant.",
        alias="SYSTEM_PROMPT",
    )

    allowed_user_ids: Union[str, List[int], int] = Field(default="", alias="ALLOWED_USER_IDS")
    allowed_user_id: Optional[int] = Field(default=None, alias="ALLOWED_USER_ID")
    allowed_user_id_file: Optional[str] = Field(default=None, alias="ALLOWED_USER_ID_FILE")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        s_dir = _resolve_secrets_dir()
        if s_dir:
            file_secret_settings = SecretsSettingsSource(settings_cls, secrets_dir=s_dir)
        return (
            init_settings,
            file_secret_settings,
            env_settings,
            dotenv_settings,
        )

    @model_validator(mode="after")
    def validate_and_post_process(self) -> "Settings":
        # 1. Чтение токена из TELEGRAM_BOT_TOKEN_FILE или fallback на secrets/telegram_bot_token[.txt]
        if not self.telegram_bot_token and self.telegram_bot_token_file:
            val = _read_file_content(self.telegram_bot_token_file)
            if val:
                self.telegram_bot_token = val

        if not self.telegram_bot_token:
            s_dir = _resolve_secrets_dir()
            if s_dir:
                for candidate in [s_dir / "telegram_bot_token", s_dir / "telegram_bot_token.txt"]:
                    val = _read_file_content(candidate)
                    if val:
                        self.telegram_bot_token = val
                        break

        # Очистка токена от возможных случайных комментариев или пробелов
        if self.telegram_bot_token:
            lines = [
                line.strip()
                for line in self.telegram_bot_token.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            self.telegram_bot_token = lines[0] if lines else None

        if not self.telegram_bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is required. Provide it via Docker Secret (/run/secrets/telegram_bot_token), "
                "TELEGRAM_BOT_TOKEN_FILE, or TELEGRAM_BOT_TOKEN environment variable."
            )

        # 2. Парсинг allowed_user_ids
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

        # 3. Чтение ID из ALLOWED_USER_ID_FILE или fallback на secrets_dir
        if self.allowed_user_id_file:
            file_id = _read_file_content(self.allowed_user_id_file)
            if file_id:
                for chunk in file_id.replace("\n", ",").split(","):
                    if chunk.strip().isdigit():
                        uid = int(chunk.strip())
                        if uid not in parsed:
                            parsed.append(uid)

        if not parsed:
            s_dir = _resolve_secrets_dir()
            if s_dir:
                for candidate in [s_dir / "allowed_user_ids", s_dir / "allowed_user_ids.txt"]:
                    val = _read_file_content(candidate)
                    if val:
                        for chunk in val.replace("\n", ",").split(","):
                            if chunk.strip().isdigit():
                                uid = int(chunk.strip())
                                if uid not in parsed:
                                    parsed.append(uid)
                        if parsed:
                            break

        # 4. Проверка одиночного ALLOWED_USER_ID
        if self.allowed_user_id is not None and self.allowed_user_id not in parsed:
            parsed.append(self.allowed_user_id)

        self.allowed_user_ids = parsed
        return self
