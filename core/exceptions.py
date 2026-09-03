"""Domain exceptions."""
from typing import Optional


class DomainError(Exception):
    """Base domain exception."""
    pass


class LLMPluginError(DomainError):
    """LLM inference provider error."""
    pass


class LLMConnectionError(LLMPluginError):
    """Connection failure with LLM backend."""
    pass


class LLMTimeoutError(LLMPluginError):
    """LLM inference timeout exceeded."""
    pass


class LLMResponseError(LLMPluginError):
    """Invalid response or error status from LLM."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
