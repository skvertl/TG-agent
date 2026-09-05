from typing import Any, Dict, Callable
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Описание инструмента для передачи в языковую модель."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Tool:
    """Обертка над исполняемой функцией инструмента."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def execute(self, **kwargs) -> str:
        return self.func(**kwargs)
