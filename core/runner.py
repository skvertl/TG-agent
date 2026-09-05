"""Domain orchestrator service."""
import json
import re
import uuid
from typing import Optional, List, Dict, Any

from core.ports.llm_plugin import BaseLLMPlugin
from core.ports.memory_store import BaseMemoryStore
from core.models import ChatMessage, PromptPayload, AgentResult
from core.observability.engine import ObservabilityEngine
from core.tools.registry import ToolRegistry


class AgentRunner:
    """Orchestrates business logic for user requests, tools, and model interaction."""

    def __init__(
        self,
        llm_plugin: BaseLLMPlugin,
        memory_store: BaseMemoryStore,
        system_prompt: str | None = None,
        default_temperature: float = 0.7,
        observability_engine: Optional[ObservabilityEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_turns: int = 10,
    ) -> None:
        self.llm_plugin = llm_plugin
        self.memory_store = memory_store
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.default_temperature = default_temperature
        self.observability_engine = observability_engine
        self.tool_registry = tool_registry
        self.max_turns = max_turns

    def _build_system_prompt_with_tools(self) -> str:
        if not self.tool_registry:
            return self.system_prompt

        defs = self.tool_registry.get_definitions()
        if not defs:
            return self.system_prompt

        tool_descriptions = []
        for d in defs:
            tool_descriptions.append(
                f"- {d['name']}: {d['description']}. Parameters: {json.dumps(d['parameters'])}"
            )

        tools_block = (
            "\n\nYou have access to the following tools to accomplish your task:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTo call a tool, use this exact format:\n"
            "Action: <tool_name>\n"
            "Action Input: <json_arguments>\n\n"
            "When you have the final answer or if no tools are needed, write your response directly or prefix with:\n"
            "Final Answer: <your response>"
        )
        return self.system_prompt + tools_block

    def _parse_action(self, text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Парсит вызов инструмента из ответа модели."""
        # Паттерн 1: Action: <name>\nAction Input: <json or string>
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*\}|[^\n]+)", text, re.DOTALL)

        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip()
            try:
                args = json.loads(raw_input)
                if isinstance(args, dict):
                    return tool_name, args
            except json.JSONDecodeError:
                pass
            return tool_name, {"input": raw_input}

        # Паттерн 2: [TOOL_CALL: name(args)]
        call_match = re.search(r"\[TOOL_CALL:\s*([a-zA-Z0-9_\-]+)\((.*?)\)\]", text)
        if call_match:
            tool_name = call_match.group(1).strip()
            raw_args = call_match.group(2).strip()
            try:
                args = json.loads(raw_args)
                if isinstance(args, dict):
                    return tool_name, args
            except json.JSONDecodeError:
                pass
            return tool_name, {"path": raw_args} if raw_args else {}

        return None

    async def run(
        self,
        session_id: str,
        user_prompt: str,
        task_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Выполняет многошаговый ReAct-цикл или одиночный вызов с полным трейсингом токенов.
        """
        task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"

        history = await self.memory_store.get_history(session_id)
        effective_system_prompt = self._build_system_prompt_with_tools()

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=effective_system_prompt)
        ]
        messages.extend(history)

        user_message = ChatMessage(role="user", content=user_prompt)
        messages.append(user_message)

        last_result: Optional[AgentResult] = None
        current_turn = 1
        model_name = "unknown"

        try:
            while current_turn <= self.max_turns:
                msg_contents = [f"{m.role}: {m.content}" for m in messages]
                payload = PromptPayload(
                    messages=messages,
                    temperature=self.default_temperature,
                )

                # Трейсинг вызова LLM
                if self.observability_engine:
                    async with self.observability_engine.track_llm(
                        task_id=task_id,
                        model=model_name,
                        turn_number=current_turn,
                        session_id=session_id,
                        messages=msg_contents,
                    ) as span:
                        last_result = await self.llm_plugin.generate(payload)
                        model_name = last_result.model
                        span.model = model_name
                        span.set_tokens(
                            input_tokens=last_result.prompt_tokens or 0,
                            output_tokens=last_result.completion_tokens or 0,
                        )
                else:
                    last_result = await self.llm_plugin.generate(payload)
                    model_name = last_result.model

                response_text = last_result.content

                # Проверяем, вызвала ли модель инструмент
                if self.tool_registry:
                    action = self._parse_action(response_text)
                    if action and current_turn < self.max_turns:
                        tool_name, tool_args = action
                        tool_output = self.tool_registry.execute(
                            tool_name=tool_name,
                            task_id=task_id,
                            turn_number=current_turn,
                            arguments=tool_args,
                        )

                        # Добавляем ход ассистента и наблюдение инструмента в контекст
                        messages.append(ChatMessage(role="assistant", content=response_text))
                        messages.append(ChatMessage(role="user", content=f"Observation: {tool_output}"))
                        current_turn += 1
                        continue

                # Если нет вызова инструментов или достигнут лимит
                break

            assert last_result is not None

            # Очистка префикса "Final Answer:" если он есть
            clean_content = last_result.content
            if "Final Answer:" in clean_content:
                clean_content = clean_content.split("Final Answer:", 1)[1].strip()

            final_result = AgentResult(
                content=clean_content,
                model=last_result.model,
                prompt_tokens=last_result.prompt_tokens,
                completion_tokens=last_result.completion_tokens,
                task_id=task_id,
            )

            # Сохранение в историю сессии
            await self.memory_store.save_message(session_id, user_message)
            await self.memory_store.save_message(
                session_id, ChatMessage(role="assistant", content=final_result.content)
            )

            # Финализация трейсинга
            if self.observability_engine:
                self.observability_engine.finish_run(
                    task_id=task_id,
                    model=model_name,
                    session_id=session_id,
                    success=True,
                )

            return final_result

        except Exception as exc:
            if self.observability_engine:
                self.observability_engine.finish_run(
                    task_id=task_id,
                    model=model_name,
                    session_id=session_id,
                    success=False,
                )
            raise exc
