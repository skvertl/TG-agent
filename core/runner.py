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


from pathlib import Path


def sanitize_output(text: str) -> str:
    """Удаляет случайные галлюцинации китайских иероглифов (CJK) и чистит оборванные фразы."""
    if not text:
        return ""
    cjk_pattern = r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff01-\uff5e]"
    if re.search(cjk_pattern, text):
        cleaned = re.sub(cjk_pattern + "+", "", text)
        # Если в конце текста остался оборванный хвост слова без завершения (например "Держите ")
        cleaned = re.sub(r"\s+[А-Яа-яЁёA-Za-z]+$", ".", cleaned.strip())
        cleaned = re.sub(r"\s+([.,!?:;])", r"\1", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        return cleaned.strip()
    return text


def compact_observation(
    content: str = "",
    max_chars: int = 250,
    observation: Optional[str] = None,
) -> str:
    """
    Compacts an older tool observation to at most max_chars,
    preserving initial status/headers and appending a truncation marker.
    Supports both `content` and `observation` keyword arguments.
    """
    text = observation if observation is not None else content
    prefix = ""
    body = text
    if text.startswith("Observation: "):
        prefix = "Observation: "
        body = text[len("Observation: "):]
    elif text.startswith("Observation:"):
        prefix = "Observation: "
        body = text[len("Observation:"):].lstrip()

    if len(body) <= max_chars:
        return text

    truncated = body[:max_chars]
    boundary = max(truncated.rfind(" "), truncated.rfind("\n"))
    if boundary > int(max_chars * 0.8):
        truncated = truncated[:boundary].rstrip()

    compacted_body = (
        f"{truncated}\n... [Observation compacted: {len(body)} -> {max_chars} chars]"
    )
    return f"{prefix}{compacted_body}"


class AgentRunner:
    """Orchestrates business logic for user requests, tools, and model interaction."""

    def __init__(
        self,
        llm_plugin: BaseLLMPlugin,
        memory_store: BaseMemoryStore,
        system_prompt: str | None = None,
        default_temperature: float = 0.3,
        observability_engine: Optional[ObservabilityEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_turns: int = 8,
        skills_dir: Optional[str] = None,
        max_observation_chars: int = 250,
        history_limit: Optional[int] = 10,
    ) -> None:
        self.llm_plugin = llm_plugin
        self.memory_store = memory_store
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.default_temperature = default_temperature
        self.observability_engine = observability_engine
        self.tool_registry = tool_registry
        self.max_turns = max_turns
        self.skills_dir = skills_dir
        self.max_observation_chars = max_observation_chars
        self.history_limit = history_limit

    def _get_skills_summary(self) -> str:
        skills_path = Path(self.skills_dir or "skills")
        if not skills_path.is_dir():
            return ""

        items = []
        for d in sorted(skills_path.iterdir()):
            sf = None
            if d.is_dir():
                for cand in ["SKILL.md", "skill.md"]:
                    if (d / cand).is_file():
                        sf = d / cand
                        break
            elif d.is_file() and d.suffix == ".md":
                sf = d

            if sf:
                desc = ""
                try:
                    for line in sf.read_text(encoding="utf-8").splitlines():
                        if line.startswith("description:"):
                            desc = line.split("description:", 1)[1].strip()
                            break
                except Exception:
                    pass
                name = d.name if d.is_dir() else d.stem
                items.append(f"- {name}: {desc or 'Specialized skill routine'}")

        if not items:
            return ""

        return (
            "\n\nSkills:\n"
            + "\n".join(items)
            + "\nTo execute, use 'read_skill' with 'skill_name'.\n"
        )

    def _build_system_prompt_with_tools(self) -> str:
        if not self.tool_registry:
            return self.system_prompt

        skills_block = self._get_skills_summary()
        base = self.system_prompt + skills_block

        defs = self.tool_registry.get_definitions()
        if not defs:
            return base

        tool_descriptions = [
            f"- {d['name']}: {d['description']}. Parameters: {json.dumps(d['parameters'], ensure_ascii=False, separators=(',', ':'))}"
            for d in defs
        ]

        tools_block = (
            "\n\nTools:\n"
            + "\n".join(tool_descriptions)
            + "\n\nRules:\n"
            "1. For skills (e.g. morning-briefing, system-health), step 1 MUST be: Action: read_skill\n"
            'Action Input: {"skill_name": "<skill_name>"}\n'
            "2. EXACTLY ONE action per turn. Format:\n"
            "Action: <tool_name>\n"
            "Action Input: <json_arguments>\n"
            "3. Wait for 'Observation:' before next action.\n"
            "4. When complete, output:\n"
            "Final Answer: <response>\n"
            "5. Output STRICTLY in Russian (Cyrillic). Never output Chinese (CJK) characters."
        )
        return base + tools_block

    def _parse_action(self, text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Парсит вызов инструмента из ответа модели с защитой от multi-action и нестандартного форматирования."""
        # Паттерн 1: Action: <name>\nAction Input: <json or string>
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text)
        if action_match:
            tool_name = action_match.group(1).strip()
            after_action = text[action_match.end():]
            input_match = re.search(r"Action Input:\s*", after_action)
            if input_match:
                payload_str = after_action[input_match.end():].strip()
                if payload_str.startswith("{"):
                    try:
                        decoder = json.JSONDecoder()
                        args, _ = decoder.raw_decode(payload_str)
                        if isinstance(args, dict):
                            return tool_name, args
                    except json.JSONDecodeError:
                        pass
                first_line = payload_str.split("\n", 1)[0].strip()
                try:
                    args = json.loads(first_line)
                    if isinstance(args, dict):
                        return tool_name, args
                except json.JSONDecodeError:
                    pass
                return tool_name, {"input": first_line}

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

    def _prepare_turn_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        turn_history: list[dict[str, Any]],
        history: Optional[list[ChatMessage]] = None,
    ) -> list[ChatMessage]:
        """
        Constructs the sequence of ChatMessages for the current turn.
        Compacts historical tool observations prior to the immediate previous turn
        if they exceed max_observation_chars.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_prompt))

        num_turns = len(turn_history)
        for idx, turn in enumerate(turn_history):
            action = turn.get("action")
            if action:
                messages.append(ChatMessage(role="assistant", content=str(action)))

            obs = turn.get("observation")
            if obs is not None:
                obs_str = str(obs)
                if idx < num_turns - 1:
                    obs_formatted = compact_observation(obs_str, self.max_observation_chars)
                else:
                    obs_formatted = obs_str

                if not obs_formatted.startswith("Observation:"):
                    obs_formatted = f"Observation: {obs_formatted}"

                messages.append(ChatMessage(role="user", content=obs_formatted))

        return messages

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

        history = await self.memory_store.get_history(session_id, limit=self.history_limit)
        effective_system_prompt = self._build_system_prompt_with_tools()

        user_message = ChatMessage(role="user", content=user_prompt)

        turn_history: list[dict[str, Any]] = []
        last_result: Optional[AgentResult] = None
        current_turn = 1
        model_name = "unknown"

        try:
            while current_turn <= self.max_turns:
                messages = self._prepare_turn_messages(
                    system_prompt=effective_system_prompt,
                    user_prompt=user_prompt,
                    turn_history=turn_history,
                    history=history,
                )
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
                    if action:
                        if current_turn < self.max_turns:
                            tool_name, tool_args = action
                            tool_output = self.tool_registry.execute(
                                tool_name=tool_name,
                                task_id=task_id,
                                turn_number=current_turn,
                                arguments=tool_args,
                            )

                            turn_history.append({
                                "action": response_text,
                                "observation": tool_output,
                            })
                            current_turn += 1
                            continue
                        else:
                            # Защита от зацикливания (Anti-Looping): принудительный синтез ответа
                            synthesis_messages = self._prepare_turn_messages(
                                system_prompt=effective_system_prompt,
                                user_prompt=user_prompt,
                                turn_history=turn_history,
                                history=history,
                            )
                            synthesis_messages.append(ChatMessage(role="assistant", content=response_text))
                            synthesis_messages.append(
                                ChatMessage(
                                    role="user",
                                    content=(
                                        f"System Notice: You have reached the maximum step limit ({self.max_turns} turns). "
                                        "Do not call any more tools. Synthesize all observations gathered so far and provide your Final Answer directly."
                                    ),
                                )
                            )
                            payload = PromptPayload(
                                messages=synthesis_messages,
                                temperature=self.default_temperature,
                            )
                            if self.observability_engine:
                                async with self.observability_engine.track_llm(
                                    task_id=task_id,
                                    model=model_name,
                                    turn_number=current_turn + 1,
                                    session_id=session_id,
                                    messages=[f"{m.role}: {m.content}" for m in synthesis_messages],
                                ) as span:
                                    last_result = await self.llm_plugin.generate(payload)
                                    span.model = last_result.model
                                    span.set_tokens(
                                        input_tokens=last_result.prompt_tokens or 0,
                                        output_tokens=last_result.completion_tokens or 0,
                                    )
                            else:
                                last_result = await self.llm_plugin.generate(payload)
                            break

                # Если нет вызова инструментов
                break

            assert last_result is not None

            # Очистка префикса "Final Answer:" если он есть и фильтрация галлюцинаций
            clean_content = last_result.content
            if "Final Answer:" in clean_content:
                clean_content = clean_content.split("Final Answer:", 1)[1].strip()
            elif "Action:" in clean_content and current_turn >= self.max_turns:
                clean_content = f"Execution reached maximum step limit ({self.max_turns} turns). Summary: Processed observations."

            clean_content = sanitize_output(clean_content)

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
