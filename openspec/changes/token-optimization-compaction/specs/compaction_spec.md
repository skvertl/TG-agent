# Specification: Token Cost & Context Optimization (`compaction_spec.md`)

## 1. Overview & Architectural Boundaries

This specification defines the implementation details for Phase 2: Token Cost & Context Optimization across the domain core and persistence adapters, adhering to Hexagonal Architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE DOMAIN (core/runner.py)                      │
│                                                                             │
│  - compact_observation(content: str, max_chars: int) -> str                 │
│  - _build_minified_system_prompt()                                          │
│  - history_limit parameter in AgentRunner.__init__                          │
│  - Multi-turn observation compaction in AgentRunner.run()                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│     PORT (core/ports/memory_store)   │  │   OBSERVABILITY (core/observ.)    │
│  - get_history(session_id, limit)    │  │   - span.set_tokens()             │
└──────────────────┬───────────────────┘  │   - overlap/repetition reduction  │
                   │                      └───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  ADAPTERS (adapters/memory/)         │
│  - SqliteMemoryStore.get_history()   │
│  - StatelessMemoryStore.get_history()│
└──────────────────────────────────────┘
```

---

## 2. Specification: Observation Compactor (`core/runner.py`)

### 2.1. Problem & Context
In a multi-turn ReAct loop (`max_turns` up to 8), each tool execution adds an `Observation: <output>` message. When multiple tool calls occur:
- Turn 1: System prompt + User query
- Turn 2: + Assistant Action 1 + Tool Observation 1
- Turn 3: + Assistant Action 2 + Tool Observation 2 (Observation 1 re-sent in full!)
- Turn 4: + Assistant Action 3 + Tool Observation 3 (Observations 1 & 2 re-sent in full!)

This causes quadratic token accumulation. In benchmark Task #6, 8 turns accumulated 19,821 input tokens with 57.2% repetition.

### 2.2. Compaction Algorithm
1. **Scope of Compaction**:
   - The observation from the **immediately preceding turn** (`current_turn - 1`) is preserved **in full** (subject only to the standard 4KB tool execution cap). This guarantees the LLM has complete data to reason about its immediate next step.
   - Historical observations from **prior turns** (`turn < current_turn - 1`) are compacted to a concise summary.

2. **Function Signature**:
   ```python
   def compact_observation(content: str, max_chars: int = 250) -> str:
       """
       Compacts an older tool observation to at most max_chars,
       preserving initial status/headers and appending a truncation marker.
       """
   ```

3. **Compaction Rules**:
   - If `len(content) <= max_chars`, return `content` as-is.
   - If `len(content) > max_chars`:
     - Extract prefix of length `max_chars` (trimmed at word or newline boundary when feasible).
     - Append: `\n... [Observation compacted: {len(content)} -> {max_chars} chars]`.
     - If the observation starts with `"Observation: "`, preserve the prefix format: `"Observation: <compacted_body>"`.

4. **Integration in `AgentRunner.run()`**:
   - At each turn `current_turn` before dispatching `llm_plugin.generate(payload)`:
     ```python
     effective_messages = self._prepare_turn_messages(messages, current_turn)
     payload = PromptPayload(messages=effective_messages, temperature=self.default_temperature)
     ```
   - In `_prepare_turn_messages`:
     - Identify all messages where `role == "user"` and `content.startswith("Observation:")`.
     - Leave the last observation untouched.
     - For all prior observations, replace content with `compact_observation(m.content, self.max_observation_chars)`.
   - `AgentRunner.__init__` receives `max_observation_chars: int = 250`.

---

## 3. Specification: Dialogue History Sliding Window

### 3.1. Port Update: `core/ports/memory_store.py`
Update `BaseMemoryStore.get_history`:
```python
@abstractmethod
async def get_history(
    self,
    session_id: str,
    limit: Optional[int] = None,
) -> list[ChatMessage]:
    """
    Retrieve message history for specified session.
    If limit is specified, returns at most the latest `limit` messages in chronological order.
    """
    pass
```

### 3.2. SQLite Adapter Implementation: `adapters/memory/sqlite.py`
In `SqliteMemoryStore.get_history(session_id: str, limit: Optional[int] = None)`:
- When `limit is None`:
  ```sql
  SELECT role, content FROM chat_messages
  WHERE session_id = ?
  ORDER BY id ASC;
  ```
- When `limit is not None and limit > 0`:
  ```sql
  SELECT role, content FROM (
      SELECT id, role, content FROM chat_messages
      WHERE session_id = ?
      ORDER BY id DESC
      LIMIT ?
  ) ORDER BY id ASC;
  ```
- Subquery selects the newest `limit` messages (descending), and outer query sorts them chronologically (ascending) so the conversation flow remains natural for the LLM.

### 3.3. Stateless Adapter Implementation: `adapters/memory/stateless.py`
Update signature to accept `limit: Optional[int] = None`:
```python
async def get_history(self, session_id: str, limit: Optional[int] = None) -> list[ChatMessage]:
    return []
```

### 3.4. Orchestrator Integration: `core/runner.py`
- Add `history_limit: Optional[int] = 10` to `AgentRunner.__init__`.
- In `AgentRunner.run()`:
  ```python
  history = await self.memory_store.get_history(session_id, limit=self.history_limit)
  ```
- A window of 10 messages corresponds to 5 complete user/assistant conversational turns, preventing unbounded linear context growth across long-lived Telegram sessions while preserving full multi-turn context for recent exchanges.

---

## 4. Specification: Tool Schema & System Prompt Minification

### 4.1. Tool Parameter Schema Minification
- Currently: `json.dumps(d['parameters'])` produces spaces: `{"type": "object", "properties": ...}`.
- Specification: Minify using `json.dumps(d['parameters'], separators=(',', ':'))`.
- Eliminate redundant whitespace in tool definitions:
  ```python
  tool_descriptions = [
      f"- {d['name']}: {d['description']}. Parameters: {json.dumps(d['parameters'], separators=(',', ':'))}"
      for d in defs
  ]
  ```

### 4.2. Directive Instruction Streamlining
The current tool instructions in `_build_system_prompt_with_tools()` contain ~180 words. The minified instructions streamline this to dense, unambiguous directives:
```python
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
```

### 4.3. Skills Summary Minification
In `_get_skills_summary()`:
- Strip redundant leading/trailing empty lines.
- Dense formatting:
  ```python
  return (
      "\n\nSkills:\n"
      + "\n".join(items)
      + "\nTo execute, use 'read_skill' with 'skill_name'.\n"
  )
  ```

---

## 5. FinOps & Observability Alignment

1. **True Token Measurement**:
   - `observability_engine.track_llm(..., messages=msg_contents)` must receive the exact compacted `effective_messages` sent to the model.
   - This ensures `input_tokens`, `repeated_tokens`, and FinOps pricing calculations record the actual savings achieved.
2. **Benchmark Comparison**:
   - `scripts/run_benchmarks.py` will run the exact 20 benchmark tasks with the optimizations enabled.
   - Output will be stored in `data/benchmark_optimized.json`.
   - `BENCHMARK_REPORT.md` will be updated with side-by-side comparative analysis:
     - Baseline vs. Phase 2 Optimized
     - Token reduction percentage per category
     - Cost savings and latency delta.
