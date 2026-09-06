# Proposal: Phase 2 — Token Cost & Context Optimization

## 1. Executive Summary & Problem Statement

### 1.1. Baseline Benchmark Findings (Phase 1.5)
On September 6, 2026, an exhaustive 20-task baseline benchmark was executed (`scripts/run_benchmarks.py`) against local model `qwen2.5:7b` to establish concrete telemetry for token consumption, context repetition, and FinOps costs.

The baseline audit revealed significant context inefficiencies:

| Metric | Baseline Value (20 Tasks) | Notes |
| :--- | :--- | :--- |
| **Total Input Tokens** | **90,035** | Cumulative prompt tokens sent across all turns |
| **Total Output Tokens** | **5,351** | Cumulative completion tokens generated |
| **Total Token Volume** | **95,386** | Total token throughput across 20 tasks |
| **Repeated Context Tokens** | **21,383** | Unnecessary duplicate tokens re-sent in multi-turn steps |
| **Repeated Context %** | **23.7%** | Inefficiency ratio (target for pruning & compaction) |
| **Total Estimated Cost** | **$0.03344 USD** | Baseline cost at $0.30/1M input, $1.20/1M output |
| **Avg Turns per Task** | **1.8** | Average ReAct thinking/tool loop turns |
| **Avg Latency per Task** | **6.64s** | Average end-to-end task duration |
| **Task Success Rate** | **100% (20/20)** | Functional baseline that must not regress |

### 1.2. Root Cause Analysis of Context Bloat
Detailed span-level analysis identified three primary contributors to token bloat:

1. **Quadratic Observation Accumulation in ReAct Loop**:
   - In multi-turn tasks (e.g., Task #6 with 8 turns consuming 20,217 tokens, Task #11 with 4 turns consuming 10,736 tokens), full verbose tool outputs from early turns (`exec`, `read_skill`, `search`, `read_file`) were preserved verbatim in every subsequent turn.
   - For example, Task #6 had **57.2% repeated tokens** (11,337 repeated tokens), as command outputs (df, uptime, free, python version, ollama status) were re-transmitted on every subsequent step.
2. **Unbounded Dialogue History Drift**:
   - The memory store (`SqliteMemoryStore.get_history`) fetched complete session history without a sliding window.
   - In conversational and multi-turn workflows, prompt size climbed monotonically from ~1,250 tokens in early tasks up to **6,066 tokens** by Task #20, even for single-turn queries requiring no tool calls.
3. **Tool Schema & System Prompt Overhead**:
   - The tool definitions and system instructions included formatted JSON with unnecessary whitespace, indentation, and verbose phrasing.
   - Because system prompts are prepended to every single model invocation across every turn, static prompt overhead compounds linearly with turn count.

---

## 2. Proposed Architecture & Core Optimizations

To achieve dramatic token reduction while strictly maintaining 100% task success rate, Phase 2 implements three complementary optimizations adhering to Hexagonal Architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AGENT RUNNER (CORE)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Tool Schema & System Prompt Minification                                │
│     - Compact JSON formatting: json.dumps(params, separators=(',', ':'))    │
│     - Dense, directive instruction format                                   │
│                                                                             │
│  2. Dialogue History Sliding Window                                         │
│     - Controlled history retrieval: get_history(session_id, limit=N)        │
│     - Prunes stale conversational turns outside active window               │
│                                                                             │
│  3. ReAct Observation Compactor                                             │
│     - Turn N: keeps full tool observation for immediate reasoning           │
│     - Turn < N: compacts prior observations into concise summaries          │
│                                                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │    SqliteMemoryStore  │             │  ObservabilityEngine  │
        │ - get_history(limit)  │             │ - Compaction tracking │
        │ - Reverse-limit query │             │ - FinOps verification │
        └───────────────────────┘             └───────────────────────┘
```

### Optimization 1: Observation Compactor in ReAct Loop (`core/runner.py`)
- **Mechanism**:
  - In multi-turn execution (`current_turn > 1`), only the observation from the immediately preceding turn (`turn = current_turn - 1`) retains its full raw output (up to the standard 4KB limit).
  - Observations from earlier turns (`turn < current_turn - 1`) are automatically compacted into a dense excerpt or summary (e.g., `Observation (compacted): <first lines / key status>... [truncated from X chars]`).
  - The model retains full situational context for immediate action, while prior history is kept in dense summary form, eliminating quadratic token growth.

### Optimization 2: Dialogue History Sliding Window (`core/ports/` & `adapters/memory/`)
- **Mechanism**:
  - Update `BaseMemoryStore.get_history(session_id: str, limit: Optional[int] = None)` port interface.
  - Implement subquery-based reverse limit in `SqliteMemoryStore`:
    ```sql
    SELECT role, content FROM (
        SELECT id, role, content FROM chat_messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
    ) ORDER BY id ASC;
    ```
  - Introduce configurable `history_limit` in `AgentRunner` (default: 10 messages / 5 conversation rounds), preventing indefinite context inflation across long sessions.

### Optimization 3: System Prompt & Tool Schema Minification (`core/runner.py`)
- **Mechanism**:
  - Minify JSON parameter schemas in `_build_system_prompt_with_tools()` using `json.dumps(d['parameters'], separators=(',', ':'))`.
  - Strip redundant newlines and whitespace from system prompts and skill listings.
  - Condense tool instruction rules into high-density directives without sacrificing model instruction-following accuracy.

---

## 3. Target Metrics & Acceptance Criteria

| Metric | Baseline (Phase 1.5) | Target (Phase 2) | Expected Impact |
| :--- | :--- | :--- | :--- |
| **Total Tokens** | 95,386 | **<= 66,770** | **>= 30% reduction** (savings >= 28,616 tokens) |
| **Repeated Context %** | 23.7% | **<= 15.0%** | Drastic pruning of multi-turn redundant tokens |
| **Total Estimated Cost** | $0.03344 USD | **<= $0.02340 USD** | FinOps operational expenditure reduction |
| **Benchmark Success Rate** | 100% (20/20) | **100% (20/20)** | Zero functional regression across all 4 prompt categories |
| **Hexagonal Compliance** | Full | **Full** | Clean separation of domain core, ports, and driven adapters |
| **Documentation Drift** | Zero | **Zero** | `README.md`, specs, and benchmark reports kept 100% synchronized |

---

## 4. Risks & Mitigations

1. **Risk**: Over-compaction of past observations could degrade model reasoning on tasks requiring multi-step aggregation (e.g., DevOps health checks or multi-file grep).
   - *Mitigation*: The immediate previous turn observation is **never** compacted. Earlier turn compaction preserves tool name, exit status, and leading informational lines up to 250 characters.
2. **Risk**: Sliding window might lose critical context from earlier in the dialogue.
   - *Mitigation*: Default window of 10 messages (5 dialogue turns) covers typical multi-turn task dependencies while bounding token accumulation. `limit=None` remains supported for stateless or complete history audits.
3. **Risk**: Minified JSON schemas might cause local small models (7B/14B) to produce malformed tool call arguments.
   - *Mitigation*: The existing robust regex and JSON decoder in `AgentRunner._parse_action` already handles flexible JSON payloads, raw strings, and fallback structures. Comprehensive unit tests will verify tool calling fidelity.
