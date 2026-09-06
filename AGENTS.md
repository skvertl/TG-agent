# Workspace Rules & Agent Instructions (TG-agent)

## 1. Autonomous Skill Usage

<!-- SKILLS_CATALOG_START -->
### Available Skills & Autonomous Activation Catalog

> [!IMPORTANT]
> Before executing any non-trivial task, the agent **MUST consult this catalog**[cite: 8].
> If the task intent matches the scope of a skill, the agent **MUST inspect its specification file (`SKILL.md`) using available workspace file-reading tools** before writing code[cite: 8].

| Skill | Purpose & Trigger Condition |
| :--- | :--- |
| **`agent-loop`** | Used for all end-to-end autonomous feature development tasks[cite: 8, 12]. Coordinates a 4-subagent loop (Planning -> Development -> Review -> Publisher) with strict TDD, two-axis code review, and automated Pull Request creation[cite: 5, 8, 12]. |
| **`code-review`** | Used for all code audit and verification tasks before committing or opening a PR[cite: 7]. Conducts a parallel review along two independent axes: Standards (repository conventions and Fowler code smells) and Spec (adherence to OpenSpec/PRD contracts with zero scope creep)[cite: 2, 7, 9]. |
| **`e2e-test`** | Used for all end-to-end (E2E) browser testing tasks for web applications via Chrome DevTools MCP[cite: 12]. Verifies server readiness, authentication, DOM assertions, and real-time streaming in an actual browser[cite: 12]. |
| **`grill-me`** | Used for all architecture stress-testing and deep technical interview tasks before implementation. Interrogates design proposals and architectural decisions one rigorous question at a time to eliminate blind spots. |
| **`manual-automation`** | Used for all manual browser testing and UI automation tasks via Chrome DevTools MCP (port management, browser launch, element clicks, form fills, and screenshots)[cite: 3, 12]. |
| **`planning-and-task-breakdown`** | Used for all architectural design and requirement decomposition tasks[cite: 2, 7]. Breaks down complex features into small, ordered, verifiable tasks with acceptance criteria and checkpoints in OpenSpec (tasks.md / plan.md)[cite: 2, 7]. |
| **`pull-request`** | Used for all tasks related to Git and GitHub: creating feature branches, writing Conventional Commits, pushing to remote repositories, and opening Pull Requests via the GitHub CLI (gh pr create)[cite: 7, 9]. |
| **`qa`** | Used for all interactive conversational QA testing and bug exploration tasks in the codebase, with automated filing of structured GitHub Issues[cite: 13]. |
| **`tdd`** | Used for all tasks involving writing or modifying production code[cite: 5, 7, 8]. Enforces strict Test-Driven Development (TDD) via the RED-GREEN-REFACTOR cycle against public interfaces and ports[cite: 5, 7, 8]. |

<!-- SKILLS_CATALOG_END -->

### Orchestration Precedence & Invocation Rules
- **End-to-End Orchestration**: For end-to-end features or multi-step tasks, activate **`agent-loop`** as the master coordinator[cite: 8, 12]. Do NOT invoke standalone sub-skills directly; let `agent-loop` spawn subagents with their dedicated skill contexts to avoid context contamination[cite: 8, 12].
- **Atomic Operations**: For targeted, single-file bugfixes or localized refactoring, invoke individual skills (**`tdd`**, **`code-review`**) directly without running the full orchestration loop[cite: 5, 7].
- **Pre-Implementation Stress-Testing**: Activate **`grill-me`** to stress-test architectural choices whenever design forks or trade-offs arise prior to drafting specs.

---

## 2. Zero Documentation Drift Policy
All changes to the codebase MUST remain synchronized with the documentation and specifications[cite: 2, 7]:

1. **Mandatory Structure Updates in `README.md`**:
   - If core modules (`core/`), adapters (`adapters/`), tools (`core/tools/`), or observability subsystems (`core/observability/`) are added or modified, the agent must ensure synchronization before task closure[cite: 2, 7]:
     - Update the architectural diagram (ASCII / Mermaid)[cite: 2].
     - Update descriptions of layers, ports, and tools[cite: 2].
     - Add instructions for any new CLI commands or endpoints[cite: 2, 4].

2. **Specifications in `openspec/` (Spec-First Integrity)**:
   - Any new feature or capability must be documented under `openspec/changes/<change-name>/`[cite: 2]:
     - `proposal.md` — architectural intent, boundary definition, and NFRs[cite: 2].
     - `specs/*.md` — DTO contracts, port interfaces, database schemas, validation rules[cite: 2].
     - `tasks.md` — task checklist including a mandatory documentation update task[cite: 2].

3. **Definition of Done & Commit Order**:
   - During TDD cycles, keep source and test edits atomic (RED -> GREEN) without premature documentation edits[cite: 5, 8].
   - Documentation updates (`README.md`, architecture diagrams) must be scheduled as an explicit final task in `tasks.md` and verified prior to closing the feature via `pull-request`[cite: 2, 7].
   - A task is NOT complete if the code is written and tests pass, but `README.md` or specifications in `openspec/` are outdated[cite: 2, 7].

---

## 3. Architectural Principles & Constraints
- **Hexagonal Architecture (Ports & Adapters)**:
  - `core/` MUST NEVER import anything from `adapters/` or external UI/networking libraries (`aiogram`, `telegram`)[cite: 2, 8].
  - Core interfaces (`ports`) define abstract communication boundaries, keeping the domain completely independent of delivery channels and concrete LLM providers[cite: 2, 8].
  - `core/observability/` MUST remain a fully standalone module, ready to be extracted into any other project with zero external dependencies.
- **Security & Sandbox**:
  - All file-system tools MUST validate paths using `relative_to(workspace_root)` to prevent Path Traversal[cite: 9].
  - All system commands (`run_tests`, etc.) MUST run without `shell=True`, using a strict whitelist against Command Injection[cite: 7].
  - Secrets (tokens) are passed strictly via Docker Secrets (`/run/secrets/`) or `.env` templates and MUST NEVER be committed to git or exposed in client-facing environments[cite: 2, 8].