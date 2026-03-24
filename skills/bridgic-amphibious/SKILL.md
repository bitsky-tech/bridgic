---
name: bridgic-amphibious
description: >-
  Build cognitive agents with bridgic-amphibious — OTA (Observe-Think-Act) loops, deterministic
  workflows, or both combined (amphibious mode). Use when working with AmphibiousAutoma,
  CognitiveWorker, CognitiveContext, think_unit, FunctionToolSpec, RunMode, ErrorStrategy,
  Exposure, snapshot, AgentFallback, or scaffolding new bridgic agent projects.
---

# Bridgic Amphibious

## Architecture

```
Orchestrator   AmphibiousAutoma   ← on_agent / on_workflow, think_unit
     ↑
Worker         CognitiveWorker    ← thinking prompts, hooks, output_schema
     ↑
Context        CognitiveContext   ← Exposure, goal, tools, skills, history
```

Worker decides **what to think**; orchestrator decides **when** and executes tool calls.

Two execution modes: **Agent** (LLM-driven OTA loops) and **Workflow** (deterministic `yield step(...)`). Implement both for **amphibious** behavior — deterministic path by default, agent fallback where complexity spikes.

## Minimal agent

```python
from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveContext, CognitiveWorker, think_unit, ErrorStrategy,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    main = think_unit(
        CognitiveWorker.inline("Plan ONE immediate next step."),
        max_attempts=30,
        on_error=ErrorStrategy.RAISE,
    )
    async def on_agent(self, ctx: CognitiveContext):
        await self.main

# Run:
from bridgic.core.model import OpenAI
agent = MyAgent(llm=OpenAI(model="gpt-4o"))
result = await agent.arun(
    goal="Research AI trends",
    tools=[FunctionToolSpec.from_raw(search_fn)],
)
# result = str (context summary). Use agent.final_answer for LLM's finishing statement.
```

## Hard rules

1. **Always use generic type**: `class X(AmphibiousAutoma[MyContext])`, never bare `AmphibiousAutoma`. Omitting the type parameter raises `TypeError` at class definition time.
2. **Workers never call tools** — only the orchestrator executes the action phase.
3. **Tools must be**: `async def`, type-hinted, docstring (becomes LLM prompt), return `str`. Register via `FunctionToolSpec.from_raw`.
4. **`thinking()` prompt** — context (goal, tools, history) is auto-injected. Do NOT repeat it. Focus on: role, strategy, when to set `finish=True`, constraints.
5. **`_DELEGATE`** — a sentinel `object()`. Default return from worker hooks (`observation`, `before_action`, `after_action`); means "fall through to agent-level hook." Checked via `is` identity.
6. **Policies** (acquiring, rehearsal, reflection) — each fires at most once per `arun()` round.
7. **Observation lifecycle** — `ctx.observation` is refreshed before **every** OTA cycle (each `_run_once`). In workflow mode, each `yield step(...)` triggers a fresh observation. It is never updated during or after tool execution within the same cycle. Safe to read between yields.

## Scaffolding a new project

Use [assets/template.md](assets/template.md) for project file structure (tools.py, context.py, workers.py, agent.py, main.py).

## Reference files

Load only what the current task requires:

| File | Open when |
|------|-----------|
| [references/orchestration.md](references/orchestration.md) | `think_unit` params, `max_attempts` tuning, `until` stop condition, `on_agent` / `on_workflow`, `step()` + `AgentFallback`, `RunMode`, `snapshot()`, agent hooks (`observation`/`before_action`/`after_action`/`action_tool_call`), `arun()` calling conventions, agent properties (`final_answer`, `spend_tokens`) |
| [references/context-and-exposure.md](references/context-and-exposure.md) | Custom context fields, `Field(display=False)`, `Field(use_llm=True)`, `LayeredExposure` / `EntireExposure` subclassing, `summary()` / `get_details()`, `CognitiveHistory` 3-tier memory (`working_memory_size`, `short_term_size`, `compress_threshold`), runtime `Skill` loading |
| [references/cognitive-worker.md](references/cognitive-worker.md) | `CognitiveWorker.inline()` vs subclass, `thinking()` prompt, hooks (`observation`/`before_action`/`after_action`), `_DELEGATE` sentinel, `build_messages()`, policies (`enable_rehearsal`/`enable_reflection`/acquiring), `output_schema` for structured output |
| [references/tools.md](references/tools.md) | `FunctionToolSpec.from_raw()`, tool requirements (async/hints/docstring/return str), docstring format, overriding `tool_name`/`tool_description` |
| [references/error-handling.md](references/error-handling.md) | `ErrorStrategy.RAISE` / `IGNORE` / `RETRY`, `max_retries`, what gets discarded on failure, retry count semantics |
| [references/tracing.md](references/tracing.md) | `trace_running=True`, `AgentTrace`, `TraceStep`, `RecordedToolCall`, `StepOutputType`, saving/loading traces |
| [references/imports.md](references/imports.md) | Consolidated import map — where to import each class |
| [references/amphibious_dynamic_workflow.md](references/amphibious_dynamic_workflow.md) | **Full worked example**: custom `CognitiveContext` + `EntireExposure`, amphibious agent with `on_agent` + `on_workflow`, dynamic Python loops over `ctx.observation` data, `before_action` hooks for sanitization/tracking, hidden runtime dependencies |

## Validation

Run `scripts/validate.sh <project_dir>` to check agent definition, tool signatures, and common mistakes.
