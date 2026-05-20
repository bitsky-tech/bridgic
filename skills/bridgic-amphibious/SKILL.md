---
name: bridgic-amphibious
description: "Build agents with the Bridgic Amphibious dual-mode framework — combining LLM-driven (agent) and deterministic (workflow) execution with peer state-machine dispatch, slot-based step-level fallback, human-in-the-loop, a built-in tool surface (shell, filesystem, search, HITL) auto-injected into every agent, and delegation of sub-goals to external coding agents (claude code, etc.) as think-agent units. Use when: (1) writing code that imports from bridgic.amphibious, (2) creating AmphibiousAutoma subclasses, (3) defining CognitiveWorker think units (think_unit / ThinkUnit) and AgentWorker think agents (think_agent / ThinkAgent), yielding ThinkUnit / ThinkAgent / EnterAgent / ActionCall / HumanCall / LLMCall / RETURN, (4) implementing on_agent/on_workflow methods, (5) working with CognitiveContext, Exposure system, or cognitive policies, (6) adding human-in-the-loop interactions (HumanCall, request_human, request_human_tool), (7) using or filtering the auto-injected built-in tools (bash, read_file/write_file/edit_file, glob, grep, request_human) via the builtin_tools class attribute or arun kwarg, (8) scaffolding a new amphibious project via CLI, (9) any task involving the bridgic-amphibious framework."
---

# Bridgic Amphibious

Dual-mode agent framework: agents operate in LLM-driven (`on_agent`) and deterministic (`on_workflow`) modes. In hybrid `AMPHIFLOW` mode the framework runs a peer state machine over both, with automatic step-level recovery via an injected `resolve_step_fallback` tool.

## Dependencies

A bridgic-amphibious project requires the following packages:

| Package | Description |
|---------|-------------|
| `bridgic-core` | Core framework (Worker, Automa, GraphAutoma) |
| `bridgic-amphibious` | Dual-mode agent framework |
| `bridgic-llms-openai` | LLM provider (omit for pure `WORKFLOW` / `ThinkAgent` projects) |
| `python-dotenv` | `.env` file loading |

Before using this package, you need to install the dependencies by using the provided install script:

```bash
bash "skills/bridgic-amphibious/scripts/install-deps.sh" "$PWD"
```

The script checks uv availability, initializes a uv project if needed, installs any missing packages via `uv add`, and runs `uv sync` to finalize the environment. When it exits successfully the project is fully initialized and ready to use — no manual `uv add` / `uv sync` follow-up is required.

## LLM Setup

Pass a `BaseLlm` instance (with the `astructured_output` protocol, from a bridgic LLM provider package) to `arun(llm=...)`. An LLM is required for any run that uses a `CognitiveWorker` (`ThinkUnit`) or `LLMCall` — i.e. typical `AGENT` and `AMPHIFLOW` runs. Pure `WORKFLOW` and pure `ThinkAgent` flows need none.

```python
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration

llm = OpenAILlm(
    api_key="your-api-key",
    api_base="https://api.openai.com/v1",  # or custom endpoint
    configuration=OpenAIConfiguration(model="gpt-4o", temperature=0.0),
)
```

Other providers with same protocol: `bridgic.llms.vllm.VllmServerLlm` (self-hosted vLLM).

## Quick Start

```python
from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveContext, CognitiveWorker, think_unit,
    ThinkUnit,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 22 C in {city}"

class WeatherAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline("Look up weather and provide a summary."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        yield ThinkUnit("planner")

agent = WeatherAgent(verbose=True)
summary = await agent.arun(
    llm=llm,
    goal="Check the weather in Tokyo and London.",
    tools=[FunctionToolSpec.from_raw(get_weather)],
)
# `summary` is the post-run ctx.summary() unless a RETURN(value) was yielded
# or a finishing step set self._final_answer.
```

## Project Scaffolding

Use the CLI to bootstrap a new project:

```bash
bridgic-amphibious create
bridgic-amphibious create --task "Navigate to example.com and extract data"
bridgic-amphibious create --base-dir /path/to/project
```

Creates a single `amphi.py` in the target directory (default: cwd). The template includes a custom `CognitiveContext` subclass, an `AmphibiousAutoma` subclass with a `think_unit` declaration, and stubs for both `on_agent` and `on_workflow`. Runtime concerns (LLM credentials, entry script) are intentionally left to the caller.

## Core Concepts

**Agent = Think Units + Yield Primitives.** Agents are defined by declaring `CognitiveWorker` think units (`think_unit(...)`) and/or `AgentWorker` think agents (`think_agent(...)`) as class-level descriptors, and orchestrating them in `on_agent()` / `on_workflow()` as async generators that yield framework primitives.

**Four-layer architecture:**
1. `Exposure` — data visibility abstraction (LayeredExposure / EntireExposure)
2. `CognitiveContext` — state container (goal, tools, skills, history)
3. `CognitiveWorker` / `AgentWorker` — peer thinking units: in-process LLM observe-think-act cycle / external coding-agent delegation
4. `AmphibiousAutoma` — orchestration engine (mode routing, dispatcher, lifecycle)

**OTC Cycle (inside one think unit):** Observe -> Think -> Act, with hook points at each phase.

**Four RunModes:** `AGENT` (LLM-driven), `WORKFLOW` (deterministic), `AMPHIFLOW` (workflow + agent recovery), `AUTO` (auto-detect from overridden methods, default).

**`AUTO` resolution:** only `on_agent` overridden → `AGENT`; only `on_workflow` overridden (as an async generator) → `WORKFLOW`; both overridden → `AMPHIFLOW`. A coroutine-form `on_workflow` (e.g. `async def on_workflow(self, ctx): pass`) is treated as a stub under `AUTO` — pass `mode=RunMode.WORKFLOW` or `RunMode.AMPHIFLOW` explicitly to drive a coroutine workflow.

## Yield Primitives

Template methods are async generators. Each yielded value tells the dispatcher what to do; the dispatcher returns the result via `asend()`. Seven primitives, four scopes — mismatches raise `RuntimeError` at dispatch time.

| Primitive | Category | Allowed in | Returns to generator |
|-----------|----------|------------|----------------------|
| `ActionCall(name, **args)` | atomic Call (tool) | `on_workflow`, hooks | `List[ToolResult]` |
| `HumanCall(prompt=, channel=)` | atomic Call (HITL) | `on_workflow`, hooks | `str` |
| `LLMCall.chat(...)` / `.structure_output(...)` / `.tool_selector(...)` | atomic Call (LLM) | `on_workflow`, hooks | protocol-specific |
| `EnterAgent(goal=, tools=, skills=, history=)` | mode-switch | `on_workflow` only | `None` |
| `ThinkUnit("name", until=, max_attempts=, tools=, skills=)` | cognitive composition | `on_agent` only | worker output (or `None`) |
| `ThinkAgent("name", goal=, expose_tools=)` | cognitive composition | `on_agent` only | external agent's result `str` (or `None`) |
| `RETURN(value)` | control flow | any scope | (closes generator; value flows to caller) |

`ActionCall` / `HumanCall` / `LLMCall` are forbidden inside `on_agent` — the agent body is reserved for orchestrating cognitive steps via `ThinkUnit`. If the LLM needs to invoke a tool or ask a human, that happens *inside* a `ThinkUnit` (the worker's tool-selection phase), not by yielding from `on_agent` directly.

`RETURN(value)` is the only way to communicate a return value from an async generator (PEP 525 forbids `return value`). From a top-level `on_agent` / `on_workflow` body it sets `self._final_answer`.

## Key Patterns

### Agent Mode — LLM decides

```python
from bridgic.amphibious import ThinkUnit

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(CognitiveWorker.inline("Decide next step."), max_attempts=10)

    async def on_agent(self, ctx):
        yield ThinkUnit("worker")
```

### Workflow Mode — Developer decides

```python
from bridgic.amphibious import ActionCall

class MyWorkflow(AmphibiousAutoma[CognitiveContext]):
    async def on_workflow(self, ctx):
        result = yield ActionCall("tool_name", arg1="value")
        # result is List[ToolResult]

# Pure workflow mode does not need an LLM.
await MyWorkflow().arun(goal="...", tools=[...])
```

### Amphiflow Mode — Workflow with agent recovery

```python
from bridgic.amphibious import RunMode, ActionCall, EnterAgent, ThinkUnit

class MyHybrid(AmphibiousAutoma[CognitiveContext]):
    fixer = think_unit(CognitiveWorker.inline("Fix the problem."), max_attempts=5)

    async def on_agent(self, ctx):
        yield ThinkUnit("fixer")

    async def on_workflow(self, ctx):
        yield ActionCall("fill_field", name="user", value="john")
        yield ActionCall("click_button", name="submit")
        # Explicit handoff for an open-ended sub-task.
        yield EnterAgent(goal="Solve the captcha", tools=["solve_captcha"])

await MyHybrid().arun(
    llm=llm, goal="...", tools=[...],
    mode=RunMode.AMPHIFLOW, max_consecutive_fallbacks=2,
)
```

### Step-Level Fallback (slot + injected `resolve_step_fallback`)

In `AMPHIFLOW`, when a yielded atomic Call (`ActionCall` / `HumanCall` / `LLMCall`) raises, the framework runs `on_agent` to recover — but the workflow generator still expects a value back from `asend()`. The framework solves this by:

1. Allocating a `_FallbackSlot` with a type-appropriate default (empty `List[ToolResult]` for ActionCall, empty `str` for HumanCall, etc.).
2. Injecting an extra `resolve_step_fallback` tool into `ctx.tools` for the fallback `on_agent` run only. The agent's LLM calls it (or doesn't) to write a value into the slot.
3. After `on_agent` exhausts, the slot value is `asend()`-ed back to the suspended workflow generator, which resumes at the next instruction.

You don't import or wire `resolve_step_fallback` — the framework injects it during fallback and removes it afterwards. If the agent never calls it, the workflow receives the benign default.

`max_consecutive_fallbacks` (default 1) bounds *consecutive* recoveries. Each successful atomic Call resets the counter; once consecutive failures reach the limit, the framework abandons the workflow and runs `on_agent` for the rest of the task (full fallback).

### Human-in-the-Loop

Two entry points share the same `@human_channel` registry: the `HumanCall` yield (deterministic, from `on_workflow` or hooks), and the auto-injected `request_human` tool listed in [Built-in Tools](#built-in-tools) (LLM-driven, callable from any `ThinkUnit`).

There is **no** code-level imperative API for asking a human from `on_agent` (no `self.request_human(...)` method, and `yield HumanCall` is rejected in agent scope). The agent body is reserved for orchestrating cognitive steps — if the agent needs to ask a human, that happens inside a `ThinkUnit` where the LLM autonomously calls the `request_human` tool.

```python
from bridgic.amphibious import ActionCall, HumanCall, ThinkUnit

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(
        CognitiveWorker.inline(
            "Execute the step. Call request_human if you need confirmation."
        ),
        max_attempts=10,
    )

    async def on_agent(self, ctx):
        # The LLM inside this ThinkUnit can autonomously call the
        # auto-injected `request_human` tool — no manual wiring.
        yield ThinkUnit("worker")

    async def on_workflow(self, ctx):
        yield ActionCall("do_something", arg="value")
        feedback = yield HumanCall(prompt="Confirm?")  # deterministic HITL

# Register a @human_channel handler on the agent class to swap the
# default stdin fallback for your own UI integration. @human_channel
# is a method decorator — apply it inside an AmphibiousAutoma subclass:
#
#   class MyAgent(AmphibiousAutoma[CognitiveContext]):
#       @human_channel("my_ui")
#       async def ask_my_ui(self, prompt: str) -> str:
#           return await my_websocket.ask(prompt)
#       ...
#
# With one handler registered on the class, both HumanCall(channel=None)
# and the auto-injected request_human tool route to it implicitly.
# With multiple handlers, both sides target a channel by name —
# workflow: HumanCall(channel="feishu", ...), agent: the LLM passes
# channel="feishu" to the request_human tool. The tool spec is rebuilt
# per agent class from the @human_channel registry, so the LLM sees the
# real channel names (description + enum-constrained `channel` param)
# without you having to spell them out in the system_prompt.
await MyAgent().arun(llm=llm, goal="...", tools=[my_tool])
```

### Custom Pydantic Output

```python
from pydantic import BaseModel
from bridgic.amphibious import ThinkUnit, RETURN

class Plan(BaseModel):
    phases: list[str]

class Planner(AmphibiousAutoma[CognitiveContext]):
    plan = think_unit(
        CognitiveWorker.inline("Create a plan.", output_schema=Plan),
        max_attempts=1,
    )

    async def on_agent(self, ctx):
        plan = yield ThinkUnit("plan")  # `plan` is a Plan instance
        yield RETURN(plan.model_dump_json())
```

### Phase Annotation (snapshot)

```python
async def on_agent(self, ctx):
    async with self.snapshot(goal="Research phase"):
        yield ThinkUnit("researcher")
    async with self.snapshot(goal="Writing phase"):
        yield ThinkUnit("writer")
```

### External Agent (ThinkAgent)

`think_agent` declares an `AgentWorker` think agent — the external-agent peer of `think_unit`. Instead of an in-process LLM cycle, it hands a sub-goal to an out-of-process coding-agent CLI (claude code, etc.). The external agent reaches the parent's project tools through an in-process MCP bridge, so every tool call it makes still flows through the parent's `before_action` / `after_action` hooks and the trace.

```python
from bridgic.amphibious import (
    AgentWorker, ClaudeCodeAgent, ThinkAgent, think_agent, RETURN,
)

class Planner(AmphibiousAutoma[CognitiveContext]):
    # AgentWorker is anchored on a BaseAgent the way CognitiveWorker is
    # anchored on a BaseLlm. ClaudeCodeAgent / CodexAgent are the drivers.
    planner = think_agent(
        AgentWorker(ClaudeCodeAgent(completion_timeout=300.0)),
    )

    async def on_agent(self, ctx):
        # yield ThinkAgent returns the string the external agent passed
        # to its `agent_done` completion signal.
        summary = yield ThinkAgent("planner", goal="Draft a 4-step plan.")
        yield RETURN(summary)

# A pure ThinkAgent flow needs no `llm` — the external agent reasons.
await Planner().arun()
```

CLI-level knobs live on the `BaseAgent`; `expose_tools` filters which project tools reach the external agent. Two `BaseAgent` drivers ship with the framework — `ClaudeCodeAgent` (claude code) and `CodexAgent` (OpenAI codex); each spawns its CLI as a subprocess, which must be installed, on `PATH`, and authenticated. For the full surface see [references/api-reference.md](references/api-reference.md#agentworker--baseagent).

## Built-in Tools

Every `AmphibiousAutoma` agent receives seven built-in tools in `context.tools` automatically during `arun()` — no manual wiring. Tool names are snake_case and work in every mode (LLM-called in agent mode, `yield ActionCall("name", ...)` in workflow mode).

| Tool | What it does |
|------|--------------|
| `request_human` | Pause and ask the human operator a question (HITL) |
| `bash` | Execute a shell command. Returns raw `stdout`; non-zero exit raises `RuntimeError`. |
| `read_file` | Read a file with line numbers; required before `write_file` / `edit_file` modify it |
| `write_file` | Create a new file, or overwrite an existing one (read-before-overwrite enforced) |
| `edit_file` | Exact-string replacement with uniqueness check; `replace_all` for refactors |
| `glob` | Find files by pattern, sorted by mtime |
| `grep` | Regex content search across files |

Subclasses can opt out of specific built-ins via a class-level frozenset, and runs can override per-call:

```python
class ReadOnlyAgent(AmphibiousAutoma[CognitiveContext]):
    # Only these three are injected; bash / write / edit are unavailable.
    builtin_tools = frozenset({"request_human", "read_file", "grep"})

# Runtime override (wins over class attr); empty iterable opts out entirely.
await agent.arun(goal="...", builtin_tools=["request_human"])
```

Unknown names raise `ValueError` at `arun()` entry — typos surface immediately rather than silently producing a missing-tool agent. User-supplied tools whose name collides with a built-in win (deduplicated by `tool_name`).

`write_file` and `edit_file` enforce a read-before-modify invariant: the path must have been read with `read_file` first, AND the file must not have been changed externally since. The tracker is reset at every `arun()` entry, so the invariant is scoped to a single run.

For the full per-tool parameter list, error contracts, and filter resolution rules see [references/api-reference.md](references/api-reference.md#built-in-tools).

## Reference Files

- **Architecture details** (state-machine dispatcher, fallback mechanism, exposure system, memory tiers, cognitive policies): See [references/architecture.md](references/architecture.md)
- **Complete API reference** (all classes, methods, parameters, types, yield primitives): See [references/api-reference.md](references/api-reference.md)
- **Full code patterns and examples** (all hook types, skills, tracing, filtering, etc.): See [references/patterns.md](references/patterns.md)
