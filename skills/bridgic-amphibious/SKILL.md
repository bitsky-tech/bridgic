---
name: bridgic-amphibious
description: "Build agents with the Bridgic Amphibious dual-mode framework — combining LLM-driven (agent) and deterministic (workflow) execution with automatic fallback, human-in-the-loop, and a built-in tool surface (shell, filesystem, search, HITL) auto-injected into every agent. Use when: (1) writing code that imports from bridgic.amphibious, (2) creating AmphibiousAutoma subclasses, (3) defining CognitiveWorker think units, (4) implementing on_agent/on_workflow methods, (5) working with CognitiveContext, Exposure system, or cognitive policies, (6) adding human-in-the-loop interactions (HumanCall, request_human, request_human_tool), (7) using or filtering the auto-injected built-in tools (bash, read_file/write_file/edit_file, glob, grep, request_human) via the builtin_tools class attribute or arun kwarg, (8) scaffolding a new amphibious project via CLI, (9) any task involving the bridgic-amphibious framework."
---

# Bridgic Amphibious

Dual-mode agent framework: agents operate in LLM-driven (`on_agent`) and deterministic (`on_workflow`) modes with automatic fallback between them.

## Dependencies

A bridgic-amphibious project requires the following packages:

| Package | Description |
|---------|-------------|
| `bridgic-core` | Core framework (Worker, Automa, GraphAutoma) |
| `bridgic-amphibious` | Dual-mode agent framework |
| `bridgic-llms-openai` | LLM provider (only required for `AGENT` / `AMPHIFLOW` modes) |
| `python-dotenv` | `.env` file loading |

Before using this package, you need to install the dependencies by using the provided install script:

```bash
bash "skills/bridgic-amphibious/scripts/install-deps.sh" "$PWD"
```

The script checks uv availability, initializes a uv project if needed, installs any missing packages via `uv add`, and runs `uv sync` to finalize the environment. When it exits successfully the project is fully initialized and ready to use — no manual `uv add` / `uv sync` follow-up is required.

## LLM Setup

Amphibious agents accept a `BaseLlm` instance with `astructure_output` protocol from a bridgic LLM provider package. The LLM is required for `AGENT` and `AMPHIFLOW` modes; pure `WORKFLOW` mode can run without one.

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
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 22°C in {city}"

class WeatherAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline("Look up weather and provide a summary."),
        max_attempts=5,
    )
    async def on_agent(self, ctx: CognitiveContext):
        await self.planner

agent = WeatherAgent(llm=llm, verbose=True)
result = await agent.arun(
    goal="Check the weather in Tokyo and London.",
    tools=[FunctionToolSpec.from_raw(get_weather)],
)
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

**Agent = Think Units + Context Orchestration.** Agents are defined by declaring `CognitiveWorker` think units and orchestrating them in `on_agent()` or `on_workflow()`.

**Four-layer architecture:**
1. `Exposure` — data visibility abstraction (LayeredExposure / EntireExposure)
2. `CognitiveContext` — state container (goal, tools, skills, history)
3. `CognitiveWorker` — pure thinking unit (observe-think-act)
4. `AmphibiousAutoma` — orchestration engine (mode routing, lifecycle)

**OTC Cycle:** Observe -> Think -> Act, with hook points at each phase.

**Four RunModes:** `AGENT` (LLM-driven), `WORKFLOW` (deterministic), `AMPHIFLOW` (workflow + agent fallback), `AUTO` (auto-detect from overridden methods, default).

**`AUTO` resolution:** only `on_agent` overridden → `AGENT`; only `on_workflow` overridden → `WORKFLOW`; both overridden → `AMPHIFLOW`.

## Built-in Tools

Every `AmphibiousAutoma` agent receives seven built-in tools in `context.tools` automatically during `arun()` — no manual wiring. Tool names are snake_case and work in every mode (LLM-called in agent mode, `yield ActionCall("name", ...)` in workflow mode).

| Tool | What it does |
|------|--------------|
| `request_human` | Pause and ask the human operator a question (HITL) |
| `bash` | Execute a shell command (stdout / stderr / exit_code captured) |
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

## Key Patterns

### Agent Mode — LLM decides

```python
class MyAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(CognitiveWorker.inline("Decide next step."), max_attempts=10)
    async def on_agent(self, ctx):
        await self.worker
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

### Amphiflow Mode — Workflow with agent fallback

```python
from bridgic.amphibious import RunMode, AgentCall

class MyHybrid(AmphibiousAutoma[CognitiveContext]):
    fixer = think_unit(CognitiveWorker.inline("Fix the problem."), max_attempts=5)
    async def on_agent(self, ctx): await self.fixer
    async def on_workflow(self, ctx):
        yield ActionCall("fill_field", name="user", value="john")
        yield ActionCall("click_button", name="submit")

await MyHybrid(llm=llm).arun(
    goal="...", tools=[...],
    mode=RunMode.AMPHIFLOW, max_consecutive_fallbacks=2,
)
```

### Human-in-the-Loop

Three entry points share one event channel: the code-level `request_human()` method, the `HumanCall` workflow yield, and the auto-injected `request_human` tool listed in [Built-in Tools](#built-in-tools).

```python
from bridgic.amphibious import ActionCall, HumanCall

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(CognitiveWorker.inline("Execute step."), max_attempts=10)

    async def on_agent(self, ctx):
        await self.worker
        feedback = await self.request_human("Proceed?")  # Entry 1: code-level

    async def on_workflow(self, ctx):
        yield ActionCall("do_something", arg="value")
        feedback = yield HumanCall(prompt="Confirm?")     # Entry 2: workflow yield

# Entry 3 is automatic — the built-in `request_human` tool is already in
# context.tools, so the LLM can call it without listing it in tools=[...].
# Override `human_input(data)` to swap the default stdin read for your own
# UI integration (WebSocket, HTTP callback, Slack bot, etc.).
await MyAgent(llm=llm).arun(goal="...", tools=[my_tool])
```

### Custom Pydantic Output

```python
from pydantic import BaseModel

class Plan(BaseModel):
    phases: list[str]

class Planner(AmphibiousAutoma[CognitiveContext]):
    plan = think_unit(
        CognitiveWorker.inline("Create a plan.", output_schema=Plan),
        max_attempts=1,
    )
    async def on_agent(self, ctx):
        result = await self.plan  # Returns Plan instance
```

### Phase Annotation (snapshot)

```python
async def on_agent(self, ctx):
    async with self.snapshot(goal="Research phase"):
        await self.researcher
    async with self.snapshot(goal="Writing phase"):
        await self.writer
```

## Reference Files

- **Architecture details** (execution modes, exposure system, memory tiers, cognitive policies): See [references/architecture.md](references/architecture.md)
- **Complete API reference** (all classes, methods, parameters, types): See [references/api-reference.md](references/api-reference.md)
- **Full code patterns and examples** (all hook types, skills, tracing, filtering, etc.): See [references/patterns.md](references/patterns.md)
