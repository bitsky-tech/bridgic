---
name: bridgic-amphibious
description: "Build agents with the Bridgic Amphibious dual-mode framework — combining LLM-driven (agent) and deterministic (workflow) execution with automatic fallback and human-in-the-loop support. Use when: (1) writing code that imports from bridgic.amphibious, (2) creating AmphibiousAutoma subclasses, (3) defining CognitiveWorker think units, (4) implementing on_agent/on_workflow methods, (5) working with CognitiveContext, Exposure system, or cognitive policies, (6) adding human-in-the-loop interactions (HumanCall, request_human, request_human_tool), (7) scaffolding a new amphibious project via CLI, (8) any task involving the bridgic-amphibious framework."
---

# Bridgic Amphibious

Dual-mode agent framework: agents operate in LLM-driven (`on_agent`) and deterministic (`on_workflow`) modes with automatic fallback between them.

## Dependencies

A bridgic-amphibious project requires the following packages:

| Package | Description |
|---------|-------------|
| `bridgic-core` | Core framework (Worker, Automa, GraphAutoma, ASL) |
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
bridgic-amphibious create -n my_project
bridgic-amphibious create -n my_project --task "Navigate to example.com and extract data"
bridgic-amphibious create -n my_project --base-dir /path/to/projects
```

Creates: `task.md`, `config.py`, `tools.py`, `workers.py`, `agents.py`, `skills/`, `result/`, `log/`.

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

# Entry 3: LLM tool — `request_human` is auto-injected into every agent's
# tools, so the LLM can call it without you listing it in tools=[...].
# (Passing `request_human_tool` explicitly is harmless — it is deduped.)
await MyAgent(llm=llm).arun(goal="...", tools=[my_tool])

# If you want to be explicit, importing and passing `request_human_tool` still OK.
from bridgic.amphibious.builtin_tools import request_human_tool
await agent.arun(goal="...", tools=[request_human_tool, ...])  # also fine
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
