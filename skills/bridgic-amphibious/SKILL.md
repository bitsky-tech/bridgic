---
name: bridgic-amphibious
description: "Build agents with the Bridgic Amphibious dual-mode framework — combining LLM-driven (agent) and deterministic (workflow) execution with automatic fallback and human-in-the-loop support. Use when: (1) writing code that imports from bridgic.amphibious, (2) creating AmphibiousAutoma subclasses, (3) defining CognitiveWorker think units, (4) implementing on_agent/on_workflow methods, (5) working with CognitiveContext, Exposure system, or cognitive policies, (6) adding human-in-the-loop interactions (HumanCall, request_human, human_request_tool), (7) scaffolding a new amphibious project via CLI, (8) any task involving the bridgic-amphibious framework."
---

# Bridgic Amphibious

Dual-mode agent framework: agents operate in LLM-driven (`on_agent`) and deterministic (`on_workflow`) modes with automatic fallback between them.

## LLM Setup

Amphibious agents require a `BaseLlm` instance with `astructure_output` protocol from a bridgic LLM provider package:

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

**Four RunModes:** `AGENT` (LLM-driven), `WORKFLOW` (deterministic), `AMPHIBIOUS` (workflow + fallback), `AUTO` (auto-detect, default).

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
    async def on_agent(self, ctx): pass
    async def on_workflow(self, ctx):
        result = yield ActionCall("tool_name", arg1="value")
        # result is List[ToolResult]
```

### Amphibious Mode — Workflow with agent fallback

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
    mode=RunMode.AMPHIBIOUS, will_fallback=True, max_consecutive_fallbacks=2,
)
```

### Human-in-the-Loop

```python
from bridgic.amphibious import ActionCall, HumanCall
from bridgic.amphibious.builtin_tools import human_request_tool

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(CognitiveWorker.inline("Execute step."), max_attempts=10)

    async def on_agent(self, ctx):
        await self.worker
        feedback = await self.request_human("Proceed?")  # Entry 1: code-level

    async def on_workflow(self, ctx):
        yield ActionCall("do_something", arg="value")
        feedback = yield HumanCall(prompt="Confirm?")     # Entry 2: workflow yield

# Entry 3: LLM tool — human_request_tool is a plain FunctionToolSpec
await MyAgent(llm=llm).arun(goal="...", tools=[my_tool, human_request_tool])
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
