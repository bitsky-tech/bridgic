---
name: bridgic-amphibious
description: "Build agents with the Bridgic Amphibious dual-mode framework — combining LLM-driven (agent) and deterministic (workflow) execution with peer state-machine dispatch, step-level fallback (a bounded agent recovery sub-run), human-in-the-loop, and a two-loop context model: a small-loop OTAContext (framework-owned: this run's user_input + observe-think-act round trace + tools) and a free-form big-loop Context (cross-turn knowledge). Tools are declared on the OTA context via OTAContext.tool — nothing is auto-injected. Sub-goals can be delegated to external coding agents (claude code, codex) as think-agent units. Use when: (1) writing code that imports from bridgic.amphibious, (2) creating AmphibiousAutoma[OTAContext, Context] subclasses, (3) defining CognitiveWorker think units (think_unit / ThinkUnit) and AgentWorker think agents (think_agent / ThinkAgent), yielding ThinkUnit / ThinkAgent / EnterAgent / ActionCall / HumanCall / LLMCall / RETURN, (4) implementing on_agent / on_workflow methods, (5) working with the OTAContext / Context two-loop model or declaring tools via OTAContext.tool, (6) adding human-in-the-loop interactions (HumanCall, request_human, request_human_tool, @human_channel), (7) using the built-in tool specs (bash, read_file/write_file/edit_file, glob, grep, request_human) by declaring them on the OTA context, (8) scaffolding a new amphibious project via CLI, (9) any task involving the bridgic-amphibious framework."
---

# Bridgic Amphibious

Dual-mode agent framework: agents operate in LLM-driven (`on_agent`) and deterministic (`on_workflow`) modes. In hybrid `AMPHIFLOW` mode the framework runs a peer state machine over both, recovering from a failed workflow step by running a bounded `on_agent` sub-run.

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

Pass a `BaseLlm` instance (from a bridgic LLM provider package) to `arun(llm=...)`. An LLM is required for any run that uses a `CognitiveWorker` (`ThinkUnit`) or `LLMCall` — i.e. typical `AGENT` and `AMPHIFLOW` runs. Pure `WORKFLOW` and pure `ThinkAgent` flows need none (the external agent does the reasoning).

```python
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration

llm = OpenAILlm(
    api_key="your-api-key",
    api_base="https://api.openai.com/v1",  # or custom endpoint
    configuration=OpenAIConfiguration(model="gpt-4o", temperature=0.0),
)
```

Other providers with the same interface: `bridgic.llms.openai_like.OpenAILikeLlm`, `bridgic.llms.vllm.VllmServerLlm` (self-hosted vLLM).

## The Two-Loop Context Model

Every amphibious agent is parameterized by **two** context types — `AmphibiousAutoma[OTAContextT, ContextT]`:

- **`OTAContext`** — the **small-loop**, **framework-owned** working context. It holds this run's `user_input`, the observe-think-act round trace (`ota_record: List[OTARecord]`), and the `tools` the run carries. The framework constructs a fresh one per `arun()`. Tools are **declared on the class** via `OTAContext.tool(...)`.
- **`Context`** — the **big-loop**, **free-form** knowledge context (cross-turn state: memory, conversation, domain knowledge). Define fields and optionally override `summary(self, fields)`; your worker's `thinking()` folds it into the prompt when you need it (nothing is auto-injected). Optional — pass it via `arun(context=...)`; it is shared read-only across the run and any delegation.

Tools are **not** a base-`Context` concern — they belong to the OTA loop that actually acts. **Nothing is auto-injected**; whatever a context declares via `OTAContext.tool` is exactly what its `tools` field holds.

## Quick Start

```python
from bridgic.amphibious import (
    AmphibiousAutoma, OTAContext, Context,
    CognitiveWorker, think_unit, ThinkUnit, RETURN,
)
from bridgic.core.model.types import Message, Role

# 1. A tool — a plain async function.
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 22 C in {city}"

# 2. Small-loop context — declare the tools this run carries (nothing is
#    auto-injected). `tool` works as a decorator OR a call.
class WeatherOTAContext(OTAContext):
    pass

WeatherOTAContext.tool(get_weather)

# 3. Big-loop context — free-form cross-turn knowledge (optional).
class WeatherContext(Context):
    pass

# 4. Think worker — assemble a prompt from the contexts, call the model, and
#    return its NATURAL result; the framework adapts it into a decision.
class WeatherThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        messages = [Message.from_text(ota_context.summary(), role=Role.USER)]
        return await self._llm.aselect_tool(
            messages=messages,
            tools=[t.to_tool() for t in ota_context.tools],
        )

# 5. Agent — declare the think unit, orchestrate it in on_agent.
class WeatherAgent(AmphibiousAutoma[WeatherOTAContext, WeatherContext]):
    planner = think_unit(WeatherThink(), max_attempts=5)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("planner")

# 6. Run — `user_input` seeds the fresh small-loop OTA context.
agent = WeatherAgent(verbose=True)
answer = await agent.arun(llm=llm, user_input="Check the weather in Tokyo and London.")
# `answer` is the finishing think step's step_content (or a RETURN(value) you
# yielded, else ota_ctx.summary()). Also available as agent.final_answer.
```

## Project Scaffolding

Use the CLI to bootstrap a new project:

```bash
bridgic-amphibious create
bridgic-amphibious create --task "Navigate to example.com and extract data"
bridgic-amphibious create --base-dir /path/to/project
```

Creates a single `amphi.py` in the target directory (default: cwd). The template includes a custom `OTAContext` subclass with the built-in tools declared via `OTAContext.tool`, a free-form `Context` subclass, a `CognitiveWorker` with a `thinking()` override, an `AmphibiousAutoma[OTAContext, Context]` subclass with a `think_unit` declaration, and stubs for both `on_agent` and `on_workflow`. Runtime concerns (LLM credentials, entry script) are intentionally left to the caller.

## Core Concepts

**Agent = Think Units + Orchestration.** Agents are defined by declaring `CognitiveWorker` think units (`think_unit(...)`) and/or `AgentWorker` think agents (`think_agent(...)`) as class-level descriptors, and orchestrating them in `on_agent()` / `on_workflow()` as async generators that yield framework primitives.

**Three layers:**
1. **Context** — `OTAContext` (small-loop, framework-owned: `user_input` + `ota_record` round trace + `tools`) and `Context` (free-form big-loop knowledge). Tools live on `OTAContext` via `OTAContext.tool`.
2. **Worker** — peer think units. `CognitiveWorker` runs one in-process observe-think-act cycle anchored on a `BaseLlm` (override its single `thinking()` template method). `AgentWorker` delegates one cycle to an external coding-agent CLI, anchored on a `BaseAgent` (`ClaudeCodeAgent` / `CodexAgent`).
3. **Orchestration** — `AmphibiousAutoma` (mode routing, the peer state-machine dispatcher, lifecycle, trace).

**OTA cycle (one round):** Observe → Think → Act. Observation is set on the round (`ota_ctx.obs_result`); the worker's `thinking()` produces the decision (`ota_ctx.think_result`); the action phase executes its tool calls (`ota_ctx.action_result`). One round = one `OTARecord`.

**Four RunModes:** `AGENT` (LLM-driven), `WORKFLOW` (deterministic), `AMPHIFLOW` (workflow + agent recovery), `AUTO` (auto-detect from overridden methods, default).

**`AUTO` resolution:** only `on_agent` overridden → `AGENT`; only `on_workflow` overridden → `WORKFLOW`; both overridden → `AMPHIFLOW`. All overridable template methods must be **async generators** (yield-driven). The framework validates this at class-creation time; if a body has no real yields, add `if False: yield` as an unreachable stub.

## Yield Primitives

Template methods are async generators. Each yielded value tells the dispatcher what to do; the dispatcher returns the result via `asend()`. Seven primitives, scoped by mode — mismatches raise `RuntimeError` at dispatch time.

| Primitive | Category | Allowed in | Returns to generator |
|-----------|----------|------------|----------------------|
| `ActionCall(name, **args)` | atomic Call (tool) | `on_workflow`, hooks | `List[ToolResult]` |
| `HumanCall(prompt=, channel=)` | atomic Call (HITL) | `on_workflow`, hooks | `str` |
| `LLMCall.chat(...)` / `.structure_output(...)` / `.tool_selector(...)` | atomic Call (LLM) | `on_workflow`, hooks | protocol-specific |
| `EnterAgent(goal=)` | mode-switch | `on_workflow` only | `None` (workflow resumes) |
| `ThinkUnit("name", until=, max_attempts=)` | cognitive composition | `on_agent` only | finishing think's `step_content` |
| `ThinkAgent("name", goal=, expose_tools=)` | cognitive composition | `on_agent` only | external agent's result `str` (or `None`) |
| `RETURN(value)` | control flow | any scope | (closes generator; value flows to caller) |

`ActionCall` / `HumanCall` / `LLMCall` are forbidden inside `on_agent` — the agent body is reserved for orchestrating cognitive steps via `ThinkUnit` / `ThinkAgent`. If the LLM needs to invoke a tool or ask a human, that happens *inside* a `ThinkUnit` (the worker's tool-selection phase, or the LLM calling the declared `request_human` tool), not by yielding from `on_agent` directly.

`RETURN(value)` is the only way to communicate a return value from an async generator (PEP 525 forbids `return value`). From a top-level `on_agent` / `on_workflow` body it sets `self._final_answer`.

## Key Patterns

### Agent Mode — LLM decides

```python
from bridgic.amphibious import ThinkUnit

class MyAgent(AmphibiousAutoma[MyOTAContext, MyContext]):
    worker = think_unit(MyThink(), max_attempts=10)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("worker")
```

### Workflow Mode — Developer decides

```python
from bridgic.amphibious import ActionCall, RETURN

class MyWorkflow(AmphibiousAutoma[MyOTAContext, MyContext]):
    async def on_workflow(self, ota_context, context=None):
        result = yield ActionCall("tool_name", arg1="value")  # List[ToolResult]
        yield RETURN(result[0].result if result else "N/A")

# Pure workflow mode does not need an LLM.
await MyWorkflow().arun(user_input="...")
```

### Amphiflow Mode — Workflow with agent recovery

```python
from bridgic.amphibious import RunMode, ActionCall, EnterAgent, ThinkUnit

class MyHybrid(AmphibiousAutoma[MyOTAContext, MyContext]):
    fixer = think_unit(MyThink(), max_attempts=5)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("fixer")

    async def on_workflow(self, ota_context, context=None):
        yield ActionCall("fill_field", name="user", value="john")
        yield ActionCall("click_button", name="submit")
        # Explicit handoff for an open-ended sub-task.
        yield EnterAgent(goal="Solve the captcha")

await MyHybrid().arun(
    llm=llm, user_input="...",
    mode=RunMode.AMPHIFLOW, max_consecutive_fallbacks=2,
)
```

### Step-Level Fallback (bounded recovery sub-run)

In `AMPHIFLOW`, when a yielded atomic Call (`ActionCall` / `HumanCall` / `LLMCall`) raises, the framework decides between step-level recovery and full fallback by a single counter:

1. `consecutive_failures += 1`. Each successful atomic Call resets it to `0`.
2. If `consecutive_failures >= max_consecutive_fallbacks` → **full fallback**: the workflow generator is closed and `on_agent` runs for the rest of the task.
3. Otherwise → **step-level recovery**: the framework runs a *bounded* `on_agent` sub-run (a fresh OTA episode whose `user_input` describes the failed step + its error). The sub-run's conclusion is shaped into the failed step's return type and `asend()`-ed back to the suspended workflow, which resumes at the next instruction.

There is no injected tool and no toolset mutation — the recovery sub-run's own conclusion *is* the resolution. A workflow generator-internal error (helper code between yields raising) is unrecoverable in place, so it escalates straight to full fallback.

`max_consecutive_fallbacks` (default 1) bounds *consecutive* recoveries before full fallback takes over.

### Human-in-the-Loop

Two entry points share the same `@human_channel` registry: the `HumanCall` yield (deterministic, from `on_workflow` or hooks), and the `request_human` tool (LLM-driven, callable from any `ThinkUnit`). The `request_human` tool is **declared on the OTA context** like any other tool — `MyOTAContext.tool(request_human_tool)` — it is not auto-injected.

There is **no** code-level imperative API for asking a human from `on_agent` (no `self.request_human(...)` method, and `yield HumanCall` is rejected in agent scope). If the agent needs to ask a human, that happens inside a `ThinkUnit` where the LLM autonomously calls the declared `request_human` tool.

```python
from bridgic.amphibious import (
    OTAContext, ActionCall, HumanCall, ThinkUnit, human_channel,
    request_human_tool,
)

class MyOTAContext(OTAContext):
    pass

MyOTAContext.tool(request_human_tool)   # the LLM can now call request_human

class MyAgent(AmphibiousAutoma[MyOTAContext, MyContext]):
    worker = think_unit(MyThink(), max_attempts=10)

    # Register a @human_channel handler to swap the default stdin fallback
    # for your own UI. It is a method decorator on the AmphibiousAutoma
    # subclass; the handler is a plain async method returning str.
    @human_channel("my_ui")
    async def ask_my_ui(self, prompt: str) -> str:
        return await my_websocket.ask(prompt)

    async def on_agent(self, ota_context, context=None):
        # The LLM inside this ThinkUnit can autonomously call request_human.
        yield ThinkUnit("worker")

    async def on_workflow(self, ota_context, context=None):
        yield ActionCall("do_something", arg="value")
        feedback = yield HumanCall(prompt="Confirm?")  # deterministic HITL
```

With one handler registered, both `HumanCall(channel=None)` and the `request_human` tool route to it implicitly. With multiple handlers, address them by name — workflow: `HumanCall(channel="feishu", ...)`; agent: the LLM passes `channel="feishu"` to `request_human`. With zero handlers, both fall back to stdin.

### External Agent (ThinkAgent)

`think_agent` declares an `AgentWorker` think agent — the external-agent peer of `think_unit`. Instead of an in-process LLM cycle, it hands a sub-goal to an out-of-process coding-agent CLI. The external agent reaches the parent's project tools through an in-process MCP bridge, so every tool call it makes still flows through the parent's `before_action` / `after_action` hooks and the trace.

```python
from bridgic.amphibious import (
    AgentWorker, ClaudeCodeAgent, ThinkAgent, think_agent, RETURN,
)

class Planner(AmphibiousAutoma[MyOTAContext, MyContext]):
    # AgentWorker is anchored on a BaseAgent the way CognitiveWorker is
    # anchored on a BaseLlm. ClaudeCodeAgent / CodexAgent are the drivers.
    planner = think_agent(
        AgentWorker(ClaudeCodeAgent(completion_timeout=300.0)),
    )

    async def on_agent(self, ota_context, context=None):
        # yield ThinkAgent returns the string the external agent passed to
        # its `agent_done` completion signal.
        summary = yield ThinkAgent("planner", goal="Draft a 4-step plan.")
        yield RETURN(summary)

# A pure ThinkAgent flow needs no `llm` — the external agent reasons.
await Planner().arun(user_input="Plan the migration.")
```

CLI-level knobs live on the `BaseAgent`; `expose_tools` (on `think_agent` / `ThinkAgent`) filters which declared project tools reach the external agent. Two `BaseAgent` drivers ship with the framework — `ClaudeCodeAgent` (claude code) and `CodexAgent` (OpenAI codex); each spawns its CLI as a subprocess, which must be installed, on `PATH`, and authenticated. For the full surface see [references/api-reference.md](references/api-reference.md#agentworker--baseagent).

## Built-in Tools

The framework ships seven `FunctionToolSpec` instances. **Nothing is auto-injected** — declare the ones a run needs on its OTA context class via `OTAContext.tool` (decorator or call). Tool names are snake_case and work in every mode (LLM-called in agent mode, `yield ActionCall("name", ...)` in workflow mode).

| Tool | What it does |
|------|--------------|
| `request_human` | Pause and ask the human operator a question (HITL) |
| `bash` | Execute a shell command. Returns raw `stdout`; non-zero exit raises `RuntimeError`. |
| `read_file` | Read a file with line numbers; required before `write_file` / `edit_file` modify it |
| `write_file` | Create a new file, or overwrite an existing one (read-before-overwrite enforced) |
| `edit_file` | Exact-string replacement with uniqueness check; `replace_all` for refactors |
| `glob` | Find files by pattern, sorted by mtime |
| `grep` | Regex content search across files |

```python
from bridgic.amphibious import (
    OTAContext, bash_tool, read_file_tool, write_file_tool, edit_file_tool,
    glob_tool, grep_tool, request_human_tool,
)

class CodeOTAContext(OTAContext):
    pass

# Declare exactly the built-ins this context carries.
for _t in (bash_tool, read_file_tool, write_file_tool, edit_file_tool,
           glob_tool, grep_tool, request_human_tool):
    CodeOTAContext.tool(_t)

# Or declare the whole set at once:
#   from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS
#   for _t in ALL_BUILTIN_TOOLS: CodeOTAContext.tool(_t)
```

`write_file` and `edit_file` enforce a read-before-modify invariant: the path must have been read with `read_file` first, AND the file must not have been changed externally since. The tracker is reset at every `arun()` entry, so the invariant is scoped to a single run.

For the full per-tool parameter list and error contracts see [references/api-reference.md](references/api-reference.md#built-in-tools).

## Reference Files

- **Architecture details** (two-loop context model, state-machine dispatcher, fallback mechanism, OTA round trace, external-agent delegation): See [references/architecture.md](references/architecture.md)
- **Complete API reference** (all classes, methods, parameters, types, yield primitives): See [references/api-reference.md](references/api-reference.md)
- **Full code patterns and examples** (all hook types, custom contexts, tracing, etc.): See [references/patterns.md](references/patterns.md)
