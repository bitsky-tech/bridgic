# Bridgic Amphibious API Reference

## Table of Contents
- [LLM Setup](#llm-setup)
- [Imports](#imports)
- [CLI Scaffolding](#cli-scaffolding)
- [AmphibiousAutoma](#amphibiousautoma)
- [CognitiveWorker](#cognitiveworker)
- [AgentWorker & BaseAgent](#agentworker--baseagent)
- [think_unit](#think_unit)
- [think_agent](#think_agent)
- [Yield Primitives](#yield-primitives)
- [Human-in-the-Loop](#human-in-the-loop)
- [Built-in Tools](#built-in-tools)
- [CognitiveContext](#cognitivecontext)
- [Context and Exposure](#context-and-exposure)
- [Data Models](#data-models)
- [Tool Definition](#tool-definition)
- [AgentTrace](#agenttrace)

---

## LLM Setup

`arun(llm=...)` accepts a `BaseLlm` instance from one of the bridgic LLM provider packages. The LLM is required for `AGENT` and `AMPHIFLOW` modes; pure `WORKFLOW` mode (where `on_workflow` is the only overridden template method) — and a pure `ThinkAgent` flow, where an external agent does the reasoning — can run without one. Install one of the provider packages:

```python
# OpenAI (GPT-4, GPT-4o, etc.)
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration

llm = OpenAILlm(
    api_key="your-api-key",
    api_base=None,                    # Optional: custom base URL
    timeout=120,                      # Optional: request timeout
    configuration=OpenAIConfiguration(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=16384,
    ),
)

# OpenAI-compatible APIs (third-party providers)
from bridgic.llms.openai_like import OpenAILikeLlm, OpenAILikeConfiguration

llm = OpenAILikeLlm(
    api_base="https://api.provider.com/v1",  # Required
    api_key="provider-api-key",               # Required
    configuration=OpenAILikeConfiguration(model="model-name"),
)

# Self-hosted vLLM
from bridgic.llms.vllm import VllmServerLlm, VllmServerConfiguration

llm = VllmServerLlm(
    api_base="http://localhost:8000/v1",  # Required
    api_key="vllm-key",                    # Required
    configuration=VllmServerConfiguration(model="meta-llama/Llama-2-70b"),
)
```

Configuration class parameters (shared across providers): `model`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `max_tokens`, `stop`.

## Imports

```python
from bridgic.amphibious import (
    # Orchestration
    AmphibiousAutoma, AgentTrace,
    think_unit, ThinkUnitDescriptor,
    think_agent, ThinkAgentDescriptor,
    # Worker — in-process (CognitiveWorker) + external-agent (AgentWorker)
    CognitiveWorker, _DELEGATE,
    AgentWorker, BaseAgent, ClaudeCodeAgent, AgentRequest, AgentResult,
    # Context
    CognitiveContext, CognitiveHistory, CognitiveTools, CognitiveSkills,
    Context, Exposure, LayeredExposure, EntireExposure,
    # Yield primitives
    ActionCall, HumanCall, LLMCall, EnterAgent, ThinkUnit, ThinkAgent, RETURN,
    # Human channel registry
    human_channel,
    # Data models
    Step, Skill, RunMode, ErrorStrategy,
    ActionResult, ActionStepResult, ToolResult,
    WorkflowDecision, StepToolCall, ToolArgument, DetailRequest,
    # Trace
    TraceStep, RecordedToolCall, StepOutputType,
    # Built-in tool specs (auto-injected; importable for explicit reuse)
    request_human_tool, bash_tool,
    read_file_tool, write_file_tool, edit_file_tool,
    glob_tool, grep_tool,
)
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message
```

## CLI Scaffolding

Bootstrap a new amphibious project:

```bash
bridgic-amphibious create [--base-dir <path>] [--task <description>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base-dir` | Current directory | Target directory for the generated file |
| `--task` | (omitted) | Injected as a top-level `# Task: ...` comment in `amphi.py` |

Generated file:

```
amphi.py    # AmphibiousAutoma stub: AmphiContext + Amphi class with think_unit, on_agent, on_workflow
```

The scaffold writes only this single file in the target directory. It does not create subdirectories and does not emit runtime configuration (e.g. `.env`) — those concerns belong to the caller's environment, not the scaffold.

Python API: `create_project(base_dir: Optional[str] = None, task: Optional[str] = None) -> Path`. Raises `FileExistsError` if `amphi.py` already exists in the target directory.

## AmphibiousAutoma

```python
class AmphibiousAutoma(Generic[CognitiveContextT]):
```

Base class for dual-mode agents. Subclass with a generic `CognitiveContext` type parameter.

### Constructor

```python
AmphibiousAutoma(
    name: str = None,             # Optional agent name
    verbose: bool = False,        # Enable execution logging
    verbose_hook: bool = False,   # Surface dispatch logs for Calls yielded from
                                  # observation / before_action / after_action hooks.
                                  # Suppressed by default — hook-yielded Calls are
                                  # internal side-effects, not workflow narrative.
)
```

The LLM is **not** a constructor argument — it is bound per run via `arun(llm=...)`.

### Class Attributes (Override in Subclasses)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `builtin_tools` | `Optional[FrozenSet[str]]` | `None` | Filter for which entries of [`ALL_BUILTIN_TOOLS`](#built-in-tools) to auto-inject during `arun()`. `None` injects all; a frozenset of names restricts to that subset; `frozenset()` opts out entirely. The runtime `arun(builtin_tools=...)` kwarg wins over this attribute. Unknown names raise `ValueError` at `arun()` entry. |

### arun() — Main Entry Point

```python
await agent.arun(
    *,  # all arguments are keyword-only

    # LLM — required for AGENT / AMPHIFLOW; a pure ThinkAgent or WORKFLOW run needs none
    llm: Optional[BaseLlm] = None,

    # Context: pass a pre-built one, OR pass goal= / tools= / skills= /
    # cognitive_history= (forwarded via **kwargs) to have one auto-created.
    context: CognitiveContextT = None,

    # Execution control
    mode: RunMode = RunMode.AUTO,
    max_consecutive_fallbacks: int = 1,          # AMPHIFLOW step-failure threshold

    # Tracing / run artifacts (orthogonal — see below)
    trace: bool = False,                         # activate the in-memory AgentTrace
    workdir: Optional[Union[Path, str]] = None,  # materialise <workdir>/runs/<run_id>/

    builtin_tools: Optional[Iterable[str]] = None,  # override class-level builtin_tools filter
    **kwargs,                                    # goal=, tools=, skills=, cognitive_history=
) -> str
```

**Return value**: by default returns `ctx.summary()` — a textual recap of the post-run context. If a `RETURN(value)` yield ran from a top-level template body OR a finishing think step set `self._final_answer`, that value is returned instead (`str(value)`).

**Tracing**: `trace` and `workdir` are orthogonal. `trace=True` activates an in-memory `AgentTrace`, kept on `self._agent_trace` after the run. `workdir=path` materialises a `<workdir>/runs/<run_id>/` run directory. With **both** set, the `AgentTrace` incrementally persists `<run>/trace.json` — the run directory's single artifact. With `trace=False, workdir=path`, the run directory is created but stays empty.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `context` | `CognitiveContextT` | Current context after `arun()` |
| `final_answer` | `Optional[str]` | Auto-captured from a finishing step's `step_content`, or set explicitly by yielding `RETURN(value)` from a top-level template body |
| `llm` | `Optional[BaseLlm]` | The LLM bound by the last `arun(llm=...)` (`None` for a pure WORKFLOW / ThinkAgent run) |

### Template Methods (Override in Subclasses)

A subclass must override at least one of `on_agent` and `on_workflow`. Under
`RunMode.AUTO` the runtime picks the mode from which methods are overridden:
agent-only → `AGENT`, workflow-only (async-gen form) → `WORKFLOW`, both →
`AMPHIFLOW`. A coroutine-form `on_workflow` (`async def on_workflow(self, ctx): pass`)
is treated as a stub under `AUTO`; use `mode=RunMode.WORKFLOW` /
`RunMode.AMPHIFLOW` to drive a coroutine workflow explicitly.

```python
# LLM-driven orchestration (override for AGENT or AMPHIFLOW modes)
async def on_agent(self, ctx: CognitiveContextT) -> AsyncGenerator: ...

# Deterministic workflow (override for WORKFLOW or AMPHIFLOW modes)
async def on_workflow(self, ctx: CognitiveContextT) -> AsyncGenerator: ...

# Pre-think / post-act hooks — accept BOTH async-gen (yield primitives)
# and plain coroutine (return value) forms; both go through _invoke_template.
async def observation(self, ctx) -> Optional[str]: ...
async def before_action(self, decision_result, ctx) -> Any: ...
async def after_action(self, step_result, ctx) -> None: ...

# Action-execution hooks — coroutine form ONLY (awaited directly, not
# routed through the dispatcher; cannot yield framework primitives).
async def action_tool_call(self, tool_list, ctx) -> ActionResult: ...
async def action_custom_output(self, decision_result, ctx) -> Any: ...
```

There is NO `human_input(data)` template method on `AmphibiousAutoma`. To replace the default stdin fallback for HITL prompts, register a `@human_channel` handler — see [Human-in-the-Loop](#human-in-the-loop).

### Setting the final answer

There is **no** `self.set_final_answer(...)` instance method. To set the final answer explicitly, yield `RETURN(value)` from a top-level template body (`on_agent` or `on_workflow`); the dispatcher writes `str(value)` to `self._final_answer`. Without an explicit `RETURN`, `final_answer` is auto-captured from the finishing step's `step_content`. If neither happens, `arun()` returns `ctx.summary()`.

### Utility Methods

```python
# Phase scoping — saves/restores listed fields, clears LayeredExposure caches
async with self.snapshot(goal="Sub-goal", **fields):
    yield ThinkUnit("...")
```

## CognitiveWorker

```python
class CognitiveWorker:
```

Pure thinking unit — decides *what to do*, never *how*.

### Constructor

```python
CognitiveWorker(
    llm: BaseLlm = None,
    enable_rehearsal: bool = False,
    enable_reflection: bool = False,
    verbose: bool = None,
    verbose_prompt: bool = None,
    output_schema: Type[BaseModel] = None,  # Typed output mode
)
```

### Factory Methods

```python
# Quick creation from prompt string
worker = CognitiveWorker.inline(
    "Plan ONE immediate next step",
    llm=None,                          # Usually injected by agent
    enable_rehearsal=False,
    enable_reflection=False,
    output_schema=None,                # Set for typed output
    verbose=None,
    verbose_prompt=None,
)

# Alias
worker = CognitiveWorker.from_prompt("...")
```

### Template Methods (Override in Subclasses)

```python
# Required: Define thinking prompt
async def thinking(self) -> str: ...

# Optional hooks — observation / before_action / after_action accept
# BOTH async-coroutine and async-generator forms (symmetric with the
# AmphibiousAutoma-level hooks; both go through _invoke_template).
#   Coroutine form: `return _DELEGATE` / `return value` / `return None`.
#   Generator form: yield ActionCall / HumanCall / LLMCall, then optionally
#                   yield RETURN(value). Exhausting without RETURN is
#                   equivalent to returning None → treated as _DELEGATE
#                   (chains to the agent-level hook).
async def observation(self, context) -> Any: ...           # Return _DELEGATE / str / None, OR yield primitives
async def build_messages(self, think_prompt, tools_description,
                         output_instructions, context_info) -> List[Message]: ...
async def before_action(self, decision_result, context) -> Any: ...
async def after_action(self, step_result, ctx) -> Any: ...
```

### Class Attribute

```python
output_schema: Optional[Type[BaseModel]] = None
# When set, worker produces a typed Pydantic instance.
# Skips tool-call loop. Acquiring policy disabled.
# yield ThinkUnit("name") returns the typed instance.
```

## AgentWorker & BaseAgent

The external-agent peer of `CognitiveWorker`. Where `CognitiveWorker` runs one in-process observe-think-act cycle anchored on a `BaseLlm`, `AgentWorker` runs one **delegated** cycle anchored on a `BaseAgent` — an external coding-agent CLI. Both are concrete classes; both expose the same `thinking` / `observation` / `before_action` / `after_action` template surface.

### AgentWorker

```python
class AgentWorker(GraphAutoma):
    def __init__(
        self,
        agent: BaseAgent,                       # the external coding-agent driver
        *,
        verbose: Optional[bool] = None,         # None = inherit from AmphibiousAutoma
        verbose_prompt: Optional[bool] = None,  # log the assembled message per delegation
    )
```

`AgentWorker` organizes context only: it MCP-ifies `ctx.tools` (boots an in-process FastMCP host), assembles the message via `thinking()`, packs an `AgentRequest`, and calls `self._agent.run(request)`. It never embeds CLI argv — that is the `BaseAgent`'s job. There are no `goal` / `tools` / `skills` knobs: those ride on the `context` the framework passes in. `AgentWorker(agent)` works out of the box; subclass only to customize.

Template methods (override in a subclass):

```python
async def thinking(self, context) -> str: ...     # assemble the message (default already works)
async def observation(self, context) -> Any: ...   # _DELEGATE / value, same contract as CognitiveWorker
async def before_action(self, decision_result, context) -> Any: ...
async def after_action(self, step_result, ctx) -> Any: ...
```

### BaseAgent

Abstract driver for one external coding-agent CLI — the anchor type of `AgentWorker`, exactly as `BaseLlm` anchors `CognitiveWorker`. One required override (`run`), one provided helper (`_run_subprocess`).

```python
class BaseAgent:
    async def run(self, request: AgentRequest) -> AgentResult: ...   # subclasses MUST override

    async def _run_subprocess(
        self, argv: List[str], *,
        stdin_payload: Optional[bytes] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 180.0,
        done_signal: Optional[asyncio.Future] = None,
    ) -> Tuple[Optional[str], Optional[int], str]: ...               # provided helper
```

Subclass per CLI (`ClaudeCodeAgent`, `CodexAgent`, …). A subclass owns exactly two things: how to spawn its CLI, and how to detect completion + extract the result. `_run_subprocess` handles the generic spawn / drain / wait mechanics (races `done_signal` vs process exit vs `timeout`).

### ClaudeCodeAgent

The shipped `claude code` driver — `claude -p` with stream-json I/O.

```python
class ClaudeCodeAgent(BaseAgent):
    def __init__(
        self,
        *,
        bin: str = "claude",                  # the claude binary (path or name on PATH)
        allowed_builtin_tools: Optional[List[str]] = None,  # claude's own tools to permit;
                                              # default: Read / Write / Edit / Bash / Glob / Grep
        permission_mode: str = "bypassPermissions",
        completion_timeout: float = 180.0,    # seconds before the subprocess is force-terminated
    )
```

`ClaudeCodeAgent` spawns the `claude` CLI (`bin`) as a subprocess — it must be installed, on `PATH`, and authenticated. The tool allow-list it passes to the CLI combines `allowed_builtin_tools` with the MCP-bridged tools from the `AgentRequest`.

### CodexAgent

The shipped OpenAI Codex driver — runs the Codex CLI's headless `codex exec` mode.

```python
class CodexAgent(BaseAgent):
    def __init__(
        self,
        *,
        bin: str = "codex",                     # the codex binary (path or name on PATH)
        sandbox_mode: str = "workspace-write",  # read-only | workspace-write | danger-full-access
        completion_timeout: float = 180.0,      # seconds before the subprocess is force-terminated
    )
```

`CodexAgent` spawns the `codex` CLI as a subprocess — it must be installed, on `PATH`, and authenticated (ChatGPT login or `OPENAI_API_KEY`). The bridged MCP host is wired in with a `-c mcp_servers.<name>.url=<url>` config override; `--ignore-user-config` isolates the run so the delegation sees only the bridged server (auth still resolves from the default `~/.codex`). `sandbox_mode` governs Codex's own shell + file edits; approvals are forced off (`codex exec` is non-interactive). Codex has no per-tool allow-list flag, so `AgentRequest.allowed_tools` is unused.

### AgentRequest / AgentResult

```python
@dataclass
class AgentRequest:
    message: str                            # the full prompt handed to the external agent
    cwd: Path                               # subprocess working directory (ephemeral tempdir)
    mcp_servers: Dict[str, Dict[str, Any]]  # MCP servers to wire into the CLI
    allowed_tools: List[str] = []           # MCP-bridged tool names + the agent_done signal
    done_signal: Optional[asyncio.Future] = None  # resolves when the agent calls agent_done

@dataclass
class AgentResult:
    output: Optional[str]    # the agent_done(result=...) string, or None if it exited unsignalled
    exit_code: Optional[int]
    completion: str          # "agent_done" | "process_exit" | "timeout"
```

`AgentWorker` assembles the `AgentRequest`; `BaseAgent.run` consumes it and returns an `AgentResult` — mirroring the `messages` + `constraint` → response pair `CognitiveWorker` hands to `BaseLlm.astructured_output`.

## think_unit

```python
think_unit(
    worker: CognitiveWorker,
    *,
    max_attempts: int = 1,
    until: Callable = None,            # Loop condition
    tools: List[str] = None,           # Tool name filter
    skills: List[str] = None,          # Skill name filter
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 0,              # For RETRY strategy
) -> ThinkUnitDescriptor
```

Use as class variable; reference by name in `on_agent` via `yield ThinkUnit("name")`:

```python
from bridgic.amphibious import ThinkUnit

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(CognitiveWorker.inline("Plan step"), max_attempts=5)

    async def on_agent(self, ctx):
        result = yield ThinkUnit("planner")                     # Single execution
        result = yield ThinkUnit("planner", until=cond)         # Loop until condition
        result = yield ThinkUnit(                               # Per-call overrides
            "planner",
            until=cond, max_attempts=50, tools=["search"],
        )
```

Each yielded `ThinkUnit("name")` overlays the descriptor's defaults; `None` means "use the descriptor's value for this field".

## think_agent

```python
think_agent(
    worker: AgentWorker,
    *,
    expose_tools: Optional[List[str]] = None,  # project-tool name filter
                                               # (None = expose every non-builtin tool)
) -> ThinkAgentDescriptor
```

The external-agent peer of `think_unit`. Use as a class variable; reference by name in `on_agent` via `yield ThinkAgent("name")`. The `AgentWorker` carries all delegate-level config (which `BaseAgent` to drive); `expose_tools` selects which project tools from `ctx.tools` are exposed to the external agent over the MCP bridge.

```python
from bridgic.amphibious import AgentWorker, ClaudeCodeAgent, think_agent, ThinkAgent

class MyAutoma(AmphibiousAutoma[CognitiveContext]):
    reviewer = think_agent(
        AgentWorker(ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"])),
        expose_tools=["record_finding"],
    )

    async def on_agent(self, ctx):
        result = yield ThinkAgent("reviewer", goal="Review the diff.")
```

Each yielded `ThinkAgent(...)` overlays the descriptor's defaults (`goal`, `expose_tools`); `None` means "use the descriptor's value". The `AgentWorker` is cloned per invocation for state isolation.

## Yield Primitives

Template methods (`on_agent`, `on_workflow`, hooks) are async generators. The dispatcher routes each yielded value by type, validates the scope, executes the call, and sends a result back via `asend()`. Mismatches raise `RuntimeError` at dispatch time.

| Primitive | Category | Allowed scopes | Returns to generator |
|-----------|----------|----------------|----------------------|
| `ActionCall` | atomic Call | `on_workflow`, hooks | `List[ToolResult]` |
| `HumanCall` | atomic Call | `on_workflow`, hooks | `str` |
| `LLMCall` | atomic Call | `on_workflow`, hooks | protocol-specific |
| `EnterAgent` | mode-switch | `on_workflow` only | `None` |
| `ThinkUnit` | cognitive composition | `on_agent` only | worker output (or `None`) |
| `ThinkAgent` | cognitive composition | `on_agent` only | external agent's result `str` (or `None`) |
| `RETURN` | control flow | any | (closes generator; value flows out) |

### ActionCall — Deterministic tool execution

```python
from bridgic.amphibious import ActionCall

# In on_workflow():
result = yield ActionCall("tool_name", arg1="value", arg2=123)
# result: List[ToolResult]
```

```python
@dataclass(init=False)
class ActionCall:
    tool_name: str
    description: str
    tool_args: Dict[str, Any]
    decision: WorkflowDecision  # repr=False; built internally

    def __init__(self, tool_name: str, *, description: str = "", **tool_args): ...
```

`ActionCall` is purely a deterministic single-tool call — there are no framework-level knobs for retry, fallback worker, or attempt budgets at the call site (every `**tool_args` keyword is forwarded as a tool argument, so something like `ActionCall("foo", worker="x")` would pass `worker="x"` to the tool, not configure the framework). If the call fails in `AMPHIFLOW`, the framework's step-level fallback mechanism (see [Workflow Fallback](architecture.md#workflow-fallback-mechanism)) recovers via `on_agent` with the injected `resolve_step_fallback` tool.

### HumanCall — Pause for human input

```python
from bridgic.amphibious import HumanCall

# In on_workflow():
feedback = yield HumanCall(prompt="Confirm this action?")
feedback = yield HumanCall(prompt="...", channel="feishu")  # named handler
# feedback: str (the human's response)
```

```python
@dataclass
class HumanCall:
    prompt: str = ""
    channel: Optional[str] = None  # None = single registered @human_channel, else stdin fallback
```

Channel resolution: `channel=None` → if exactly one `@human_channel` handler is registered use it, if zero are registered fall back to built-in stdin, otherwise raise `RuntimeError`. `channel="name"` → invoke that named handler. Per-call timeouts are not exposed; the channel handler should enforce its own.

### LLMCall — Direct LLM invocation

```python
from bridgic.amphibious import LLMCall

# In on_workflow():
text = yield LLMCall.chat("What is 2+2?")
parsed = yield LLMCall.structure_output("Extract...", constraint=PydanticModel(model=Schema))
calls, reply = yield LLMCall.tool_selector("...", tools=[...])
```

```python
@dataclass(frozen=True)
class LLMCall:
    protocol: Literal["chat", "structure_output", "tool_selector"]
    prompt: str = ""
    history: Optional[List[Message]] = None
    constraint: Optional[Constraint] = None  # required iff protocol == "structure_output"
    tools: Optional[List[Tool]] = None       # required iff protocol == "tool_selector"
```

Returns by protocol:
- `"chat"` → `str` (extracted from `Response.message.content`)
- `"structure_output"` → typed value from `astructured_output` (typically a Pydantic instance)
- `"tool_selector"` → `Tuple[List[ToolCall], Optional[str]]`

### EnterAgent — Switch on_workflow → on_agent

```python
from bridgic.amphibious import EnterAgent

# In on_workflow():
yield EnterAgent(goal="Handle the login popup")
yield EnterAgent(goal="Pick a flight", tools=["search_flights", "book"])
yield EnterAgent(goal="Summarize", history=prior_messages, skills=["summary"])
```

```python
@dataclass
class EnterAgent:
    goal: str = ""
    history: Optional[Any] = None       # Optional[CognitiveHistory]; None → fresh CognitiveHistory()
    tools: Optional[List[str]] = None   # Tool-name filter applied to ctx.tools while in agent mode
    skills: Optional[List[str]] = None  # Skill-name filter applied to ctx.skills while in agent mode
```

`EnterAgent` is a **mode-switch signal**, not a function call. The state-machine dispatcher suspends the workflow generator and creates a fresh `on_agent` generator under a context snapshot built from the listed fields. When the agent generator naturally exhausts (an implicit "switch back to workflow" signal), the suspended workflow resumes at the next instruction. There is no stack, no recursion, no resumable agent state across switches.

`worker=` and `max_attempts=` are **not** accepted — `EnterAgent` controls *what the agent sees*, not *how it thinks*. Declare a `think_unit` and `yield ThinkUnit("name")` from inside `on_agent` for fine-grained cognitive control.

Requires the agent class to override `on_agent`; raises `RuntimeError` at dispatch time otherwise.

### ThinkUnit — Invoke a named cognitive step

```python
from bridgic.amphibious import ThinkUnit

# In on_agent():
result = yield ThinkUnit("main_think")
result = yield ThinkUnit("exec_think", until=lambda c: c.done, max_attempts=20)
```

```python
@dataclass(frozen=True)
class ThinkUnit:
    name: str
    until: Optional[Callable[..., Union[bool, Awaitable[bool]]]] = None
    max_attempts: Optional[int] = None
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
```

Resolves `name` via `getattr(type(self), name)` and expects a `ThinkUnitDescriptor`. Each non-`None` field overrides the descriptor's default for this single yield. The result returned via `asend()` is the worker's typed output (or `None` if the worker has no `output_schema`).

Only valid inside `on_agent` (`scope='agent'`). For a direct LLM invocation outside the cognitive loop, use `LLMCall` from `on_workflow`.

### ThinkAgent — Delegate a sub-goal to an external agent

```python
from bridgic.amphibious import ThinkAgent

# In on_agent():
result = yield ThinkAgent("reviewer")
result = yield ThinkAgent("reviewer", goal="Review the diff", expose_tools=["record_finding"])
```

```python
@dataclass(frozen=True)
class ThinkAgent:
    name: str
    goal: Optional[str] = None
    expose_tools: Optional[List[str]] = None
```

Resolves `name` via `getattr(type(self), name)` and expects a `ThinkAgentDescriptor`. The dispatcher clones the descriptor's `AgentWorker`, overlays `goal` / `expose_tools` onto `ctx` for the delegation, and drives one external-agent cycle. Every tool call the external agent makes over the MCP bridge is surfaced back as a decision and executed through `_run_action_call` — the parent's `before_action` / `after_action` hooks fire and the call lands in the trace.

The `asend()` result is the string the external agent passed to `agent_done(result=...)`, or `None` if it exited without signalling.

Only valid inside `on_agent` (`scope='agent'`) — same as `ThinkUnit`. Unlike `ThinkUnit`, there are no `until=` / `max_attempts=` knobs: one `ThinkAgent` yield is exactly one delegated cycle. See [think_agent](#think_agent) for the descriptor and [AgentWorker & BaseAgent](#agentworker--baseagent) for the worker surface.

### RETURN — Communicate a return value

```python
from bridgic.amphibious import RETURN

async def on_agent(self, ctx):
    yield ThinkUnit("main_think", max_attempts=20)
    yield RETURN(ctx.cognitive_history.get_all()[-1].content)
```

```python
@dataclass(frozen=True)
class RETURN:
    value: Any = None
```

PEP 525 forbids `return value` inside async generators (only bare `return` is allowed). `RETURN(value)` is the framework-level workaround: when the dispatcher receives it, it captures the value, immediately closes the generator, and returns the value to the caller. Anything yielded after a `RETURN` is unreachable.

For top-level template-method generators (`on_agent` / `on_workflow`), the captured value is written to `self._final_answer` (overriding the auto-capture from history).

`RETURN` is allowed in any scope (workflow, agent, hook).

## Human-in-the-Loop

Two entry points for requesting human input — both go through the same `@human_channel` registry (or stdin fallback if no handlers are registered):

| Entry Point | Where | Usage |
|-------------|-------|-------|
| `HumanCall` | `on_workflow()`, hooks (rejected in `on_agent`) | `feedback = yield HumanCall(prompt="Confirm?")` |
| `request_human` tool | LLM tool-call inside any `ThinkUnit`, any mode | Auto-injected into `context.tools`; no setup needed |

There is **no** code-level imperative API on `AmphibiousAutoma` — no `self.request_human(...)` method. If `on_agent` needs to ask a human, the LLM does it autonomously via the auto-injected tool inside a `ThinkUnit`.

### request_human as a built-in tool

`request_human` is one of the seven [built-in tools](#built-in-tools) auto-injected into `context.tools` during `arun()`, so the LLM can call it in any mode (AGENT, WORKFLOW fallback, AMPHIFLOW) without you wiring it through `tools=[...]`:

```python
await agent.arun(goal="Plan a trip, ask me if you need confirmation.", tools=[search_tool])
```

Passing `request_human_tool` explicitly is harmless — the injection step deduplicates by tool name. The tool resolves to the running agent through `current_agent` (a `contextvars.ContextVar`), so each concurrent `arun()` task gets its own binding and parallel agents do not interfere.

### @human_channel — register handlers for HumanCall + request_human

`@human_channel` is a **method decorator** — apply it to an `async` method of your `AmphibiousAutoma` subclass. The framework walks the MRO at class-definition time (`__init_subclass__`) and builds a per-class `_human_channels: Dict[str, str]` registry mapping channel names to method names.

```python
from bridgic.amphibious import AmphibiousAutoma, CognitiveContext, human_channel

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    @human_channel("feishu")                  # explicit channel name
    async def ask_feishu(self, prompt: str) -> str:
        return await send_to_feishu_and_wait(prompt)

    @human_channel                            # bare form: channel name = method name
    async def terminal(self, prompt: str) -> str:
        return await read_from_terminal(prompt)
```

When `HumanCall(channel="feishu", ...)` dispatches, the registered handler is invoked. With one handler registered on the class and `channel=None`, the dispatcher uses that handler implicitly; with zero handlers it falls back to stdin; with two or more, an explicit `channel=` is required (or the dispatcher raises `RuntimeError`).

The same `_human_channels` registry also drives the auto-injected `request_human` built-in tool — the LLM passes `channel="name"` from `on_agent`/any `ThinkUnit`, and the call routes through identical `_dispatch_human_channel` logic. The tool's JSON schema is rebuilt per agent class from this registry (see [request_human as a built-in tool](#request_human-as-a-built-in-tool) above), so the LLM is `enum`-constrained to valid names — it cannot fabricate a channel that would later be rejected.

Channel handlers are plain `async def` methods returning `str`. They are leaf I/O operations and do not dispatch yields (don't `yield ActionCall` / `HumanCall` / etc. inside).

## Built-in Tools

Seven `FunctionToolSpec` instances are auto-injected into every `AmphibiousAutoma` agent's `context.tools` during `arun()`, subject to the [`builtin_tools` filter](#class-attributes-override-in-subclasses). They are exported from both `bridgic.amphibious` and `bridgic.amphibious.builtin_tools` as `*_tool` constants.

```python
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
# ALL_BUILTIN_TOOLS: tuple of all seven specs in display order.
# current_agent:    ContextVar bound to the running AmphibiousAutoma during arun().
```

**Error contract.** Tools raise on validation failures. The framework's per-tool exception handler (the `_run_one` inner function inside `AmphibiousAutoma.action_tool_call`) catches every exception and produces:

```python
ActionStepResult(success=False, error=str(exc), tool_result=None)
```

— so the LLM sees the error in the next observation, and `on_workflow` (without fallback) propagates it as `RuntimeError("Tool execution failed for: ...")`. Tools never wrap errors as `<error>...</error>` strings at their own layer.

### request_human

```python
async def request_human(prompt: str, channel: str | None = None) -> str
```

Pause and ask the human operator a question. Internally resolves the running agent via the `current_agent` ContextVar and routes through `agent._dispatch_human_channel(prompt, channel=channel)` — the same dispatcher used by `yield HumanCall(prompt=..., channel=...)` from `on_workflow`.

| Param | Description |
|-------|-------------|
| `prompt` | The question or message shown to the human. |
| `channel` | Optional. Name of a registered `@human_channel` to route through (e.g. `"feishu"`, `"slack"`). Omit for the implicit default: sole registered channel, or stdin fallback if none are registered. **Required when 2+ channels are registered** — otherwise the dispatcher raises an ambiguity error. |

**Dynamic schema**: The spec the LLM actually sees is *not* the module-level `request_human_tool`; it is rebuilt per `arun()` from the agent class's `@human_channel` registry. When one or more channels are registered, the injected spec has its `channel` parameter constrained to a JSON-schema `enum` of the registered channel names, and the top-level description lists them — so the LLM picks from real names instead of guessing. With zero channels, the static spec is reused as-is. The factory is `build_request_human_tool(channel_names)` in `bridgic.amphibious.builtin_tools.human.request_human`; you normally do not call it directly.

See [Human-in-the-Loop](#human-in-the-loop) for the full HITL story.

### bash

```python
async def bash(command: str, timeout: int = 120000, cwd: str = "") -> str
```

Execute a shell command via the user's default shell. Returns the captured `stdout` **verbatim** — no envelope, no tags, no decoration. Downstream consumers (workflow `yield ActionCall("bash", ...)` or LLM tool dispatch) get the raw shell output and parse it directly.

Failure handling matches `subprocess.check_output`: a non-zero exit code raises `RuntimeError` whose message contains the exit code and any captured `stderr` (falling back to `stdout` when `stderr` is empty). The framework's `_action` then surfaces it as `ActionStepResult(success=False, error=...)`, so callers never need to inspect tags to detect failure.

`stderr` is NOT mixed into the return value on success — it's typical progress / warning noise. If a command writes its useful output to `stderr` (some tools do), append `2>&1` to redirect it into stdout.

| Param | Description |
|-------|-------------|
| `command` | Shell command. Multiple commands may be chained with `&&` / `\|\|` / `;`. Append `2>&1` to merge stderr into stdout. |
| `timeout` | Milliseconds before the process is killed. Default 120000 (2 min); maximum 600000 (10 min). |
| `cwd` | Working directory. Empty string inherits the parent process's cwd. |

Stateless — does not depend on the running agent. Non-zero exit codes are NOT exceptions; they are reported via the envelope so the LLM can interpret them. Output past 30 KB is truncated with a marker.

Raises:
- `ValueError` if `command` is empty.
- `TimeoutError` if the command exceeds `timeout` (process killed and awaited before the raise).

### read_file

```python
async def read_file(file_path: str, offset: int = 0, limit: int = 0) -> str
```

Read a file's contents in `cat -n` format (line number + tab + content). The line-numbered output is the format that `edit_file` expects you to base its `old_string` on (line numbers excluded — only the actual content matches).

Calling `read_file` records the file's mtime on the agent's per-run `_read_tracker` dict; `write_file` and `edit_file` consult it to enforce the read-before-modify invariant.

| Param | Description |
|-------|-------------|
| `file_path` | Absolute path. Relative paths are rejected. |
| `offset` | 1-based line number to start from. `0` means the first line. |
| `limit` | Max lines to return. `0` means the default cap of 2000 lines. |

Maximum file size is 5 MB. Lines longer than 2000 chars are truncated with a marker. Empty files and offsets past the end return informational text rather than empty strings or exceptions.

Raises:
- `ValueError` if `file_path` is empty / not absolute / not a regular file / file too large.
- `FileNotFoundError` if the file does not exist.

### write_file

```python
async def write_file(file_path: str, content: str) -> str
```

Create a new file or overwrite an existing one. Creating new files has no preconditions; overwriting an existing file requires that `read_file` was called on the path AND the file has not changed externally since that read.

Raises:
- `ValueError` if `file_path` is empty / not absolute / target exists but is not a regular file.
- `FileNotFoundError` if the parent directory does not exist.
- `RuntimeError` if the file exists and was not read first, or has been modified externally since the read.

### edit_file

```python
async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str
```

Replace `old_string` with `new_string`. By default `old_string` must occur exactly once — supply more surrounding context if it doesn't, or pass `replace_all=True` for rename refactors. Enforces the read-before-modify invariant.

Raises:
- `ValueError` if `file_path` is invalid / `old_string` is empty / equals `new_string` / not found / occurs multiple times without `replace_all`.
- `FileNotFoundError` if the file does not exist.
- `RuntimeError` if the file has not been read first or was modified externally since the read.

### glob

```python
async def glob(pattern: str, path: str = "") -> str
```

Find files matching a glob pattern (e.g. `**/*.py`, `src/**/*.ts`). Returns matching paths sorted by mtime descending, so recently-touched files surface first.

| Param | Description |
|-------|-------------|
| `pattern` | Glob pattern relative to `path`. |
| `path` | Absolute search root. Empty string means the process's cwd. |

Capped at 100 results; "no match" returns informational text.

Raises:
- `ValueError` for empty `pattern` or non-absolute `path`.
- `NotADirectoryError` if `path` is not a directory.

### grep

```python
async def grep(
    pattern: str,
    path: str = "",
    glob: str = "",
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
    head_limit: int = 0,
) -> str
```

Regex content search across files. Pure-Python via the standard `re` module — not a ripgrep replacement.

| Param | Description |
|-------|-------------|
| `pattern` | Python regex. |
| `path` | Absolute search root; empty = cwd. |
| `glob` | Optional glob filter on file paths (e.g. `*.py`). Empty = scan all files recursively. |
| `output_mode` | `"files_with_matches"` (default), `"count"` (`path:N`), or `"content"` (`path:lineno:line`). |
| `case_insensitive` | If True, match case-insensitively. |
| `head_limit` | Max result lines. `0` = default cap of 200. |

Hidden directories (path components starting with `.`) are skipped — keeps `.git`, `.venv` and similar metadata trees out of results. Capped at 5000 files scanned.

Raises:
- `ValueError` for empty `pattern`, non-absolute `path`, or unknown `output_mode`.
- `NotADirectoryError` if `path` is not a directory.
- `re.error` on invalid regex.

### Filter resolution

`AmphibiousAutoma.arun()` resolves which built-ins to inject in this order:

1. `arun(builtin_tools=...)` runtime kwarg, if provided.
2. Otherwise the class-level `builtin_tools` attribute.
3. Otherwise `None`, which means "inject every entry of `ALL_BUILTIN_TOOLS`".

A non-`None` resolution must reference only valid tool names; unknown entries (typos, stale references) raise `ValueError` at `arun()` entry. The selected set is intersected with already-present `context.tools` by `tool_name` — user-supplied tools win, the colliding built-in is silently skipped.

### Read-before-modify tracker

`AmphibiousAutoma._read_tracker: Dict[str, float]` maps absolute path → mtime at last successful `read_file`. It is reset at every `arun()` entry (so the invariant is scoped to a single run) and accessed by the filesystem tools through `current_agent`. `track_read` is a best-effort hook — a failed `os.stat` after a successful read is silently swallowed; the tracker simply has no entry, which causes a subsequent `edit_file` / `write_file` to correctly demand a re-read.

### resolve_step_fallback (injected during step-level fallback)

`resolve_step_fallback` is **not** part of `ALL_BUILTIN_TOOLS`. It is injected by the state-machine dispatcher only while running `on_agent` to recover from a failed atomic Call in `AMPHIFLOW` mode (see [Workflow Fallback Mechanism](architecture.md#workflow-fallback-mechanism)).

The tool's signature is shaped to match the failed yield:

| Failed yield | Tool signature | Slot default |
|--------------|----------------|--------------|
| `ActionCall` | `resolve_step_fallback(result: Any) -> str` | `[]` (empty `List[ToolResult]`) |
| `HumanCall` | `resolve_step_fallback(response: str) -> str` | `""` |
| `LLMCall.chat` | `resolve_step_fallback(text: str) -> str` | `""` |
| `LLMCall.structure_output` | `resolve_step_fallback(value_json: str) -> str` | `None` |
| `LLMCall.tool_selector` | `resolve_step_fallback() -> str` | `([], None)` |

The agent's LLM invokes it (or doesn't) to set the slot value; on agent generator exhaustion, the framework `asend()`s the slot value back to the suspended workflow generator. Users do not import or wire this tool — the framework injects and removes it transparently.

## CognitiveContext

```python
class CognitiveContext(Context):
```

Default context combining goal, tools, skills, and history.

### Fields

| Field | Type | Exposure | Description |
|-------|------|----------|-------------|
| `goal` | `str` | Plain | The goal to achieve |
| `tools` | `CognitiveTools` | EntireExposure | Available tools |
| `skills` | `CognitiveSkills` | LayeredExposure | Available skills |
| `cognitive_history` | `CognitiveHistory` | LayeredExposure | Execution history |
| `observation` | `Optional[str]` | Hidden (`display=False`) | Current observation |

### Custom Context

```python
from pydantic import Field, ConfigDict

class MyContext(CognitiveContext):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_page: str = Field(default="", description="Current page URL")
    extracted_data: dict = Field(
        default_factory=dict,
        json_schema_extra={"display": False}  # Hidden from LLM
    )
```

### CognitiveHistory Configuration

```python
CognitiveHistory(
    working_memory_size: int = 5,    # Recent steps with full details
    short_term_size: int = 20,       # Older steps as summaries
    compress_threshold: int = 10,    # Trigger LLM compression
)
```

### CognitiveSkills Methods

```python
skills = CognitiveSkills()
skills.add(Skill(name="...", description="...", content="..."))
skills.add_from_file("path/to/SKILL.md")
skills.add_from_markdown("---\nname: ...\n---\nContent")
skills.load_from_directory("skills/")
```

## Context and Exposure

### Context Base Class Methods

```python
ctx.summary() -> Dict[str, str]               # All field summaries
ctx.format_summary(include=None, exclude=None) -> str  # Formatted string
ctx.get_details(field: str, idx: int) -> Optional[str]  # LayeredExposure detail
ctx.get_field(field: str) -> Tuple[Optional[List[str]], Any]
ctx.get_revealed_items() -> List[Tuple[str, int]]
ctx.reset_revealed() -> None
ctx.set_llm(llm) -> None                      # Propagate LLM to Exposure fields
```

### Creating Custom Exposure Fields

```python
class MyExposure(LayeredExposure[MyItem]):
    def summary(self) -> List[str]: ...
    def get_details(self, index: int) -> Optional[str]: ...

class MyContext(CognitiveContext):
    my_field: MyExposure = Field(default_factory=MyExposure)
```

## Data Models

### ErrorStrategy

```python
class ErrorStrategy(Enum):
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently skip failed cycles
    RETRY = "retry"    # Retry up to max_retries times
```

### RunMode

```python
class RunMode(str, Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    AMPHIFLOW = "amphiflow"
    AUTO = "auto"
```

### Skill

```python
class Skill(BaseModel):
    name: str
    description: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Step

```python
class Step(BaseModel):
    content: str = ""
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[bool] = None
```

### ToolResult (returned by yield ActionCall)

```python
@dataclass
class ToolResult:
    tool_name: str
    tool_arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None
```

### ActionResult / ActionStepResult

```python
class ActionResult(BaseModel):
    results: List[ActionStepResult]

class ActionStepResult(BaseModel):
    tool_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    success: bool = True
    error: Optional[str] = None
```

### WorkflowDecision

```python
class WorkflowDecision(BaseModel):
    step_content: str = ""
    output: List[StepToolCall] = Field(default_factory=list)
```

Built internally by `ActionCall(...)`; not normally constructed by user code.

## Tool Definition

```python
from bridgic.core.agentic.tool_specs import FunctionToolSpec

# From async function
async def my_tool(param1: str, param2: int) -> str:
    """Tool description visible to LLM."""
    return "result"

tool_spec = FunctionToolSpec.from_raw(my_tool)
```

## AgentTrace

```python
# Enable tracing
result = await agent.arun(..., trace=True)
# With workdir set too, the trace also persists to <workdir>/runs/<run_id>/trace.json
result = await agent.arun(..., trace=True, workdir="./.bridgic")

# Access the trace (kept on the agent after the run)
trace = agent._agent_trace.build()
# Returns a flat unified dict:
#   {"goal": str, "metadata": {...}, "history": [TraceStep, ...]}
# history: every recorded step in order — ThinkUnit, ThinkAgent, ActionCall,
#          HumanCall, LLMCall — each a TraceStep.

# Save / Load
agent._agent_trace.save("trace.json")
loaded = AgentTrace.load("trace.json")  # Returns plain dict
```

Each `TraceStep` carries `name`, `step_content`, `tool_calls` (`List[RecordedToolCall]`), `observation`, `output_type` (`StepOutputType`), `structured_output`, and `think_agent_name` (set on `ThinkAgent` steps).
