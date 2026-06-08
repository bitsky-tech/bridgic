# Bridgic Amphibious API Reference

## Table of Contents
- [LLM Setup](#llm-setup)
- [Imports](#imports)
- [CLI Scaffolding](#cli-scaffolding)
- [The Two-Loop Context Model](#the-two-loop-context-model)
- [AmphibiousAutoma](#amphibiousautoma)
- [CognitiveWorker](#cognitiveworker)
- [AgentWorker & BaseAgent](#agentworker--baseagent)
- [think_unit](#think_unit)
- [think_agent](#think_agent)
- [Yield Primitives](#yield-primitives)
- [Human-in-the-Loop](#human-in-the-loop)
- [Built-in Tools](#built-in-tools)
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
    AgentWorker, BaseAgent, ClaudeCodeAgent, CodexAgent, AgentRequest, AgentResult,
    # Context — two-loop model
    OTAContext, Context,
    # Yield primitives
    ActionCall, HumanCall, LLMCall, EnterAgent, ThinkUnit, ThinkAgent, RETURN,
    # Human channel registry
    human_channel,
    # Data models
    Step, OTARecord, RunMode, ErrorStrategy,
    ActionResult, ActionStepResult, ToolResult,
    StepToolCall, ToolArgument,
    # Trace
    TraceStep, RecordedToolCall, StepOutputType,
    # Built-in tool specs (declared on the OTA context via OTAContext.tool)
    request_human_tool, bash_tool,
    read_file_tool, write_file_tool, edit_file_tool,
    glob_tool, grep_tool,
)
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message, Role
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

The scaffold writes only a single `amphi.py` (an `OTAContext` subclass with built-ins declared via `OTAContext.tool`, a `Context` subclass, a `CognitiveWorker`, and an `AmphibiousAutoma[OTAContext, Context]` with `think_unit` + `on_agent` / `on_workflow` stubs). It does not create subdirectories and does not emit runtime configuration (e.g. `.env`).

Python API: `create_project(base_dir: Optional[str] = None, task: Optional[str] = None) -> Path`. Raises `FileExistsError` if `amphi.py` already exists in the target directory.

## The Two-Loop Context Model

An agent is parameterized by two context types: `AmphibiousAutoma[OTAContextT, ContextT]`. Both are required.

### Context (big-loop, free-form)

```python
class Context(BaseModel):
```

Free-form cross-turn knowledge (memory, conversation, domain state). A bare `Context` has just fields plus an overridable `summary`. **Tools are NOT a base-`Context` concern** — they live on `OTAContext`.

```python
from bridgic.amphibious import Context

class MyContext(Context):
    notes: str = ""
    facts: list[str] = []

    # `fields` is the raw {name: value} dict, auto-injected. Compose any
    # prompt-facing rendering you like (usually a str). Default returns the dict.
    def summary(self, fields):
        return f"Notes: {fields['notes']}\nFacts: {fields['facts']}"
```

Supply it at run time via `arun(context=...)` (optional — the framework creates an empty `_context_class` when omitted). It is shared read-only across the run and any delegation.

### OTAContext (small-loop, framework-owned)

```python
class OTAContext(Context):
    user_input: str = ""                       # this run's question / objective
    ota_record: List[OTARecord] = []           # observe-think-act round trace
    tools: List[ToolSpec] = []                 # action-phase affordances (declared)
```

The framework constructs a fresh `OTAContext` per `arun()`, seeding `user_input`. During the run it drives the round trace through per-round result accessors and `open_record()`.

| Member | Kind | Description |
|--------|------|-------------|
| `user_input` | field | The single question / objective this run answers |
| `ota_record` | field | `List[OTARecord]` — one record per observe-think-act round |
| `tools` | field | The tool specs this run carries (seeded from the class's declared tools) |
| `obs_result` | property | Read/write the current round's observation result |
| `think_result` | property | Read/write the current round's decision (`ThinkResult`) |
| `action_result` | property | Read/write the current round's action result |
| `open_record()` | method | Append a fresh `OTARecord` (open a new round) |
| `add_tool(spec)` | method | Add a tool to this run's `tools` at run time |
| `summary(fields=None)` | method | Render `user_input` + the round trace for the prompt (overridable) |
| `OTAContext.tool(obj)` | classmethod | **Declare** a tool on the class (decorator or call) |

### Declaring tools — `OTAContext.tool`

**Nothing is auto-injected.** Every OTA context declares the tools it carries via the `tool` classmethod, usable as a decorator *and* a call:

```python
from bridgic.amphibious import OTAContext, bash_tool, request_human_tool

class MyOTAContext(OTAContext):
    pass

@MyOTAContext.tool                  # decorate a standalone function
async def fetch(url: str) -> str:
    """Fetch a URL."""
    ...

MyOTAContext.tool(bash_tool)        # register an existing ToolSpec
MyOTAContext.tool(request_human_tool)
MyOTAContext.tool(obj.method)       # bound method — keeps `obj` as self
```

Accepted forms (normalized to a `ToolSpec`): a `ToolSpec`, a bound method (kept bound to its `self`), or any plain callable. A subclass inherits its bases' declared tools and may add more. Each run's `tools` field is seeded from the class's declared set at construction; pass an explicit `tools=` to the constructor (or use `add_tool`) to override per run.

## AmphibiousAutoma

```python
class AmphibiousAutoma(GraphAutoma, Generic[OTAContextT, ContextT]):
```

Base class for dual-mode agents. Subclass with **two** generic arguments: the small-loop `OTAContext` (arg 1) and the free-form big-loop `Context` (arg 2), e.g. `class MyAgent(AmphibiousAutoma[MyOTAContext, MyContext])`. Both are validated at class-creation time.

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

### arun() — Main Entry Point

```python
await agent.arun(
    *,  # all arguments are keyword-only

    # The small-loop objective — seeds the fresh per-run OTAContext's user_input.
    user_input: str = "",

    # LLM — required for AGENT / AMPHIFLOW; a pure ThinkAgent or WORKFLOW run needs none.
    llm: Optional[BaseLlm] = None,

    # Big-loop knowledge context (free-form). None → a fresh empty _context_class.
    context: Optional[ContextT] = None,

    # Pre-built small-loop context. None → the framework builds one from
    # _ota_context_class seeded with user_input. When supplied, it is used
    # verbatim (its own user_input stands) — lets you seed per-run small-loop state.
    ota_context: Optional[OTAContextT] = None,

    # Execution control
    mode: RunMode = RunMode.AUTO,
    max_consecutive_fallbacks: int = 1,          # AMPHIFLOW step-failure threshold

    # Tracing / run artifacts (orthogonal — see below)
    trace: bool = False,                         # activate the in-memory AgentTrace
    workdir: Optional[Union[Path, str]] = None,  # materialise <workdir>/runs/<run_id>/
) -> str
```

**Tools are not passed here.** Each OTA context declares the tools it carries on its class via `OTAContext.tool`; `arun` does not assemble or merge any toolset.

**Return value**: a `str`. By default the post-run small-loop `ota_ctx.summary()`. If a finishing think step set `self._final_answer`, or a top-level template body yielded `RETURN(value)`, that value is returned instead (`str(value)`).

**Tracing**: `trace` and `workdir` are orthogonal. `trace=True` activates an in-memory `AgentTrace`, kept on `self._agent_trace` after the run. `workdir=path` materialises a `<workdir>/runs/<run_id>/` run directory. With **both** set, the `AgentTrace` incrementally persists `<run>/trace.json` — the run directory's single artifact. With `trace=False, workdir=path`, the run directory is created but stays empty.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `ota_ctx` | `Optional[OTAContextT]` | The active small-loop context (fresh per run; swapped to a sub-context during a delegation) |
| `ctx` | `Optional[ContextT]` | The big-loop knowledge context (shared read-only across the run) |
| `final_answer` | `Optional[str]` | Auto-captured from the finishing step's `step_content`, or set by yielding `RETURN(value)` from a top-level body |
| `llm` | `Optional[BaseLlm]` | The LLM bound by the last `arun(llm=...)` (`None` for a pure WORKFLOW / ThinkAgent run) |

### Template Methods (Override in Subclasses)

A subclass must override at least one of `on_agent` and `on_workflow`. Under `RunMode.AUTO` the runtime picks the mode from which methods are overridden: agent-only → `AGENT`, workflow-only → `WORKFLOW`, both → `AMPHIFLOW`.

**All overridable template methods must be async generators** (yield-driven). The framework validates this at class-creation time and raises `TypeError` for a coroutine-form override. If a body has no real yields, add `if False: yield` as an unreachable stub.

```python
# LLM-driven orchestration — yields ThinkUnit / ThinkAgent / RETURN only.
async def on_agent(self, ota_context, context=None) -> AsyncGenerator: ...

# Deterministic workflow — yields ActionCall / HumanCall / LLMCall / EnterAgent / RETURN.
async def on_workflow(self, ota_context, context=None) -> AsyncGenerator: ...

# Agent-level OTA hooks — async generators, payload-free (read state off ota_context).
# Worker-level hooks delegating here (returning _DELEGATE / None) fall through to these.
async def observation(self, ota_context, context=None):     # yield RETURN(text) to set obs_result
    ...
async def before_action(self, ota_context, context=None):   # read ota_context.think_result;
    ...                                                      # yield RETURN(decision) to override it
async def after_action(self, ota_context, context=None):    # read ota_context.action_result
    ...

# Action-execution hook — a coroutine (NOT an async generator). Reads the
# decision off ota_context.think_result, runs its tool calls, returns an ActionResult.
async def action_tool_call(self, ota_context, context=None) -> ActionResult: ...
```

To set the final answer explicitly, yield `RETURN(value)` from a top-level body. There is no `self.set_final_answer(...)` method. To replace the default stdin HITL fallback, register a `@human_channel` handler — see [Human-in-the-Loop](#human-in-the-loop).

## CognitiveWorker

```python
class CognitiveWorker(GraphAutoma):
```

The in-process think unit — one observe-think-act cycle, anchored on a `BaseLlm`. Observation and action execution are handled by `AmphibiousAutoma`; the worker owns exactly one thing: the **thinking** step.

### Constructor

```python
CognitiveWorker(
    llm: Optional[BaseLlm] = None,   # usually injected by the agent at runtime
    verbose: Optional[bool] = None,
)
```

### Template Methods (Override in Subclasses)

```python
# THE override point. Assemble a prompt from the two contexts, call self._llm
# however the model needs, and return that call's NATURAL result —
# _assemble_decision adapts it into the framework's decision.
async def thinking(self, ota_context, context=None) -> Any: ...

# Optional hooks — coroutine form (return _DELEGATE / value) OR async-generator
# form (yield ActionCall / HumanCall / LLMCall, then optionally RETURN(value)).
# Payload-free: read state off ota_context. Returning _DELEGATE / None chains to
# the matching AmphibiousAutoma-level hook.
async def observation(self, ota_context, context=None) -> Any: ...
async def before_action(self, ota_context, context=None) -> Any: ...
async def after_action(self, ota_context, context=None) -> Any: ...
```

### What `thinking()` returns — `_assemble_decision`

`thinking()` returns the bridgic protocol's natural result; the framework adapts each shape into a flat decision (`ThinkResult`: `step_content` + `tool_calls`). A decision with **no** tool calls is the finish.

| `thinking()` returns | Produced by | Becomes |
|----------------------|-------------|---------|
| `Response` | `achat` | content-only (`step_content` = reply text) |
| `(tool_calls, content)` | `aselect_tool` | tool-calling (note: tool_calls FIRST) |
| a pydantic `BaseModel` / `dict` | `astructured_output` | content-only; value serialized to JSON in `step_content` |
| `str` | plain text / accumulated stream | content-only |

```python
class MyThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        messages = [Message.from_text(ota_context.summary(), role=Role.USER)]
        return await self._llm.aselect_tool(
            messages=messages,
            tools=[t.to_tool() for t in ota_context.tools],
        )
```

A `CognitiveWorker` is cloned per `yield ThinkUnit(...)` for state isolation (`_clone()`; the `BaseLlm` is shared and re-bound at runtime). Subclasses with extra `__init__` params should override `_clone()`.

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

`AgentWorker` organizes context only: it MCP-ifies the OTA context's `tools` (boots an in-process FastMCP host), assembles the message via `thinking()`, packs an `AgentRequest`, and calls `self._agent.run(request)`. It never embeds CLI argv — that is the `BaseAgent`'s job. There are no `goal` / `tools` knobs: those ride on the contexts the framework passes in. `AgentWorker(agent)` works out of the box; subclass only to customize.

Template methods (override in a subclass):

```python
async def thinking(self, ota_context, context=None) -> str: ...   # assemble the message (default works)
async def observation(self, ota_context, context=None) -> Any: ... # _DELEGATE / value, as CognitiveWorker
async def before_action(self, ota_context, context=None) -> Any: ...
async def after_action(self, ota_context, context=None) -> Any: ...
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

A subclass owns exactly two things: how to spawn its CLI, and how to detect completion + extract the result. `_run_subprocess` handles the generic spawn / drain / wait mechanics (races `done_signal` vs process exit vs `timeout`).

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

The CLI (`bin`) must be installed, on `PATH`, and authenticated. The tool allow-list it passes to the CLI combines `allowed_builtin_tools` with the MCP-bridged tools from the `AgentRequest`.

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

The `codex` CLI must be installed, on `PATH`, and authenticated (ChatGPT login or `OPENAI_API_KEY`). The bridged MCP server is wired in with `-c mcp_servers.<name>.url=<url>` config overrides; `--ignore-user-config` isolates the run so the delegation sees only the bridged server (auth still resolves from the default `~/.codex`). Because `codex exec` is non-interactive, approvals are disabled at the config level (`-c approval_policy=never`, plus per-server `-c default_tools_approval_mode=auto`). `sandbox_mode` governs Codex's own shell + file edits.

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

`AgentWorker` assembles the `AgentRequest`; `BaseAgent.run` consumes it and returns an `AgentResult`.

## think_unit

```python
think_unit(
    worker: CognitiveWorker,
    *,
    until: Callable = None,            # Loop condition (stop early when true)
    max_attempts: int = 1,             # OTA cycle cap
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 0,              # For the RETRY strategy
) -> ThinkUnitDescriptor
```

Wraps a `CognitiveWorker` (cloned per invocation for state isolation). A think unit owns only thinking-orchestration knobs — the toolset comes from the contexts the worker's `thinking()` assembles, not from here (there is no `tools` / `skills` filter). Use as a class variable; reference by name in `on_agent` via `yield ThinkUnit("name")`:

```python
from bridgic.amphibious import ThinkUnit

class MyAgent(AmphibiousAutoma[MyOTAContext, MyContext]):
    planner = think_unit(MyThink(), max_attempts=5)

    async def on_agent(self, ota_context, context=None):
        result = yield ThinkUnit("planner")                         # single execution
        result = yield ThinkUnit("planner", until=cond)             # loop until condition
        result = yield ThinkUnit("planner", until=cond, max_attempts=50)  # per-call overrides
```

Each yielded `ThinkUnit("name")` overlays the descriptor's defaults; `None` means "use the descriptor's value". (`on_error` / `max_retries` are descriptor-only — no per-yield overlay.)

## think_agent

```python
think_agent(
    worker: AgentWorker,
    *,
    expose_tools: Optional[List[str]] = None,  # project-tool name filter
                                               # (None = expose every non-builtin tool)
) -> ThinkAgentDescriptor
```

The external-agent peer of `think_unit`. Use as a class variable; reference by name in `on_agent` via `yield ThinkAgent("name")`. The `AgentWorker` carries all delegate-level config (which `BaseAgent` to drive); `expose_tools` selects which declared project tools are exposed to the external agent over the MCP bridge.

```python
from bridgic.amphibious import AgentWorker, ClaudeCodeAgent, think_agent, ThinkAgent

class MyAutoma(AmphibiousAutoma[MyOTAContext, MyContext]):
    reviewer = think_agent(
        AgentWorker(ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"])),
        expose_tools=["record_finding"],
    )

    async def on_agent(self, ota_context, context=None):
        result = yield ThinkAgent("reviewer", goal="Review the diff.")
```

Each yielded `ThinkAgent(...)` overlays the descriptor's defaults (`goal`, `expose_tools`); `None` means "use the descriptor's value". The `AgentWorker` is cloned per invocation for state isolation.

## Yield Primitives

Template methods are async generators. The dispatcher routes each yielded value by type, validates the scope, executes the call, and sends a result back via `asend()`. Mismatches raise `RuntimeError` at dispatch time.

| Primitive | Category | Allowed scopes | Returns to generator |
|-----------|----------|----------------|----------------------|
| `ActionCall` | atomic Call | `on_workflow`, hooks | `List[ToolResult]` |
| `HumanCall` | atomic Call | `on_workflow`, hooks | `str` |
| `LLMCall` | atomic Call | `on_workflow`, hooks | protocol-specific |
| `EnterAgent` | mode-switch | `on_workflow` only | `None` (workflow resumes) |
| `ThinkUnit` | cognitive composition | `on_agent` only | finishing think's `step_content` (`str`) |
| `ThinkAgent` | cognitive composition | `on_agent` only | external agent's result `str` (or `None`) |
| `RETURN` | control flow | any | (closes generator; value flows out) |

### ActionCall — Deterministic tool execution

```python
result = yield ActionCall("tool_name", arg1="value", arg2=123)   # in on_workflow
# result: List[ToolResult]
```

```python
@dataclass(init=False)
class ActionCall:
    tool_name: str
    description: str
    tool_args: Dict[str, Any]

    def __init__(self, tool_name: str, *, description: str = "", **tool_args): ...
```

Each `ActionCall` wraps exactly one tool call. Every `**tool_args` keyword is forwarded as a tool argument — the `**kwargs` form cannot express a tool whose own parameter is named `tool_name` or `description` (those names are claimed by the signature). If the call fails in `AMPHIFLOW`, the framework's step-level fallback recovers via a bounded `on_agent` sub-run (see [architecture.md](architecture.md#workflow-fallback-mechanism)).

### HumanCall — Pause for human input

```python
feedback = yield HumanCall(prompt="Confirm this action?")          # in on_workflow
feedback = yield HumanCall(prompt="...", channel="feishu")         # named handler
# feedback: str
```

```python
@dataclass
class HumanCall:
    prompt: str = ""
    channel: Optional[str] = None
```

Channel resolution: `channel=None` → if exactly one `@human_channel` handler is registered use it, if zero are registered fall back to built-in stdin, otherwise raise `RuntimeError`. `channel="name"` → invoke that named handler.

### LLMCall — Direct LLM invocation

```python
text = yield LLMCall.chat("What is 2+2?")                                   # in on_workflow
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
yield EnterAgent(goal="Handle the login popup")    # in on_workflow
```

```python
@dataclass
class EnterAgent:
    goal: str = ""
```

`EnterAgent` is a **mode-switch signal**, not a function call. The state-machine dispatcher suspends the workflow generator and runs a **fresh nested OTA episode** of `on_agent`: a new `OTAContext` is built with `goal` as its `user_input`, carrying the OTA context class's declared tools; the big-loop `Context` is shared read-only. When the agent generator naturally exhausts, the suspended workflow resumes at the next instruction. No stack, no recursion — delegation is fresh-instance (isolation by construction); the parent OTA context is restored, never mutated.

The resumed `yield EnterAgent(...)` evaluates to **`None`** — the sub-run communicates through shared state and side-effects, not a return value. Note: a `RETURN(value)` yielded *inside* that `on_agent` ends the **entire** run with `value` (the workflow does not resume); let `on_agent` exhaust normally if you want the workflow to continue.

`EnterAgent` controls *what sub-task the agent gets* (`goal`), not *how it thinks* — there is no `worker=` / `max_attempts=` / `tools=` / `skills=`. For fine-grained cognitive control, `yield ThinkUnit("name")` from inside `on_agent`. Requires the class to override `on_agent`; raises `RuntimeError` otherwise.

### ThinkUnit — Invoke a named cognitive step

```python
result = yield ThinkUnit("main_think")                                    # in on_agent
result = yield ThinkUnit("exec_think", until=lambda c: c.done, max_attempts=20)
```

```python
@dataclass(frozen=True)
class ThinkUnit:
    name: str
    until: Optional[Callable[..., Union[bool, Awaitable[bool]]]] = None
    max_attempts: Optional[int] = None
```

Resolves `name` against the class and expects a `ThinkUnitDescriptor`. Each non-`None` field overrides the descriptor's default for this single yield. The `asend()` result is the finishing think's `step_content` (a `str`). Only valid inside `on_agent`.

### ThinkAgent — Delegate a sub-goal to an external agent

```python
result = yield ThinkAgent("reviewer")                                     # in on_agent
result = yield ThinkAgent("reviewer", goal="Review the diff", expose_tools=["record_finding"])
```

```python
@dataclass(frozen=True)
class ThinkAgent:
    name: str
    goal: Optional[str] = None
    expose_tools: Optional[List[str]] = None
```

Resolves `name` against the class and expects a `ThinkAgentDescriptor`. The dispatcher clones the descriptor's `AgentWorker` and drives one external-agent cycle. Every tool call the external agent makes over the MCP bridge is surfaced back as a decision and executed through the action phase — the parent's `before_action` / `after_action` hooks fire and the call lands in the trace. The `asend()` result is the string the external agent passed to `agent_done(result=...)`, or `None` if it exited without signalling. Only valid inside `on_agent`. Unlike `ThinkUnit`, there are no `until=` / `max_attempts=` knobs — one `ThinkAgent` yield is exactly one delegated cycle.

### RETURN — Communicate a return value

```python
async def on_agent(self, ota_context, context=None):
    answer = yield ThinkUnit("main_think", max_attempts=20)
    yield RETURN(answer)
```

```python
@dataclass(frozen=True)
class RETURN:
    value: Any = None
```

PEP 525 forbids `return value` inside async generators. `RETURN(value)` is the framework-level workaround: the dispatcher captures the value, closes the generator, and returns it to the caller. Anything yielded after a `RETURN` is unreachable. For top-level template bodies, the captured value is written to `self._final_answer`. Allowed in any scope.

## Human-in-the-Loop

Two entry points for requesting human input — both go through the same `@human_channel` registry (or stdin fallback if no handlers are registered):

| Entry Point | Where | Usage |
|-------------|-------|-------|
| `HumanCall` | `on_workflow()`, hooks (rejected in `on_agent`) | `feedback = yield HumanCall(prompt="Confirm?")` |
| `request_human` tool | LLM tool-call inside any `ThinkUnit`, any mode | Declared via `OTAContext.tool(request_human_tool)` |

There is **no** code-level imperative API on `AmphibiousAutoma` (no `self.request_human(...)`). If `on_agent` needs to ask a human, the LLM does it autonomously via the declared tool inside a `ThinkUnit`.

### request_human tool

`request_human_tool` is a plain `FunctionToolSpec` — declare it on the OTA context that wants HITL: `MyOTAContext.tool(request_human_tool)`. It resolves the running agent via the `current_agent` `ContextVar` and routes through `agent._run_human_call(HumanCall(prompt, channel))` — the same driver `yield HumanCall(...)` uses. Because of the ContextVar binding, each concurrent `arun()` task gets its own isolated agent, so parallel agents do not interfere.

```python
async def request_human(prompt: str, channel: str | None = None) -> str
```

| Param | Description |
|-------|-------------|
| `prompt` | The question shown to the human. |
| `channel` | Optional. A registered `@human_channel` name. Omit for the implicit default: sole registered channel, or stdin fallback if none. **Required when 2+ channels are registered.** |

### @human_channel — register handlers

`@human_channel` is a **method decorator** — apply it to an `async` method of your `AmphibiousAutoma` subclass. The framework walks the MRO at class-creation time and builds a per-class `_human_channels: Dict[str, str]` registry (channel name → method name).

```python
from bridgic.amphibious import AmphibiousAutoma, human_channel

class MyAgent(AmphibiousAutoma[MyOTAContext, MyContext]):
    @human_channel("feishu")                  # explicit channel name
    async def ask_feishu(self, prompt: str) -> str:
        return await send_to_feishu_and_wait(prompt)

    @human_channel                            # bare form: channel name = method name
    async def terminal(self, prompt: str) -> str:
        return await read_from_terminal(prompt)
```

With one handler registered, `HumanCall(channel=None)` and the `request_human` tool route to it implicitly; with zero, both fall back to stdin; with 2+, an explicit `channel=` is required. Channel handlers are plain `async def` methods returning `str` — leaf I/O operations; they do not yield framework primitives.

## Built-in Tools

Seven `FunctionToolSpec` instances ship with the framework. **Nothing is auto-injected** — declare the ones a run needs on its OTA context class via `OTAContext.tool` (decorator or call). They are exported from both `bridgic.amphibious` and `bridgic.amphibious.builtin_tools` as `*_tool` constants.

```python
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
# ALL_BUILTIN_TOOLS: tuple of all seven specs in display order — declare the
#                    whole set at once: for t in ALL_BUILTIN_TOOLS: MyOTACtx.tool(t)
# current_agent:     ContextVar bound to the running AmphibiousAutoma during arun().
```

**Error contract.** Tools raise on validation failures. The framework's per-tool exception handler (the `_run_one` inner function inside `AmphibiousAutoma.action_tool_call`) catches every exception and produces `ActionStepResult(success=False, error=str(exc), tool_result=None)` — so the LLM sees the error in the next observation, and `on_workflow` (without fallback) propagates it as `RuntimeError("Tool execution failed for: ...")`. Tools never wrap errors as strings at their own layer.

### request_human

```python
async def request_human(prompt: str, channel: str | None = None) -> str
```

Pause and ask the human operator a question. See [Human-in-the-Loop](#human-in-the-loop).

### bash

```python
async def bash(command: str, timeout: int = 120000, cwd: str = "") -> str
```

Execute a shell command via the user's default shell. Returns the captured `stdout` **verbatim**. A non-zero exit code raises `RuntimeError` (message contains the exit code and any captured `stderr`). `stderr` is not mixed into the success return — append `2>&1` to merge it into stdout.

| Param | Description |
|-------|-------------|
| `command` | Shell command. Chain with `&&` / `\|\|` / `;`. Append `2>&1` to merge stderr into stdout. |
| `timeout` | Milliseconds before the process is killed. Default 120000 (2 min); maximum 600000 (10 min). |
| `cwd` | Working directory. Empty string inherits the parent process's cwd. |

Raises `ValueError` if `command` is empty; `TimeoutError` on timeout. Output past 30 KB is truncated.

### read_file

```python
async def read_file(file_path: str, offset: int = 0, limit: int = 0) -> str
```

Read a file in `cat -n` format (line number + tab + content) — the format `edit_file` expects you to base `old_string` on (line numbers excluded; only content matches). Records the file's mtime on the agent's per-run `_read_tracker` so `write_file` / `edit_file` can enforce read-before-modify.

| Param | Description |
|-------|-------------|
| `file_path` | Absolute path. Relative paths are rejected. |
| `offset` | 1-based start line. `0` = first line. |
| `limit` | Max lines. `0` = default cap of 2000 lines. |

Max file size 5 MB; lines over 2000 chars are truncated. Raises `ValueError` (empty / not absolute / not a regular file / too large) or `FileNotFoundError`.

### write_file

```python
async def write_file(file_path: str, content: str) -> str
```

Create a new file, or overwrite an existing one (overwrite requires a prior `read_file` on the path AND no external change since). Raises `ValueError`, `FileNotFoundError` (missing parent dir), or `RuntimeError` (not read first / changed externally).

### edit_file

```python
async def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str
```

Exact-string replacement. By default `old_string` must occur exactly once — add surrounding context, or pass `replace_all=True` for rename refactors. Enforces read-before-modify. Raises `ValueError`, `FileNotFoundError`, or `RuntimeError`.

### glob

```python
async def glob(pattern: str, path: str = "") -> str
```

Find files matching a glob pattern (e.g. `**/*.py`). Returns paths sorted by mtime descending. `path` empty = cwd. Capped at 100 results. Raises `ValueError` / `NotADirectoryError`.

### grep

```python
async def grep(pattern: str, path: str = "", glob: str = "",
               output_mode: str = "files_with_matches",
               case_insensitive: bool = False, head_limit: int = 0) -> str
```

Regex content search (pure-Python `re`). `output_mode`: `"files_with_matches"` (default) / `"count"` (`path:N`) / `"content"` (`path:lineno:line`). Hidden directories are skipped; capped at 5000 files. `head_limit` `0` = default cap of 200. Raises `ValueError` / `NotADirectoryError` / `re.error`.

### Read-before-modify tracker

`AmphibiousAutoma._read_tracker: Dict[str, float]` maps absolute path → mtime at last successful `read_file`. Reset at every `arun()` entry (scoping the invariant to one run); accessed by the filesystem tools through `current_agent`.

## Data Models

### RunMode

```python
class RunMode(str, Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    AMPHIFLOW = "amphiflow"
    AUTO = "auto"
```

### ErrorStrategy

```python
class ErrorStrategy(Enum):
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently ignore exceptions
    RETRY = "retry"    # Retry up to max_retries times
```

### OTARecord (one observe-think-act round)

```python
class OTARecord(BaseModel):
    model_config = ConfigDict(extra="allow")   # hooks may fold custom per-round fields
    observation_result: Optional[Any] = None
    think_result: Optional[Any] = None
    action_result: Optional[Any] = None
```

One round = one think-decision = one action result. `extra="allow"` lets a `before_action` / `after_action` hook attach custom per-round fields (e.g. a `permission_result`) via `ota_ctx._current_record()`.

### ThinkResult (a worker's decision)

```python
class ThinkResult(BaseModel):
    step_content: str = ""                       # think text / final answer / serialized structured result
    tool_calls: List[StepToolCall] = []          # tool calls to execute this step
```

A flat decision assembled by the worker. No `tool_calls` IS the finish. Both the `yield ThinkUnit(...)` result and the run's `final_answer` are this decision's `step_content`.

### Step

```python
class Step(BaseModel):
    result: Optional[Any] = None    # the act-phase result (an ActionResult, or None for a content-only finish)
```

### ToolResult (returned by `yield ActionCall`)

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

### StepToolCall / ToolArgument

```python
class StepToolCall(BaseModel):
    tool: str
    tool_arguments: List[ToolArgument]

class ToolArgument(BaseModel):
    name: str
    value: str    # coerced to str
```

## Tool Definition

```python
from bridgic.core.agentic.tool_specs import FunctionToolSpec

async def my_tool(param1: str, param2: int) -> str:
    """Tool description visible to LLM."""
    return "result"

tool_spec = FunctionToolSpec.from_raw(my_tool)
# Declare on an OTA context: MyOTAContext.tool(tool_spec)  — or @MyOTAContext.tool on the function.
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
#          HumanCall, LLMCall, EnterAgent — each a TraceStep.

# Save / Load
agent._agent_trace.save("trace.json")
loaded = AgentTrace.load("trace.json")  # Returns plain dict
```

Each `TraceStep` carries `name`, `step_content`, `tool_calls` (`List[RecordedToolCall]`), `observation`, `observation_hash`, `output_type` (`StepOutputType`), `structured_output`, `structured_output_class`, `llm_call_protocol` (set on `LLMCall` steps), and `think_agent_name` (set on `ThinkAgent` steps).

`StepOutputType`: `TOOL_CALLS` / `CONTENT_ONLY` / `LLM_CALL` / `THINK_AGENT` / `HUMAN_CALL` / `ENTER_AGENT`.
