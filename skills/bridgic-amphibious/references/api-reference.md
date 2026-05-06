# Bridgic Amphibious API Reference

## Table of Contents
- [LLM Setup](#llm-setup)
- [Imports](#imports)
- [CLI Scaffolding](#cli-scaffolding)
- [AmphibiousAutoma](#amphibiousautoma)
- [CognitiveWorker](#cognitiveworker)
- [think_unit](#think_unit)
- [ActionCall, HumanCall, AgentCall](#actioncall-humancall-agentcall)
- [Human-in-the-Loop](#human-in-the-loop)
- [Built-in Tools](#built-in-tools)
- [CognitiveContext](#cognitivecontext)
- [Context and Exposure](#context-and-exposure)
- [Data Models](#data-models)
- [Tool Definition](#tool-definition)
- [AgentTrace](#agenttrace)

---

## LLM Setup

Amphibious agents accept a `BaseLlm` instance from one of the bridgic LLM provider packages. The LLM is required for `AGENT` and `AMPHIFLOW` modes; pure `WORKFLOW` mode (where `on_workflow` is the only overridden template method) can run without one. Install one of the provider packages:

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
    AmphibiousAutoma, think_unit, AgentTrace, ThinkUnitDescriptor,
    # Worker
    CognitiveWorker, _DELEGATE,
    # Context
    CognitiveContext, CognitiveHistory, CognitiveTools, CognitiveSkills,
    Context, Exposure, LayeredExposure, EntireExposure,
    # Workflow yield types
    ActionCall, HumanCall, AgentCall, HUMAN_INPUT_EVENT_TYPE,
    # Data models
    Step, Skill, RunMode, ErrorStrategy,
    ActionResult, ActionStepResult, ToolResult,
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
    llm: Optional[BaseLlm] = None,  # Optional. Required for AGENT/AMPHIFLOW modes
    name: str = None,                # Optional agent name
    verbose: bool = False,           # Enable execution logging
)
```

### Class Attributes (Override in Subclasses)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `builtin_tools` | `Optional[FrozenSet[str]]` | `None` | Filter for which entries of [`ALL_BUILTIN_TOOLS`](#built-in-tools) to auto-inject during `arun()`. `None` injects all; a frozenset of names restricts to that subset; `frozenset()` opts out entirely. The runtime `arun(builtin_tools=...)` kwarg wins over this attribute. Unknown names raise `ValueError` at `arun()` entry. |
| `WORKFLOW_STEP_FALLBACK_MAX_ATTEMPTS` | `int` | `5` | Max think-unit attempts when a workflow step falls back to agent mode for per-step recovery. |

### arun() — Main Entry Point

```python
await agent.arun(
    # Context: either pre-built or auto-created
    context: CognitiveContextT = None,  # Pre-built context
    goal: str = "",                      # Auto-create: goal
    tools: List[ToolSpec] = [],          # Auto-create: tools
    skills: List[Skill] = [],            # Auto-create: skills
    cognitive_history: CognitiveHistory = None,  # Auto-create: custom history

    # Execution control
    mode: RunMode = RunMode.AUTO,
    trace_running: bool = False,
    max_consecutive_fallbacks: int = 1,
    builtin_tools: Optional[Iterable[str]] = None,  # Override class-level builtin_tools filter
) -> str
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `context` | `CognitiveContextT` | Current context after `arun()` |
| `final_answer` | `Optional[str]` | Auto-captured from finishing step's `step_content` |
| `llm` | `Optional[BaseLlm]` | The agent's LLM (`None` is allowed for pure WORKFLOW mode) |
| `spent_tokens` | `int` | Token usage for last `arun()` |
| `spent_time` | `float` | Time in seconds for last `arun()` |

### Template Methods (Override in Subclasses)

A subclass must override at least one of `on_agent` and `on_workflow`. Under
`RunMode.AUTO` the runtime picks the mode from which methods are overridden:
agent-only → `AGENT`, workflow-only → `WORKFLOW`, both → `AMPHIFLOW`.

```python
# LLM-driven orchestration (override for AGENT or AMPHIFLOW modes)
async def on_agent(self, ctx: CognitiveContextT) -> None: ...

# Deterministic workflow (override for WORKFLOW or AMPHIFLOW modes)
async def on_workflow(self, ctx: CognitiveContextT) -> AsyncGenerator: ...

# Optional hooks
async def observation(self, ctx) -> Optional[str]: ...
async def before_action(self, decision_result, ctx) -> Any: ...
async def after_action(self, step_result, ctx) -> None: ...
async def action_tool_call(self, tool_list, ctx) -> ActionResult: ...
async def action_custom_output(self, decision_result, ctx) -> Any: ...

# Human-in-the-loop (override to integrate with your UI)
async def human_input(self, data: Dict[str, Any]) -> str: ...
```

### Human-in-the-Loop Methods

```python
# Request human input (use in on_agent or from tools)
await self.request_human(
    prompt: str,            # Question to present to the human
    timeout: float = None,  # Seconds before TimeoutError; None = wait forever
) -> str

# Template method — override to replace default stdin with your UI
async def human_input(self, data: Dict[str, Any]) -> str:
    # data contains: {"prompt": "...", "timeout": ...}
    # Default: reads from stdin via run_in_executor
    ...
```

### Utility Methods

```python
self.set_final_answer(answer: str)  # Explicitly set final answer

# Phase scoping
async with self.snapshot(goal="Sub-goal", **fields):
    await self.worker
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

# Optional hooks
async def observation(self, context) -> Any: ...           # Return _DELEGATE or str
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
# await think_unit returns the typed instance.
```

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

Use as class variable:

```python
class MyAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(CognitiveWorker.inline("Plan step"), max_attempts=5)

    async def on_agent(self, ctx):
        await self.planner                        # Single execution
        await self.planner.until(condition)        # Loop until condition
        await self.planner.until(                  # With overrides
            condition, max_attempts=50, tools=["search"]
        )
```

### .until() Parameters

```python
await self.think_unit.until(
    condition: Callable[[ctx], bool],  # Sync or async callable
    *,
    max_attempts: int = None,          # Override descriptor max_attempts
    tools: List[str] = None,           # Override tool filter
    skills: List[str] = None,          # Override skill filter
)
```

## ActionCall, HumanCall, AgentCall

Three yield types for `on_workflow()`:

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
    worker: Optional[Any]        # Custom worker for fallback
    tool_args: Dict[str, Any]

    def __init__(self, tool_name: str, *, description: str = "", worker=None, **tool_args): ...
```

### HumanCall — Pause for human input

```python
from bridgic.amphibious import HumanCall

# In on_workflow():
feedback = yield HumanCall(prompt="Confirm this action?")
# feedback: str (the human's response)
```

```python
@dataclass
class HumanCall:
    prompt: str = ""
    timeout: Optional[float] = None  # Seconds; None = wait forever
```

### AgentCall — Delegate to LLM agent mode

```python
from bridgic.amphibious import AgentCall

yield AgentCall(goal="Handle complex case", max_attempts=5)
```

```python
@dataclass
class AgentCall:
    goal: str = ""
    tools: Optional[Any] = None      # None → use context's tools
    skills: Optional[Any] = None     # None → use context's skills
    history: Optional[Any] = None    # None → fresh CognitiveHistory()
    max_attempts: int = 1
    worker: Optional[Any] = None     # None → framework default
```

## Human-in-the-Loop

Three entry points for requesting human input:

| Entry Point | Where | Usage |
|-------------|-------|-------|
| `request_human()` | `on_agent()` | `await self.request_human("Proceed?")` |
| `HumanCall` | `on_workflow()` | `feedback = yield HumanCall(prompt="Confirm?")` |
| `request_human` tool | LLM tool-call, any mode | Auto-injected into `context.tools`; no setup needed |

### request_human as a built-in tool

`request_human` is one of the seven [built-in tools](#built-in-tools) auto-injected into `context.tools` during `arun()`, so the LLM can call it in any mode (AGENT, WORKFLOW fallback, AMPHIFLOW) without you wiring it through `tools=[...]`:

```python
await agent.arun(goal="Plan a trip, ask me if you need confirmation.", tools=[search_tool])
```

Passing `request_human_tool` explicitly is harmless — the injection step deduplicates by tool name. The tool resolves to the running agent through `current_agent` (a `contextvars.ContextVar`), so each concurrent `arun()` task gets its own binding and parallel agents do not interfere.

### HUMAN_INPUT_EVENT_TYPE

```python
from bridgic.amphibious import HUMAN_INPUT_EVENT_TYPE
# Value: "HUMAN_INPUT_REQUEST"
```

Framework-level event type constant used by all three HITL entry points.

## Built-in Tools

Seven `FunctionToolSpec` instances are auto-injected into every `AmphibiousAutoma` agent's `context.tools` during `arun()`, subject to the [`builtin_tools` filter](#class-attributes-override-in-subclasses). They are exported from both `bridgic.amphibious` and `bridgic.amphibious.builtin_tools` as `*_tool` constants.

```python
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
# ALL_BUILTIN_TOOLS: tuple of all seven specs in display order.
# current_agent:    ContextVar bound to the running AmphibiousAutoma during arun().
```

**Error contract.** Tools raise on validation failures. The framework's per-tool exception handler (`_action_tool_call._run_one`) catches every exception and produces:

```python
ActionStepResult(success=False, error=str(exc), tool_result=None)
```

— so the LLM sees the error in the next observation, and `on_workflow` (without fallback) propagates it as `RuntimeError("Tool execution failed for: ...")`. Tools never wrap errors as `<error>...</error>` strings at their own layer.

### request_human

```python
async def request_human(prompt: str) -> str
```

Pause and ask the human operator a question. Internally delegates to `agent.request_human(prompt)` — the same code path used by `await self.request_human(...)` in `on_agent` and `yield HumanCall(prompt=...)` in `on_workflow`. See [Human-in-the-Loop](#human-in-the-loop) for the full HITL story.

### bash

```python
async def bash(command: str, timeout: int = 120000, cwd: str = "") -> str
```

Execute a shell command via the user's default shell. Returns a tagged envelope:

```
<stdout>
...captured stdout...
</stdout>
<stderr>
...captured stderr...
</stderr>
<exit_code>0</exit_code>
```

| Param | Description |
|-------|-------------|
| `command` | Shell command. Multiple commands may be chained with `&&` / `\|\|` / `;`. |
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
result = await agent.arun(..., trace_running=True)

# Access trace
trace = agent._agent_trace.build()
# Returns: {"phases": [...], "orphan_steps": [...], "metadata": {...}}
# phases: steps grouped by self.snapshot() blocks (empty if no phase annotations)
# orphan_steps: steps outside any phase annotation

# Save / Load
agent._agent_trace.save("trace.json")
loaded = AgentTrace.load("trace.json")  # Returns plain dict
```
