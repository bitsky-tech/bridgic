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
- [CognitiveContext](#cognitivecontext)
- [Context and Exposure](#context-and-exposure)
- [Data Models](#data-models)
- [Tool Definition](#tool-definition)
- [AgentTrace](#agenttrace)

---

## LLM Setup

Amphibious agents require a `BaseLlm` instance. Install one of the bridgic LLM provider packages:

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
)
from bridgic.amphibious.builtin_tools import human_request_tool
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message
```

## CLI Scaffolding

Bootstrap a new amphibious project:

```bash
bridgic-amphibious create -n <project-name> [--base-dir <path>] [--task <description>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --name` | (required) | Project directory name |
| `--base-dir` | Current directory | Parent directory for the project |
| `--task` | "Describe your task here." | Initial task description for `task.md` |

Generated structure:

```
<project-name>/
├── task.md          # Task description (input)
├── config.py        # LLM configuration (API base, key, model)
├── tools/           # Tool definitions
│   └── __init__.py
├── workers.py       # Context and data models (ProjectContext)
├── agents.py        # AmphibiousAutoma subclass (TODO template)
├── skills/          # Amphibious skills
├── result/          # Trace and analysis results
└── log/             # Runtime logs
```

## AmphibiousAutoma

```python
class AmphibiousAutoma(Generic[CognitiveContextT]):
```

Base class for dual-mode agents. Subclass with a generic `CognitiveContext` type parameter.

### Constructor

```python
AmphibiousAutoma(
    llm: BaseLlm,           # Required. LLM for workers and auxiliary tasks
    name: str = None,        # Optional agent name
    verbose: bool = False,   # Enable execution logging
)
```

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
    will_fallback: bool = True,
    max_consecutive_fallbacks: int = 1,
) -> str
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `context` | `CognitiveContextT` | Current context after `arun()` |
| `final_answer` | `Optional[str]` | Auto-captured from finishing step's `step_content` |
| `llm` | `BaseLlm` | The agent's LLM |
| `spent_tokens` | `int` | Token usage for last `arun()` |
| `spent_time` | `float` | Time in seconds for last `arun()` |

### Template Methods (Override in Subclasses)

```python
# Required: LLM-driven orchestration
async def on_agent(self, ctx: CognitiveContextT) -> None: ...

# Optional: Deterministic workflow (async generator)
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
| `human_request_tool` | `arun(tools=[...])` | LLM autonomously calls `ask_human` tool |

### human_request_tool — Built-in FunctionToolSpec

```python
from bridgic.amphibious.builtin_tools import human_request_tool

# Use like any other tool — no factory call needed
await agent.arun(goal="...", tools=[search_tool, human_request_tool])
```

Uses `contextvars.ContextVar` for late-binding to the running agent. Each concurrent `arun()` task gets its own binding — safe for parallel execution.

### HUMAN_INPUT_EVENT_TYPE

```python
from bridgic.amphibious import HUMAN_INPUT_EVENT_TYPE
# Value: "HUMAN_INPUT_REQUEST"
```

Framework-level event type constant used by all three HITL entry points.

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
    AMPHIBIOUS = "amphibious"
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
