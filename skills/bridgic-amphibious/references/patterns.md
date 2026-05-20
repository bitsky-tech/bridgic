# Bridgic Amphibious Code Patterns

## Table of Contents
- [Minimal Agent (Agent Mode)](#minimal-agent-agent-mode)
- [Workflow Mode](#workflow-mode)
- [Built-in Tools](#built-in-tools)
- [Human-in-the-Loop](#human-in-the-loop)
- [Amphiflow Mode](#amphiflow-mode)
- [EnterAgent in Workflow](#enteragent-in-workflow)
- [LLMCall in Workflow](#llmcall-in-workflow)
- [RETURN — explicit return values](#return--explicit-return-values)
- [Custom Worker](#custom-worker)
- [Think Agent (External Agent Delegation)](#think-agent-external-agent-delegation)
- [Structured Output (output_schema)](#structured-output-output_schema)
- [Custom Context](#custom-context)
- [Phase Annotation](#phase-annotation)
- [Cognitive Policies](#cognitive-policies)
- [OTC Hooks](#otc-hooks)
- [Skills Usage](#skills-usage)
- [Memory Configuration](#memory-configuration)
- [Conditional Loops](#conditional-loops)
- [Tool & Skill Filtering](#tool--skill-filtering)
- [Execution Tracing](#execution-tracing)

---

## Minimal Agent (Agent Mode)

```python
from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveContext, CognitiveWorker, think_unit,
    ThinkUnit,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

# 1. Define tools
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22 C in {city}"

get_weather_tool = FunctionToolSpec.from_raw(get_weather)

# 2. Define agent — on_agent yields ThinkUnit("name") to invoke the descriptor.
class WeatherAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline("Look up weather and provide a summary."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        yield ThinkUnit("planner")

# 3. Run
agent = WeatherAgent(verbose=True)
summary = await agent.arun(
    llm=llm,
    goal="Check the weather in Tokyo and London.",
    tools=[get_weather_tool],
)
print(agent.final_answer)  # auto-captured from finishing step's step_content
```

## Workflow Mode

Pure workflow mode runs deterministically and does not need an LLM — only
override `on_workflow`, leave `on_agent` alone.

```python
from bridgic.amphibious import ActionCall, RETURN

class WeatherWorkflow(AmphibiousAutoma[CognitiveContext]):
    async def on_workflow(self, ctx: CognitiveContext):
        tokyo = yield ActionCall("get_weather", city="Tokyo")
        london = yield ActionCall("get_weather", city="London")

        tokyo_val = tokyo[0].result if tokyo else "N/A"
        london_val = london[0].result if london else "N/A"
        yield RETURN(f"Tokyo: {tokyo_val}, London: {london_val}")

workflow = WeatherWorkflow()  # No LLM needed for pure workflow mode
result = await workflow.arun(
    goal="Check weather",
    tools=[get_weather_tool],
)
```

## Built-in Tools

Every `AmphibiousAutoma` agent receives seven built-in tools in `context.tools` automatically — no manual wiring. Names are snake_case and work in every mode.

| Tool | Purpose |
|------|---------|
| `request_human` | Ask the human operator a question (HITL) |
| `bash` | Execute a shell command |
| `read_file` | Read a file with line numbers (required before `write_file` / `edit_file`) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Exact-string replacement with uniqueness check |
| `glob` | Find files by pattern |
| `grep` | Regex search across files |

### Default — relying on auto-injection

```python
class CodeAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(
        CognitiveWorker.inline("Investigate the codebase and report findings."),
        max_attempts=20,
    )
    async def on_agent(self, ctx):
        yield ThinkUnit("worker")

# All seven built-ins are present. Anything you pass in tools=[...] is added
# on top, deduped by name.
await CodeAgent().arun(llm=llm, goal="What does this repo do?")
```

### Calling built-ins from on_workflow

```python
from bridgic.amphibious import ActionCall

class ConfigPatcher(AmphibiousAutoma[CognitiveContext]):
    async def on_workflow(self, ctx):
        files = yield ActionCall("glob", pattern="**/conf.yaml", path="/abs/repo")
        # Read-before-edit invariant: each path must be read first.
        yield ActionCall("read_file", file_path="/abs/repo/conf.yaml")
        yield ActionCall(
            "edit_file",
            file_path="/abs/repo/conf.yaml",
            old_string="threshold: 5",
            new_string="threshold: 10",
        )
```

### Restricting which built-ins are injected

```python
class ReadOnlyAgent(AmphibiousAutoma[CognitiveContext]):
    # Class-level filter — these and only these are injected by arun().
    builtin_tools = frozenset({"request_human", "read_file", "glob", "grep"})

    worker = think_unit(CognitiveWorker.inline("Audit the code."), max_attempts=10)
    async def on_agent(self, ctx):
        yield ThinkUnit("worker")
```

```python
# Runtime override — wins over the class attribute.
await agent.arun(goal="quick read-only sweep", builtin_tools=["read_file", "grep"])

# Empty iterable opts out entirely.
await agent.arun(goal="...", builtin_tools=[])
```

Unknown names fail loudly at `arun()` entry — `frozenset({"read_files"})` (typo) raises `ValueError` rather than silently producing a tool-less agent.

### Combining with think_unit tool filters

`think_unit(tools=[...])` filters by tool name. Built-in names work the same as any user tool, which lets you gate phases by capability:

```python
class PhaseGated(AmphibiousAutoma[CognitiveContext]):
    investigate = think_unit(
        CognitiveWorker.inline("Investigate."),
        tools=["read_file", "glob", "grep"],   # exploration only
        max_attempts=10,
    )
    apply = think_unit(
        CognitiveWorker.inline("Apply the planned change."),
        tools=["read_file", "edit_file"],       # no bash, no overwrite
        max_attempts=5,
    )
    async def on_agent(self, ctx):
        yield ThinkUnit("investigate")
        yield ThinkUnit("apply")
```

### Read-before-modify safety in practice

`write_file` (for existing files) and `edit_file` refuse to act on a path that hasn't been read in the current `arun()` call, AND refuse if the file's mtime advanced between read and modify. The tracker is reset at every `arun()` entry, so the invariant is per-run.

```python
async def on_workflow(self, ctx):
    # Without this read, the next ActionCall raises RuntimeError.
    yield ActionCall("read_file", file_path="/abs/conf.yaml")
    yield ActionCall(
        "edit_file",
        file_path="/abs/conf.yaml",
        old_string="threshold: 5",
        new_string="threshold: 10",
    )
```

## Human-in-the-Loop

Two entry points for requesting human input — both go through the same `@human_channel` registry:

| Entry | Where | Driver |
|-------|-------|--------|
| `yield HumanCall(prompt=, channel=)` | `on_workflow`, hooks | Deterministic — you decide when to ask |
| Auto-injected `request_human` tool | LLM-driven, inside a `ThinkUnit` | Autonomous — the LLM decides when to ask |

There is **no** code-level imperative API like `self.request_human(...)`, and `yield HumanCall` is rejected in `on_agent` scope. If the agent needs to ask a human, that happens through the LLM-driven tool path inside a `ThinkUnit`.

### Entry 1: HumanCall in on_workflow() (deterministic)

```python
from bridgic.amphibious import ActionCall, HumanCall, RETURN

class ConfirmableWorkflow(AmphibiousAutoma[CognitiveContext]):
    async def on_workflow(self, ctx: CognitiveContext):
        result = yield ActionCall(
            "search_flights", origin="Beijing", destination="Tokyo", date="2024-06-01",
        )
        feedback = yield HumanCall(prompt="Found flights. Book CA123?")
        if feedback == "yes":
            yield ActionCall("book_flight", flight_number="CA123")
        else:
            yield RETURN("Booking cancelled by user.")
```

### Entry 2: LLM tool (autonomous, any mode)

`request_human` is auto-injected as one of the [built-in tools](#built-in-tools), so the LLM can call it from any `ThinkUnit` (in `AGENT`, workflow step-level fallback, full fallback, or `AMPHIFLOW`'s `on_agent`) without manual wiring:

```python
class AutonomousAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(
        CognitiveWorker.inline(
            "Execute the task. Call request_human when you need user input."
        ),
        max_attempts=10,
    )
    async def on_agent(self, ctx):
        yield ThinkUnit("worker")

agent = AutonomousAgent()
await agent.arun(llm=llm, goal="Plan a trip", tools=[search_tool])
```

### Custom UI Integration via @human_channel

`@human_channel` is the only customization mechanism for HITL UI integration — there is no `human_input` template method on `AmphibiousAutoma`. **It is a method decorator**: apply it to an `async` method of your `AmphibiousAutoma` subclass. The framework collects all decorated methods into a per-class registry at class-definition time (via `__init_subclass__`).

#### Single implicit handler

Register exactly one handler on the agent class, leave `channel=None` everywhere — both `HumanCall(channel=None)` and the auto-injected `request_human` tool use it.

```python
from bridgic.amphibious import human_channel, HumanCall, RETURN

class WebAgent(AmphibiousAutoma[CognitiveContext]):
    @human_channel("web")
    async def ask_web(self, prompt: str) -> str:
        return await websocket.send_and_receive(prompt)

    async def on_workflow(self, ctx):
        # Implicit channel resolution: only one handler registered on
        # this class, so HumanCall(channel=None) routes to ask_web.
        feedback = yield HumanCall(prompt="Confirm deploy?")
        yield RETURN(feedback)
```

You can also use the bare form `@human_channel` (no parens) — the channel name then defaults to the method name.

#### Multiple named handlers

With 2+ handlers registered on the same class, address each by name from either side. Workflow uses `HumanCall(channel="name", ...)`; agent mode has the LLM pass `channel="name"` to the auto-injected `request_human` tool. Both routes go through the same `@human_channel` registry, so the agent loop can choose its target as freely as the workflow can.

The LLM does not need to memorise channel names — the auto-injected `request_human` spec is rebuilt per agent class from its `@human_channel` registry, so the `channel` parameter's JSON schema is constrained to an `enum` of the actual registered names and the description lists them. Even with no extra wording in the system prompt, the LLM is hard-bounded to valid channels.

```python
class HybridAgent(AmphibiousAutoma[CognitiveContext]):
    @human_channel("feishu")
    async def ask_feishu(self, prompt: str) -> str:
        return await send_to_feishu_and_wait(prompt)

    @human_channel("slack")
    async def ask_slack(self, prompt: str) -> str:
        return await send_to_slack_and_wait(prompt)

    triage = think_unit(
        CognitiveWorker.inline(
            "Route the question to the right human. Use channel='feishu' "
            "for engineering questions, channel='slack' for product/design. "
            "Call request_human(prompt, channel=...) — the channel arg is "
            "required when multiple channels are registered."
        )
    )

    async def on_workflow(self, ctx):
        # Workflow side picks the channel explicitly.
        approval = yield HumanCall(channel="feishu", prompt="Approve deploy?")
        followup = yield HumanCall(channel="slack", prompt="Anything else?")

    async def on_agent(self, ctx):
        # Agent side: the LLM passes `channel="feishu"` or `channel="slack"`
        # to the request_human tool based on the goal.
        yield ThinkUnit("triage")
```

## Amphiflow Mode

When a class overrides both `on_agent` and `on_workflow` (with `on_workflow` as
an async generator), `RunMode.AUTO` resolves to `AMPHIFLOW`: the workflow runs
deterministically through the peer state-machine dispatcher, and on a step
failure the framework runs `on_agent` to recover via the slot + injected
`resolve_step_fallback` tool. You may also pass `mode=RunMode.AMPHIFLOW`
explicitly.

```python
from bridgic.amphibious import RunMode, ActionCall, ThinkUnit

class FormFiller(AmphibiousAutoma[CognitiveContext]):
    fixer = think_unit(
        CognitiveWorker.inline("Diagnose the problem and fix it."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        yield ThinkUnit("fixer")

    async def on_workflow(self, ctx: CognitiveContext):
        yield ActionCall("fill_field", field_name="username", value="john")
        yield ActionCall("fill_field", field_name="email", value="john@example.com")
        yield ActionCall("click_button", button_name="submit")

# Workflow runs; on a failed ActionCall the framework runs on_agent under a
# snapshot, allocates a _FallbackSlot (default = empty List[ToolResult] for
# ActionCall), and injects a resolve_step_fallback(result: Any) -> str tool.
# Whatever the LLM passes to that tool is asend()'d back to the workflow
# generator.  Each successful ActionCall resets the consecutive-failure
# counter; reaching max_consecutive_fallbacks triggers full fallback.
agent = FormFiller(verbose=True)
result = await agent.arun(
    llm=llm,
    goal="Fill and submit the form",
    tools=[fill_field_tool, click_button_tool, solve_captcha_tool],
    mode=RunMode.AMPHIFLOW,
    max_consecutive_fallbacks=2,
)
```

## EnterAgent in Workflow

`EnterAgent` is the *explicit* mode-switch from deterministic workflow to LLM-driven agent. The state-machine dispatcher suspends the workflow, snapshots the context (per the `goal` / `tools` / `skills` / `history` fields), runs `on_agent`, and resumes when the agent generator naturally exhausts.

```python
from bridgic.amphibious import EnterAgent, ActionCall, ThinkUnit

class PriceComparer(AmphibiousAutoma[CognitiveContext]):
    analyst = think_unit(CognitiveWorker.inline("Analyze prices."), max_attempts=3)

    async def on_agent(self, ctx):
        yield ThinkUnit("analyst")

    async def on_workflow(self, ctx):
        yield ActionCall("search_price", platform="Amazon", product="laptop")
        yield ActionCall("search_price", platform="eBay", product="laptop")

        # Delegate open-ended analysis to LLM, scoped to a sub-goal.
        # The agent sees only the listed tools/skills while inside this snapshot.
        yield EnterAgent(
            goal="Analyze prices and decide if we need more platforms.",
            tools=["search_price"],
        )

        # When on_agent exhausts, control returns here.
        yield ActionCall("publish_decision")
```

`EnterAgent` does **not** accept `worker=` or `max_attempts=` — those control *how* the agent thinks, which belongs in the `think_unit` declaration, not at the call site. The agent uses whatever `on_agent` does (typically `yield ThinkUnit("...")`).

## LLMCall in Workflow

`LLMCall` lets `on_workflow` invoke the agent's LLM directly through one of three protocols, without wrapping the call in a `CognitiveWorker`.

```python
from bridgic.amphibious import LLMCall, RETURN
from bridgic.core.model.protocols import PydanticModel
from pydantic import BaseModel

class Outline(BaseModel):
    sections: list[str]

class OutlineWriter(AmphibiousAutoma[CognitiveContext]):
    async def on_workflow(self, ctx):
        # 1. Free-form chat.
        notes = yield LLMCall.chat("Brainstorm a topic for a 5-minute talk.")

        # 2. Structured output via Constraint.
        outline = yield LLMCall.structure_output(
            f"Turn these notes into a 5-section outline:\n{notes}",
            constraint=PydanticModel(model=Outline),
        )

        # 3. tool_selector returns (List[ToolCall], Optional[reply_text]).
        # tool_calls, reply = yield LLMCall.tool_selector("...", tools=[...])

        yield RETURN(outline.model_dump_json())
```

`LLMCall` is **not** allowed inside `on_agent` — the agent body is reserved for orchestrating cognitive steps via `ThinkUnit`. Direct LLM calls belong in `on_workflow`, hooks, or inside a `CognitiveWorker`'s `thinking()` method.

## RETURN — explicit return values

PEP 525 forbids `return value` inside async generators. `RETURN(value)` is the framework's workaround:

```python
from bridgic.amphibious import RETURN, ThinkUnit

class Summarizer(AmphibiousAutoma[CognitiveContext]):
    summarizer = think_unit(CognitiveWorker.inline("Summarize."), max_attempts=3)

    async def on_agent(self, ctx):
        yield ThinkUnit("summarizer")
        # Override the auto-captured final answer with something explicit.
        last = ctx.cognitive_history.get_all()[-1].content
        yield RETURN(f"FINAL: {last}")
```

When yielded from a top-level `on_agent` / `on_workflow`, `RETURN(value)` writes `str(value)` to `self._final_answer` and closes the generator. Anything yielded after a `RETURN` is unreachable.

`RETURN` is allowed in any scope (workflow / agent / hook), but its value-handoff semantics are the *only* extension to PEP 525 — the framework does not extend `RETURN` for fallback value handoff (that's the `_FallbackSlot` + `resolve_step_fallback` mechanism, see [architecture.md](architecture.md#workflow-fallback-mechanism)).

## Custom Worker

```python
class DestinationAnalyzer(CognitiveWorker):
    async def thinking(self) -> str:
        return "Analyze the destination and suggest a day-by-day plan."

    async def observation(self, context: CognitiveContext):
        return (
            f"Current goal: {context.goal}\n"
            f"Tip: Visit attractions early morning to avoid crowds."
        )

class TravelPlanner(AmphibiousAutoma[CognitiveContext]):
    analyzer = think_unit(DestinationAnalyzer(), max_attempts=3)
    planner = think_unit(
        CognitiveWorker.inline("Create a detailed itinerary."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        yield ThinkUnit("analyzer")
        yield ThinkUnit("planner")
```

## Think Agent (External Agent Delegation)

`think_agent` wraps an `AgentWorker` — the external-agent peer of `think_unit`. It delegates a sub-goal to an out-of-process coding-agent CLI — `claude code` (`ClaudeCodeAgent`) or OpenAI `codex` (`CodexAgent`) — instead of an in-process LLM cycle. Project tools reach the external agent through an in-process MCP bridge, so its tool calls still flow through the parent's hooks and trace.

### Default — claude code as the think agent

```python
from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveContext, AgentWorker, ClaudeCodeAgent,
    ThinkAgent, think_agent, RETURN,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

async def record_finding(text: str) -> str:
    """Record one review finding."""
    return f"recorded: {text}"

class Reviewer(AmphibiousAutoma[CognitiveContext]):
    # ClaudeCodeAgent (a BaseAgent) is the shipped claude-code driver.
    reviewer = think_agent(
        AgentWorker(ClaudeCodeAgent(
            allowed_builtin_tools=["Read", "Grep"],   # claude's own tools
            completion_timeout=300.0,
        )),
        expose_tools=["record_finding"],              # project tools sent over MCP
    )

    async def on_agent(self, ctx):
        # yield ThinkAgent returns the string the external agent passed
        # to its `agent_done` completion signal.
        summary = yield ThinkAgent("reviewer", goal="Review ./src and record findings.")
        yield RETURN(summary)

# A pure ThinkAgent flow needs no `llm` — the external agent is the brain.
agent = Reviewer()
result = await agent.arun(tools=[FunctionToolSpec.from_raw(record_finding)])
```

`CodexAgent` (OpenAI codex) is a drop-in alternative — `AgentWorker(CodexAgent())` in place of `AgentWorker(ClaudeCodeAgent(...))`; the rest of the pattern is identical.

### Custom AgentWorker — reshape the prompt

Subclass `AgentWorker` and override `thinking()` to restructure the message handed to the external agent (the `AgentWorker` analog of `CognitiveWorker.thinking()`):

```python
class StrictReviewer(AgentWorker):
    async def thinking(self, context) -> str:
        base = await super().thinking(context)
        return base + "\n\nBe extremely thorough. Flag every TODO."

class Reviewer(AmphibiousAutoma[CognitiveContext]):
    reviewer = think_agent(StrictReviewer(ClaudeCodeAgent()))

    async def on_agent(self, ctx):
        yield ThinkAgent("reviewer", goal="Audit the module.")
```

CLI-level knobs (which binary, the sandbox / permission policy, the completion timeout) live on the `BaseAgent` subclass; cognitive customization (`thinking` / `observation` / `before_action` / `after_action`) lives on the `AgentWorker` — the same split as `BaseLlm` vs `CognitiveWorker`.

## Structured Output (output_schema)

```python
from pydantic import BaseModel, Field
from bridgic.amphibious import ThinkUnit, RETURN

class PlanResult(BaseModel):
    phases: list[str] = Field(description="Execution phases")
    estimated_steps: int = Field(description="Total steps needed")

class PlannerAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline(
            "Create a step-by-step execution plan.",
            output_schema=PlanResult,
        ),
        max_attempts=1,
    )

    async def on_agent(self, ctx: CognitiveContext):
        plan = yield ThinkUnit("planner")  # Returns a PlanResult instance
        yield RETURN(plan.model_dump_json())
```

## Custom Context

```python
from pydantic import Field, ConfigDict
from bridgic.amphibious import (
    CognitiveContext, CognitiveHistory, ActionResult, ThinkUnit,
)

class DocumentContext(CognitiveContext):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_document: str = Field(
        default="",
        description="Name of the document being analyzed"
    )
    analysis_results: dict = Field(
        default_factory=dict,
        description="Accumulated results keyed by document name"
    )
    internal_state: str = Field(
        default="",
        json_schema_extra={"display": False}  # Hidden from LLM
    )

class DocumentAnalyzer(AmphibiousAutoma[DocumentContext]):
    analyzer = think_unit(
        CognitiveWorker.inline("Analyze the current document."),
        max_attempts=5,
    )

    async def after_action(self, step_result, ctx: DocumentContext):
        """Keep custom context in sync with tool results.

        Hook can be a coroutine OR an async generator — pick whichever you
        need. Coroutines are fine when there's nothing to yield.
        """
        action_result = step_result.result
        if not isinstance(action_result, ActionResult):
            return
        for step in action_result.results:
            if step.success and step.tool_name == "read_document":
                doc_name = step.tool_arguments.get("doc_name", "")
                ctx.current_document = doc_name
                ctx.analysis_results[doc_name] = step.tool_result

    async def on_agent(self, ctx: DocumentContext):
        yield ThinkUnit("analyzer")
```

## Phase Annotation

```python
class ContentCreator(AmphibiousAutoma[CognitiveContext]):
    researcher = think_unit(
        CognitiveWorker.inline("Research the topic thoroughly."),
        max_attempts=3,
    )
    writer = think_unit(
        CognitiveWorker.inline("Write the article using gathered research."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        # Phase 1: Research
        async with self.snapshot(goal="Gather research material on renewable energy"):
            yield ThinkUnit("researcher")

        # Phase 2: Write
        async with self.snapshot(goal="Write the article using the research"):
            yield ThinkUnit("writer")
```

## Cognitive Policies

```python
# Enable all three policies on a single worker
class AnalystAgent(AmphibiousAutoma[CognitiveContext]):
    analyst = think_unit(
        CognitiveWorker.inline(
            "Perform a comprehensive analysis.",
            enable_rehearsal=True,    # Mental simulation
            enable_reflection=True,   # Information assessment
            # Acquiring is always active by default
        ),
        max_attempts=10,
    )

    async def on_agent(self, ctx: CognitiveContext):
        yield ThinkUnit("analyst")
```

## OTC Hooks

OTC hooks split into two groups by how the framework invokes them:

- **Pre-think / post-act hooks** (`observation`, `before_action`, `after_action`) go through `_invoke_template` and accept BOTH async-coroutine and async-generator forms. Use the coroutine form when you just want a return value; use the generator form when you want to yield framework primitives (`ActionCall`, `HumanCall`, `LLMCall`, `RETURN`) inside the hook.
- **Action-execution hooks** (`action_tool_call`, `action_custom_output`) are awaited directly by the action phase, NOT routed through the dispatcher. They must be plain coroutines — yielding framework primitives from these hooks does nothing.

### observation — Inject Custom Perception

```python
class SecurityWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return "Analyze the system for security issues."

    async def observation(self, context: CognitiveContext):
        return f"Security policy: Read-only audit mode."

class SecurityAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(SecurityWorker(), max_attempts=5)

    # Agent-level observation: shared across all workers
    async def observation(self, ctx: CognitiveContext):
        return f"System: production-server-01, Uptime: 45 days"

    async def on_agent(self, ctx):
        yield ThinkUnit("auditor")
```

#### Generator-form observation — capture a fresh snapshot every cycle

When the observation needs a live tool call (e.g. take a browser snapshot, query an external system) before each `ThinkUnit`, write the hook as an async generator and `yield ActionCall` / `RETURN`:

```python
class BrowserAgent(AmphibiousAutoma[CognitiveContext]):
    explorer = think_unit(CognitiveWorker.inline("Decide next click."), max_attempts=10)

    # Async-generator form: yield ActionCall to fetch the snapshot, then
    # RETURN it so the framework writes it into ctx.observation for the
    # upcoming think step.
    async def observation(self, ctx):
        snapshot = yield ActionCall("bridgic_browser_snapshot")
        yield RETURN(snapshot[0].result if snapshot else None)

    async def on_agent(self, ctx):
        yield ThinkUnit("explorer")
```

Hook-scope semantics: the `yield ActionCall(...)` here is a raw tool execution. It does NOT re-enter `observation` / `before_action` / `after_action` (hooks are not OTC participants — only `on_workflow` is). The same generator form applies to `before_action` (e.g. yield an audit-log ActionCall before every tool dispatch) and `after_action` (e.g. yield a refresh ActionCall after each step). Exhausting the generator without `RETURN` is treated as no-op / passthrough; `RETURN(value)` becomes the hook's effective return value.

### build_messages — Reshape LLM Messages

```python
from bridgic.core.model.types import Message

class StrictWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return "Perform a security audit."

    async def build_messages(self, think_prompt, tools_description,
                             output_instructions, context_info):
        rules = "\n\nRULES:\n1. NEVER call delete_file.\n2. NEVER read .env files."
        system = f"{think_prompt}{rules}\n\n{tools_description}\n\n{output_instructions}"
        return [
            Message.from_text(text=system, role="system"),
            Message.from_text(text=context_info, role="user"),
        ]
```

### before_action — Filter Dangerous Calls

```python
class SafeAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(CognitiveWorker.inline("Audit the system."), max_attempts=5)

    async def before_action(self, decision_result, ctx):
        if isinstance(decision_result, list):
            blocked = {"delete_file", "drop_table"}
            return [(tc, ts) for tc, ts in decision_result
                    if ts.tool_name not in blocked] or decision_result
        return decision_result

    async def on_agent(self, ctx):
        yield ThinkUnit("auditor")
```

### after_action — Update Context After Execution

```python
class TrackingAgent(AmphibiousAutoma[MyContext]):
    worker = think_unit(CognitiveWorker.inline("Process data."), max_attempts=5)

    async def after_action(self, step_result, ctx: MyContext):
        action_result = step_result.result
        if isinstance(action_result, ActionResult):
            for r in action_result.results:
                if r.success:
                    ctx.processed_count += 1

    async def on_agent(self, ctx):
        yield ThinkUnit("worker")
```

### action_custom_output — Post-process Structured Output

```python
from pydantic import BaseModel

class AuditReport(BaseModel):
    findings: list[str]
    risk_level: str

class RedactingAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(
        CognitiveWorker.inline("Produce an audit report.", output_schema=AuditReport),
        max_attempts=1,
    )

    async def action_custom_output(self, decision_result, ctx):
        if isinstance(decision_result, AuditReport):
            decision_result.findings = [
                f.replace("sk-xxx", "[REDACTED]") for f in decision_result.findings
            ]
        return decision_result

    async def on_agent(self, ctx):
        yield ThinkUnit("auditor")
```

## Skills Usage

```python
from bridgic.amphibious import Skill

fundamental_skill = Skill(
    name="fundamental-analysis",
    description="Evaluate stock's intrinsic value using financial metrics",
    content="## Procedure\n1. Get financials\n2. Evaluate P/E ratio\n...",
)

agent = MyAgent()
result = await agent.arun(
    llm=llm,
    goal="Analyze AAPL stock",
    tools=[get_financials_tool],
    skills=[fundamental_skill],
)

# Or load from file
ctx = CognitiveContext(goal="...")
ctx.skills.add_from_file("skills/analysis/SKILL.md")
ctx.skills.load_from_directory("skills/")
```

## Memory Configuration

```python
from bridgic.amphibious import CognitiveHistory

# Short tasks: large working memory
history = CognitiveHistory(working_memory_size=10, short_term_size=30)

# Long tasks: aggressive compression
history = CognitiveHistory(
    working_memory_size=2,
    short_term_size=5,
    compress_threshold=3,
)

agent = MyAgent()
result = await agent.arun(
    llm=llm,
    goal="Long running task",
    tools=[...],
    cognitive_history=history,
)
```

## Conditional Loops

`ThinkUnit("name", until=...)` loops the named think unit until the condition holds, with optional per-call overrides.

```python
class IterativeAgent(AmphibiousAutoma[CognitiveContext]):
    researcher = think_unit(
        CognitiveWorker.inline("Research ONE aspect of the topic."),
        max_attempts=10,
    )

    async def on_agent(self, ctx: CognitiveContext):
        # Loop until condition met (uses the descriptor's max_attempts).
        yield ThinkUnit(
            "researcher",
            until=lambda ctx: len(ctx.cognitive_history) >= 3,
        )

        # Loop with per-call overrides.
        yield ThinkUnit(
            "researcher",
            until=lambda ctx: some_condition(ctx),
            max_attempts=50,
            tools=["search"],
        )
```

## Tool & Skill Filtering

```python
class MultiPhaseAgent(AmphibiousAutoma[CognitiveContext]):
    searcher = think_unit(
        CognitiveWorker.inline("Search for information."),
        max_attempts=5,
        tools=["search", "browse"],       # Only these tools visible
        skills=["research"],              # Only these skills visible
    )
    writer = think_unit(
        CognitiveWorker.inline("Write the report."),
        max_attempts=3,
        tools=["write_file"],
    )

    async def on_agent(self, ctx):
        yield ThinkUnit("searcher")
        yield ThinkUnit("writer")
```

## Execution Tracing

```python
agent = MyAgent(verbose=True)
result = await agent.arun(
    llm=llm,
    goal="...",
    tools=[...],
    trace=True,                 # activate the in-memory AgentTrace
    workdir="./.bridgic",       # optional: also persist <workdir>/runs/<run_id>/trace.json
)

# Access the trace — a flat unified dict, kept on the agent after the run.
trace = agent._agent_trace.build()
# {"goal": str, "metadata": {...}, "history": [TraceStep, ...]}

for step in trace["history"]:
    print(f"  {step.step_content[:80]}")
    for tc in step.tool_calls:          # List[RecordedToolCall]
        print(f"    -> {tc.tool_name}")

# Save / Load
agent._agent_trace.save("trace.json")
loaded = AgentTrace.load("trace.json")  # Returns plain dict
```
