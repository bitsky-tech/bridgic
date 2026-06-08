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
- [Structured Output](#structured-output)
- [Custom Context](#custom-context)
- [OTA Hooks](#ota-hooks)
- [Conditional Loops](#conditional-loops)
- [Execution Tracing](#execution-tracing)

---

## Minimal Agent (Agent Mode)

```python
from bridgic.amphibious import (
    AmphibiousAutoma, OTAContext, Context,
    CognitiveWorker, think_unit, ThinkUnit,
)
from bridgic.core.model.types import Message, Role

# 1. A tool — a plain async function.
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22 C in {city}"

# 2. Small-loop context — declare the tools this run carries.
class WeatherOTAContext(OTAContext):
    pass

WeatherOTAContext.tool(get_weather)

# 3. Big-loop context — free-form knowledge (optional; bare here).
class WeatherContext(Context):
    pass

# 4. Think worker — assemble a prompt and call the model; return its
#    natural result (the framework adapts it into a decision).
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

# 6. Run.
agent = WeatherAgent(verbose=True)
answer = await agent.arun(llm=llm, user_input="Check the weather in Tokyo and London.")
print(agent.final_answer)  # auto-captured from the finishing step's step_content
```

## Workflow Mode

Pure workflow mode runs deterministically and does not need an LLM — only override `on_workflow`, leave `on_agent` alone.

```python
from bridgic.amphibious import OTAContext, Context, ActionCall, RETURN

class WeatherOTAContext(OTAContext):
    pass

WeatherOTAContext.tool(get_weather)

class WeatherWorkflow(AmphibiousAutoma[WeatherOTAContext, Context]):
    async def on_workflow(self, ota_context, context=None):
        tokyo = yield ActionCall("get_weather", city="Tokyo")     # List[ToolResult]
        london = yield ActionCall("get_weather", city="London")

        tokyo_val = tokyo[0].result if tokyo else "N/A"
        london_val = london[0].result if london else "N/A"
        yield RETURN(f"Tokyo: {tokyo_val}, London: {london_val}")

workflow = WeatherWorkflow()  # No LLM needed for pure workflow mode
result = await workflow.arun(user_input="Check weather")
```

## Built-in Tools

The framework ships seven built-in tool specs. **Nothing is auto-injected** — declare the ones a run needs on its OTA context class via `OTAContext.tool`. Names are snake_case and work in every mode.

| Tool | Purpose |
|------|---------|
| `request_human` | Ask the human operator a question (HITL) |
| `bash` | Execute a shell command |
| `read_file` | Read a file with line numbers (required before `write_file` / `edit_file`) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Exact-string replacement with uniqueness check |
| `glob` | Find files by pattern |
| `grep` | Regex search across files |

### Declaring built-ins on the OTA context

```python
from bridgic.amphibious import (
    AmphibiousAutoma, OTAContext, Context, ThinkUnit, think_unit,
    bash_tool, read_file_tool, write_file_tool, edit_file_tool,
    glob_tool, grep_tool, request_human_tool,
)

class CodeOTAContext(OTAContext):
    pass

for _t in (bash_tool, read_file_tool, write_file_tool, edit_file_tool,
           glob_tool, grep_tool, request_human_tool):
    CodeOTAContext.tool(_t)

# Or declare the whole set at once:
#   from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS
#   for _t in ALL_BUILTIN_TOOLS: CodeOTAContext.tool(_t)

class CodeAgent(AmphibiousAutoma[CodeOTAContext, Context]):
    worker = think_unit(CodeThink(), max_attempts=20)
    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("worker")

await CodeAgent().arun(llm=llm, user_input="What does this repo do?")
```

### Read-only context — declare only safe built-ins

```python
class ReadOnlyOTAContext(OTAContext):
    pass

# Only these are carried; bash / write / edit are unavailable to this run.
for _t in (read_file_tool, glob_tool, grep_tool, request_human_tool):
    ReadOnlyOTAContext.tool(_t)
```

### Calling built-ins from on_workflow

```python
from bridgic.amphibious import ActionCall

class ConfigPatcher(AmphibiousAutoma[CodeOTAContext, Context]):
    async def on_workflow(self, ota_context, context=None):
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

`write_file` (for existing files) and `edit_file` refuse to act on a path that hasn't been read in the current `arun()`, and refuse if the file's mtime advanced between read and modify. The tracker resets at every `arun()` entry.

## Human-in-the-Loop

Two entry points for requesting human input — both go through the same `@human_channel` registry. The `request_human` tool is declared on the OTA context like any other tool.

| Entry | Where | Driver |
|-------|-------|--------|
| `yield HumanCall(prompt=, channel=)` | `on_workflow`, hooks | Deterministic — you decide when to ask |
| `request_human` tool | LLM-driven, inside a `ThinkUnit` | Autonomous — the LLM decides when to ask |

There is **no** code-level imperative API like `self.request_human(...)`, and `yield HumanCall` is rejected in `on_agent`.

### Entry 1: HumanCall in on_workflow() (deterministic)

```python
from bridgic.amphibious import ActionCall, HumanCall, RETURN

class ConfirmableWorkflow(AmphibiousAutoma[MyOTAContext, Context]):
    async def on_workflow(self, ota_context, context=None):
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

Declare `request_human_tool` on the OTA context, and the LLM can call it from any `ThinkUnit`:

```python
from bridgic.amphibious import OTAContext, request_human_tool, ThinkUnit, think_unit

class TripOTAContext(OTAContext):
    pass

TripOTAContext.tool(request_human_tool)
# ... declare any other tools the worker needs ...

class AutonomousAgent(AmphibiousAutoma[TripOTAContext, Context]):
    worker = think_unit(TripThink(), max_attempts=10)
    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("worker")

await AutonomousAgent().arun(llm=llm, user_input="Plan a trip; ask me if you need input.")
```

### Custom UI Integration via @human_channel

`@human_channel` is the only customization mechanism for HITL UI — there is no `human_input` template method. **It is a method decorator** on an `async` method of your `AmphibiousAutoma` subclass.

#### Single implicit handler

```python
from bridgic.amphibious import human_channel, HumanCall, RETURN

class WebAgent(AmphibiousAutoma[MyOTAContext, Context]):
    @human_channel("web")
    async def ask_web(self, prompt: str) -> str:
        return await websocket.send_and_receive(prompt)

    async def on_workflow(self, ota_context, context=None):
        # Only one handler registered → HumanCall(channel=None) routes to ask_web.
        feedback = yield HumanCall(prompt="Confirm deploy?")
        yield RETURN(feedback)
```

You can also use the bare form `@human_channel` (no parens) — the channel name then defaults to the method name.

#### Multiple named handlers

With 2+ handlers registered, address each by name. Workflow uses `HumanCall(channel="name", ...)`; agent mode has the LLM pass `channel="name"` to the `request_human` tool.

```python
class HybridAgent(AmphibiousAutoma[TriageOTAContext, Context]):
    @human_channel("feishu")
    async def ask_feishu(self, prompt: str) -> str:
        return await send_to_feishu_and_wait(prompt)

    @human_channel("slack")
    async def ask_slack(self, prompt: str) -> str:
        return await send_to_slack_and_wait(prompt)

    triage = think_unit(TriageThink())  # prompt the model to pass channel="feishu"/"slack"

    async def on_workflow(self, ota_context, context=None):
        approval = yield HumanCall(channel="feishu", prompt="Approve deploy?")
        followup = yield HumanCall(channel="slack", prompt="Anything else?")

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("triage")  # the LLM picks channel via the request_human tool
```

(For the agent side, declare `request_human_tool` on `TriageOTAContext`. With 2+ channels registered, the `channel` argument is required.)

## Amphiflow Mode

When a class overrides both `on_agent` and `on_workflow`, `RunMode.AUTO` resolves to `AMPHIFLOW`: the workflow runs deterministically through the peer state-machine dispatcher, and on a step failure the framework runs a bounded `on_agent` recovery sub-run. You may also pass `mode=RunMode.AMPHIFLOW` explicitly.

```python
from bridgic.amphibious import RunMode, ActionCall, ThinkUnit, think_unit

class FormFiller(AmphibiousAutoma[FormOTAContext, Context]):
    fixer = think_unit(FixerThink(), max_attempts=5)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("fixer")

    async def on_workflow(self, ota_context, context=None):
        yield ActionCall("fill_field", field_name="username", value="john")
        yield ActionCall("fill_field", field_name="email", value="john@example.com")
        yield ActionCall("click_button", button_name="submit")

# On a failed ActionCall the framework runs a bounded on_agent recovery sub-run
# (fresh OTA episode against a goal describing the failed step). Its conclusion is
# shaped into the failed step's return type and asend()'d back to the workflow.
# Each successful ActionCall resets the consecutive-failure counter; reaching
# max_consecutive_fallbacks triggers full fallback (on_agent for the rest of the run).
agent = FormFiller(verbose=True)
result = await agent.arun(
    llm=llm,
    user_input="Fill and submit the form",
    mode=RunMode.AMPHIFLOW,
    max_consecutive_fallbacks=2,
)
```

## EnterAgent in Workflow

`EnterAgent` is the *explicit* mode-switch from deterministic workflow to LLM-driven agent. The dispatcher suspends the workflow, runs a fresh `on_agent` sub-run with `goal` as the new `user_input`, and resumes when the agent generator exhausts.

```python
from bridgic.amphibious import EnterAgent, ActionCall, ThinkUnit, think_unit

class PriceComparer(AmphibiousAutoma[PriceOTAContext, Context]):
    analyst = think_unit(AnalystThink(), max_attempts=3)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("analyst")

    async def on_workflow(self, ota_context, context=None):
        yield ActionCall("search_price", platform="Amazon", product="laptop")
        yield ActionCall("search_price", platform="eBay", product="laptop")

        # Delegate open-ended analysis to the agent, scoped to a sub-goal.
        yield EnterAgent(goal="Analyze prices and decide if we need more platforms.")

        # When on_agent exhausts, control returns here.
        yield ActionCall("publish_decision")
```

`EnterAgent` accepts only `goal=` — it controls *what sub-task the agent gets*, not *how it thinks* (no `worker=` / `max_attempts=` / `tools=` / `skills=`). The sub-run carries the OTA context class's declared tools and a fresh OTA context; the big-loop `Context` is shared.

## LLMCall in Workflow

`LLMCall` lets `on_workflow` invoke the agent's LLM directly through one of three protocols, without wrapping the call in a `CognitiveWorker`.

```python
from bridgic.amphibious import LLMCall, RETURN
from bridgic.core.model.protocols import PydanticModel
from pydantic import BaseModel

class Outline(BaseModel):
    sections: list[str]

class OutlineWriter(AmphibiousAutoma[MyOTAContext, Context]):
    async def on_workflow(self, ota_context, context=None):
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

`LLMCall` is **not** allowed inside `on_agent` — direct LLM calls belong in `on_workflow`, a hook, or inside a `CognitiveWorker`'s `thinking()` method.

## RETURN — explicit return values

PEP 525 forbids `return value` inside async generators. `RETURN(value)` is the framework's workaround:

```python
from bridgic.amphibious import RETURN, ThinkUnit, think_unit

class Summarizer(AmphibiousAutoma[MyOTAContext, Context]):
    summarizer = think_unit(SummarizerThink(), max_attempts=3)

    async def on_agent(self, ota_context, context=None):
        answer = yield ThinkUnit("summarizer")   # finishing think's step_content
        yield RETURN(f"FINAL: {answer}")
```

When yielded from a top-level `on_agent` / `on_workflow`, `RETURN(value)` writes `str(value)` to `self._final_answer` and closes the generator. Anything yielded after a `RETURN` is unreachable. Allowed in any scope.

## Custom Worker

A `CognitiveWorker` subclass owns one template method: `thinking(self, ota_context, context=None)`. It assembles a prompt from the two contexts, calls `self._llm`, and returns that call's natural result — the framework adapts it into a decision. `thinking()` has no default; every subclass implements it.

```python
from bridgic.amphibious import CognitiveWorker, think_unit, ThinkUnit
from bridgic.core.model.types import Message, Role

class DestinationAnalyzer(CognitiveWorker):
    async def observation(self, ota_context, context=None):
        # Worker-level observation: returned value becomes this round's obs_result.
        return "Tip: visit attractions early morning to avoid crowds."

    async def thinking(self, ota_context, context=None):
        messages = [
            Message.from_text(
                "Analyze the destination and suggest a day-by-day plan.\n\n"
                + ota_context.summary(),
                role=Role.USER,
            ),
        ]
        return await self._llm.aselect_tool(
            messages=messages,
            tools=[t.to_tool() for t in ota_context.tools],
        )

class TravelPlanner(AmphibiousAutoma[TravelOTAContext, Context]):
    analyzer = think_unit(DestinationAnalyzer(), max_attempts=3)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("analyzer")
```

To use a model without native function-calling, call `self._llm.achat(...)` in `thinking()` and return the `Response` (content-only finish), or parse tool calls yourself and return a `(tool_calls, content)` pair where each call is an object with `.name` / `.arguments` or a `{"name": ..., "arguments": {...}}` dict.

## Think Agent (External Agent Delegation)

`think_agent` wraps an `AgentWorker` — the external-agent peer of `think_unit`. It delegates a sub-goal to an out-of-process coding-agent CLI — `claude code` (`ClaudeCodeAgent`) or OpenAI `codex` (`CodexAgent`). Project tools reach the external agent through an in-process MCP bridge, so its tool calls still flow through the parent's hooks and trace.

### Default — claude code as the think agent

```python
from bridgic.amphibious import (
    AmphibiousAutoma, OTAContext, Context, AgentWorker, ClaudeCodeAgent,
    ThinkAgent, think_agent, RETURN,
)

async def record_finding(text: str) -> str:
    """Record one review finding."""
    return f"recorded: {text}"

class ReviewOTAContext(OTAContext):
    pass

ReviewOTAContext.tool(record_finding)   # project tool exposed to the external agent

class Reviewer(AmphibiousAutoma[ReviewOTAContext, Context]):
    reviewer = think_agent(
        AgentWorker(ClaudeCodeAgent(
            allowed_builtin_tools=["Read", "Grep"],   # claude's own tools
            completion_timeout=300.0,
        )),
        expose_tools=["record_finding"],              # which declared tools to bridge
    )

    async def on_agent(self, ota_context, context=None):
        summary = yield ThinkAgent("reviewer", goal="Review ./src and record findings.")
        yield RETURN(summary)

# A pure ThinkAgent flow needs no `llm` — the external agent is the brain.
result = await Reviewer().arun(user_input="Review the source tree.")
```

`CodexAgent` (OpenAI codex) is a drop-in alternative — `AgentWorker(CodexAgent())` in place of `AgentWorker(ClaudeCodeAgent(...))`; the rest is identical.

### Custom AgentWorker — reshape the prompt

Subclass `AgentWorker` and override `thinking()` to restructure the message handed to the external agent:

```python
class StrictReviewer(AgentWorker):
    async def thinking(self, ota_context, context=None) -> str:
        base = await super().thinking(ota_context, context)
        return base + "\n\nBe extremely thorough. Flag every TODO."

class Reviewer(AmphibiousAutoma[ReviewOTAContext, Context]):
    reviewer = think_agent(StrictReviewer(ClaudeCodeAgent()))

    async def on_agent(self, ota_context, context=None):
        yield ThinkAgent("reviewer", goal="Audit the module.")
```

CLI-level knobs (binary, sandbox / permission policy, completion timeout) live on the `BaseAgent`; cognitive customization (`thinking` / `observation` / `before_action` / `after_action`) lives on the `AgentWorker` — the same split as `BaseLlm` vs `CognitiveWorker`.

## Structured Output

There is no `output_schema` knob anymore — return a Pydantic model straight from `thinking()`. `_assemble_decision` serializes it into the decision's `step_content` (JSON); the `yield ThinkUnit(...)` result is that JSON string, which you parse.

```python
from pydantic import BaseModel, Field
from bridgic.amphibious import CognitiveWorker, think_unit, ThinkUnit, RETURN
from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import Message, Role

class PlanResult(BaseModel):
    phases: list[str] = Field(description="Execution phases")
    estimated_steps: int = Field(description="Total steps needed")

class PlannerThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        messages = [Message.from_text(
            "Create a step-by-step execution plan.\n\n" + ota_context.summary(),
            role=Role.USER,
        )]
        # astructured_output returns a PlanResult instance.
        return await self._llm.astructured_output(messages, PydanticModel(model=PlanResult))

class PlannerAgent(AmphibiousAutoma[PlanOTAContext, Context]):
    planner = think_unit(PlannerThink(), max_attempts=1)

    async def on_agent(self, ota_context, context=None):
        plan_json = yield ThinkUnit("planner")          # JSON string
        plan = PlanResult.model_validate_json(plan_json)
        yield RETURN(f"{plan.estimated_steps} steps across {len(plan.phases)} phases")
```

## Custom Context

Subclass `OTAContext` for per-run small-loop state, and `Context` for big-loop knowledge. The **OTA context** is the run's mutable working state — fold tool results onto it from `after_action`. The **big-loop `Context`** is rendered into the prompt via `summary()`; treat it as read-only during a run (the framework shares one instance across the run and any delegation, so mutating it mid-run is not a supported seam).

```python
from pydantic import Field
from bridgic.amphibious import (
    AmphibiousAutoma, OTAContext, Context, CognitiveWorker,
    ActionResult, ThinkUnit, think_unit,
)
from bridgic.core.model.types import Message, Role

# Big-loop knowledge — caller-supplied, read into the prompt by the worker.
class DocContext(Context):
    guidelines: str = ""

    def summary(self, fields):
        return f"Review guidelines:\n{fields['guidelines']}"

# Small-loop context — per-run working state + the tools this run carries.
class DocOTAContext(OTAContext):
    current_document: str = ""
    analyzed: list[str] = Field(default_factory=list)

DocOTAContext.tool(read_document_tool)

# The worker is what folds the big-loop knowledge into the prompt — nothing is
# auto-injected, so a custom Context only reaches the model if thinking() reads it.
class DocAnalyzerThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        knowledge = context.summary() if context is not None else ""
        prompt = f"{knowledge}\n\n{ota_context.summary()}" if knowledge else ota_context.summary()
        return await self._llm.aselect_tool(
            messages=[Message.from_text(prompt, role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )

class DocumentAnalyzer(AmphibiousAutoma[DocOTAContext, DocContext]):
    analyzer = think_unit(DocAnalyzerThink(), max_attempts=5)

    async def after_action(self, ota_context, context=None):
        # Payload-free: read the action result off the current round and fold
        # derived state onto the OTA context (the run's working state).
        action_result = ota_context.action_result
        if isinstance(action_result, ActionResult):
            for step in action_result.results:
                if step.success and step.tool_name == "read_document":
                    doc = step.tool_arguments.get("doc_name", "")
                    ota_context.current_document = doc
                    ota_context.analyzed.append(doc)
        if False:  # keep this an async generator even when nothing yields
            yield

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("analyzer")

# The caller seeds the big-loop knowledge; the run reads it via summary().
# await DocumentAnalyzer().arun(llm=llm, user_input="Analyze the docs",
#                               context=DocContext(guidelines="Flag any PII."))
```

## OTA Hooks

Hooks customize the observe-think-act cycle. Worker-level hooks (`CognitiveWorker.observation` / `before_action` / `after_action`) accept BOTH a coroutine form (`return _DELEGATE` / a value) and an async-generator form (yield `ActionCall` / `HumanCall` / `LLMCall`, then optionally `RETURN`). Agent-level hooks on `AmphibiousAutoma` are async generators. All hooks are **payload-free** — read the current round's state off `ota_context` (`obs_result` / `think_result` / `action_result`).

### observation — Inject Custom Perception

```python
class SecurityAgent(AmphibiousAutoma[SecOTAContext, Context]):
    auditor = think_unit(SecurityThink(), max_attempts=5)

    # Agent-level observation: shared across all workers. Yield RETURN(text)
    # to set this round's obs_result; exhausting without RETURN preserves it.
    async def observation(self, ota_context, context=None):
        yield RETURN("System: production-server-01, read-only audit mode.")

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("auditor")
```

#### Generator-form observation — fresh snapshot every cycle

When the observation needs a live tool call before each `ThinkUnit` (e.g. a browser snapshot), yield an `ActionCall` then `RETURN` its result:

```python
class BrowserAgent(AmphibiousAutoma[BrowserOTAContext, Context]):
    explorer = think_unit(ExplorerThink(), max_attempts=10)

    async def observation(self, ota_context, context=None):
        snapshot = yield ActionCall("bash", command="bridgic-browser snapshot")
        yield RETURN(snapshot[0].result if snapshot else None)

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("explorer")
```

The `yield ActionCall(...)` here is a raw tool execution — it does NOT re-enter the hook chain (hooks are not OTA participants). The same generator form applies to `before_action` (e.g. an audit-log call before each dispatch) and `after_action` (e.g. a refresh call after each step).

### before_action — Override the Decision

```python
class SafeAgent(AmphibiousAutoma[SafeOTAContext, Context]):
    auditor = think_unit(AuditThink(), max_attempts=5)

    async def before_action(self, ota_context, context=None):
        # Read the pending decision off the current round; drop blocked calls.
        decision = ota_context.think_result
        blocked = {"delete_file", "drop_table"}
        decision.tool_calls = [c for c in decision.tool_calls if c.tool not in blocked]
        yield RETURN(decision)   # override the decision before the act phase

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("auditor")
```

### after_action — React to the Result

```python
class TrackingAgent(AmphibiousAutoma[TrackOTAContext, Context]):
    worker = think_unit(WorkerThink(), max_attempts=5)

    async def after_action(self, ota_context, context=None):
        action_result = ota_context.action_result
        if isinstance(action_result, ActionResult):
            ok = sum(1 for r in action_result.results if r.success)
            # Fold a custom field onto the current round (OTARecord is extra="allow").
            ota_context._current_record().succeeded = ok
        if False:
            yield

    async def on_agent(self, ota_context, context=None):
        yield ThinkUnit("worker")
```

## Conditional Loops

`ThinkUnit("name", until=...)` loops the named think unit until the condition holds, capped by `max_attempts`.

```python
class IterativeAgent(AmphibiousAutoma[ResearchOTAContext, Context]):
    researcher = think_unit(ResearchThink(), max_attempts=10)

    async def on_agent(self, ota_context, context=None):
        # Loop until at least 3 rounds have been recorded (per-call overrides allowed).
        yield ThinkUnit(
            "researcher",
            until=lambda ota: len(ota.ota_record) >= 3,
            max_attempts=50,
        )
```

## Execution Tracing

```python
agent = MyAgent(verbose=True)
result = await agent.arun(
    llm=llm,
    user_input="...",
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
