# Orchestration — AmphibiousAutoma

## Table of Contents

- [Declaring an agent](#declaring-an-agent)
- [think_unit — declaring a reusable think step](#think_unit--declaring-a-reusable-think-step)
- [Three ways to execute think steps](#three-ways-to-execute-think-steps)
- [Workflow mode (on_workflow)](#workflow-mode-on_workflow)
- [RunMode — choosing how the agent runs](#runmode--choosing-how-the-agent-runs)
- [snapshot — scoped context override](#snapshot--scoped-context-override)
- [Agent-level hooks](#agent-level-hooks)
- [Agent properties](#agent-properties)
- [Running — arun()](#running--arun)

---

## What the orchestrator does

`AmphibiousAutoma` is the agent class. It wires workers, runs the OTA cycle, manages context, and handles mode switching. You subclass it to define your agent's behavior.

## Declaring an agent

```python
from bridgic.amphibious import AmphibiousAutoma, CognitiveContext, think_unit, CognitiveWorker, ErrorStrategy

class MyAgent(AmphibiousAutoma[CognitiveContext]):
    main_think = think_unit(
        CognitiveWorker.inline("Plan ONE immediate next step."),
        max_attempts=30,
        on_error=ErrorStrategy.RAISE,
    )

    async def on_agent(self, ctx: CognitiveContext):
        await self.main_think
```

**Always use a generic type parameter**: `AmphibiousAutoma[MyContext]`. Omitting it raises `TypeError` at class definition time — `__init_subclass__` validates that a `CognitiveContext` subclass is provided via the generic parameter.

## `think_unit(...)` — declaring a reusable think step

```python
think_unit(
    worker,              # CognitiveWorker template (cloned per execution for state isolation)
    until=None,          # Callable(ctx) -> bool — custom stop condition
    max_attempts=1,      # Max OTA cycles (1 = single-shot, 30 = long ReAct loop)
    tools=None,          # List[str] — tool name filter (None = all registered tools)
    skills=None,         # List[str] — skill name filter (None = all loaded skills)
    on_error=ErrorStrategy.RAISE,
    max_retries=0,       # Only used with ErrorStrategy.RETRY
)
```

`think_unit` is a **class-level descriptor** — it defines the step configuration but doesn't execute until awaited. The worker template is cloned per execution via `type(template)(...)` — this creates a fresh instance with clean runtime state (tokens, time, execution state) while copying configuration (policies, output_schema, verbose). LLM is injected at runtime.

**`max_attempts` guidance:**
- `1` — Single-shot decisions, planning steps, structured output
- `3–5` — Short targeted tasks ("find this one thing")
- `10–20` — Medium-complexity tasks
- `30+` — Open-ended ReAct agents that need many tool calls

**`until` — custom stop condition:**
```python
main = think_unit(
    worker,
    max_attempts=50,
    until=lambda ctx: len(ctx.findings) >= 5,  # Stop when we have 5 findings
)
```
The loop stops when `until(ctx)` returns True OR `finish=True` OR `max_attempts` reached.

## Three ways to execute think steps

### 1. `await self.think_unit_name` — use declared defaults

```python
async def on_agent(self, ctx):
    await self.main_think  # Uses max_attempts, tools, etc. from declaration
```

### 2. `.until(...)` — override stop condition and limits at runtime

```python
async def on_agent(self, ctx):
    await self.main_think.until(
        lambda ctx: some_condition(ctx),
        max_attempts=50,
        tools=["specific_tool"],  # Restrict tools for this execution
    )
```

### 3. `self._run(worker, ...)` — ad-hoc worker execution

For one-off workers not declared as class-level think_units:

```python
async def on_agent(self, ctx):
    worker = CognitiveWorker.inline("Do something specific", llm=self.llm)
    await self._run(worker, max_attempts=5, tools=["tool_a", "tool_b"])
```

**When to use which:**
- **think_unit** — reusable, declared at class level, configuration lives with the agent definition
- **_run** — ad-hoc, dynamic workers created at runtime based on context

## Workflow mode (`on_workflow`)

For deterministic sequences where you know exactly which tools to call in what order:

```python
from bridgic.amphibious import step, AgentFallback

class MyAgent(AmphibiousAutoma[MyContext]):
    async def on_workflow(self, ctx):
        yield step("navigate_to", url="http://example.com")
        result = yield step("click_element", ref="42")
        # result is List[ToolResult] — the tool execution results

        # Hand off complex sub-tasks to agent mode
        yield AgentFallback(goal="Handle login if CAPTCHA appears", max_attempts=10)
```

**`step()` function:**
```python
step(tool_name, *, description="", worker=None, **tool_args) -> WorkflowStep
```
- `tool_name` — name of a registered tool
- `**tool_args` — arguments passed to the tool
- `worker` — optional custom worker for fallback (defaults to generic fixer)

**`AgentFallback`** — switches to agent mode for a sub-task:
```python
AgentFallback(
    goal="...",           # Goal for the agent sub-task
    max_attempts=10,      # Max OTA cycles
    worker=None,          # Optional custom worker (default: generic fixer)
    tools=None,           # Optional tool filter
    skills=None,          # Optional skill filter
)
```

**When a workflow step fails:** If `will_fallback=True`, the framework automatically falls back to agent mode for that specific step. The fallback agent gets the original step's intent as its goal.

### Dynamic workflows with Python control flow

`on_workflow` is a Python async generator — you can use **any Python control flow** (loops, conditionals, variable assignment) alongside `yield step(...)`:

```python
async def on_workflow(self, ctx):
    # Sequential setup
    yield step("navigate_to", url="https://app.example.com/list")
    yield step("click_element", ref="search_btn")

    # Wait, then read dynamic data from the latest observation
    yield step("wait_for", time_seconds="3")
    items = extract_items(ctx.observation)  # Python helper, not LLM

    # Loop over dynamically discovered items
    for ref, item_id in items:
        if item_id in ctx.tracker._seen_ids:
            continue  # Skip already processed

        yield step("click_element", ref=ref)
        yield step("wait_for", time_seconds="2")

        # Compute dynamic step arguments from observation
        detail_page = find_newest_page(ctx.observation)

        # Delegate complex extraction to agent
        yield AgentFallback(
            goal=f"Extract data from item {item_id} and save it.",
            tools=["save_record"],
            max_attempts=3,
        )
```

**`ctx.observation` lifecycle in workflows**: Each `yield step(...)` triggers a fresh observation cycle — `observation()` is called and `ctx.observation` is updated **before** the step's tools execute. After the step completes, `ctx.observation` still holds the pre-execution observation (it is NOT refreshed after tool execution). It is refreshed again at the **next** `yield step(...)`. Safe to read between yields.

**In agent mode (OTA loops)**: `ctx.observation` is refreshed at the start of **every** OTA cycle (`_run_once`), before the Think phase. Tools run concurrently via `asyncio.gather()` during Act, but observation is never updated during or after tool execution within the same cycle.

**When to use `step()` vs `AgentFallback`:**

| Use | When |
|-----|------|
| `step("tool", ...)` | Arguments are known (deterministic, computed from observation) |
| `AgentFallback(goal="...")` | Task needs LLM reasoning (page understanding, content extraction) |

## `RunMode` — choosing how the agent runs

```python
await agent.arun(context=ctx, mode=RunMode.AUTO)
```

| Mode | What happens | Use when |
|------|-------------|----------|
| `AUTO` (default) | Amphibious if `on_workflow` is overridden, otherwise agent | Most cases — let the framework decide |
| `AGENT` | Only runs `on_agent()`, ignores `on_workflow()` | Force LLM-driven behavior |
| `WORKFLOW` | Only runs `on_workflow()`, **no fallback** on failure | Strict deterministic execution |
| `AMPHIBIOUS` | Runs `on_workflow()` with automatic agent fallback | Deterministic path with safety net |

**Amphibious-specific parameters:**
```python
await agent.arun(
    context=ctx,
    mode=RunMode.AMPHIBIOUS,
    will_fallback=True,              # Allow fallback to agent on step failure
    max_consecutive_fallbacks=1,     # N consecutive failures → switch to full on_agent()
)
```

## `snapshot()` — scoped context override

**Problem it solves**: Multi-phase agents need different goals, tool subsets, or state for each phase, but you don't want to manually save/restore fields.

```python
async def on_agent(self, ctx):
    # Phase 1: Research
    async with self.snapshot(goal="Gather information about the topic"):
        await self.researcher
    # goal is automatically restored here

    # Phase 2: Synthesis
    async with self.snapshot(goal="Synthesize findings into a report"):
        await self.synthesizer
```

**What snapshot does:**
1. **Enter**: saves current field values, applies overrides
2. **Inside**: agent runs with overridden fields
3. **Exit**: restores original values (even if an exception occurs)

**Parameters:**
```python
self.snapshot(
    goal="...",           # Override any context field by name
    keep_revealed={...},  # Control LayeredExposure revealed state
    **fields,             # Any other context fields to override
)
```

**`keep_revealed`**: Controls what disclosed details survive the snapshot boundary.
- `None` (default) — clears all revealed details on enter (fresh start)
- `{"skills": [0, 1]}` — keeps revealed details for skill indices 0 and 1

## Agent-level hooks

These hooks apply to **all** workers unless a worker overrides them (returns something other than `_DELEGATE`).

```python
class MyAgent(AmphibiousAutoma[MyContext]):
    async def observation(self, ctx) -> Optional[str]:
        """Shared observation for all workers. Injected into every OTA cycle.
        Return a string with current state info, or None for no observation.
        Use for: browser state, sensor data, environment info."""
        return f"Current page: {ctx.current_url}"

    async def before_action(self, decision_result, ctx) -> Any:
        """Intercept tool calls before execution. Can filter or modify.
        decision_result is List[Tuple[ToolCall, ToolSpec]] for tool calls,
        or a BaseModel for output_schema.
        Use for: safety filters, logging, argument modification."""
        return decision_result

    async def after_action(self, step_result, ctx) -> None:
        """Post-process after tool execution. Update context fields.
        step_result is a Step with .content (reasoning) and .result (ActionResult).
        Use for: extracting findings, updating counters, state transitions."""
        pass

    async def action_tool_call(self, tool_list, context) -> ActionResult:
        """Override the actual tool execution mechanism.
        Default: concurrent execution via asyncio.gather().
        Use for: sequential execution, rate limiting, sandboxing, logging."""
        ...

    async def action_custom_output(self, decision_result, context) -> Any:
        """Handle output_schema results before they're stored in history.
        Default: returns unchanged.
        Use for: validation, transformation, side effects."""
        return decision_result
```

## Agent properties

```python
agent.llm            # The LLM instance
agent.context        # Current context (available during/after arun)
agent.final_answer   # Last step_content from the finishing step
agent.spend_tokens   # Total tokens used in last arun()
agent.spend_time     # Wall-clock seconds for last arun()

agent.set_final_answer("...")  # Explicitly override the auto-captured answer
```

## Running — `arun()`

Two calling conventions:

```python
# Option 1: Pre-built context (full control)
ctx = CognitiveContext(goal="Plan a trip")
ctx.tools.add(FunctionToolSpec.from_raw(search))
result = await agent.arun(context=ctx)

# Option 2: Shorthand (auto-creates CognitiveContext)
result = await agent.arun(
    goal="Plan a trip",
    tools=[FunctionToolSpec.from_raw(search)],
    skills=[Skill(name="x", description="y", content="z")],
)
```

**Returns `str`** — the context summary after execution. Use `agent.final_answer` for the LLM's finishing statement.

**With tracing:**
```python
result = await agent.arun(context=ctx, trace_running=True)
agent._agent_trace.build()           # Get trace data
agent._agent_trace.save("trace.json") # Save to file
```

For trace details, see [tracing.md](tracing.md). For error strategies, see [error-handling.md](error-handling.md). For worker hooks and policies, see [cognitive-worker.md](cognitive-worker.md).
