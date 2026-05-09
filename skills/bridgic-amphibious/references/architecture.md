# Bridgic Amphibious Architecture Reference

## Table of Contents
- [Four-Layer Architecture](#four-layer-architecture)
- [Observe-Think-Act (OTC) Cycle](#observe-think-act-otc-cycle)
- [Execution Modes (RunMode)](#execution-modes-runmode)
- [Yield Primitive Categories](#yield-primitive-categories)
- [Peer State-Machine Dispatcher](#peer-state-machine-dispatcher)
- [Workflow Fallback Mechanism](#workflow-fallback-mechanism)
- [Data Exposure System](#data-exposure-system)
- [Cognitive Policies](#cognitive-policies)
- [Memory Architecture (CognitiveHistory)](#memory-architecture-cognitivehistory)
- [Think Unit Descriptor Pattern](#think-unit-descriptor-pattern)
- [Phase Annotation (snapshot)](#phase-annotation-snapshot)
- [Built-in Tools Subsystem](#built-in-tools-subsystem)
- [Human-in-the-Loop](#human-in-the-loop)

---

## Four-Layer Architecture

```
Layer 4: AmphibiousAutoma (Orchestration)
  ├─ on_agent()         → LLM-driven async generator yielding ThinkUnit / RETURN
  ├─ on_workflow()      → Deterministic async generator yielding atomic Calls /
  │                        EnterAgent / RETURN
  ├─ _drive_amphiflow() → Peer state machine driving on_workflow + on_agent
  ├─ _invoke_template() → Single-generator driver (AGENT / WORKFLOW / hooks)
  └─ _dispatch_call()   → Per-yield handler with scope validation

Layer 3: CognitiveWorker (Think Unit)
  └─ thinking()         → LLM decision logic
  └─ Policies           → acquiring, rehearsal, reflection

Layer 2: CognitiveContext (State Management)
  ├─ goal, tools, skills, history
  └─ Exposure system    → data visibility control

Layer 1: Exposure (Data Abstraction)
  ├─ LayeredExposure    → progressive disclosure
  └─ EntireExposure     → full exposure
```

## Observe-Think-Act (OTC) Cycle

Each think unit execution follows:

1. **Observe**: Gather current state
   - Worker `observation(context)` called first
   - If returns `_DELEGATE`, falls through to agent `observation(context)`
   - Result stored in `context.observation`

2. **Think**: LLM decides next action
   - `CognitiveWorker._thinking(context)` runs LLM
   - Multi-round loop if cognitive policies fire
   - Returns decision with `step_content`, `finish`, `output`

3. **Act**: Execute tools or produce structured output
   - `before_action()` hooks (worker → agent delegation)
   - Route to `action_tool_call()` for tool calls
   - Route to `action_custom_output()` for structured output (output_schema)
   - `after_action()` hooks (worker → agent delegation)
   - Record result in `CognitiveHistory`

## Execution Modes (RunMode)

| Mode | Driver | Best For | Fallback |
|------|--------|----------|----------|
| `AGENT` | `_invoke_template(on_agent)` | Open-ended, adaptive tasks | N/A |
| `WORKFLOW` | `_invoke_template(on_workflow)` | Known, repeatable processes | N/A |
| `AMPHIFLOW` | `_drive_amphiflow` (state machine) | Robust hybrid execution | Step-level + full |
| `AUTO` (default) | Auto-detect from overridden methods | Most subclasses | Inherits from resolved mode |

`AUTO` resolution rules:
- only `on_agent` overridden → `AGENT`
- only `on_workflow` overridden (as **async generator**) → `WORKFLOW`
- both overridden → `AMPHIFLOW`
- neither overridden → `RuntimeError` at run time

A coroutine-form `on_workflow` (`async def on_workflow(self, ctx): pass` — produces a coroutine, not an async generator) is treated as a stub under `AUTO`. This shields users from AI-generated stub `on_workflow` methods that would otherwise force `AMPHIFLOW`. To run a real coroutine workflow, force `mode=RunMode.WORKFLOW` or `RunMode.AMPHIFLOW`; the dispatcher handles both forms in those paths (`_drive_amphiflow` short-circuits to `await workflow_obj` when the workflow is a coroutine).

LLM requirement: `AGENT` and `AMPHIFLOW` require an LLM at `arun()` time; pure `WORKFLOW` does not.

## Yield Primitive Categories

The dispatcher's `_dispatch_call` recognizes six yield types in three categories. Scope validation happens at dispatch time — mismatches raise `RuntimeError`.

| Category | Primitive | Allowed scopes |
|----------|-----------|----------------|
| **Atomic Call** (operations on the world) | `ActionCall` (deterministic single-tool) | `workflow`, `hook` |
|  | `HumanCall` (HITL via `@human_channel`) | `workflow`, `hook` |
|  | `LLMCall` (chat / structure_output / tool_selector) | `workflow`, `hook` |
| **Mode-switch** (state-machine transition) | `EnterAgent` (suspend workflow → run on_agent) | `workflow` only |
| **Cognitive composition** (inside on_agent) | `ThinkUnit` (named class-level descriptor) | `agent` only |
| **Control flow** | `RETURN` (PEP 525 return-value workaround) | any |

The asymmetry — atomic Calls forbidden in `agent` scope — is intentional: `on_agent` is reserved for orchestrating cognitive steps via `ThinkUnit`. Tool / human / LLM operations the agent needs to perform happen *inside* a `ThinkUnit` (the worker's tool-selection phase), not by yielding from `on_agent` directly. There's no "switch back to workflow" yield; agent-generator exhaustion is the implicit signal.

## Peer State-Machine Dispatcher

`AMPHIFLOW` is driven by `_drive_amphiflow`, a single while-loop holding two generator slots:

```
┌──────────────────────────────────────────────────────────┐
│  _drive_amphiflow                                         │
│   ├─ workflow_gen   (created from on_workflow)            │
│   ├─ agent_gen      (created on EnterAgent / fallback)    │
│   ├─ snapshot stack (AsyncExitStack across iterations)    │
│   └─ counter        (consecutive_failures)                │
│                                                            │
│   while True:                                              │
│     active = agent_gen if agent_gen else workflow_gen      │
│     item = active.asend(prev_value)  # or aclose on RETURN │
│     ├─ RETURN(v)     → close active, propagate v           │
│     ├─ EnterAgent    → snapshot ctx, create agent_gen      │
│     ├─ atomic Call   → _dispatch_call (in active scope)    │
│     │                  → on raise, run step-level fallback │
│     └─ ThinkUnit     → _dispatch_call (agent scope only)   │
│                                                            │
│   StopAsyncIteration on agent_gen → switch back to workflow│
│   StopAsyncIteration on workflow_gen → finish              │
└──────────────────────────────────────────────────────────┘
```

Key invariants:

- **Single while-loop** — no recursion, no nested driver. EnterAgent simply changes which slot the loop reads from on the next iteration.
- **Implicit switch-back** — agent-generator exhaustion (`StopAsyncIteration`) signals "I'm done, return control to the workflow." The framework `asend()`s the agent's last value (or the fallback slot value, see below) into the suspended workflow generator.
- **AsyncExitStack** for snapshot lifetimes — each `EnterAgent` pushes its `self.snapshot(**kwargs)` onto a stack; the snapshot stays open across multiple loop iterations (one per yield within that `on_agent` body) and is unwound only when the agent generator exhausts.
- **Cleanup ordering** — on early exit (RETURN, exception): close generators first (their `finally` blocks see snapshotted ctx), then unwind the snapshot stack, then close the workflow generator.
- **Coroutine short-circuit** — if `self.on_workflow(ctx)` returns a coroutine instead of an async generator (`inspect.isasyncgen` is False), `_drive_amphiflow` simply `await`s it and returns. The state machine starts only for true async-generator workflows.

`AGENT` and `WORKFLOW` modes use the simpler `_invoke_template` single-generator driver — there's nothing to alternate between. `EnterAgent` yielded inside forced-`WORKFLOW` mode falls through `_dispatch_call`'s recursive branch (it calls `_invoke_template(self.on_agent(ctx), scope='agent')` inline), so the scope rules still hold.

## Workflow Fallback Mechanism

`AMPHIFLOW` defends against two distinct failure sources.

### Generator-internal exception (helper / inline logic between yields raises)

The generator is unrecoverable after a raise — `asend()` cannot resume it — so step-level fallback is impossible. The framework jumps directly to **full fallback**: `on_agent(ctx)` takes over the remaining task.

- Pure WORKFLOW mode (`will_fallback=False`): the original exception is re-raised — no fallback.
- AMPHIFLOW with `on_agent` overridden: hand off to `on_agent(ctx)`.
- AMPHIFLOW forced via `mode=` without an `on_agent` override: a `RuntimeError` is raised.

`workflow_gen.aclose()` is wrapped in `try / except` during full-fallback unwinding — if the user's `finally` block raises, the fallback agent still runs.

### Atomic-Call failure (an `ActionCall` / `HumanCall` / `LLMCall` raises)

The dispatcher catches the exception, increments the consecutive-failures counter, and decides between **step-level recovery** and **full fallback** based on the counter alone:

```
counter < max_consecutive_fallbacks  → step-level recovery (counter++)
counter >= max_consecutive_fallbacks → full fallback (counter == limit)
```

Each successful atomic Call resets the counter to 0. `EnterAgent` does **not** reset the counter — it's a mode switch, not a successful atomic Call.

#### Step-level recovery: slot + injected `resolve_step_fallback` tool

When step-level recovery fires, the framework:

1. **Allocates a `_FallbackSlot`** with a type-appropriate default value:

   | Failed yield | Slot default |
   |--------------|--------------|
   | `ActionCall` | `[]` (empty `List[ToolResult]`) |
   | `HumanCall` | `""` |
   | `LLMCall.chat` | `""` |
   | `LLMCall.structure_output` | `None` |
   | `LLMCall.tool_selector` | `([], None)` |

2. **Injects a `resolve_step_fallback` tool** into `ctx.tools` for the duration of this fallback `on_agent` run only. The tool's signature is shaped to match the failed yield (e.g. `resolve_step_fallback(result: Any) -> str` for `ActionCall`, `(response: str) -> str` for `HumanCall`). It closes over the slot, so calling it writes the agent's result into the slot.

3. **Runs `on_agent(ctx)`** under a snapshot scoped to the failed yield's goal. The agent's LLM either calls `resolve_step_fallback(...)` (writing a value to the slot) or doesn't (slot keeps its default).

4. **Removes the injected tool** from `ctx.tools` after the agent generator exhausts.

5. **Resumes the workflow generator** by `asend()`-ing the slot's current value.

**Counter-only escalation** (not "did the agent call the tool"): the framework does not extract values heuristically from the agent's behaviour. The slot value going back to the workflow is *just a value* — its presence/absence does not signal escalation. Escalation is decided exclusively by the consecutive-failures counter against `max_consecutive_fallbacks`.

This design preserves a clean separation:
- The yield primitive set stays small (no new "fallback" primitive).
- `RETURN` keeps its narrow PEP-525 workaround semantics (not extended for value handoff).
- "Agent gave up" (no `resolve_step_fallback` call) is not conflated with "void Call needs no value" (e.g. an `ActionCall` whose tool is purely side-effecting — the empty default is genuinely the right answer).

### EnterAgent vs fallback

`EnterAgent` yield is orthogonal to fallback — it's the user's *explicit* mode-switch signal, not a failure recovery. `EnterAgent` runs without injecting `resolve_step_fallback` (no failed Call to recover from), and its agent run is bounded only by the agent generator's own logic.

## Data Exposure System

Controls how context data is visible to the LLM.

### EntireExposure[T]

All data visible at once. Used for tools.

- Methods: `summary()` only
- Implementation: `CognitiveTools`

### LayeredExposure[T]

Progressive disclosure with details on demand.

- Methods: `summary()` + `get_details(index)` + `reveal(index)`
- Caching: `_revealed` dict stores cached details
- Reset: `reset_revealed()` clears cache (at phase boundaries)
- Implementations: `CognitiveSkills`, `CognitiveHistory`

### Context Field Detection

`Context` base class auto-detects `Exposure`-typed fields and classifies them as `layered` or `entire`. Custom fields that are plain types (str, dict, etc.) appear directly in the summary.

- Hide a field from summary: `json_schema_extra={"display": False}`
- Enable LLM propagation to an Exposure field: `json_schema_extra={"use_llm": True}`

## Cognitive Policies

Multi-round thinking within a single OTC cycle. Each policy fires **at most once**, then closes.

### Acquiring (built-in, always active when no output_schema)

LLM requests details from `LayeredExposure` fields (skills, cognitive_history).

```
LLM fills: details: [{field: "skills", index: 0}]
→ Framework reveals full content
→ Re-think with revealed data
```

### Rehearsal (opt-in: `enable_rehearsal=True`)

LLM mentally simulates planned action.

```
LLM fills: rehearsal: "If I call search_tool, I expect..."
→ Prediction injected as context
→ Re-think with simulation
```

### Reflection (opt-in: `enable_reflection=True`)

LLM assesses information quality.

```
LLM fills: reflection: "The data is inconsistent because..."
→ Assessment injected as context
→ Re-think with assessment
```

Policy execution order: **Acquiring → Rehearsal → Reflection**. After all active policies fire, LLM must commit to a final action.

## Memory Architecture (CognitiveHistory)

Four-tier layered memory with automatic compression:

```
New step added
    │
    v
[Working Memory]    ← latest N steps, full details shown
    │
    v (overflow)
[Short-term Memory] ← next M steps, summaries only, queryable via Acquiring
    │
    v (overflow, triggers compression)
[Long-term Pending] ← brief summaries, awaiting batch compression
    │
    v (compress_threshold reached + LLM available)
[Long-term Compressed] ← LLM-compressed concise paragraph
```

Default parameters:
- `working_memory_size=5`
- `short_term_size=20`
- `compress_threshold=10`

## Think Unit Descriptor Pattern

Think units use Python descriptors for class-level declaration:

1. `think_unit()` factory returns `ThinkUnitDescriptor`
2. On instance access (`self.main_think`), returns `_BoundThinkUnit` (used internally — direct `await self.main_think` still works)
3. Canonical orchestration is `yield ThinkUnit("main_think")` from inside `on_agent` — this routes through the dispatcher, supports per-yield overrides (`until=`, `max_attempts=`, `tools=`, `skills=`), and returns the worker's typed output
4. Fresh worker clone per execution (state isolation)

## Phase Annotation (snapshot)

`self.snapshot(**fields)` creates scoped context overrides:

```python
async with self.snapshot(goal="Sub-task A"):
    # Original fields saved, overrides applied
    # LayeredExposure._revealed cleared
    yield ThinkUnit("worker")  # LLM sees goal = "Sub-task A"
# Original fields + revealed state restored
```

- Provides sub-goal scoping for focused thinking
- Exception-safe via async context manager
- Used internally by `EnterAgent` (one snapshot per yield, lifetime managed by `AsyncExitStack`) and by step-level fallback (one snapshot scoped to the failed call's goal)

## Built-in Tools Subsystem

`AmphibiousAutoma.arun()` injects a fixed roster of built-in tools into `context.tools` so every agent has a baseline capability surface — shell, filesystem, search, human input — without any per-project wiring. The roster lives in `bridgic.amphibious.builtin_tools.ALL_BUILTIN_TOOLS`; adding a new built-in only requires appending its `FunctionToolSpec` to that tuple.

### Injection resolution

```
arun(builtin_tools=...)        ← runtime kwarg (highest priority)
    └─ if None → class.builtin_tools (frozenset or None)
        └─ if None → inject every entry of ALL_BUILTIN_TOOLS
```

A non-`None` resolution must reference only valid tool names; unknown entries raise `ValueError` at `arun()` entry, surfacing typos before the LLM ever sees a missing tool. The resulting set is intersected with already-present `context.tools` by `tool_name` — user-supplied tools win, so a built-in whose name collides is silently skipped (dedup behaviour).

`resolve_step_fallback` is *not* part of `ALL_BUILTIN_TOOLS`. It is allocated and injected only during step-level fallback and removed before the workflow resumes — see [Workflow Fallback Mechanism](#workflow-fallback-mechanism).

### Read-before-modify invariant

The filesystem-mutating built-ins (`write_file`, `edit_file`) require a prior `read_file` on the same path AND that the file has not changed externally since that read. Mechanism:

- `AmphibiousAutoma._read_tracker: Dict[str, float]` — a per-agent dict mapping absolute path → mtime at last read. Reset at every `arun()` entry, scoping the invariant to a single run.
- `read_file` records the file's mtime after a successful read (best-effort: a failed `os.stat` here is silently swallowed so it cannot mask the successful read).
- `write_file` (for existing files) and `edit_file` consult the tracker and raise `RuntimeError` if (a) the path was never read, or (b) the current mtime is newer than the recorded one.

The tools resolve the agent through the same `current_agent` ContextVar used by `request_human`, so the tracker is implicitly per-`asyncio.Task`: concurrent `arun()` calls from separate agents never share state.

### Tool exception path

Built-in tools raise on validation failures (`ValueError`, `FileNotFoundError`, `RuntimeError`, `TimeoutError`, …). They do not catch and wrap errors as `<error>...</error>` strings. The framework's per-tool exception handling — in `_action_tool_call._run_one` — captures every exception and produces:

```python
ActionStepResult(success=False, error=str(exc), tool_result=None)
```

In agent mode this becomes part of the next observation, letting the LLM see what went wrong and adapt. In workflow mode, `_run_workflow` aggregates failed `ActionStepResult`s into a `RuntimeError("Tool execution failed for: ... — ...")` and either falls back to `on_agent` (AMPHIFLOW within `max_consecutive_fallbacks`) or re-raises (pure WORKFLOW).

## Human-in-the-Loop

Two entry points for requesting human input — both share the same `@human_channel` registry:

| Entry Point | Context | Mechanism |
|-------------|---------|-----------|
| `yield HumanCall(prompt=, channel=)` | `on_workflow()` body, hooks (rejected in `on_agent`) | State-machine dispatcher routes through the `@human_channel` registry, `asend()`s the response back to the generator |
| `request_human` tool (auto-injected) | LLM-driven — called from inside any `ThinkUnit`, in any mode | Built-in tool injected into `context.tools` during `arun()`; resolves the running agent via `current_agent` ContextVar and routes through `_dispatch_human_channel` |

There is **no** code-level imperative API on `AmphibiousAutoma` (no `self.request_human(...)`, no `self.ask_human(...)`). The agent's `on_agent` body is reserved for orchestrating cognitive steps via `ThinkUnit`; HITL inside `on_agent` happens autonomously through the LLM calling the auto-injected tool.

**Channel resolution** (applies to both `HumanCall` dispatch and the auto-injected `request_human` tool, since both go through `_dispatch_human_channel`):
- `channel=None` + zero `@human_channel` handlers → built-in stdin handler.
- `channel=None` + one handler → that handler used implicitly.
- `channel=None` + 2+ handlers → `RuntimeError` requiring explicit channel.
- `channel="name"` → invoke that named handler.

The auto-injected `request_human` tool always passes `channel=None`, so it picks the implicit default. To route LLM-driven HITL to a specific channel, register exactly one `@human_channel` handler.

**Customization**: Override the `human_input(data)` template method to replace the default stdin fallback with your own UI integration (WebSocket, HTTP callback, Slack bot, etc.). Alternatively, register named `@human_channel` handlers and yield `HumanCall(channel="name", ...)` from `on_workflow`.

**Auto-injection**: `request_human` is one of the seven tools injected by `arun()` (see [Built-in Tools Subsystem](#built-in-tools-subsystem) above). Auto-injection is what gives `on_agent`, workflow step-level fallback, and full agent fallback the same autonomous HITL capability as `HumanCall` provides to `on_workflow`. Users can still pass `request_human_tool` explicitly — it is a no-op thanks to the dedupe.

**Concurrency**: `request_human` uses `contextvars.ContextVar` for late-binding. Each `asyncio.Task` (each `arun()`) gets its own isolated binding — concurrent agents sharing the same tool object never interfere.
