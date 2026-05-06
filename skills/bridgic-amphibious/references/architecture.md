# Bridgic Amphibious Architecture Reference

## Table of Contents
- [Four-Layer Architecture](#four-layer-architecture)
- [Observe-Think-Act (OTC) Cycle](#observe-think-act-otc-cycle)
- [Execution Modes (RunMode)](#execution-modes-runmode)
- [Data Exposure System](#data-exposure-system)
- [Cognitive Policies](#cognitive-policies)
- [Memory Architecture (CognitiveHistory)](#memory-architecture-cognitivehistory)
- [Think Unit Descriptor Pattern](#think-unit-descriptor-pattern)
- [Phase Annotation (snapshot)](#phase-annotation-snapshot)
- [Workflow Fallback Mechanism](#workflow-fallback-mechanism)
- [Built-in Tools Subsystem](#built-in-tools-subsystem)
- [Human-in-the-Loop](#human-in-the-loop)

---

## Four-Layer Architecture

```
Layer 4: AmphibiousAutoma (Orchestration)
  ├─ on_agent()    → LLM-driven thinking
  ├─ on_workflow() → Deterministic steps
  └─ _run_once()   → Single OTC cycle

Layer 3: CognitiveWorker (Think Unit)
  └─ thinking()    → LLM decision logic
  └─ Policies      → acquiring, rehearsal, reflection

Layer 2: CognitiveContext (State Management)
  ├─ goal, tools, skills, history
  └─ Exposure system → data visibility control

Layer 1: Exposure (Data Abstraction)
  ├─ LayeredExposure → progressive disclosure
  └─ EntireExposure  → full exposure
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
| `AGENT` | LLM (`on_agent`) | Open-ended, adaptive tasks | N/A |
| `WORKFLOW` | Code (`on_workflow`) | Known, repeatable processes | N/A |
| `AMPHIFLOW` | Workflow + LLM fallback | Robust hybrid execution | Automatic |
| `AUTO` (default) | Auto-detect from overridden methods | Most subclasses | Inherits from resolved mode |

- `AUTO` resolution rules:
  - only `on_agent` overridden → `AGENT`
  - only `on_workflow` overridden → `WORKFLOW`
  - both overridden → `AMPHIFLOW`
  - neither overridden → `RuntimeError` at run time
- LLM requirement: `AGENT` and `AMPHIFLOW` require an LLM at `arun()` time; pure `WORKFLOW` does not.

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
2. On instance access (`self.main_think`), returns `_BoundThinkUnit`
3. `_BoundThinkUnit` is awaitable (`await self.main_think`)
4. Supports `.until()` for conditional loops
5. Fresh worker clone per execution (state isolation)

## Phase Annotation (snapshot)

`self.snapshot()` creates scoped context overrides:

```python
async with self.snapshot(goal="Sub-task A"):
    # Original fields saved, overrides applied
    # LayeredExposure._revealed cleared
    await self.worker  # LLM sees goal = "Sub-task A"
# Original fields + revealed state restored
```

- Provides sub-goal scoping for focused thinking
- Exception-safe via async context manager

## Workflow Fallback Mechanism

Two distinct failure sources are handled in AMPHIFLOW mode:

**ActionCall tool failure** (a yielded tool raises during execution):

1. Step fails → check `consecutive_failures < max_consecutive_fallbacks`
2. Within limit: agent fixes the specific step (scoped goal via `snapshot`); generator resumes
3. Limit exceeded: abandon workflow → call `on_agent()` for full agent mode

**Generator-internal exception** (helper / inline logic between yields raises):

- The generator is unrecoverable after a raise — `asend()` cannot resume it — so step-level fallback is impossible. The framework jumps directly to full fallback: `on_agent(ctx)` takes over the remaining task.
- Pure WORKFLOW mode (`will_fallback=False`): the original exception is re-raised — no fallback.
- AMPHIFLOW with `on_agent` overridden: hand off to `on_agent(ctx)`.
- AMPHIFLOW forced via `mode=` without an `on_agent` override: a `RuntimeError` is raised, tagged with the failing step index.

`AgentCall` yield is orthogonal to fallback — it explicitly delegates a sub-task to agent mode (with a clean context snapshot) regardless of failure state.

## Built-in Tools Subsystem

`AmphibiousAutoma.arun()` injects a fixed roster of built-in tools into `context.tools` so every agent has a baseline capability surface — shell, filesystem, search, human input — without any per-project wiring. The roster lives in `bridgic.amphibious.builtin_tools.ALL_BUILTIN_TOOLS`; adding a new built-in only requires appending its `FunctionToolSpec` to that tuple.

### Injection resolution

```
arun(builtin_tools=...)        ← runtime kwarg (highest priority)
    └─ if None → class.builtin_tools (frozenset or None)
        └─ if None → inject every entry of ALL_BUILTIN_TOOLS
```

A non-`None` resolution must reference only valid tool names; unknown entries raise `ValueError` at `arun()` entry, surfacing typos before the LLM ever sees a missing tool. The resulting set is intersected with already-present `context.tools` by `tool_name` — user-supplied tools win, so a built-in whose name collides is silently skipped (dedup behaviour).

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

Three entry points for requesting human input, all built on `request_feedback_async`:

| Entry Point | Context | Mechanism |
|-------------|---------|-----------|
| `await self.request_human(prompt)` | `on_agent()` — between think units | Direct async call |
| `yield HumanCall(prompt=...)` | `on_workflow()` — pause generator | Framework calls `request_human`, sends response via `asend()` |
| `request_human` tool (auto-injected) | LLM-driven — any mode | Built-in tool injected into `context.tools` during `arun()`, resolved via `ContextVar` |

**Customization**: Override `human_input(data)` template method to replace default stdin with your UI (WebSocket, HTTP callback, Slack bot, etc.).

**Auto-injection**: `request_human` is one of the seven tools injected by `arun()` (see [Built-in Tools Subsystem](#built-in-tools-subsystem) above). Auto-injection is what gives `on_agent`, workflow step-level fallback, and full agent fallback the same autonomous HITL capability as `HumanCall` provides to `on_workflow`. Users can still pass `request_human_tool` explicitly — it is a no-op thanks to the dedupe.

**Concurrency**: `request_human` uses `contextvars.ContextVar` for late-binding. Each `asyncio.Task` (each `arun()`) gets its own isolated binding — concurrent agents sharing the same tool object never interfere.
