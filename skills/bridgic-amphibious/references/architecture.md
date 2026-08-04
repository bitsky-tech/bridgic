# Bridgic Amphibious Architecture Reference

## Table of Contents
- [Three-Layer Architecture](#three-layer-architecture)
- [The Two-Loop Context Model](#the-two-loop-context-model)
- [Observe-Think-Act (OTA) Cycle](#observe-think-act-ota-cycle)
- [Execution Modes (RunMode)](#execution-modes-runmode)
- [Yield Primitive Categories](#yield-primitive-categories)
- [Peer State-Machine Dispatcher](#peer-state-machine-dispatcher)
- [Workflow Fallback Mechanism](#workflow-fallback-mechanism)
- [Tool Declaration & Built-in Tools Subsystem](#tool-declaration--built-in-tools-subsystem)
- [Think Unit Descriptor Pattern](#think-unit-descriptor-pattern)
- [External Agent Delegation (ThinkAgent)](#external-agent-delegation-thinkagent)
- [Human-in-the-Loop](#human-in-the-loop)

---

## Three-Layer Architecture

```
Layer 3: AmphibiousAutoma (Orchestration)
  ├─ on_agent()         → LLM-driven async generator yielding ThinkUnit /
  │                        ThinkAgent / RETURN
  ├─ on_workflow()      → Deterministic async generator yielding atomic Calls /
  │                        EnterAgent / RETURN
  ├─ _amphiflow()       → Peer state machine driving on_workflow + on_agent
  ├─ _invoke_template() → Single-generator driver (AGENT / WORKFLOW / hooks)
  └─ _dispatch_step()   → Per-yield handler with scope validation

Layer 2: CognitiveWorker / AgentWorker (Think Units — peers)
  ├─ CognitiveWorker    → in-process LLM cycle, anchored on a BaseLlm
  │   └─ thinking()     → single template method: talk to the LLM, return its
  │                       natural result (Response / (tool_calls, content) /
  │                       BaseModel / str); _assemble_decision adapts it
  └─ AgentWorker        → one delegated cycle to an external coding agent,
                          anchored on a BaseAgent (ClaudeCodeAgent / CodexAgent)

Layer 1: Context (State — two loops)
  ├─ OTAContext         → small loop: user_input + ota_record trace + tools
  │                       (framework-owned; tools declared via OTAContext.tool)
  └─ Context            → big loop: free-form cross-turn knowledge (summary())
```

There is no separate "exposure" layer and no `CognitiveContext` / `CognitiveHistory` / skills subsystem — those were removed in the OTA-context rebuild. Context state is two plain Pydantic models; tools are an OTA-loop concern declared on the OTA context class.

## The Two-Loop Context Model

`AmphibiousAutoma[OTAContextT, ContextT]` carries two contexts, resolved from its generic parameters at class-creation time.

### OTAContext — the small loop (framework-owned)

The working context for one run. The framework constructs a fresh instance per `arun()` (seeding `user_input`) and drives it directly:

- `user_input: str` — the run's question / objective.
- `ota_record: List[OTARecord]` — the observe-think-act round trace, one `OTARecord` per round.
- `tools: List[ToolSpec]` — the action-phase affordances this run carries, seeded from the class's declared tools.

Per-round result accessors (`obs_result` / `think_result` / `action_result`) read/write the latest `OTARecord`; `open_record()` opens a new round. `OTARecord` is `extra="allow"`, so a `before_action` / `after_action` hook can fold custom per-round fields (e.g. a `permission_result`) without subclassing.

`OTAContext.summary()` renders `user_input` + the round trace for the prompt (overridable).

### Context — the big loop (free-form)

Cross-turn knowledge (memory, conversation, domain state). Just fields plus an overridable `summary(self, fields)`: an override is handed the raw `{name: value}` dict and composes whatever prompt-facing rendering it wants. Supplied via `arun(context=...)` (optional); shared read-only across the run and any delegation — only the small loop is isolated per sub-run.

### Tool declaration (no auto-injection)

Tools belong to the OTA loop that acts. **Nothing is auto-injected.** Each OTA context declares its tools on the class via `OTAContext.tool(obj)` (decorator or call), which normalizes `obj` (a `ToolSpec`, a bound method, or a plain callable) into a `ToolSpec` and appends it to the class's `_declared_tools`. A subclass inherits its bases' declared tools and may add more. At construction, each run's `tools` field is seeded from the class's declared set (an explicit `tools=` is preserved). `arun` does not assemble or merge any toolset — pure dispatch.

## Observe-Think-Act (OTA) Cycle

One round of the small loop, bracketed as one `OTARecord`:

1. **Observe** — set the round's observation.
   - The worker's `observation(ota_context, context)` runs first.
   - If it returns `_DELEGATE` / `None`, the agent-level `observation` runs.
   - The result lands on `ota_context.obs_result` (an agent-level hook yields `RETURN(text)` to set it; exhausting without `RETURN` preserves the previous value).

2. **Think** — the worker decides.
   - `CognitiveWorker.thinking(ota_context, context)` talks to `self._llm` and returns the protocol's natural result.
   - `_assemble_decision` adapts it into a flat `ThinkResult` (`step_content` + `tool_calls`) stored on `ota_context.think_result`.
   - A decision with **no** `tool_calls` IS the finish.

3. **Act** — execute the decision.
   - `before_action` hooks run (worker → agent delegation); a hook may override the decision.
   - `action_tool_call(ota_context, context)` runs the decision's tool calls concurrently against `ota_context.tools`, producing an `ActionResult` on `ota_context.action_result`.
   - `after_action` hooks run (worker → agent delegation).

A `think_unit`'s `max_attempts` caps how many rounds run; an optional `until` predicate stops early.

## Execution Modes (RunMode)

| Mode | Driver | Best For | Fallback |
|------|--------|----------|----------|
| `AGENT` | `_invoke_template(on_agent)` | Open-ended, adaptive tasks | N/A |
| `WORKFLOW` | `_invoke_template(on_workflow)` | Known, repeatable processes | N/A |
| `AMPHIFLOW` | `_amphiflow` (state machine) | Robust hybrid execution | Step-level + full |
| `AUTO` (default) | Auto-detect from overridden methods | Most subclasses | Inherits from resolved mode |

`AUTO` resolution rules:
- only `on_agent` overridden → `AGENT`
- only `on_workflow` overridden → `WORKFLOW`
- both overridden → `AMPHIFLOW`

All overridable template methods must be **async generators** — the dispatch model is yield-driven. The framework validates this in `__init_subclass__` and raises `TypeError` for a coroutine-form override (`async def` with no `yield`). The base defaults are stub async generators (`if False: yield`), so not overriding is fine; if a real override has no yields, add `if False: yield`.

LLM requirement: an LLM is needed wherever a `CognitiveWorker` runs (any `ThinkUnit`, plus an `AMPHIFLOW` step-recovery sub-run) or an `LLMCall` fires — i.e. typical `AGENT` / `AMPHIFLOW` runs. Pure `WORKFLOW` and pure `ThinkAgent` flows need none.

## Yield Primitive Categories

`_dispatch_step` recognizes seven yield types in three categories. Scope validation happens at dispatch time — mismatches raise `RuntimeError`.

| Category | Primitive | Allowed scopes |
|----------|-----------|----------------|
| **Atomic Call** (operations on the world) | `ActionCall` (deterministic single-tool) | `workflow`, `hook` |
|  | `HumanCall` (HITL via `@human_channel`) | `workflow`, `hook` |
|  | `LLMCall` (chat / structure_output / tool_selector) | `workflow`, `hook` |
| **Mode-switch** (state-machine transition) | `EnterAgent` (suspend workflow → run on_agent) | `workflow` only |
| **Cognitive composition** (inside on_agent) | `ThinkUnit` (in-process CognitiveWorker cycle) | `agent` only |
|  | `ThinkAgent` (delegated AgentWorker cycle) | `agent` only |
| **Control flow** | `RETURN` (PEP 525 return-value workaround) | any |

The asymmetry — atomic Calls forbidden in `agent` scope — is intentional: `on_agent` is reserved for orchestrating cognitive steps via `ThinkUnit` / `ThinkAgent`. Tool / human / LLM operations happen *inside* a `ThinkUnit` (the worker's tool-selection phase, or the LLM calling the declared `request_human` tool), not by yielding from `on_agent`. There is no "switch back to workflow" yield; agent-generator exhaustion is the implicit signal. `RETURN` is intercepted by the drivers directly (not routed through `_dispatch_step`) — it is a control signal, not an operation.

## Peer State-Machine Dispatcher

`AMPHIFLOW` is driven by `_amphiflow`, a single while-loop holding two generator slots on `self._amphi` (an `_AmphiState`):

```
┌──────────────────────────────────────────────────────────┐
│  _amphiflow                                               │
│   ├─ workflow_gen     (created from on_workflow)          │
│   ├─ agent_gen        (created on EnterAgent / fallback)  │
│   ├─ agent_mode_stack (AsyncExitStack for the OTA scope)  │
│   ├─ scope            ("workflow" | "agent")              │
│   └─ consecutive_failures                                 │
│                                                            │
│   while not should_break:                                  │
│     active = agent_gen if scope=="agent" else workflow_gen │
│     item = active.asend(send_value)   # or __anext__       │
│     ├─ RETURN(v)     → set return_value, should_break      │
│     ├─ EnterAgent    → fresh OTA sub-context, agent_gen    │
│     ├─ atomic Call   → _dispatch_step (in active scope)    │
│     │                  → on raise, step-level / full fallback│
│     └─ ThinkUnit     → _dispatch_step (agent scope only)   │
│                                                            │
│   StopAsyncIteration on agent_gen → switch back to workflow│
│   StopAsyncIteration on workflow_gen → finish              │
└──────────────────────────────────────────────────────────┘
```

Key invariants:

- **Single while-loop** — no recursion, no nested driver. `EnterAgent` changes which slot the loop reads from on the next iteration.
- **Fresh-instance delegation** — `EnterAgent` installs a fresh `OTAContext` (its `user_input` = the `EnterAgent.goal`, carrying the OTA class's declared tools) on an `AsyncExitStack` (`_ota_scope`), then hands a fresh `on_agent` generator to the loop. The big-loop `Context` is shared read-only. The parent OTA context is restored when the agent generator exhausts.
- **Implicit switch-back** — agent-generator exhaustion (`StopAsyncIteration`) signals "return control to the workflow"; the suspended workflow resumes (the `yield EnterAgent(...)` evaluates to `None`). A `RETURN(value)` yielded *inside* an `EnterAgent`-driven `on_agent` instead ends the whole run with `value` — agent-scope `RETURN` is a run-level terminate, not a sub-flow return.
- **Cleanup ordering** — on exit (RETURN, exception): close the agent generator (may be mid-yield), unwind the agent-mode `AsyncExitStack` (the OTA scope), then close the workflow generator.

`AGENT` and `WORKFLOW` modes use the simpler `_invoke_template` single-generator driver — nothing to alternate between.

## Workflow Fallback Mechanism

`AMPHIFLOW` defends against two failure sources.

### Generator-internal exception (helper / inline logic between yields raises)

The generator is unrecoverable after a raise — `asend()` cannot resume it — so step-level recovery is impossible. The framework jumps directly to **full fallback**: the workflow generator is dropped (`workflow_gen = None`) and `_enter_agent()` runs `on_agent` for the remaining task. `workflow_gen.aclose()` is wrapped in `try/except` during unwinding, so a raising `finally` still lets the fallback agent run.

### Atomic-Call failure (an `ActionCall` / `HumanCall` / `LLMCall` raises)

`_dispatch_step` raises; the FSM catches it and decides by a single counter:

```
consecutive_failures += 1   (each successful atomic Call resets it to 0)
consecutive_failures >= max_consecutive_fallbacks  → full fallback
consecutive_failures <  max_consecutive_fallbacks  → step-level recovery
```

**Step-level recovery (bounded inline sub-run).** The framework builds a fallback goal describing the failed step + its error, then runs a *bounded* recovery via `_run_fallback_agent(goal)`: a fresh OTA episode of `on_agent` runs to completion against that goal (isolated sub-context, the OTA class's declared tools). Its conclusion — the sub-run's `RETURN` value, else its last think step's `step_content` — is shaped into the failed step's return type by `_shape_fallback_value` and `asend()`-ed back to the suspended workflow, which resumes at the next instruction.

There is **no** injected tool and **no** toolset mutation — the recovery sub-run's own conclusion *is* the resolution. (The recovered value is an internal step value, not the run's answer; the run's `final_answer` comes from the resuming workflow or `summary()`.)

**Full fallback.** When the counter reaches `max_consecutive_fallbacks`, the workflow generator is closed and `_enter_agent()` (no `item`) runs `on_agent` — inheriting the parent's `user_input` — for the rest of the run.

`EnterAgent` is orthogonal to fallback: it is the user's *explicit* mode-switch, not a failure recovery, and does not touch the counter.

## Tool Declaration & Built-in Tools Subsystem

Tools are declared on the OTA context class — see [The Two-Loop Context Model](#the-two-loop-context-model). The framework ships a fixed roster of built-in `FunctionToolSpec` instances in `bridgic.amphibious.builtin_tools.ALL_BUILTIN_TOOLS` (request_human, bash, read_file, write_file, edit_file, glob, grep). They are **not** auto-injected — a run carries a built-in only if its OTA context declares it via `OTAContext.tool(spec)` (or `for t in ALL_BUILTIN_TOOLS: MyOTACtx.tool(t)`). Adding a new built-in means importing its spec into `builtin_tools` and appending it to that tuple.

### Read-before-modify invariant

The filesystem-mutating built-ins (`write_file`, `edit_file`) require a prior `read_file` on the same path AND that the file has not changed externally since. Mechanism:

- `AmphibiousAutoma._read_tracker: Dict[str, float]` — a per-agent dict mapping absolute path → mtime at last read. Reset at every `arun()` entry, scoping the invariant to a single run.
- `read_file` records the file's mtime after a successful read (best-effort: a failed `os.stat` here is silently swallowed so it cannot mask the read).
- `write_file` (for existing files) and `edit_file` consult the tracker and raise `RuntimeError` if the path was never read, or the current mtime is newer than the recorded one.

The tools resolve the agent through the same `current_agent` `ContextVar` used by `request_human`, so the tracker is per-`asyncio.Task`: concurrent `arun()` calls from separate agents never share state.

### Tool exception path

Built-in tools raise on validation failures (`ValueError`, `FileNotFoundError`, `RuntimeError`, `TimeoutError`, …); they do not wrap errors as strings. The per-tool exception handler — the `_run_one` inner function inside `AmphibiousAutoma.action_tool_call` — captures every exception and produces `ActionStepResult(success=False, error=str(exc), tool_result=None)`. In agent mode this becomes part of the next observation; in workflow mode, the `ActionCall` branch aggregates failed results into `RuntimeError("Tool execution failed for: ...")`, which then drives the AMPHIFLOW fallback (or re-raises in pure WORKFLOW).

## Think Unit Descriptor Pattern

Think units use Python descriptors for class-level declaration:

1. `think_unit(worker, *, until=, max_attempts=, on_error=, max_retries=)` returns a `ThinkUnitDescriptor` wrapping one `CognitiveWorker` template.
2. Class- and instance-level access both return the descriptor itself; invocation goes through `yield ThinkUnit("name")` from inside `on_agent`.
3. The dispatcher resolves the name, clones the worker template (`_clone_worker` → `worker._clone()`) for state isolation, resolves per-yield overlays (`until` / `max_attempts`) against descriptor defaults, injects the LLM, and runs the OTA loop via `_run_think_unit`.
4. The `asend()` value is the finishing think's `step_content`.

`ThinkAgentDescriptor` mirrors this for external-agent delegation (cloning an `AgentWorker`); the two cognitive-composition descriptors share the same dispatch contract.

## External Agent Delegation (ThinkAgent)

`ThinkAgent` is the cognitive-composition peer of `ThinkUnit`: where `ThinkUnit` drives an in-process `CognitiveWorker` (one LLM observe-think-act cycle), `ThinkAgent` drives an `AgentWorker` that hands the sub-goal to an **external** coding-agent CLI (`claude code` or OpenAI `codex`; add others by subclassing `BaseAgent`).

### Layer split

```
ThinkAgent (yield primitive)
    │  resolved by AmphibiousAutoma._run_think_agent
    v
AgentWorker  ── context organization: MCP-ify the OTA context's tools, assemble
    │            the message via thinking(), pack an AgentRequest
    v
BaseAgent    ── CLI mechanics: argv, subprocess, completion detection
    │            (ClaudeCodeAgent / CodexAgent ship with the framework)
    v
external coding-agent CLI subprocess
```

`AgentWorker` : `BaseAgent` mirrors `CognitiveWorker` : `BaseLlm` — the worker organizes context and never embeds CLI internals; the base type executes.

### MCP bridge

The parent's project tools (the OTA context's `tools`, minus the framework built-ins; further filtered by `expose_tools`) are exposed to the external agent through an **in-process FastMCP host** booted for the delegation. The external agent discovers and calls them as `mcp__<server>__<tool>`; a synthetic `agent_done` MCP tool is the completion signal. The host is torn down when the delegation ends. (`fastmcp` / `uvicorn` are imported lazily — projects that never use `AgentWorker` pay zero install / import cost.)

### Decision channel

`AgentWorker` does **not** execute the external agent's tool calls — it only *produces* decisions, exactly like `CognitiveWorker`. Each MCP tool call is surfaced onto a per-delegation `asyncio.Queue` as a `(decision, future)` pair. `_run_think_agent` runs a consumer task — alive only for this one delegation — that pulls each decision, runs it through the action phase (so `before_action` / `after_action` hooks fire and the call is recorded in the `AgentTrace`), and resolves the future with the result. `AmphibiousAutoma` remains the only component that *acts*; the worker only thinks.

The `yield ThinkAgent` result is the string the external agent passed to `agent_done(result=...)` (`AgentResult.output`), or `None` if it exited without signalling.

## Human-in-the-Loop

Two entry points for requesting human input — both share the same `@human_channel` registry:

| Entry Point | Context | Mechanism |
|-------------|---------|-----------|
| `yield HumanCall(prompt=, channel=)` | `on_workflow()` body, hooks (rejected in `on_agent`) | The dispatcher routes through `_run_human_call`, then `asend()`s the response back to the generator |
| `request_human` tool | LLM-driven — called from inside any `ThinkUnit`, in any mode | Declared on the OTA context via `OTAContext.tool(request_human_tool)`; resolves the running agent via `current_agent` and routes through `_run_human_call` |

There is **no** code-level imperative API on `AmphibiousAutoma` (no `self.request_human(...)`). The `on_agent` body is reserved for orchestrating cognitive steps via `ThinkUnit`; HITL inside `on_agent` happens autonomously through the LLM calling the declared tool.

**Channel resolution** (applies to both `HumanCall` and the `request_human` tool, since both go through `_run_human_call`):
- `channel=None` + zero `@human_channel` handlers → built-in stdin handler.
- `channel=None` + one handler → that handler used implicitly.
- `channel=None` + 2+ handlers → `RuntimeError` requiring an explicit channel.
- `channel="name"` → invoke that named handler (`RuntimeError` if unknown).

**Registry**: `@human_channel` is a method decorator; `__init_subclass__` walks the MRO (bottom-up so subclass overrides win) and builds a per-class `_human_channels: Dict[str, str]` (channel name → method name). Channel handlers are plain `async def` methods returning `str` — leaf I/O operations; they do not yield framework primitives.

**`request_human` spec**: the exported `request_human_tool` is a plain static `FunctionToolSpec`. The LLM passes `channel="name"` matching a registered `@human_channel` key (the tool's docstring tells it the accepted names follow the agent class's registered channels); routing then goes through the same dispatcher as workflow-side `HumanCall(channel="name", ...)`.

**Concurrency**: `request_human` uses `contextvars.ContextVar` for late-binding. Each `asyncio.Task` (each `arun()`) gets its own isolated binding — concurrent agents sharing the same tool object never interfere.
