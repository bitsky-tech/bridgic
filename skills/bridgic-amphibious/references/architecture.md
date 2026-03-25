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
| `AMPHIBIOUS` | Workflow + LLM fallback | Robust hybrid execution | Automatic |
| `AUTO` (default) | Auto-detect | Detects if `on_workflow` is overridden | Auto |

- `AUTO`: if `on_workflow()` is overridden → AMPHIBIOUS; otherwise → AGENT.

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

In AMPHIBIOUS mode:

1. Deterministic step fails → check `consecutive_failures < max_consecutive_fallbacks`
2. If within limit: agent fixes the specific step (scoped goal via `snapshot`)
3. If exceeded: abandon workflow → call `on_agent()` for full agent mode
4. `AgentFallback` yield explicitly delegates a sub-task to agent mode
