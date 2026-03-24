# Worker — CognitiveWorker

## Table of Contents

- [Inline vs Subclass — when to use which](#inline-vs-subclass--when-to-use-which)
- [Template methods — hooks in the OTA cycle](#template-methods--what-each-hook-does-in-the-ota-cycle)
- [Constructor parameters](#constructor-parameters)
- [Cognitive policies (acquiring / rehearsal / reflection)](#cognitive-policies--what-they-actually-do)
- [output_schema — structured output without tools](#output_schema--structured-output-without-tools)

---

## What a worker is

A `CognitiveWorker` is a **pure thinking unit**. It shapes the LLM prompt and hooks into the OTA cycle, but it does **not** execute tools — that's the agent's job.

Think of it as the "personality" of one thinking step: what role the LLM plays, what it observes, how it processes results.

## Inline vs Subclass — when to use which

**Inline** — simple, single-purpose thinking with no custom hooks:

```python
worker = CognitiveWorker.inline("Plan ONE immediate next step.")
worker = CognitiveWorker.inline("Analyze the situation.", enable_rehearsal=True)
worker = CognitiveWorker.inline("Produce a plan.", output_schema=PlanResult)
```

Use inline when:
- The thinking prompt is a simple string
- You don't need custom observation, before_action, or after_action
- The worker is straightforward ReAct-style

**Subclass** — when you need custom hooks or dynamic prompts:

```python
class ResearchWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return "You are a research assistant. Plan ONE immediate next step."

    async def observation(self, context):
        """Inject domain-specific state the LLM should see."""
        return f"Documents found: {len(context.documents)}\nBudget remaining: {context.budget}"

    async def after_action(self, step_result, ctx):
        """Update context after each tool execution."""
        ctx.total_searches += 1
        return _DELEGATE  # Also run agent-level after_action
```

Use subclass when:
- You need custom observation (domain-specific state beyond context summary)
- You want to intercept/modify tool calls (before_action)
- You want to update context after execution (after_action)
- You need full control over the LLM message structure (build_messages)

## Template methods — what each hook does in the OTA cycle

```
① Observe   →  worker.observation(ctx)     — what extra info does the LLM see?
② Think     →  worker.thinking()           — what role/instructions does the LLM follow?
                worker.build_messages(...)  — (advanced) full control over prompt assembly
③ Act       →  worker.before_action(...)   — can I filter/modify the tool calls?
                [agent executes tools]
                worker.after_action(...)    — should I update context with results?
```

| Method | Default | When to override |
|--------|---------|-----------------|
| `thinking()` | `NotImplementedError` | **Always required.** The system prompt for the LLM. |
| `observation(ctx)` | `_DELEGATE` | When a worker needs domain-specific observations beyond what the agent provides. E.g., page content for a browser worker. |
| `before_action(decision, ctx)` | `_DELEGATE` | When you need to filter dangerous tools, modify arguments, or block certain calls. The `decision` is a list of `(ToolCall, ToolSpec)` tuples. |
| `after_action(result, ctx)` | `_DELEGATE` | When you need to update context fields based on tool results. E.g., increment counters, extract findings. |
| `build_messages(...)` | Standard assembly | When you need full control over the LLM message structure (rare). |

**`_DELEGATE`**: A bare `object()` sentinel. Returning it from any hook means "fall through to the agent-level hook." Always checked via `is` identity (`if result is _DELEGATE`). This is the default for all hooks — override only the ones you need. When a worker's `before_action` returns `_DELEGATE`, the agent-level `before_action` receives the **original** decision (not the sentinel).

### `build_messages()` — default behavior

By default, the LLM receives two messages:

```
System: [thinking prompt] + [tools description] + [output instructions]
User:   [context summary: goal, status, history] + [observation]
```

Override `build_messages()` only when you need a different structure (e.g., few-shot examples, separate tool descriptions, custom message roles):

```python
async def build_messages(self, think_prompt, tools_description,
                          output_instructions, context_info):
    return [
        Message.from_text(text=f"{think_prompt}\n\n{tools_description}\n\n{output_instructions}", role="system"),
        Message.from_text(text=context_info, role="user"),
    ]
```

## Constructor parameters

```python
CognitiveWorker(
    llm=None,               # Injected by agent at runtime — leave as None
    enable_rehearsal=False,  # Enable mental simulation before acting
    enable_reflection=False, # Enable information quality check before acting
    verbose=None,            # Override agent's verbose setting
    verbose_prompt=None,     # Also print the full prompt sent to LLM
    output_schema=None,      # Pydantic BaseModel → typed output, skips tool loop
)
```

## Cognitive policies — what they actually do

All policies are **fire-once**: they activate at most once per `arun()` round, then close permanently. This prevents the LLM from looping endlessly on meta-cognition.

### Acquiring (built-in, always available)

**Problem it solves**: LayeredExposure fields (skills, history, custom) show summaries by default. The LLM might need full details to make a decision.

**How it works**:
1. LLM sees a `details` field in its output schema: `details: [{field: "skills", index: 0}]`
2. If filled: system calls `context.get_details("skills", 0)` → reveals full content
3. Acquiring closes, LLM re-thinks with the new details in the prompt
4. LLM makes its actual decision with full information

### Rehearsal (opt-in: `enable_rehearsal=True`)

**Problem it solves**: The LLM might act impulsively. Rehearsal forces it to mentally simulate the outcome before committing.

**How it works**:
1. LLM sees a `rehearsal` field: "To ensure accuracy, mentally simulate the next action first..."
2. If filled: the simulation text is injected into the next round as `## Mental Simulation (Rehearsal): ...`
3. Rehearsal closes, LLM re-thinks with its own simulation in context
4. LLM decides whether to proceed or adjust its plan

**Use when**: Actions are expensive or irreversible (API calls, file writes, purchases).

### Reflection (opt-in: `enable_reflection=True`)

**Problem it solves**: The LLM might act with insufficient information. Reflection forces it to evaluate information quality first.

**How it works**:
1. LLM sees a `reflection` field: "Evaluate information sufficiency before committing..."
2. If filled: the assessment is injected as `## Information Assessment (Reflection): ...`
3. Reflection closes, LLM re-thinks with its quality assessment in context
4. LLM can decide to gather more info or proceed

**Use when**: Information quality matters (research tasks, analysis, decisions with consequences).

### Policy interaction in the OTA cycle

```
Round 1: LLM has [acquiring + rehearsal + reflection] open
         LLM fills "details" → acquiring fires, closes → re-think

Round 2: LLM has [rehearsal + reflection] open, plus disclosed details
         LLM fills "rehearsal" → rehearsal fires, closes → re-think

Round 3: LLM has [reflection] open, plus details + simulation
         LLM fills nothing → proceeds to tool calls → done
```

Each policy adds at most one extra thinking round. With all three enabled, worst case is 3 extra rounds before the actual action.

## `output_schema` — structured output without tools

When set, the worker returns a typed Pydantic instance **instead of** tool calls. The action phase is skipped entirely — the result is stored directly in history.

**Use when**: You need structured data from the LLM (plans, classifications, extractions) rather than tool invocations.

```python
from pydantic import BaseModel

class ResearchPlan(BaseModel):
    queries: list[str]
    focus_areas: list[str]
    estimated_steps: int

planner = think_unit(
    CognitiveWorker.inline("Analyze the topic and produce a structured plan.", output_schema=ResearchPlan),
    # max_attempts=1 (default) — one-shot decision, no looping
)

# In on_agent:
plan_result = await self.planner
plan = plan_result.output  # ResearchPlan instance
```

The `output` field on the result holds the Pydantic instance. `finish` is automatically True for output_schema workers.

See also: [orchestration.md](orchestration.md) for `think_unit` parameters and `arun()`, [error-handling.md](error-handling.md) for what happens when a cycle fails.
