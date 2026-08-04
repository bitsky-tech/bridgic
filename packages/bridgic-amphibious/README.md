# Bridgic Amphibious

Bridgic Amphibious is a yield-driven orchestration framework for combining
LLM-directed agents, deterministic workflows, and external coding-agent
delegation in one Python class.

Version 0.2.0 introduces a two-loop context model and an explicit set of yield
primitives. The examples below describe the 0.2.0 API.

## Highlights

- **Two execution loops:** a per-run `OTAContext` for observe-think-act state,
  plus a free-form `Context` for longer-lived knowledge.
- **Three run modes:** agent, workflow, and amphibious workflow-with-fallback.
- **Explicit composition:** `ThinkUnit` drives an in-process
  `CognitiveWorker`; `ThinkAgent` delegates to an external CLI agent.
- **One action pipeline:** model-selected, workflow-selected, and
  external-agent-selected tools are executed by `AmphibiousAutoma`.
- **Human-in-the-loop routing:** deterministic `HumanCall` operations and the
  LLM-callable `request_human` tool share the same named channel registry.

## Installation

Bridgic Amphibious 0.2 supports Python 3.10 through 3.13.

```bash
pip install bridgic-amphibious
```

The package also provides a small project scaffold:

```bash
bridgic-amphibious create --task "Investigate order A-42"
```

This creates `amphi.py` in the current directory. Use `--base-dir` to choose
another directory. The command refuses to overwrite an existing `amphi.py`.

## Quick start: an LLM-directed agent

Every `AmphibiousAutoma` declares two context types. The first generic
argument must be an `OTAContext`; the second must be a `Context`.

```python
from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveWorker,
    Context,
    OTAContext,
    RETURN,
    ThinkUnit,
    think_unit,
)
from bridgic.core.model.types import Message, Role


class SupportOTAContext(OTAContext):
    """Per-run input, tools, and observe-think-act records."""

    def summary(self, fields):
        return (
            f"User input: {fields['user_input']}\n"
            f"OTA history: {fields['ota_record']}"
        )


@SupportOTAContext.tool
async def lookup_order(order_id: str) -> str:
    """Return the current status of an order."""
    return f"Order {order_id} is packed and waiting for pickup."


class SupportKnowledge(Context):
    customer_tier: str = "standard"

    def summary(self, fields):
        return f"Customer tier: {fields['customer_tier']}"


class SupportThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        knowledge = context.summary() if context is not None else ""
        prompt = "\n\n".join(part for part in (
            knowledge,
            ota_context.summary(),
            "Use the available tools when needed, then answer the user.",
        ) if part)
        return await self._llm.aselect_tool(
            messages=[Message.from_text(prompt, role=Role.USER)],
            tools=[tool.to_tool() for tool in ota_context.tools],
        )


class SupportAgent(AmphibiousAutoma[SupportOTAContext, SupportKnowledge]):
    support = think_unit(SupportThink(), max_attempts=6)

    async def on_agent(self, ota_context, context=None):
        answer = yield ThinkUnit("support")
        yield RETURN(answer)


agent = SupportAgent()
answer = await agent.arun(
    llm=my_llm,
    user_input="Where is order A-42?",
    context=SupportKnowledge(customer_tier="priority"),
)
print(answer)
```

`my_llm` is a Bridgic-compatible LLM implementation that supports tool
selection. A `CognitiveWorker` implements `thinking()` and returns the natural
result of the LLM protocol it uses:

- `achat()` returns a response and finishes the think unit.
- `aselect_tool()` returns `(tool_calls, content)`; the worker continues while
  tool calls remain and finishes when the list is empty.
- `astructured_output()` returns a Pydantic model or dictionary, which is
  serialized into the step content.
- Plain text also finishes the think unit.

`think_unit(...)` is a class-level declaration. It is invoked from
`on_agent()` by yielding `ThinkUnit("name")`; the descriptor itself is not
awaited. `max_attempts` caps observe-think-act cycles, while an optional
`until` callable can stop after a non-finishing cycle:

```python
answer = yield ThinkUnit(
    "support",
    until=lambda ota: len(ota.ota_record) >= 3,
    max_attempts=10,
)
```

## The two contexts

`OTAContext` is the framework-owned small-loop context. A fresh instance is
normally created for each `arun()` and contains:

- `user_input`: the run objective or another caller-defined payload.
- `ota_record`: one `OTARecord` per observe-think-act round.
- `tools`: the exact action affordances declared by the context class.
- `obs_result`, `think_result`, and `action_result`: accessors for the current
  round.

`Context` is free-form big-loop knowledge. Subclasses can declare Pydantic
fields and override `summary(fields)` to render them for a prompt. The same
big-loop context is shared with nested agent delegations; each delegation gets
its own isolated `OTAContext`.

A caller may provide a pre-built small-loop context with
`arun(ota_context=...)`. In that case its own `user_input` and tools are used
as-is.

## Declaring tools

No tools are injected automatically. Register every tool the small loop should
carry on its `OTAContext` subclass. `tool()` accepts a callable, a bound method,
or an existing `ToolSpec`, and works both as a decorator and a class method:

```python
from bridgic.amphibious import (
    bash_tool,
    read_file_tool,
    request_human_tool,
)


SupportOTAContext.tool(read_file_tool)
SupportOTAContext.tool(bash_tool)
SupportOTAContext.tool(request_human_tool)
```

The shipped tool specs are `request_human_tool`, `bash_tool`,
`read_file_tool`, `write_file_tool`, `edit_file_tool`, `glob_tool`, and
`grep_tool`. Register only the capabilities the run needs.

## Run modes and yield primitives

`RunMode.AUTO` is the default and resolves from the template methods a class
implements:

| Implemented methods | Resolved mode |
| --- | --- |
| `on_agent()` only | `RunMode.AGENT` |
| `on_workflow()` only | `RunMode.WORKFLOW` |
| Both | `RunMode.AMPHIFLOW` |

In amphibious mode, execution starts in the workflow. An atomic workflow step
that fails can be handed to a bounded agent recovery flow; repeated failures
can switch the remainder of the run to agent mode. Configure the threshold
with `max_consecutive_fallbacks`.

Each primitive has an explicit scope and sends a result back into the async
generator:

| Primitive | Valid scope | Value sent back by `yield` |
| --- | --- | --- |
| `ActionCall` | `on_workflow` or a hook | `list[ToolResult]` |
| `HumanCall` | `on_workflow` or a hook | Human response string |
| `LLMCall` | `on_workflow` or a hook | Protocol-specific LLM result |
| `EnterAgent` | `on_workflow` | Nested result in workflow mode; control transfer in amphibious mode |
| `ThinkUnit` | `on_agent` | Latest think step content |
| `ThinkAgent` | `on_agent` | External agent's completion string, or `None` |
| `RETURN` | Any framework async generator | Return value for the current template |

`on_agent()` is reserved for `ThinkUnit`, `ThinkAgent`, and `RETURN`.
Deterministic tool calls, human pauses, and direct LLM calls belong in
`on_workflow()` or in `observation`, `before_action`, and `after_action` hooks.
`AmphibiousAutoma` template overrides must be async generators; if an override
has no real yield, include `if False: yield` to preserve that shape.

## Deterministic workflow, direct LLM call, and agent entry

The following class inherits the `on_agent()` strategy from the quick start
and adds a workflow. Because both strategies are present, `RunMode.AUTO` would
select amphibious mode. This example explicitly selects `RunMode.WORKFLOW` so
`EnterAgent` behaves as a nested call and then resumes the workflow.

```python
import asyncio

from bridgic.amphibious import (
    ActionCall,
    EnterAgent,
    HumanCall,
    LLMCall,
    RETURN,
    RunMode,
    human_channel,
)


class SupportFlow(SupportAgent):
    @human_channel("terminal")
    async def ask_terminal(self, prompt: str) -> str:
        return await asyncio.to_thread(input, f"{prompt}\n> ")

    async def on_workflow(self, ota_context, context=None):
        results = yield ActionCall("lookup_order", order_id="A-42")
        status = results[0].result if results else "No order result"

        summary = yield LLMCall.chat(
            f"Summarize this status for the customer: {status}"
        )
        approved = yield HumanCall(
            channel="terminal",
            prompt=f"Send this reply?\n{summary}",
        )
        if approved.strip().lower() != "yes":
            yield RETURN("Reply cancelled")

        escalation = yield EnterAgent(
            goal="Check whether order A-42 needs proactive intervention"
        )
        yield RETURN(escalation or summary)


answer = await SupportFlow().arun(
    llm=my_llm,
    user_input="Resolve the order question",
    context=SupportKnowledge(customer_tier="priority"),
    mode=RunMode.WORKFLOW,
)
```

`ActionCall` executes one named tool and returns a list of `ToolResult`
objects. `LLMCall.chat()` requires `llm=` on `arun()`. The other direct
protocol constructors are `LLMCall.structure_output(..., constraint=...)` and
`LLMCall.tool_selector(..., tools=...)`; the LLM must implement the matching
Bridgic protocol.

In `RunMode.WORKFLOW`, `EnterAgent` suspends the workflow, runs `on_agent()`
with a fresh small-loop context whose `user_input` is the supplied goal, and
then sends the nested result back to the workflow. The big-loop `Context`
remains shared. In `RunMode.AMPHIFLOW`, it is a state-machine control transfer;
an agent-side `RETURN` completes the overall run instead of resuming the
workflow.

## Human channels

Decorate a plain async method with `@human_channel` or
`@human_channel("name")`. It accepts a prompt and returns a string.

- With no registered channel, `HumanCall(channel=None)` uses stdin.
- With exactly one registered channel, its name may be omitted.
- With multiple registered channels, each call must name one explicitly.

Registering `request_human_tool` on the `OTAContext` lets an LLM call
`request_human(prompt, channel=None)` during a `ThinkUnit`. It resolves through
the same channel registry as a workflow's `HumanCall`.

## Delegating to Codex or Claude Code

`ThinkAgent` delegates one cognitive step to an external CLI through an
`AgentWorker`. Project tools from the nested `OTAContext` are exposed to the
CLI through an in-process MCP server, and calls return through the parent
automa's action and hook pipeline.

```python
from bridgic.amphibious import (
    AgentWorker,
    AmphibiousAutoma,
    ClaudeCodeAgent,
    CodexAgent,
    RETURN,
    ThinkAgent,
    think_agent,
)


class DelegatingAgent(AmphibiousAutoma[SupportOTAContext, SupportKnowledge]):
    investigate_with_codex = think_agent(
        AgentWorker(CodexAgent(sandbox_mode="workspace-write")),
        expose_tools=["lookup_order"],
    )
    review_with_claude = think_agent(
        AgentWorker(
            ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"])
        ),
        expose_tools=["lookup_order"],
    )

    async def on_agent(self, ota_context, context=None):
        investigation = yield ThinkAgent(
            "investigate_with_codex",
            goal="Investigate order A-42 with the exposed project tools.",
        )
        review = yield ThinkAgent(
            "review_with_claude",
            goal=f"Review this investigation and return a concise answer:\n{investigation}",
        )
        yield RETURN(review or investigation)
```

The per-yield `goal=` overrides the nested run's objective.
`expose_tools=[...]` narrows the project tools available over MCP; omitting it
exposes all registered non-builtin project tools. Each delegation uses a
temporary working directory, so the CLI's own filesystem tools do not see the
caller's project automatically. Expose explicit project tools over MCP when
the delegated agent needs project data or mutations.

The `codex` or `claude` executable must be installed and authenticated
separately. `CodexAgent` accepts `bin`, `sandbox_mode`, and
`completion_timeout`. `ClaudeCodeAgent` accepts `bin`,
`allowed_builtin_tools`, `permission_mode`, and `completion_timeout`.

## Tracing

Tracing is opt-in:

```python
answer = await agent.arun(
    llm=my_llm,
    user_input="Investigate order A-42",
    trace=True,
    workdir=".bridgic",
)
```

`trace=True` enables in-memory `AgentTrace` capture. Supplying `workdir` creates
`<workdir>/runs/<run_id>/`; when both options are set, the trace is also
persisted there as `trace.json`.

## License

Bridgic Amphibious is released under the MIT License. See the repository's
`LICENSE` file for the full text.
