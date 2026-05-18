"""
Amphibious Agent Framework — Consolidated Data Models.

All Pydantic models, dataclasses, enums, and type aliases used across the
amphibious module are gathered here as a single source of truth.

Sections are annotated with the module(s) that consume each model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from bridgic.core.model.types import Message, Tool
    from bridgic.core.model.protocols import Constraint


################################################################################################################
# Context layer models  (used by: _context.py)
################################################################################################################

class Step(BaseModel):
    """A single execution step with content, result, and metadata.

    Used by: _context.py (CognitiveHistory, CognitiveContext.add_info),
             _amphibious_automa.py (_action, _record_trace_step)
    """
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[bool] = None  # Optional status flag for backward compatibility


class Skill(BaseModel):
    """A skill definition following SKILL.md format.

    Used by: _context.py (CognitiveSkills)
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


################################################################################################################
# Worker layer models  (used by: _cognitive_worker.py, _amphibious_automa.py)
################################################################################################################

class RunMode(str, Enum):
    """The mode of the run.

    Used by: _amphibious_automa.py (arun, router)
    """
    AGENT = "agent"
    WORKFLOW = "workflow"
    AMPHIFLOW = "amphiflow"
    AUTO = "auto"

class DetailRequest(BaseModel):
    """Request for detailed information about a specific item in a LayeredExposure field.

    Used by: _cognitive_worker.py (acquiring policy)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["field", "index"],
            "additionalProperties": False,
        }
    )
    field: str = Field(description="Name of the field to get details from (e.g., 'cognitive_history', 'skills')")
    index: int = Field(description="0-based index of the item to get details for")


class ToolArgument(BaseModel):
    """A single tool argument as name-value pair.

    Used by: _cognitive_worker.py (StepToolCall), _amphibious_automa.py (action phase)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["name", "value"],
            "additionalProperties": False,
        }
    )
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value as string")

    @field_validator('value', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        return str(v) if not isinstance(v, str) else v


class StepToolCall(BaseModel):
    """A single tool call specification.

    Used by: _cognitive_worker.py (ThinkModel output), _amphibious_automa.py (action phase)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool", "tool_arguments"],
            "additionalProperties": False,
        }
    )
    tool: str = Field(description="Name of the tool to call")
    tool_arguments: List[ToolArgument] = Field(
        description="Arguments as list of name-value pairs, e.g., [{name: 'city', value: 'Beijing'}]"
    )


class _ThinkBase(BaseModel):
    """Unified base for all dynamically-generated ThinkModel variants.

    Factory (_create_think_model) adds: output, details, rehearsal,
    reflection — all optional and conditional on configuration.

    Used by: _cognitive_worker.py (_create_think_model)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["step_content"],
            "additionalProperties": False,
        }
    )

    step_content: str = Field(
        default="",
        description="Description of what to do in this step, or your analysis/reasoning"
    )
    finish: bool = Field(
        default=False,
        description="Set True when your current sub-task is FULLY complete and no more steps are needed."
    )

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)


def _coerce_none_to_list(v: Any) -> list:
    """Coerce non-list values to empty list for field validation.

    Some LLMs may place summary text or a dict in the ``output`` field
    instead of a proper ``List[StepToolCall]``.  Rather than letting
    Pydantic raise a ``ValidationError``, silently discard the invalid
    value so the cycle can still finish gracefully (the content lives
    in ``step_content`` anyway).
    """
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return []


################################################################################################################
# Yield primitives
#
# Used by every user-facing template method (on_workflow / on_agent /
# observation / before_action / after_action) as an async generator
# yielding these. Three categories, six atomic types:
#
#   Atomic Calls (operations on the world):
#     ActionCall  — deterministic single-tool execution         (on_workflow / hooks)
#     HumanCall   — pause and request human input via           (on_workflow / hooks)
#                   ``@human_channel`` registry
#     LLMCall     — direct LLM invocation via a bridgic-core    (on_workflow / hooks)
#                   protocol (chat / structure_output / tool_selector)
#
#   Mode-switch signals (state-machine transitions):
#     EnterAgent  — suspend on_workflow, run on_agent in a      (on_workflow only)
#                   snapshotted context; agent generator
#                   exhaustion implicitly resumes workflow
#
#   Cognitive composition (inside on_agent):
#     ThinkUnit   — invoke a class-level ``think_unit`` by name (on_agent only)
#
#   Control flow:
#     RETURN      — communicate a "return value" out of an      (any scope)
#                   async generator (PEP 525 forbids native
#                   ``return value``); from a top-level body
#                   this terminates the entire arun
#
# Scope validation lives in ``_dispatch_step``; mismatches raise
# ``RuntimeError`` at dispatch time.
################################################################################################################

class WorkflowDecision(BaseModel):
    """
    Single-step deterministic decision for workflow mode.

    Used by: _amphibious_automa.py (state-machine driver, ActionCall)
    """
    model_config = ConfigDict(extra="forbid")

    step_content: str = ""
    output: List[StepToolCall] = Field(default_factory=list)


@dataclass(init=False)
class ActionCall:
    """Yielded by on_workflow() / hooks for deterministic single-tool execution.

    Each instance wraps exactly one tool call.

    Used by: _amphibious_automa.py (state-machine driver)

    Usage::
        yield ActionCall("navigate_to", url="http://example.com")
        yield ActionCall("click_element_by_ref", description="Click submit", ref="e42")
        result = yield ActionCall("fill_field", name="user", value="john")
    """
    tool_name: str
    description: str
    tool_args: Dict[str, Any]
    decision: WorkflowDecision = field(repr=False)

    def __init__(
        self,
        tool_name: str,
        *,
        description: str = "",
        **tool_args: Any,
    ) -> None:
        self.tool_name = tool_name
        self.description = description
        self.tool_args = tool_args
        self.decision = WorkflowDecision(
            step_content=description,
            output=[StepToolCall(
                tool=tool_name,
                tool_arguments=[
                    ToolArgument(name=k, value=str(v)) for k, v in tool_args.items()
                ],
            )],
        )


@dataclass
class HumanCall:
    """Yielded to pause execution and request human input.

    Execution is suspended until the registered ``@human_channel`` handler
    provides a response, which is returned to the generator via ``asend()``
    as a plain string.

    Channel resolution (at dispatch time):

    * ``channel=None`` → if exactly one ``@human_channel`` handler is
      registered, use it; if zero handlers are registered, the framework
      falls back to a built-in stdin handler; if 2+ handlers are
      registered, raises ``RuntimeError`` requiring explicit channel.
    * ``channel="name"`` → invoke the handler registered under that name.

    Per-call timeouts are not exposed; if needed, the channel handler
    should enforce its own timeout.

    Used by: _amphibious_automa.py (_dispatch_step)

    Usage::
        feedback = yield HumanCall(prompt="Please verify (yes/no):")
        feedback = yield HumanCall(channel="feishu", prompt="Confirm?")
    """
    prompt: str = ""
    channel: Optional[str] = None


@dataclass
class EnterAgent:
    """Yielded to suspend ``on_workflow`` and switch into ``on_agent``.

    A **mode-switch** signal (not a function call): the workflow
    generator suspends; a fresh agent generator runs until it exhausts;
    workflow resumes at the next instruction after this yield. No stack,
    no recursion — each EnterAgent creates a fresh agent generator.

    The optional fields snapshot context for the agent flow only:
    ``goal`` overrides ``ctx.goal``; ``history`` overrides
    ``ctx.cognitive_history`` (``None`` → fresh); ``tools`` / ``skills``
    are name-filters applied to ``ctx.tools`` / ``ctx.skills``. All
    overrides restore when the agent exhausts.

    EnterAgent scopes *what the agent sees*; it does not control *how it
    thinks*. For a single named cognitive step, use ``ThinkUnit`` from
    inside ``on_agent``. Requires the class to override ``on_agent``.

    >>> yield EnterAgent(goal="Handle the login popup")
    >>> yield EnterAgent(goal="Pick a flight", tools=["search_flights", "book"])
    >>> yield EnterAgent(goal="Summarize", history=prior_messages, skills=["summary"])
    """
    goal: str = ""
    history: Optional[Any] = None       # Optional[CognitiveHistory]; None → fresh CognitiveHistory()
    tools: Optional[List[str]] = None   # Tool-name filter applied to ctx.tools while in agent mode
    skills: Optional[List[str]] = None  # Skill-name filter applied to ctx.skills while in agent mode


LLMCallProtocol = Literal["chat", "structure_output", "tool_selector"]


@dataclass(frozen=True)
class LLMCall:
    """Yielded by ``on_workflow`` to invoke ``self._llm`` via a bridgic-core protocol.

    Result via ``asend()`` by protocol:

    * ``"chat"`` → ``str`` (extracted from ``Response.message.content``)
    * ``"structure_output"`` → value from ``StructuredOutput.astructured_output()``
    * ``"tool_selector"`` → ``Tuple[List[ToolCall], Optional[str]]``

    ``prompt`` becomes the final ``Role.USER`` message; ``history`` (if
    given) is prepended verbatim. Per-call temperature / kwargs are
    deliberately not exposed — those are baked at LLM construction time.

    >>> text = yield LLMCall.chat("What is 2+2?")
    >>> parsed = yield LLMCall.structure_output("Extract...", constraint=PydanticModel(model=Schema))
    >>> calls, reply = yield LLMCall.tool_selector("...", tools=[...])
    """
    protocol: LLMCallProtocol
    prompt: str = ""
    history: Optional[List["Message"]] = None
    constraint: Optional["Constraint"] = None     # required iff protocol == "structure_output"
    tools: Optional[List["Tool"]] = None          # required iff protocol == "tool_selector"

    def __post_init__(self) -> None:
        if self.protocol == "structure_output" and self.constraint is None:
            raise ValueError(
                "LLMCall(protocol='structure_output') requires a `constraint=` argument."
            )
        if self.protocol == "tool_selector" and not self.tools:
            raise ValueError(
                "LLMCall(protocol='tool_selector') requires a non-empty `tools=` argument."
            )
        if self.protocol == "chat" and (self.constraint is not None or self.tools is not None):
            raise ValueError(
                "LLMCall(protocol='chat') does not accept `constraint` or `tools`."
            )

    @classmethod
    def chat(
        cls,
        prompt: str,
        *,
        history: Optional[List["Message"]] = None,
    ) -> "LLMCall":
        """Construct a ``protocol='chat'`` LLMCall."""
        return cls(protocol="chat", prompt=prompt, history=history)

    @classmethod
    def structure_output(
        cls,
        prompt: str,
        *,
        constraint: "Constraint",
        history: Optional[List["Message"]] = None,
    ) -> "LLMCall":
        """Construct a ``protocol='structure_output'`` LLMCall."""
        return cls(
            protocol="structure_output",
            prompt=prompt,
            history=history,
            constraint=constraint,
        )

    @classmethod
    def tool_selector(
        cls,
        prompt: str,
        *,
        tools: List["Tool"],
        history: Optional[List["Message"]] = None,
    ) -> "LLMCall":
        """Construct a ``protocol='tool_selector'`` LLMCall."""
        return cls(
            protocol="tool_selector",
            prompt=prompt,
            history=history,
            tools=tools,
        )


@dataclass(frozen=True)
class ThinkUnit:
    """Yielded inside ``on_agent`` to invoke a class-level ``think_unit``.

    The dispatcher resolves ``name`` against the class and runs the
    associated ``ThinkUnitDescriptor`` via ``_ThinkUnitRuntime``. Fields
    beyond ``name`` overlay the descriptor's defaults (``None`` =
    descriptor value). The ``asend()`` result is the worker's typed
    output (or ``None`` if there is no ``output_schema``).

    >>> result = yield ThinkUnit("main_think")
    >>> result = yield ThinkUnit("exec_think", until=lambda c: c.done, max_attempts=20)
    """
    name: str
    until: Optional[Callable[..., Union[bool, Awaitable[bool]]]] = None
    max_attempts: Optional[int] = None
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None


@dataclass(frozen=True)
class ThinkAgent:
    """Yielded to invoke a class-level ``think_agent`` declaration by name.

    Unlike ``ThinkUnit`` (one in-process OTC cycle), ``ThinkAgent`` hands
    the sub-goal off to an **external** agent runtime (currently
    ``claude code``). The external agent is bound to the parent's task
    tools via an in-process MCP server, so every tool call routes back
    through ``_run_action_call`` and the parent's hooks fire normally.

    Fields beyond ``name`` overlay the descriptor's defaults (``None`` =
    descriptor value). The ``asend()`` result is the string the external
    agent passed to ``agent_done(result=...)``, or the last chunk of
    assistant text if the agent exited without signalling.

    >>> class MyAutoma(AmphibiousAutoma[MyContext]):
    ...     write_article = think_agent()
    ...     async def on_agent(self, ctx):
    ...         result = yield ThinkAgent("write_article", goal="Write the article.")
    ...         yield RETURN(result)
    """
    name: str
    goal: Optional[str] = None
    expose_tools: Optional[List[str]] = None
    allowed_builtin_tools: Optional[List[str]] = None
    permission_mode: Optional[str] = None


@dataclass(frozen=True)
class RETURN:
    """Yielded to communicate a return value out of an async generator.

    PEP 525 forbids ``return value`` inside async generators (only bare
    ``return`` is allowed). ``RETURN(value)`` is the framework-level
    workaround: when the dispatcher receives it, it captures
    ``RETURN.value``, immediately closes the generator, and returns the
    value to its caller. Anything yielded after a ``RETURN`` is
    unreachable.

    For top-level template-method generators (``on_agent`` /
    ``on_workflow``), the captured value is written to
    ``self._final_answer`` (overriding the auto-capture from history).

    Used by: _amphibious_automa.py (_dispatch_step)

    Usage::
        async def on_agent(self, ctx):
            yield ThinkUnit("main_think", max_attempts=20)
            yield RETURN(ctx.cognitive_history.get_all()[-1].content)
    """
    value: Any = None


################################################################################################################
# Action result models  (used by: _amphibious_automa.py)
################################################################################################################

class ErrorStrategy(Enum):
    """Error handling strategy for worker execution via ``_run()``.

    Used by: _amphibious_automa.py (_run method), _amphibious_automa.py (ThinkUnitDescriptor)
    """
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently ignore exceptions
    RETRY = "retry"    # Retry up to max_retries times


class ActionStepResult(BaseModel):
    """Result of executing one tool in the action phase.

    Used by: _amphibious_automa.py (action_tool_call)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool_id", "tool_name", "tool_arguments", "tool_result", "success"],
            "additionalProperties": False,
        }
    )
    tool_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    success: bool = True
    error: Optional[str] = None


class ActionResult(BaseModel):
    """Overall result of the action phase (one or more tool executions).

    Used by: _amphibious_automa.py (action_tool_call, _record_trace_step)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["results"],
            "additionalProperties": False,
        }
    )
    results: List[ActionStepResult]


@dataclass
class ToolResult:
    """Single tool execution result returned to workflow generator via asend().

    Used by: _amphibious_automa.py (state-machine driver), on_workflow user code
    """
    tool_name: str
    tool_arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None


################################################################################################################
# Trace data models  (used by: _amphibious_automa.py — AgentTrace)
################################################################################################################

class StepOutputType(str, Enum):
    """Discriminator for the kind of output a trace step produced.

    Used by: _amphibious_automa.py (AgentTrace, _record_trace_step,
    _record_llm_call_trace, _record_think_agent_trace)
    """
    TOOL_CALLS = "tool_calls"
    STRUCTURED = "structured"
    CONTENT_ONLY = "content_only"
    LLM_CALL = "llm_call"
    THINK_AGENT = "think_agent"


class RecordedToolCall(BaseModel):
    """A complete record of one tool invocation.

    Used by: _amphibious_automa.py (AgentTrace.build)
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    success: bool = True
    error: Optional[str] = None


class TraceStep(BaseModel):
    """Record of one observe-think-act cycle.

    Used by: _amphibious_automa.py (AgentTrace.build)
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    step_content: str
    tool_calls: List[RecordedToolCall] = Field(default_factory=list)
    observation: Optional[str] = None
    observation_hash: Optional[str] = None
    output_type: StepOutputType = StepOutputType.TOOL_CALLS
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_class: Optional[str] = None
    llm_call_protocol: Optional[str] = None  # set when output_type == LLM_CALL
    think_agent_name: Optional[str] = None   # set when output_type == THINK_AGENT


################################################################################################################
# Utility functions  (used by: _amphibious_automa.py)
################################################################################################################

def observation_fingerprint(obs: Any) -> Optional[str]:
    """Compute a stable hash fingerprint of an observation value.

    Used for divergence detection during replay. Returns None for
    None observations.
    """
    if obs is None:
        return None
    try:
        serialized = json.dumps(obs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(obs)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
