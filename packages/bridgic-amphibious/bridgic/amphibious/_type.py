"""
Amphibious Agent Framework — Consolidated Data Models.

All Pydantic models, dataclasses, enums, and type aliases used across the
amphibious module are gathered here as a single source of truth.

Sections are annotated with the module(s) that consume each model.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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
    """One act-phase result — the tool-execution outcome of a single
    observe-think-act cycle.

    Carries only the result payload (an ``ActionResult`` for tool calls, or
    ``None`` for a content-only finish). The think text is NOT here — it
    lives on the round's ``think_result`` (``ThinkResult.step_content``).

    Used by: _amphibious_automa.py (_run_action_call act-result envelope)
    """
    model_config = ConfigDict(extra="forbid")

    result: Optional[Any] = None


class OTARecord(BaseModel):
    """One OTA (observe-think-act) round of the small loop.

    INVARIANT: one round == one think-decision == one ``_execute`` == one
    ``ActionResult`` (the N tool calls of a single decision aggregate into
    ONE ``action_result``).

    ``model_config`` uses ``extra="allow"`` so that user hooks (e.g.
    ``before_action``/``after_action``) can fold custom per-round fields
    onto the current round — for example a ``permission_result`` — without
    subclassing ``OTARecord``.

    Used by: _context.py (OTAContext.ota_record)
    """
    model_config = ConfigDict(extra="allow")

    observation_result: Optional[Any] = None
    think_result: Optional[Any] = None
    action_result: Optional[Any] = None


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


def generate_tool_call_id() -> str:
    """Generate a local id for tool calls that arrive without provider ids."""
    return f"call_{uuid.uuid4().hex[:25]}"


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
    call_id: str = Field(
        default_factory=generate_tool_call_id,
        description="Provider tool-call id, or a framework-generated local id.",
    )
    tool: str = Field(description="Name of the tool to call")
    tool_arguments: List[ToolArgument] = Field(
        description="Arguments as list of name-value pairs, e.g., [{name: 'city', value: 'Beijing'}]"
    )

    @field_validator("call_id", mode="before")
    @classmethod
    def ensure_call_id(cls, v: Any) -> str:
        return str(v) if v not in (None, "") else generate_tool_call_id()


class ThinkResult(BaseModel):
    """The worker's assembled decision: what to say + which tools to call.

    A flat structure — ``step_content`` (the think text / final answer, or a
    serialized structured result) plus ``tool_calls``. There is no explicit
    ``finish`` flag: a decision with NO ``tool_calls`` IS the finish (the
    agent stops calling tools). Both the ``yield ThinkUnit(...)`` result and
    the run's ``final_answer`` are this decision's ``step_content``.

    Used by: _cognitive_worker.py (assembled by _assemble_decision),
    _amphibious_automa.py (the act phase reads ``tool_calls``)
    """
    model_config = ConfigDict(extra="forbid")

    step_content: str = Field(
        default="",
        description="Description of what to do in this step, or your analysis/reasoning"
    )
    tool_calls: List[StepToolCall] = Field(
        default_factory=list,
        description="Tool calls to execute this step.",
    )

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
#                   fresh sub-run OTA context; agent generator
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


@dataclass(init=False)
class ActionCall:
    """Yielded by on_workflow() / hooks for deterministic single-tool execution.

    Each instance wraps exactly one tool call.

    The ``**kwargs`` constructor is the ergonomic form for hand-written
    workflow code, but it cannot express a tool whose own parameter is
    named ``tool_name`` or ``description`` — those names are claimed by
    the signature. When the tool arguments come from an external source
    (e.g. an MCP-bridged call, where the agent may pick any parameter
    names), use the collision-free ``ActionCall.from_tool_args(...)``.

    Used by: _amphibious_automa.py (state-machine driver)

    Usage::
        yield ActionCall("navigate_to", url="http://example.com")
        yield ActionCall("click_element_by_ref", description="Click submit", ref="e42")
        result = yield ActionCall("fill_field", name="user", value="john")
        ActionCall.from_tool_args("save", {"description": "draft"})  # collision-free
    """
    tool_name: str
    description: str
    tool_args: Dict[str, Any]

    def __init__(self, tool_name: str, *, description: str = "", **tool_args: Any) -> None:
        self.tool_name = tool_name
        self.description = description
        self.tool_args = tool_args


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

    Delegation is **fresh-instance** (isolation by construction): a new
    small-loop ``OTAContext`` is built for the sub-flow with ``goal`` as
    its ``user_input``, carrying the OTA context class's declared tools
    (``OTAContext.tool``). The sub-run owns its ``rounds``; the parent OTA
    context is restored (never mutated) when the agent exhausts. The
    big-loop knowledge context is shared (read-only) across parent and
    sub-run.

    EnterAgent hands the agent a sub-task (``goal``); it does not control
    *how it thinks*. For a single named cognitive step, use ``ThinkUnit``
    from inside ``on_agent``. Requires the class to override ``on_agent``.

    >>> yield EnterAgent(goal="Handle the login popup")
    """
    goal: str = ""


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
    associated ``ThinkUnitDescriptor`` through
    ``AmphibiousAutoma._run_think_unit``. Fields beyond ``name`` overlay
    the descriptor's defaults (``None`` = descriptor value). The
    ``asend()`` result is the finishing think's ``step_content`` (a
    ``str``).

    >>> result = yield ThinkUnit("main_think")
    >>> result = yield ThinkUnit("exec_think", until=lambda c: c.done, max_attempts=20)
    """
    name: str
    until: Optional[Callable[..., Union[bool, Awaitable[bool]]]] = None
    max_attempts: Optional[int] = None


@dataclass(frozen=True)
class ThinkAgent:
    """Yielded to invoke a class-level ``think_agent`` declaration by name.

    Unlike ``ThinkUnit`` (one in-process OTC cycle driven by a
    ``CognitiveWorker``), ``ThinkAgent`` drives an ``AgentWorker`` that
    hands the sub-goal off to an **external** agent (today: ``claude
    code``; add others by subclassing ``BaseAgent``). The external
    agent is bound to the parent's task tools via an in-process MCP
    server, so every tool call it makes is surfaced back as a decision
    and executed by ``_run_action_call`` — the parent's hooks fire
    normally.

    Fields beyond ``name`` overlay the descriptor's defaults (``None`` =
    descriptor value). CLI-level knobs (``allowed_builtin_tools`` /
    ``permission_mode`` / ``completion_timeout`` / …) live on the
    ``BaseAgent`` the ``AgentWorker`` wraps — analogous to how LLM /
    cognitive-policy knobs live on ``CognitiveWorker``, not on
    ``ThinkUnit``.

    The ``asend()`` result is the string the external agent passed to
    ``agent_done(result=...)``, or ``None`` if the agent exited
    without signalling.

    >>> class MyAutoma(AmphibiousAutoma[OTAContext, Context]):
    ...     write_article = think_agent(
    ...         AgentWorker(ClaudeCodeAgent(allowed_builtin_tools=["Write"])),
    ...     )
    ...     async def on_agent(self, ota_ctx):
    ...         result = yield ThinkAgent("write_article", goal="Write the article.")
    ...         yield RETURN(result)
    """
    name: str
    goal: Optional[str] = None
    expose_tools: Optional[List[str]] = None


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
        async def on_agent(self, ota_context, context=None):
            answer = yield ThinkUnit("main_think", max_attempts=20)
            yield RETURN(answer)   # ``answer`` is the finishing think's step_content
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

    One value per ``_record_<primitive>`` family on ``AmphibiousAutoma``
    (used by: _amphibious_automa.py — AgentTrace + the ``_record_*``
    methods).
    """
    TOOL_CALLS = "tool_calls"
    CONTENT_ONLY = "content_only"
    LLM_CALL = "llm_call"
    THINK_AGENT = "think_agent"
    HUMAN_CALL = "human_call"
    ENTER_AGENT = "enter_agent"


class RecordedToolCall(BaseModel):
    """A complete record of one tool invocation.

    Used by: _amphibious_automa.py (AgentTrace.build)
    """
    model_config = ConfigDict(extra="forbid")

    tool_id: Optional[str] = None
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
