import asyncio
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any, AsyncGenerator, Awaitable, Callable, ClassVar, Dict, FrozenSet, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union,
    get_args, get_origin
)

from pydantic import BaseModel

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._automa import RunningOptions
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model.types import Message, Role, ToolCall
from bridgic.core.model.protocols import StructuredOutput, ToolSelection
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec, FunctionToolSpec
from bridgic.core.utils._console import printer
from bridgic.amphibious._context import CognitiveContext, CognitiveTools, CognitiveSkills, CognitiveHistory, Exposure, LayeredExposure
from bridgic.amphibious._cognitive_worker import CognitiveWorker, _DELEGATE
from bridgic.amphibious._worker_runner import WorkerRunner
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
from bridgic.amphibious.builtin_tools.human.request_human import (
    build_request_human_tool,
)
from bridgic.amphibious._type import (
    RunMode,
    Step,
    StepToolCall,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    ThinkUnit,
    RETURN,
    ErrorStrategy,
    ActionStepResult,
    ActionResult,
    ToolResult,
    StepOutputType,
    TraceStep,
    RecordedToolCall,
    observation_fingerprint,
)


################################################################################################################
# Type Aliases
#
# Module-level type names and constants used throughout the framework.
# ``CognitiveContextT`` is the generic parameter that lets subclasses
# declare their own context type via ``AmphibiousAutoma[MyContext]``.
# ``_BUILTIN_TOOLS`` is the immutable tuple of built-in tools auto-injected
# into every agent's tool set during ``arun()``.
################################################################################################################

# Generic type for the agent's cognitive context, allowing users to define their own context classes.
CognitiveContextT = TypeVar("CognitiveContextT", bound=CognitiveContext)

# Built-in tools auto-injected into every AmphibiousAutoma agent's tool set.
# Sourced from ``builtin_tools.ALL_BUILTIN_TOOLS`` so that adding a new
# built-in tool only requires touching the ``builtin_tools`` package.
_BUILTIN_TOOLS: Tuple[ToolSpec, ...] = ALL_BUILTIN_TOOLS


################################################################################################################
# AgentTrace — flat execution path recorder
#
# Captures one ``TraceStep`` per observe-think-act cycle (CognitiveWorker
# runs) or per dispatched APICall (workflow yields). Optional capture —
# only active when ``arun(trace_running=True)`` is set. ``record_step``
# appends; ``build`` materialises the structured dict; ``save`` / ``load``
# persist to JSON.
################################################################################################################


class AgentTrace:
    """Flat trace recorder that captures each observe-think-act cycle.

    ``record_step()`` appends step data to the execution path.
    ``build()`` returns the collected steps as a structured dict.
    ``save()`` / ``load()`` provide JSON serialization.
    """

    def __init__(self):
        self._steps: List[dict] = []

    def record_step(self, step_data: dict) -> None:
        """Append a trace step to the execution path."""
        self._steps.append(step_data)

    def build(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return collected trace data as a structured dict."""
        steps = [
            TraceStep(
                name=s["name"],
                step_content=s.get("step_content", ""),
                tool_calls=[
                    RecordedToolCall(**tc) for tc in s.get("tool_calls", [])
                ],
                observation=s.get("observation"),
                observation_hash=s.get("observation_hash"),
                output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                structured_output=s.get("structured_output"),
                structured_output_class=s.get("structured_output_class"),
                llm_call_protocol=s.get("llm_call_protocol"),
            )
            for s in self._steps
        ]

        return {
            "steps": steps,
            "metadata": metadata or {},
        }

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Serialize the trace to a JSON file."""
        trace_data = self.build(metadata=metadata)
        serializable = self._to_serializable(trace_data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Deserialize a trace from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _to_serializable(self, data: Any) -> Any:
        """Recursively convert Pydantic models and enums to plain dicts/values."""
        from enum import Enum
        if isinstance(data, BaseModel):
            return self._to_serializable(data.model_dump())
        if isinstance(data, dict):
            return {k: self._to_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_serializable(item) for item in data]
        if isinstance(data, Enum):
            return data.value
        return data


################################################################################################################
# _AgentSnapshot — async context manager for scoped context mutation
#
# Internal helper used by ``snapshot()`` to save/restore field values and
# every ``LayeredExposure._revealed`` dict around a sub-task. Two modes:
# clear-all (default — sub-agent sees a fresh revealed view) or custom
# keep-list. EnterAgent dispatch always goes through this so the parent
# agent's revealed state is restored when the sub-task returns.
################################################################################################################

class _AgentSnapshot:
    """
    Async context manager for exception-safe temporary field overrides on a Context,
    with additional management of LayeredExposure._revealed state.

    Used internally by ``snapshot()`` context manager in AmphibiousAutoma.

    On enter:
    1. Saves original field values and applies overrides.
    2. Snapshots all LayeredExposure._revealed dicts.
    3. Manages _revealed according to the chosen mode.

    On exit:
    - Restores all field values.
    - Restores all _revealed dicts to their pre-enter state.

    Modes (keep_revealed parameter)
    --------------------------------
    None (default)  — Clear All: clears all _revealed on enter, restores on exit.
    dict            — Custom: {field_name: [indices]} specifying which items to keep.
    """

    def __init__(self, ctx, fields: Dict[str, Any], keep_revealed=None):
        self._ctx = ctx
        self._fields = fields
        self._keep_revealed = keep_revealed
        self._originals: Dict[str, Any] = {}
        self._saved_revealed: Dict[str, Dict[int, str]] = {}

    async def __aenter__(self):
        # 1. Save field values and apply overrides
        self._originals = {k: getattr(self._ctx, k) for k in self._fields}
        for k, v in self._fields.items():
            setattr(self._ctx, k, v)

        # 2. Save all LayeredExposure._revealed snapshots
        for fname, fval in self._ctx:
            if isinstance(fval, LayeredExposure):
                self._saved_revealed[fname] = dict(fval._revealed)

        # 3. Apply revealed management based on mode
        if self._keep_revealed is None:
            for _, fval in self._ctx:
                if isinstance(fval, LayeredExposure):
                    fval._revealed.clear()
        else:
            self._apply_filter(self._keep_revealed)

        return self._ctx

    async def __aexit__(self, *exc) -> None:
        # Restore field values
        for k, v in self._originals.items():
            setattr(self._ctx, k, v)
        # Restore _revealed state for all LayeredExposure fields
        for fname, fval in self._ctx:
            if isinstance(fval, LayeredExposure):
                fval._revealed.clear()
                if fname in self._saved_revealed:
                    fval._revealed.update(self._saved_revealed[fname])

    def _apply_filter(self, keep: Dict[str, List[int]]) -> None:
        """Remove revealed items not in the keep dict."""
        for fname, fval in self._ctx:
            if not isinstance(fval, LayeredExposure):
                continue
            allowed = set(keep.get(fname, []))
            to_remove = [idx for idx in fval._revealed if idx not in allowed]
            for idx in to_remove:
                del fval._revealed[idx]


################################################################################################################
# ThinkUnit — descriptor-based think-step declaration
#
# Class-level marker placed via the ``think_unit(...)`` factory and
# invoked by ``yield ThinkUnit("name")`` from inside ``on_agent``. Stores
# the worker template plus per-unit overlays (until / max_attempts /
# tools / skills / on_error / max_retries); the dispatcher clones a fresh
# CognitiveWorker instance per call for state isolation, or uses a
# WorkerRunner template directly.
#
# This section also defines the ``@human_channel`` decorator used to
# register named HumanCall handlers on agent subclasses.
################################################################################################################

class ThinkUnitDescriptor:
    """Class-level marker for a declared think unit.

    Both class-level (``MyAgent.main_think``) and instance-level
    (``self.main_think``) access return the descriptor itself. Invocation
    happens via ``yield ThinkUnit("main_think", ...)`` inside an
    async-generator template method; the dispatcher resolves the name
    against the class, picks up the descriptor, clones its worker
    template, and runs it through ``_run``.
    """

    def __init__(
        self,
        worker: Any,
        *,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: int = 1,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ):
        self._worker_template = worker
        self._until = until
        self._max_attempts = max_attempts
        self._tools = tools
        self._skills = skills
        self._on_error = on_error
        self._max_retries = max_retries

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "ThinkUnitDescriptor":
        # Both class- and instance-level access return the descriptor
        # itself. Invocation goes through ``yield ThinkUnit("name")``.
        return self

    @staticmethod
    def _clone_worker(template: CognitiveWorker) -> CognitiveWorker:
        """Clone a worker from its template for state isolation.

        Copies configuration (policies, output_schema, verbose settings) but
        creates a fresh instance with clean runtime state (tokens, time,
        GraphAutoma execution state). LLM is left as None — injected by
        the agent at runtime via ``_run()``.

        Used by ``_dispatch_call`` when handling a ``ThinkUnit`` yield.
        """
        clone = type(template)(
            llm=None,
            enable_rehearsal=template.enable_rehearsal,
            enable_reflection=template.enable_reflection,
            verbose=template._verbose,
            verbose_prompt=template._verbose_prompt,
            output_schema=template.output_schema,
        )
        return clone
    

def think_unit(
    worker: Any,
    *,
    until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
    max_attempts: int = 1,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 0,
) -> ThinkUnitDescriptor:
    """Declare a think unit, invoked via ``yield ThinkUnit(name)``.

    Factory function that returns a ``ThinkUnitDescriptor``. Use as a
    class variable::

        class MyAgent(AmphibiousAutoma[MyContext]):
            main_think = think_unit(
                CognitiveWorker.inline("Plan ONE immediate next step"),
                max_attempts=80,
                on_error=ErrorStrategy.RAISE,
            )

            async def on_agent(self, ctx):
                yield ThinkUnit("main_think")

    Parameters
    ----------
    worker : CognitiveWorker | WorkerRunner
        The worker template. For ``CognitiveWorker`` a fresh clone is
        created for each ``ThinkUnit`` (state isolation). For an
        external ``WorkerRunner`` implementation the template is used
        directly (the runner manages its own state). The
        ``until`` / ``max_attempts`` / ``tools`` / ``skills`` overlays
        apply to the CognitiveWorker path only.
    until : Optional callable
        Loop condition (CognitiveWorker only): repeats until this
        returns True or LLM signals finish.
    max_attempts : int
        Maximum execution attempts (default 1 = single shot).
    tools : Optional[List[str]]
        Tool filter (CognitiveWorker only): only these tools are visible.
    skills : Optional[List[str]]
        Skill filter (CognitiveWorker only): only these skills are visible.
    on_error : ErrorStrategy
        Error handling strategy (default: RAISE).
    max_retries : int
        Max retries for RETRY strategy.
    """
    return ThinkUnitDescriptor(
        worker,
        until=until,
        max_attempts=max_attempts,
        tools=tools,
        skills=skills,
        on_error=on_error,
        max_retries=max_retries,
    )


_HUMAN_CHANNEL_MARKER: str = "_human_channel_name"
def human_channel(arg: Any = None) -> Any:
    """Decorator that registers an async method as a human-input channel.

    Two usage forms::

        @human_channel("feishu")           # explicit channel name
        async def ask_feishu(self, prompt: str) -> str: ...

        @human_channel                     # bare — channel name = method name
        async def terminal(self, prompt: str) -> str: ...

    Channel handlers are *plain async methods returning ``str``*, not
    generators. They are leaf I/O operations and do not dispatch inner
    yields.

    The framework collects all decorated methods into a class-level
    ``_human_channels: Dict[str, str]`` registry (channel-name →
    method-name) inside ``AmphibiousAutoma.__init_subclass__``. At
    dispatch time, ``HumanCall(channel=...)`` is routed via this
    registry.

    Used by: AmphibiousAutoma (channel registry), HumanCall dispatch.
    """
    # Bare form: @human_channel (no parens) → arg is the method itself.
    if callable(arg) and not isinstance(arg, str):
        method = arg
        setattr(method, _HUMAN_CHANNEL_MARKER, method.__name__)
        return method

    # Parameterised form: @human_channel("name") or @human_channel()
    name: Optional[str] = arg

    def _decorator(method):
        setattr(method, _HUMAN_CHANNEL_MARKER, name or method.__name__)
        return method

    return _decorator


@dataclass
class _FlowState:
    """Mutable per-flow state for body-mode fallback bookkeeping.

    Created fresh per ``_drive_amphiflow`` invocation. Tracks the
    consecutive-failure counter that the state-machine driver uses to
    decide between step-level fallback (snapshot + agent + resume) and
    full fallback (close workflow + agent + end).

    Attributes
    ----------
    max_consecutive_fallbacks : int
        Step-failure threshold before full fallback to ``on_agent``.
    consecutive_failures : int, default 0
        Running count of consecutive atomic-Call failures, reset on success.
    step_index : int, default 0
        Running step counter (informational, surfaced in error messages).
    failed_steps : List[str]
        Accumulated descriptions of failed steps for diagnostic output.
    """
    max_consecutive_fallbacks: int
    consecutive_failures: int = 0
    step_index: int = 0
    failed_steps: List[str] = field(default_factory=list)


class _FallbackSlot:
    """Mailbox for a single step-level fallback's resolved value.

    Created fresh per step-level fallback by ``_drive_amphiflow``.
    Initialized with a benign default appropriate for the failed
    Call's expected return type (e.g. ``[]`` for ActionCall, ``""``
    for HumanCall). The agent can override the default by calling the
    auto-injected ``resolve_step_fallback`` tool, which closes over
    this slot and writes through ``set()``.

    On agent generator exhaustion, ``_drive_amphiflow`` reads
    ``self.value`` and asends it to the workflow generator's failed
    yield, resuming the workflow as if the original Call had
    returned that value.
    """
    __slots__ = ("value",)

    def __init__(self, default: Any) -> None:
        self.value = default

    def set(self, value: Any) -> None:
        self.value = value


################################################################################################################
# AmphibiousAutoma
################################################################################################################

class AmphibiousAutoma(GraphAutoma, Generic[CognitiveContextT]):
    """Base class for amphibious agents — dual-mode orchestration engine.

    Supports three execution modes:

    - **Agent mode** (``on_agent``): LLM-driven cognitive flow. Yields
      ``ThinkUnit`` to invoke named ``think_unit`` declarations.
    - **Workflow mode** (``on_workflow``): Deterministic flow. Yields
      ``ActionCall`` / ``HumanCall`` / ``LLMCall`` / ``EnterAgent``.
    - **Amphiflow mode** (``on_workflow`` + ``on_agent``): workflow-first
      with automatic agent fallback when a step fails.

    Subclasses define behavior by implementing ``on_agent()`` and/or
    ``on_workflow()``. Under ``RunMode.AUTO`` (the default) the runtime
    picks the mode from which template methods are overridden:

    - only ``on_agent`` overridden → ``RunMode.AGENT``
    - only ``on_workflow`` overridden → ``RunMode.WORKFLOW``
    - both overridden → ``RunMode.AMPHIFLOW``

    Parameters
    ----------
    llm : Optional[BaseLlm]
        Default LLM for workers and auxiliary tasks (e.g. history
        compression). Individual workers can specify their own LLM.
    name : Optional[str]
        Optional name for the agent instance.
    verbose : bool, default False
        Enable logging of execution summary (tokens, time).
    verbose_hook_calls : bool, default False
        Whether to emit dispatch logs for Calls yielded from hook-scope
        generators (``observation`` / ``before_action`` / ``after_action``).
        These are internal side-effects and would clutter the workflow
        narrative; suppressed by default. Flip to ``True`` to surface
        them when debugging a hook.

    Notes
    -----
    Yield-type ↔ scope rules:

    ===========  ============  ========  =====
    primitive    on_workflow   on_agent  hooks
    ===========  ============  ========  =====
    ActionCall   ✓             ✗         ✓
    HumanCall    ✓             ✗         ✓
    LLMCall      ✓             ✗         ✓
    EnterAgent   ✓             ✗         ✗
    ThinkUnit    ✗             ✓         ✗
    RETURN       ✓             ✓         ✓
    ===========  ============  ========  =====

    on_agent body is reserved for orchestrating ``ThinkUnit`` cycles —
    deterministic tool / HITL / direct-LLM operations belong in
    on_workflow or inside a worker hook.

    Examples
    --------
    >>> class MyAgent(AmphibiousAutoma[CognitiveContext]):
    ...     main_think = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=20)
    ...     async def on_agent(self, ctx: CognitiveContext):
    ...         yield ThinkUnit("main_think")
    ...
    >>> answer = await MyAgent(llm=llm).arun(goal="Complete the task", tools=[...])
    """

    ############################################################################
    # Class Attributes and Initialization
    #
    # Class-level state populated automatically by ``__init_subclass__``:
    # ``_context_class`` (resolved from the ``Generic[T]`` parameter) and
    # ``_human_channels`` (registry walked from the MRO so subclass
    # overrides win over parent declarations). Subclasses can also
    # override the ``builtin_tools`` filter.
    ############################################################################

    _context_class: Optional[Type[CognitiveContext]] = None

    #: Filter applied to ``_BUILTIN_TOOLS`` during ``arun()`` injection.
    #: ``None`` (default) injects every built-in tool. A ``frozenset`` of
    #: tool names restricts injection to that subset; an empty frozenset
    #: opts out entirely. ``arun(builtin_tools=...)`` overrides at runtime.
    builtin_tools: ClassVar[Optional[FrozenSet[str]]] = None

    #: ``@human_channel``-decorated registry, populated by
    #: ``__init_subclass__``. Maps channel-name → method-name. Empty on
    #: the base class; subclasses inherit and may add or override.
    _human_channels: ClassVar[Dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Per-subclass initialisation.

        Two responsibilities:

        1. Extract the ``CognitiveContext`` type from the generic
           parameter so ``cls._context_class`` is set.
        2. Build the ``cls._human_channels`` registry by walking the MRO
           and collecting every method tagged via ``@human_channel``.
           Subclass overrides win over parent declarations.
        """
        super().__init_subclass__(**kwargs)
        cls._detect_context_class()
        cls._build_human_channel_registry()

    @classmethod
    def _detect_context_class(cls) -> None:
        """Resolve ``cls._context_class`` from the ``Generic[T]`` parameter."""
        for base in getattr(cls, "__orig_bases__", []):
            origin = get_origin(base)
            if origin is not None:
                args = get_args(base)
                if args:
                    context_type = args[0]
                    if isinstance(context_type, type) and issubclass(context_type, CognitiveContext):
                        cls._context_class = context_type
                        return
        for base in cls.__bases__:
            if hasattr(base, "_context_class") and base._context_class is not None:
                cls._context_class = base._context_class
                break

        if cls._context_class is None or not issubclass(cls._context_class, CognitiveContext):
            raise TypeError(
                f"{cls.__name__} must specify a CognitiveContext type via generic parameter, "
                f"e.g., class {cls.__name__}(AmphibiousAutoma[MyContext])"
            )

    @classmethod
    def _build_human_channel_registry(cls) -> None:
        """Walk MRO bottom-up so subclass overrides win, populate registry."""
        registry: Dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            for attr_name, attr in vars(klass).items():
                channel_name = getattr(attr, _HUMAN_CHANNEL_MARKER, None)
                if channel_name is not None:
                    registry[channel_name] = attr_name
        cls._human_channels = registry

    ############################################################################
    # Instance Attributes and Initialization
    #
    # Per-instance runtime state set up in ``__init__``: the LLM, the
    # current cognitive context, the optional trace recorder, the final
    # answer, usage stats (tokens / time), and the read-before-modify
    # tracker shared with the filesystem built-in tools. Three read-only
    # properties (``llm`` / ``context`` / ``final_answer``) expose the
    # most common reads.
    ############################################################################

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
        verbose_hook_calls: bool = False,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        self._llm = llm
        self._current_context: Optional[CognitiveContextT] = None
        self._verbose = verbose
        # Hook-scope dispatch logs are suppressed by default so the visible
        # log focuses on workflow narrative. Flip ``verbose_hook_calls`` to
        # ``True`` to surface them while debugging a hook generator.
        self._verbose_hook_calls = verbose_hook_calls

        # Trace capture
        self._agent_trace: Optional[AgentTrace] = None

        # Final answer — auto-captured from the finishing step or
        # explicitly set by yielding ``RETURN(value)`` from a
        # top-level template method.
        self._final_answer: Optional[str] = None

        # Usage stats (reset per arun call)
        self.spent_tokens: int = 0
        self.spent_time: float = 0.0

        # Per-agent read-before-modify tracker shared with the filesystem
        # built-in tools (read_file/write_file/edit_file). Maps absolute
        # file path → mtime at the time of the last read. Reset per arun call.
        self._read_tracker: Dict[str, float] = {}

    @property
    def llm(self) -> Optional[Any]:
        """Access the agent's default LLM."""
        return self._llm

    @property
    def context(self) -> Optional[CognitiveContextT]:
        """Access the current context."""
        return self._current_context

    @property
    def final_answer(self) -> Optional[str]:
        """The final answer produced by the last ``arun()`` call.

        Automatically captured from the ``step_content`` of the finishing
        step (agent mode) or the last executed step (workflow mode).
        Top-level template-method generators may override the auto-captured
        value by yielding ``RETURN(value)``.
        """
        return self._final_answer

    ############################################################################
    # Template methods (override by user to customize the behavior)
    #
    # All template methods may be written as async generators that yield
    # framework primitives, OR as plain async coroutines. The dispatcher
    # supports both forms. New code should prefer the yield form.
    #
    # Yield-type ↔ scope rules:
    #   ActionCall / HumanCall / LLMCall — on_workflow + hooks (NOT on_agent)
    #   EnterAgent                       — on_workflow only
    #   ThinkUnit                        — on_agent only
    #   RETURN                           — any scope
    ############################################################################
    async def observation(self, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent-level default observation, shared across all workers.

        Called before each thinking phase. Workers can enhance this via
        their own ``observation()`` method, which delegates here when
        it returns ``_DELEGATE`` or ``None``.

        Parameters
        ----------
        ctx : CognitiveContextT
            The current cognitive context.

        Yields
        ------
        ActionCall | HumanCall | LLMCall | RETURN
            Hook scope — ``EnterAgent`` and ``ThinkUnit`` are rejected.
            Yield ``RETURN(text)`` to set ``ctx.observation`` for this
            cycle. Exhausting without ``RETURN`` (or yielding
            ``RETURN(None)``) **preserves** the previous
            ``ctx.observation`` instead of overwriting it — so
            ``after_action``-driven refresh patterns work without a
            dedicated passthrough override.

        Examples
        --------
        >>> async def observation(self, ctx):
        ...     snapshot = yield ActionCall("bash", command="bridgic-browser snapshot")
        ...     yield RETURN(snapshot[0].result)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_agent(self, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent mode: LLM-driven cognitive flow.

        Override this method to declare the agent's strategy. A subclass
        may override this, ``on_workflow``, or both; the default is a
        no-op so that subclasses which only implement ``on_workflow``
        remain instantiable.

        Parameters
        ----------
        ctx : CognitiveContextT
            The current cognitive context.

        Yields
        ------
        ThinkUnit | RETURN
            Agent scope — only ``ThinkUnit`` (named cognitive step) and
            ``RETURN`` (explicit final answer) are allowed. The atomic
            Calls ``ActionCall`` / ``HumanCall`` / ``LLMCall`` and the
            mode-switch ``EnterAgent`` are all rejected: on_agent body
            is reserved for orchestrating cognitive steps; deterministic
            tool / HITL / direct-LLM operations belong in on_workflow
            or inside a worker hook. The framework auto-captures the
            final answer from the finishing think step's
            ``step_content`` if no ``RETURN`` is yielded.

        Examples
        --------
        >>> async def on_agent(self, ctx):
        ...     yield ThinkUnit("main_think", max_attempts=20)
        ...     yield ThinkUnit("exec_think", until=lambda c: c.done)
        ...     yield RETURN(ctx.cognitive_history.get_all()[-1].content)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_workflow(self, ctx: CognitiveContextT) -> AsyncGenerator[Union[ActionCall, HumanCall, EnterAgent, LLMCall], None]:
        """Workflow mode: deterministic flow as an async generator.

        Override this method to declare a deterministic workflow. When
        overridden, ``arun()`` automatically routes to workflow mode
        instead of ``on_agent``.

        Parameters
        ----------
        ctx : CognitiveContextT
            The current cognitive context.

        Yields
        ------
        ActionCall | HumanCall | LLMCall | EnterAgent | RETURN
            Workflow scope — ``ThinkUnit`` is rejected. Use
            ``EnterAgent`` to enter an autonomous sub-flow, or
            ``LLMCall`` for a one-shot LLM call. Use
            ``result = yield ActionCall(...)`` to receive tool
            execution results via ``asend()``. The generator exhausting
            signals workflow completion — no finish signal needed.

        Examples
        --------
        >>> async def on_workflow(self, ctx):
        ...     yield ActionCall("navigate_to", url="http://example.com")
        ...     result = yield ActionCall("click_element_by_ref", ref="42")
        ...     summary = yield LLMCall.chat("Summarize the page in one line.")
        ...     yield EnterAgent(goal="Handle complex case")
        """
        if False:  # pragma: no cover — makes this a proper async generator stub
            yield

    async def before_action(
        self,
        decision_result: Any,
        ctx: CognitiveContextT,
    ) -> AsyncGenerator[Any, Any]:
        """Agent-level before_action hook, shared across all workers.

        Called when a worker's ``before_action()`` returns ``_DELEGATE``
        (or ``None``, which is treated identically). Override to
        intercept and modify tool calls at the agent level.

        Parameters
        ----------
        decision_result : Any
            The pending action decision (``List[Tuple[ToolCall,
            ToolSpec]]`` for tool-call output, or a typed Pydantic
            instance for structured output).
        ctx : CognitiveContextT
            The current cognitive context.

        Yields
        ------
        ActionCall | HumanCall | LLMCall | RETURN
            Hook scope — ``EnterAgent`` and ``ThinkUnit`` are rejected.
            Yield ``RETURN(modified_decision)`` to override the decision.
            Exhausting without RETURN (or returning ``None`` from a
            coroutine override) is treated as passthrough — the
            original ``decision_result`` is preserved.

        Examples
        --------
        >>> async def before_action(self, decision_result, ctx):
        ...     adjusted = sanitize(decision_result)
        ...     yield RETURN(adjusted)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def action_tool_call(self, tool_list: List[Tuple[ToolCall, ToolSpec]], context: CognitiveContextT) -> ActionResult:
        """
        Execute tool calls concurrently and collect results.

        Override this method to customize tool execution behavior
        (e.g., sequential execution, rate limiting, sandboxing).

        Parameters
        ----------
        tool_list : List[Tuple[ToolCall, ToolSpec]]
            Matched tool call / spec pairs to execute.
        context : CognitiveContextT
            The current cognitive context.

        Returns
        -------
        ActionResult
            Aggregated results with per-tool success/failure status.
        """

        async def _run_one(tool_call: ToolCall, tool_spec: ToolSpec) -> ActionStepResult:
            tool_worker = tool_spec.create_worker()
            sandbox = ConcurrentAutoma()
            worker_key = f"tool_{tool_call.name}_{tool_call.id}"
            sandbox.add_worker(
                key=worker_key,
                worker=tool_worker,
                args_mapping_rule=ArgsMappingRule.UNPACK,
            )
            try:
                results = await sandbox.arun(InOrder([tool_call.arguments]))
                result = results[0] if results else None
                return ActionStepResult(
                    tool_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_arguments=tool_call.arguments,
                    tool_result=result,
                    success=True,
                )
            except Exception as e:
                return ActionStepResult(
                    tool_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_arguments=tool_call.arguments,
                    tool_result=None,
                    success=False,
                    error=str(e),
                )

        step_results = await asyncio.gather(
            *(_run_one(tc, ts) for tc, ts in tool_list)
        )
        return ActionResult(results=list(step_results))

    async def action_custom_output(self, decision_result: Any, context: CognitiveContextT) -> Any:
        """
        Handle structured output from a worker with ``output_schema`` set.

        Called instead of ``action_tool_call()`` when the worker produces
        a typed Pydantic instance (via ``output_schema``) rather than tool calls.
        Override to post-process or validate structured output.

        Parameters
        ----------
        decision_result : Any
            The structured output instance produced by the worker.
        context : CognitiveContextT
            The current cognitive context.

        Returns
        -------
        Any
            The (optionally processed) result to store in execution history.

            Returning ``None`` (e.g. an empty ``pass`` override) is treated
            as passthrough — the original ``decision_result`` is preserved
            so that stub overrides do not silently drop the typed output.
        """
        return decision_result

    async def after_action(self, step_result: Any, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent-level after_action hook.

        Called after action execution and before the result is returned.
        Override to update custom context fields or trigger follow-up
        primitives based on tool results.

        Parameters
        ----------
        step_result : Any
            The just-executed Step (as returned by ``_action``).
        ctx : CognitiveContextT
            The current cognitive context.

        Yields
        ------
        ActionCall | HumanCall | LLMCall
            Hook scope — ``EnterAgent`` and ``ThinkUnit`` are rejected.
            ``RETURN`` is unused here; the hook's "return value" is
            ignored by the framework.

        Examples
        --------
        >>> async def after_action(self, step_result, ctx):
        ...     summary = yield LLMCall.chat(f"Summarize: {step_result}")
        ...     ctx.last_summary = summary
        """
        if False:  # pragma: no cover — async generator stub
            yield

    ############################################################################
    # Core methods
    #
    # The dispatcher and the worker runner — the framework's two engines.
    #
    # Dispatcher:
    # * ``_drive_amphiflow``    — state-machine driver for AMPHIFLOW mode.
    #                             Holds the workflow generator and creates
    #                             fresh on_agent generators on EnterAgent /
    #                             step-level fallback. Workflow is the entry
    #                             mode; on_agent generator exhaustion implicitly
    #                             resumes the workflow generator.
    # * ``_invoke_template``    — single-generator driver. Used for hooks
    #                             (observation / before_action / after_action),
    #                             for AGENT-mode body, for WORKFLOW-mode body,
    #                             and recursively by EnterAgent /
    #                             step-level fallback in the state machine.
    # * ``_dispatch_call``      — per-yield handler. Validates scope, dispatches
    #                             by isinstance. Failures from atomic Calls
    #                             propagate; fallback bookkeeping lives in the
    #                             state-machine driver.
    #
    # Helpers: ``_dispatch_human_channel`` / ``_stdin_human_fallback`` (HumanCall
    # routing); ``_run_llm_call`` (LLMCall protocol dispatch);
    # ``_build_enter_agent_snapshot`` / ``_describe_call`` /
    # ``_build_fallback_goal`` (state-machine helpers).
    #
    # Worker runner (``_run`` / ``_run_once`` / ``_action``): drives a
    # CognitiveWorker through one or more observe-think-act cycles. Used
    # by ThinkUnit dispatch and any user code that opts into a coroutine-form
    # ``on_agent``.
    #
    # ``snapshot`` / ``_phase_context`` provide the scoped-context
    # mechanism that EnterAgent and step-level fallback build on.
    ############################################################################
    async def _drive_amphiflow(
        self,
        ctx: CognitiveContextT,
        max_consecutive_fallbacks: int,
    ) -> Any:
        """Peer state-machine driver for AMPHIFLOW mode.

        Workflow is the entry mode. The driver holds two generator
        slots — ``workflow_gen`` (always alive until exhaustion or full
        fallback) and ``agent_gen`` (lazy-created on EnterAgent or
        step-level fallback, disposed on exhaustion). A single while
        loop alternates between them.

        Mode transitions
        ----------------
        * **EnterAgent** (yielded from on_workflow): suspend
          workflow_gen, push a snapshot via ``AsyncExitStack``,
          create a fresh agent_gen, switch ``current = "agent"``.
        * **Agent gen exhaustion** (StopAsyncIteration): pop the
          snapshot, dispose agent_gen. If a fallback slot is active,
          asend ``slot.value`` to the suspended workflow_gen. Switch
          back to ``current = "workflow"``.
        * **Atomic-Call failure in workflow + counter < threshold**:
          synthesise an EnterAgent — push snapshot with fallback goal
          AND an injected ``resolve_step_fallback`` tool bound to a
          fresh slot. Same path as user-yielded EnterAgent from here.
        * **Atomic-Call failure in workflow + counter >= threshold**,
          OR **workflow generator-internal exception**: close
          workflow_gen entirely and drive on_agent linearly via
          ``_invoke_template`` (full fallback — workflow does not
          resume).

        Parameters
        ----------
        ctx : CognitiveContextT
            The current cognitive context.
        max_consecutive_fallbacks : int
            Atomic-Call step-failure threshold before full fallback.

        Returns
        -------
        Any
            The value captured from a ``RETURN(value)`` yield, or
            ``None`` if the run ends without RETURN.
        """
        workflow_obj = self.on_workflow(ctx)
        if not inspect.isasyncgen(workflow_obj):
            # Coroutine-form on_workflow: no yields, so no fallback
            # machinery applies. Just await it; treat the coroutine's
            # return value as RETURN-equivalent (None means no override).
            return await workflow_obj
        workflow_gen = workflow_obj
        agent_gen: Optional[Any] = None
        agent_mode_stack: Optional[AsyncExitStack] = None
        fallback_slot: Optional[_FallbackSlot] = None

        workflow_send: Any = None
        agent_send: Any = None
        state = _FlowState(max_consecutive_fallbacks=max_consecutive_fallbacks)
        return_value: Any = None

        try:
            while True:
                # Pick the active generator slot.
                if agent_gen is not None:
                    gen = agent_gen
                    send = agent_send
                    agent_send = None
                    scope = "agent"
                else:
                    gen = workflow_gen
                    send = workflow_send
                    workflow_send = None
                    scope = "workflow"

                # Advance the chosen generator.
                try:
                    if send is None:
                        item = await gen.__anext__()
                    else:
                        item = await gen.asend(send)
                except StopAsyncIteration:
                    if scope == "agent":
                        # Implicit "switch back to workflow".
                        if fallback_slot is not None:
                            workflow_send = fallback_slot.value
                            fallback_slot = None
                        if agent_mode_stack is not None:
                            await agent_mode_stack.__aexit__(None, None, None)
                            agent_mode_stack = None
                        agent_gen = None
                        continue
                    # workflow exhausted naturally → run done.
                    break
                except Exception as e:
                    if scope == "agent":
                        # Agent body raised — snapshot must be popped, then
                        # propagate. We do not auto-escalate agent failures.
                        if agent_mode_stack is not None:
                            await agent_mode_stack.__aexit__(
                                type(e), e, e.__traceback__,
                            )
                            agent_mode_stack = None
                        agent_gen = None
                        fallback_slot = None
                        raise
                    # Workflow generator-internal error → full fallback.
                    if not self._has_agent():
                        raise RuntimeError(
                            f"Generator raised at step {state.step_index}: {e}\n"
                            f"on_agent() is not overridden, cannot fall back."
                        ) from e
                    self._log(
                        "Dispatch",
                        f"[ERROR] Generator code raised at step {state.step_index}: {e} — "
                        f"falling back to on_agent().",
                        color="red",
                    )
                    # workflow_gen is already dead from the raise; drive agent
                    # linearly with the original context.
                    workflow_gen = None
                    agent_return = await self._invoke_template(
                        self.on_agent(ctx), ctx, scope="agent",
                    )
                    if agent_return is not None:
                        return_value = agent_return
                    return return_value

                # Successfully advanced — handle the yielded item.

                # RETURN: terminates the entire run regardless of which mode
                # produced it (RETURN's only role is "set final answer + end").
                if isinstance(item, RETURN):
                    return_value = item.value
                    preview = str(item.value)
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    self._log("Dispatch", f"RETURN: {preview}", color="cyan")
                    break

                # EnterAgent: user-yielded mode switch (workflow → agent).
                if isinstance(item, EnterAgent):
                    if scope != "workflow":
                        raise RuntimeError(
                            f"EnterAgent(goal={item.goal!r}) is only valid inside "
                            f"on_workflow (scope='workflow'); got scope={scope!r}. "
                            "EnterAgent is the deterministic→autonomous mode "
                            "switch; once you are inside on_agent, keep "
                            "thinking via ThinkUnit instead."
                        )
                    if not self._has_agent():
                        raise RuntimeError(
                            f"EnterAgent(goal={item.goal!r}) requires an "
                            "on_agent() override on the agent class."
                        )
                    self._log(
                        "Dispatch",
                        f"EnterAgent(goal={item.goal!r}) → switching to on_agent",
                        color="cyan",
                    )
                    snapshot_kwargs = self._build_enter_agent_snapshot(item, ctx)
                    agent_obj = self.on_agent(ctx)
                    if not inspect.isasyncgen(agent_obj):
                        # Coroutine-form on_agent: state-machine interleaving
                        # is moot (no per-yield interleaving inside a coro).
                        # Drive it inline under the snapshot.
                        async with self.snapshot(**snapshot_kwargs):
                            coro_return = await agent_obj
                        if coro_return is not None:
                            return_value = coro_return
                            break  # terminate run
                        # Natural exhaustion → resume workflow with asend(None).
                        workflow_send = None
                        continue
                    # Generator-form on_agent: set up state-machine slot.
                    agent_mode_stack = AsyncExitStack()
                    await agent_mode_stack.__aenter__()
                    await agent_mode_stack.enter_async_context(
                        self.snapshot(**snapshot_kwargs),
                    )
                    agent_gen = agent_obj
                    # No fallback slot for user-yielded EnterAgent; workflow
                    # resumes via asend(None) when agent exhausts.
                    continue

                # Other yields → _dispatch_call, with fallback wrapping for
                # the three atomic Calls in workflow scope.
                is_atomic_call = isinstance(item, (ActionCall, HumanCall, LLMCall))
                try:
                    result = await self._dispatch_call(item, ctx, scope=scope)
                    if scope == "agent":
                        agent_send = result
                    else:
                        workflow_send = result
                        if is_atomic_call:
                            state.consecutive_failures = 0
                            state.step_index += 1
                except Exception as e:
                    if scope == "agent" or not is_atomic_call:
                        # ThinkUnit failure / non-atomic error / atomic-Call
                        # error from within agent scope: propagate.
                        raise

                    # Atomic-Call failure in workflow scope → fallback.
                    state.consecutive_failures += 1
                    state.step_index += 1
                    item_label = self._describe_call(item)
                    state.failed_steps.append(
                        f"Step {state.step_index}: {item_label} — {e}"
                    )
                    self._log(
                        "Dispatch",
                        f"[ERROR] Step {state.step_index} failed "
                        f"({state.consecutive_failures}/{state.max_consecutive_fallbacks}): {e}",
                        color="red",
                    )

                    if state.consecutive_failures >= state.max_consecutive_fallbacks:
                        # Full fallback.
                        if not self._has_agent():
                            raise RuntimeError(
                                f"Workflow degradation failed: consecutive "
                                f"failures reached {state.max_consecutive_fallbacks}.\n"
                                f"Failed steps:\n" + "\n".join(state.failed_steps)
                            ) from e
                        self._log(
                            "Dispatch",
                            "[ERROR] threshold breached → full fallback to on_agent",
                            color="red",
                        )
                        # Close workflow_gen best-effort; do not let an aclose
                        # exception block the fallback we are about to run.
                        try:
                            await workflow_gen.aclose()
                        except Exception:
                            pass
                        workflow_gen = None
                        agent_return = await self._invoke_template(
                            self.on_agent(ctx), ctx, scope="agent",
                        )
                        if agent_return is not None:
                            return_value = agent_return
                        return return_value

                    # Step-level fallback: synthesise an EnterAgent with a
                    # fallback-goal snapshot AND an injected
                    # resolve_step_fallback tool bound to a fresh slot.
                    if not self._has_agent():
                        raise  # No on_agent — re-raise the original failure.
                    fallback_slot = self._make_fallback_slot(item)
                    resolve_tool = self._make_resolve_tool(fallback_slot, item)
                    fallback_goal = self._build_fallback_goal(
                        item, item_label, e, state,
                    )
                    augmented_tools = CognitiveTools()
                    for t in ctx.tools.get_all():
                        augmented_tools.add(t)
                    augmented_tools.add(resolve_tool)
                    self._log(
                        "Dispatch",
                        f"Step-level fallback to on_agent for: {item_label}",
                        color="yellow",
                    )
                    agent_obj = self.on_agent(ctx)
                    if not inspect.isasyncgen(agent_obj):
                        # Coroutine-form on_agent: drive inline under snapshot.
                        async with self.snapshot(goal=fallback_goal, tools=augmented_tools):
                            coro_return = await agent_obj
                        if coro_return is not None:
                            return_value = coro_return
                            fallback_slot = None
                            break  # terminate run
                        # Natural exhaustion → asend slot.value to workflow.
                        workflow_send = fallback_slot.value
                        fallback_slot = None
                        continue
                    # Generator-form: set up state-machine slot.
                    agent_mode_stack = AsyncExitStack()
                    await agent_mode_stack.__aenter__()
                    await agent_mode_stack.enter_async_context(
                        self.snapshot(goal=fallback_goal, tools=augmented_tools),
                    )
                    agent_gen = agent_obj
                    # workflow_send stays unset; once agent_gen exhausts, the
                    # StopAsyncIteration branch will populate workflow_send
                    # with fallback_slot.value.
                    continue
        finally:
            # Cleanup order: close suspended generators first (so their
            # finally blocks see the snapshotted ctx, matching the view
            # they had during execution), then roll back the snapshot,
            # then close workflow_gen. All are best-effort — never mask
            # the primary control-flow exception.
            if agent_gen is not None:
                try:
                    await agent_gen.aclose()
                except Exception:
                    pass
            if agent_mode_stack is not None:
                try:
                    await agent_mode_stack.__aexit__(None, None, None)
                except Exception:
                    pass
            if workflow_gen is not None:
                try:
                    await workflow_gen.aclose()
                except Exception:
                    pass

        return return_value

    def _build_enter_agent_snapshot(
        self,
        item: EnterAgent,
        ctx: CognitiveContextT,
    ) -> Dict[str, Any]:
        """Build snapshot kwargs for an EnterAgent transition.

        Maps EnterAgent fields onto ``snapshot()`` overrides:
        ``goal`` and ``history`` directly; ``tools`` / ``skills`` filter
        the ctx surface by name.
        """
        history = item.history if item.history is not None else CognitiveHistory()
        snapshot_kwargs: Dict[str, Any] = {
            "goal": item.goal,
            "cognitive_history": history,
        }
        if item.tools is not None:
            allowed = set(item.tools)
            filtered_tools = CognitiveTools()
            for tool in ctx.tools.get_all():
                if tool.tool_name in allowed:
                    filtered_tools.add(tool)
            snapshot_kwargs["tools"] = filtered_tools
        if item.skills is not None:
            allowed = set(item.skills)
            filtered_skills = CognitiveSkills()
            for skill in ctx.skills.get_all():
                if skill.name in allowed:
                    filtered_skills.add(skill)
            snapshot_kwargs["skills"] = filtered_skills
        return snapshot_kwargs

    @staticmethod
    def _describe_call(item: Any) -> str:
        """One-line description of an atomic Call for logs / fallback goals."""
        if isinstance(item, ActionCall):
            return f"ActionCall(tool_name={item.tool_name!r})"
        if isinstance(item, HumanCall):
            channel = item.channel or "<default>"
            return f"HumanCall(channel={channel!r})"
        if isinstance(item, LLMCall):
            return f"LLMCall(protocol={item.protocol!r})"
        return type(item).__name__

    @staticmethod
    def _make_fallback_slot(item: Any) -> _FallbackSlot:
        """Create a fallback slot pre-loaded with a benign default value.

        The default is what the workflow's failed yield will receive if
        the agent does not call the auto-injected
        ``resolve_step_fallback`` tool. Defaults are chosen so a "void"
        atomic Call (one whose return value the workflow does not use)
        can be left alone without blowing up downstream code:

        ============================  ====================================
        Failed Call                   Default slot value
        ============================  ====================================
        ActionCall                    one ToolResult with result=None
        HumanCall                     ""
        LLMCall(protocol="chat")      ""
        LLMCall("structure_output")   None
        LLMCall("tool_selector")      ([], None)
        ============================  ====================================
        """
        if isinstance(item, ActionCall):
            default: Any = [
                ToolResult(
                    tool_name=item.tool_name,
                    tool_arguments=dict(item.tool_args),
                    result=None,
                    success=True,
                )
            ]
        elif isinstance(item, HumanCall):
            default = ""
        elif isinstance(item, LLMCall):
            if item.protocol == "chat":
                default = ""
            elif item.protocol == "tool_selector":
                default = ([], None)
            else:
                default = None
        else:
            default = None
        return _FallbackSlot(default)

    def _make_resolve_tool(
        self,
        slot: _FallbackSlot,
        item: Any,
    ) -> FunctionToolSpec:
        """Build a ``resolve_step_fallback`` tool bound to ``slot``.

        Each step-level fallback gets a fresh tool instance (closure
        captures ``slot`` and ``item``) with a signature tuned to the
        failed Call's expected return type. The tool overwrites the
        slot's default; if the agent never calls it, the default
        applies.

        Tool-name collisions with user tools are unlikely in practice
        — the ``resolve_step_fallback`` name is reserved for this
        framework purpose. The tool is only present in ``ctx.tools``
        for the duration of one step-level fallback (snapshot scope).
        """
        if isinstance(item, ActionCall):
            tool_name = item.tool_name
            tool_args = dict(item.tool_args)

            async def resolve_step_fallback(result: Any) -> str:
                """Submit the recovered result for the failed workflow step.

                Call this once when you have produced the value the
                failed step should have returned. The workflow will
                resume with this value as if the original step had
                succeeded.

                Parameters
                ----------
                result : Any
                    The result the failed step should have produced
                    (whatever type its tool would normally return).
                """
                slot.set([
                    ToolResult(
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        result=result,
                        success=True,
                    )
                ])
                return "Result submitted; workflow will resume after you finish."

            return FunctionToolSpec.from_raw(resolve_step_fallback)

        if isinstance(item, HumanCall):
            async def resolve_step_fallback(response: str) -> str:
                """Submit the human response for the failed step.

                Call this once with the response the human would have
                given. The workflow will resume with this string as the
                HumanCall's return value.

                Parameters
                ----------
                response : str
                    The response text to feed back to the workflow.
                """
                slot.set(response)
                return "Response submitted; workflow will resume after you finish."

            return FunctionToolSpec.from_raw(resolve_step_fallback)

        if isinstance(item, LLMCall):
            protocol = item.protocol
            if protocol == "chat":
                async def resolve_step_fallback(text: str) -> str:
                    """Submit text for the failed LLMCall(chat).

                    Parameters
                    ----------
                    text : str
                        The text content the failed chat call should
                        have returned.
                    """
                    slot.set(text)
                    return "Text submitted; workflow will resume after you finish."

                return FunctionToolSpec.from_raw(resolve_step_fallback)
            if protocol == "structure_output":
                constraint = item.constraint

                async def resolve_step_fallback(value_json: str) -> str:
                    """Submit a JSON-encoded value for the failed structure_output LLMCall.

                    Parameters
                    ----------
                    value_json : str
                        JSON string conforming to the constraint's
                        schema. The framework will parse it into the
                        expected typed instance.
                    """
                    from bridgic.core.model.protocols import PydanticModel
                    if isinstance(constraint, PydanticModel):
                        slot.set(constraint.model.model_validate_json(value_json))
                    else:
                        slot.set(value_json)
                    return "Value submitted; workflow will resume after you finish."

                return FunctionToolSpec.from_raw(resolve_step_fallback)
            # tool_selector protocol — its return type is hard to
            # express cleanly via tool args. Inject a no-op tool that
            # just acknowledges; the slot keeps its ([], None) default.

            async def resolve_step_fallback() -> str:
                """No-op for failed LLMCall(tool_selector) — the
                framework will resume the workflow with the empty
                tool-selection default. Submit explicit recovery is
                not supported for this protocol."""
                return (
                    "Acknowledged. The workflow will resume with the "
                    "empty tool-selection default; explicit submission "
                    "is not supported for tool_selector failures."
                )

            return FunctionToolSpec.from_raw(resolve_step_fallback)

        raise ValueError(
            f"Cannot build resolve_step_fallback tool for item of type "
            f"{type(item).__name__}."
        )

    @staticmethod
    def _build_fallback_goal(
        item: Any,
        item_label: str,
        error: BaseException,
        state: _FlowState,
    ) -> str:
        """Goal text fed to on_agent on step-level fallback.

        Tells the agent (a) what failed and why, (b) that it should
        recover however it sees fit, and (c) how to feed the result
        back to the workflow via ``resolve_step_fallback``. The
        framework auto-injects that tool for the duration of this
        fallback; calling it once with the recovered value is the only
        way to override the slot's default value.

        Agent generator exhaustion (with or without calling
        ``resolve_step_fallback``) is the implicit "I am done with
        this scoped task" signal — the state-machine driver then
        asends ``slot.value`` to the workflow's failed yield and
        resumes the workflow.
        """
        if isinstance(item, ActionCall):
            intent = item.decision.step_content or item.tool_name
        else:
            intent = item_label
        return (
            f"[Workflow fallback] Step {state.step_index} failed.\n"
            f"Step intent: {intent}\n"
            f"Failed call: {item_label}\n"
            f"Error: {error}\n\n"
            f"Recover however you see fit. When you have produced the "
            f"value the failed step should have returned, call the "
            f"`resolve_step_fallback` tool with that value to feed it "
            f"back to the workflow. The workflow will resume with it "
            f"as if the original step had succeeded.\n\n"
            f"If the failed call's return value is not used downstream, "
            f"you may omit the `resolve_step_fallback` call — a benign "
            f"default value is then sent back. End your reasoning when "
            f"recovery is complete; the workflow will resume "
            f"automatically."
        )

    async def _dispatch_call(
        self,
        item: Any,
        ctx: CognitiveContextT,
        *,
        scope: str = "hook",
    ) -> Any:
        """Per-yield handler.

        Routes one yielded primitive by isinstance, validates yield-type
        ↔ scope compatibility, executes the call. Failures from atomic
        Calls (ActionCall / HumanCall / LLMCall) propagate — fallback
        bookkeeping lives in ``_drive_amphiflow``.

        Parameters
        ----------
        item : Any
            The yielded primitive (ActionCall / HumanCall / LLMCall /
            EnterAgent / ThinkUnit).
        ctx : CognitiveContextT
            The current cognitive context.
        scope : {"workflow", "agent", "hook"}, default "hook"
            Caller's flow scope, used for yield-type validation.

        Returns
        -------
        Any
            The value to send back to the generator via ``asend()``.
            Type depends on the yield type (e.g. ``str`` for HumanCall,
            ``List[ToolResult]`` for ActionCall).

        Raises
        ------
        RuntimeError
            ``EnterAgent`` yielded outside ``scope='workflow'``,
            ``ThinkUnit`` yielded outside ``scope='agent'``, or
            ``ActionCall`` / ``HumanCall`` / ``LLMCall`` yielded with
            ``scope='agent'``.
        TypeError
            ``item`` is not one of the recognised primitive types.

        Notes
        -----
        Yield-type ↔ scope rules:

        * ``ActionCall`` / ``HumanCall`` / ``LLMCall`` — allowed in
          ``"workflow"`` and ``"hook"`` scope. Rejected in
          ``"agent"`` (on_agent body should only orchestrate cognitive
          steps via ``ThinkUnit``; deterministic tool/HITL/LLM calls
          belong in on_workflow or in a worker's hook).
        * ``EnterAgent`` — only ``"workflow"``.
        * ``ThinkUnit`` — only ``"agent"``.

        ActionCall semantics differ by scope:

        * ``scope == "workflow"`` — full OTC wrap: agent-level
          ``observation`` runs first, then ``_action`` runs (which
          invokes ``before_action`` + tool + ``after_action``), and a
          workflow trace step is recorded.
        * ``scope == "hook"`` — raw tool execution via ``_action_raw``:
          no observation, no ``before_action`` / ``after_action`` wrap,
          no trace step. Hooks are imperative side-effect channels and
          are explicitly NOT OTC participants — re-entering the hook
          chain here would recurse into the same generator that yielded
          the ActionCall and blow the stack.

        EnterAgent dispatch here (called from ``_invoke_template`` in
        WORKFLOW mode without state-machine wrapping) recursively drives
        ``on_agent`` via ``_invoke_template``. In AMPHIFLOW mode the
        state-machine driver intercepts EnterAgent before this method is
        reached.
        """
        if isinstance(item, EnterAgent):
            if scope != "workflow":
                raise RuntimeError(
                    f"EnterAgent(goal={item.goal!r}) is only valid inside "
                    f"on_workflow (scope='workflow'); got scope={scope!r}. "
                    "EnterAgent is the deterministic→autonomous mode "
                    "switch; once you are inside on_agent, keep "
                    "thinking via ThinkUnit instead."
                )
            if not self._has_agent():
                raise RuntimeError(
                    f"EnterAgent(goal={item.goal!r}) requires an on_agent() "
                    "override on the agent class."
                )
            snapshot_kwargs = self._build_enter_agent_snapshot(item, ctx)
            self._log(
                "Dispatch",
                f"EnterAgent(goal={item.goal!r}) → switching to on_agent",
                color="cyan",
            )
            async with self.snapshot(**snapshot_kwargs):
                await self._invoke_template(
                    self.on_agent(ctx), ctx, scope="agent",
                )
            return None

        if isinstance(item, HumanCall):
            if scope == "agent":
                raise RuntimeError(
                    f"HumanCall(prompt={item.prompt!r}) is not allowed inside "
                    "on_agent — the agent should request human input via the "
                    "auto-injected ``request_human`` tool (called by the LLM "
                    "during a ThinkUnit), not by yielding HumanCall directly. "
                    "If you need a deterministic human step, put it in "
                    "on_workflow."
                )
            channel_label = item.channel or "<default>"
            self._log_call(
                scope,
                "Dispatch",
                f"Requesting human input via {channel_label}: {item.prompt}",
                color="yellow",
            )
            response = await self._dispatch_human_channel(item.prompt, channel=item.channel)
            self._log_call(
                scope,
                "Dispatch",
                f"Human responded: {response[:100]}{'...' if len(response) > 100 else ''}",
                color="green",
            )
            return response

        if isinstance(item, LLMCall):
            if scope == "agent":
                raise RuntimeError(
                    f"LLMCall(protocol={item.protocol!r}) is not allowed inside "
                    "on_agent — on_agent body is reserved for orchestrating "
                    "cognitive steps via ThinkUnit. Direct LLM calls belong "
                    "in on_workflow, in a hook, or inside a CognitiveWorker's "
                    "thinking() method."
                )
            prompt_preview = item.prompt[:80] + ("..." if len(item.prompt) > 80 else "")
            self._log_call(
                scope,
                "Dispatch",
                f"LLMCall protocol={item.protocol} prompt={prompt_preview}",
                color="cyan",
            )
            result = await self._run_llm_call(item)
            result_preview = str(result)
            if len(result_preview) > 120:
                result_preview = result_preview[:120] + "..."
            self._log_call(scope, "Dispatch", f"LLMCall result: {result_preview}", color="green")
            # Trace is workflow-narrative only — hook-scope LLM calls are
            # internal side-effects (mirror of the ActionCall trace policy).
            if scope != "hook":
                self._record_llm_call_trace(item, result)
            return result

        if isinstance(item, ThinkUnit):
            if scope != "agent":
                raise RuntimeError(
                    f"ThinkUnit(name={item.name!r}) is only valid inside "
                    f"on_agent (scope='agent'); got scope={scope!r}. "
                    "ThinkUnit references a class-level think_unit and "
                    "represents a step in the agent's cognitive strategy; "
                    "use LLMCall for a direct LLM invocation outside the "
                    "cognitive loop, or EnterAgent to enter an on_agent flow."
                )
            descriptor = getattr(type(self), item.name, None)
            if not isinstance(descriptor, ThinkUnitDescriptor):
                raise AttributeError(
                    f"ThinkUnit(name={item.name!r}) does not match any "
                    f"think_unit declaration on {type(self).__name__}."
                )
            template = descriptor._worker_template
            if isinstance(template, CognitiveWorker):
                worker = ThinkUnitDescriptor._clone_worker(template)
            else:
                # External WorkerRunner — use the template directly.
                worker = template
            until = item.until if item.until is not None else descriptor._until
            max_attempts = (
                item.max_attempts if item.max_attempts is not None
                else descriptor._max_attempts
            )
            tools = item.tools if item.tools is not None else descriptor._tools
            skills = item.skills if item.skills is not None else descriptor._skills
            self._log(
                "Dispatch",
                f"ThinkUnit name={item.name} max_attempts={max_attempts}",
                color="cyan",
            )
            await self._run(
                worker,
                until=until,
                max_attempts=max_attempts,
                tools=tools,
                skills=skills,
                on_error=descriptor._on_error,
                max_retries=descriptor._max_retries,
            )
            worker_output: Any = None
            if (
                isinstance(worker, CognitiveWorker)
                and worker.output_schema is not None
                and ctx is not None
                and len(ctx.cognitive_history) > 0
            ):
                last_step = ctx.cognitive_history.get_all()[-1]
                if last_step.result is not None:
                    worker_output = last_step.result
            return worker_output

        if isinstance(item, ActionCall):
            if scope == "agent":
                raise RuntimeError(
                    f"ActionCall(tool_name={item.tool_name!r}) is not allowed "
                    "inside on_agent — let the LLM decide tool calls inside a "
                    "ThinkUnit. If you need a deterministic tool call, put it "
                    "in on_workflow or in a worker hook (observation / "
                    "before_action / after_action)."
                )
            decision = item.decision

            # ActionCall dispatch splits by scope, reflecting the framework's
            # design philosophy: hooks (``observation`` / ``before_action`` /
            # ``after_action``) are NOT OTC participants — they are
            # imperative side-effect channels. Only ``on_workflow`` does an
            # OTC-wrapped dispatch.
            #
            # * ``scope == "hook"``     — raw tool execution. No
            #   observation, no ``before_action`` / ``after_action`` wrap,
            #   no workflow trace step. Re-entering the hook chain here
            #   would recurse into the same generator that yielded this
            #   ActionCall and blow the stack.
            # * ``scope == "workflow"`` — full OTC wrap (observation +
            #   ``_action`` which runs before/after_action, plus a
            #   workflow trace step).
            if scope == "hook":
                action_result = await self._action_raw(decision, ctx)
            else:
                # 1. Observe via agent-level hook.
                #
                # ``None`` (observation hook didn't yield ``RETURN`` — the
                # default stub case, or a deliberate "no fresh observation
                # here") is treated as "preserve the previous
                # ``ctx.observation``" so that snapshots written by
                # ``after_action`` survive across yields where
                # ``observation`` is intentionally a no-op.
                obs = await self._invoke_template(self.observation(ctx), ctx)
                if obs is not None:
                    ctx.observation = obs

                obs_str = str(obs) if obs is not None else "None"
                if len(obs_str) > 200:
                    obs_str = obs_str[:200] + "..."
                self._log("Observe", f"dispatch: {obs_str}", color="green")
                self._log("Think", f"dispatch: {decision.step_content}", color="cyan")

                # 2. Act with before/after_action wrap. Tool failures bubble
                # up as RuntimeError; the state-machine driver's fallback
                # wrapping catches them.
                action_result = await self._action(decision, ctx, _worker=None)

            inner = getattr(action_result, "result", None)
            if isinstance(inner, ActionResult):
                failed = [r for r in inner.results if not r.success]
                if failed:
                    errors = "; ".join(f"{r.tool_name}: {r.error}" for r in failed)
                    raise RuntimeError(
                        f"Tool execution failed for: "
                        f"{decision.step_content} — {errors}"
                    )

            if action_result is not None:
                formatted = action_result.model_dump_json(indent=4)
                log_label = "hook-dispatch" if scope == "hook" else "dispatch"
                self._log_call(scope, "Act", f"{log_label}:\n{formatted}", color="purple")

            # Trace recording is workflow-narrative only. Hook-scope tool
            # executions are internal side-effects and do not appear in the
            # workflow trace.
            if scope != "hook":
                self._record_trace_step(None, obs, decision, action_result, ctx)
            return self._build_tool_results(action_result)

        raise TypeError(
            f"Unknown yield type: {type(item).__name__}. Expected one of "
            "RETURN / ActionCall / HumanCall / LLMCall / EnterAgent / ThinkUnit."
        )

    async def _dispatch_human_channel(
        self, prompt: str, channel: Optional[str] = None
    ) -> str:
        """Resolve and invoke a registered ``@human_channel`` handler.

        Parameters
        ----------
        prompt : str
            The text shown to the human responder.
        channel : Optional[str], default None
            Name of the registered channel to use. ``None`` triggers
            implicit-default resolution (see Notes).

        Returns
        -------
        str
            The response text from the resolved channel handler.

        Raises
        ------
        RuntimeError
            ``channel=None`` was passed but multiple channels are
            registered, or an explicitly named ``channel`` is not in
            the registry.

        Notes
        -----
        Channel resolution rules:

        * Zero channels registered → use the framework's stdin fallback.
        * One channel registered → it becomes the implicit default.
        * Multiple channels registered → ``channel`` must be specified.
        * ``channel="name"`` provided → look up in ``cls._human_channels``
          and call the bound method.

        Used by the ``HumanCall`` dispatch branch and by
        ``request_human_tool`` (the LLM-facing built-in tool).
        """
        registry = type(self)._human_channels
        if not registry:
            return await self._stdin_human_fallback(prompt)
        if channel is None:
            if len(registry) != 1:
                raise RuntimeError(
                    "HumanCall(channel=None) is ambiguous: "
                    f"{len(registry)} channels registered "
                    f"({sorted(registry.keys())}). Specify channel='name' "
                    "explicitly."
                )
            channel = next(iter(registry))
        method_name = registry.get(channel)
        if method_name is None:
            raise RuntimeError(
                f"Unknown human channel: {channel!r}. "
                f"Registered: {sorted(registry.keys())}"
            )
        return await getattr(self, method_name)(prompt)

    async def _stdin_human_fallback(self, prompt: str) -> str:
        """Default human-input source when no ``@human_channel`` is registered.

        Reads a single line from stdin in a thread executor so the event
        loop is not blocked. Subclasses normally do not call this
        directly — register a ``@human_channel`` instead.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, input, f"\n[HumanInput] {prompt}\n> "
        )

    async def _run_llm_call(self, item: LLMCall) -> Any:
        """Execute an ``LLMCall`` against ``self._llm`` using the requested protocol.

        Builds the message list as ``history + [Message(prompt,
        role=USER)]``, then dispatches to one of three async LLM
        methods.

        Parameters
        ----------
        item : LLMCall
            The yielded LLMCall — carries ``protocol`` and any
            protocol-specific arguments (``constraint`` / ``tools``).

        Returns
        -------
        Any
            * ``protocol='chat'`` — ``str`` (the message content).
            * ``protocol='structure_output'`` — the typed instance from
              ``StructuredOutput.astructured_output``.
            * ``protocol='tool_selector'`` — the tuple from
              ``ToolSelection.aselect_tool``.

        Raises
        ------
        RuntimeError
            No LLM was passed to the AmphibiousAutoma constructor.
        TypeError
            The configured LLM does not implement the requested protocol.
        ValueError
            ``item.protocol`` is not one of the recognised values.

        Notes
        -----
        Used by ``_dispatch_call`` (LLMCall dispatch branch).
        """
        if self._llm is None:
            raise RuntimeError(
                f"LLMCall(protocol={item.protocol!r}) requires self._llm, "
                "but no LLM was passed to the AmphibiousAutoma constructor."
            )

        messages: List[Message] = []
        if item.history:
            messages.extend(item.history)
        messages.append(Message.from_text(item.prompt, role=Role.USER))

        if item.protocol == "chat":
            response = await self._llm.achat(messages)
            text = ""
            msg = getattr(response, "message", None)
            if msg is not None:
                text = msg.content or ""
            if not text:
                # Defensive fallback when the provider returns a Response
                # without a Message — surface the raw repr rather than ""
                # so user code can detect the degenerate case.
                text = str(response)
            return text

        if item.protocol == "structure_output":
            if not isinstance(self._llm, StructuredOutput):
                raise TypeError(
                    f"LLM {type(self._llm).__name__} does not implement the "
                    "StructuredOutput protocol; cannot satisfy "
                    "LLMCall(protocol='structure_output')."
                )
            return await self._llm.astructured_output(messages, item.constraint)

        if item.protocol == "tool_selector":
            if not isinstance(self._llm, ToolSelection):
                raise TypeError(
                    f"LLM {type(self._llm).__name__} does not implement the "
                    "ToolSelection protocol; cannot satisfy "
                    "LLMCall(protocol='tool_selector')."
                )
            return await self._llm.aselect_tool(messages, item.tools)

        raise ValueError(
            f"Unknown LLMCall protocol: {item.protocol!r}. "
            "Expected 'chat', 'structure_output', or 'tool_selector'."
        )

    async def _invoke_template(
        self,
        gen_or_coro: Any,
        ctx: CognitiveContextT,
        *,
        scope: str = "hook",
    ) -> Any:
        """Generic template-method driver. No fallback policy.

        User-facing template methods (``observation``, ``on_agent``,
        ``before_action``, ``after_action``, ``on_workflow``) may be
        written as async generators (the idiomatic yield-driven form)
        OR as plain async coroutines. This helper bridges both forms:
        coroutines are awaited directly; async generators are driven
        with ``__anext__`` / ``asend``, dispatching each yielded item
        through ``_dispatch_call`` and capturing ``RETURN.value`` as
        the return.

        Parameters
        ----------
        gen_or_coro : Any
            An async generator object or coroutine returned from a
            template-method invocation.
        ctx : CognitiveContextT
            The current cognitive context, forwarded to ``_dispatch_call``.
        scope : {"workflow", "agent", "hook"}, default "hook"
            Identifies the kind of generator being driven so
            ``_dispatch_call`` can enforce yield-type restrictions.

        Returns
        -------
        Any
            The coroutine's return value, or the value captured from a
            ``RETURN(value)`` yield, or ``None`` if the generator
            exhausts without RETURN.

        Notes
        -----
        Used by hook callsites, by AGENT / WORKFLOW mode entry points,
        by ``_dispatch_call``'s EnterAgent recursive call (in
        WORKFLOW mode), and by ``_drive_amphiflow`` for EnterAgent /
        step-level / full-fallback agent runs. Errors propagate;
        body-level fallback policy lives in ``_drive_amphiflow``.
        """
        if not inspect.isasyncgen(gen_or_coro):
            return await gen_or_coro

        send_value: Any = None
        return_value: Any = None
        try:
            while True:
                try:
                    if send_value is None:
                        item = await gen_or_coro.__anext__()
                    else:
                        item = await gen_or_coro.asend(send_value)
                    send_value = None
                except StopAsyncIteration:
                    break

                if isinstance(item, RETURN):
                    return_value = item.value
                    break

                send_value = await self._dispatch_call(
                    item, ctx, scope=scope,
                )
        finally:
            await gen_or_coro.aclose()

        return return_value

    async def _run(
        self,
        worker: CognitiveWorker,
        *,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: int = 1,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> None:
        """
        Execute a worker against the current context.

        Two execution paths exist, distinguished by worker type:

        * ``CognitiveWorker`` — runs through the framework's observe-
          think-act cycle (one cycle per attempt, with ``until`` /
          ``max_attempts`` controlling the loop).
        * ``WorkerRunner`` (external implementation; not a
          ``CognitiveWorker``) — single ``await worker.run(self, ctx)``
          call; the worker is responsible for its own loop. The
          ``until`` / ``max_attempts`` / ``tools`` / ``skills`` /
          ``on_error`` / ``max_retries`` overlays are ignored on this
          path because they are CognitiveWorker concepts.

        Internal method used by ``_dispatch_call`` (ThinkUnit path and
        the ActionCall step-level fallback) and by user code written as
        a coroutine-form ``on_agent`` that drives workers manually.

        Parameters
        ----------
        worker : CognitiveWorker | WorkerRunner
            The worker to execute. If neither, ``TypeError`` is raised.
        until, max_attempts, tools, skills, on_error, max_retries
            See class docstring. Effective only on the CognitiveWorker path.
        """
        context = self._current_context
        if context is None:
            raise RuntimeError(
                "Cannot call _run(): no active context. "
                "_run() must be called within an on_agent() method."
            )

        # External WorkerRunner path — bypass observe-think-act entirely.
        if not isinstance(worker, CognitiveWorker):
            if isinstance(worker, WorkerRunner):
                self._log(
                    "Run",
                    f"WorkerRunner {type(worker).__name__} taking control of run loop",
                    color="cyan",
                )
                await worker.run(self, context)
                return
            raise TypeError(
                f"_run() expects a CognitiveWorker or a WorkerRunner; "
                f"got {type(worker).__name__}."
            )

        # CognitiveWorker path — framework-managed observe-think-act.
        async def _execute():
            if until is not None or max_attempts > 1:
                for _ in range(max_attempts):
                    finished = await self._run_once(
                        worker, tools=tools, skills=skills,
                        on_error=on_error, max_retries=max_retries,
                    )
                    if finished:
                        return
                    if until is not None:
                        cond_result = until(context)
                        if inspect.iscoroutine(cond_result):
                            cond_result = await cond_result
                        if cond_result:
                            return
            else:
                await self._run_once(
                    worker, tools=tools, skills=skills,
                    on_error=on_error, max_retries=max_retries,
                )

        await _execute()
    
    async def _run_once(
        self,
        worker: CognitiveWorker,
        *,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> bool:
        """Execute a single observe-think-act cycle. Returns whether the worker signalled finish."""
        async def _run_observe_think_act(worker: CognitiveWorker, context: CognitiveContextT) -> bool:
            worker_name = worker.__class__.__name__

            # 1. Observe
            # Worker-level ``None`` (e.g. an AI-generated ``pass`` stub) is
            # treated identically to ``_DELEGATE`` so the agent-level
            # observation fallback still runs. Both worker- and agent-level
            # hooks may be written as plain coroutines OR async generators
            # — ``_invoke_template`` handles both forms uniformly.
            #
            # If both layers ultimately return ``None``, preserve the
            # previous ``context.observation`` instead of overwriting it.
            # This lets ``after_action``-driven refresh patterns work
            # without forcing the user to write a passthrough
            # ``observation`` override solely to defeat the overwrite.
            obs = await self._invoke_template(worker.observation(context), context)
            if obs is _DELEGATE or obs is None:
                obs = await self._invoke_template(self.observation(context), context)
            if obs is not None:
                context.observation = obs

            obs_str = str(obs) if obs is not None else "None"
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."
            self._log("Observe", f"{worker_name}: {obs_str}", color="green")

            # 2. Think
            decision = await worker.arun(context=context)
            step_str = getattr(decision, 'step_content', str(decision))
            finished = getattr(decision, 'finish', False)
            self._log("Think", f"{worker_name}: finish={finished}, step={step_str}", color="cyan")

            # 3. Act
            action_result = await self._action(decision, context, _worker=worker) if decision is not None else None
            if action_result is not None:
                formatted = action_result.model_dump_json(indent=4)
                self._log("Act", f"{worker_name}:\n{formatted}", color="purple")

            # Record trace step
            self._record_trace_step(worker, obs, decision, action_result, context)

            # Auto-capture final answer when the worker signals finish
            if decision.finish and decision.step_content:
                self._final_answer = decision.step_content

            return decision.finish


        ########################
        # Initialize CognitiveWorker
        # runtime environment
        ########################
        context = self._current_context
        worker_label = worker.__class__.__name__

        # Init LLM
        if worker._llm is None and self._llm is not None:
            worker.set_llm(self._llm)

        # Init verbose
        injected_verbose = False
        if worker._verbose is None:
            worker._verbose = self._verbose
            injected_verbose = True

        # Init tools
        original_tools = None
        if tools is not None:
            original_tools = context.tools
            filtered_tools = CognitiveTools()
            for tool in original_tools.get_all():
                if tool.tool_name in tools:
                    filtered_tools.add(tool)
            context.tools = filtered_tools

        # Init skills (all bindings declared up-front so the `finally` block
        # can read them unconditionally regardless of whether filtering ran).
        original_skills: Optional[CognitiveSkills] = None
        filtered_skills: Optional[CognitiveSkills] = None
        filtered_to_orig: Dict[int, int] = {}
        if skills is not None:
            original_skills = context.skills
            filtered_skills = CognitiveSkills()
            orig_to_filtered: Dict[int, int] = {}
            for orig_idx, skill in enumerate(original_skills.get_all()):
                if skill.name in skills:
                    new_idx = len(filtered_skills)
                    filtered_skills.add(skill)
                    orig_to_filtered[orig_idx] = new_idx
                    filtered_to_orig[new_idx] = orig_idx
            for orig_idx, detail in original_skills._revealed.items():
                if orig_idx in orig_to_filtered:
                    filtered_skills._revealed[orig_to_filtered[orig_idx]] = detail
            context.skills = filtered_skills

        # Init spent status
        tokens_before = worker.spent_tokens

        ########################
        # Run CognitiveWorker
        ########################
        finished = False
        try:
            finished = await _run_observe_think_act(worker, context)
        except Exception as e:
            if on_error == ErrorStrategy.RAISE:
                raise RuntimeError(
                    f"Worker '{worker_label}' failed during "
                    f"observe-think-act cycle: {e}"
                ) from e
            elif on_error == ErrorStrategy.IGNORE:
                pass
            elif on_error == ErrorStrategy.RETRY:
                for attempt in range(max_retries + 1):
                    try:
                        finished = await _run_observe_think_act(worker, context)
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            raise RuntimeError(
                                f"Worker '{worker_label}' failed after "
                                f"{max_retries + 1} retries: {e}"
                            ) from e
        finally:
            # Record and restore the execution status of the worker
            self.spent_tokens += worker.spent_tokens - tokens_before
            if injected_verbose:
                worker._verbose = None
            if original_tools is not None:
                context.tools = original_tools
            if original_skills is not None:
                if filtered_skills is not None:
                    for filtered_idx, detail in filtered_skills._revealed.items():
                        orig_idx = filtered_to_orig.get(filtered_idx)
                        if orig_idx is not None:
                            original_skills._revealed[orig_idx] = detail
                context.skills = original_skills

        return finished

    def _parse_decision_for_action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
    ) -> Tuple[bool, Optional[List[StepToolCall]], Any]:
        """Parse a thinking decision into its execution form.

        Returns
        -------
        (is_tool_call_form, calls, decision_result)
            * ``is_tool_call_form`` — True when ``decision.output`` is
              declared as ``List[StepToolCall]`` (tool-call path); False
              when it is a BaseModel (custom-output path).
            * ``calls`` — the raw ``StepToolCall`` list when tool-call
              form, else ``None``.
            * ``decision_result`` — ``List[Tuple[ToolCall, ToolSpec]]``
              for tool-call form (ready for ``action_tool_call``), or
              the raw output BaseModel for custom-output form.

        Pure helper shared by ``_action`` (OTC-wrapped) and
        ``_action_raw`` (hook-scope, no wrapping).
        """
        def _is_list_step_tool_call(d: Any) -> bool:
            if not isinstance(d, BaseModel):
                return False
            fi = type(d).model_fields.get('output')
            if fi is None:
                return False
            ann = fi.annotation
            if get_origin(ann) is Annotated:
                ann = get_args(ann)[0]
            if ann is None:
                return False
            origin = get_origin(ann)
            if origin is list:
                args = get_args(ann)
                return len(args) == 1 and args[0] is StepToolCall
            return False

        def _convert_decision_to_tool_calls(calls: List, ctx: CognitiveContextT) -> List[ToolCall]:
            """Convert a list of StepToolCall into ToolCall objects with type-coerced arguments."""
            _, tool_specs = ctx.get_field('tools')
            tool_calls = []

            for idx, call in enumerate(calls):
                tool_spec = next((s for s in tool_specs if s.tool_name == call.tool), None)
                param_types: Dict[str, str] = {}
                if tool_spec and tool_spec.tool_parameters:
                    for name, info in tool_spec.tool_parameters.get('properties', {}).items():
                        param_types[name] = info.get('type', 'string')

                arguments: Dict[str, Any] = {}
                for arg in call.tool_arguments:
                    value: Any = arg.value
                    param_type = param_types.get(arg.name, 'string')
                    if param_type == 'integer':
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            pass
                    elif param_type == 'number':
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif param_type == 'boolean':
                        value = value.lower() in ('true', '1', 'yes')
                    arguments[arg.name] = value

                tool_calls.append(ToolCall(id=f"call_{idx}", name=call.tool, arguments=arguments))

            return tool_calls

        def _match_tool_calls(tool_calls: List[ToolCall], ctx: CognitiveContextT) -> List[Tuple[ToolCall, ToolSpec]]:
            """Match each ToolCall to its ToolSpec by name."""
            _, tool_specs = ctx.get_field('tools')
            matched: List[Tuple[ToolCall, ToolSpec]] = []
            for tc in tool_calls:
                for spec in tool_specs:
                    if tc.name == spec.tool_name:
                        if tc.arguments.get("__args__") is not None:
                            props = list(spec.tool_parameters.get('properties', {}).keys())
                            args = tc.arguments.get("__args__")
                            if isinstance(args, list):
                                tc.arguments = dict(zip(props, args))
                            else:
                                tc.arguments = {props[0]: args} if props else {}
                        matched.append((tc, spec))
                        break
            return matched

        output = getattr(decision, 'output', None)
        if _is_list_step_tool_call(decision):
            calls = output
            tool_calls = _convert_decision_to_tool_calls(calls, ctx)
            decision_result = _match_tool_calls(tool_calls, ctx)
            return True, calls, decision_result
        return False, None, output

    async def _execute_parsed_decision(
        self,
        decision: Any,
        decision_result: Any,
        ctx: CognitiveContextT,
        *,
        is_tool_call_form: bool,
        calls: Optional[List[StepToolCall]],
    ) -> Step:
        """Run the action proper — no ``before_action`` / ``after_action`` wrapping.

        Dispatches to ``action_tool_call`` (tool-call form) or
        ``action_custom_output`` (custom-output form) and builds the
        resulting ``Step``. Shared by ``_action`` and ``_action_raw``.
        """
        if is_tool_call_form:
            if not calls:
                result = Step(
                    content=decision.step_content,
                    result=None,
                    metadata={"tool_calls": []},
                )
                ctx.add_info(result)
                return result
            action_result = await self.action_tool_call(decision_result, ctx)
            result = Step(
                content=decision.step_content,
                result=action_result,
                metadata={},
            )
            ctx.add_info(result)
            return result

        # Custom-output form. ``None`` (e.g. from an AI-generated ``pass``
        # stub of ``action_custom_output``) is treated as passthrough so the
        # typed output is preserved instead of being silently dropped.
        custom_ret = await self.action_custom_output(decision_result, ctx)
        action_result = decision_result if custom_ret is None else custom_ret
        result = Step(content=decision.step_content, result=action_result, metadata={})
        ctx.add_info(result)
        return result

    async def _action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: Optional[CognitiveWorker] = None,
    ) -> Step:
        """OTC-wrapped action: ``before_action`` → execute → ``after_action``.

        Routes to ``action_tool_call()`` for tool-call output or
        ``action_custom_output()`` for structured output (output_schema).
        Calls ``before_action()`` and ``after_action()`` on both the
        worker and agent level (with delegation via ``_DELEGATE``).

        Used by the OTC cycle (``_run_observe_think_act``) and by
        ``_dispatch_call``'s workflow-scope ActionCall path. The hook-scope
        ActionCall path uses ``_action_raw()`` instead, because hooks are
        not OTC participants — re-entering the hook chain there would
        recurse into the same generator that yielded the ActionCall.

        Parameters
        ----------
        decision : Any
            The thinking decision with 'output' field (List[StepToolCall] or BaseModel).
        ctx : CognitiveContextT
            The cognitive context.
        _worker : Optional[CognitiveWorker]
            The worker that produced this decision (used for before_action callback).
        """
        is_tool_call_form, calls, decision_result = self._parse_decision_for_action(decision, ctx)

        # before_action delegation: worker → agent.
        # ``None`` is treated as "no-op override" so that AI-generated stubs
        # (``async def before_action(...): pass``) behave identically to not
        # overriding the hook at all:
        #   - worker-level None ≡ _DELEGATE → fall through to agent-level
        #   - agent-level None  ≡ passthrough → keep original decision_result
        # Both layers accept coroutine OR async-generator form;
        # ``_invoke_template`` drives either uniformly.
        original_decision_result = decision_result
        if _worker is not None:
            worker_ret = await self._invoke_template(
                _worker.before_action(decision_result, ctx), ctx,
            )
            if worker_ret is _DELEGATE or worker_ret is None:
                agent_ret = await self._invoke_template(
                    self.before_action(original_decision_result, ctx), ctx,
                )
                decision_result = original_decision_result if agent_ret is None else agent_ret
            else:
                decision_result = worker_ret
        else:
            agent_ret = await self._invoke_template(
                self.before_action(decision_result, ctx), ctx,
            )
            decision_result = original_decision_result if agent_ret is None else agent_ret

        result = await self._execute_parsed_decision(
            decision, decision_result, ctx,
            is_tool_call_form=is_tool_call_form, calls=calls,
        )

        # after_action delegation: worker → agent.
        # Worker-level ``None`` ≡ ``_DELEGATE`` so AI-generated ``pass`` stubs
        # still chain to the agent-level hook. Both layers accept coroutine
        # OR async-generator form; ``_invoke_template`` drives either
        # uniformly.
        if _worker is not None:
            delegate = await self._invoke_template(
                _worker.after_action(result, ctx), ctx,
            )
            if delegate is _DELEGATE or delegate is None:
                await self._invoke_template(self.after_action(result, ctx), ctx)
        else:
            await self._invoke_template(self.after_action(result, ctx), ctx)

        return result

    async def _action_raw(
        self,
        decision: Any,
        ctx: CognitiveContextT,
    ) -> Step:
        """Execute an action with NO hook participation.

        Used by ``_dispatch_call`` when an ActionCall is yielded from
        within a hook (``scope="hook"``). Hooks (``observation`` /
        ``before_action`` / ``after_action``) are **not** OTC participants
        — they are imperative side-effect channels that may yield to
        request a one-off tool execution. Running the OTC wrap
        (observation + before_action + after_action) here would re-enter
        the same hook generator that yielded this ActionCall and recurse
        infinitely.

        The two paths intentionally diverge:
        * ``scope="workflow"`` ActionCall → ``_action()`` (OTC-wrapped,
          recorded as a workflow trace step).
        * ``scope="hook"``     ActionCall → ``_action_raw()`` (raw tool
          execution, no hooks, no workflow-trace step).

        See ``_action()`` for the OTC-wrapped variant.
        """
        is_tool_call_form, calls, decision_result = self._parse_decision_for_action(decision, ctx)
        return await self._execute_parsed_decision(
            decision, decision_result, ctx,
            is_tool_call_form=is_tool_call_form, calls=calls,
        )

    @staticmethod
    def _build_tool_results(action_result: Optional[Step]) -> List[ToolResult]:
        """Convert an action Step into a List[ToolResult] for asend() back to the generator."""
        if action_result is None:
            return []
        inner = getattr(action_result, "result", None)
        if inner is not None and isinstance(inner, ActionResult):
            return [
                ToolResult(
                    tool_name=r.tool_name,
                    tool_arguments=r.tool_arguments,
                    result=r.tool_result,
                    success=r.success,
                    error=r.error,
                )
                for r in inner.results
            ]
        return []

    @asynccontextmanager
    async def snapshot(self, *, goal: Optional[str] = None,
                   keep_revealed: Optional[Dict[str, List[int]]] = None,
                   **snapshot_fields):
        """
        Temporarily override context fields for the duration of the block.

        Parameters
        ----------
        goal : Optional[str]
            Temporary goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context(keep_revealed=keep_revealed,
                                       goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def _phase_context(self, *,
                             keep_revealed: Optional[Dict[str, List[int]]] = None,
                             **snapshot_fields):
        """Shared implementation for snapshot() context manager.

        Subclasses can override this to inject custom behavior around
        snapshot blocks (e.g., logging, metrics, extended trace capture).

        Parameters
        ----------
        keep_revealed : Optional[Dict[str, List[int]]]
            Revealed state management mode for ``_AgentSnapshot``.
        **snapshot_fields
            Context fields to temporarily override during this phase.
        """
        context = self._current_context
        fields = {k: v for k, v in snapshot_fields.items() if v is not None}

        if not fields and keep_revealed is None:
            raise ValueError(
                "snapshot(): no snapshot fields, keep_revealed, "
                "or goal provided. If no context state needs to be scoped, "
                "use _run() directly — the behavior is identical."
            )

        snap = _AgentSnapshot(context, fields, keep_revealed=keep_revealed)
        await snap.__aenter__()
        try:
            yield
        finally:
            await snap.__aexit__(None, None, None)

    ############################################################################
    # Internal Helper methods
    #
    # Verbose logging (``_log``), trace recording (``_record_trace_step``
    # for OTC cycles / ActionCall dispatch, ``_record_llm_call_trace`` for
    # LLMCall yields), and template-override detection (``_has_workflow``
    # / ``_has_agent``) used by ``_resolve_mode``. These are called by the
    # dispatcher and ``_run_once`` — never by user code.
    ############################################################################

    def _log_call(self, scope: str, stage: str, message: str, *, color: str = "white") -> None:
        """Scope-aware ``_log`` for primitive dispatch.

        Hook-scope Calls (yielded from a generator-form ``observation`` /
        ``before_action`` / ``after_action``) are internal side-effects,
        not workflow narrative. Their dispatch logs are suppressed by
        default so the visible log stays focused on the workflow. Set
        ``verbose_hook_calls=True`` on the constructor to surface them
        for debugging.

        Workflow-scope Calls always log (subject to the usual
        ``self._verbose`` gate enforced by ``_log``).
        """
        if scope == "hook" and not self._verbose_hook_calls:
            return
        self._log(stage, message, color=color)

    def _log(self, stage: str, message: str, data: Any = None, color: str = "white"):
        """Log formatted message with timestamp and caller location.

        Format: ``[HH:MM:SS.mmm] [Stage] (file:line) message``

        Only prints when ``self._verbose`` is True.
        """
        if not self._verbose:
            return
        import inspect
        from datetime import datetime
        from os.path import basename

        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                filename = basename(caller.f_code.co_filename)
                lineno = caller.f_lineno
            else:
                filename, lineno = "?", 0
        finally:
            del frame

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] [{stage}] ({filename}:{lineno}) {message}"
        printer.print(line, color=color)
        if data is not None:
            printer.print(str(data), color="gray")
    
    def _record_trace_step(self, worker: Optional[CognitiveWorker], obs: str, decision: Any, action_result: Step, context: Any) -> None:
        """Record a trace step to the workflow builder (if capture is active).

        Detects the output type from the action_result:
        - Tool calls (ActionResult with results) → TOOL_CALLS
        - Structured BaseModel output → STRUCTURED
        - Everything else (content only, no action) → CONTENT_ONLY
        """
        if self._agent_trace is None:
            return

        tool_calls = []
        output_type = StepOutputType.CONTENT_ONLY
        structured_output = None
        structured_output_class = None

        if action_result is not None and isinstance(action_result, Step):
            result_obj = action_result.result
            if isinstance(result_obj, ActionResult):
                # Tool call output
                output_type = StepOutputType.TOOL_CALLS
                for r in result_obj.results:
                    tool_calls.append({
                        "tool_name": r.tool_name,
                        "tool_arguments": r.tool_arguments,
                        "tool_result": r.tool_result,
                        "success": r.success,
                        "error": r.error,
                    })
            elif result_obj is not None and isinstance(result_obj, BaseModel):
                # Structured BaseModel output
                output_type = StepOutputType.STRUCTURED
                structured_output = result_obj.model_dump()
                structured_output_class = (
                    f"{result_obj.__class__.__module__}.{result_obj.__class__.__qualname__}"
                )
            elif result_obj is not None:
                # Non-BaseModel custom output — store as structured with no class
                output_type = StepOutputType.STRUCTURED
                try:
                    structured_output = {"__value__": result_obj}
                except Exception:
                    structured_output = {"__value__": str(result_obj)}
            # else: result_obj is None → CONTENT_ONLY (default)

        self._agent_trace.record_step({
            "name": worker.__class__.__name__ if worker is not None else "workflow",
            "step_content": getattr(decision, "step_content", ""),
            "tool_calls": tool_calls,
            "observation": str(obs) if obs is not None else None,
            "observation_hash": observation_fingerprint(obs),
            "output_type": output_type.value,
            "structured_output": structured_output,
            "structured_output_class": structured_output_class,
        })

    def _record_llm_call_trace(self, item: LLMCall, result: Any) -> None:
        """Record an ``LLM_CALL`` trace step.

        LLMCall does not produce a tool-call result, observation, or
        worker, so it cannot be folded into ``_record_trace_step`` without
        forcing fake objects through that signature. The recorded step
        carries: the protocol name, the prompt as the "observation"
        slot (it is the closest analog), and the result serialized into
        ``structured_output``.
        """
        if self._agent_trace is None:
            return

        try:
            if isinstance(result, BaseModel):
                serialized = result.model_dump()
                cls_name = (
                    f"{result.__class__.__module__}.{result.__class__.__qualname__}"
                )
            else:
                serialized = {"__value__": result}
                cls_name = type(result).__qualname__
        except Exception:
            serialized = {"__value__": str(result)}
            cls_name = type(result).__qualname__

        self._agent_trace.record_step({
            "name": "llm_call",
            "step_content": f"LLMCall({item.protocol})",
            "tool_calls": [],
            "observation": item.prompt or None,
            "observation_hash": observation_fingerprint(item.prompt),
            "output_type": StepOutputType.LLM_CALL.value,
            "structured_output": serialized,
            "structured_output_class": cls_name,
            "llm_call_protocol": item.protocol,
        })

    def _has_workflow(self) -> bool:
        """Check whether the subclass has overridden on_workflow() with a real async generator.

        A subclass that writes ``async def on_workflow(...): pass`` (e.g. an
        AI-generated stub) produces a coroutine, not an async generator —
        treat that as "not overridden" so RunMode.AUTO falls back to the
        agent path. Users who deliberately want a coroutine-form workflow
        can force ``mode=RunMode.WORKFLOW`` or ``RunMode.AMPHIFLOW``; the
        dispatcher handles both forms in those paths.
        """
        impl = type(self).on_workflow
        if impl is AmphibiousAutoma.on_workflow:
            return False
        return inspect.isasyncgenfunction(impl)

    def _has_agent(self) -> bool:
        """Check whether the subclass has overridden on_agent()."""
        return type(self).on_agent is not AmphibiousAutoma.on_agent

    ############################################################################
    # Entry point
    #
    # Public ``arun()`` and the GraphAutoma plumbing it dispatches through.
    # ``_resolve_mode`` collapses ``RunMode.AUTO`` to a concrete mode
    # based on which template methods are overridden;
    # ``_resolve_builtin_filter`` decides which built-in tools to inject.
    # ``router`` is the GraphAutoma start worker that ferries to the
    # chosen mode entry — ``_agent`` (AGENT), ``_workflow`` (WORKFLOW),
    # or ``_amphiflow`` (AMPHIFLOW). AGENT and WORKFLOW use
    # ``_invoke_template`` (single-generator drive). AMPHIFLOW uses
    # ``_drive_amphiflow`` (state-machine drive with fallback).
    ############################################################################
    def _resolve_builtin_filter(
        self, override: Optional[Iterable[str]]
    ) -> Optional[FrozenSet[str]]:
        """Resolve which built-in tool names should be auto-injected.

        Resolution order: ``arun(builtin_tools=...)`` argument →
        class-level ``builtin_tools`` attribute → ``None`` (inject all).

        Returns
        -------
        Optional[FrozenSet[str]]
            ``None`` means inject every entry of ``_BUILTIN_TOOLS``;
            a frozenset (possibly empty) restricts injection to the
            listed tool names.
        """
        if override is not None:
            return frozenset(override)
        return type(self).builtin_tools

    def _resolve_mode(self, mode: RunMode) -> RunMode:
        """Resolve ``RunMode.AUTO`` to a concrete mode based on overridden template methods.

        Resolution rules:
        - both ``on_agent`` and ``on_workflow`` overridden → ``RunMode.AMPHIFLOW``
        - only ``on_workflow`` overridden → ``RunMode.WORKFLOW``
        - only ``on_agent`` overridden → ``RunMode.AGENT``
        - neither overridden → ``RuntimeError``

        Non-AUTO modes are returned unchanged.
        """
        if mode is not RunMode.AUTO:
            return mode
        has_agent = self._has_agent()
        has_workflow = self._has_workflow()
        if has_agent and has_workflow:
            return RunMode.AMPHIFLOW
        if has_workflow:
            return RunMode.WORKFLOW
        if has_agent:
            return RunMode.AGENT
        raise RuntimeError(
            f"{type(self).__name__} must override on_agent() or on_workflow()."
        )
    
    @worker(is_start=True)
    async def router(self, mode: RunMode, max_consecutive_fallbacks: int) -> str:
        """
        Router worker: dispatches to the correct execution mode.

        ``RunMode.AUTO`` is resolved upstream in ``arun()``, so this worker
        always receives a concrete mode.
        """
        if mode is RunMode.AGENT:
            self._log("Router", "Ferrying to AGENT mode", color="green")
            self.ferry_to("_agent")
        elif mode is RunMode.WORKFLOW:
            self._log("Router", "Ferrying to WORKFLOW mode", color="green")
            self.ferry_to("_workflow")
        elif mode is RunMode.AMPHIFLOW:
            self._log("Router", f"Ferrying to AMPHIFLOW mode, max_consecutive_fallbacks={max_consecutive_fallbacks}", color="green")
            self.ferry_to("_amphiflow", max_consecutive_fallbacks=max_consecutive_fallbacks)
        else:
            raise RuntimeError(f"Unsupported run mode: {mode!r}")

    @worker(is_output=True)
    async def _agent(self) -> str:
        """AGENT mode entry point.

        Drives ``on_agent`` through ``_invoke_template`` with
        ``scope='agent'``. No state machine, no fallback (agent IS
        already the autonomous tier).

        Returns
        -------
        str
            ``self._final_answer`` (if set by a ``RETURN(value)`` yield
            or by a worker's ``finish=True``) or ``ctx.summary()``.
        """
        ctx = self._current_context
        return_value = await self._invoke_template(
            self.on_agent(ctx), ctx, scope="agent",
        )
        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or ctx.summary()

    @worker(is_output=True)
    async def _workflow(self) -> str:
        """WORKFLOW mode entry point.

        Drives ``on_workflow`` through ``_invoke_template`` with
        ``scope='workflow'``. No fallback — failures propagate.
        ``EnterAgent`` yielded from on_workflow is dispatched through
        ``_dispatch_call``'s recursive ``_invoke_template`` path (works
        without a state machine because there is no fallback to track).

        Returns
        -------
        str
            ``self._final_answer`` (if set by a ``RETURN(value)`` yield)
            or ``ctx.summary()``.
        """
        ctx = self._current_context
        return_value = await self._invoke_template(
            self.on_workflow(ctx), ctx, scope="workflow",
        )
        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or ctx.summary()

    @worker(is_output=True)
    async def _amphiflow(self, max_consecutive_fallbacks: int) -> str:
        """AMPHIFLOW mode entry point.

        Drives ``on_workflow`` through ``_drive_amphiflow`` — the
        state-machine driver. ``EnterAgent`` (user-yielded) and
        step-level fallback (synthesized on atomic-Call failure within
        threshold) suspend the workflow generator and run a fresh
        on_agent generator in a snapshotted context; on_agent generator
        exhaustion implicitly resumes the workflow generator. Full
        fallback (threshold breached or generator-internal exception)
        closes the workflow generator and runs on_agent with the
        original context, ending the run.

        Parameters
        ----------
        max_consecutive_fallbacks : int
            Atomic-Call step-failure threshold before full fallback.

        Returns
        -------
        str
            ``self._final_answer`` (if set by a ``RETURN(value)`` yield
            from any flow, or by a fallback worker's ``finish=True``)
            or ``ctx.summary()``.
        """
        ctx = self._current_context
        return_value = await self._drive_amphiflow(ctx, max_consecutive_fallbacks)
        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or ctx.summary()

    async def arun(
        self,
        *,
        context: Optional[CognitiveContextT] = None,
        trace_running: bool = False,
        mode: Optional[RunMode] = RunMode.AUTO,
        max_consecutive_fallbacks: int = 1,
        builtin_tools: Optional[Iterable[str]] = None,
        **kwargs
    ) -> str:
        """
        Run the agent.

        Routes to one of the execution modes:
        1. Agent mode — LLM-driven ``on_agent()`` path.
        2. Workflow mode — deterministic ``on_workflow()`` path (no fallback).
        3. Amphiflow mode — ``on_workflow()`` with automatic agent fallback.
        4. Auto mode (default) — resolved from which template methods the
           subclass overrides: both → AMPHIFLOW, only ``on_workflow`` → WORKFLOW,
           only ``on_agent`` → AGENT.

        Context initialization has two paths:
        1. Pre-created: ``arun(context=my_ctx)``
        2. Auto-created: ``arun(goal="...", tools=[...], skills=[...])``

        Parameters
        ----------
        context : Optional[CognitiveContextT]
            Pre-created context object. If provided, uses this context directly.
        trace_running : bool
            If True, enables trace capture via AgentTrace during execution.
        mode : Optional[RunMode]
            Execution mode. ``RunMode.AGENT`` forces agent mode,
            ``RunMode.WORKFLOW`` forces workflow mode (no fallback),
            ``RunMode.AMPHIFLOW`` forces workflow with agent fallback,
            ``RunMode.AUTO`` (default) auto-detects from which template methods
            are overridden.
        max_consecutive_fallbacks : int
            Maximum consecutive workflow step failures before switching
            to full agent mode. Only applies to amphiflow mode. Default is 1.
        builtin_tools : Optional[Iterable[str]]
            Runtime filter for which built-in tools to inject. ``None``
            (the default) defers to the class-level ``builtin_tools``
            attribute, which itself defaults to ``None`` meaning "all".
            Pass an explicit iterable (e.g. ``["request_human"]``) to
            inject only those tools, or an empty iterable to opt out
            entirely. Tools the user has already passed via ``tools=[...]``
            are never overwritten.
        **kwargs
            Arguments passed to CognitiveContext constructor when ``context``
            is not provided (e.g., ``goal``, ``tools``, ``skills``).

        Returns
        -------
        str
            Summary of the context after execution.
        """
        def _build_trace(automa: "AmphibiousAutoma") -> Dict[str, Any]:
            """Build a trace dict from the workflow builder of the last run."""
            import time as _time
            metadata = {
                "automa_class": f"{automa.__class__.__module__}.{automa.__class__.__qualname__}",
                "context_class": (
                    f"{automa._context_class.__module__}.{automa._context_class.__qualname__}"
                    if automa._context_class else None
                ),
                "timestamp": _time.time(),
                "spent_tokens": automa.spent_tokens,
                "spent_time": automa.spent_time,
            }
            return automa._agent_trace.build(metadata=metadata)

        async def _run_and_report(context: CognitiveContextT) -> str:
            """Run the agent, measure time, and log summary."""
            start_time = time.time()
            result = await GraphAutoma.arun(
                self, resolved_mode,
                max_consecutive_fallbacks=max_consecutive_fallbacks,
            )
            self.spent_time = time.time() - start_time

            if self._verbose:
                agent_name = self.name or self.__class__.__name__
                separator = "=" * 50
                printer.print(separator, color="cyan")
                printer.print(
                    f"  {agent_name} | Completed\n"
                    f"Tokens: {self.spent_tokens} | "
                    f"Time: {self.spent_time:.2f}s",
                    color="cyan"
                )
                printer.print(separator, color="cyan")

            return result
        
        ########################
        # Pre-initialize status
        ########################
        self.spent_tokens = 0
        self.spent_time = 0.0
        self._final_answer = None
        self._read_tracker = {}

        resolved_mode = self._resolve_mode(mode if mode is not None else RunMode.AUTO)
        if self._llm is None and resolved_mode in (RunMode.AGENT, RunMode.AMPHIFLOW):
            raise RuntimeError(
                f"AmphibiousAutoma must be initialized with an LLM for "
                f"{resolved_mode.value} mode."
            )

        if trace_running:
            self._agent_trace = AgentTrace()
        else:
            self._agent_trace = None

        ########################
        # Initialize context
        ########################
        if context is not None:
            if not isinstance(context, self._context_class):
                raise ValueError(
                    f"Context must be an instance of {self._context_class.__name__}, "
                    f"got {type(context).__name__}"
                )

        # Separate Exposure fields (tools, skills, etc.) from regular constructor args
        exposure_fields = self._context_class._exposure_fields
        if exposure_fields is None:
            exposure_fields = self._context_class._detect_exposure_fields()
            self._context_class._exposure_fields = exposure_fields

        # Create the context
        constructor_kwargs = {}
        exposure_items = {}  # {field_name: list_of_items}
        for key, value in kwargs.items():  # Add fields to the context that can directly be added to the context
            if key in exposure_fields and isinstance(value, (list, tuple)):
                exposure_items[key] = value
            elif key in exposure_fields and isinstance(value, Exposure):
                constructor_kwargs[key] = value
            else:
                constructor_kwargs[key] = value
        
        if context is None:
            context = self._context_class(**constructor_kwargs)  # Create the context
        for field_name, items in exposure_items.items():  # Add items to Exposure fields
            attr = getattr(context, field_name)
            for item in items:
                attr.add(item)

        # Inject built-in tools (e.g. request_human, bash, read_file, ...)
        # so on_agent execution can trigger framework-level capabilities
        # autonomously. The ``builtin_tools`` arun kwarg takes priority over
        # the class-level ``builtin_tools`` attribute, which itself defaults
        # to ``None`` meaning "inject all".
        allowed_builtins = self._resolve_builtin_filter(builtin_tools)
        valid_builtin_names = {t.tool_name for t in _BUILTIN_TOOLS}
        if allowed_builtins is not None:
            unknown = allowed_builtins - valid_builtin_names
            if unknown:
                raise ValueError(
                    f"Unknown built-in tool name(s) in builtin_tools filter: "
                    f"{sorted(unknown)}. Valid names: {sorted(valid_builtin_names)}."
                )
        existing_tool_names = {t.tool_name for t in context.tools.get_all()}
        for builtin in _BUILTIN_TOOLS:
            if allowed_builtins is not None and builtin.tool_name not in allowed_builtins:
                continue
            if builtin.tool_name in existing_tool_names:
                continue
            # Specialise ``request_human`` against this agent class's
            # ``@human_channel`` registry so the LLM sees the actual
            # channel names — both in the tool description and as an
            # ``enum`` constraint on the ``channel`` parameter. With no
            # channels registered the factory returns the generic spec.
            if builtin.tool_name == "request_human":
                channels = type(self)._human_channels.keys()
                builtin = build_request_human_tool(channels)
            context.tools.add(builtin)

        # Set the LLM to the context
        if self._llm is not None:
            context.set_llm(self._llm)
        self._current_context = context
        
    
        ########################
        # Run the amphibious automa
        ########################
        token = current_agent.set(self)
        try:
            result = await _run_and_report(context=context)
            if trace_running and self._agent_trace:
                _build_trace(self)
            return result
        finally:
            current_agent.reset(token)
