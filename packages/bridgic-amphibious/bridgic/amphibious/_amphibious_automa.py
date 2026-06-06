import hashlib
import asyncio
import inspect
import json
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
import contextlib
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Awaitable, Callable, ClassVar, Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union,
    get_args, get_origin
)

from pydantic import BaseModel

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._automa import RunningOptions
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Role, ToolCall
from bridgic.core.model.protocols import StructuredOutput, ToolSelection
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.utils._console import printer
from bridgic.amphibious._context import (
    Context,
    OTAContext,
)
from bridgic.amphibious._cognitive_worker import CognitiveWorker, _DELEGATE
from bridgic.amphibious._agent_worker import AgentWorker
from bridgic.amphibious._think_unit import ThinkUnitDescriptor
from bridgic.amphibious._think_agent import ThinkAgentDescriptor
from bridgic.amphibious._run_dir import ensure_run_dir, make_run_id
from bridgic.amphibious.builtin_tools import current_agent
from bridgic.amphibious._type import (
    RunMode,
    Step,
    StepToolCall,
    ToolArgument,
    ThinkResult,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    ThinkUnit,
    ThinkAgent,
    RETURN,
    ErrorStrategy,
    ActionStepResult,
    ActionResult,
    ToolResult,
    StepOutputType,
    TraceStep,
    RecordedToolCall,
)


################################################################################################################
# Module-level type names + constants
################################################################################################################

# Two-loop generics — the small-loop (OTA) context and the loop context.
OTAContextT = TypeVar("OTAContextT", bound=OTAContext)
ContextT = TypeVar("ContextT", bound=Context)

# Sentinel put on a ThinkAgent delegation's decision channel to tell the
# per-delegation consumer task (``_run_think_agent._execute_decisions``)
# that the worker has finished and no more decisions will arrive.
_DELEGATION_DONE: Any = object()


################################################################################################################
# AgentTrace — flat execution path recorder
################################################################################################################

_LOG_BRIEF_CHARS: int = 2000

# Width of the ``[HH:MM:SS.mmm] `` timestamp prefix on header lines.
# Arrow sub-phase lines lead with a same-width spacer so ``->`` aligns
# under the header's ``[Label]`` column.
_LOG_TS_PREFIX_WIDTH: int = 15

# Fixed terminal width for wrapping log lines — typical modern setups are
# wider, but this keeps things readable when piped to a narrower view.
_LOG_TERMINAL_WIDTH: int = 120


def _brief(value: Any, n: int = _LOG_BRIEF_CHARS) -> str:
    """One-line truncated repr suitable for log lines.

    Collapses newlines / tabs, trims surrounding whitespace, and caps
    length at ``n`` chars with an ellipsis if longer.
    """
    if value is None:
        return ""
    text = str(value)
    text = " ".join(text.split())
    if len(text) > n:
        return text[:n] + "…"
    return text


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


class AgentTrace:
    """Unified trace recorder for one ``arun`` invocation.

    Owns ALL workdir persistence: when constructed with a non-``None``
    ``workdir``, every lifecycle event and step record triggers an
    incremental write of ``<workdir>/trace.json``. That one file is the
    single artifact for a run — it replaced an earlier multi-file layout
    (separate ``meta.json`` + ``ctx_initial.json`` + ``ctx_final.json``
    + a steps file).

    Trace data layout (``build()`` and the on-disk JSON share it)::

        {
            "goal":     "<the original arun goal>",
            "metadata": {agent_class, agent_name, context_class, mode,
                         run_id, start_time, end_time, spent_tokens,
                         spent_time, cost_time, ...},
            "history":  [TraceStep, ...],  # one entry per yield primitive
        }

    Semantic split from ``OTAContext.ota_record``: the small-loop round trace
    is summarised for the agent's own consumption (prompts), while this
    trace history is the detailed audit log of every step's outcome.
    """

    def __init__(self, workdir: Optional[Path] = None):
        self._workdir = workdir
        self._goal: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._steps: List[dict] = []

    ############################################################################
    # Lifecycle — called by ``AmphibiousAutoma.arun``
    ############################################################################

    def begin_run(
        self,
        *,
        goal: str,
        agent_class: str,
        agent_name: Optional[str],
        context_class: Optional[str],
        mode: str,
        run_id: Optional[str],
        max_consecutive_fallbacks: int,
        start_time: float,
    ) -> None:
        """Record run start; persist."""
        self._goal = goal
        self._metadata.update({
            "agent_class": agent_class,
            "agent_name": agent_name,
            "context_class": context_class,
            "mode": mode,
            "run_id": run_id,
            "max_consecutive_fallbacks": max_consecutive_fallbacks,
            "start_time": start_time,
            "start_time_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(start_time)
            ),
        })
        self._persist()

    def end_run(
        self,
        *,
        end_time: float,
        spent_tokens: int,
        spent_time: float,
    ) -> None:
        """Record run end; persist."""
        self._metadata.update({
            "end_time": end_time,
            "end_time_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(end_time)
            ),
            "spent_tokens": spent_tokens,
            "spent_time": spent_time,
            "cost_time": round(spent_time, 3),
        })
        self._persist()

    ############################################################################
    # Step recording — called by ``_record_*_trace`` in the dispatcher
    ############################################################################

    def record_step(self, step_data: dict) -> None:
        """Append a step record; persist incrementally."""
        self._steps.append(step_data)
        self._persist()

    ############################################################################
    # Snapshot / serialization
    ############################################################################

    def build(self) -> Dict[str, Any]:
        """Return the unified trace dict; pure (no IO)."""
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
                think_agent_name=s.get("think_agent_name"),
            )
            for s in self._steps
        ]
        return {
            "goal": self._goal,
            "metadata": dict(self._metadata),
            "history": steps,
        }

    def save(self, path: str) -> None:
        """Explicit one-shot write to ``path``.

        Equivalent to what ``_persist()`` writes to ``workdir/trace.json``,
        but writable anywhere; useful for tests and ad-hoc snapshotting.
        """
        data = self._to_serializable(self.build())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Deserialize a trace from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    ############################################################################
    # Internals
    ############################################################################

    def _persist(self) -> None:
        """Best-effort write to ``<workdir>/trace.json``.

        No-op when ``workdir`` is ``None`` (in-memory only). Wrapped in a
        try/except so an artifact-write failure can never mask the run's
        primary control flow.
        """
        if self._workdir is None:
            return
        try:
            data = self._to_serializable(self.build())
            (self._workdir / "trace.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

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
# @human_channel decorator — registers async methods as named HumanCall handlers
################################################################################################################

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


################################################################################################################
# AMPHIFLOW state machine internals — per-run state on ``self._amphi``
################################################################################################################


@dataclass
class _AmphiState:
    """Per-AMPHIFLOW-run FSM state.

    Held on ``AmphibiousAutoma._amphi`` for the lifetime of one
    ``_amphiflow`` call. ``_dispatch_step`` reads / writes it directly
    when handling primitives so the dispatcher (not the driver) owns
    mode transitions, send-slot fills, and the termination signal.

    Attributes
    ----------
    workflow_gen, workflow_send
        Workflow generator slot and value to feed into its next
        ``.asend()``. Workflow is the entry mode — this slot lives for
        the whole run unless full fallback closes it.
    agent_gen, agent_send
        Agent generator slot (lazy-created on ``EnterAgent`` or
        step-level fallback) and value to feed into its next
        ``.asend()``. Disposed on agent exhaustion.
    scope
        Which generator is currently the active one (``"workflow"`` or
        ``"agent"``). Updated explicitly by ``_dispatch_step`` at every
        mode transition and by the driver's StopAsyncIteration handler
        when agent generator exhausts back to workflow.
    agent_mode_stack
        ``AsyncExitStack`` holding the agent-mode scope. Pushed when
        entering agent mode (via ``EnterAgent`` or full fallback); popped
        when the agent generator exhausts or raises. (Step-level fallback
        does not use it — it runs a bounded inline recovery sub-run.)
    max_consecutive_fallbacks, consecutive_failures, step_index, failed_steps
        Step-level fallback bookkeeping. Counts atomic-Call failures
        across the workflow; threshold breach escalates to full
        fallback.
    return_value, should_break
        Termination signal. ``RETURN(value)`` yielded from any flow —
        and full-fallback agent runs funneled through
        ``_dispatch_step(RETURN(...))`` — populate ``return_value`` and
        set ``should_break = True``; the driver's outer
        ``while not self._amphi.should_break:`` loop then exits.
    """
    # Current worked mode's generator.
    workflow_gen: Any
    workflow_send: Any = None
    agent_gen: Optional[Any] = None
    agent_send: Any = None
    scope: Literal["workflow", "agent"] = "workflow"

    # Agent-mode snapshot stack for nested EnterAgent or step-level fallbacks.
    agent_mode_stack: Optional[AsyncExitStack] = None

    # Step-level fallback bookkeeping + configuration.
    failed_steps: List[str] = field(default_factory=list)
    max_consecutive_fallbacks: int = 1
    consecutive_failures: int = 0

    # State record for `Amphi`
    step_index: int = 0
    return_value: Any = None
    should_break: bool = False


################################################################################################################
# Decision parsing — turn a worker's ``ThinkResult`` (its ``tool_calls``, a
# list of ``StepToolCall``) into matched ``(ToolCall, ToolSpec)`` pairs. Shared
# by ``_run_action_call`` and the ``action_tool_call`` template method. A
# decision with no ``tool_calls`` is the finish — there is no action to run.
################################################################################################################

def _decision_to_matched_calls(
    decision: Any, tools: List[ToolSpec]
) -> List[Tuple[ToolCall, ToolSpec]]:
    """Turn a ``decision``'s ``tool_calls`` (a list of ``StepToolCall``) into
    matched ``(ToolCall, ToolSpec)`` pairs against ``tools``.

    Coerces each argument to its declared parameter type (integer / number /
    boolean) and resolves a positional ``__args__`` against the spec's
    property order. Calls whose name matches no spec are dropped.
    """
    calls = getattr(decision, "tool_calls", None) or []

    # 1. StepToolCall -> ToolCall (with type-coerced arguments).
    tool_calls: List[ToolCall] = []
    for idx, call in enumerate(calls):
        tool_spec = next((s for s in tools if s.tool_name == call.tool), None)
        param_types: Dict[str, str] = {}
        if tool_spec and tool_spec.tool_parameters:
            for name, info in tool_spec.tool_parameters.get("properties", {}).items():
                param_types[name] = info.get("type", "string")
        arguments: Dict[str, Any] = {}
        for arg in call.tool_arguments:
            value: Any = arg.value
            param_type = param_types.get(arg.name, "string")
            if param_type == "integer":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif param_type == "number":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            elif param_type == "boolean":
                value = value.lower() in ("true", "1", "yes")
            arguments[arg.name] = value
        tool_calls.append(ToolCall(id=f"call_{idx}", name=call.tool, arguments=arguments))

    # 2. Match each ToolCall to its ToolSpec by name.
    matched: List[Tuple[ToolCall, ToolSpec]] = []
    for tc in tool_calls:
        for spec in tools:
            if tc.name == spec.tool_name:
                if tc.arguments.get("__args__") is not None:
                    props = list(spec.tool_parameters.get("properties", {}).keys())
                    args = tc.arguments.get("__args__")
                    if isinstance(args, list):
                        tc.arguments = dict(zip(props, args))
                    else:
                        tc.arguments = {props[0]: args} if props else {}
                matched.append((tc, spec))
                break
    return matched


################################################################################################################
# AmphibiousAutoma
################################################################################################################

class AmphibiousAutoma(GraphAutoma, Generic[OTAContextT, ContextT]):
    """Base class for amphibious agents — dual-mode orchestration engine.

    Subclasses define behavior by implementing ``on_agent()`` (LLM-driven,
    yields ``ThinkUnit`` / ``ThinkAgent``) and/or ``on_workflow()``
    (deterministic, yields ``ActionCall`` / ``HumanCall`` / ``LLMCall`` /
    ``EnterAgent``). Under ``RunMode.AUTO`` (default), only-on_agent →
    AGENT, only-on_workflow → WORKFLOW, both → AMPHIFLOW (workflow-first
    with agent fallback on step failure).

    Yield-type ↔ scope rules:

    ===========  ============  ========  =====
    primitive    on_workflow   on_agent  hooks
    ===========  ============  ========  =====
    ActionCall   ✓             ✗         ✓
    HumanCall    ✓             ✗         ✓
    LLMCall      ✓             ✗         ✓
    EnterAgent   ✓             ✗         ✗
    ThinkUnit    ✗             ✓         ✗
    ThinkAgent   ✗             ✓         ✗
    RETURN       ✓             ✓         ✓
    ===========  ============  ========  =====

    Constructor params: ``llm`` (default LLM for workers), ``name``
    (instance name), ``verbose`` (log execution summary), and
    ``verbose_hook`` (surface dispatch logs for Calls yielded from
    hooks — suppressed by default since hooks are internal side-effects).

    Examples
    --------
    >>> class MyThink(CognitiveWorker):
    ...     async def thinking(self, ota_context, context=None):
    ...         return await self._llm.aselect_tool(messages=[...], tools=[...])
    >>> class MyAgent(AmphibiousAutoma[OTAContext, Context]):
    ...     main_think = think_unit(MyThink(), max_attempts=20)
    ...     async def on_agent(self, ota_context, context=None):
    ...         yield ThinkUnit("main_think")
    ...
    >>> answer = await MyAgent().arun(llm=llm, user_input="Complete the task")
    """

    ############################################################################
    # Class attributes — populated by ``__init_subclass__``
    ############################################################################

    #: Small-loop context type, resolved from the first generic argument by
    #: ``_detect_context_classes``. The framework constructs a fresh instance
    #: of this per ``arun`` (seeding ``goal``), so it must be ``OTAContext``
    #: (or a subclass).
    _ota_context_class: Optional[Type[OTAContext]] = None
    #: Big-loop context type, resolved from the second generic argument. A
    #: free-form ``Context`` subclass; supplied at run time via ``arun(context=)``
    #: (optional — the framework reads its ``summary()`` and its declared
    #: ``tools``).
    _context_class: Optional[Type[Context]] = None

    #: ``@human_channel``-decorated registry, populated by
    #: ``__init_subclass__``. Maps channel-name → method-name. Empty on
    #: the base class; subclasses inherit and may add or override.
    _human_channels: ClassVar[Dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Per-subclass initialisation.

        Three responsibilities:

        1. Extract the two context types (``OTAContext`` small-loop +
           ``Context`` loop) from the ``Generic[OTAContextT, ContextT]``
           parameters so ``cls._ota_context_class`` / ``cls._context_class``
           are set.
        2. Build the ``cls._human_channels`` registry by walking the MRO
           and collecting every method tagged via ``@human_channel``.
           Subclass overrides win over parent declarations.
        3. Validate that every overridden template method is an async
           generator (the only shape the dispatch model supports).
        """
        super().__init_subclass__(**kwargs)
        cls._detect_context_classes()
        cls._build_human_channel_registry()
        cls._validate_template_forms()

    @classmethod
    def _validate_template_forms(cls) -> None:
        """Reject coroutine-form template overrides at class-creation time.

        The dispatch model is yield-driven: every framework primitive
        (``ActionCall`` / ``HumanCall`` / ``LLMCall`` / ``EnterAgent`` /
        ``ThinkUnit`` / ``ThinkAgent`` / ``RETURN``) reaches the framework
        via ``yield``. A coroutine-form override (``async def`` without
        any ``yield``) cannot use any of these primitives, so the
        framework no longer accepts that shape. The base class defaults
        are themselves stub async generators (``if False: yield``), so
        not overriding is also fine.
        """
        template_names = (
            "on_agent", "on_workflow",
            "observation", "before_action", "after_action",
        )
        for name in template_names:
            impl = getattr(cls, name, None)
            base_impl = getattr(AmphibiousAutoma, name, None)
            # Not overridden — base default is already a proper async-gen.
            if impl is base_impl:
                continue
            if not inspect.isasyncgenfunction(impl):
                raise TypeError(
                    f"{cls.__name__}.{name} must be an ``async def`` "
                    f"function with at least one ``yield`` statement in "
                    f"its body. The framework's dispatch model is "
                    f"yield-driven — every primitive (ActionCall / "
                    f"HumanCall / LLMCall / EnterAgent / ThinkUnit / "
                    f"ThinkAgent / RETURN) reaches the framework via "
                    f"``yield``, so a coroutine-form template override "
                    f"(no ``yield``) cannot use the framework. If the "
                    f"body has no real yields, add ``if False: yield`` "
                    f"as an unreachable stub to keep the async-generator "
                    f"shape."
                )

    @classmethod
    def _detect_context_classes(cls) -> None:
        """Resolve the two context types from ``Generic[OTAContextT, ContextT]``.

        Parses ``__orig_bases__`` for a parametrization carrying exactly two
        arguments and validates each against its bound: the first must be an
        :class:`OTAContext` (framework-owned small loop), the second a
        :class:`Context` (free-form loop). Both are required — there is no
        single-argument form.

        A subclass of an already-parametrized agent (whose own
        ``__orig_bases__`` no longer name the generic) inherits both classes
        from its base. The error path fires only when neither parametrization
        nor inheritance yields a valid pair.
        """
        for base in getattr(cls, "__orig_bases__", []):
            if get_origin(base) is None:
                continue
            args = get_args(base)
            if len(args) != 2:
                continue
            ota_type, loop_type = args
            # Skip bare-TypeVar / unresolved parametrizations; inheritance
            # (below) covers concrete-parametrized intermediate subclasses.
            if not (isinstance(ota_type, type) and isinstance(loop_type, type)):
                continue
            if not issubclass(ota_type, OTAContext):
                raise TypeError(
                    f"{cls.__name__}: the first generic argument {ota_type.__name__!r} "
                    f"is not an OTAContext. AmphibiousAutoma[OTAContextT, ContextT] "
                    f"requires the small-loop context (arg 1) to subclass OTAContext, "
                    f"e.g. class {cls.__name__}(AmphibiousAutoma[OTAContext, Context])."
                )
            if not issubclass(loop_type, Context):
                raise TypeError(
                    f"{cls.__name__}: the second generic argument {loop_type.__name__!r} "
                    f"is not a Context. AmphibiousAutoma[OTAContextT, ContextT] requires "
                    f"the loop context (arg 2) to subclass Context, "
                    f"e.g. class {cls.__name__}(AmphibiousAutoma[OTAContext, Context])."
                )
            cls._ota_context_class = ota_type
            cls._context_class = loop_type
            return

        # Inheritance: a subclass of an already-parametrized agent.
        for base in cls.__bases__:
            ota_inherited = getattr(base, "_ota_context_class", None)
            loop_inherited = getattr(base, "_context_class", None)
            if ota_inherited is not None and loop_inherited is not None:
                cls._ota_context_class = ota_inherited
                cls._context_class = loop_inherited
                break

        if cls._ota_context_class is None or cls._context_class is None:
            raise TypeError(
                f"{cls.__name__} must specify both context types via the generic "
                f"parameters, e.g. "
                f"class {cls.__name__}(AmphibiousAutoma[OTAContext, Context]). "
                f"Arg 1 (small loop) must subclass OTAContext; arg 2 (loop) must "
                f"subclass Context."
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
    # Instance attributes — set up in ``__init__``
    ############################################################################

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        verbose: bool = False,
        verbose_hook: bool = False,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        # User-facing state. Two context slots for the two loops, each read
        # through a property accessor (``self.ota_ctx`` / ``self.ctx``) so
        # internal methods reach the active context off ``self`` rather than
        # threading it as a parameter:
        #   * ``_current_ota_context`` — the small-loop OTA context, freshly
        #     constructed per ``arun`` and swapped to a nested sub-context for
        #     the duration of a delegation (via ``_ota_scope``). Read: ``ota_ctx``.
        #   * ``_current_context`` — the loop knowledge context, supplied by
        #     the caller and read-only to the run. Read: ``ctx``.
        self._llm = None
        self._current_ota_context: Optional[OTAContextT] = None
        self._current_context: Optional[ContextT] = None
        self._run_mode: Optional[RunMode] = None

        # Log configuration
        self._verbose = verbose
        self._verbose_hook = verbose_hook
        self._log_depth: int = 0
        self._log_hook_name: Optional[str] = None

        # Trace capture
        self._agent_trace: Optional[AgentTrace] = None
        self._read_tracker: Dict[str, float] = {}
        self._current_run_dir: Optional[Path] = None

        # Running results
        self._final_answer: Optional[str] = None
        self.spent_tokens: int = 0
        self.spent_time: float = 0.0

        # AMPHIFLOW FSM state — set by ``_amphiflow`` for the duration of one AMPHIFLOW run
        self._amphi: Optional[_AmphiState] = None

    @property
    def llm(self) -> Optional[Any]:
        """LLM of the active or most recent ``arun`` (``None`` before
        the first run and after ``arun`` clears it in ``finally``)."""
        return self._llm

    @property
    def ota_ctx(self) -> Optional[OTAContextT]:
        """The active small-loop (OTA) context.

        Freshly constructed per ``arun`` and swapped to a nested sub-context
        for the span of a delegation (``EnterAgent`` / ``ThinkAgent`` /
        step-level fallback) by ``_ota_scope``. Internal methods read the
        active context through this accessor instead of threading it as a
        parameter; the underlying slot (``_current_ota_context``) is written
        only by ``arun`` and ``_ota_scope``.
        """
        return self._current_ota_context

    @property
    def ctx(self) -> Optional[ContextT]:
        """The loop (knowledge) context — the free-form context passed to
        ``arun(context=)`` (or a fresh default when none was supplied).

        Shared read-only across the parent run and any nested delegation;
        only the small loop (``ota_ctx``) is isolated per sub-run.
        """
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
    # Template methods — overridable hooks. May be written as async-generator
    # (yielding framework primitives) or plain coroutine; dispatcher accepts
    # both. Scope rules are documented on the class docstring above.
    ############################################################################
    async def observation(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> AsyncGenerator[Any, Any]:
        """Agent-level default observation, shared across all workers.

        Called before each thinking phase; workers' own ``observation()``
        delegates here when it returns ``_DELEGATE`` / ``None``.

        Yield ``RETURN(text)`` to set ``ota_context.obs_result`` for this
        cycle. Exhausting without ``RETURN`` (or yielding ``RETURN(None)``)
        **preserves** the previous ``ota_context.obs_result`` — so
        ``after_action``-driven refresh patterns work without a dedicated
        passthrough override.

        >>> async def observation(self, ota_context, context=None):
        ...     snapshot = yield ActionCall("bash", command="bridgic-browser snapshot")
        ...     yield RETURN(snapshot[0].result)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_agent(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> AsyncGenerator[Any, Any]:
        """Agent mode: LLM-driven cognitive flow.

        Override to declare the agent's strategy. on_agent body is
        reserved for orchestrating cognitive steps — only ``ThinkUnit``
        / ``ThinkAgent`` / ``RETURN`` are allowed (deterministic tool /
        HITL / direct-LLM operations belong in on_workflow or a hook).
        Without ``RETURN``, the framework auto-captures the final answer
        from the last think step's ``step_content``.

        >>> async def on_agent(self, ota_context, context=None):
        ...     yield ThinkUnit("main_think", max_attempts=20)
        ...     yield ThinkUnit("exec_think", until=lambda c: c.done)
        ...     yield RETURN(ota_context.ota_record[-1].think_result.step_content)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_workflow(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> AsyncGenerator[Union[ActionCall, HumanCall, EnterAgent, LLMCall], None]:
        """Workflow mode: deterministic flow as an async generator.

        Override to declare a deterministic workflow. Yield ``ActionCall``
        / ``HumanCall`` / ``LLMCall`` for atomic steps, ``EnterAgent`` to
        enter an autonomous sub-flow, ``RETURN(value)`` to terminate
        early. Use ``result = yield ActionCall(...)`` to receive results
        via ``asend()``.

        >>> async def on_workflow(self, ota_context, context=None):
        ...     yield ActionCall("navigate_to", url="http://example.com")
        ...     result = yield ActionCall("click_element_by_ref", ref="42")
        ...     summary = yield LLMCall.chat("Summarize the page in one line.")
        ...     yield EnterAgent(goal="Handle complex case")
        """
        if False:  # pragma: no cover — makes this a proper async generator stub
            yield

    async def before_action(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> AsyncGenerator[Any, Any]:
        """Agent-level before_action hook, shared across all workers.

        Called when a worker's ``before_action()`` returns ``_DELEGATE``
        / ``None``. Payload-free — read the pending decision from
        ``ota_context.think_result``. Yield ``RETURN(modified_decision)``
        to override the decision; exhausting without RETURN (or returning
        ``None`` from a coroutine override) is passthrough — the folded
        decision stands.

        >>> async def before_action(self, ota_context, context=None):
        ...     adjusted = sanitize(ota_context.think_result)
        ...     yield RETURN(adjusted)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def action_tool_call(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> ActionResult:
        """Execute the current decision's tool calls concurrently, collect results.

        The calls are read off the decision on ``ota_context.think_result``
        (a ``before_action`` hook may have already filtered or replaced it)
        and matched against ``ota_context.tools``. Override to customize
        execution (sequential, rate-limited, sandboxed); both contexts are
        passed for parity with the other template methods.
        """
        matched = _decision_to_matched_calls(
            ota_context.think_result, ota_context.tools
        )

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
            *(_run_one(tc, ts) for tc, ts in matched)
        )
        return ActionResult(results=list(step_results))

    async def after_action(self, ota_context: OTAContextT, context: Optional[ContextT] = None) -> AsyncGenerator[Any, Any]:
        """Agent-level after_action hook.

        Called after action execution. Payload-free — read the action
        result from ``ota_context.action_result``. Override to update
        custom context fields or trigger follow-up primitives based on
        it. ``RETURN`` is unused here — the hook's return value is
        ignored.

        >>> async def after_action(self, ota_context, context=None):
        ...     summary = yield LLMCall.chat(f"Summarize: {ota_context.action_result}")
        ...     ota_context.action_result  # action payload on the current round
        """
        if False:  # pragma: no cover — async generator stub
            yield

    ############################################################################
    # Core methods
    #
    # Two engines: ``_invoke_template`` / ``_amphiflow`` drive generators,
    # ``_dispatch_step`` routes each yield to a ``_run_<primitive>`` /
    # ``_enter_agent`` handler. ``RETURN`` is the only yield NOT routed
    # through dispatch — the two drivers intercept it directly as a loop
    # control signal. ``_ota_scope`` provides the fresh-instance delegation
    # mechanism EnterAgent, ThinkAgent, and step-level fallback rely on
    # (each runs a nested OTA episode with its own ``OTAContext``).
    ############################################################################

    async def _invoke_template(
        self,
        gen_or_coro: Any,
        *,
        scope: str = "hook",
    ) -> Any:
        """Generic template-method driver. No fallback policy.

        Supports two template shapes:

        * **Async-generator** — driven with ``__anext__`` / ``asend``,
          dispatching each yielded item through ``_dispatch_step``,
          capturing ``RETURN(value)`` as the return.
        * **Coroutine** — ``await`` and return the awaited value. Used
          by ``CognitiveWorker`` hooks (``observation`` /
          ``before_action`` / ``after_action``) whose natural shape is
          ``return _DELEGATE`` / ``return value``.

        ``scope`` is one of ``"workflow"`` / ``"agent"`` / ``"hook"``
        and gates which primitives ``_dispatch_step`` accepts. Errors
        propagate; body-level fallback lives in ``_amphiflow``.
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
                else:
                    if isinstance(item, RETURN):
                        return_value = item.value
                        break
                    send_value = await self._dispatch_step(item, scope=scope)
        finally:
            # Cleanup
            try:
                await gen_or_coro.aclose()
            except Exception:
                pass

        return return_value

    async def _dispatch_step(
        self,
        item: Any,
        *,
        scope: str = "hook",
    ) -> Any:
        """Per-yield handler — the single place that knows framework primitives.

        Routes each operation primitive to its ``_run_<primitive>`` /
        ``_enter_agent`` and returns the raw result for the caller to
        forward via ``.asend()`` (inline) or write to
        ``fsm.{agent,workflow}_send`` (AMPHIFLOW). ``RETURN`` is
        intercepted by the callers themselves (``_invoke_template`` /
        ``_amphiflow``) — it is a control-flow signal, not an operation.

        Scope rules:

        * ``ActionCall`` / ``HumanCall`` / ``LLMCall`` — ``workflow`` or
          ``hook``, never ``agent``.
        * ``EnterAgent`` — ``workflow`` only.
        * ``ThinkUnit`` / ``ThinkAgent`` — ``agent`` only.

        ActionCall in ``scope="hook"`` skips before/after_action hooks
        (``_run_action_call(..., with_hooks=False)``): hooks are not OTC
        participants — re-entering the hook chain would recurse into
        the generator that yielded the call.
        """
        if isinstance(item, EnterAgent):
            # Scope validation
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

            # Mode switch
            return await self._enter_agent(item=item)

        if isinstance(item, HumanCall):
            # Scope validation
            if scope == "agent":
                raise RuntimeError(
                    f"HumanCall(prompt={item.prompt!r}) is not allowed inside "
                    "on_agent — the agent should request human input via the "
                    "auto-injected ``request_human`` tool (called by the LLM "
                    "during a ThinkUnit), not by yielding HumanCall directly. "
                    "If you need a deterministic human step, put it in "
                    "on_workflow."
                )
            
            # Human call
            return await self._run_human_call(item)

        if isinstance(item, LLMCall):
            # Scope validation: LLMCall is an atomic step and must be handled
            if scope == "agent":
                raise RuntimeError(
                    f"LLMCall(protocol={item.protocol!r}) is not allowed inside "
                    "on_agent — on_agent body is reserved for orchestrating "
                    "cognitive steps via ThinkUnit. Direct LLM calls belong "
                    "in on_workflow, in a hook, or inside a CognitiveWorker's "
                    "thinking() method."
                )
            
            # LLM call
            return await self._run_llm_call(item)

        if isinstance(item, ThinkUnit):
            # Scope validation: ThinkUnit is a cognitive step and must be handled by on_agent.
            if scope != "agent":
                raise RuntimeError(
                    f"ThinkUnit(name={item.name!r}) is only valid inside "
                    f"on_agent (scope='agent'); got scope={scope!r}. "
                    "ThinkUnit references a class-level think_unit and "
                    "represents a step in the agent's cognitive strategy; "
                    "use LLMCall for a direct LLM invocation outside the "
                    "cognitive loop, or EnterAgent to enter an on_agent flow."
                )
            # Run ThinkUnit
            return await self._run_think_unit(item)

        if isinstance(item, ThinkAgent):
            # Scope validation: ThinkAgent is a cognitive step that delegates to an external agent runtime.
            if scope != "agent":
                raise RuntimeError(
                    f"ThinkAgent(name={item.name!r}) is only valid inside "
                    f"on_agent (scope='agent'); got scope={scope!r}. "
                    "ThinkAgent hands the sub-goal off to an external agent "
                    "runtime and is part of the cognitive-composition layer; "
                    "use EnterAgent from on_workflow if you need to enter the "
                    "agent flow, then yield ThinkAgent from there."
                )
            
            # ThinkAgent
            return await self._run_think_agent(item)

        if isinstance(item, ActionCall):
            # Scope validation: ActionCall is an atomic step and must be handled by on_workflow or a hook, never on_agent.
            if scope == "agent":
                raise RuntimeError(
                    f"ActionCall(tool_name={item.tool_name!r}) is not allowed "
                    "inside on_agent — let the LLM decide tool calls inside a "
                    "ThinkUnit. If you need a deterministic tool call, put it "
                    "in on_workflow or in a worker hook (observation / "
                    "before_action / after_action)."
                )
            
            # ActionCall — wrap the single tool call into a ThinkResult decision.
            decision = ThinkResult(
                step_content=item.description,
                tool_calls=[StepToolCall(
                    tool=item.tool_name,
                    tool_arguments=[
                        ToolArgument(name=k, value=str(v)) for k, v in item.tool_args.items()
                    ],
                )],
            )
            if scope == "hook":
                action_result = await self._run_action_call(decision, with_hooks=False, top_level=False)
            else:
                action_result = await self._run_action_call(decision, _worker=None)

            inner = getattr(action_result, "result", None)
            if isinstance(inner, ActionResult):
                failed = [r for r in inner.results if not r.success]
                if failed:
                    errors = "; ".join(f"{r.tool_name}: {r.error}" for r in failed)
                    raise RuntimeError(
                        f"Tool execution failed for: "
                        f"{decision.step_content} — {errors}"
                    )

            return self._build_tool_results(action_result)
        
        raise TypeError(
            f"Unknown yield type: {type(item).__name__}. Expected one of "
            "ActionCall / HumanCall / LLMCall / EnterAgent / ThinkUnit / ThinkAgent. "
            "(RETURN is a control-flow signal handled upstream in "
            "``_invoke_template`` / ``_amphiflow`` before dispatch.)"
        )
    
    async def _enter_agent(
        self,
        *,
        item: Optional[EnterAgent] = None,
    ) -> Any:
        """Run a fresh nested OTA episode of the agent's ``on_agent`` strategy.

        Delegation = a fresh sub-run (isolation by construction), not a
        snapshot of the parent context. A fresh :class:`OTAContext` is
        built (own ``user_input`` / ``ota_record``, carrying the OTA context
        class's declared tools) and installed as
        ``self._current_ota_context`` for the sub-flow; the parent's OTA
        context is restored when the sub-flow ends. The loop knowledge
        context is **shared** (read via the ``current_agent`` ContextVar) —
        only the small loop is isolated.

        Two entry shapes:

        * ``item`` (a yielded ``EnterAgent``) — sub-goal is ``item.goal``.
          Emits the ``[EnterAgent]`` header + ``-> final:`` closer.
        * no ``item`` (AMPHIFLOW full fallback) — the sub-run inherits the
          parent's goal. No envelope.

        (Step-level fallback no longer routes through here — it runs a
        bounded inline recovery via ``_run_fallback_agent``.)

        The sub-run's tools always come from the OTA context class
        declaration (``OTAContext.tool``); nothing is filtered or passed in.
        The inherited goal is read off ``self.ota_ctx`` (still the parent —
        the scope swap happens afterwards, step 3 / 4).
        """
        # 1. Resolve the fresh sub-run's goal from the entry shape (its tools
        #    come from the OTA context class declaration, so there is nothing
        #    to pass or filter). ``self.ota_ctx`` is still the parent here.
        if item is not None:
            sub_goal: str = item.goal
        else:
            sub_goal = self.ota_ctx.user_input

        # 2. Build the fresh small-loop OTA context (isolation by construction;
        #    it auto-carries the OTA context class's declared tools).
        sub_ctx = self._ota_context_class(user_input=sub_goal)

        envelope = item is not None
        if envelope:
            self._log(
                "EnterAgent",
                f"goal={_brief(item.goal)}",
                color="yellow",
            )
            self._log_depth += 1
        try:
            # 3. AMPHIFLOW path: install the sub-context on an
            #    ``AsyncExitStack`` (restored when the agent generator
            #    exhausts, via ``fsm.agent_mode_stack``), then hand the
            #    fresh ``on_agent`` generator to the state machine.
            #    Mirrors the legacy snapshot hand-off — only the scoped
            #    object changed (a fresh OTA context, not field overrides).
            if self._run_mode is RunMode.AMPHIFLOW:
                fsm = self._amphi
                assert fsm is not None, (
                    "AMPHIFLOW run_mode but ``self._amphi`` is None — "
                    "``_enter_agent`` was called outside ``_amphiflow``'s "
                    "state machine. Check the run-mode / FSM lifecycle."
                )
                stack = AsyncExitStack()
                agent_obj = None
                try:
                    await stack.__aenter__()
                    await stack.enter_async_context(self._ota_scope(sub_ctx))
                    # Build the generator AFTER the swap so its ``ctx``
                    # parameter is the fresh sub-context.
                    agent_obj = self.on_agent(sub_ctx, self.ctx)
                except BaseException:
                    if agent_obj is not None:
                        try:
                            await agent_obj.aclose()
                        except Exception:
                            pass
                    try:
                        await stack.__aexit__(None, None, None)
                    except Exception:
                        pass
                    raise
                fsm.agent_mode_stack = stack
                fsm.agent_gen = agent_obj
                fsm.scope = "agent"
                result = None
            else:
                # 4. Inline path: drive the fresh sub-run to completion
                #    against the fresh sub-context, then restore the parent.
                async with self._ota_scope(sub_ctx):
                    result = await self._invoke_template(
                        self.on_agent(sub_ctx, self.ctx), scope="agent",
                    )

            if envelope:
                self._record_enter_agent(item, result)
            return result
        finally:
            if envelope:
                self._log_depth -= 1

    async def _run_fallback_agent(self, goal: str) -> Any:
        """Run a bounded recovery sub-run inline and return its conclusion.

        Step-level fallback — unlike full fallback, which hands ``on_agent``
        to the state machine for the rest of the run — is a *bounded*
        recovery: a fresh OTA episode runs to completion against ``goal``,
        and its conclusion is what the caller shapes into the failed step's
        return type and asends to the resuming workflow.

        The conclusion is read off the **isolated sub-context** — the
        sub-run's ``RETURN`` value, else its last think step's
        ``step_content``. It deliberately never touches
        ``self._final_answer``: that slot is owned by the run drivers
        (``_agent`` / ``_workflow`` / ``_amphiflow``) for the run's *own*
        final answer (``return self._final_answer or summary()``), so a
        helper must not reset it. The recovery agent's think step still
        updates ``self._final_answer`` naturally (as any agent run does) —
        last meaningful answer wins, overwritten if the resuming workflow
        yields its own ``RETURN``.

        Nothing is injected and no toolset is mutated — the sub-run carries
        the OTA context class's declared tools, same as any other sub-run.
        """
        sub_ctx = self._ota_context_class(user_input=goal)
        async with self._ota_scope(sub_ctx):
            result = await self._invoke_template(
                self.on_agent(sub_ctx, self.ctx), scope="agent",
            )
        if result is not None:
            return result
        # No RETURN — fall back to the recovery run's last think conclusion,
        # read off the isolated sub-context (never ``self._final_answer``).
        last_decision = sub_ctx.think_result
        return getattr(last_decision, "step_content", None) or None

    async def _run_human_call(self, item: "HumanCall") -> str:
        """Run one HumanCall and emit ``[Human Interaction]`` header +
        ``-> result:`` arrow. ``_record_human_call`` is invoked here so
        the trace + log live in this method (not the dispatcher).
        """
        async def _stdin_human_fallback(prompt: str) -> str:
            """Default human-input source when no ``@human_channel`` is registered.

            Reads a single line from stdin in a thread executor so the
            event loop is not blocked. Tests stub by monkey-patching
            ``builtins.input``.
            """
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, input, f"\n[HumanInput] {prompt}\n> "
            )

        prompt = item.prompt
        channel = item.channel

        channel_str = channel or "default"
        self._log(
            "Human Interaction",
            f"{channel_str}: {_brief(prompt or '')}",
            color="purple",
        )
        self._log_depth += 1
        try:
            registry = type(self)._human_channels
            if not registry:
                response = await _stdin_human_fallback(prompt)
            else:
                ch = channel
                if ch is None:
                    if len(registry) != 1:
                        raise RuntimeError(
                            "HumanCall(channel=None) is ambiguous: "
                            f"{len(registry)} channels registered "
                            f"({sorted(registry.keys())}). Specify channel='name' "
                            "explicitly."
                        )
                    ch = next(iter(registry))
                method_name = registry.get(ch)
                if method_name is None:
                    raise RuntimeError(
                        f"Unknown human channel: {ch!r}. "
                        f"Registered: {sorted(registry.keys())}"
                    )
                response = await getattr(self, method_name)(prompt)
            self._record_human_call(item, response)
            return response
        finally:
            self._log_depth -= 1

    async def _run_llm_call(self, item: LLMCall) -> Any:
        """Run one LLMCall and emit ``[LLM Query]`` header +
        ``-> result:`` arrow. ``_record_llm_call`` is invoked here so
        the trace + log live in this method (not the dispatcher).
        """
        self._log("LLM Query", item.protocol, color="white")
        self._log_depth += 1
        try:
            if self._llm is None:
                raise RuntimeError(
                    f"LLMCall(protocol={item.protocol!r}) requires self._llm, "
                    "but no LLM was passed to arun(llm=...)."
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
                    text = str(response)
                result = text
            elif item.protocol == "structure_output":
                if not isinstance(self._llm, StructuredOutput):
                    raise TypeError(
                        f"LLM {type(self._llm).__name__} does not implement the "
                        "StructuredOutput protocol; cannot satisfy "
                        "LLMCall(protocol='structure_output')."
                    )
                result = await self._llm.astructured_output(messages, item.constraint)
            elif item.protocol == "tool_selector":
                if not isinstance(self._llm, ToolSelection):
                    raise TypeError(
                        f"LLM {type(self._llm).__name__} does not implement the "
                        "ToolSelection protocol; cannot satisfy "
                        "LLMCall(protocol='tool_selector')."
                    )
                result = await self._llm.aselect_tool(messages, item.tools)
            else:
                raise ValueError(
                    f"Unknown LLMCall protocol: {item.protocol!r}. "
                    "Expected 'chat', 'structure_output', or 'tool_selector'."
                )

            self._record_llm_call(item, result)
            return result
        finally:
            self._log_depth -= 1

    async def _run_think_unit(
        self,
        item: ThinkUnit,
    ) -> Any:
        """Drive one ``ThinkUnit`` yield through its observe-think-act cycle.

        Returns the think unit's result — the finishing think's
        ``step_content`` — which becomes the ``yield ThinkUnit(...)`` value.

        Resolves the descriptor from ``item.name``, clones the
        ``CognitiveWorker`` template (state isolation), resolves
        per-yield overlays against descriptor defaults, sets up the
        runtime env (LLM injection, verbose), and runs the OTC loop.

        Emits the ``[Think] <name>`` header + bumps ``_log_depth`` so
        per-cycle arrows nest underneath. Mirrors ``_run_think_agent``
        in shape.
        """
        ########################
        # Resolve descriptor + overlays
        ########################
        descriptor = getattr(type(self), item.name, None)
        if not isinstance(descriptor, ThinkUnitDescriptor):
            raise AttributeError(
                f"ThinkUnit(name={item.name!r}) does not match any "
                f"think_unit declaration on {type(self).__name__}."
            )

        until = item.until if item.until is not None else descriptor._until
        max_attempts: int = (
            item.max_attempts if item.max_attempts is not None
            else descriptor._max_attempts
        )
        # on_error / max_retries are descriptor-only (no per-yield overlay).
        on_error: ErrorStrategy = descriptor._on_error
        max_retries: int = descriptor._max_retries

        ########################
        # Clone worker (state isolation)
        ########################
        worker = ThinkUnitDescriptor._clone_worker(descriptor._worker_template)
        worker_label = worker.__class__.__name__

        ########################
        # [Think] header + depth bump
        ########################
        self._log("Think", item.name, color="cyan")
        self._log_depth += 1
        try:
            result = await self._run_think_unit_body(
                worker, worker_label,
                until=until,
                max_attempts=max_attempts,
                on_error=on_error,
                max_retries=max_retries,
            )
        finally:
            self._log_depth -= 1
        
        return result

    async def _run_think_unit_body(
        self,
        worker: CognitiveWorker,
        worker_label: str,
        *,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: int = 1,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> None:
        """OTC body — the actual ``CognitiveWorker`` observe-think-act loop.

        Split out from ``_run_think_unit`` so the verbose-injection /
        token-tracking ``try / finally`` keeps its shape; the outer method
        wraps it with descriptor resolution + ``[Think]`` header.

        The toolset the LLM sees is whatever the worker's ``thinking()``
        assembles from ``ota_context.tools`` (the OTA loop owns the tool
        registry) — the think unit no longer narrows it. Its only
        knobs are loop control (``max_attempts`` / ``until``) and the
        per-cycle error policy (``on_error`` / ``max_retries``).
        """
        ########################
        # Setup runtime env.
        ########################
        # The active small-loop OTA context the worker operates on (stable for
        # the whole OTC loop — a ThinkUnit never swaps the delegation scope).
        ota_ctx = self.ota_ctx
        if ota_ctx is None:
            raise RuntimeError(
                "Cannot call _run_think_unit(): no active context. "
                "_run_think_unit() must be called within an on_agent() method."
            )

        # LLM (final CognitiveWorker has no set_llm — LLM is set directly)
        if worker._llm is None and self._llm is not None:
            worker._llm = self._llm
        if worker._llm is None:
            raise RuntimeError(
                f"ThinkUnit's CognitiveWorker ({worker_label}) has no LLM. "
                "Either pass llm=... to arun(), or set llm on the "
                "CognitiveWorker template itself."
            )

        # verbose
        injected_verbose = False
        if worker._verbose is None:
            worker._verbose = self._verbose
            injected_verbose = True

        # spent-tokens delta tracker
        tokens_before = worker.spent_tokens

        ########################
        # OTC cycle closure
        ########################
        async def _run_observe_think_act(cycle: int) -> bool:
            # 0. Open a fresh OTA round at cycle start
            ota_ctx.open_record()

            # 1. Observe (worker → agent fallback) → ``ota.obs_result``.
            obs = await self._invoke_template(worker.observation(ota_ctx))
            if obs is _DELEGATE or obs is None:
                obs = await self._invoke_template(self.observation(ota_ctx, self.ctx))
            if obs is not None:
                ota_ctx.obs_result = obs

            # 2. Think (worker reads the small-loop OTA ``ota_ctx``
            decision = await worker.arun(ota_context=ota_ctx, context=self.ctx)
            self._record_think_unit(worker, obs, decision, cycle=cycle)
            # Fold the full decision onto the round (the act path re-folds the
            # same via _run_action_call; the finish path below relies on it too).
            self.ota_ctx.think_result = decision
            if decision.tool_calls == []:
                # No tool calls → this IS the finish; the think text is both
                # the run's final answer and the ``yield ThinkUnit`` result.
                self.ota_ctx.action_result = None
                self._final_answer = decision.step_content
                return True, decision.step_content

            # 3. Act — execute the tool calls. On a non-finishing cycle the
            # ``yield ThinkUnit`` result is still the latest think text.
            await self._run_action_call(decision, _worker=worker, top_level=False)
            return False, decision.step_content

        # Run loop with on_error handling, then restore environment.
        result: Any = None
        try:
            for cycle_idx in range(max_attempts):
                cycle_num = cycle_idx + 1
                try:
                    finished, result = await _run_observe_think_act(cycle_num)
                except Exception as e:
                    if on_error == ErrorStrategy.RAISE:
                        raise RuntimeError(
                            f"Worker '{worker_label}' failed during "
                            f"observe-think-act cycle: {e}"
                        ) from e
                    elif on_error == ErrorStrategy.IGNORE:
                        # ``result`` keeps the last successful cycle's
                        # ``step_content`` (None if none has succeeded) —
                        # an ignored cycle contributes no value.
                        finished = False
                    elif on_error == ErrorStrategy.RETRY:
                        finished = False
                        for attempt in range(max_retries + 1):
                            try:
                                finished, result = await _run_observe_think_act(cycle_num)
                                break
                            except Exception as retry_e:
                                if attempt == max_retries:
                                    raise RuntimeError(
                                        f"Worker '{worker_label}' failed after "
                                        f"{max_retries + 1} retries: {retry_e}"
                                    ) from retry_e
                else:
                    if finished:
                        break
                    if until is not None:
                        cond_result = until(ota_ctx)
                        if inspect.iscoroutine(cond_result):
                            cond_result = await cond_result
                        if cond_result:
                            break
        finally:
            self.spent_tokens += worker.spent_tokens - tokens_before
            if injected_verbose:
                worker._verbose = None
        
        return result

    async def _run_think_agent(self, item: ThinkAgent) -> Any:
        """Drive one ``ThinkAgent`` yield through one delegated cycle.

        Resolves the descriptor from ``item.name``, clones the
        ``AgentWorker`` template (state isolation), runs the delegation
        against a **fresh nested OTA context** (its ``goal`` is the
        per-yield ``goal`` overlaid on the parent's; its ``tools`` are the
        ``expose_tools``-filtered parent toolset), runs the observation
        phase (worker → agent fallback), then drives the worker. The
        parent OTA context is restored when the delegation ends — it is
        never mutated (isolation by construction, replacing the removed
        snapshot mechanism).

        **Decision channel.** The worker does NOT execute the external
        agent's tool calls — it only *produces* decisions, exactly like
        ``CognitiveWorker``. Each MCP tool call the external agent makes
        is surfaced onto a per-delegation ``asyncio.Queue`` as a
        ``(decision, future)`` pair; a consumer task — owned here, alive
        only for this one delegation — pulls each, runs it through
        ``_run_action_call`` against the fresh sub-context (so
        ``before_action`` / ``after_action`` hooks fire and the call lands
        in the sub-run's rounds + the trace), and resolves the future.
        ``AmphibiousAutoma`` is the only place that *acts*.

        Per-yield knobs flow through the fresh sub-context exactly the way
        ``CognitiveWorker`` reads its inputs — no worker-side slots, no
        second protocol. ``AgentWorker.thinking()`` reads ``context.goal``
        directly (the fresh sub-context's).

        Mirrors ``_run_think_unit`` in shape — the two cognitive-
        composition drivers share the same skeleton.
        """
        ########################
        # Resolve descriptor + per-yield overlays
        ########################
        descriptor = getattr(type(self), item.name, None)
        if not isinstance(descriptor, ThinkAgentDescriptor):
            raise AttributeError(
                f"ThinkAgent(name={item.name!r}) does not match any "
                f"think_agent declaration on {type(self).__name__}."
            )

        expose_tools_filter: Optional[List[str]] = (
            item.expose_tools if item.expose_tools is not None
            else descriptor._expose_tools
        )

        ########################
        # Build the fresh nested OTA context for this delegation
        ########################
        # ``yield ThinkAgent(goal=...)`` becomes the sub-context's goal
        # (else inherit the parent's); ``expose_tools`` narrows the parent's
        # ``ota.tools`` to the whitelisted set — the external agent only ever
        # sees / calls these (it MCP-ifies ``ota.tools``), so the filter is a
        # real construction-time narrowing here, not just a render-time one.
        # ``self.ota_ctx`` is still the parent here — the scope swap to
        # ``sub_ctx`` happens below via ``_ota_scope``.
        parent_ota = self.ota_ctx
        sub_goal: str = item.goal if item.goal is not None else parent_ota.user_input
        sub_tools = self._filter_tools(parent_ota.tools, expose_tools_filter)
        sub_ctx = self._ota_context_class(user_input=sub_goal, tools=sub_tools)

        ########################
        # Clone worker (state isolation) + wire the decision channel
        ########################
        worker = ThinkAgentDescriptor._clone_worker(descriptor._worker_template)
        decision_channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = decision_channel

        async def _execute_decisions() -> None:
            """Per-delegation consumer: pull the decisions the worker
            surfaces from the external agent's MCP tool calls, run each
            through ``_run_action_call`` (folding onto the fresh
            sub-context), resolve the result future.

            Lives only for this one delegation — it ends as soon as the
            ``_DELEGATION_DONE`` sentinel arrives (put in the ``finally``
            below, after the worker has finished and the channel is
            drained).
            """
            while True:
                msg = await decision_channel.get()
                if msg is _DELEGATION_DONE:
                    return
                decision, result_future = msg
                try:
                    step = await self._run_action_call(decision, _worker=worker, top_level=False)
                    result_future.set_result(step)
                except Exception as exc:  # surface to the worker's await
                    result_future.set_exception(exc)

        ########################
        # [ThinkAgent] header + depth bump
        ########################
        self._log(
            "ThinkAgent",
            f"{item.name}  goal={_brief(item.goal or '')}",
            color="yellow",
        )
        self._log_depth += 1
        try:
            ########################
            # Drive within the fresh sub-context, alongside the consumer
            ########################
            consumer = asyncio.create_task(_execute_decisions())
            try:
                async with self._ota_scope(sub_ctx):
                    result = await self._run_think_agent_body(worker)
            finally:
                # Worker is done → no more decisions can arrive (the
                # external agent blocks on each call's result, so the
                # channel is already drained). Signal the consumer to
                # stop and join it.
                await decision_channel.put(_DELEGATION_DONE)
                await consumer

            # ``result`` is an ``AgentResult`` — captured from
            # ``worker.arun``'s return value, mirroring how
            # ``CognitiveWorker.arun`` hands its decision back. The
            # worker keeps no result slot of its own.
            self._record_think_agent(item, result, worker)
            return result.output
        finally:
            self._log_depth -= 1

    async def _run_think_agent_body(
        self,
        worker: AgentWorker,
    ) -> Any:
        """Delegate body — observation + ``AgentWorker.arun``.

        Split out from ``_run_think_agent`` so the verbose-injection +
        token-tracking ``try / finally`` keeps its shape; the outer
        method handles descriptor resolution, installing the fresh
        sub-context (``_ota_scope``), header / depth orchestration, and
        the ``_record_think_agent`` envelope.

        Tests that want to bypass the actual MCP host + subprocess can
        patch this method — by the time it runs, ``self.ota_ctx`` is the
        fresh sub-context (its ``user_input`` / ``tools`` reflect the
        per-yield overlay) so introspection works.

        Returns the ``AgentResult`` from ``worker.arun`` — the parent
        unwraps ``.output`` for the ``yield ThinkAgent`` asend value and
        folds the whole result into the trace envelope.
        """
        ota_ctx = self.ota_ctx
        if ota_ctx is None:
            raise RuntimeError(
                "Cannot call _run_think_agent(): no active context. "
                "_run_think_agent() must be called within an on_agent() method."
            )

        # verbose
        injected_verbose = False
        if worker._verbose is None:
            worker._verbose = self._verbose
            injected_verbose = True

        try:
            ########################
            # Observe (worker → agent fallback)
            ########################
            obs = await self._invoke_template(worker.observation(ota_ctx))
            if obs is _DELEGATE or obs is None:
                obs = await self._invoke_template(self.observation(ota_ctx, self.ctx))
            if obs is not None:
                ota_ctx.obs_result = obs

            ########################
            # Drive the worker — return value is the AgentResult
            ########################
            return await worker.arun(ota_context=ota_ctx, context=self.ctx)
        finally:
            if injected_verbose:
                worker._verbose = None

    async def _run_action_call(
        self,
        decision: Any,
        *,
        _worker: Optional[CognitiveWorker] = None,
        with_hooks: bool = True,
        top_level: bool = True,
    ) -> Step:
        """Execute a thinking decision — the single canonical action executor.

        Executes the decision's ``tool_calls`` via ``action_tool_call()``
        (a decision with none is the finish — no action runs), optionally
        wrapped by ``before_action`` / ``after_action`` hooks. When
        ``_worker`` is given AND ``with_hooks`` is True, the worker-level
        hooks run first and delegate to the agent level via ``_DELEGATE``.

        ``with_hooks=False`` skips ALL before/after_action — reserved for
        the hook-scope ActionCall path, where re-entering the hook chain
        would recurse into the generator that yielded the call.

        ``top_level=True`` (default) emits the ``[Action Execution]``
        header + ``-> observation:`` arrow and bumps ``_log_depth``.
        Callers that wrap this in their own scope (worker OTC inside
        ThinkUnit, the dispatcher's hook-scope branch, MCP bridge in
        ThinkAgent) pass ``top_level=False`` to suppress the header.
        """
        # The active small-loop OTA context
        ota_ctx = self.ota_ctx

        # A nested hook-scope ActionCall (``with_hooks=False``) runs INSIDE the
        # outer round, which already folded the outer decision onto
        # ``think_result``. Since the act phase re-reads ``think_result``, this
        # nested call must not leave its own decision there — save the outer's
        # and restore it on exit. Top-level / OTC calls persist think_result:
        # it is the round's real think.
        _restore_think = not with_hooks
        _outer_think = ota_ctx.think_result if _restore_think else None

        # Top-level: Log
        if top_level:
            desc = getattr(decision, "step_content", "") or ""
            if not desc:
                calls = getattr(decision, "tool_calls", None) or []
                names = [getattr(c, "tool", None) for c in calls]
                names = [t for t in names if t]
                desc = ", ".join(names) if names else "<action>"
            self._log("Action Execution", _brief(desc), color="green")
            self._log_depth += 1

        try:
            # Top-level observation gathering (agent-level only).
            if top_level:
                obs = await self._invoke_template(self.observation(ota_ctx, self.ctx))
                if obs is not None:
                    ota_ctx.obs_result = obs
                    self._log("Observation", _brief(obs), color="green")

            # Fold the decision onto the current round BEFORE before_action so
            # the payload-free hook reads it via ``ota_context.think_result``.
            ota_ctx.think_result = decision

            # Before_action hooks
            if with_hooks:
                with self._hook_log_scope("before_action"):
                    if _worker is not None:
                        worker_ret = await self._invoke_template(
                            _worker.before_action(ota_ctx, self.ctx),
                        )
                        if worker_ret is _DELEGATE or worker_ret is None:
                            agent_ret = await self._invoke_template(
                                self.before_action(ota_ctx, self.ctx),
                            )
                            if agent_ret is not None:
                                ota_ctx.think_result = agent_ret
                        else:
                            ota_ctx.think_result = worker_ret
                    else:
                        agent_ret = await self._invoke_template(
                            self.before_action(ota_ctx, self.ctx),
                        )
                        if agent_ret is not None:
                            ota_ctx.think_result = agent_ret

            # Execute the (possibly hook-modified) decision off
            # ``ota_context.think_result``: run its ``tool_calls`` via
            # ``action_tool_call``, or — when there are none — finish (the
            # content-only think has no action payload).
            final_decision = ota_ctx.think_result
            if getattr(final_decision, "tool_calls", None):
                action_result = await self.action_tool_call(ota_ctx, self.ctx)
                result = Step(result=action_result)
            else:
                # No tool calls — a content-only finish; no action payload.
                result = Step(result=None)

            # Fold the action payload onto the current round BEFORE after_action
            # so the payload-free hook reads it via ``ota_context.action_result``.
            ota_ctx.action_result = result.result

            # Record (trace + ``-> result:`` arrow). Sits between hooks
            # so the arrow lands chronologically between ``-> before_action:``
            # and ``-> after_action:`` arrows.
            self._record_action_call(
                _worker, obs=ota_ctx.obs_result, decision=final_decision,
                action_result=result,
            )

            # After_action hooks — skipped entirely when with_hooks=False.
            if with_hooks:
                with self._hook_log_scope("after_action"):
                    if _worker is not None:
                        delegate = await self._invoke_template(
                            _worker.after_action(ota_ctx, self.ctx),
                        )
                        if delegate is _DELEGATE or delegate is None:
                            await self._invoke_template(self.after_action(ota_ctx, self.ctx))
                    else:
                        await self._invoke_template(self.after_action(ota_ctx, self.ctx))

            return result
        finally:
            if top_level:
                self._log_depth -= 1
            if _restore_think:
                ota_ctx.think_result = _outer_think

    @asynccontextmanager
    async def _ota_scope(self, sub_ctx: OTAContextT):
        """Install ``sub_ctx`` as the active small-loop context for a block.

        Delegation isolation by construction (replaces the removed
        ``snapshot`` / ``_AgentSnapshot`` field-override machinery): a
        fresh :class:`OTAContext` becomes ``self._current_ota_context`` for
        the duration of the block, and the parent OTA context is restored
        on exit (including on exception). The parent context is **never
        mutated** — the sub-run's ``rounds`` accumulate on its own object.

        Parameters
        ----------
        sub_ctx : OTAContextT
            The fresh small-loop context to make active.
        """
        parent_ctx = self._current_ota_context
        self._current_ota_context = sub_ctx
        try:
            yield sub_ctx
        finally:
            self._current_ota_context = parent_ctx

    @staticmethod
    def _filter_tools(
        tools: List[ToolSpec],
        allowed_names: Optional[List[str]],
    ) -> List[ToolSpec]:
        """Return a fresh ``list`` of ``tools`` narrowed to a whitelist.

        When ``allowed_names`` is ``None`` the result is a verbatim copy
        (no narrowing). Always returns a NEW list — the input is never
        mutated, so a sub-run's filtered toolset never aliases the parent's.
        """
        if allowed_names is None:
            return AmphibiousAutoma._clone_tools(tools)
        allowed = set(allowed_names)
        return [tool for tool in tools if tool.tool_name in allowed]

    @staticmethod
    def _clone_tools(tools: List[ToolSpec]) -> List[ToolSpec]:
        """Return a fresh ``list`` carrying the same specs.

        A shallow copy of the list (the ``ToolSpec`` instances are
        shared — they are stateless), so a sub-run's ``ota.tools`` is an
        independent list from the parent's.
        """
        return list(tools)

    ############################################################################
    # Internal helpers — logging, trace recording, override detection.
    ############################################################################

    @contextlib.contextmanager
    def _hook_log_scope(self, hook_name: str):
        """Mark log entries inside the block as belonging to a hook.

        While active, ``_log`` lazily emits a ``[<hook_name>]`` header on
        first call, indents subsequent lines +1, and overrides color to
        gray. Gated by ``self._verbose_hook`` (independent of
        main-flow ``self._verbose``).
        """
        prev = self._log_hook_name
        self._log_hook_name = hook_name
        try:
            yield
        finally:
            self._log_hook_name = prev

    def _log(self, label: str, content: str = "", *, color: str = "white") -> None:
        """Render one log line.

        Two display forms, chosen automatically:

        * **Header** (``depth == 0``, not in hook scope) —
          ``[HH:MM:SS.mmm] [<label>] <content>``. Used for top-level
          Calls (``ActionCall`` / ``LLMCall`` / ``HumanCall`` /
          ``ThinkUnit`` / ``ThinkAgent`` / ``EnterAgent``) and the
          one-shot ``Router`` event.
        * **Arrow** (``depth > 0`` or in hook scope) —
          ``[HH:MM:SS.mmm]   -> <phase>: <content>``. Used for
          sub-phases of a Call: ``observation`` / ``before_action`` /
          ``result`` / ``after_action`` / ``think`` etc. When inside
          ``_hook_log_scope(<name>)`` the phase tag is overridden to
          ``<name>`` (the original ``label`` is discarded) and color
          is forced to gray.

        Long content wraps so continuation lines align with the start
        of the first line's body (after the ``[<label>] `` or
        ``-> <phase>: `` prefix), keeping the column layout intact.

        Gating: main-scope lines need ``self._verbose``; hook-scope
        lines need ``self._verbose_hook`` (independent flags).
        """
        in_hook = self._log_hook_name is not None
        # Gating
        if in_hook:
            if not self._verbose_hook:
                return
        else:
            if not self._verbose:
                return

        # Compose prefix + body. Only header lines (depth 0, not in
        # hook scope) get the timestamp marker — arrow sub-phase lines
        # stay quiet AND lead with a timestamp-width spacer so ``->``
        # aligns under the header's ``[Label]`` column.
        #
        # Indent shape:
        #   depth 0, not hook  →  ``[ts] [Label] content``
        #   depth ≥1 or hook   →  ``<ts_spacer><(depth-1)*2 spaces>-> phase: content``
        if in_hook:
            arrow_indent = " " * (
                _LOG_TS_PREFIX_WIDTH + max(0, self._log_depth - 1) * 2
            )
            prefix = f"{arrow_indent}-> {self._log_hook_name}: "
            body = str(content) if content else str(label)
            # Preserve red for failure visibility — gray would hide a
            # ``✗`` tool-call failure inside a hook. All other colors
            # collapse to gray (hooks are visually subordinate).
            final_color = "red" if color == "red" else "gray"
        elif self._log_depth == 0:
            ts = datetime.now().strftime("[%H:%M:%S.%f]")[:-4] + "]"
            prefix = f"{ts} [{label}] "
            body = str(content)
            final_color = color
        else:
            arrow_indent = " " * (
                _LOG_TS_PREFIX_WIDTH + max(0, self._log_depth - 1) * 2
            )
            prefix = f"{arrow_indent}-> {label}: "
            body = str(content)
            final_color = color

        # Wrap so continuation lines align with body start, at the
        # fixed ``_LOG_TERMINAL_WIDTH``.
        plain = prefix + body
        if len(plain) <= _LOG_TERMINAL_WIDTH or not body:
            printer.print(plain, color=final_color)
            return
        body_width = max(_LOG_TERMINAL_WIDTH - len(prefix), 20)
        wrapped = textwrap.wrap(
            body,
            width=body_width,
            initial_indent="",
            subsequent_indent="",
            break_long_words=True,
            break_on_hyphens=False,
        )
        cont_indent = " " * len(prefix)
        printer.print(prefix + (wrapped[0] if wrapped else ""), color=final_color)
        for line in wrapped[1:]:
            printer.print(cont_indent + line, color=final_color)

    def _record_action_call(
        self,
        worker: Optional[CognitiveWorker],
        obs: Any,
        decision: Any,
        action_result: Step,
    ) -> None:
        """Record + log an act-phase step.

        Called both from worker OTC (per cycle, ``worker`` supplied) and
        from the dispatcher (per ``yield ActionCall``, ``worker=None``).
        The act result is either an ``ActionResult`` (tool calls →
        ``TOOL_CALLS``) or ``None`` (a content-only finish → ``CONTENT_ONLY``).
        """
        tool_calls = []
        output_type = StepOutputType.CONTENT_ONLY
        result_obj = None

        if action_result is not None and isinstance(action_result, Step):
            result_obj = action_result.result
            if isinstance(result_obj, ActionResult):
                output_type = StepOutputType.TOOL_CALLS
                for r in result_obj.results:
                    tool_calls.append({
                        "tool_name": r.tool_name,
                        "tool_arguments": r.tool_arguments,
                        "tool_result": r.tool_result,
                        "success": r.success,
                        "error": r.error,
                    })

        # Trace storage
        if self._agent_trace is not None:
            self._agent_trace.record_step({
                "name": worker.__class__.__name__ if worker is not None else "workflow",
                "step_content": getattr(decision, "step_content", ""),
                "tool_calls": tool_calls,
                "observation": str(obs) if obs is not None else None,
                "observation_hash": observation_fingerprint(obs),
                "output_type": output_type.value,
            })

        # Log arrow(s). One ``-> result: ...`` line per tool call (so
        # success/failure each get visibility); structured / content-only
        # fall back to a single summary line. Under a ``_hook_log_scope``
        # the "result" label is overridden to the hook name by ``_log``.
        if output_type == StepOutputType.TOOL_CALLS:
            for tc in tool_calls:
                mark = "✓" if tc["success"] else "✗"
                line_color = "green" if tc["success"] else "red"
                content = (
                    f"{tc['tool_name']}({_brief(tc['tool_arguments'])}) "
                    f"{mark} {_brief(tc['tool_result'])}"
                )
                if not tc["success"] and tc["error"]:
                    content += f" — {_brief(tc['error'], n=200)}"
                self._log("result", content, color=line_color)
        else:  # CONTENT_ONLY
            step_content = getattr(decision, "step_content", "")
            if step_content:
                self._log("result", _brief(step_content), color="green")

    def _record_think_unit(
        self,
        worker: CognitiveWorker,
        obs: Any,
        decision: Any,
        cycle: int = 0,
    ) -> None:
        """Log one ``ThinkUnit`` OTC cycle's observation + think phases.

        Called from worker OTC inside ``_run_think_unit`` before
        ``_run_action_call`` fires. The cycle's act phase is logged
        separately by ``_record_action_call`` (invoked inside
        ``_run_action_call`` between the worker hooks). No trace
        storage — the cycle's outcome is captured by the per-cycle
        ``_record_action_call`` trace step.

        ``cycle`` (1-based) — when ``>= 2`` an ``-- cycle N --`` gray
        divider line is emitted before the phase arrows so multi-cycle
        OTC runs are visually delimited. Cycle 1 is the natural opener
        right after the ``[Think]`` header, so no divider for it.
        """
        # Cycle divider (cycle 1+). Gated by ``_verbose`` since this
        # is main-flow output, not hook-scope. Same spacer width as
        # arrow lines so the divider aligns with the surrounding ``->``
        # column.
        if cycle >= 1 and self._verbose:
            divider_indent = " " * (
                _LOG_TS_PREFIX_WIDTH + max(0, self._log_depth - 1) * 2
            )
            printer.print(f"{divider_indent}-- cycle {cycle} --", color="gray")

        if obs is not None:
            self._log("observation", _brief(obs), color="green")
        if decision is not None:
            worker_label = worker.__class__.__name__
            step_content = getattr(decision, "step_content", "") or ""
            content = (
                f"{worker_label}: {_brief(step_content)}" if step_content
                else worker_label
            )
            self._log("think", content, color="cyan")

    def _record_human_call(self, item: "HumanCall", response: str) -> None:
        """Record + log a ``HUMAN_CALL`` step.

        The prompt sits in the ``observation`` slot (closest analog);
        the response and channel land in ``structured_output``.
        """
        response_text = "" if response is None else str(response)

        if self._agent_trace is not None:
            self._agent_trace.record_step({
                "name": "human_call",
                "step_content": f"HumanCall(channel={item.channel or 'default'})",
                "tool_calls": [],
                "observation": item.prompt or None,
                "observation_hash": observation_fingerprint(item.prompt),
                "output_type": StepOutputType.HUMAN_CALL.value,
                "structured_output": {
                    "channel": item.channel,
                    "response": _brief(response_text),
                },
                "structured_output_class": None,
            })

        self._log("result", _brief(response_text), color="purple")

    def _record_llm_call(self, item: LLMCall, result: Any) -> None:
        """Record + log an ``LLM_CALL`` step.

        The prompt sits in the ``observation`` slot; the result is
        serialised into ``structured_output``. ``llm_call_protocol`` on
        the top-level step records which LLM contract was invoked.
        """
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

        if self._agent_trace is not None:
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

        self._log("result", _brief(result), color="white")

    def _record_think_agent(
        self,
        item: "ThinkAgent",
        result: Any,
        worker: AgentWorker,
    ) -> None:
        """Record a ``THINK_AGENT`` trace step.

        The MCP-bridged tool calls fired *inside* the external agent
        each generate their own ``_record_action_call`` entries via the
        decision-channel path (``AgentWorker._emit_decision`` →
        ``_run_think_agent``'s consumer → ``_run_action_call``). This
        record is the outer envelope marking the yield itself.

        ``result`` is the ``AgentResult`` returned by ``worker.arun`` —
        its ``output`` / ``exit_code`` / ``completion`` plus the worker
        / agent class identities are folded into ``structured_output``
        so the unified ``AgentTrace`` is the single source of truth for
        delegate runs. (The worker keeps no result slot — the outcome
        flows through the return value, mirroring ``CognitiveWorker``.)
        """
        output = getattr(result, "output", None)
        result_preview = _brief(output) if output is not None else ""

        # Record the yield-supplied goal verbatim. The "resolved goal"
        # — whatever the delegation's sub-context ``user_input`` was —
        # is already captured by ``AgentTrace``'s top-level ``goal``
        # field (the original arun input) plus any ``EnterAgent`` step
        # in scope. No need to duplicate here.
        goal = item.goal
        step_content = f"ThinkAgent({item.name!r})"
        if goal:
            step_content += f": {_brief(goal, n=200)}"

        structured: Dict[str, Any] = {
            "goal": goal,
            "result": result_preview or None,
            "exit_code": getattr(result, "exit_code", None),
            "completion_signal": getattr(result, "completion", None),
            "worker_class": (
                f"{type(worker).__module__}.{type(worker).__qualname__}"
            ),
        }
        base_agent = getattr(worker, "_agent", None)
        if base_agent is not None:
            structured["agent_class"] = (
                f"{type(base_agent).__module__}.{type(base_agent).__qualname__}"
            )

        if self._agent_trace is not None:
            self._agent_trace.record_step({
                "name": "think_agent",
                "step_content": step_content,
                "tool_calls": [],
                "observation": None,
                "observation_hash": None,
                "output_type": StepOutputType.THINK_AGENT.value,
                "structured_output": structured,
                "structured_output_class": None,
                "think_agent_name": item.name,
            })

        self._log("final", result_preview or "(no return value)", color="yellow")

    def _record_enter_agent(self, item: "EnterAgent", result: Any) -> None:
        """Record + log an ``ENTER_AGENT`` scope-switch closer.

        Minimal marker — the actual steps that ran inside the agent
        scope each produced their own records (ThinkUnit cycles,
        ThinkAgent yields, etc.). This step exists so the trace
        timeline reflects "we entered agent mode here with this goal
        and exited with this final answer".
        """
        result_text = _brief(result) if result is not None else ""

        if self._agent_trace is not None:
            self._agent_trace.record_step({
                "name": "enter_agent",
                "step_content": f"EnterAgent(goal={item.goal!r})",
                "tool_calls": [],
                "observation": None,
                "observation_hash": None,
                "output_type": StepOutputType.ENTER_AGENT.value,
                "structured_output": {
                    "goal": item.goal,
                    "result": result_text or None,
                },
                "structured_output_class": None,
            })

        self._log("final", result_text or "(no return value)", color="yellow")

    def _has_workflow(self) -> bool:
        """Check whether the subclass has overridden on_workflow().

        ``_validate_template_forms`` at class creation guarantees that an
        overridden ``on_workflow`` is always an async generator function,
        so a plain identity check against the base method is sufficient.
        """
        return type(self).on_workflow is not AmphibiousAutoma.on_workflow

    def _has_agent(self) -> bool:
        """Check whether the subclass has overridden on_agent()."""
        return type(self).on_agent is not AmphibiousAutoma.on_agent

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
    
    ############################################################################
    # Entry point — ``arun`` + mode resolution + GraphAutoma router plumbing.
    # ``router`` ferries to ``_agent`` / ``_workflow`` / ``_amphiflow``
    # based on the resolved mode.
    ############################################################################

    def _resolve_mode(self, mode: RunMode) -> RunMode:
        """Resolve and validate the run mode against overridden template methods.

        Resolution (when ``mode is RunMode.AUTO``):

        - both ``on_agent`` and ``on_workflow`` overridden → ``RunMode.AMPHIFLOW``
        - only ``on_workflow`` overridden → ``RunMode.WORKFLOW``
        - only ``on_agent`` overridden → ``RunMode.AGENT``
        - neither overridden → ``RuntimeError``

        Validation (when an explicit mode is supplied):

        - ``RunMode.AGENT`` requires ``on_agent`` overridden
        - ``RunMode.WORKFLOW`` requires ``on_workflow`` overridden
        - ``RunMode.AMPHIFLOW`` requires both ``on_agent`` and ``on_workflow``
          overridden

        Establishing this invariant at the routing boundary lets the
        downstream drivers (``_agent`` / ``_workflow`` / ``_amphiflow``)
        rely on the presence of the templates they need without
        repeatedly re-checking. In particular, ``_amphiflow`` can assume
        ``on_agent`` is always available for step-level and full
        fallback, eliminating defensive ``_has_agent()`` branches.
        """
        has_agent = self._has_agent()
        has_workflow = self._has_workflow()

        if mode is RunMode.AUTO:
            if has_agent and has_workflow:
                return RunMode.AMPHIFLOW
            if has_workflow:
                return RunMode.WORKFLOW
            if has_agent:
                return RunMode.AGENT
            raise RuntimeError(
                f"{type(self).__name__} must override on_agent() or on_workflow()."
            )

        if mode is RunMode.AGENT and not has_agent:
            raise RuntimeError(
                f"{type(self).__name__} requested mode=RunMode.AGENT but "
                f"does not override on_agent()."
            )
        if mode is RunMode.WORKFLOW and not has_workflow:
            raise RuntimeError(
                f"{type(self).__name__} requested mode=RunMode.WORKFLOW but "
                f"does not override on_workflow()."
            )
        if mode is RunMode.AMPHIFLOW and not (has_agent and has_workflow):
            missing = []
            if not has_agent:
                missing.append("on_agent()")
            if not has_workflow:
                missing.append("on_workflow()")
            raise RuntimeError(
                f"{type(self).__name__} requested mode=RunMode.AMPHIFLOW but "
                f"does not override {' and '.join(missing)}."
            )
        return mode

    @worker(is_start=True)
    async def router(self, mode: RunMode, max_consecutive_fallbacks: int) -> str:
        """
        Router worker: dispatches to the correct execution mode.

        ``RunMode.AUTO`` is resolved upstream in ``arun()``, so this worker
        always receives a concrete mode.
        """
        if mode is RunMode.AGENT:
            self._log("Router", "Ferrying to AGENT mode")
            self.ferry_to("_agent")
        elif mode is RunMode.WORKFLOW:
            self._log("Router", "Ferrying to WORKFLOW mode")
            self.ferry_to("_workflow")
        elif mode is RunMode.AMPHIFLOW:
            self._log("Router", f"Ferrying to AMPHIFLOW mode, max_consecutive_fallbacks={max_consecutive_fallbacks}")
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
            or by a worker finishing via empty ``tool_calls``) or the
            OTA context's ``summary()``.
        """
        return_value = await self._invoke_template(
            self.on_agent(self.ota_ctx, self.ctx), scope="agent",
        )
        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or self.ota_ctx.summary()

    @worker(is_output=True)
    async def _workflow(self) -> str:
        """WORKFLOW mode entry point.

        Drives ``on_workflow`` through ``_invoke_template`` with
        ``scope='workflow'``. No fallback — failures propagate.
        ``EnterAgent`` yielded from on_workflow is dispatched through
        ``_dispatch_step``'s recursive ``_invoke_template`` path (works
        without a state machine because there is no fallback to track).

        Returns
        -------
        str
            ``self._final_answer`` (if set by a ``RETURN(value)`` yield)
            or the OTA context's ``summary()``.
        """
        return_value = await self._invoke_template(
            self.on_workflow(self.ota_ctx, self.ctx), scope="workflow",
        )
        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or self.ota_ctx.summary()

    @worker(is_output=True)
    async def _amphiflow(self, max_consecutive_fallbacks: int) -> str:
        """AMPHIFLOW mode entry point + peer state machine.

        Drives interleaved ``on_workflow`` and ``on_agent`` generators
        with step-level fallback for atomic-Call failures. The loop
        intercepts ``RETURN`` (terminate), and defers other primitives
        to ``_dispatch_step``.

        Step-level fallback is the only thing this driver owns above
        plain dispatch: when ``_dispatch_step`` raises and the failed
        primitive was an atomic Call in workflow scope, it counts the
        failure, runs a bounded inline recovery sub-run
        (``_run_fallback_agent``), shapes that sub-run's final answer
        into the failed step's return type, and asends it to the
        resuming workflow. On threshold breach or a workflow
        generator-internal exception, the workflow generator is closed
        and ``on_agent`` runs linearly via ``_invoke_template`` (full
        fallback — workflow does not resume).
        """
        def _is_atomic_step(item: Any) -> bool:
            """
            Whether ``item`` is a recognized framework atomic primitive.
            """
            return isinstance(item, (
                ActionCall, HumanCall, LLMCall,
                EnterAgent, ThinkUnit, ThinkAgent,
            ))
        
        def _describe_atomic_step(item: Any) -> str:
            """
            One-line description of an atomic step for logs / fallback goals.
            """
            if isinstance(item, ActionCall):
                return f"ActionCall(tool_name={item.tool_name!r})"
            if isinstance(item, HumanCall):
                channel = item.channel or "<default>"
                return f"HumanCall(channel={channel!r})"
            if isinstance(item, LLMCall):
                return f"LLMCall(protocol={item.protocol!r})"
            return type(item).__name__
        
        def _shape_fallback_value(item: Any, answer: Any) -> Any:
            """Shape the recovery agent's final answer into the failed step's
            expected return type, so the suspended workflow resumes as if the
            step had produced it. ``answer is None`` (the agent produced
            nothing) degrades to a benign default — chosen so a "void" atomic
            Call (one whose return value the workflow does not use) resumes
            without blowing up downstream code:

            ============================  ====================================
            Failed Call                   Shaped value (``answer`` = agent's)
            ============================  ====================================
            ActionCall                    one ToolResult(result=answer)
            HumanCall                     answer or ""
            LLMCall(protocol="chat")      answer or ""
            LLMCall("structure_output")   answer (best-effort passthrough)
            LLMCall("tool_selector")      ([], answer)
            ============================  ====================================
            """
            if isinstance(item, ActionCall):
                return [
                    ToolResult(
                        tool_name=item.tool_name,
                        tool_arguments=dict(item.tool_args),
                        result=answer,
                        success=True,
                    )
                ]
            if isinstance(item, HumanCall):
                return answer or ""
            if isinstance(item, LLMCall):
                if item.protocol == "chat":
                    return answer or ""
                if item.protocol == "tool_selector":
                    return ([], answer)
                return answer
            return answer

        def _build_fallback_goal(
            item: Any,
            item_label: str,
            error: BaseException,
            fsm: "_AmphiState",
        ) -> str:
            """Goal text fed to on_agent on step-level fallback.

            Tells the agent (a) what failed and why, and (b) that its final
            answer becomes the value the failed step should have returned —
            the workflow resumes with it. There is no tool to call and no
            toolset is touched: the recovery sub-run's conclusion IS the
            resolution.
            """
            if isinstance(item, ActionCall):
                intent = item.description or item.tool_name
            else:
                intent = item_label
            return (
                f"[Workflow fallback] Step {fsm.step_index} failed.\n"
                f"Step intent: {intent}\n"
                f"Failed call: {item_label}\n"
                f"Error: {error}\n\n"
                f"Recover however you see fit. Your final answer becomes the "
                f"value the failed step should have returned — the workflow "
                f"resumes with it as if the step had succeeded. If the failed "
                f"call's return value is not used downstream, a brief "
                f"acknowledgement is fine."
            )
        
        # Initialize the state machine with the workflow generator before `while` loop
        # With workflow as the main focus
        workflow_gen = self.on_workflow(self.ota_ctx, self.ctx)
        self._amphi = _AmphiState(
            workflow_gen=workflow_gen,
            max_consecutive_fallbacks=max_consecutive_fallbacks,
        )
        fsm = self._amphi

        ########################
        # State machine main loop
        ########################
        try:
            while not fsm.should_break:
                # Pick the active generator slot based on scope.
                if fsm.scope == "agent":
                    gen, send = fsm.agent_gen, fsm.agent_send
                    fsm.agent_send = None
                else:
                    gen, send = fsm.workflow_gen, fsm.workflow_send
                    fsm.workflow_send = None

                # Advance the chosen generator.
                try:
                    if send is None:
                        item = await gen.__anext__()
                    else:
                        item = await gen.asend(send)
                except StopAsyncIteration:
                    # If it is Agent Mode stop.
                    if fsm.scope == "agent":
                        # TODO: introduce a symmetric agent → workflow switch primitive.
                        #
                        # Today the handoff is asymmetric: workflow → agent is explicit (the user yields ``EnterAgent``, and
                        # ``_enter_agent`` performs the switch), but agent → workflow is implicit — an on_agent run
                        # is bounded by a snapshot scope (a "sub-task"), and "returning to workflow" is signalled by the
                        # generator naturally exhausting. There is no agent-side primitive that mirrors ``EnterAgent``;
                        # exhaustion under a bounded scope is the only way the state machine learns the agent is done.
                        #
                        # Symmetric design: add an agent-side primitive (e.g. ``enter_workflow(value)``) that the agent tool call
                        # to deliberately hand control back. ``_dispatch_step`` would handle that yield the way this branch
                        # currently handles ``StopAsyncIteration`` — tear down the agent slot, forward ``value`` via
                        # ``workflow_send``, restore ``scope = "workflow"``. Exhaustion would degrade
                        # to a default empty-return fallback (or be a strict-mode error).
                        #
                        # Until that primitive exists, this branch is the single point where the implicit "agent done →
                        # resume workflow / terminate run" decision lives.

                        # Agent generator exhausted.
                        if fsm.agent_mode_stack is not None:
                            await fsm.agent_mode_stack.__aexit__(None, None, None)
                            fsm.agent_mode_stack = None
                        fsm.agent_gen = None

                        # Full-fallback exhaustion: the workflow is dead, the
                        # run is over.
                        if fsm.workflow_gen is None:
                            fsm.should_break = True
                        # User-yielded EnterAgent exhausted: resume the workflow
                        # at the instruction after the EnterAgent yield. (Step-
                        # level fallback never reaches here — it runs inline.)
                        else:
                            fsm.scope = "workflow"
                        continue

                    # If it is Workflow Mode stop. 
                    else:
                        fsm.should_break = True
                        continue 
                except Exception as e:
                    # If agent body raised
                    if fsm.scope == "agent":
                        if fsm.agent_mode_stack is not None:
                            await fsm.agent_mode_stack.__aexit__(type(e), e, e.__traceback__)
                            fsm.agent_mode_stack = None
                        fsm.agent_gen = None
                        raise

                    # If Workflow generator-internal error → full fallback.
                    fsm.workflow_gen = None
                    await self._enter_agent()
                    continue

                # RETURN is a control-flow signal — terminate the FSM
                # loop with the carried value. All other primitives
                # (including EnterAgent) go through the dispatcher.
                if isinstance(item, RETURN):
                    fsm.return_value = item.value
                    fsm.should_break = True
                    continue

                try:
                    outcome = await self._dispatch_step(item, scope=fsm.scope)
                except Exception as e:
                    # Agent-scope error / unknown yield type: no fallback, just propagate the exception.
                    if fsm.scope == "agent" or not _is_atomic_step(item):
                        raise

                    # Set up for fallback info
                    fsm.consecutive_failures += 1
                    fsm.step_index += 1
                    item_label = _describe_atomic_step(item)
                    fsm.failed_steps.append(f"Step {fsm.step_index}: {item_label} — {e}")

                    # Full fallback.
                    if fsm.consecutive_failures >= fsm.max_consecutive_fallbacks:
                        try:
                            await fsm.workflow_gen.aclose()
                        except Exception:
                            pass
                        fsm.workflow_gen = None
                        await self._enter_agent()
                        continue

                    # Step-level fallback. Run a bounded recovery sub-run
                    # inline; its final answer, shaped into the failed step's
                    # return type, is asend-ed to the resuming workflow. No
                    # tool is injected and no toolset is mutated — the recovery
                    # sub-run's conclusion IS the resolution.
                    #
                    # TODO(amphiflow step-level fallback): this recovery
                    # semantics is still awkward and needs a rethink. Mapping
                    # the recovery agent's free-form conclusion (a str) onto the
                    # failed step's *typed* return via ``_shape_fallback_value``
                    # is a loose fit (esp. ActionCall -> List[ToolResult] and
                    # structure_output), and clearing ``_final_answer`` below is
                    # a patch over the recovery sub-run's think-step writing the
                    # shared run state. Revisit: a dedicated recovery-value
                    # channel, or reconsider whether a recovered value should
                    # feed back into the workflow at all. Left as-is for now.
                    else:
                        fallback_goal = _build_fallback_goal(item, item_label, e, fsm)
                        recovered = await self._run_fallback_agent(fallback_goal)
                        fsm.workflow_send = _shape_fallback_value(item, recovered)
                        # The recovered value flows to the workflow via
                        # ``workflow_send`` — it is an internal step value, not
                        # the run's answer. Clear the ``_final_answer`` that the
                        # recovery sub-run's think step set on the shared run
                        # state, so the run's final answer comes from the
                        # resuming workflow (or ``summary()``), never the
                        # internal recovery.
                        self._final_answer = None
                        continue
                else:
                    if fsm.scope == "agent":
                        fsm.agent_send = outcome
                    else:
                        fsm.workflow_send = outcome
                        fsm.consecutive_failures = 0
                        fsm.step_index += 1
        finally:
            # Cleanup order: agent_gen (may be mid-yield) → agent_mode_stack (snapshot) → workflow_gen. 
            if fsm.agent_gen is not None:
                try:
                    await fsm.agent_gen.aclose()
                except Exception:
                    pass
            if fsm.agent_mode_stack is not None:
                try:
                    await fsm.agent_mode_stack.__aexit__(None, None, None)
                except Exception:
                    pass
            if fsm.workflow_gen is not None:
                try:
                    await fsm.workflow_gen.aclose()
                except Exception:
                    pass
            return_value = fsm.return_value
            self._amphi = None

        if return_value is not None:
            self._final_answer = str(return_value)
        return self._final_answer or self.ota_ctx.summary()

    async def arun(
        self,
        *,
        user_input: str = "",
        llm: Optional[BaseLlm] = None,
        context: Optional[ContextT] = None,
        ota_context: Optional[OTAContextT] = None,
        mode: Optional[RunMode] = RunMode.AUTO,
        max_consecutive_fallbacks: int = 1,
        trace: bool = False,
        workdir: Optional[Union[Path, str]] = None,
    ) -> str:
        """Run the agent. Returns a summary of the final context.

        Two contexts, two loops. ``context=`` is the **loop** knowledge
        context (free-form, optional — defaults to an empty ``_context_class``);
        the framework constructs a fresh **small-loop** ``OTAContext`` per run
        from ``_ota_context_class``, seeded with ``user_input`` — unless the
        caller passes a pre-built ``ota_context=``, which is then used verbatim
        (its own ``user_input`` stands). That lets the caller seed per-run
        small-loop state — e.g. inject a per-turn resource the workers/hooks
        read back off ``ota_context``. Pure dispatch:
        the automa only schedules and no longer assembles any toolset — each
        OTA context declares the tools it carries on its class via
        ``OTAContext.tool`` (framework builtins it wants + its own), so whatever
        a context declared is exactly what its ``tools`` field holds.

        ``mode=RunMode.AUTO`` (default) picks AMPHIFLOW / WORKFLOW / AGENT
        from which template methods the subclass overrides.

        ``trace`` and ``workdir`` are orthogonal:

        * ``trace=True`` activates an in-memory ``AgentTrace`` (survives
          on ``self._agent_trace`` after the run for inspection).
        * ``workdir=path`` materialises ``<workdir>/runs/<run_id>/`` —
          the run directory — independent of whether the trace is active.
        * Both set ⇒ ``AgentTrace`` incrementally persists the single
          ``<run>/trace.json`` (goal + metadata + history) — the run
          directory's only artifact.
        * ``trace=False, workdir=path`` ⇒ the run dir is created but
          empty (nothing writes ``trace.json``).

        ``max_consecutive_fallbacks`` (AMPHIFLOW only) is the workflow
        step-failure threshold before switching to full agent mode.
        """
        async def _run_and_report(context: ContextT) -> str:
            """Run the agent, measure time, and log summary."""
            start_time = time.time()
            result = await GraphAutoma.arun(
                self, self._run_mode,
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
        # Config
        self._llm = llm
        self._run_mode = self._resolve_mode(mode if mode is not None else RunMode.AUTO)

        # Trace
        self._read_tracker = {}
        self._current_run_dir = None
        self._agent_trace = None
        
        # State
        self._final_answer = None
        self.spent_tokens = 0
        self.spent_time = 0.0
        self._log_depth = 0
        self._log_hook_name = None

        ########################
        # Run-dir + trace activation — orthogonal axes.
        #   trace=True  → AgentTrace is created (in-memory recorder).
        #   workdir set → <workdir>/runs/<run_id>/ is materialised.
        #   Both set    → AgentTrace also persists trace.json there.
        ########################
        run_id: Optional[str] = None
        if workdir is not None:
            run_id = make_run_id()
            self._current_run_dir = ensure_run_dir(Path(workdir).expanduser().resolve(), run_id)

        if trace:
            self._agent_trace = AgentTrace(workdir=self._current_run_dir)

        ########################
        # Initialize the two contexts (pure dispatch — no toolset assembly).
        #   * Loop — caller-supplied free-form knowledge context, or a fresh
        #            default ``_context_class`` when none was passed.
        #   * OTA  — a caller-supplied pre-built small-loop context, or a fresh
        #            one constructed per run seeded with ``user_input``.
        # Each context already carries its own declared ``tools`` (populated
        # from its class's ``OTAContext.tool`` registrations); the framework
        # does not inject or merge any tools here.
        ########################
        if context is not None and not isinstance(context, self._context_class):
            raise ValueError(
                f"context= must be an instance of {self._context_class.__name__} "
                f"(the loop context), got {type(context).__name__}."
            )
        if ota_context is not None and not isinstance(ota_context, self._ota_context_class):
            raise ValueError(
                f"ota_context= must be an instance of {self._ota_context_class.__name__} "
                f"(the small-loop context), got {type(ota_context).__name__}."
            )

        loop_ctx = context if context is not None else self._context_class()
        ota_ctx = ota_context if ota_context is not None else self._ota_context_class(user_input=user_input)
        self._current_ota_context = ota_ctx
        self._current_context = loop_ctx

        ########################
        # Run the amphibious automa
        ########################
        token = current_agent.set(self)
        try:
            # Trace lifecycle — begin.
            if self._agent_trace is not None:
                self._agent_trace.begin_run(
                    goal=ota_ctx.user_input or "",
                    agent_class=f"{type(self).__module__}.{type(self).__qualname__}",
                    agent_name=self.name,
                    context_class=(
                        f"{self._context_class.__module__}.{self._context_class.__qualname__}"
                        if self._context_class else None
                    ),
                    mode=self._run_mode.value,
                    run_id=run_id,
                    max_consecutive_fallbacks=max_consecutive_fallbacks,
                    start_time=time.time(),
                )
            return await _run_and_report(context=loop_ctx)
        finally:
            # Trace lifecycle — end.
            if self._agent_trace is not None:
                try:
                    self._agent_trace.end_run(
                        end_time=time.time(),
                        spent_tokens=self.spent_tokens,
                        spent_time=self.spent_time,
                    )
                except Exception:
                    pass
            self._current_run_dir = None
            self._run_mode = None
            self._llm = None
            current_agent.reset(token)
