import hashlib
import asyncio
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any, AsyncGenerator, Awaitable, Callable, ClassVar, Dict, FrozenSet, Generic, Iterable, List, Literal, Optional, Tuple, Type, TypeVar, Union,
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
from bridgic.core.agentic.tool_specs import ToolSpec, FunctionToolSpec
from bridgic.core.utils._console import printer
from bridgic.amphibious._context import CognitiveContext, CognitiveTools, CognitiveSkills, CognitiveHistory, Exposure, LayeredExposure
from bridgic.amphibious._cognitive_worker import CognitiveWorker, WorkerRunner, _DELEGATE
from bridgic.amphibious._think_unit import ThinkUnitDescriptor, _ThinkUnitRuntime
from bridgic.amphibious._think_agent import ThinkAgentDescriptor, _ThinkAgentRuntime
from bridgic.amphibious._run_dir import ensure_run_dir, make_run_id
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
from bridgic.amphibious.builtin_tools.human.request_human import build_request_human_tool
from bridgic.amphibious._type import (
    RunMode,
    Step,
    StepToolCall,
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

# Generic type for the agent's cognitive context, allowing users to define their own context classes.
CognitiveContextT = TypeVar("CognitiveContextT", bound=CognitiveContext)

# Built-in tools auto-injected into every AmphibiousAutoma agent's tool set.
_BUILTIN_TOOLS: Tuple[ToolSpec, ...] = ALL_BUILTIN_TOOLS


################################################################################################################
# AgentTrace — flat execution path recorder
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


class AgentTrace:
    """Unified trace recorder for one ``arun`` invocation.

    Owns ALL workdir persistence: when constructed with a non-``None``
    ``workdir``, every lifecycle event and step record triggers an
    incremental write of ``<workdir>/trace.json``. This is the single
    artifact for a run (replacing the legacy ``meta.json`` +
    ``ctx_initial.json`` + ``ctx_final.json`` + ``trace.json`` quartet).

    Trace data layout (``build()`` and the on-disk JSON share it)::

        {
            "goal":     "<the original arun goal>",
            "metadata": {agent_class, agent_name, context_class, mode,
                         run_id, start_time, end_time, spent_tokens,
                         spent_time, cost_time, ...},
            "history":  [TraceStep, ...],  # one entry per yield primitive
        }

    Semantic split from ``CognitiveContext.cognitive_history``: the
    context history is summarised for the agent's own consumption
    (prompts), while the trace history is the detailed audit log of
    every step's outcome.
    """

    def __init__(self, workdir: Optional[Path] = None):
        self._workdir = workdir
        self._goal: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._steps: List[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle — called by ``AmphibiousAutoma.arun``
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Step recording — called by ``_record_*_trace`` in the dispatcher
    # ------------------------------------------------------------------

    def record_step(self, step_data: dict) -> None:
        """Append a step record; persist incrementally."""
        self._steps.append(step_data)
        self._persist()

    # ------------------------------------------------------------------
    # Snapshot / serialization
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
# _AgentSnapshot — async context manager for scoped context mutation
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


class _FallbackSlot:
    """Mailbox for a single step-level fallback's resolved value.

    Created fresh per step-level fallback. Initialized with a benign
    default appropriate for the failed Call's expected return type
    (e.g. ``[]`` for ActionCall, ``""`` for HumanCall). The agent can
    override the default by calling the auto-injected
    ``resolve_step_fallback`` tool, which closes over this slot and
    writes through ``set()``.

    On agent generator exhaustion, the state-machine driver reads
    ``self.value`` and asends it to the workflow generator's failed
    yield, resuming the workflow as if the original Call had returned
    that value.
    """
    __slots__ = ("value",)

    def __init__(self, default: Any) -> None:
        self.value = default

    def set(self, value: Any) -> None:
        self.value = value


def _make_resolve_tool(
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
        ``AsyncExitStack`` holding the agent-mode snapshot. Pushed when
        entering agent mode (via ``EnterAgent`` or step-level fallback);
        popped when the agent generator exhausts or raises.
    fallback_slot
        Set during step-level fallback only — carries the agent's
        recovered value back to the failed workflow yield via
        ``resolve_step_fallback``. ``None`` for user-yielded EnterAgent.
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

    # Fallback state for the current step-level fallback and configuration.
    failed_steps: List[str] = field(default_factory=list)
    fallback_slot: Optional[_FallbackSlot] = None
    max_consecutive_fallbacks: int = 1
    consecutive_failures: int = 0

    # State record for `Amphi`
    step_index: int = 0
    return_value: Any = None
    should_break: bool = False


################################################################################################################
# AmphibiousAutoma
################################################################################################################

class AmphibiousAutoma(GraphAutoma, Generic[CognitiveContextT]):
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
    ``verbose_hook_calls`` (surface dispatch logs for Calls yielded from
    hooks — suppressed by default since hooks are internal side-effects).

    Examples
    --------
    >>> class MyAgent(AmphibiousAutoma[CognitiveContext]):
    ...     main_think = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=20)
    ...     async def on_agent(self, ctx: CognitiveContext):
    ...         yield ThinkUnit("main_think")
    ...
    >>> answer = await MyAgent().arun(llm=llm, goal="Complete the task", tools=[...])
    """

    ############################################################################
    # Class attributes — populated by ``__init_subclass__``
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

        Three responsibilities:

        1. Extract the ``CognitiveContext`` type from the generic
           parameter so ``cls._context_class`` is set.
        2. Build the ``cls._human_channels`` registry by walking the MRO
           and collecting every method tagged via ``@human_channel``.
           Subclass overrides win over parent declarations.
        3. Validate that every overridden template method is an async
           generator (the only shape the dispatch model supports).
        """
        super().__init_subclass__(**kwargs)
        cls._detect_context_class()
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
    # Instance attributes — set up in ``__init__``
    ############################################################################

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        verbose: bool = False,
        verbose_hook_calls: bool = False,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        # User-facing state
        self._llm = None
        self._current_context: Optional[CognitiveContextT] = None
        self._run_mode: Optional[RunMode] = None
        self._verbose = verbose
        self._verbose_hook_calls = verbose_hook_calls

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
    # Template methods — overridable hooks. May be written as async-generator
    # (yielding framework primitives) or plain coroutine; dispatcher accepts
    # both. Scope rules are documented on the class docstring above.
    ############################################################################
    async def observation(self, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent-level default observation, shared across all workers.

        Called before each thinking phase; workers' own ``observation()``
        delegates here when it returns ``_DELEGATE`` / ``None``.

        Yield ``RETURN(text)`` to set ``ctx.observation`` for this cycle.
        Exhausting without ``RETURN`` (or yielding ``RETURN(None)``)
        **preserves** the previous ``ctx.observation`` — so
        ``after_action``-driven refresh patterns work without a dedicated
        passthrough override.

        >>> async def observation(self, ctx):
        ...     snapshot = yield ActionCall("bash", command="bridgic-browser snapshot")
        ...     yield RETURN(snapshot[0].result)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_agent(self, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent mode: LLM-driven cognitive flow.

        Override to declare the agent's strategy. on_agent body is
        reserved for orchestrating cognitive steps — only ``ThinkUnit``
        / ``ThinkAgent`` / ``RETURN`` are allowed (deterministic tool /
        HITL / direct-LLM operations belong in on_workflow or a hook).
        Without ``RETURN``, the framework auto-captures the final answer
        from the last think step's ``step_content``.

        >>> async def on_agent(self, ctx):
        ...     yield ThinkUnit("main_think", max_attempts=20)
        ...     yield ThinkUnit("exec_think", until=lambda c: c.done)
        ...     yield RETURN(ctx.cognitive_history.get_all()[-1].content)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def on_workflow(self, ctx: CognitiveContextT) -> AsyncGenerator[Union[ActionCall, HumanCall, EnterAgent, LLMCall], None]:
        """Workflow mode: deterministic flow as an async generator.

        Override to declare a deterministic workflow. Yield ``ActionCall``
        / ``HumanCall`` / ``LLMCall`` for atomic steps, ``EnterAgent`` to
        enter an autonomous sub-flow, ``RETURN(value)`` to terminate
        early. Use ``result = yield ActionCall(...)`` to receive results
        via ``asend()``.

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
        / ``None``. Yield ``RETURN(modified_decision)`` to override the
        decision; exhausting without RETURN (or returning ``None`` from
        a coroutine override) is passthrough — the original
        ``decision_result`` is preserved.

        >>> async def before_action(self, decision_result, ctx):
        ...     adjusted = sanitize(decision_result)
        ...     yield RETURN(adjusted)
        """
        if False:  # pragma: no cover — async generator stub
            yield

    async def action_tool_call(self, tool_list: List[Tuple[ToolCall, ToolSpec]], context: CognitiveContextT) -> ActionResult:
        """Execute tool calls concurrently and collect results.

        Override to customize tool execution (e.g. sequential, rate
        limiting, sandboxing).
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
        """Handle structured output from a worker with ``output_schema`` set.

        Called instead of ``action_tool_call()`` when the worker
        produces a typed Pydantic instance. Override to post-process or
        validate. Returning ``None`` (e.g. ``pass`` stub) is treated as
        passthrough — the original ``decision_result`` is preserved so
        stub overrides don't silently drop the typed output.
        """
        return decision_result

    async def after_action(self, step_result: Any, ctx: CognitiveContextT) -> AsyncGenerator[Any, Any]:
        """Agent-level after_action hook.

        Called after action execution. Override to update custom
        context fields or trigger follow-up primitives based on
        results. ``RETURN`` is unused here — the hook's return value is
        ignored.

        >>> async def after_action(self, step_result, ctx):
        ...     summary = yield LLMCall.chat(f"Summarize: {step_result}")
        ...     ctx.last_summary = summary
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
    # control signal. ``snapshot`` / ``_phase_context`` provide the
    # scoped-context mechanism EnterAgent and step-level fallback rely on.
    ############################################################################

    async def _invoke_template(
        self,
        gen_or_coro: Any,
        ctx: CognitiveContextT,
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
                    send_value = await self._dispatch_step(item, ctx, scope=scope)
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
        ctx: CognitiveContextT,
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

            # Handle the mode switch.
            result = await self._enter_agent(self.on_agent(ctx), ctx, item=item)
            self._record_enter_agent(item, result)
            return result

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
            
            # Handle the human call and return the response.
            response = await self._run_human_call(item.prompt, channel=item.channel)
            self._record_human_call(item, response)
            return response

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
            
            # Handle the LLM call and return the response.
            result = await self._run_llm_call(item)
            self._record_llm_call(item, result)
            return result

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
            descriptor = getattr(type(self), item.name, None)
            if not isinstance(descriptor, ThinkUnitDescriptor):
                raise AttributeError(
                    f"ThinkUnit(name={item.name!r}) does not match any "
                    f"think_unit declaration on {type(self).__name__}."
                )
            
            # Run the think unit.
            runtime = _ThinkUnitRuntime(descriptor, item)
            return await runtime.run(self, ctx)

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
            
            # Run the delegated agent and return its final answer.
            result = await self._run_think_agent(item, ctx)
            self._record_think_agent(item, result)
            return result

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
            
            # Handle the action call.
            decision = item.decision
            if scope == "hook":
                action_result = await self._run_action_call(decision, ctx, with_hooks=False)
            else:
                obs = await self._invoke_template(self.observation(ctx), ctx)
                if obs is not None:
                    ctx.observation = obs
                action_result = await self._run_action_call(decision, ctx, _worker=None)

            self._record_action_call(
                None, ctx.observation, decision, action_result, ctx,
            )

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
        agent_obj: Any,
        ctx: CognitiveContextT,
        *,
        item: Optional[EnterAgent] = None,
        snapshot_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # 1. If from EnterAgent. Build the snapshot kwargs 
        if item is not None:
            history = (
                item.history if item.history is not None
                else CognitiveHistory()
            )
            snapshot_kwargs = {
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

        # 2. AMPHIFLOW path: hand off to the state machine, return None.
        if self._run_mode is RunMode.AMPHIFLOW:
            fsm = self._amphi
            assert fsm is not None, (
                "AMPHIFLOW run_mode but ``self._amphi`` is None — "
                "``_enter_agent`` was called outside ``_amphiflow``'s "
                "state machine. Check the run-mode / FSM lifecycle."
            )
            if snapshot_kwargs is not None:
                stack = AsyncExitStack()
                try:
                    await stack.__aenter__()
                    await stack.enter_async_context(
                        self.snapshot(**snapshot_kwargs),
                    )
                except BaseException:
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
            return None

        # 3. Inline path: drive the agent to completion and return the inner RETURN value. 
        if snapshot_kwargs is not None:
            async with self.snapshot(**snapshot_kwargs):
                return await self._invoke_template(
                    agent_obj, ctx, scope="agent",
                )
        return await self._invoke_template(
            agent_obj, ctx, scope="agent",
        )
    
    async def _run_human_call(
        self, prompt: str, channel: Optional[str] = None
    ) -> str:
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

        registry = type(self)._human_channels
        if not registry:
            return await _stdin_human_fallback(prompt)
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

    async def _run_llm_call(self, item: LLMCall) -> Any:
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

    async def _run_think_unit(
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
        """Drive one ``CognitiveWorker`` through its observe-think-act cycle.

        Two paths by worker type: a ``CognitiveWorker`` goes through the
        framework's OTC cycle (one cycle per attempt, ``until`` /
        ``max_attempts`` control the loop); a ``WorkerRunner`` gets a
        single ``await worker.run(self, ctx)`` (it owns its own loop, so
        the per-yield overlays are ignored).
        """
        ########################
        # External WorkerRunner.
        ########################
        if not isinstance(worker, CognitiveWorker):
            if isinstance(worker, WorkerRunner):
                await worker.run(self, self._current_context)
                return
            raise TypeError(
                f"_run_think_unit() expects a CognitiveWorker or a WorkerRunner; "
                f"got {type(worker).__name__}."
            )

        worker_label = worker.__class__.__name__

        ########################
        # Setup runtime env.
        ########################
        # Context
        context = self._current_context
        if context is None:
            raise RuntimeError(
                "Cannot call _run_think_unit(): no active context. "
                "_run_think_unit() must be called within an on_agent() method."
            )

        # LLM
        if worker._llm is None and self._llm is not None:
            worker.set_llm(self._llm)
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

        # tools filter
        original_tools: Optional[CognitiveTools] = None
        if tools is not None:
            original_tools = context.tools
            filtered_tools = CognitiveTools()
            for tool in original_tools.get_all():
                if tool.tool_name in tools:
                    filtered_tools.add(tool)
            context.tools = filtered_tools

        # skills filter
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

        # spent-tokens delta tracker
        tokens_before = worker.spent_tokens

        ########################
        # OTC cycle closure
        ########################
        async def _run_observe_think_act() -> bool:
            # 1. Observe (worker → agent fallback)
            obs = await self._invoke_template(worker.observation(context), context)
            if obs is _DELEGATE or obs is None:
                obs = await self._invoke_template(self.observation(context), context)
            if obs is not None:
                context.observation = obs

            obs_str = str(obs) if obs is not None else "None"
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."
            self._log("Observe", f"{worker_label}: {obs_str}", color="green")

            # 2. Think
            decision = await worker.arun(context=context)
            step_str = getattr(decision, 'step_content', str(decision))
            finished = getattr(decision, 'finish', False)
            self._log("Think", f"{worker_label}: finish={finished}, step={step_str}", color="cyan")

            # 3. Act
            action_result = (
                await self._run_action_call(decision, context, _worker=worker)
                if decision is not None else None
            )
            if action_result is not None:
                formatted = action_result.model_dump_json(indent=4)
                self._log("Act", f"{worker_label}:\n{formatted}", color="purple")

            # Trace + auto-captured final answer
            self._record_action_call(worker, obs, decision, action_result, context)
            if decision.finish and decision.step_content:
                self._final_answer = decision.step_content

            return decision.finish

        # Run loop with on_error handling, then restore environment.
        try:
            for _ in range(max_attempts):
                try:
                    finished = await _run_observe_think_act()
                except Exception as e:
                    if on_error == ErrorStrategy.RAISE:
                        raise RuntimeError(
                            f"Worker '{worker_label}' failed during "
                            f"observe-think-act cycle: {e}"
                        ) from e
                    elif on_error == ErrorStrategy.IGNORE:
                        finished = False
                    elif on_error == ErrorStrategy.RETRY:
                        finished = False
                        for attempt in range(max_retries + 1):
                            try:
                                finished = await _run_observe_think_act()
                                break
                            except Exception as retry_e:
                                if attempt == max_retries:
                                    raise RuntimeError(
                                        f"Worker '{worker_label}' failed after "
                                        f"{max_retries + 1} retries: {retry_e}"
                                    ) from retry_e

                if finished:
                    break
                if until is not None:
                    cond_result = until(context)
                    if inspect.iscoroutine(cond_result):
                        cond_result = await cond_result
                    if cond_result:
                        break
        finally:
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

    async def _run_think_agent(
        self,
        item: ThinkAgent,
        ctx: CognitiveContextT,
    ) -> Any:
        descriptor = getattr(type(self), item.name, None)
        if not isinstance(descriptor, ThinkAgentDescriptor):
            raise AttributeError(
                f"ThinkAgent(name={item.name!r}) does not match any "
                f"think_agent declaration on {type(self).__name__}."
            )
        runtime = _ThinkAgentRuntime(descriptor, item)
        result_value = await runtime.run(self, ctx)
        return result_value

    async def _run_action_call(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: Optional[CognitiveWorker] = None,
        with_hooks: bool = True,
    ) -> Step:
        """Execute a thinking decision — the single canonical action executor.

        Routes to ``action_tool_call()`` (tool-call output) or
        ``action_custom_output()`` (custom output_schema), optionally
        wrapped by ``before_action`` / ``after_action`` hooks. When
        ``_worker`` is given AND ``with_hooks`` is True, the worker-level
        hooks run first and delegate to the agent level via ``_DELEGATE``.

        ``with_hooks=False`` skips ALL before/after_action — reserved for
        the hook-scope ActionCall path, where re-entering the hook chain
        would recurse into the generator that yielded the call.
        """
        # Parsing closures — turn a thinking decision into either a
        # matched (ToolCall, ToolSpec) list (tool-call form) or a raw
        # BaseModel (custom-output form).
        def _parse_decision(
            decision: Any,
        ) -> Tuple[bool, Optional[List[StepToolCall]], Any]:
            """Returns ``(is_tool_call_form, calls, decision_result)``.

            * ``is_tool_call_form`` — True when ``decision.output`` is
              declared as ``List[StepToolCall]`` (tool-call path);
              False when it is a BaseModel (custom-output path).
            * ``calls`` — the raw ``StepToolCall`` list when tool-call
              form, else ``None``.
            * ``decision_result`` — ``List[Tuple[ToolCall, ToolSpec]]``
              for tool-call form (ready for ``action_tool_call``), or
              the raw output BaseModel for custom-output form.
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

            def _convert_to_tool_calls(calls: List) -> List[ToolCall]:
                """Convert StepToolCall list into ToolCall objects with type-coerced arguments."""
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

            def _match_tool_calls(tool_calls: List[ToolCall]) -> List[Tuple[ToolCall, ToolSpec]]:
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
                tool_calls = _convert_to_tool_calls(calls)
                return True, calls, _match_tool_calls(tool_calls)
            return False, None, output

        # Execution closure — dispatch to action_tool_call (tool-call
        # form) or action_custom_output (custom-output form) and build
        # the resulting Step.
        async def _execute(
            decision_result: Any,
            *,
            is_tool_call_form: bool,
            calls: Optional[List[StepToolCall]],
        ) -> Step:
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

            # Custom-output form. ``None`` (e.g. from an AI-generated
            # ``pass`` stub of ``action_custom_output``) is treated as
            # passthrough so the typed output is preserved instead of
            # being silently dropped.
            custom_ret = await self.action_custom_output(decision_result, ctx)
            action_result = decision_result if custom_ret is None else custom_ret
            result = Step(content=decision.step_content, result=action_result, metadata={})
            ctx.add_info(result)
            return result

        # Parse
        is_tool_call_form, calls, decision_result = _parse_decision(decision)

        # Before_action hooks — skipped entirely when with_hooks=False.
        if with_hooks:
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

        # Execute
        result = await _execute(
            decision_result,
            is_tool_call_form=is_tool_call_form, calls=calls,
        )

        # After_action hooks — skipped entirely when with_hooks=False.
        if with_hooks:
            if _worker is not None:
                delegate = await self._invoke_template(
                    _worker.after_action(result, ctx), ctx,
                )
                if delegate is _DELEGATE or delegate is None:
                    await self._invoke_template(self.after_action(result, ctx), ctx)
            else:
                await self._invoke_template(self.after_action(result, ctx), ctx)

        return result

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
    # Internal helpers — logging, trace recording, override detection.
    ############################################################################

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

    def _record_action_call(
        self,
        worker: Optional[CognitiveWorker],
        obs: Any,
        decision: Any,
        action_result: Step,
        context: Any,
    ) -> None:
        """Record an act-phase step.

        Called both from worker OTC (per cycle, ``worker`` supplied) and
        from the dispatcher (per ``yield ActionCall``, ``worker=None``).
        Detects the output type from the action_result:

        * tool calls (ActionResult with results) → ``TOOL_CALLS``
        * structured BaseModel output → ``STRUCTURED``
        * everything else (content only) → ``CONTENT_ONLY``
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
                output_type = StepOutputType.STRUCTURED
                structured_output = result_obj.model_dump()
                structured_output_class = (
                    f"{result_obj.__class__.__module__}.{result_obj.__class__.__qualname__}"
                )
            elif result_obj is not None:
                output_type = StepOutputType.STRUCTURED
                try:
                    structured_output = {"__value__": result_obj}
                except Exception:
                    structured_output = {"__value__": str(result_obj)}

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

    def _record_human_call(self, item: "HumanCall", response: str) -> None:
        """Record a ``HUMAN_CALL`` trace step.

        The prompt sits in the ``observation`` slot (closest analog);
        the response and channel land in ``structured_output``.
        """
        if self._agent_trace is None:
            return

        response_text = "" if response is None else str(response)
        response_preview = (
            (response_text[:1000] + "...")
            if len(response_text) > 1000 else response_text
        )

        self._agent_trace.record_step({
            "name": "human_call",
            "step_content": f"HumanCall(channel={item.channel or 'default'})",
            "tool_calls": [],
            "observation": item.prompt or None,
            "observation_hash": observation_fingerprint(item.prompt),
            "output_type": StepOutputType.HUMAN_CALL.value,
            "structured_output": {
                "channel": item.channel,
                "response": response_preview,
            },
            "structured_output_class": None,
        })

    def _record_llm_call(self, item: LLMCall, result: Any) -> None:
        """Record an ``LLM_CALL`` trace step.

        The prompt sits in the ``observation`` slot; the result is
        serialised into ``structured_output``. ``llm_call_protocol`` on
        the top-level step records which LLM contract was invoked.
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

    def _record_think_agent(self, item: "ThinkAgent", result: Any) -> None:
        """Record a ``THINK_AGENT`` trace step.

        The MCP-bridged tool calls fired *inside* the external agent
        each generate their own ``_record_action_call`` entries via the
        ``_dispatch_project_tool`` → ``_run_action_call`` path. This
        record is the outer envelope marking the yield itself.
        """
        if self._agent_trace is None:
            return

        result_preview: Optional[str]
        if result is None:
            result_preview = None
        else:
            text = str(result)
            result_preview = (text[:1000] + "...") if len(text) > 1000 else text

        goal_preview = item.goal or ""
        step_content = f"ThinkAgent({item.name!r})"
        if goal_preview:
            step_content += f": {goal_preview[:200]}{'...' if len(goal_preview) > 200 else ''}"

        self._agent_trace.record_step({
            "name": "think_agent",
            "step_content": step_content,
            "tool_calls": [],
            "observation": None,
            "observation_hash": None,
            "output_type": StepOutputType.THINK_AGENT.value,
            "structured_output": {
                "goal": item.goal,
                "result": result_preview,
            },
            "structured_output_class": None,
            "think_agent_name": item.name,
        })

    def _record_enter_agent(self, item: "EnterAgent", result: Any) -> None:
        """Record an ``ENTER_AGENT`` scope-switch marker.

        Minimal marker — the actual steps that ran inside the agent
        scope each produced their own records (ThinkUnit cycles,
        ThinkAgent yields, etc.). This step exists so the trace
        timeline reflects "we entered agent mode here with this goal
        and exited with this final answer".
        """
        if self._agent_trace is None:
            return

        result_text: Optional[str]
        if result is None:
            result_text = None
        else:
            text = str(result)
            result_text = (text[:1000] + "...") if len(text) > 1000 else text

        self._agent_trace.record_step({
            "name": "enter_agent",
            "step_content": f"EnterAgent(goal={item.goal!r})",
            "tool_calls": [],
            "observation": None,
            "observation_hash": None,
            "output_type": StepOutputType.ENTER_AGENT.value,
            "structured_output": {
                "goal": item.goal,
                "result": result_text,
            },
            "structured_output_class": None,
        })

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

    def _resolve_builtin_filter(
        self, override: Optional[Iterable[str]],
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
        ``_dispatch_step``'s recursive ``_invoke_template`` path (works
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
        """AMPHIFLOW mode entry point + peer state machine.

        Drives interleaved ``on_workflow`` and ``on_agent`` generators
        with step-level fallback for atomic-Call failures. The loop
        intercepts ``RETURN`` (terminate), and defers other primitives
        to ``_dispatch_step``.

        Step-level fallback is the only thing this driver owns above
        plain dispatch: when ``_dispatch_step`` raises and the failed
        primitive was an atomic Call in workflow scope, it counts the
        failure, synthesises a fallback EnterAgent (snapshot with
        ``fallback_goal`` + auto-injected ``resolve_step_fallback``),
        and lets the FSM drive the recovery agent. On threshold
        breach or a workflow generator-internal exception, the
        workflow generator is closed and ``on_agent`` runs linearly
        via ``_invoke_template`` (full fallback — workflow does not
        resume).
        """
        def _is_atomic_step(item: Any) -> bool:
            """
            Whether ``item`` is a recognized framework atomic primitive.
            """
            return isinstance(item, (
                ActionCall, HumanCall, LLMCall,
                EnterAgent, ThinkUnit, ThinkAgent, RETURN,
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
        
        def _make_fallback_slot(item: Any) -> _FallbackSlot:
            """
            Create a fallback slot pre-loaded with a benign default value.

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

        def _build_fallback_goal(
            item: Any,
            item_label: str,
            error: BaseException,
            fsm: "_AmphiState",
        ) -> str:
            """Goal text fed to on_agent on step-level fallback.

            Tells the agent (a) what failed and why, (b) that it should
            recover however it sees fit, and (c) how to feed the result
            back to the workflow via ``resolve_step_fallback``. The
            framework auto-injects that tool for the duration of this
            fallback; calling it once with the recovered value is the only
            way to override the slot's default value.
            """
            if isinstance(item, ActionCall):
                intent = item.decision.step_content or item.tool_name
            else:
                intent = item_label
            return (
                f"[Workflow fallback] Step {fsm.step_index} failed.\n"
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
        
        # Initialize the state machine with the workflow generator before `while` loop
        # With workflow as the main focus
        ctx = self._current_context
        workflow_gen = self.on_workflow(ctx)
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
                        # Today the handoff is asymmetric: workflow → agentis explicit (the user yields ``EnterAgent``, and
                        # ``_enter_agent`` performs the switch), but agent → workflow is implicit — an on_agent run
                        # is bounded by a snapshot scope (a "sub-task"), and "returning to workflow" is signalled by the
                        # generator naturally exhausting. There is no agent-side primitive that mirrors ``EnterAgent``;
                        # exhaustion under a bounded scope is the only way the state machine learns the agent is done.
                        #
                        # Symmetric design: add an agent-side primitive (e.g. ``enter_workflow(value)``) that the agent tool call
                        # to deliberately hand control back. ``_dispatch_step`` would handle that yield the way this branch
                        # currently handles ``StopAsyncIteration`` — tear down the agent slot, forward ``value`` via
                        # ``fallback_slot`` / ``workflow_send``, restore ``scope = "workflow"``. Exhaustion would degrade
                        # to a default empty-return fallback (or be a strict-mode error).
                        #
                        # Until that primitive exists, this branch is the single point where the implicit "agent done →
                        # resume workflow / terminate run" decision lives.

                        # Agent generator exhausted.
                        if fsm.agent_mode_stack is not None:
                            await fsm.agent_mode_stack.__aexit__(None, None, None)
                            fsm.agent_mode_stack = None
                        fsm.agent_gen = None

                        # If Full-fallback exhaustion: workflow is dead.
                        if fsm.workflow_gen is None:  
                            fsm.should_break = True

                        # If Step-level fallback or user-yielded EnterAgent
                        else:
                            if fsm.fallback_slot is not None:
                                fsm.workflow_send = fsm.fallback_slot.value
                                fsm.fallback_slot = None
                            fsm.scope = "workflow"

                        # In all cases, switch to workflow scope
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
                        fsm.fallback_slot = None
                        raise

                    # If Workflow generator-internal error → full fallback.
                    fsm.workflow_gen = None
                    await self._enter_agent(self.on_agent(ctx), ctx)
                    continue

                # RETURN is a control-flow signal — terminate the FSM
                # loop with the carried value. All other primitives
                # (including EnterAgent) go through the dispatcher.
                if isinstance(item, RETURN):
                    fsm.return_value = item.value
                    fsm.should_break = True
                    continue

                try:
                    outcome = await self._dispatch_step(item, ctx, scope=fsm.scope)
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
                        await self._enter_agent(self.on_agent(ctx), ctx)
                        continue

                    # Step-level fallback.
                    else:
                        fsm.fallback_slot = _make_fallback_slot(item)
                        resolve_tool = _make_resolve_tool(fsm.fallback_slot, item)
                        augmented_tools = CognitiveTools()
                        for t in ctx.tools.get_all():
                            augmented_tools.add(t)
                        augmented_tools.add(resolve_tool)
                        fallback_goal = _build_fallback_goal(item, item_label, e, fsm)
                        await self._enter_agent(self.on_agent(ctx), ctx, snapshot_kwargs={"goal": fallback_goal, "tools": augmented_tools})
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
        return self._final_answer or ctx.summary()
    
    async def arun(
        self,
        *,
        llm: Optional[BaseLlm] = None,
        context: Optional[CognitiveContextT] = None,
        mode: Optional[RunMode] = RunMode.AUTO,
        max_consecutive_fallbacks: int = 1,
        trace: bool = False,
        workdir: Optional[Union[Path, str]] = None,
        builtin_tools: Optional[Iterable[str]] = None,
        **kwargs
    ) -> str:
        """Run the agent. Returns a summary of the final context.

        Two ways to initialize context: pass ``context=my_ctx`` to use a
        pre-built one, or pass ``goal=`` / ``tools=`` / ``skills=`` (via
        ``**kwargs``) to have one auto-created.

        ``mode=RunMode.AUTO`` (default) picks AMPHIFLOW / WORKFLOW / AGENT
        from which template methods the subclass overrides.

        ``trace`` and ``workdir`` are orthogonal:

        * ``trace=True`` activates an in-memory ``AgentTrace`` (survives
          on ``self._agent_trace`` after the run for inspection).
        * ``workdir=path`` materialises ``<workdir>/runs/<run_id>/`` so
          ThinkAgent delegate subdirs (``delegates/<n>/...``) have a
          home — independent of whether the trace is active.
        * Both set ⇒ ``AgentTrace`` also incrementally persists
          ``<run>/trace.json`` (goal + metadata + history).
        * ``trace=False, workdir=path`` ⇒ run dir exists for delegate
          artifacts, no ``trace.json`` is written.

        ``builtin_tools`` filters which built-in tools to inject. ``None``
        (default) defers to the class-level ``builtin_tools`` attribute;
        an iterable selects exactly those tools; an empty iterable opts
        out. User-supplied ``tools=[...]`` are never overwritten.

        ``max_consecutive_fallbacks`` (AMPHIFLOW only) is the workflow
        step-failure threshold before switching to full agent mode.
        """
        async def _run_and_report(context: CognitiveContextT) -> str:
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
        self._llm = llm
        self._final_answer = None
        self._read_tracker = {}
        self._current_run_dir = None
        self._agent_trace = None
        self._run_mode = self._resolve_mode(mode if mode is not None else RunMode.AUTO)
        self.spent_tokens = 0
        self.spent_time = 0.0

        ########################
        # Run-dir + trace activation — orthogonal axes.
        #   trace=True  → AgentTrace is created (in-memory recorder).
        #   workdir set → <workdir>/runs/<run_id>/ is materialised so
        #                  ThinkAgent delegate subdirs have a home.
        #   Both set    → AgentTrace also persists to that run dir.
        ########################
        run_id: Optional[str] = None
        if workdir is not None:
            run_id = make_run_id()
            self._current_run_dir = ensure_run_dir(Path(workdir).expanduser().resolve(), run_id)

        if trace:
            self._agent_trace = AgentTrace(workdir=self._current_run_dir)

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
            # Trace lifecycle — begin
            if self._agent_trace is not None:
                self._agent_trace.begin_run(
                    goal=getattr(context, "goal", "") or "",
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
            return await _run_and_report(context=context)
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
