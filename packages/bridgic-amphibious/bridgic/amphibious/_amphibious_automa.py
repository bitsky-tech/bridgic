import asyncio
import inspect
import json
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Annotated,
    Any, AsyncGenerator, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union,
    get_args, get_origin
)

from pydantic import BaseModel

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._automa import RunningOptions
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model.types import ToolCall
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.utils._console import printer
from bridgic.amphibious._context import CognitiveContext, CognitiveTools, CognitiveSkills, Exposure, LayeredExposure
from bridgic.amphibious._cognitive_worker import CognitiveWorker, _DELEGATE
from bridgic.amphibious._type import (
    RunMode,
    Step,
    StepToolCall,
    WorkflowStep,
    AgentFallback,
    ErrorStrategy,
    ActionStepResult,
    ActionResult,
    ToolResult,
    StepOutputType,
    TraceStep,
    RecordedToolCall,
    RunConfig,
    observation_fingerprint,
)


################################################################################################################
# Type Aliases
################################################################################################################

CognitiveContextT = TypeVar("CognitiveContextT", bound=CognitiveContext)


################################################################################################################
# AgentTrace — trace collection (replaces WorkflowBuilder)
################################################################################################################


class _PhaseAccumulator:
    """Accumulates trace steps for a single structural phase (sequential or loop)."""
    __slots__ = ("phase_type", "goal", "steps", "run_config")

    def __init__(self, phase_type: str, goal: Optional[str] = None):
        self.phase_type: str = phase_type  # "sequential" | "loop"
        self.goal: Optional[str] = goal
        self.steps: List[dict] = []
        self.run_config: Optional[dict] = None


class AgentTrace:
    """Stack-based builder that captures phase-annotated trace steps.

    ``begin_phase()`` / ``end_phase()`` bracket a structural phase.
    ``record_step()`` routes step data to the active phase (or orphan list).
    ``set_run_config()`` attaches run() parameters to the current phase.
    ``build()`` returns the collected phases as structured dicts for inspection.
    """

    def __init__(self):
        self._stack: List[_PhaseAccumulator] = []
        self._completed: List[_PhaseAccumulator] = []
        self._orphan_steps: List[dict] = []

    def begin_phase(self, phase_type: str, goal: Optional[str] = None) -> None:
        self._stack.append(_PhaseAccumulator(phase_type, goal=goal))

    def end_phase(self) -> None:
        if self._stack:
            self._completed.append(self._stack.pop())

    def record_step(self, step_data: dict) -> None:
        """Route a trace step to the active phase or the orphan list."""
        if self._stack:
            self._stack[-1].steps.append(step_data)
        else:
            self._orphan_steps.append(step_data)

    def set_run_config(self, config: dict) -> None:
        """Attach run() parameters to the current (topmost) phase."""
        if self._stack:
            self._stack[-1].run_config = config

    def is_loop_pattern_confirmed(self) -> bool:
        """Check if the current loop phase has detected a repeating pattern (probe mode)."""
        if not self._stack:
            return False
        acc = self._stack[-1]
        if acc.phase_type != "loop":
            return False
        # Minimal implementation: require at least 6 steps to consider pattern detected
        return len(acc.steps) >= 6

    @contextmanager
    def phase(self, phase_type: str, goal: Optional[str] = None):
        """Context manager that brackets a structural phase."""
        self.begin_phase(phase_type, goal=goal)
        try:
            yield
        finally:
            self.end_phase()

    def build(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return collected trace data as a structured dict."""
        phases = []
        for phase in self._completed:
            trace_steps = [
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
                )
                for s in phase.steps
            ]
            run_config = RunConfig(**(phase.run_config or {"worker_class": "unknown"}))
            phases.append({
                "phase_type": phase.phase_type,
                "goal": phase.goal,
                "run_config": run_config,
                "steps": trace_steps,
            })

        orphan_steps = [
            TraceStep(
                name=s["name"],
                step_content=s.get("step_content", ""),
                tool_calls=[RecordedToolCall(**tc) for tc in s.get("tool_calls", [])],
                observation=s.get("observation"),
                observation_hash=s.get("observation_hash"),
                output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                structured_output=s.get("structured_output"),
                structured_output_class=s.get("structured_output_class"),
            )
            for s in self._orphan_steps
        ]

        return {
            "phases": phases,
            "orphan_steps": orphan_steps,
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
# _AgentSnapshot — async context manager for safe field mutation + revealed state management
################################################################################################################

class _AgentSnapshot:
    """
    Async context manager for exception-safe temporary field overrides on a Context,
    with additional management of LayeredExposure._revealed state.

    Used internally by ``sequential()`` and ``loop()`` context managers in AmphibiousAutoma.

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
# ThinkUnit — Descriptor-based think step declaration
################################################################################################################

class _BoundThinkUnit:
    """A think unit bound to a specific agent instance. Supports await and .until().

    Created by ThinkUnitDescriptor.__get__() — not instantiated directly by users.
    """

    def __init__(self, agent: "AmphibiousAutoma", descriptor: "ThinkUnitDescriptor"):
        self._agent = agent
        self._desc = descriptor

    def __await__(self):
        """Support ``await self.main_think`` syntax."""
        return self._execute().__await__()

    async def _execute(
        self,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: Optional[int] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> Any:
        """Execute the think unit. Clones the worker template for state isolation."""
        desc = self._desc
        worker = self._clone_worker(desc._worker_template)

        await self._agent._run(
            worker,
            until=until if until is not None else desc._until,
            max_attempts=max_attempts if max_attempts is not None else desc._max_attempts,
            tools=tools if tools is not None else desc._tools,
            skills=skills if skills is not None else desc._skills,
            on_error=desc._on_error,
            max_retries=desc._max_retries,
        )

        # Return output_schema result if the worker has one
        if worker.output_schema is not None:
            ctx = self._agent._current_context
            if ctx is not None and len(ctx.cognitive_history) > 0:
                last_step = ctx.cognitive_history.get_all()[-1]
                if last_step.result is not None:
                    return last_step.result
        return None

    async def until(
        self,
        condition: Union[Callable[..., bool], Callable[..., Awaitable[bool]]],
        *,
        max_attempts: Optional[int] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> Any:
        """Execute with a dynamic loop condition, optionally overriding parameters.

        Usage::

            await self.exec_think.until(
                lambda ctx: not ctx.last_step_has_tools,
                max_attempts=50,
                tools=["save_information"],
            )
        """
        return await self._execute(
            until=condition,
            max_attempts=max_attempts,
            tools=tools,
            skills=skills,
        )

    @staticmethod
    def _clone_worker(template: CognitiveWorker) -> CognitiveWorker:
        """Clone a worker from its template for state isolation.

        Copies configuration (policies, output_schema, verbose settings) but
        creates a fresh instance with clean runtime state (tokens, time,
        GraphAutoma execution state). LLM is left as None — injected by
        the agent at runtime via _run().
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


class ThinkUnitDescriptor:
    """Python descriptor enabling class-level think unit declaration.

    When accessed on an instance (``self.main_think``), returns a
    ``_BoundThinkUnit`` that is awaitable and supports ``.until()``.

    When accessed on the class, returns the descriptor itself.
    """

    def __init__(
        self,
        worker: CognitiveWorker,
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

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name (Python 3.6+ descriptor protocol)."""
        self._attr_name = name

    def __get__(self, obj: Any, objtype: type = None) -> Any:
        if obj is None:
            return self  # Class-level access returns the descriptor
        return _BoundThinkUnit(obj, self)


def think_unit(
    worker: CognitiveWorker,
    *,
    until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
    max_attempts: int = 1,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 0,
) -> ThinkUnitDescriptor:
    """Declare a think unit for use in on_agent().

    Factory function that returns a ThinkUnitDescriptor. Use as a class variable::

        class MyAgent(AmphibiousAutoma[MyContext]):
            main_think = think_unit(
                CognitiveWorker.inline("Plan ONE immediate next step"),
                max_attempts=80,
                on_error=ErrorStrategy.RAISE,
            )

            async def on_agent(self, ctx):
                await self.main_think

    Parameters
    ----------
    worker : CognitiveWorker
        The worker template. A fresh clone is created for each execution.
    until : Optional callable
        Loop condition: repeats until this returns True or LLM signals finish.
    max_attempts : int
        Maximum execution attempts (default 1 = single shot).
    tools : Optional[List[str]]
        Tool filter: only these tools are visible to the worker.
    skills : Optional[List[str]]
        Skill filter: only these skills are visible to the worker.
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


################################################################################################################
# AmphibiousAutoma
################################################################################################################

class AmphibiousAutoma(GraphAutoma, Generic[CognitiveContextT]):
    """
    Base class for amphibious agents — dual-mode orchestration engine.

    Supports two execution modes:
    - **Agent mode** (``on_agent``): LLM-driven observe-think-act cycles via think_unit
    - **Workflow mode** (``on_workflow``): Deterministic step execution via yield

    Subclasses define behavior by implementing ``on_agent()`` and/or ``on_workflow()``.

    Parameters
    ----------
    llm : Optional[BaseLlm]
        Default LLM for workers and auxiliary tasks (e.g., history compression).
        Individual workers can specify their own LLM.
    name : Optional[str]
        Optional name for the agent instance.
    verbose : bool
        Enable logging of execution summary (tokens, time). Default is False.

    Examples
    --------
    >>> class MyAgent(AmphibiousAutoma[CognitiveContext]):
    ...     main_think = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=20)
    ...     async def on_agent(self, ctx: CognitiveContext):
    ...         await self.main_think
    ...
    >>> ctx = await MyAgent(llm=llm).arun(goal="Complete the task", tools=[...])
    """

    _context_class: Optional[Type[CognitiveContext]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Extract the CognitiveContext type from the generic parameter."""
        super().__init_subclass__(**kwargs)
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
        
        # Check if there are any initialization issues with the context generic class.
        if cls._context_class is None or not issubclass(cls._context_class, CognitiveContext):
            raise TypeError(
                f"{cls.__name__} must specify a CognitiveContext type via generic parameter, "
                f"e.g., class {cls.__name__}(AmphibiousAutoma[MyContext])"
            )

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        self._llm = llm
        self._current_context: Optional[CognitiveContextT] = None
        self._verbose = verbose

        # Trace capture
        self._agent_trace: Optional[AgentTrace] = None

        # Usage stats (reset per arun call)
        self.spend_tokens: int = 0
        self.spend_time: float = 0.0

    @property
    def llm(self) -> Optional[Any]:
        """Access the agent's default LLM."""
        return self._llm

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################
    async def observation(self, ctx: CognitiveContextT) -> Optional[str]:
        """
        Agent-level default observation, shared across all workers.

        Called by ``_run()`` before each thinking phase. Workers can
        enhance this via their own ``observation()`` method.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context.

        Returns
        -------
        Optional[str]
            Custom observation text, or None.
        """
        return None

    @abstractmethod
    async def on_agent(self, ctx: CognitiveContextT) -> None:
        """
        Agent mode: LLM-driven orchestration logic.

        Subclasses implement this method to define how think_units are
        orchestrated. Use standard Python control flow combined with
        ``await self.think_unit`` calls.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context, used for accessing state and conditions.

        Examples
        --------
        >>> async def on_agent(self, ctx: MyContext):
        ...     await self.main_think
        ...     await self.exec_think.until(lambda ctx: ctx.done, max_attempts=20)
        """
        ...

    async def on_workflow(self, ctx: CognitiveContextT) -> AsyncGenerator[Union[WorkflowStep, AgentFallback], None]:
        """Workflow mode: deterministic execution as an async generator.

        Override this method to define a deterministic workflow. When overridden,
        ``arun()`` automatically routes to workflow mode instead of ``on_agent()``.

        Yield ``WorkflowStep`` for deterministic tool execution, or
        ``AgentFallback`` to delegate a sub-task to agent mode.

        The generator exhausting signals workflow completion — no finish signal needed.
        Use ``result = yield step(...)`` to receive tool execution results via asend().

        Examples
        --------
        >>> async def on_workflow(self, ctx):
        ...     yield step("navigate_to", url="http://example.com")
        ...     result = yield step("click_element_by_ref", ref="42")
        ...     yield AgentFallback(goal="Handle complex case", max_attempts=10)
        """
        ...

    async def before_action(
        self,
        decision_result: Any,
        ctx: CognitiveContextT,
    ) -> Any:
        """
        Agent-level before_action hook, shared across all workers.

        Called when a worker's ``before_action()`` returns ``_DELEGATE``.
        Override to intercept and modify tool calls at the agent level.

        Parameters
        ----------
        decision_result : Any
            The decision result (typically List[Tuple[ToolCall, ToolSpec]]).
        ctx : CognitiveContextT
            The cognitive context.

        Returns
        -------
        Any
            Verified/adjusted decision result.
        """
        return decision_result

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
        """
        return decision_result

    ############################################################################
    # Core methods
    ############################################################################
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
        Execute a CognitiveWorker through observe-think-act cycle(s).

        Internal method used by think_unit descriptors and _run_workflow.
        Not intended for direct use by subclasses — use think_unit instead.

        Parameters
        ----------
        worker : CognitiveWorker
            The worker to execute.
        until : Optional callable
            Loop condition: if provided, the step repeats until this returns True
            or LLM signals finish. Supports sync and async callables.
        max_attempts : int
            Maximum execution attempts (default 1 = single shot).
        tools : Optional[List[str]]
            Tool filter: only these tools are visible to the worker.
        skills : Optional[List[str]]
            Skill filter: only these skills are visible to the worker.
        on_error : ErrorStrategy
            Error handling strategy.
        max_retries : int
            Max retries for RETRY strategy.
        """
        async def _execute():
            if until is not None or max_attempts > 1:
                for _ in range(max_attempts):
                    finished = await self._run_once(
                        worker, tools=tools, skills=skills,
                        on_error=on_error, max_retries=max_retries,
                    )
                    if finished:
                        return
                    # Early termination — stop once loop pattern is confirmed
                    if (self._agent_trace
                            and self._agent_trace.is_loop_pattern_confirmed()):
                        self._log(
                            "Workflow",
                            "Loop pattern confirmed after "
                            f"{len(self._agent_trace._stack[-1].steps)} steps "
                            "— stopping early (probe mode)",
                            color="green",
                        )
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

        context = self._current_context
        if context is None:
            raise RuntimeError(
                "Cannot call _run(): no active context. "
                "_run() must be called within an on_agent() method."
            )

        # Capture run config for trace
        if self._agent_trace:
            self._agent_trace.set_run_config({
                "worker_class": f"{worker.__class__.__module__}.{worker.__class__.__qualname__}",
                "tools": tools,
                "skills": skills,
                "max_attempts": max_attempts,
                "on_error": on_error.value,
            })

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
            obs = await worker.observation(context)
            if obs is _DELEGATE:
                obs = await self.observation(context)
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

        # Init skills
        original_skills = None
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

        # Init spend status
        tokens_before = worker.spend_tokens

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
            self.spend_tokens += worker.spend_tokens - tokens_before
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

    @asynccontextmanager
    async def sequential(self, *, goal: Optional[str] = None,
                         keep_revealed: Optional[Dict[str, List[int]]] = None,
                         **snapshot_fields):
        """
        Mark a structural phase as sequential for workflow capture.

        Use as an async context manager around ``_run()`` calls to group them
        into a ``SequentialBlock`` when ``trace_running=True``.

        Parameters
        ----------
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("sequential",
                                       keep_revealed=keep_revealed,
                                       goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def loop(self, *, goal: Optional[str] = None,
                   keep_revealed: Optional[Dict[str, List[int]]] = None,
                   **snapshot_fields):
        """
        Mark a structural phase as a loop for workflow capture.

        Use as an async context manager around ``_run()`` calls to group them
        into a ``LoopBlock`` with pattern detection when ``trace_running=True``.

        Parameters
        ----------
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("loop",
                                       keep_revealed=keep_revealed,
                                       goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def snapshot(self, *, goal: Optional[str] = None,
                   keep_revealed: Optional[Dict[str, List[int]]] = None,
                   **snapshot_fields):
        """
        Temporarily override context fields for the duration of the block.

        A non-structural snapshot that does not create a named phase in the trace.
        Use ``sequential()`` or ``loop()`` when you need phase annotation.

        Parameters
        ----------
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("snapshot",
                                       keep_revealed=keep_revealed,
                                       goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def _phase_context(self, phase_type: str, *,
                             keep_revealed: Optional[Dict[str, List[int]]] = None,
                             **snapshot_fields):
        """Shared implementation for sequential() and loop() context managers.

        Parameters
        ----------
        phase_type : str
            Phase type identifier (``"sequential"`` or ``"loop"`` or ``"snapshot"``).
        keep_revealed : Optional[Dict[str, List[int]]]
            Revealed state management mode for ``_AgentSnapshot``.
        **snapshot_fields
            Context fields to temporarily override during this phase.
        """
        # Init the variables
        context = self._current_context
        _phase_goal = snapshot_fields.pop("_phase_goal", None)
        fields = {k: v for k, v in snapshot_fields.items() if v is not None}
        phase_type = phase_type if phase_type != "snapshot" else "sequential"

        # If goal was passed as a snapshot field (e.g. sequential(goal=...)),
        # use it as the phase goal for the workflow builder as well.
        if _phase_goal is None and "goal" in fields:
            _phase_goal = fields["goal"]

        if not fields and keep_revealed is None and _phase_goal is None:
            raise ValueError(
                f"_phase_context('{phase_type}'): no snapshot fields, keep_revealed, "
                f"or goal provided. If no context state needs to be scoped for "
                f"this phase, use _run() directly — the behavior is identical."
            )

        # Create the snapshot
        snap = _AgentSnapshot(context, fields, keep_revealed=keep_revealed)
        await snap.__aenter__()
        if self._agent_trace:
            self._agent_trace.begin_phase(phase_type, goal=_phase_goal)
        try:
            yield
        finally:
            if self._agent_trace:
                self._agent_trace.end_phase()
            await snap.__aexit__(None, None, None)

    async def _run_workflow(
        self,
        ctx: CognitiveContextT,
        *,
        will_fallback: bool = True,
        max_consecutive_fallbacks: int = 1,
    ) -> None:
        """Consume on_workflow() generator, executing each step.

        Uses asend() to return tool execution results (List[ToolResult]) back
        to the generator, enabling ``result = yield step(...)`` syntax.

        For each yielded item:
        - ``WorkflowStep``: run observe → log → act deterministically.
          If execution fails, fall back to agent mode for the current step.
        - ``AgentFallback``: delegate to ``_run()`` for LLM-driven execution.

        If consecutive failures exceed ``max_consecutive_fallbacks``, abandon
        the workflow and delegate the remaining task to ``on_agent()``
        (full agent mode).
        """
        consecutive_failures = 0
        if not will_fallback:
            max_consecutive_fallbacks = 0
        step_index = 0
        failed_steps = []  # Track failed step info for error reporting

        gen = self.on_workflow(ctx)
        send_value = None  # First iteration uses None (equivalent to __anext__)

        try:
            while True:
                # Get next item from generator (send previous result back)
                try:
                    if send_value is None:
                        item = await gen.__anext__()
                    else:
                        item = await gen.asend(send_value)
                    send_value = None  # Reset for next iteration
                except StopAsyncIteration:
                    break

                if isinstance(item, AgentFallback):
                    fallback_worker = item.worker if item.worker is not None else self._create_fallback_worker()
                    await self._run(
                        fallback_worker,
                        max_attempts=item.max_attempts,
                        tools=item.tools or None,
                        skills=item.skills or None,
                    )
                    consecutive_failures = 0
                    send_value = None
                    continue

                # Deterministic execution
                worker = item.worker
                decision = item.decision

                # 1. Observe
                if worker is not None:
                    obs = await worker.observation(ctx)
                    if obs is _DELEGATE:
                        obs = await self.observation(ctx)
                else:
                    obs = await self.observation(ctx)
                ctx.observation = obs

                obs_str = str(obs) if obs is not None else "None"
                if len(obs_str) > 200:
                    obs_str = obs_str[:200] + "..."
                self._log("Observe", f"workflow: {obs_str}", color="green")
                self._log("Think", f"workflow: {decision.step_content}", color="cyan")

                # 2. Act — with fallback on failure
                try:
                    action_result = await self._action(decision, ctx, _worker=worker)

                    # Check if any tool execution failed
                    inner = getattr(action_result, "result", None)
                    if (
                        inner is not None
                        and hasattr(inner, "results")
                    ):
                        failed = [
                            r for r in inner.results
                            if not r.success
                        ]
                        if failed:
                            errors = "; ".join(
                                f"{r.tool_name}: {r.error}" for r in failed
                            )
                            raise RuntimeError(
                                f"Tool execution failed for: "
                                f"{decision.step_content} — {errors}"
                            )

                    if action_result is not None:
                        formatted = action_result.model_dump_json(indent=4)
                        self._log("Act", f"workflow:\n{formatted}", color="purple")

                    # 3. Record trace step (if capturing)
                    self._record_trace_step(worker, obs, decision, action_result, ctx)
                    consecutive_failures = 0
                    step_index += 1

                    # 4. Build ToolResult list to send back via asend()
                    send_value = self._build_tool_results(action_result)

                except Exception as e:
                    if not will_fallback:
                        raise e

                    consecutive_failures += 1
                    step_index += 1
                    failed_steps.append(f"Step {step_index}: {decision.step_content} — {e}")
                    self._log(
                        "Workflow",
                        f"Step {step_index} failed "
                        f"({consecutive_failures}/{max_consecutive_fallbacks}): {e}",
                        color="red",
                    )

                    # Check if consecutive failures exceed threshold → full fallback
                    if consecutive_failures >= max_consecutive_fallbacks:
                        if not self._has_agent():
                            failed_summary = "\n".join(failed_steps)
                            raise RuntimeError(
                                f"Workflow degradation failed: consecutive failures reached "
                                f"{max_consecutive_fallbacks}, but on_agent() is not overridden "
                                f"so fallback cannot proceed.\n"
                                f"Failed steps:\n{failed_summary}"
                            )
                        self._log(
                            "Workflow",
                            f"Consecutive failures reached {max_consecutive_fallbacks}, "
                            f"falling back to full agent mode (on_agent).",
                            color="red",
                        )
                        await self.on_agent(ctx)
                        return

                    # Step-level fallback: construct a scoped goal and let agent fix it
                    fallback_goal = (
                        f"[Workflow fallback] Step {step_index} failed.\n"
                        f"Step intent: {decision.step_content}\n"
                        f"Error: {e}\n\n"
                        f"You must do TWO things:\n"
                        f"1. Resolve the error — fix whatever is blocking this step (e.g. login, navigate, wait for page load).\n"
                        f"2. Complete the original step intent: {decision.step_content}\n\n"
                        f"Set finish=True ONLY after both are done. "
                        f"Do NOT continue with subsequent steps."
                    )
                    self._log(
                        "Workflow",
                        f"Falling back to agent mode for step {step_index}: "
                        f"{decision.step_content}",
                        color="yellow",
                    )
                    async with self.snapshot(goal=fallback_goal):
                        await self._run(
                            worker if worker is not None else self._create_fallback_worker(),
                            max_attempts=5,
                        )
                    send_value = None  # No result to send back after fallback
        finally:
            await gen.aclose()

    # TODO: This locally downgraded CognitiveWorker needs to have a good way to be opened up 
    # in the future to allow for customization.
    def _create_fallback_worker(self) -> CognitiveWorker:
        """Create a default fallback worker for workflow mode.

        Used when a WorkflowStep or AgentFallback has no explicit worker.
        The worker's observation() and before_action() both return _DELEGATE,
        delegating to the agent-level hooks. Inherits the full tool/skill set
        configured via arun().

        Override this method to provide a custom fallback worker.
        """
        return CognitiveWorker.inline(
            "Fix the current error you observe, then complete the step that failed. "
            "Work in ReAct style: observe the situation, reason about the next action, "
            "act once, then observe again. Take one step at a time—do not plan ahead; "
            "react to what you see after each action.",
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

    async def _action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: Optional[CognitiveWorker] = None,
    ) -> Step:
        """
        Execute the action phase based on the thinking decision.

        Routes to ``action_tool_call()`` for tool-call output or
        ``action_custom_output()`` for structured output (output_schema).
        Calls ``before_action()`` on both the worker and agent level
        (with delegation via ``_DELEGATE``).

        Parameters
        ----------
        decision : Any
            The thinking decision with 'output' field (List[StepToolCall] or BaseModel).
        ctx : CognitiveContextT
            The cognitive context.
        _worker : Optional[CognitiveWorker]
            The worker that produced this decision (used for before_action callback).
        """ 
        def _is_list_step_tool_call(d: Any) -> bool:
            # Get the declared type of the output field
            if not isinstance(d, BaseModel):
                return False
            fi = type(d).model_fields.get('output')
            if fi is None:
                return False
            ann = fi.annotation
            if get_origin(ann) is Annotated:
                ann = get_args(ann)[0]

            # if the type is not List[StepToolCall], return False
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

        ########################
        # Analysis of the decision output
        ########################
        # Get the output structure of the decision
        output = getattr(decision, 'output', None)

        # If the output is a list of StepToolCall
        if _is_list_step_tool_call(decision):
            calls = output
            tool_calls = _convert_decision_to_tool_calls(calls, ctx)
            decision_result = _match_tool_calls(tool_calls, ctx)

        # If the output is a BaseModel
        else:
            decision_result = output


        ########################
        # Execution of the action based on the decision result
        ########################
        # before_action delegation: worker → agent (if _DELEGATE)
        original_decision_result = decision_result
        if _worker is not None:
            decision_result = await _worker.before_action(decision_result, ctx)
            if decision_result is _DELEGATE:
                decision_result = await self.before_action(original_decision_result, ctx)
        else:
            decision_result = await self.before_action(decision_result, ctx)
        result = None
        if _is_list_step_tool_call(decision):
            if not calls:
                result = Step(
                    content=decision.step_content,
                    result=None,
                    metadata={"tool_calls": []}
                )
                ctx.add_info(result)
            else:
                action_result = await self.action_tool_call(decision_result, ctx)
                result = Step(
                    content=decision.step_content,
                    result=action_result,
                    metadata={}
                )
                ctx.add_info(result)
        else:
            action_result = await self.action_custom_output(decision_result, ctx)
            result = Step(content=decision.step_content, result=action_result, metadata={})
            ctx.add_info(result)

        return result

    ############################################################################
    # Internal Helper methods
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

    def _has_workflow(self) -> bool:
        """Check whether the subclass has overridden on_workflow()."""
        return type(self).on_workflow is not AmphibiousAutoma.on_workflow

    def _has_agent(self) -> bool:
        """Check whether the subclass has overridden on_agent()."""
        return type(self).on_agent is not AmphibiousAutoma.on_agent

    ############################################################################
    # Entry point
    ############################################################################
    @worker(is_start=True)
    async def router(self, mode: RunMode, will_fallback: bool, max_consecutive_fallbacks: int) -> str:
        """
        Router worker: determines and dispatches to the correct execution mode.
        """
        if mode is RunMode.AGENT:
            self._log("Router", "Ferrying to AGENT mode", color="green")
            self.ferry_to("_cognition")
        elif mode is RunMode.WORKFLOW:
            self._log("Router", "Ferrying to WORKFLOW mode", color="green")
            self.ferry_to("_workflow")
        elif mode is RunMode.AMPHIBIOUS:
            self._log("Router", f"Ferrying to AMPHIBIOUS mode, will_fallback={will_fallback}, max_consecutive_fallbacks={max_consecutive_fallbacks}", color="green")
            self.ferry_to("_amphibious", will_fallback=will_fallback, max_consecutive_fallbacks=max_consecutive_fallbacks)
        elif mode is RunMode.AUTO:
            self._log("Router", "Auto-detecting execution mode", color="green")
            has_workflow = self._has_workflow()
            if has_workflow:
                self._log("Router", f"Detected AMPHIBIOUS mode (on_workflow overridden), will_fallback={will_fallback}, max_consecutive_fallbacks={max_consecutive_fallbacks}", color="green")
                self.ferry_to("_amphibious", will_fallback=will_fallback, max_consecutive_fallbacks=max_consecutive_fallbacks)
            else:
                self._log("Router", "Detected AGENT mode", color="green")
                self.ferry_to("_cognition")

    @worker(is_output=True)
    async def _cognition(self) -> str:
        """
        Agent mode entry point: delegates to the user-defined on_agent() method.
        """
        await self.on_agent(self._current_context)
        return self._current_context.summary()

    @worker(is_output=True)
    async def _workflow(self) -> str:
        """
        Workflow mode entry point: runs on_workflow() without agent fallback.
        """
        await self._run_workflow(self._current_context, will_fallback=False)
        return self._current_context.summary()

    @worker(is_output=True)
    async def _amphibious(self, will_fallback: bool, max_consecutive_fallbacks: int) -> str:
        """
        Entry point: runs on_workflow() with agent fallback support (amphibious mode).
        """
        await self._run_workflow(self._current_context, will_fallback=will_fallback, max_consecutive_fallbacks=max_consecutive_fallbacks)
        return self._current_context.summary()

    async def arun(
        self,
        *,
        context: Optional[CognitiveContextT] = None,
        trace_running: bool = False,
        mode: Optional[RunMode] = RunMode.AUTO,
        will_fallback: bool = True,
        max_consecutive_fallbacks: int = 1,
        **kwargs
    ) -> str:
        """
        Run the agent.

        Routes to one of the execution modes:
        1. Agent mode — LLM-driven ``on_agent()`` path.
        2. Workflow mode — deterministic ``on_workflow()`` path (no fallback).
        3. Amphibious mode — ``on_workflow()`` with automatic agent fallback.
        4. Auto mode (default) — amphibious if ``on_workflow()`` is overridden,
           otherwise agent.

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
            ``RunMode.AMPHIBIOUS`` forces workflow with agent fallback,
            ``RunMode.AUTO`` (default) auto-detects based on whether
            ``on_workflow()`` is overridden.
        will_fallback : bool
            Whether workflow failures can fall back to agent mode.
            Only applies to amphibious mode. Default is True.
        max_consecutive_fallbacks : int
            Maximum consecutive workflow step failures before switching
            to full agent mode. Default is 1.
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
                "spend_tokens": automa.spend_tokens,
                "spend_time": automa.spend_time,
            }
            return automa._agent_trace.build(metadata=metadata)

        async def _run_and_report(context: CognitiveContextT) -> str:
            """Run the agent, measure time, and log summary."""
            start_time = time.time()
            result = await GraphAutoma.arun(
                self, mode, context,
                will_fallback=will_fallback,
                max_consecutive_fallbacks=max_consecutive_fallbacks,
            )
            self.spend_time = time.time() - start_time

            if self._verbose:
                agent_name = self.name or self.__class__.__name__
                separator = "=" * 50
                printer.print(separator, color="cyan")
                printer.print(
                    f"  {agent_name} | Completed\n"
                    f"Tokens: {self.spend_tokens} | "
                    f"Time: {self.spend_time:.2f}s",
                    color="cyan"
                )
                printer.print(separator, color="cyan")

            return result
        
        ########################
        # Pre-initialize status
        ########################
        self.spend_tokens = 0
        self.spend_time = 0.0

        if self._llm is None:
            raise RuntimeError("AmphibiousAutoma must be initialized with an LLM.")

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
        else:
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
            context = self._context_class(**constructor_kwargs)  # Create the context
            for field_name, items in exposure_items.items():  # Add items to Exposure fields
                attr = getattr(context, field_name)
                for item in items:
                    attr.add(item)

        # Set the LLM to the context
        if self._llm is not None:
            context.set_llm(self._llm)
        self._current_context = context
        
    
        ########################
        # Run the amphibious automa
        ########################
        await _run_and_report(context=context)
        if trace_running and self._agent_trace:
            _build_trace(self)
