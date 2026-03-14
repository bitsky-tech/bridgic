import inspect
import time
import traceback
from abc import abstractmethod
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Annotated, 
    Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, 
    get_args, get_origin
)

from pydantic import BaseModel, ConfigDict

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._automa import RunningOptions
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model.types import ToolCall
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.cognitive._context import CognitiveContext, CognitiveTools, CognitiveSkills, Exposure, LayeredExposure, Step
from bridgic.core.cognitive._cognitive_worker import CognitiveWorker, _DELEGATE, StepToolCall
from bridgic.core.utils._console import printer


################################################################################################################
# Type Aliases
################################################################################################################

CognitiveContextT = TypeVar("CognitiveContextT", bound=CognitiveContext)


class ErrorStrategy(Enum):
    """Error handling strategy for worker execution via ``self.run()``."""
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently ignore exceptions
    RETRY = "retry"    # Retry up to max_retries times


################################################################################################################
# Action Result Data Structures
################################################################################################################

class ActionStepResult(BaseModel):
    """
    Result of executing one tool in the action phase.

    Attributes
    ----------
    tool_id : str
        ID of the tool call.
    tool_name : str
        Name of the tool.
    tool_arguments : Dict[str, Any]
        Arguments passed to the tool.
    tool_result : Any
        Raw result returned by the tool.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool_id", "tool_name", "tool_arguments", "tool_result"],
            "additionalProperties": False,
        }
    )
    tool_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any


class ActionResult(BaseModel):
    """
    Overall result of the action phase (one or more tool executions).

    Attributes
    ----------
    status : bool
        True if all tools succeeded, False otherwise.
    results : List[ActionStepResult]
        Per-tool results in order.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["results"],
            "additionalProperties": False,
        }
    )
    results: List[ActionStepResult]


################################################################################################################
# _AgentSnapshot — async context manager for safe field mutation + revealed state management
################################################################################################################

class _AgentSnapshot:
    """
    Async context manager for exception-safe temporary field overrides on a Context,
    with additional management of LayeredExposure._revealed state.

    Used internally by ``sequential()`` and ``loop()`` context managers in AgentAutoma.

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
# Workflow Builder (internal)
################################################################################################################

class _PhaseAccumulator:
    """Accumulates trace steps for a single structural phase (sequential or loop)."""

    __slots__ = ("phase_type", "name", "steps")

    def __init__(self, phase_type: str, name: str):
        self.phase_type: str = phase_type  # "sequential" | "loop"
        self.name: str = name
        self.steps: List[dict] = []


class _WorkflowBuilder:
    """Stack-based builder that converts phase-annotated trace steps into structured Workflow blocks.

    ``begin_phase()`` / ``end_phase()`` bracket a structural phase.
    ``record_step()`` routes step data to the active phase (or orphan list).
    ``build()`` converts completed phases into the appropriate block types.
    """

    def __init__(self):
        self._stack: List[_PhaseAccumulator] = []
        self._completed: List[_PhaseAccumulator] = []
        self._orphan_steps: List[dict] = []

    def begin_phase(self, phase_type: str, name: str) -> None:
        self._stack.append(_PhaseAccumulator(phase_type, name or phase_type))

    def end_phase(self) -> None:
        if self._stack:
            self._completed.append(self._stack.pop())

    @contextmanager
    def phase(self, phase_type: str, name: str):
        """Context manager that brackets a structural phase."""
        self.begin_phase(phase_type, name)
        try:
            yield
        finally:
            self.end_phase()

    def record_step(self, step_data: dict) -> None:
        if self._stack:
            self._stack[-1].steps.append(step_data)
        else:
            self._orphan_steps.append(step_data)

    def has_content(self) -> bool:
        return bool(self._completed) or bool(self._orphan_steps)

    def has_phases(self) -> bool:
        """True if at least one phase was opened (sequential/loop was used)."""
        return bool(self._completed) or bool(self._stack)

    @property
    def steps_count(self) -> int:
        """Total number of recorded steps across all phases and orphans."""
        total = len(self._orphan_steps)
        for phase in self._completed:
            total += len(phase.steps)
        for phase in self._stack:
            total += len(phase.steps)
        return total

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Convert accumulated phases into a ``Workflow``."""
        from bridgic.core.cognitive._workflow import (
            LinearTraceBlock,
            LoopBlock,
            Workflow,
        )

        wf = Workflow(metadata={"structured": True})

        # Flush any remaining open phases (shouldn't happen, but be safe)
        while self._stack:
            self._completed.append(self._stack.pop())

        # Prepend orphan steps that arrived before the first phase
        if self._orphan_steps:
            wf.blocks.append(LinearTraceBlock(name="pre", steps=list(self._orphan_steps)))

        for phase in self._completed:
            if phase.phase_type == "loop":
                pattern, iterations = self._detect_loop_pattern(phase.steps)
                wf.blocks.append(LoopBlock(
                    name=phase.name,
                    pattern_template=pattern,
                    iterations=iterations,
                    max_attempts=max(len(iterations) * 3, 10),
                ))
            else:
                # sequential (or any other type) → LinearTraceBlock
                wf.blocks.append(LinearTraceBlock(
                    name=phase.name,
                    steps=list(phase.steps),
                ))

        return wf

    # ------------------------------------------------------------------
    # Loop pattern detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_loop_pattern(steps: List[dict]) -> tuple:
        """Detect repeating tool-name subsequences and split steps into iterations.

        Returns ``(pattern_template, iterations)`` where *pattern_template* is
        the canonical tool-name sequence for one iteration and *iterations* is
        a list of step-data lists, one per detected iteration.
        """
        # 1. Extract tool name sequence (flatten multi-tool steps)
        tool_names: List[str] = []
        step_indices: List[int] = []  # maps each tool_name entry → step index
        for idx, step in enumerate(steps):
            tcs = step.get("tool_calls", [])
            for tc in tcs:
                name = tc.get("tool_name") or tc.get("name", "")
                if name:
                    tool_names.append(name)
                    step_indices.append(idx)

        if not tool_names:
            return [], [list(steps)] if steps else []

        # 2. Find best repeating pattern via sliding window + frequency
        #    Prefer longer patterns when counts are equal (more specific).
        best_pattern: List[str] = []
        best_count = 0

        for length in range(2, min(len(tool_names) // 2 + 1, 16)):
            candidate = tool_names[:length]
            count = _count_pattern_occurrences(tool_names, candidate)
            if count >= 2 and count >= best_count:
                best_count = count
                best_pattern = candidate

        if best_count < 2:
            # No repeating pattern found — treat as single iteration
            return tool_names, [list(steps)]

        # 3. Split steps into iterations using the detected pattern
        iterations: List[List[dict]] = []
        current_iter_steps: List[dict] = []
        pattern_pos = 0

        for step in steps:
            current_iter_steps.append(step)

            tcs = step.get("tool_calls", [])
            for tc in tcs:
                tn = tc.get("tool_name") or tc.get("name", "")
                if tn and pattern_pos < len(best_pattern) and tn == best_pattern[pattern_pos]:
                    pattern_pos += 1
                    if pattern_pos >= len(best_pattern):
                        # Completed one iteration
                        iterations.append(current_iter_steps)
                        current_iter_steps = []
                        pattern_pos = 0
                        break

        # Remaining steps go into a partial iteration
        if current_iter_steps:
            iterations.append(current_iter_steps)

        return best_pattern, iterations


def _count_pattern_occurrences(sequence: List[str], pattern: List[str]) -> int:
    """Count non-overlapping occurrences of *pattern* in *sequence*, tolerating noise."""
    if not pattern:
        return 0
    count = 0
    seq_pos = 0
    while seq_pos < len(sequence):
        pat_pos = 0
        scan = seq_pos
        while scan < len(sequence) and pat_pos < len(pattern):
            if sequence[scan] == pattern[pat_pos]:
                pat_pos += 1
            scan += 1
        if pat_pos == len(pattern):
            count += 1
            seq_pos = scan
        else:
            break
    return count


################################################################################################################
# AgentAutoma
################################################################################################################

class AgentAutoma(GraphAutoma, Generic[CognitiveContextT]):
    """
    Base class for cognitive agents — the "amphibious" engine.

    Subclasses define agent behavior by implementing ``cognition()`` using:
    - ``self.run(worker, ...)`` for observe-think-act cycles
    - ``self.execute_plan(operator)`` for structured flow operators
    - Standard Python control flow (if/else, while, for)

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
    >>> class MyAgent(AgentAutoma[CognitiveContext]):
    ...     async def cognition(self, ctx: CognitiveContext):
    ...         planner = CognitiveWorker.inline("Plan approach", llm=self.llm)
    ...         executor = CognitiveWorker.inline("Execute step", llm=self.llm)
    ...         await self.run(planner, name="plan")
    ...         await self.run(executor, name="execute",
    ...                        until=lambda ctx: ctx.done, max_attempts=20)
    ...
    >>> ctx = await MyAgent(llm=llm).arun(goal="Complete the task", tools=[...])
    """

    _context_class: Optional[Type[CognitiveContext]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Initialize the context generic class."""
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
                f"e.g., class {cls.__name__}(AgentAutoma[MyContext])"
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

        # Workflow capture
        self._workflow_builder: Optional[_WorkflowBuilder] = None

        # Usage stats (reset per arun call)
        self.spend_tokens: int = 0
        self.spend_time: float = 0.0

    @property
    def llm(self) -> Optional[Any]:
        """Access the agent's default LLM."""
        return self._llm

    ############################################################################
    # Internal core methods (GraphAutoma execution flow)
    ############################################################################
    @worker(is_start=True, is_output=True)
    async def _cognition(self) -> str:
        """
        Cognition: runs the user-defined cognition method.

        This is the start worker of AgentAutoma. It sets up the context
        and delegates to the cognition() method for orchestration logic.
        """
        await self.cognition(self._current_context)
        return self._current_context.summary()


    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################
    async def observation(self, ctx: CognitiveContextT) -> Optional[str]:
        """
        Agent-level default observation, shared across all workers.

        Called by ``self.run()`` before each thinking phase. Workers can
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
    async def cognition(self, ctx: CognitiveContextT) -> None:
        """
        Define the cognitive orchestration logic.

        Subclasses must implement this method to define how thinking steps
        are orchestrated. Use standard Python control flow combined with
        ``self.run()`` and ``self.execute_plan()``.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context, used for accessing state and conditions.

        Examples
        --------
        >>> async def cognition(self, ctx: MyContext):
        ...     planner = CognitiveWorker.inline("Plan", llm=self.llm)
        ...     executor = CognitiveWorker.inline("Execute", llm=self.llm)
        ...     await self.run(planner, name="plan")
        ...     await self.run(executor, name="execute",
        ...                    until=lambda ctx: ctx.done, max_attempts=20)
        """
        ...

    async def action_tool_call(self, tool_list: List[Tuple[ToolCall, ToolSpec]], context: CognitiveContextT) -> Any:
        """
        Define the action logic.
        """
        # The sand box is used to execute the tools in parallel
        sandbox = ConcurrentAutoma()

        # Create the tool calls and add them to the sandbox
        tool_calls = []
        for tool_call, tool_spec in tool_list:
            tool_calls.append(tool_call)
            tool_worker = tool_spec.create_worker()
            worker_key = f"tool_{tool_call.name}_{tool_call.id}"
            sandbox.add_worker(
                key=worker_key,
                worker=tool_worker,
                args_mapping_rule=ArgsMappingRule.UNPACK
            )
        tool_args = [tc.arguments for tc in tool_calls]

        # Execute the tools in parallel
        try:
            results = await sandbox.arun(InOrder(tool_args))
            step_results = [
                ActionStepResult(
                    tool_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=tc.arguments,
                    tool_result=result
                )
                for tc, result in zip(tool_calls, results)
            ]
            return ActionResult(results=step_results)
        except Exception:
            return ActionResult(results=[])

    async def action_custom_output(self, decision_result: Any, context: CognitiveContextT) -> Any:
        """
        Define the action logic for custom output.
        """
        return decision_result

    ############################################################################
    # Core: self.run() — the primary execution method
    ############################################################################
    async def run(
        self,
        worker: CognitiveWorker,
        *,
        name: Optional[str] = None,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: int = 1,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> None:
        """
        Execute a CognitiveWorker through observe-think-act cycle(s).

        This is the primary method for orchestrating workers in ``cognition()``.
        It handles LLM injection, tool/skill filtering, observation, thinking,
        action, trace recording, and looping.

        Parameters
        ----------
        worker : CognitiveWorker
            The worker to execute.
        name : Optional[str]
            Step name, used for trace/workflow mapping.
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

        Returns
        -------
        Any
            The result from the last execution (typed instance in output_schema mode,
            or None in standard tool-call mode).

        Examples
        --------
        >>> # Single step
        >>> await self.run(planner, name="plan")
        >>>
        >>> # Looping step
        >>> await self.run(executor, name="execute",
        ...                until=lambda ctx: ctx.done, max_attempts=20,
        ...                tools=["click", "type"])
        """
        async def _execute():
            if until is not None or max_attempts > 1:
                for _ in range(max_attempts):
                    finished = await self._run_once(
                        worker, name=step_name, tools=tools, skills=skills,
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
                    worker, name=step_name, tools=tools, skills=skills,
                    on_error=on_error, max_retries=max_retries,
                )

        step_name = name or worker.__class__.__name__
        context = self._current_context
        if context is None:
            raise RuntimeError(
                "Cannot call self.run(): no active context. "
                "run() must be called within a cognition() method."
            )

        await _execute()

    @asynccontextmanager
    async def sequential(self, name: str, *, goal: Optional[str] = None,
                         keep_revealed: Optional[Dict[str, List[int]]] = None,
                         **snapshot_fields):
        """
        Mark a structural phase as sequential for workflow capture.

        Use as an async context manager around ``self.run()`` calls to group them
        into a ``LinearTraceBlock`` when ``capture_workflow=True``.

        Parameters
        ----------
        name : str
            Phase name, used in the workflow block.
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("sequential", name,
                                       keep_revealed=keep_revealed,
                                       _phase_goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def loop(self, name: str, *, goal: Optional[str] = None,
                   keep_revealed: Optional[Dict[str, List[int]]] = None,
                   **snapshot_fields):
        """
        Mark a structural phase as a loop for workflow capture.

        Use as an async context manager around ``self.run()`` calls to group them
        into a ``LoopBlock`` with pattern detection when ``capture_workflow=True``.

        Parameters
        ----------
        name : str
            Phase name, used in the workflow block.
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("loop", name,
                                       keep_revealed=keep_revealed,
                                       _phase_goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def snapshot(self, name: str, *, goal: Optional[str] = None,
                   keep_revealed: Optional[Dict[str, List[int]]] = None,
                   **snapshot_fields):
        """
        Take a snapshot of the context temporarily for the duration of the agent manager.

        Parameters
        ----------
        name : str
            Phase name, used in the workflow block.
        goal : Optional[str]
            Phase-level goal injected into the context so the LLM knows
            the purpose of this phase.
        keep_revealed : Optional[Dict[str, List[int]]]
            Passed to ``_AgentSnapshot`` for revealed state management.
        **snapshot_fields
            Additional context fields to temporarily override during this phase.
        """
        async with self._phase_context("loop", name,
                                       keep_revealed=keep_revealed,
                                       _phase_goal=goal, **snapshot_fields):
            yield

    @asynccontextmanager
    async def _phase_context(self, phase_type: str, name: str, *,
                             keep_revealed: Optional[Dict[str, List[int]]] = None,
                             **snapshot_fields):
        """Shared implementation for sequential() and loop() context managers.

        Parameters
        ----------
        phase_type : str
            Phase type identifier (``"sequential"`` or ``"loop"``).
        name : str
            Phase name, used in the workflow block.
        keep_revealed : Optional[Dict[str, List[int]]]
            Revealed state management mode for ``_AgentSnapshot``.
        **snapshot_fields
            Context fields to temporarily override during this phase.
        """
        context = self._current_context
        fields = {k: v for k, v in snapshot_fields.items() if v is not None}

        if not fields and keep_revealed is None:
            raise ValueError(
                f"_phase_context('{name}'): no snapshot fields or keep_revealed provided. "
                f"If no context state needs to be scoped for this phase, "
                f"use self.run() directly — the behavior is identical."
            )

        snap = _AgentSnapshot(context, fields, keep_revealed=keep_revealed)
        await snap.__aenter__()
        if self._workflow_builder:
            self._workflow_builder.begin_phase(phase_type, name)
        try:
            yield
        finally:
            if self._workflow_builder:
                self._workflow_builder.end_phase()
            await snap.__aexit__(None, None, None)

    async def _run_once(
        self,
        worker: CognitiveWorker,
        *,
        name: str = "",
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> bool:
        """Execute a single observe-think-act cycle. Returns whether the worker signalled finish."""
        async def _run_observe_think_act(worker: CognitiveWorker, context: CognitiveContextT) -> bool:
            # 1. Observe
            obs = await worker.observation(context)
            if obs is _DELEGATE:
                obs = await self.observation(context)
            context.observation = obs

            # TODO: log design
            # if worker._verbose:
            #     obs_str = str(obs) if obs is not None else "None"
            #     worker._log(
            #         "Observe", "Observation",
            #         obs_str[:200] + ("..." if len(obs_str) > 200 else ""),
            #         color="cyan"
            #     )

            # 2. Think
            decision = await worker.arun(context=context)

            # 3. Act
            if decision is not None:
                await self._action(decision, context, _worker=worker)

            # Record trace step
            self._record_trace_step(name, obs, decision, context)
            return decision.finish


        ########################
        # Initialize CognitiveWorker 
        # runtime environment
        ########################
        context = self._current_context

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
                    f"Worker '{name or worker.__class__.__name__}' failed during "
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
                                f"Worker '{name or worker.__class__.__name__}' failed after "
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


    def _record_trace_step(self, name: str, obs: Any, decision: Any, context: Any) -> None:
        """Record a trace step to the workflow builder (if capture is active)."""
        # TODO: Here, you have to judge for yourself whether it is a step that needs to be recorded.
        if self._workflow_builder is None:
            return

        from bridgic.core.cognitive._workflow import WorkflowToolCall

        tool_calls = []
        tool_results = []
        step_content = ""

        if decision is not None:
            step_content = getattr(decision, 'step_content', '')
            calls = getattr(decision, 'output', None)
            if isinstance(calls, list):
                if context.cognitive_history._items:
                    last_step = context.cognitive_history._items[-1]
                    action_results = last_step.metadata.get("action_results", [])
                    for ar in action_results:
                        tool_calls.append(WorkflowToolCall(
                            tool_name=ar["tool_name"],
                            tool_arguments=ar["tool_arguments"],
                            tool_result=ar.get("tool_result"),
                        ))
                        tool_results.append(ar.get("tool_result"))

        self._workflow_builder.record_step({
            "name": name,
            "observation": str(obs) if obs is not None else None,
            "step_content": step_content,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "tool_results": tool_results,
        })

    async def _action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: CognitiveWorker,
    ) -> None:
        """
        Agent-level action execution. Override to change the execution engine.

        Calls worker.before_action() as callback.

        Parameters
        ----------
        decision : Any
            The thinking decision with 'output' field (List[StepToolCall]).
        ctx : CognitiveContextT
            The cognitive context.
        _worker : CognitiveWorker
            The worker that produced this decision (used for callbacks).
        """ 
        def _is_list_step_tool_call(d: Any) -> bool:
            # Get the declared type of the output field
            if not isinstance(d, BaseModel):
                return None
            fi = type(d).model_fields.get('output')
            if fi is None:
                return None
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
            if not calls:
                ctx.add_info(Step(
                    content=decision.step_content,
                    result=None,
                    metadata={"tool_calls": []}
                ))
                return
            
            tool_calls = _convert_decision_to_tool_calls(calls, ctx)
            decision_result = _match_tool_calls(tool_calls, ctx)
        
        # If the output is a BaseModel
        else:
            decision_result = output


        ########################
        # Execution of the action based on the decision result
        ########################
        decision_result = await _worker.before_action(decision_result, ctx)
        if _is_list_step_tool_call(decision):
            action_result = await self.action_tool_call(decision_result, ctx)
            ctx.add_info(Step(
                content=decision.step_content,
                result=action_result.results,
                metadata={
                    "tool_calls": [tc[0].name for tc in decision_result],
                    "tool_arguments": [tc[0].arguments for tc in decision_result],
                    "action_results": [
                        {
                            "tool_name": r.tool_name,
                            "tool_arguments": r.tool_arguments,
                            "tool_result": r.tool_result,
                        }
                        for r in action_result.results
                    ]
                }
            ))
        else:
            action_result = await self.action_custom_output(decision_result, ctx)
            ctx.add_info(Step(content=decision.step_content, result=action_result, metadata={}))


    ############################################################################
    # Entry point
    ############################################################################

    async def arun(
        self,
        *,
        context: Optional[CognitiveContextT] = None,
        capture_workflow: bool = False,
        workflow: Optional[Any] = None,
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs
    ) -> Union[CognitiveContextT, Tuple[CognitiveContextT, Any]]:
        """
        Run the agent.

        Context initialization has two paths:
        1. Pre-created: ``arun(context=my_ctx)``
        2. Auto-created: ``arun(goal="...", tools=[...], skills=[...])``

        For auto-created contexts, all ``**kwargs`` are passed to the
        CognitiveContext constructor. Exposure fields (tools, skills) can be
        passed as lists — items are added via ``add()`` after construction.

        Parameters
        ----------
        context : Optional[CognitiveContextT]
            Pre-created context object. If provided, uses this context directly.
        capture_workflow : bool
            If True, returns a ``(context, Workflow)`` tuple containing the
            structured execution trace of the run.
        workflow : Optional[Workflow]
            If provided, runs in amphibious mode: replay the workflow
            deterministically, falling back to agent mode on divergence.
        feedback_data : Optional[InteractionFeedback | List[InteractionFeedback]]
            Reserved for future use (currently unused).
        **kwargs
            Arguments passed to CognitiveContext constructor when ``context``
            is not provided. Exposure fields (e.g., tools, skills) can be
            passed as lists and will be added via ``add()``.

        Returns
        -------
        CognitiveContextT | Tuple[CognitiveContextT, Workflow]
            The context after execution (or a tuple when capture_workflow=True).

        Examples
        --------
        >>> # Auto-create context with tools and skills
        >>> ctx = await agent.arun(
        ...     goal="Complete the task",
        ...     tools=[tool1, tool2],
        ...     skills=[skill_path1],
        ... )
        >>>
        >>> # Pre-created context
        >>> ctx = MyContext(goal="...", browser=browser)
        >>> ctx = await agent.arun(context=ctx)
        >>>
        >>> # Capture workflow for later replay
        >>> ctx, wf = await agent.arun(goal="...", capture_workflow=True)
        """
        async def _run_and_report(context: CognitiveContextT) -> CognitiveContextT:
            """Run the agent, measure time, and log summary."""
            start_time = time.time()
            await super().arun(context=context)
            self.spend_time = time.time() - start_time

            result = self._current_context

            if self._verbose:
                agent_name = self.name or self.__class__.__name__
                steps_count = len(result.cognitive_history) if hasattr(result, 'cognitive_history') else 0
                separator = "=" * 50
                printer.print(separator, color="cyan")
                printer.print(
                    f"  {agent_name} | Completed\n"
                    f"  Steps: {steps_count} | "
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
            raise RuntimeError(
                "AgentAutoma must be initialized with an LLM."
            )

        if capture_workflow:
            self._workflow_builder = _WorkflowBuilder()
        else:
            self._workflow_builder = None

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
        # Run the amphibious agent
        ########################
        if workflow is not None:
            return await self._run_amphibious(context, workflow)
        else:
            result = await _run_and_report(context=context)
            if capture_workflow:  # If capture_workflow is True, return the result and the workflow
                return result, self._build_workflow()
            return result

    async def _run_amphibious(self, context: CognitiveContextT, workflow) -> CognitiveContextT:
        """Run in amphibious mode using AmphibiousRunner."""
        from bridgic.core.cognitive._amphibious import AmphibiousRunner
        runner = AmphibiousRunner(agent=self, workflow=workflow)
        return await runner.run(context)

    def _build_workflow(self) -> Any:
        """
        Build a Workflow from the workflow builder of the last run.

        Returns
        -------
        Workflow
            Structured Workflow from the captured steps.
        """
        from bridgic.core.cognitive._workflow import Workflow

        if self._workflow_builder is None or not self._workflow_builder.has_content():
            return Workflow(metadata={"agent_class": self.__class__.__name__})

        wf = self._workflow_builder.build()
        wf.metadata.update({
            "agent_class": self.__class__.__name__,
            "steps_count": self._workflow_builder.steps_count,
        })
        return wf
