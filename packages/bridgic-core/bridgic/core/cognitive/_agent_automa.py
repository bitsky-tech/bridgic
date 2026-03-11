import inspect
import time
import traceback

from abc import abstractmethod
from contextlib import asynccontextmanager
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._automa import RunningOptions
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.utils._console import printer
from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, ToolCall
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec

from pydantic import BaseModel, ConfigDict

from bridgic.core.cognitive._context import CognitiveContext, CognitiveTools, CognitiveSkills, Exposure, Step
from bridgic.core.cognitive._cognitive_worker import (
    CognitiveWorker, _DELEGATE,
    ThinkDecision,
)


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
            "required": ["status", "results"],
            "additionalProperties": False,
        }
    )
    status: bool
    results: List[ActionStepResult]


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
        self._in_exception_scope = False

        # Execution trace for workflow capture
        self._execution_trace = None

        # Usage stats (reset per arun call)
        self.spend_tokens: int = 0
        self.spend_time: float = 0.0

    @property
    def llm(self) -> Optional[Any]:
        """Access the agent's default LLM."""
        return self._llm

    ############################################################################
    # Worker methods (GraphAutoma execution flow)
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _cognition(self, context: CognitiveContextT) -> CognitiveContextT:
        """
        Cognition: runs the user-defined cognition method.

        This is the start worker of AgentAutoma. It sets up the context
        and delegates to the cognition() method for orchestration logic.
        """
        self._current_context = context
        await self.cognition(context)
        return self._current_context

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

    async def action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: CognitiveWorker,
    ) -> None:
        """
        Agent-level action execution. Override to change the execution engine.

        Calls worker.before_action() and worker.after_action() as callbacks.

        Parameters
        ----------
        decision : Any
            The thinking decision with 'output' field (List[StepToolCall]).
        ctx : CognitiveContextT
            The cognitive context.
        _worker : CognitiveWorker
            The worker that produced this decision (used for callbacks).
        """
        # Get the tool calls from the decision, which are the core obj in action()
        calls = getattr(decision, 'output', None)

        # Validate the type of the tool calls
        if not isinstance(calls, list):
            raise TypeError(
                f"action() received a non-tool-call decision (output type: {type(calls).__name__}). "
                "output_schema decisions should not reach action()."
            )

        ########################
        # If no tool calls
        ########################
        if not calls:
            ctx.add_info(Step(
                content=decision.step_content,
                status=True,
                result=None,
                metadata={"tool_calls": []}
            ))
            return

        ########################
        # If there are tool calls
        ########################
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
        
        async def _run_tools(matched_list: List[Tuple[ToolCall, ToolSpec]]) -> ActionResult:
            """Execute a list of (ToolCall, ToolSpec) pairs and return ActionResult."""
            # The sand box is used to execute the tools in parallel
            sandbox = ConcurrentAutoma()

            # Create the tool calls and add them to the sandbox
            tool_calls = []
            for tool_call, tool_spec in matched_list:
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
                return ActionResult(status=True, results=step_results)
            except Exception:
                return ActionResult(
                    status=False,
                    results=[ActionStepResult(
                        tool_id="",
                        tool_name="",
                        tool_arguments={},
                        tool_result=traceback.format_exc()
                    )]
                )
        
        tool_calls = _convert_decision_to_tool_calls(calls, ctx)
        matched = _match_tool_calls(tool_calls, ctx)

        # 1. BEFORE ACTION (worker callback)
        verified = await _worker.before_action(matched, ctx)
        if not verified:
            raise ValueError(f"No matching tools found for: {[tc.name for tc in tool_calls]}")
        for tc, _ in verified:
            _worker._log("Action", f"Executing: {tc.name}({tc.arguments})", color="green")

        # 2. EXECUTE TOOLS
        action_result = await _run_tools(verified)

        # 3. AFTER ACTION (worker callback)
        consequence = await _worker.after_action(action_result.results, ctx)

        ctx.add_info(Step(
            content=decision.step_content,
            status=action_result.status,
            result=consequence,
            metadata={
                "tool_calls": [tc[0].name for tc in verified],
                "action_results": [
                    {
                        "tool_name": r.tool_name,
                        "tool_arguments": r.tool_arguments,
                        "tool_result": r.tool_result,
                    }
                    for r in action_result.results
                ],
            }
        ))

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
    ) -> Any:
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
        context = self._current_context
        if context is None:
            raise RuntimeError(
                "Cannot call self.run(): no active context. "
                "run() must be called within a cognition() method."
            )

        step_name = name or worker.__class__.__name__

        if until is not None or max_attempts > 1:
            # Looping execution
            result = None
            for attempt in range(max_attempts):
                result, finished = await self._run_once(
                    worker, name=step_name, tools=tools, skills=skills,
                    on_error=on_error, max_retries=max_retries,
                )
                if finished:
                    return result
                if until is not None:
                    cond_result = until(context)
                    if inspect.iscoroutine(cond_result):
                        cond_result = await cond_result
                    if cond_result:
                        return result
            return result
        else:
            # Single execution
            result, _ = await self._run_once(
                worker, name=step_name, tools=tools, skills=skills,
                on_error=on_error, max_retries=max_retries,
            )
            return result

    async def _run_once(
        self,
        worker: CognitiveWorker,
        *,
        name: str = "",
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> Tuple[Any, bool]:
        """Execute a single observe-think-act cycle. Returns (result, finished)."""
        context = self._current_context

        # Inject agent's default LLM if worker doesn't have one
        if worker._llm is None and self._llm is not None:
            worker.set_llm(self._llm)

        # Inject verbose
        injected_verbose = False
        if worker._verbose is None:
            worker._verbose = self._verbose
            injected_verbose = True

        # Apply tool/skill filtering
        original_tools = None
        original_skills = None
        filtered_skills = None
        filtered_to_orig: Dict[int, int] = {}

        if tools is not None:
            original_tools = context.tools
            filtered_tools = CognitiveTools()
            for tool in original_tools.get_all():
                if tool.tool_name in tools:
                    filtered_tools.add(tool)
            context.tools = filtered_tools

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

        tokens_before = worker.spend_tokens

        result = None
        finished = False
        try:
            # ① Observe
            default_obs = await self.observation(context)
            obs = await worker.observation(context, default_observation=default_obs)
            if obs is _DELEGATE:
                obs = default_obs
            context.observation = obs

            if worker._verbose:
                obs_str = str(obs) if obs is not None else "None"
                worker._log(
                    "Observe", "Observation",
                    obs_str[:200] + ("..." if len(obs_str) > 200 else ""),
                    color="cyan"
                )

            # ② Think
            decision = await self._run_worker_with_error_handling(
                worker, context, on_error, max_retries
            )

            # ③ Act
            if decision is not None:
                if worker.output_schema is not None:
                    result = decision.output
                    finished = getattr(decision, 'finish', False)
                else:
                    await self.action(decision, context, _worker=worker)
                    result = None
                    finished = getattr(decision, 'finish', False)

                    # Record step name in metadata for workflow building
                    if name and context.cognitive_history._items:
                        context.cognitive_history._items[-1].metadata["step_key"] = name

            # Record trace step
            self._record_trace_step(name, obs, decision, context)

        finally:
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

        return result, finished

    async def _run_worker_with_error_handling(
        self, worker: CognitiveWorker, context, on_error: ErrorStrategy, max_retries: int
    ) -> Any:
        """Run the worker with configured error handling strategy."""
        if on_error == ErrorStrategy.RAISE:
            return await worker.arun(context=context)
        elif on_error == ErrorStrategy.IGNORE:
            try:
                return await worker.arun(context=context)
            except Exception:
                return None
        elif on_error == ErrorStrategy.RETRY:
            for attempt in range(max_retries + 1):
                try:
                    return await worker.arun(context=context)
                except Exception as e:
                    if attempt == max_retries:
                        raise

    def _record_trace_step(self, name: str, obs: Any, decision: Any, context: Any) -> None:
        """Record a TraceStep to the execution trace (if capture is active)."""
        if self._execution_trace is None:
            return

        from bridgic.core.cognitive._trace import TraceStep
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

        self._execution_trace.add_step(
            name=name,
            observation=str(obs) if obs is not None else None,
            step_content=step_content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            is_exception_handler=self._in_exception_scope,
            source="agent",
        )

    ############################################################################
    # Structured operator execution
    ############################################################################

    async def execute_plan(self, operator) -> Any:
        """
        Execute a structured flow operator (Step, Loop, Sequence, Branch).

        Operators express flow control in a way that serializes cleanly to
        workflow blocks.

        Parameters
        ----------
        operator : Step | Loop | Sequence | Branch
            The operator to execute.

        Returns
        -------
        Any
            The result from the last execution.

        Examples
        --------
        >>> from bridgic.core.cognitive import Loop
        >>> main_flow = Loop(
        ...     worker=executor,
        ...     name="process_orders",
        ...     until=lambda ctx: ctx.all_done,
        ...     max_attempts=50,
        ... )
        >>> await self.execute_plan(main_flow)
        """
        from bridgic.core.cognitive._operators import Step as OpStep, Loop, Sequence, Branch

        if isinstance(operator, OpStep):
            return await self.run(
                operator.worker, name=operator.name,
                tools=operator.tools, skills=operator.skills,
            )
        elif isinstance(operator, Loop):
            return await self.run(
                operator.worker, name=operator.name,
                until=operator.until, max_attempts=operator.max_attempts,
                tools=operator.tools, skills=operator.skills,
            )
        elif isinstance(operator, Sequence):
            result = None
            for sub_op in operator.steps:
                result = await self.execute_plan(sub_op)
            return result
        elif isinstance(operator, Branch):
            key = operator.condition(self._current_context)
            if inspect.iscoroutine(key):
                key = await key
            chosen = operator.branches.get(key, operator.default)
            if chosen is not None:
                return await self.execute_plan(chosen)
            return None
        else:
            raise TypeError(f"Unknown operator type: {type(operator).__name__}")

    ############################################################################
    # Exception scope
    ############################################################################

    @asynccontextmanager
    async def exception_scope(self):
        """
        Mark subsequent steps as exception handling (not recorded in workflow).

        Steps executed within this scope have ``is_exception_handler=True``
        in their TraceStep. When generating a Workflow, these steps are stored
        as known exception patterns at that position, not as normal flow steps.

        Examples
        --------
        >>> async def cognition(self, ctx):
        ...     try:
        ...         await self.run(executor, name="navigate")
        ...     except LoginPopupDetected:
        ...         async with self.exception_scope():
        ...             await self.run(login_handler, name="handle_login")
        ...         await self.run(executor, name="navigate")  # retry
        """
        self._in_exception_scope = True
        try:
            yield
        finally:
            self._in_exception_scope = False


    ############################################################################
    # Task completion check
    ############################################################################

    async def check_task_completion(self, ctx: CognitiveContextT) -> bool:
        """
        Use LLM to determine if the task has been completed.

        Not called automatically — use in until() conditions or cognition() logic.
        """
        if self._llm is None:
            raise RuntimeError(
                "check_task_completion requires an LLM. "
                "Set llm= when creating the agent."
            )

        prompt = self._build_completion_check_prompt(ctx)

        system_msg = (
            "You are a task completion evaluator. Determine if the task has been completed "
            "based on the goal, execution history, and current observation.\n\n"
            "Respond with ONLY 'YES' if the task is fully completed, or 'NO' if more work is needed.\n"
            "Do not include any explanation or other text."
        )

        response = await self._llm.agenerate(
            messages=[
                Message.from_text(text=system_msg, role="system"),
                Message.from_text(text=prompt, role="user")
            ]
        )

        return response.strip().upper() == "YES"

    def _build_completion_check_prompt(self, ctx: CognitiveContextT) -> str:
        """Build the prompt for task completion check."""
        context_info = ctx.format_summary()
        if ctx.observation:
            context_info += f"\n\nCurrent Observation:\n{ctx.observation}"
        prompt = f"{context_info}\n\nIs the task completed? Respond with YES or NO."
        return prompt

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
        self.spend_tokens = 0
        self.spend_time = 0.0

        # Initialize execution trace for workflow capture
        if capture_workflow:
            from bridgic.core.cognitive._trace import ExecutionTrace
            self._execution_trace = ExecutionTrace()
        else:
            self._execution_trace = None

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
    
        ########################
        # Run the amphibious agent
        ########################
        if workflow is not None:
            result = await self._run_amphibious(context, workflow)
        else:
            result = await self._run_and_report(context=context)

        # If capture_workflow is True, return the result and the workflow
        if capture_workflow:
            return result, self._build_workflow()
        return result

    async def _run_amphibious(self, context: CognitiveContextT, workflow) -> CognitiveContextT:
        """Run in amphibious mode using AmphibiousRunner."""
        from bridgic.core.cognitive._amphibious import AmphibiousRunner
        runner = AmphibiousRunner(agent=self, workflow=workflow)
        return await runner.run(context)

    def _build_workflow(self) -> Any:
        """
        Build a Workflow from the execution trace of the last run.

        Returns
        -------
        Workflow
            Structured Workflow from the execution trace.
        """
        if self._execution_trace is not None:
            return self._build_structured_workflow()

        # Fallback: no trace active (capture_workflow=False path shouldn't reach here,
        # but guard anyway by returning an empty Workflow).
        from bridgic.core.cognitive._workflow import Workflow
        wf = Workflow(metadata={"agent_class": self.__class__.__name__})
        return wf

    def _build_structured_workflow(self):
        """Build a structured Workflow from the execution trace."""
        from bridgic.core.cognitive._workflow import Workflow

        wf = Workflow(metadata={
            "agent_class": self.__class__.__name__,
            "steps_count": len(self._execution_trace.steps),
        })

        # Group consecutive non-exception steps into linear trace blocks
        current_steps = []
        for trace_step in self._execution_trace.steps:
            if trace_step.is_exception_handler:
                continue
            current_steps.append(trace_step.model_dump())

        if current_steps:
            wf.add_linear_trace_block(name="main", steps=current_steps)

        return wf

    async def _run_and_report(self, context: CognitiveContextT) -> CognitiveContextT:
        """Run the agent, measure time, and log summary."""
        start_time = time.time()
        result = await super().arun(context=context)
        self.spend_time = time.time() - start_time

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
