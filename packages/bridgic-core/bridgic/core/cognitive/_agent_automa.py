import inspect
import time
import traceback

from abc import abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._graph_meta import GraphMeta
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
    """Error handling strategy for think_step execution."""
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently ignore exceptions
    RETRY = "retry"    # Retry up to max_retries times


################################################################################################################
# ThinkStep Descriptor
################################################################################################################

class _BoundStep:
    """
    Per-instance binding of a ThinkStepDescriptor to an agent.

    Created fresh by ThinkStepDescriptor.__get__ on each attribute access,
    so concurrent agent instances never share mutable runtime state.
    """

    def __init__(self, descriptor: "ThinkStepDescriptor", agent: "AgentAutoma"):
        self._descriptor = descriptor
        self._agent = agent
        self._runtime_tools: Optional[List[str]] = None
        self._runtime_skills: Optional[List[str]] = None

    def __await__(self):
        """Make the bound step directly awaitable."""
        tools, self._runtime_tools = self._runtime_tools, None
        skills, self._runtime_skills = self._runtime_skills, None
        return self._descriptor._execute(self._agent, tools=tools, skills=skills).__await__()

    def with_tools(self, tools: List[str]) -> "_BoundStep":
        """Restrict tools for this execution. Chainable with with_skills() and until()."""
        self._runtime_tools = tools
        return self

    def with_skills(self, skills: List[str]) -> "_BoundStep":
        """Restrict skills for this execution. Chainable with with_tools() and until()."""
        self._runtime_skills = skills
        return self

    def until(
        self,
        condition: Optional[Union[Callable[[CognitiveContext], bool], Callable[[CognitiveContext], Awaitable[bool]]]] = None,
        max_attempts: Optional[int] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        """Run until condition is met, LLM signals finish, or max attempts reached."""
        effective_tools = tools if tools is not None else self._runtime_tools
        effective_skills = skills if skills is not None else self._runtime_skills
        attempts = max_attempts if max_attempts is not None else self._descriptor.max_retries
        return self._descriptor._execute_until(
            self._agent, condition, attempts,
            tools=effective_tools, skills=effective_skills,
        )

    @property
    def worker(self) -> CognitiveWorker:
        """Access the underlying worker (delegates to descriptor)."""
        return self._descriptor.worker

    def _execute_once(self, agent: "AgentAutoma", *, tools=None, skills=None):
        """Delegate _execute_once to the descriptor (for testing). Returns (result, finished) tuple."""
        return self._descriptor._execute_once(agent, tools=tools, skills=skills)


class ThinkStepDescriptor:
    """
    Descriptor for declaring cognitive thinking steps.

    When accessed on an agent instance, returns a fresh ``_BoundStep`` that
    captures the agent reference. This ensures concurrent agent instances
    never share mutable runtime state.

    Parameters
    ----------
    worker : CognitiveWorker
        The cognitive worker to execute for this step.
    on_error : ErrorStrategy
        Error handling strategy.
    max_retries : int
        Maximum retry attempts when on_error=ErrorStrategy.RETRY.
    tools : Optional[List[str]]
        If set, only these tools are visible to this step.
    skills : Optional[List[str]]
        If set, only these skills are visible to this step.
    """

    def __init__(
        self,
        worker: CognitiveWorker,
        *,
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 3,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        self.worker = worker
        self.on_error = on_error
        self.max_retries = max_retries
        self.tools_filter = tools
        self.skills_filter = skills
        self._name: Optional[str] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Optional["AgentAutoma"], objtype: Optional[Type] = None) -> Any:
        if obj is None:
            return self
        # Return a fresh _BoundStep per access — no shared mutable state
        return _BoundStep(self, obj)

    def bind(self, agent: "AgentAutoma") -> "_BoundStep":
        """Bind this step to an agent instance for dynamic step creation.

        Used when creating think_step instances inside cognition() as local
        variables (descriptor protocol only works for class attributes).

        Parameters
        ----------
        agent : AgentAutoma
            The agent instance to bind to.

        Returns
        -------
        _BoundStep
            A bound step ready for awaiting or chaining.

        Example
        -------
        >>> async def cognition(self, ctx):
        ...     step = think_step(Worker()).bind(self)
        ...     await step.until(lambda ctx: ctx.done, max_attempts=5)
        """
        return _BoundStep(self, agent)

    def until(
        self,
        condition: Optional[Union[Callable[[CognitiveContext], bool], Callable[[CognitiveContext], Awaitable[bool]]]] = None,
        max_attempts: Optional[int] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        """
        Create a repeating execution that runs until condition is met, LLM signals finish,
        or max attempts is reached.

        Parameters
        ----------
        condition : Optional callable
            Condition function (sync or async) returning True when done.
            If None, the loop continues until LLM sets finish=True or max_attempts is reached.
        max_attempts : Optional[int]
            Maximum attempts. If None, uses self.max_retries.
        tools : Optional[List[str]]
            If set, dynamically restrict tools for this execution only.
        skills : Optional[List[str]]
            If set, dynamically restrict skills for this execution only.

        Returns
        -------
        Awaitable that executes the step repeatedly until stopping condition or max attempts.

        Examples
        --------
        # Rely on LLM finish signal (no explicit condition)
        >>> await self.execute.until(max_attempts=20, skills=["my-skill"])

        # Sync condition
        >>> await self.search.until(lambda ctx: ctx.found_results, max_attempts=5)

        # Async condition
        >>> async def task_done(ctx):
        ...     if ctx.last_step_has_tools:
        ...         return False
        ...     return await self.check_task_completion(ctx)
        >>> await self.execute.until(task_done, max_attempts=10)
        """
        raise RuntimeError(
            f"Cannot call until() directly on think_step '{self._name}': "
            "access via agent instance first: self.step_name.until(...)"
        )

    async def _execute_until(
        self,
        agent: "AgentAutoma",
        condition: Optional[Union[Callable[[CognitiveContext], bool], Callable[[CognitiveContext], Awaitable[bool]]]],
        max_attempts: int,
        *,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> None:
        """Execute the step repeatedly until finish signal, condition is met, or max attempts reached."""
        for _ in range(max_attempts):
            _, finished = await self._execute_once(agent, tools=tools, skills=skills)

            # LLM explicit finish signal — highest priority
            if finished:
                return

            # Check condition (supports both sync and async)
            if condition is not None:
                result = condition(agent._current_context)
                if inspect.iscoroutine(result):
                    result = await result
                if result:
                    return

    async def _execute(self, agent: "AgentAutoma", *, tools: Optional[List[str]] = None, skills: Optional[List[str]] = None) -> Any:
        """Execute the thinking step with error handling and optional tool/skill filtering."""
        context = agent._current_context
        if context is None:
            raise RuntimeError(
                f"Cannot execute think_step '{self._name}': no active context. "
                "This step must be called within a cognition method."
            )

        result, _ = await self._execute_once(agent, tools=tools, skills=skills)
        return result

    async def _execute_once(self, agent: "AgentAutoma", *, tools: Optional[List[str]] = None, skills: Optional[List[str]] = None) -> Tuple[Any, bool]:
        """Execute the step once with all setup and teardown.

        Returns
        -------
        Tuple[Any, bool]
            (result, finished) where:
            - result: typed instance (output_schema mode) or None (standard mode)
            - finished: True if LLM set finish=True on this round
        """
        context = agent._current_context

        # Inject agent's default LLM if worker doesn't have one
        if self.worker._llm is None and agent._llm is not None:
            self.worker.set_llm(agent._llm)

        # Inject agent's verbose if worker doesn't have its own
        injected_verbose = False
        if self.worker._verbose is None:
            self.worker._verbose = agent._verbose
            injected_verbose = True

        # Apply tool/skill filtering: use override when provided, else descriptor's filter
        effective_tools = tools if tools is not None else self.tools_filter
        effective_skills = skills if skills is not None else self.skills_filter

        original_tools = None
        original_skills = None
        filtered_skills = None
        filtered_to_orig: Dict[int, int] = {}

        if effective_tools is not None:
            original_tools = context.tools
            filtered_tools = CognitiveTools()
            for tool in original_tools.get_all():
                if tool.tool_name in effective_tools:
                    filtered_tools.add(tool)
            context.tools = filtered_tools

        if effective_skills is not None:
            original_skills = context.skills
            filtered_skills = CognitiveSkills()
            # Build index mappings and copy revealed state from original to filtered.
            # This preserves skill details disclosed in earlier iterations of until(),
            # so the LLM doesn't need to re-request the same skill details each round.
            orig_to_filtered: Dict[int, int] = {}
            for orig_idx, skill in enumerate(original_skills.get_all()):
                if skill.name in effective_skills:
                    new_idx = len(filtered_skills)
                    filtered_skills.add(skill)
                    orig_to_filtered[orig_idx] = new_idx
                    filtered_to_orig[new_idx] = orig_idx
            # Forward-copy cached reveals with remapped indices
            for orig_idx, detail in original_skills._revealed.items():
                if orig_idx in orig_to_filtered:
                    filtered_skills._revealed[orig_to_filtered[orig_idx]] = detail
            context.skills = filtered_skills

        # Track worker token delta
        tokens_before = self.worker.spend_tokens

        result = None
        finished = False
        try:
            # ① Observe (enhancement semantics)
            # Step 1: Get default observation from agent
            default_obs = await agent.observation(context)

            # Step 2: Let worker enhance it (pass default_observation)
            obs = await self.worker.observation(context, default_observation=default_obs)

            # Step 3: If worker returns _DELEGATE (legacy mode), use default
            if obs is _DELEGATE:
                obs = default_obs

            # Write observation to context for the thinking phase
            context.observation = obs

            # Log observation if verbose
            if self.worker._verbose:
                obs_str = str(obs) if obs is not None else "None"
                self.worker._log(
                    "Observe", "Observation",
                    obs_str[:200] + ("..." if len(obs_str) > 200 else ""),
                    color="cyan"
                )

            # ② Think (pure thinking — worker returns decision directly via is_output=True)
            decision = await self._run_with_error_handling(context)

            # ③ Act (agent-level execution)
            if decision is not None:
                if self.worker.output_schema is not None:
                    # output_schema mode: decision.output is typed instance, skip action
                    result = decision.output
                    finished = getattr(decision, 'finish', False)
                else:
                    # standard mode: decision.output is List[StepToolCall], execute tools
                    await agent.action(decision, context, _worker=self.worker)
                    result = None
                    finished = getattr(decision, 'finish', False)

                    # Record descriptor name in the last step's metadata for workflow replay.
                    # This lets _build_workflow() find the exact worker for each step.
                    if self._name and context.cognitive_history._items:
                        context.cognitive_history._items[-1].metadata["step_key"] = self._name

        finally:
            # Accumulate token delta to agent
            agent.spend_tokens += self.worker.spend_tokens - tokens_before

            # Restore injected verbose
            if injected_verbose:
                self.worker._verbose = None

            # Restore original tools/skills
            if original_tools is not None:
                context.tools = original_tools
            if original_skills is not None:
                # Write back any new reveals from filtered_skills to original_skills
                # (with reverse index mapping) so they persist across until() iterations.
                if filtered_skills is not None:
                    for filtered_idx, detail in filtered_skills._revealed.items():
                        orig_idx = filtered_to_orig.get(filtered_idx)
                        if orig_idx is not None:
                            original_skills._revealed[orig_idx] = detail
                context.skills = original_skills

        return result, finished

    async def _run_with_error_handling(self, context) -> Any:
        """Run the worker with configured error handling strategy, returning the decision."""
        if self.on_error == ErrorStrategy.RAISE:
            return await self.worker.arun(context=context)
        elif self.on_error == ErrorStrategy.IGNORE:
            try:
                return await self.worker.arun(context=context)
            except Exception:
                return None
        elif self.on_error == ErrorStrategy.RETRY:
            for attempt in range(self.max_retries + 1):
                try:
                    return await self.worker.arun(context=context)
                except Exception as e:
                    if attempt == self.max_retries:
                        raise

    @property
    def name(self) -> Optional[str]:
        """Get the step name assigned by the descriptor protocol."""
        return self._name


def think_step(
    worker: CognitiveWorker,
    *,
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 3,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
) -> ThinkStepDescriptor:
    """
    Create a thinking step that wraps a CognitiveWorker.

    When accessed on an AgentAutoma instance, returns an awaitable coroutine.
    Simply use `await self.step_name` in your cognition method.

    Parameters
    ----------
    worker : CognitiveWorker
        The cognitive worker to execute for this step.
    on_error : ErrorStrategy
        Error handling strategy:
        - ErrorStrategy.RAISE: Re-raise exceptions (default)
        - ErrorStrategy.IGNORE: Silently ignore exceptions
        - ErrorStrategy.RETRY: Retry up to max_retries times
    max_retries : int
        Maximum retry attempts when on_error=ErrorStrategy.RETRY (default: 3).
    tools : Optional[List[str]]
        If set, only these tools are visible to this step.
        Other tools in the context will be temporarily hidden during execution.
    skills : Optional[List[str]]
        If set, only these skills are visible to this step.
        Other skills in the context will be temporarily hidden during execution.

    Returns
    -------
    ThinkStepDescriptor
        A descriptor that returns an awaitable when accessed.

    Examples
    --------
    >>> class MyAgent(AgentAutoma[MyContext]):
    ...     # Class-level step definitions
    ...     analyze = think_step(ThinkWorker(llm=llm))
    ...
    ...     # Step with only specific tools
    ...     search = think_step(
    ...         SearchWorker(llm=llm),
    ...         tools=["search_flight", "search_hotel"]
    ...     )
    ...
    ...     # Step with only specific skills
    ...     book = think_step(
    ...         BookingWorker(llm=llm),
    ...         skills=["booking"]
    ...     )
    ...
    ...     async def cognition(self, ctx: MyContext):
    ...         # Direct execution
    ...         await self.analyze
    ...
    ...         # Conditional repeat execution
    ...         await self.search.until(lambda ctx: ctx.found_results, max_attempts=5)
    ...
    ...         # Dynamic tools/skills at runtime (in cognition), chainable
    ...         if ctx.phase == "search":
    ...             await self.step.with_tools(["search_flight", "search_hotel"])
    ...         else:
    ...             await self.step.with_tools(["book_flight"]).with_skills(["booking"]).until(
    ...                 lambda ctx: ctx.done,
    ...                 max_attempts=3
    ...             )
    ...
    ...         # Dynamic step creation (defined in cognition method)
    ...         plan = think_step(PlanWorker(llm=llm), tools=["register_phase"]).bind(self)
    ...         await plan.until(lambda ctx: bool(ctx.phases), max_attempts=3)
    ...
    ...         await self.book
    """
    return ThinkStepDescriptor(
        worker,
        on_error=on_error,
        max_retries=max_retries,
        tools=tools,
        skills=skills,
    )


def _think_step_inline(
    thinking_prompt: str,
    llm: Optional[BaseLlm] = None,
    *,
    enable_rehearsal: bool = False,
    enable_reflection: bool = False,
    output_schema: Optional[Type] = None,
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 3,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
) -> ThinkStepDescriptor:
    """
    Create a think step from a thinking prompt string.

    Convenience function for creating simple workers without defining a class.

    Parameters
    ----------
    thinking_prompt : str
        The thinking prompt to use.
    llm : Optional[BaseLlm]
        LLM instance to use.
    enable_rehearsal : bool
        Enable rehearsal policy.
    enable_reflection : bool
        Enable reflection policy.
    output_schema : Optional[Type[BaseModel]]
        If set, the worker produces a typed instance directly instead of
        going through the standard tool-call loop.
    on_error : ErrorStrategy
        Error handling strategy.
    max_retries : int
        Maximum retry attempts.
    tools : Optional[List[str]]
        Tool filter for this step.
    skills : Optional[List[str]]
        Skill filter for this step.

    Returns
    -------
    ThinkStepDescriptor
        A descriptor that returns an awaitable when accessed.

    Examples
    --------
    >>> class MyAgent(AgentAutoma):
    ...     step = think_step.inline(
    ...         "Plan ONE step",
    ...         llm=llm,
    ...         enable_rehearsal=True
    ...     )
    ...     plan_step = think_step.inline(
    ...         "Produce a phased plan.",
    ...         output_schema=PlanningResult,
    ...     )
    ...
    ...     async def cognition(self, ctx):
    ...         plan = await self.plan_step
    ...         while not ctx.finish:
    ...             await self.step
    """
    worker = CognitiveWorker.inline(
        thinking_prompt=thinking_prompt,
        llm=llm,
        enable_rehearsal=enable_rehearsal,
        enable_reflection=enable_reflection,
        output_schema=output_schema,
    )
    return think_step(
        worker=worker,
        on_error=on_error,
        max_retries=max_retries,
        tools=tools,
        skills=skills,
    )


# Attach as a static method to think_step
think_step.inline = _think_step_inline
think_step.from_prompt = _think_step_inline


################################################################################################################
# AgentAutoma Metaclass
################################################################################################################

class AgentAutomaMeta(GraphMeta):
    """
    Metaclass for AgentAutoma, inheriting from GraphMeta.

    Responsibilities
    ----------------
    1. Collect ThinkStepDescriptor declarations from class definition
    2. Extract Context type from generic parameter
    3. Validate that cognition method is defined
    """

    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any], **kwargs) -> type:
        # Collect ThinkStepDescriptor declarations
        steps: Dict[str, ThinkStepDescriptor] = {}
        for attr_name, attr_value in list(namespace.items()):
            if isinstance(attr_value, ThinkStepDescriptor):
                steps[attr_name] = attr_value

        # Inherit steps from parent classes
        for base in bases:
            if hasattr(base, "_declared_steps"):
                for step_name, step in base._declared_steps.items():
                    if step_name not in steps:
                        steps[step_name] = step

        namespace["_declared_steps"] = steps

        # Create class via parent metaclass
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Extract Context type from generic parameter
        context_class = mcs._extract_context_class(cls, bases)
        cls._context_class = context_class

        return cls

    @staticmethod
    def _extract_context_class(cls: type, bases: Tuple[type, ...]) -> Optional[Type[CognitiveContext]]:
        """
        Extract CognitiveContext type from generic parameter.

        Parameters
        ----------
        cls : type
            The class being created.
        bases : Tuple[type, ...]
            Base classes.

        Returns
        -------
        Optional[Type[CognitiveContext]]
            The CognitiveContext subclass from generic parameter, or None.
        """
        # Check __orig_bases__ for generic parameter
        for base in getattr(cls, "__orig_bases__", []):
            origin = get_origin(base)
            if origin is not None:
                args = get_args(base)
                if args:
                    context_type = args[0]
                    if isinstance(context_type, type) and issubclass(context_type, CognitiveContext):
                        return context_type

        # Inherit from parent class
        for base in bases:
            if hasattr(base, "_context_class") and base._context_class is not None:
                return base._context_class

        return None


################################################################################################################
# AgentAutoma
################################################################################################################

class AgentAutoma(GraphAutoma, Generic[CognitiveContextT], metaclass=AgentAutomaMeta):
    """
    Base class for cognitive agents, inheriting from GraphAutoma.

    Subclasses define agent behavior by:
    1. Declaring thinking steps using think_step()
    2. Implementing the cognition() method with Python async control flow

    The CognitiveContext type is specified via generic parameter, and the framework
    automatically creates and initializes the context.

    Parameters
    ----------
    llm : Optional[BaseLlm]
        Default LLM for auxiliary tasks (e.g., history compression).
        Individual think_step workers can specify their own LLM.
    name : Optional[str]
        Optional name for the agent instance.
    ctx_init : Optional[Dict[str, Any]]
        Context field initialization dict. Keys match context model fields.
    verbose : bool
        Enable logging of execution summary (tokens, time). Default is False.

    Examples
    --------
    >>> class MyAgent(AgentAutoma[MyContext]):
    ...     analyze = think_step(ThinkWorker(llm=llm))
    ...     reflect = think_step(ReflectWorker(llm=llm))
    ...
    ...     async def cognition(self, ctx: MyContext):
    ...         await self.analyze
    ...         while not ctx.finish:
    ...             await self.reflect
    ...
    >>> # Option 1: Pass goal directly, framework creates MyContext
    >>> result = await MyAgent(llm=llm).arun(goal="...")
    ...
    >>> # Option 2: Create context manually
    >>> ctx = MyContext(goal="...")
    >>> result = await MyAgent(llm=llm).arun(context=ctx)
    ...
    >>> # Option 3: Use ctx_init to initialize context fields without subclassing
    >>> result = await MyAgent(
    ...     llm=llm,
    ...     ctx_init={"tools": [tool1, tool2], "skills": [skill1]}
    ... ).arun(goal="...")
    """

    _declared_steps: Dict[str, ThinkStepDescriptor] = {}
    _context_class: Optional[Type[CognitiveContext]] = None

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        llm: Optional[Any] = None,
        ctx_init: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)
        self._llm = llm  # Default LLM for auxiliary tasks
        self._ctx_init = ctx_init
        self._current_context: Optional[CognitiveContextT] = None
        self._verbose = verbose

        # Usage stats (reset per arun call)
        self.spend_tokens: int = 0
        self.spend_time: float = 0.0

    ############################################################################
    # Worker methods (GraphAutoma execution flow)
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _cognition(self, context: CognitiveContextT) -> CognitiveContextT:
        """
        Cognition: runs the user-defined cognition method.

        This is the start worker of AgentAutoma. It sets up the context
        and delegates to the cognition() method for orchestration logic.

        Parameters
        ----------
        context : CognitiveContextT
            The cognitive context.

        Returns
        -------
        CognitiveContextT
            The context after cognition completes.
        """
        self._current_context = context
        self._apply_ctx_init(context)
        await self.cognition(context)
        return self._current_context

    def _apply_ctx_init(self, ctx: CognitiveContextT) -> None:
        """
        Apply ctx_init configuration to the context.

        For each key-value pair in ctx_init:
        - If key is not a model field on the context, skip it.
        - If key is an Exposure field, value must be a list/tuple; items are added via add().
        - Otherwise (regular field), only applied when context was pre-created
          (i.e. ``_ctx_init_exposure_only`` is False). For auto-created contexts,
          regular fields are already merged into the constructor kwargs by ``arun()``.

        Raises
        ------
        TypeError
            If value type doesn't match the field type.
        """
        if not self._ctx_init:
            return

        exposure_only = getattr(self, '_ctx_init_exposure_only', False)
        exposure_fields = type(ctx)._exposure_fields or {}

        for key, value in self._ctx_init.items():
            if key not in type(ctx).model_fields:
                continue

            if key in exposure_fields:
                if isinstance(value, Exposure):
                    # Exposure instance: replace directly (enables shared state)
                    setattr(ctx, key, value)
                elif isinstance(value, (list, tuple)):
                    # List/tuple: add items to the default Exposure
                    attr = getattr(ctx, key)
                    for item in value:
                        attr.add(item)
                else:
                    raise TypeError(
                        f"ctx_init['{key}']: expected a list or Exposure instance "
                        f"for Exposure field, got {type(value).__name__}"
                    )
            elif not exposure_only:
                # Regular field: only for pre-created contexts
                field_annotation = type(ctx).model_fields[key].annotation
                expected_type = field_annotation
                if not isinstance(expected_type, type):
                    expected_type = get_origin(expected_type)

                if isinstance(expected_type, type) and not isinstance(value, expected_type):
                    type_name = getattr(field_annotation, '__name__', str(field_annotation))
                    raise TypeError(
                        f"ctx_init['{key}']: expected {type_name}, "
                        f"got {type(value).__name__}"
                    )
                setattr(ctx, key, value)

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    @abstractmethod
    async def cognition(self, ctx: CognitiveContextT) -> None:
        """
        Define the cognitive orchestration logic.

        Subclasses must implement this method to define how thinking steps
        are orchestrated. Use standard Python control flow (if/else, while,
        for) combined with await on thinking steps.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context, used for accessing state and conditions.

        Examples
        --------
        >>> async def cognition(self, ctx: MyContext):
        ...     await self.analyze
        ...     if ctx.needs_reflection:
        ...         await self.reflect
        ...     while not ctx.finish:
        ...         await self.iterate
        """
        ...

    async def observation(self, ctx: CognitiveContextT) -> Optional[str]:
        """
        Agent-level default observation, shared across all think steps.

        Called by ThinkStepDescriptor when a worker's observation() returns _DELEGATE.
        Override to provide custom context to all workers by default.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context.

        Returns
        -------
        Optional[str]
            Custom observation to include in the thinking context.
            Return None to include no observation.
        """
        return None

    async def action(
        self,
        decision: Any,
        ctx: CognitiveContextT,
        *,
        _worker: CognitiveWorker,
    ) -> None:
        """
        Agent-level action execution. Override to change the execution engine.

        Calls worker.before_action() and worker.after_action() as callbacks,
        allowing per-worker customization within the shared infrastructure.

        Parameters
        ----------
        decision : Any
            The thinking decision (ThinkDecision or _ThinkResultModel instance).
            Must have an 'output' field of type List[StepToolCall].
        ctx : CognitiveContextT
            The cognitive context.
        _worker : CognitiveWorker
            The worker that produced this decision (used for callbacks).

        Raises
        ------
        TypeError
            If decision.output is not a list (output_schema decisions should not reach here).
        """
        # Read tool calls from 'output' field (unified field name across all models)
        calls = getattr(decision, 'output', None)
        if not isinstance(calls, list):
            raise TypeError(
                f"action() received a non-tool-call decision (output type: {type(calls).__name__}). "
                "output_schema decisions should not reach action()."
            )

        # No tool calls: record the thinking step and continue
        if not calls:
            ctx.add_info(Step(
                content=decision.step_content,
                status=True,
                result=None,
                metadata={"tool_calls": []}
            ))
            ctx.last_step_has_tools = False
            return

        tool_calls = self._convert_decision_to_tool_calls(calls, ctx)
        matched = self._match_tool_calls(tool_calls, ctx)

        # ① BEFORE ACTION (worker callback)
        verified = await _worker.before_action(matched, ctx)
        if not verified:
            raise ValueError(f"No matching tools found for: {[tc.name for tc in tool_calls]}")

        # Log tool execution
        for tc, _ in verified:
            _worker._log("Action", f"Executing: {tc.name}({tc.arguments})", color="green")

        # ② EXECUTE TOOLS
        action_result = await self._run_tools(verified)

        # ③ AFTER ACTION (worker callback)
        consequence = await _worker.after_action(action_result.results, ctx)

        # Log result
        status = "success" if action_result.status else "failed"
        _worker._log("Action", f"Result: {status}", consequence, color="green")

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
        ctx.last_step_has_tools = True

    ############################################################################
    # Execution helpers (migrated from CognitiveWorker)
    ############################################################################

    def _convert_decision_to_tool_calls(
        self,
        calls: List,
        ctx: CognitiveContextT,
    ) -> List[ToolCall]:
        """Convert a list of StepToolCall into ToolCall objects with type-coerced arguments."""
        _, tool_specs = ctx.get_field('tools')
        tool_calls = []

        for idx, call in enumerate(calls):
            # Look up param types for this tool
            tool_spec = next((s for s in tool_specs if s.tool_name == call.tool), None)
            param_types: Dict[str, str] = {}
            if tool_spec and tool_spec.tool_parameters:
                for name, info in tool_spec.tool_parameters.get('properties', {}).items():
                    param_types[name] = info.get('type', 'string')

            # Build arguments dict with type coercion
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

    def _match_tool_calls(
        self,
        tool_calls: List[ToolCall],
        ctx: CognitiveContextT,
    ) -> List[Tuple[ToolCall, ToolSpec]]:
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

    async def _run_tools(self, matched_list: List[Tuple[ToolCall, ToolSpec]]) -> ActionResult:
        """Execute a list of (ToolCall, ToolSpec) pairs and return ActionResult."""
        sandbox = ConcurrentAutoma()
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

    ############################################################################
    # Task completion check
    ############################################################################

    async def check_task_completion(self, ctx: CognitiveContextT) -> bool:
        """
        Use LLM to determine if the task has been completed.

        This method is NOT called automatically. Users should explicitly call it
        in their until() conditions or cognition() logic.

        Evaluates based on:
        - Goal: what we want to achieve
        - Execution history: what has been done
        - Current observation: custom observation from the last observation phase
        - Other displayable context fields

        Returns
        -------
        bool
            True if the task is completed, False otherwise.

        Raises
        ------
        RuntimeError
            If no LLM is configured for this agent.

        Examples
        --------
        >>> async def task_done(ctx):
        ...     # Only check when no tools were called
        ...     if ctx.last_step_has_tools:
        ...         return False
        ...     return await self.check_task_completion(ctx)
        >>>
        >>> await self.execute.until(task_done, max_attempts=10)
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
        """
        Build the prompt for task completion check.

        Uses ctx.format_summary() to include all displayable context fields,
        similar to how CognitiveWorker builds thinking prompts.

        Subclasses can override this to customize what context is included.

        Parameters
        ----------
        ctx : CognitiveContextT
            The cognitive context.

        Returns
        -------
        str
            The formatted prompt for LLM evaluation.

        Examples
        --------
        Override to customize the prompt:

        >>> def _build_completion_check_prompt(self, ctx):
        ...     # Include only specific fields
        ...     context_info = ctx.format_summary(include=['goal', 'cognitive_history'])
        ...     return f"{context_info}\\n\\nIs the task completed? YES or NO."
        """
        # Use format_summary, similar to CognitiveWorker._build_prompts
        context_info = ctx.format_summary()

        # observation field is not in summary (display=False), but we need it for completion check
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
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs
    ) -> Union[CognitiveContextT, Tuple[CognitiveContextT, "GraphAutoma"]]:
        """
        Run the agent.

        Parameters
        ----------
        context : Optional[CognitiveContextT]
            Pre-created context object. If provided, uses this context directly.
        capture_workflow : bool
            If True, returns a ``(context, workflow)`` tuple where ``workflow``
            is a ``GraphAutoma`` representing the flat replay of this run.
        **kwargs
            Additional arguments passed to CognitiveContext constructor.

        Returns
        -------
        CognitiveContextT | Tuple[CognitiveContextT, GraphAutoma]
            The context after execution (or a tuple when capture_workflow=True).

        Raises
        ------
        ValueError
            If CognitiveContext type is not specified in generic parameter.
        """
        # Reset stats for this run
        self.spend_tokens = 0
        self.spend_time = 0.0

        if context is not None:
            # Pre-created context: _apply_ctx_init handles all fields
            self._ctx_init_exposure_only = False
            # Set LLM for history compression if available
            if self._llm is not None:
                context.cognitive_history.set_llm(self._llm)
            result = await self._run_and_report(context=context)
            if capture_workflow:
                return result, self._build_workflow()
            return result

        if self._context_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must specify a CognitiveContext type via generic parameter, "
                f"e.g., class {self.__class__.__name__}(AgentAutoma[MyContext])"
            )

        # Merge non-Exposure ctx_init fields into constructor kwargs,
        # so required fields (e.g. goal) can be provided via ctx_init.
        # Exposure fields (tools, skills, etc.) are handled later by _apply_ctx_init.
        # Auto-created: regular fields already in constructor, _apply_ctx_init only handles Exposure.
        self._ctx_init_exposure_only = True
        if self._ctx_init:
            exposure_fields = self._context_class._exposure_fields
            if exposure_fields is None:
                exposure_fields = self._context_class._detect_exposure_fields()
                self._context_class._exposure_fields = exposure_fields
            merged = dict(kwargs)
            for key, value in self._ctx_init.items():
                if key in self._context_class.model_fields and key not in exposure_fields and key not in merged:
                    merged[key] = value
            context = self._context_class(**merged)
        else:
            context = self._context_class(**kwargs)

        # Set LLM for history compression if available
        if self._llm is not None:
            context.cognitive_history.set_llm(self._llm)

        result = await self._run_and_report(context=context)
        if capture_workflow:
            return result, self._build_workflow()
        return result

    def _build_workflow(self) -> "GraphAutoma":
        """
        Build a linear GraphAutoma from the execution history of the last run.

        Each step in ``cognitive_history`` becomes a ``WorkflowStepWorker`` node.
        Tool call data is taken from the ``action_results`` entry in step metadata
        (populated by ``action()`` during the live run).

        If a step recorded a ``step_key`` in its metadata, the corresponding
        worker from ``_declared_steps`` is used to build an ``observation_fn``
        that replicates the original observe-think-act observation chain on replay.

        Returns
        -------
        GraphAutoma
            A ready-to-run workflow that replays all tool calls without LLM.
        """
        from ._workflow import WorkflowStepWorker, WorkflowToolCall

        graph = GraphAutoma()
        steps = list(self._current_context.cognitive_history.get_all())

        for i, step in enumerate(steps):
            raw = step.metadata.get("action_results", [])
            tool_calls = [
                WorkflowToolCall(
                    tool_name=r["tool_name"],
                    tool_arguments=r["tool_arguments"],
                    tool_result=r.get("tool_result"),
                )
                for r in raw
            ]

            # Build observation_fn bound to the exact worker that produced this step.
            obs_fn = None
            step_key = step.metadata.get("step_key")
            if step_key and step_key in self._declared_steps:
                the_worker = self._declared_steps[step_key].worker

                async def _obs(context, _agent=self, _worker=the_worker):
                    default_obs = await _agent.observation(context)
                    await _worker.observation(context, default_observation=default_obs)

                obs_fn = _obs

            step_worker = WorkflowStepWorker(
                tool_calls=tool_calls,
                step_content=step.content,
                observation_fn=obs_fn,
            )
            is_first = (i == 0)
            is_last = (i == len(steps) - 1)
            graph.add_worker(
                key=f"step_{i}",
                worker=step_worker,
                dependencies=[] if is_first else [f"step_{i - 1}"],
                is_start=is_first,
                is_output=is_last,
                args_mapping_rule=ArgsMappingRule.AS_IS,
            )

        return graph

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

    @property
    def ctx(self) -> Optional[CognitiveContextT]:
        """
        Get the current context.

        Returns
        -------
        Optional[CognitiveContextT]
            The current context, or None if not yet set.
        """
        return self._current_context
