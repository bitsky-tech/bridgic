import time

from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, get_args, get_origin

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._graph_meta import GraphMeta
from bridgic.core.utils._console import printer

from bridgic.core.cognitive._context import CognitiveContext, CognitiveTools, CognitiveSkills, Exposure
from bridgic.core.cognitive._cognitive_worker import CognitiveWorker


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

class ThinkStepOverride:
    """
    Wrapper that applies dynamic tools/skills override when executing a think step.

    Returned by ThinkStepDescriptor.with_tools() and .with_skills().
    Supports both direct await and .until() chaining.
    """

    def __init__(
        self,
        descriptor: "ThinkStepDescriptor",
        agent: "AgentAutoma",
        *,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        self._descriptor = descriptor
        self._agent = agent
        self._tools = tools
        self._skills = skills

    def __await__(self):
        """Make the override directly awaitable."""
        return self._descriptor._execute(
            self._agent, tools=self._tools, skills=self._skills
        ).__await__()

    def until(
        self,
        condition: Callable[[CognitiveContext], bool],
        max_attempts: Optional[int] = None,
    ):
        """Execute repeatedly until condition is met, with tools/skills override."""
        return self._descriptor._execute_until(
            self._agent,
            condition,
            max_attempts or self._descriptor.max_retries,
            tools=self._tools,
            skills=self._skills,
        )

    def with_tools(self, tools: List[str]) -> "ThinkStepOverride":
        """Chain: override tools (replaces any previous tools override)."""
        return ThinkStepOverride(
            self._descriptor, self._agent, tools=tools, skills=self._skills
        )

    def with_skills(self, skills: List[str]) -> "ThinkStepOverride":
        """Chain: override skills (replaces any previous skills override)."""
        return ThinkStepOverride(
            self._descriptor, self._agent, tools=self._tools, skills=skills
        )


class ThinkStepDescriptor:
    """
    Descriptor for declaring cognitive thinking steps.

    When accessed on an instance, returns an awaitable coroutine that
    executes the wrapped CognitiveWorker with the agent's current context.

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
        # Return self to allow both direct await and .until() chaining
        self._agent = obj
        return self

    def __await__(self):
        """Make the descriptor directly awaitable."""
        if not hasattr(self, '_agent') or self._agent is None:
            raise RuntimeError(
                f"Cannot await think_step '{self._name}': not accessed from agent instance. "
                "Use 'await self.step_name' within cognition method."
            )
        return self._execute(self._agent).__await__()

    def bind(self, agent: "AgentAutoma") -> "ThinkStepDescriptor":
        """Bind this step to an agent instance for dynamic step creation.

        This is needed when creating think_step instances inside cognition()
        method as local variables, since the descriptor protocol only works
        for class attributes.

        Parameters
        ----------
        agent : AgentAutoma
            The agent instance to bind to.

        Returns
        -------
        ThinkStepDescriptor
            Returns self for method chaining.

        Example
        -------
        >>> async def cognition(self, ctx):
        ...     step = think_step(Worker()).bind(self)
        ...     await step.until(lambda ctx: ctx.done, max_attempts=5)
        """
        self._agent = agent
        return self

    def with_tools(self, tools: List[str]) -> ThinkStepOverride:
        """
        Dynamically restrict tools for this execution (in cognition method).

        Returns a wrapper that can be awaited or chained with .until().

        Example
        -------
        >>> async def cognition(self, ctx):
        ...     # Use different tools based on runtime state
        ...     if ctx.phase == "search":
        ...         await self.step.with_tools(["search_flights", "search_hotels"])
        ...     else:
        ...         await self.step.with_tools(["book_flight", "book_hotel"])
        """
        self._ensure_agent()
        return ThinkStepOverride(self, self._agent, tools=tools, skills=None)

    def with_skills(self, skills: List[str]) -> ThinkStepOverride:
        """
        Dynamically restrict skills for this execution (in cognition method).

        Returns a wrapper that can be awaited or chained with .until().

        Example
        -------
        >>> async def cognition(self, ctx):
        ...     await self.step.with_skills(["travel-planning"]).until(
        ...         lambda ctx: ctx.plan_complete, max_attempts=5
        ...     )
        """
        self._ensure_agent()
        return ThinkStepOverride(self, self._agent, tools=None, skills=skills)

    def _ensure_agent(self) -> None:
        if not hasattr(self, '_agent') or self._agent is None:
            raise RuntimeError(
                f"Cannot use with_tools/with_skills on think_step '{self._name}': "
                "not accessed from agent instance. Use within cognition method."
            )

    def until(
        self,
        condition: Callable[[CognitiveContext], bool],
        max_attempts: Optional[int] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        """
        Create a repeating execution that runs until condition is met.

        Parameters
        ----------
        condition : Callable[[CognitiveContext], bool]
            Function that receives context and returns True when done.
        max_attempts : Optional[int]
            Maximum attempts. If None, uses self.max_retries.
        tools : Optional[List[str]]
            If set, dynamically restrict tools for this execution only.
        skills : Optional[List[str]]
            If set, dynamically restrict skills for this execution only.

        Returns
        -------
        Awaitable that executes the step repeatedly until condition or max attempts.

        Example
        -------
        >>> await self.plan_step.until(lambda ctx: bool(ctx.phases), max_attempts=5)
        >>> await self.search.until(
        ...     lambda ctx: ctx.found,
        ...     tools=["search_flights"],
        ...     max_attempts=3
        ... )
        """
        if not hasattr(self, '_agent') or self._agent is None:
            raise RuntimeError(
                f"Cannot use until() on think_step '{self._name}': not accessed from agent instance. "
                "Use within cognition method: await self.step_name.until(...)"
            )

        attempts = max_attempts if max_attempts is not None else self.max_retries
        return self._execute_until(self._agent, condition, attempts, tools=tools, skills=skills)

    async def _execute_until(
        self,
        agent: "AgentAutoma",
        condition: Callable[[CognitiveContext], bool],
        max_attempts: int,
        *,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> None:
        """Execute the step repeatedly until condition is met or max attempts reached."""
        for _ in range(max_attempts):
            await self._execute_once(agent, tools=tools, skills=skills)
            if condition(agent._current_context):
                return

    async def _execute(self, agent: "AgentAutoma", *, tools: Optional[List[str]] = None, skills: Optional[List[str]] = None) -> None:
        """Execute the thinking step with error handling and optional tool/skill filtering."""
        context = agent._current_context
        if context is None:
            raise RuntimeError(
                f"Cannot execute think_step '{self._name}': no active context. "
                "This step must be called within a cognition method."
            )

        await self._execute_once(agent, tools=tools, skills=skills)

    async def _execute_once(self, agent: "AgentAutoma", *, tools: Optional[List[str]] = None, skills: Optional[List[str]] = None) -> None:
        """Execute the step once with all setup and teardown."""
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
            for skill in original_skills.get_all():
                if skill.name in effective_skills:
                    filtered_skills.add(skill)
            context.skills = filtered_skills

        # Track worker token delta
        tokens_before = self.worker.spend_tokens

        try:
            await self._run_with_error_handling(context)
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
                context.skills = original_skills

    async def _run_with_error_handling(self, context) -> None:
        """Run the worker with configured error handling strategy."""
        if self.on_error == ErrorStrategy.RAISE:
            await self.worker.arun(context=context)
        elif self.on_error == ErrorStrategy.IGNORE:
            try:
                await self.worker.arun(context=context)
            except Exception:
                pass
        elif self.on_error == ErrorStrategy.RETRY:
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    await self.worker.arun(context=context)
                    return
                except Exception as e:
                    last_error = e
                    if attempt == self.max_retries:
                        raise last_error

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
    ...         # Dynamic tools/skills at runtime (in cognition)
    ...         if ctx.phase == "search":
    ...             await self.step.with_tools(["search_flight", "search_hotel"])
    ...         else:
    ...             await self.step.until(
    ...                 lambda ctx: ctx.done,
    ...                 tools=["book_flight"],
    ...                 skills=["booking"],
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
        llm: Optional[Any] = None,
        name: Optional[str] = None,
        ctx_init: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name)
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
        - Otherwise, value type must match the field annotation; assigned via setattr.

        Raises
        ------
        TypeError
            If value type doesn't match the field type.
        """
        if not self._ctx_init:
            return

        exposure_fields = type(ctx)._exposure_fields or {}

        for key, value in self._ctx_init.items():
            if key not in type(ctx).model_fields:
                continue

            if key in exposure_fields:
                # Exposure field: value must be a list/tuple
                if not isinstance(value, (list, tuple)):
                    raise TypeError(
                        f"ctx_init['{key}']: expected a list for Exposure field, "
                        f"got {type(value).__name__}"
                    )
                attr = getattr(ctx, key)
                for item in value:
                    attr.add(item)
            else:
                # Regular field: type check
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

    ############################################################################
    # Entry point
    ############################################################################

    async def arun(self, goal: Optional[str] = None, *, context: Optional[CognitiveContextT] = None, **kwargs) -> CognitiveContextT:
        """
        Run the agent.

        Parameters
        ----------
        goal : Optional[str]
            Goal description. If provided, automatically creates the CognitiveContext
            type specified in the generic parameter.
        context : Optional[CognitiveContextT]
            Pre-created context object. If provided, uses this context directly.
        **kwargs
            Additional arguments passed to CognitiveContext constructor.

        Returns
        -------
        CognitiveContextT
            The context object after execution completes.

        Raises
        ------
        ValueError
            If neither goal nor context is provided.
        ValueError
            If CognitiveContext type is not specified in generic parameter.
        """
        # Reset stats for this run
        self.spend_tokens = 0
        self.spend_time = 0.0

        if context is not None:
            # Set LLM for history compression if available
            if self._llm is not None:
                context.cognitive_history.set_llm(self._llm)
            return await self._run_and_report(context=context)

        if goal is None:
            raise ValueError("Must provide either 'goal' or 'context'")

        if self._context_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must specify a CognitiveContext type via generic parameter, "
                f"e.g., class {self.__class__.__name__}(AgentAutoma[MyContext])"
            )

        context = self._context_class(goal=goal, **kwargs)

        # Set LLM for history compression if available
        if self._llm is not None:
            context.cognitive_history.set_llm(self._llm)

        return await self._run_and_report(context=context)

    async def _run_and_report(self, context: CognitiveContextT) -> CognitiveContextT:
        """Run the agent, measure time, and log summary."""
        start_time = time.time()
        result = await super().arun(context=context)
        self.spend_time = time.time() - start_time

        if self._verbose:
            agent_name = self.name or self.__class__.__name__
            steps_count = len(result.cognitive_history) if hasattr(result, 'cognitive_history') else 0
            status = "Completed" if getattr(result, 'finish', False) else "In Progress"
            separator = "=" * 50
            printer.print(separator, color="cyan")
            printer.print(
                f"  {agent_name} | {status}\n"
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
