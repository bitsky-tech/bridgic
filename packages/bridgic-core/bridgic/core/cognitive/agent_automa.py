from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, get_args, get_origin

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa._graph_meta import GraphMeta

from .context import CognitiveContext, CognitiveTools, CognitiveSkills
from .cognitive_worker import CognitiveWorker


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
        return self._execute(obj)

    async def _execute(self, agent: "AgentAutoma") -> None:
        """Execute the thinking step with error handling and optional tool/skill filtering."""
        context = agent._current_context
        if context is None:
            raise RuntimeError(
                f"Cannot execute think_step '{self._name}': no active context. "
                "This step must be called within a cognition method."
            )

        # Apply tool/skill filtering if configured
        original_tools = None
        original_skills = None

        if self.tools_filter is not None:
            original_tools = context.tools
            filtered_tools = CognitiveTools()
            for tool in original_tools.get_all():
                if tool.tool_name in self.tools_filter:
                    filtered_tools.add(tool)
            context.tools = filtered_tools

        if self.skills_filter is not None:
            original_skills = context.skills
            filtered_skills = CognitiveSkills()
            for skill in original_skills.get_all():
                if skill.name in self.skills_filter:
                    filtered_skills.add(skill)
            context.skills = filtered_skills

        try:
            await self._run_with_error_handling(context)
        finally:
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
    ...     # Step with all tools/skills
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
    ...         await self.analyze
    ...         await self.search
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
    """

    _declared_steps: Dict[str, ThinkStepDescriptor] = {}
    _context_class: Optional[Type[CognitiveContext]] = None

    def __init__(self, llm: Optional[Any] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self._current_context: Optional[CognitiveContextT] = None
        self._llm = llm  # Default LLM for auxiliary tasks

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
        await self.cognition(context)
        return self._current_context

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
        if context is not None:
            # Set LLM for history compression if available
            if self._llm is not None:
                context.cognitive_history.set_llm(self._llm)
            return await super().arun(context=context)

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

        return await super().arun(context=context)

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
