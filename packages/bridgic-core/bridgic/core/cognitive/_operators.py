"""
Structured operators for expressing flow control in cognition methods.

These operators can be used inside ``cognition()`` to express loops,
sequences, branches, and single steps in a way that serializes cleanly
to workflow blocks.

Usage::

    async def cognition(self, ctx):
        planner = CognitiveWorker.inline("Plan", llm=self.llm)
        executor = CognitiveWorker.inline("Execute", llm=self.llm)

        # Free-form step
        await self.run(planner, name="plan")

        # Structured loop (serializable to LoopBlock)
        main_flow = Loop(
            worker=executor,
            name="process_orders",
            until=lambda ctx: ctx.all_done,
            max_attempts=50,
            tools=["click", "extract", "save_info"],
        )
        await self.execute_plan(main_flow)
"""
from typing import Any, Awaitable, Callable, List, Optional, Union

from bridgic.core.cognitive._cognitive_worker import CognitiveWorker


class Step:
    """Single observe-think-act execution.

    Parameters
    ----------
    worker : CognitiveWorker
        The worker to execute.
    name : str
        Step name for workflow mapping.
    tools : Optional[List[str]]
        Tool filter for this step.
    skills : Optional[List[str]]
        Skill filter for this step.
    """

    def __init__(
        self,
        worker: CognitiveWorker,
        *,
        name: str = "",
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        self.worker = worker
        self.name = name
        self.tools = tools
        self.skills = skills


class Loop:
    """Repeated observe-think-act until condition or max_attempts.

    Parameters
    ----------
    worker : CognitiveWorker
        The worker to execute each iteration.
    name : str
        Loop name for workflow mapping.
    until : Optional callable
        Stop condition. Takes context, returns bool.
    max_attempts : int
        Maximum iterations.
    tools : Optional[List[str]]
        Tool filter for this loop.
    skills : Optional[List[str]]
        Skill filter for this loop.
    """

    def __init__(
        self,
        worker: CognitiveWorker,
        *,
        name: str = "",
        until: Optional[Union[
            Callable[..., bool],
            Callable[..., Awaitable[bool]],
        ]] = None,
        max_attempts: int = 10,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        self.worker = worker
        self.name = name
        self.until = until
        self.max_attempts = max_attempts
        self.tools = tools
        self.skills = skills


class Sequence:
    """Ordered list of operators executed in order.

    Parameters
    ----------
    steps : List
        Operators (Step, Loop, Sequence, Branch) to execute.
    name : str
        Sequence name for workflow mapping.
    """

    def __init__(
        self,
        steps: List[Any],
        *,
        name: str = "",
    ):
        self.steps = steps
        self.name = name


class Branch:
    """Conditional execution: pick one branch based on a condition.

    Parameters
    ----------
    condition : callable
        Takes context, returns a key identifying which branch to take.
    branches : dict
        Mapping from condition result to operator.
    name : str
        Branch name for workflow mapping.
    default : Optional
        Default operator if condition result not in branches.
    """

    def __init__(
        self,
        condition: Callable,
        branches: dict,
        *,
        name: str = "",
        default: Any = None,
    ):
        self.condition = condition
        self.branches = branches
        self.name = name
        self.default = default
