"""ThinkUnit — declarative class-level think-step declaration.

Layout (mirrors ``_think_agent.py``): ``ThinkUnitDescriptor`` + factory
``think_unit(...)`` + ``_ThinkUnitRuntime``. The runtime clones the
``CognitiveWorker`` template (state isolation), drives via
``agent._run_think_unit``, and surfaces the worker's structured output
(if any) to the dispatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional, Union

from bridgic.amphibious._cognitive_worker import CognitiveWorker
from bridgic.amphibious._type import ErrorStrategy, ThinkUnit

if TYPE_CHECKING:
    from bridgic.amphibious._amphibious_automa import AmphibiousAutoma
    from bridgic.amphibious._context import CognitiveContext


class ThinkUnitDescriptor:
    """Class-level marker for a declared ``think_unit``.

    Invocation goes through ``yield ThinkUnit("name", ...)`` inside
    ``on_agent``; the dispatcher resolves the name, picks up the
    descriptor, and hands it to ``_ThinkUnitRuntime`` for execution.
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

        Used by ``_ThinkUnitRuntime`` at the start of each execution.
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

    Pass a ``CognitiveWorker`` (cloned per invocation for state
    isolation) or a ``WorkerRunner`` (used directly — manages own loop;
    overlays ignored). Overlays apply to the ``CognitiveWorker`` path:
    ``until`` (loop condition), ``max_attempts`` (default 1),
    ``tools`` / ``skills`` (name filters), ``on_error`` (default RAISE),
    ``max_retries`` (for RETRY strategy).

    >>> class MyAgent(AmphibiousAutoma[MyContext]):
    ...     main_think = think_unit(
    ...         CognitiveWorker.inline("Plan ONE immediate next step"),
    ...         max_attempts=80,
    ...     )
    ...     async def on_agent(self, ctx):
    ...         yield ThinkUnit("main_think")
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


# ----------------------------------------------------------------------
# Runtime — does the actual think-step execution
# ----------------------------------------------------------------------


class _ThinkUnitRuntime:
    """Per-invocation runtime; instantiated by ``_dispatch_step`` and
    used once via ``await runtime.run(agent, ctx)``.

    Mirrors ``_think_agent._ThinkAgentRuntime`` in shape — the two
    cognitive-composition primitives share the same dispatch contract.
    """

    def __init__(
        self,
        descriptor: ThinkUnitDescriptor,
        item: ThinkUnit,
    ) -> None:
        self.descriptor = descriptor
        # Resolve overlays (item-level beats descriptor-level). Worker
        # cloning is deferred to ``run`` so each invocation gets a
        # fresh CognitiveWorker with clean runtime state.
        self.until = (
            item.until if item.until is not None else descriptor._until
        )
        self.max_attempts: int = (
            item.max_attempts if item.max_attempts is not None
            else descriptor._max_attempts
        )
        self.tools: Optional[List[str]] = (
            item.tools if item.tools is not None else descriptor._tools
        )
        self.skills: Optional[List[str]] = (
            item.skills if item.skills is not None else descriptor._skills
        )
        # These two are descriptor-only (no per-yield overlay today).
        self.on_error: ErrorStrategy = descriptor._on_error
        self.max_retries: int = descriptor._max_retries

    async def run(
        self,
        agent: "AmphibiousAutoma",
        ctx: "CognitiveContext",
    ) -> Any:
        """Execute the think unit and return its structured output (or None).

        Phases:

        1. Resolve worker — clone the ``CognitiveWorker`` template for
           state isolation; pass an external ``WorkerRunner`` through
           as-is (the runner manages its own state, and the per-yield
           overlays are CognitiveWorker concepts that the runner
           ignores).
        2. Drive through ``agent._run_think_unit`` with the resolved overlays.
        3. Surface structured output: when the worker is a
           ``CognitiveWorker`` with an ``output_schema``, return the
           ``.result`` of the last step in ``ctx.cognitive_history``;
           otherwise return ``None``. Atomic ``ActionCall`` /
           ``HumanCall`` / ``LLMCall`` yields have their results flow
           directly back via ``.asend()`` and don't go through this
           surfacing layer.
        """
        # 1. Resolve worker.
        template = self.descriptor._worker_template
        if isinstance(template, CognitiveWorker):
            worker = ThinkUnitDescriptor._clone_worker(template)
        else:
            # External WorkerRunner — use the template directly.
            worker = template

        # 2. Drive.
        await agent._run_think_unit(
            worker,
            until=self.until,
            max_attempts=self.max_attempts,
            tools=self.tools,
            skills=self.skills,
            on_error=self.on_error,
            max_retries=self.max_retries,
        )

        # 3. Surface structured output.
        if (
            isinstance(worker, CognitiveWorker)
            and worker.output_schema is not None
            and ctx is not None
            and len(ctx.cognitive_history) > 0
        ):
            last_step = ctx.cognitive_history.get_all()[-1]
            if last_step.result is not None:
                return last_step.result
        return None


__all__ = [
    "ThinkUnitDescriptor",
    "think_unit",
]
