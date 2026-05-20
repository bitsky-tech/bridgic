"""ThinkUnit — declarative class-level think-step declaration.

Layout (symmetric to ``_think_agent.py``):

* ``ThinkUnitDescriptor`` — class-level marker holding a
  ``CognitiveWorker`` template + descriptor-level overlays (``until``
  / ``max_attempts`` / ``tools`` / ``skills`` / ``on_error`` /
  ``max_retries``). Exposes a ``_clone_worker`` static helper for
  state-isolated cloning per invocation.
* ``think_unit(worker, *, ...)`` — factory that wraps one
  ``CognitiveWorker`` instance into a descriptor.

The actual OTC cycle (observe → think → act loop) lives on
``AmphibiousAutoma._run_think_unit``; driving the worker per yield is
the job of the dispatcher there.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, Union

from bridgic.amphibious._cognitive_worker import CognitiveWorker
from bridgic.amphibious._type import ErrorStrategy


################################################################################################################
# Descriptor + factory
################################################################################################################


class ThinkUnitDescriptor:
    """Class-level marker for a declared ``think_unit``.

    Invocation goes through ``yield ThinkUnit("name", ...)`` inside
    ``on_agent``; the dispatcher resolves the name, picks up the
    descriptor, clones its ``CognitiveWorker`` template (state
    isolation), and hands the clone to
    ``AmphibiousAutoma._run_think_unit``.

    Mirrors ``ThinkAgentDescriptor`` in shape — the two cognitive-
    composition descriptors share the same dispatch contract.
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
    ) -> None:
        if not isinstance(worker, CognitiveWorker):
            raise TypeError(
                f"think_unit(worker, ...) requires a CognitiveWorker "
                f"instance; got {type(worker).__name__}. Use "
                "CognitiveWorker.inline(prompt) or subclass CognitiveWorker."
            )
        self._worker_template: CognitiveWorker = worker
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

        Copies configuration (policies, output_schema, verbose settings)
        but creates a fresh instance with clean runtime state (tokens,
        time, GraphAutoma execution state). LLM is left as None — the
        agent injects it via ``set_llm()`` at runtime.

        Mirrors ``ThinkAgentDescriptor._clone_worker``.
        """
        return type(template)(
            llm=None,
            enable_rehearsal=template.enable_rehearsal,
            enable_reflection=template.enable_reflection,
            verbose=template._verbose,
            verbose_prompt=template._verbose_prompt,
            output_schema=template.output_schema,
        )


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
    """Declare a think unit, invoked via ``yield ThinkUnit(name)``.

    Wraps a ``CognitiveWorker`` (cloned per invocation for state
    isolation). Descriptor-level overlays:

    * ``until`` — loop condition (stop early when true).
    * ``max_attempts`` — OTC cycle cap (default 1).
    * ``tools`` / ``skills`` — name-based filters scoped to this unit.
    * ``on_error`` — error policy (default RAISE).
    * ``max_retries`` — for the RETRY strategy.

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


__all__ = [
    "ThinkUnitDescriptor",
    "think_unit",
]
