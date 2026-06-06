"""ThinkUnit — declarative class-level think-step declaration.

Layout (symmetric to ``_think_agent.py``):

* ``ThinkUnitDescriptor`` — class-level marker holding a
  ``CognitiveWorker`` template + descriptor-level orchestration knobs
  (``until`` / ``max_attempts`` / ``on_error`` / ``max_retries``).
  Exposes a ``_clone_worker`` static helper for state-isolated cloning
  per invocation.
* ``think_unit(worker, *, ...)`` — factory that wraps one
  ``CognitiveWorker`` instance into a descriptor.

The actual OTC cycle (observe → think → act loop) lives on
``AmphibiousAutoma._run_think_unit``; driving the worker per yield is
the job of the dispatcher there.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Union

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
        on_error: ErrorStrategy = ErrorStrategy.RAISE,
        max_retries: int = 0,
    ) -> None:
        if not isinstance(worker, CognitiveWorker):
            raise TypeError(
                f"think_unit(worker, ...) requires a CognitiveWorker "
                f"instance; got {type(worker).__name__}. Subclass "
                "CognitiveWorker and implement thinking()."
            )
        self._worker_template: CognitiveWorker = worker
        self._until = until
        self._max_attempts = max_attempts
        self._on_error = on_error
        self._max_retries = max_retries

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "ThinkUnitDescriptor":
        # Both class- and instance-level access return the descriptor
        # itself. Invocation goes through ``yield ThinkUnit("name")``.
        return self

    @staticmethod
    def _clone_worker(template: CognitiveWorker) -> CognitiveWorker:
        """Clone a worker from its template for state isolation.

        Delegates to ``template._clone()`` — each CognitiveWorker subclass
        owns the contract of preserving its own config (since constructor
        params vary per subclass, the framework can't copy them
        generically). The default clone carries verbose and leaves the LLM
        as None — the agent sets it at runtime.

        Mirrors ``ThinkAgentDescriptor._clone_worker``.
        """
        return template._clone()


def think_unit(
    worker: CognitiveWorker,
    *,
    until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
    max_attempts: int = 1,
    on_error: ErrorStrategy = ErrorStrategy.RAISE,
    max_retries: int = 0,
) -> ThinkUnitDescriptor:
    """Declare a think unit, invoked via ``yield ThinkUnit(name)``.

    Wraps a ``CognitiveWorker`` (cloned per invocation for state
    isolation). A think unit owns only the thinking-orchestration knobs —
    the toolset comes from the contexts the worker's ``thinking()``
    assembles, not from here:

    * ``until`` — loop condition (stop early when true).
    * ``max_attempts`` — OTC cycle cap (default 1).
    * ``on_error`` — error policy (default RAISE).
    * ``max_retries`` — for the RETRY strategy.

    >>> class MyThink(CognitiveWorker):
    ...     async def thinking(self, ota_context, context=None):
    ...         return await self._llm.aselect_tool(messages=[...], tools=[...])
    >>> class MyAgent(AmphibiousAutoma[OTAContext, Context]):
    ...     main_think = think_unit(MyThink(), max_attempts=80)
    ...     async def on_agent(self, ota_ctx):
    ...         yield ThinkUnit("main_think")
    """
    return ThinkUnitDescriptor(
        worker,
        until=until,
        max_attempts=max_attempts,
        on_error=on_error,
        max_retries=max_retries,
    )


__all__ = [
    "ThinkUnitDescriptor",
    "think_unit",
]
