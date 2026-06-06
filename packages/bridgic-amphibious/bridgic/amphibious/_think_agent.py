"""ThinkAgent — declarative external-agent think primitive.

Layout (symmetric to ``_think_unit.py``):

* ``ThinkAgentDescriptor`` — class-level marker holding an
  ``AgentWorker`` template + descriptor-level overlays
  (``expose_tools``). The descriptor exposes a ``_clone_worker`` static
  helper for state-isolated cloning per invocation, mirroring
  ``ThinkUnitDescriptor``.
* ``think_agent(worker, *, expose_tools=None)`` — factory that wraps
  one ``AgentWorker`` instance into a descriptor.

The actual orchestration (MCP host, subprocess, prompt assembly) lives
on the ``AgentWorker`` class itself (``_agent_worker.py``); driving the
worker per yield is the job of ``AmphibiousAutoma._run_think_agent``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from bridgic.amphibious._agent_worker import AgentWorker


################################################################################################################
# Descriptor + factory
################################################################################################################


class ThinkAgentDescriptor:
    """Class-level marker for a declared ``think_agent``.

    Invocation goes through ``yield ThinkAgent("name", ...)`` inside
    ``on_agent``; the dispatcher resolves the name, picks up the
    descriptor, clones its ``AgentWorker`` template (state isolation),
    and hands the clone to ``AmphibiousAutoma._run_think_agent``.

    Mirrors ``ThinkUnitDescriptor`` in shape — the two cognitive-
    composition descriptors share the same dispatch contract.
    """

    def __init__(
        self,
        worker: AgentWorker,
        *,
        expose_tools: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(worker, AgentWorker):
            raise TypeError(
                f"think_agent(worker, ...) requires an AgentWorker instance; "
                f"got {type(worker).__name__}. Use AgentWorker(ClaudeCodeAgent(...)) "
                "or subclass AgentWorker."
            )
        self._worker_template: AgentWorker = worker
        self._expose_tools: Optional[List[str]] = (
            list(expose_tools) if expose_tools is not None else None
        )

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "ThinkAgentDescriptor":
        # Both class- and instance-level access return the descriptor
        # itself. Invocation goes through ``yield ThinkAgent("name")``.
        return self

    @staticmethod
    def _clone_worker(template: AgentWorker) -> AgentWorker:
        """Clone an ``AgentWorker`` for state isolation.

        Delegates to ``template._clone()`` — each AgentWorker subclass
        owns the contract of preserving its own config (since
        constructor params vary per subclass, the framework can't
        copy them generically).

        Mirrors ``ThinkUnitDescriptor._clone_worker`` in role.
        """
        return template._clone()


def think_agent(
    worker: AgentWorker,
    *,
    expose_tools: Optional[List[str]] = None,
) -> ThinkAgentDescriptor:
    """Declare a think-agent unit, invoked via ``yield ThinkAgent(name, ...)``.

    Mirrors ``think_unit(worker, ...)`` but wraps an ``AgentWorker``
    instead of a ``CognitiveWorker``. The worker carries all the
    delegate-level config (which CLI backend to spawn, which built-in
    tools to allow, permission mode, completion timeout, …);
    ``expose_tools`` is the descriptor-level filter selecting which
    project tools from ``ctx.tools`` to expose via MCP (``None`` =
    expose every non-builtin tool).

    >>> class ReviewerWorker(AgentWorker):
    ...     async def thinking(self, ota_ctx, big_ctx=None):
    ...         return "Review the file and record findings."
    ...
    >>> class MyAutoma(AmphibiousAutoma[OTAContext, Context]):
    ...     reviewer = think_agent(
    ...         ReviewerWorker(ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"])),
    ...     )
    ...     async def on_agent(self, ota_ctx):
    ...         result = yield ThinkAgent("reviewer")
    ...         yield RETURN(result)
    """
    return ThinkAgentDescriptor(worker, expose_tools=expose_tools)


__all__ = [
    "ThinkAgentDescriptor",
    "think_agent",
]
