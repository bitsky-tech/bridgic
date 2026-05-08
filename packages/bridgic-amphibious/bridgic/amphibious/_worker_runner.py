"""WorkerRunner — minimal runtime interface for plug-in worker implementations.

Background
----------

The framework's default worker is ``CognitiveWorker`` (see
``_cognitive_worker.py``), which decomposes its execution into the
explicit observe-think-act cycle that the framework drives via
``_run`` / ``_run_once``. That decomposition is convenient when you
want the framework to manage iteration, tool selection, and trace
recording — but it is rigid when the goal is to plug in an *external*
agent runtime that already has its own internal loop (Claude Code,
OpenAI Agents, a bespoke ReAct stack, ...).

``WorkerRunner`` is the minimal alternative: a worker that implements
this Protocol gets a single ``run(agent, ctx)`` callback per invocation,
takes full responsibility for completing the sub-task, and mutates
``ctx.cognitive_history`` directly when it wants framework-level
visibility.

Usage
-----

Implement the Protocol on your class::

    class ClaudeCodeWorker:
        async def run(self, agent: AmphibiousAutoma, ctx: CognitiveContext) -> None:
            # subprocess out to claude, feed it ctx.goal, write the
            # resulting transcript into ctx.cognitive_history.
            ...

Then declare it as a think_unit just like any other worker::

    class MyAgent(AmphibiousAutoma[CognitiveContext]):
        external_think = think_unit(ClaudeCodeWorker())

        async def on_agent(self, ctx):
            yield ThinkCall("external_think")

The dispatcher detects that the worker is *not* a ``CognitiveWorker``
and skips the observe-think-act cycle, calling ``run(agent, ctx)``
directly instead. ``until`` / ``max_attempts`` / ``tools`` / ``skills``
overlays are ignored on the WorkerRunner path — the worker manages
its own loop and tool exposure.

Notes
-----

* ``CognitiveWorker`` does **not** itself satisfy ``WorkerRunner`` — the
  two paths are kept distinct so the dispatcher's ``isinstance`` check
  is unambiguous. The framework prefers the CognitiveWorker path for
  any object that is also a CognitiveWorker.
* The Protocol is ``@runtime_checkable``, so ``isinstance(obj, WorkerRunner)``
  is a structural check on attribute presence (the ``run`` method).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bridgic.amphibious._amphibious_automa import AmphibiousAutoma
    from bridgic.amphibious._context import CognitiveContext


@runtime_checkable
class WorkerRunner(Protocol):
    """Minimal runtime interface for an external worker implementation.

    Implement this Protocol on classes that wrap an external agent
    runtime and plug them in via ``think_unit(...)`` — invoked from
    ``on_agent`` by ``yield ThinkCall("name")``.
    """

    async def run(
        self,
        agent: "AmphibiousAutoma",
        ctx: "CognitiveContext",
    ) -> None:
        """Execute the worker's own loop against ``ctx``.

        Parameters
        ----------
        agent : AmphibiousAutoma
            The agent that owns this worker. Provided so the worker can
            access ``agent._llm``, ``agent.spent_tokens``, etc.
        ctx : CognitiveContext
            The current cognitive context. The worker is expected to
            ``ctx.add_info(Step(...))`` for any history it wants the
            framework to see.
        """
        ...


__all__ = ["WorkerRunner"]
