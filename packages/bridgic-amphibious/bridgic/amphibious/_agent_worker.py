"""
AgentWorker — the external-agent "think" unit of the amphibious framework.

Symmetric counterpart of ``CognitiveWorker``:

* ``CognitiveWorker`` runs **one observe-think-act cycle in-process**,
  anchored on a ``BaseLlm``. It organizes context (builds messages) and
  calls ``llm.astructured_output(...)``.

* ``AgentWorker`` runs **one delegated cycle out-of-process**, anchored
  on a ``BaseAgent``. It organizes context (MCP-ifies the project
  tools, assembles the message) and calls ``agent.run(request)``.

The two are **peers** — equal status, equal abstraction level. Both are
concrete classes you can instantiate directly; both expose the same
template-method surface (``thinking`` / ``observation`` /
``before_action`` / ``after_action``) for subclasses that want to
customize.

The layer split:

* ``AgentWorker`` — context organization only. It builds the in-process
  FastMCP host (the "MCP-ify tools" step), assembles the message, packs
  an ``AgentRequest``, and hands it to ``self._agent.run(...)``. It does
  NOT know any CLI's argv.
* ``BaseAgent`` (``_base_agent.py``) — owns the CLI mechanics: argv,
  subprocess, completion detection, result extraction.

``fastmcp`` / ``uvicorn`` are imported lazily via ``_mcp_host`` — users
who never use ``AgentWorker`` pay zero install / import cost.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.utils._console import printer

from bridgic.amphibious._context import CognitiveContext
from bridgic.amphibious._cognitive_worker import _DELEGATE
from bridgic.amphibious.temp._base_agent import AgentRequest, AgentResult, BaseAgent
from bridgic.amphibious._mcp_host import MCPHost, MCPToolBinding
from bridgic.amphibious._type import ActionCall, ActionResult, Step
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS


################################################################################################################
# AgentWorker
################################################################################################################


class AgentWorker(GraphAutoma):
    """External-agent worker — one delegated cycle.

    Concrete class, peer of ``CognitiveWorker``. Anchored on a
    ``BaseAgent`` (its BASE, exactly as ``CognitiveWorker`` is anchored
    on a ``BaseLlm``): the worker only ever calls ``self._agent.run()``;
    the ``BaseAgent`` owns how the CLI is actually driven.

    Constructor
    -----------
    agent : BaseAgent
        The external coding-agent driver (``ClaudeCodeAgent``, …).
    verbose : Optional[bool]
        Logging override (``None`` = inherit from ``AmphibiousAutoma``).
    verbose_prompt : Optional[bool]
        When truthy, log the assembled message before each delegation
        (parity with ``CognitiveWorker.verbose_prompt``).

    No ``goal`` / ``tools`` / ``skills`` / ``history`` knobs — those are
    all carried by the ``context`` the framework passes in, and read
    straight from it (``thinking()`` reads ``context.goal`` etc.).

    Customize by subclassing and overriding the template methods —
    ``thinking`` (assemble the message), ``observation``,
    ``before_action``, ``after_action``. The default ``thinking``
    already produces a sensible message, so ``AgentWorker(agent)`` works
    out of the box with no subclass.

    >>> reviewer = think_agent(AgentWorker(
    ...     ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"]),
    ... ))
    >>>
    >>> class StrictReviewer(AgentWorker):
    ...     async def thinking(self, context):
    ...         base = await super().thinking(context)
    ...         return base + "\\n\\nBe extremely thorough."
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"AgentWorker(agent, ...) requires a BaseAgent instance; "
                f"got {type(agent).__name__}. Use ClaudeCodeAgent(...) or "
                "subclass BaseAgent."
            )
        # The BASE — mirrors ``CognitiveWorker._llm``.
        self._agent: BaseAgent = agent

        # Logging runtime — parity with ``CognitiveWorker``'s
        # ``verbose`` / ``verbose_prompt`` (``None`` = inherit / off).
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Framework-injected per-call wiring (set by
        # ``AmphibiousAutoma._run_think_agent`` before driving). Each
        # CLI tool call is surfaced onto this channel as a decision; the
        # framework's consumer executes it and returns the result. The
        # worker itself never touches ``_run_action_call`` — same spirit
        # as ``CognitiveWorker`` only ever holding a ``_llm``.
        self._decision_channel: Optional["asyncio.Queue"] = None

        # Wall-clock spent across this worker's delegations. (No
        # ``spent_tokens`` — an external CLI agent's token usage is not
        # visible to the framework; its stdout is not captured.)
        self.spent_time = 0.0

    ############################################################################
    # Core dispatch — NOT user-overridable
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _think(self, context: CognitiveContext) -> Any:
        """Think-phase entry (mirrors ``CognitiveWorker._thinking``).

        Reads ``context.observation`` (set by
        ``AmphibiousAutoma._run_think_agent`` before calling ``arun``).
        Returns the ``AgentResult`` from the delegation — the parent
        dispatcher unwraps ``.output`` for the ``yield ThinkAgent``
        asend value.
        """
        if not isinstance(context, CognitiveContext):
            raise TypeError(
                f"Expected CognitiveContext, got {type(context).__name__}. "
                "AgentWorker requires CognitiveContext or its subclass."
            )
        return await self._run_think(context)

    async def _run_think(self, context: CognitiveContext) -> AgentResult:
        """Organize context, then delegate to ``self._agent.run(...)``.

        Phases:

        1. MCP-ify ``ctx.tools`` — derive bindings, boot the in-process
           FastMCP host (project tools + the ``agent_done`` signal).
        2. Assemble the message via the ``thinking()`` template.
        3. Pack an ``AgentRequest`` (message + cwd + mcp servers +
           allow-list + completion future).
        4. ``await self._agent.run(request)`` — the BaseAgent owns the
           CLI mechanics from here.
        5. Tear the host down; return the ``AgentResult``.

        This method is the AgentWorker's whole job: context organization
        + delegation. It is not user-overridable — override ``thinking``
        / the hooks instead.

        Returns the ``AgentResult`` straight from ``self._agent.run`` —
        no result is stashed on the worker; the outcome flows out
        through the return value, mirroring ``CognitiveWorker.arun``.
        """
        ########################
        # 1. MCP-ify ctx.tools → boot the host
        ########################
        builtin_names = {t.tool_name for t in ALL_BUILTIN_TOOLS}
        bindings = self._build_bindings_from_ctx(context, builtin_names)

        loop = asyncio.get_event_loop()
        agent_done_future: asyncio.Future[str] = loop.create_future()

        async def _on_tool_call(tool_name: str, args: Dict[str, Any]) -> Any:
            return await self._emit_decision(tool_name, args)

        def _on_agent_done(result: str) -> None:
            if not agent_done_future.done():
                agent_done_future.set_result(result)

        host = MCPHost(
            server_name=self._mcp_server_name(),
            bindings=bindings,
            on_tool_call=_on_tool_call,
            on_agent_done=_on_agent_done,
        )
        await host.start()

        ########################
        # 2-5. Assemble request → delegate → teardown
        ########################
        try:
            with tempfile.TemporaryDirectory(prefix="amphi-delegate-") as tmp:
                cwd = Path(tmp)

                message = await self.thinking(context)
                if self._verbose_prompt:
                    printer.print(
                        f"[AgentWorker] prompt → {type(self._agent).__name__}",
                        color="cyan",
                    )
                    printer.print(message, color="gray")

                exposed_tool_names = [
                    f"mcp__{host.server_name}__{b.name}" for b in bindings
                ]
                exposed_tool_names.append(f"mcp__{host.server_name}__agent_done")

                request = AgentRequest(
                    message=message,
                    cwd=cwd,
                    mcp_servers={
                        host.server_name: {"type": "http", "url": host.url},
                    },
                    allowed_tools=exposed_tool_names,
                    done_signal=agent_done_future,
                )
                result = await self._agent.run(request)
        finally:
            await host.stop()

        return result

    def _mcp_server_name(self) -> str:
        """MCP server name advertised to the external agent.

        Subclasses may override. The name prefixes every bridged tool's
        external identifier (``mcp__<server>__<tool>``).
        """
        return "amphi-bridge"

    ############################################################################
    # Internal helpers — MCP bindings / decision emit / cloning
    ############################################################################

    def _build_bindings_from_ctx(
        self, ctx: CognitiveContext, builtin_names: set,
    ) -> List[MCPToolBinding]:
        """Derive MCP tool bindings from ``ctx.tools``.

        Skips framework built-ins (the CLI has its own); any
        ``expose_tools`` filtering already happened upstream where the
        parent dispatcher snapshotted ``ctx.tools`` down to the
        whitelisted set before invoking us.
        """
        bindings: List[MCPToolBinding] = []
        for tool in ctx.tools.get_all():
            name = tool.tool_name
            if name in builtin_names:
                continue
            bindings.append(MCPToolBinding(
                name=name,
                description=tool.tool_description or name,
                parameters=tool.tool_parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ))
        return bindings

    async def _emit_decision(
        self, tool_name: str, args: Dict[str, Any],
    ) -> Any:
        """Surface a CLI tool call as a decision for the framework to execute.

        Each tool call the external agent makes over MCP is, in effect,
        a ``think_result`` — "I want to call this tool." The worker does
        NOT execute it. It builds the decision, hands it to the framework
        over ``self._decision_channel``, and relays back whatever the
        framework's consumer fills in.

        This keeps ``AgentWorker`` symmetric with ``CognitiveWorker``:
        both only *produce* decisions; ``AmphibiousAutoma`` is the one
        that *acts*. The worker never references ``_run_action_call``.

        The framework's consumer runs each decision through
        ``_run_action_call`` (worker ``before_action`` / ``after_action``
        hooks fire, the call lands in ``ctx.cognitive_history`` + the
        ``AgentTrace``), then resolves the future with the resulting
        ``Step`` — which is unwrapped here into the raw tool result the
        external agent expects back.
        """
        if self._decision_channel is None:
            raise RuntimeError(
                "AgentWorker has no decision channel — it must be driven "
                "by AmphibiousAutoma._run_think_agent, which wires the "
                "channel and runs the consumer that executes each decision."
            )
        # ``from_tool_args`` (not the ``**kwargs`` constructor): the
        # external agent may name a tool parameter ``description`` or
        # ``tool_name``, which would collide with ActionCall's own
        # signature if splatted as kwargs.
        action_call = ActionCall.from_tool_args(
            tool_name,
            args,
            description=f"[think_agent] {tool_name}",
        )
        result_future: "asyncio.Future" = (
            asyncio.get_event_loop().create_future()
        )
        await self._decision_channel.put((action_call.decision, result_future))
        step = await result_future
        return _extract_tool_result(step)

    def _clone(self) -> "AgentWorker":
        """Return a fresh worker with the same configuration.

        The ``BaseAgent`` is *shared* across clones — it is stateless
        per call (every dynamic input arrives via ``AgentRequest``),
        exactly as a ``BaseLlm`` is shared across ``CognitiveWorker``
        clones. Subclasses with extra ``__init__`` params should
        override.

        Used by ``ThinkAgentDescriptor._clone_worker`` for state
        isolation at every ``yield ThinkAgent(...)``.
        """
        return type(self)(
            self._agent,
            verbose=self._verbose,
            verbose_prompt=self._verbose_prompt,
        )

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    async def observation(self, context: CognitiveContext) -> Any:
        """Worker-level observation hook. Override to customize.

        Same contract as ``CognitiveWorker.observation``: returning
        ``_DELEGATE`` (or ``None``) hands off to
        ``AmphibiousAutoma.observation()``. Other values are used as
        the observation directly.
        """
        return _DELEGATE

    async def thinking(self, context: CognitiveContext) -> str:
        """Assemble the message handed to the external agent.

        The AgentWorker analog of ``CognitiveWorker.thinking()`` — it
        produces the prompt. The default layout is goal → parent context
        (summary + observation) → completion contract; override to
        restructure or inject domain instructions.

        ``context.goal`` is the resolved goal: the dispatcher snapshots
        ``yield ThinkAgent(goal=...)`` onto it before this method runs
        (or leaves the original ``ctx.goal`` when the yield omits it).
        The project tools are NOT enumerated here — the agent discovers
        them through the MCP server; the contract just needs to tell it
        they exist and how to finish.
        """
        goal = getattr(context, "goal", None) or ""
        context_info = _format_context_info(context, context.observation)

        parts = [
            "You are running as the external agent layer of an AmphibiousAutoma.",
            "",
            "GOAL:",
            goal or "(no goal supplied)",
        ]
        if context_info:
            parts += ["", "PARENT CONTEXT:", context_info]
        parts += [
            "",
            "PROJECT TOOLS:",
            "  Project tools are exposed to you over MCP. Calling them "
            "routes back into the parent automa's hook pipeline.",
            "",
            "COMPLETION CONTRACT:",
            "  When the goal is fully complete, call the `agent_done` "
            "tool with `result` set to the final answer / summary "
            "string, then finish. The parent automa resumes with the "
            "value you pass.",
        ]
        return "\n".join(parts)

    async def before_action(
        self,
        decision_result: Any,
        context: CognitiveContext,
    ) -> Any:
        """Worker-level before_action hook. Override to intercept bridged tool calls.

        Same contract as ``CognitiveWorker.before_action``: ``_DELEGATE`` /
        ``None`` chains to the agent-level hook; any other value
        overrides the decision.
        """
        return _DELEGATE

    async def after_action(
        self,
        step_result: Any,
        ctx: CognitiveContext,
    ) -> Any:
        """Worker-level after_action hook. Override for side-effects.

        Same contract as ``CognitiveWorker.after_action``: ``_DELEGATE`` /
        ``None`` chains to the agent-level hook; any other value
        suppresses it.
        """
        return _DELEGATE

    ############################################################################
    # Entry point
    ############################################################################

    async def arun(
        self,
        *args: Any,
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the think phase.

        Observation must be pre-set in ``context.observation`` (handled
        by ``AmphibiousAutoma._run_think_agent``). Returns the
        ``AgentResult`` for the delegation (mirrors how
        ``CognitiveWorker.arun`` returns its decision).
        """
        start_time = time.time()
        result = await super().arun(*args, feedback_data=feedback_data, **kwargs)
        self.spent_time += time.time() - start_time
        return result


################################################################################################################
# Module-level helpers
################################################################################################################


def _extract_tool_result(step: Optional[Step]) -> Any:
    """Unwrap the raw tool result from a ``Step`` produced by
    ``_run_action_call`` — what the external agent expects back from its
    MCP tool call. ``None`` step → ``None``; a tool-call step → the first
    result's ``tool_result``; anything else → ``step.result`` as-is.
    """
    if step is None:
        return None
    inner = step.result
    if isinstance(inner, ActionResult) and inner.results:
        return inner.results[0].tool_result
    return inner


def _format_context_info(
    ctx: CognitiveContext, observation: Optional[str],
) -> str:
    """A compact context block to inline into the message.

    Mirrors ``CognitiveWorker._build_messages``'s context_info
    construction — picks up the context summary (excluding tools /
    skills, advertised separately) plus the observation if set.
    """
    try:
        summary = ctx.summary()
    except Exception:
        summary = {}
    info_parts = [
        summary[f]
        for f in summary
        if f not in ("tools", "skills") and summary.get(f)
    ]
    info = "\n".join(info_parts)
    if observation is not None:
        obs_str = str(observation)
        if info:
            info += f"\n\nObservation:\n{obs_str}"
        else:
            info = f"Observation:\n{obs_str}"
    return info


__all__ = [
    "AgentWorker",
]
