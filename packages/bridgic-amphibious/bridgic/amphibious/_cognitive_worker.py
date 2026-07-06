"""
CognitiveWorker — the in-process "think" unit of the amphibious framework.

Each ``arun`` call performs exactly the thinking phase: observation is
injected by ``AmphibiousAutoma._run_think_unit`` before calling, and
action is executed by ``AmphibiousAutoma._run_action_call`` after.

The thinking phase is the single template method
:meth:`CognitiveWorker.thinking` — you override it to assemble the prompt
from the two contexts and call the model however you like (chat, streaming,
tool-select, or structured output). Whatever it returns,
:meth:`CognitiveWorker._assemble_decision` adapts into a decision the
framework's orchestration consumes.

For *external*-agent delegation (out-of-process CLI), see the symmetric
``AgentWorker`` in ``_agent_worker.py``.
"""

import json
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from pydantic import BaseModel

from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Response
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.amphibious._context import Context, OTAContext
from bridgic.amphibious._type import (
    StepToolCall,
    ToolArgument,
    ThinkResult
)


#############################################################################
# Sentinel
#############################################################################

_DELEGATE = object()  # Worker returns this to delegate observation to Agent


#############################################################################
# CognitiveWorker
#############################################################################

class CognitiveWorker(GraphAutoma):
    """Cognitive worker — one observe-think-act cycle.

    Observation and action execution are handled by ``AmphibiousAutoma`` as
    shared infrastructure. The worker owns one thing: the **thinking** step.

    :meth:`thinking` is abstract — subclass and implement it to assemble the
    prompt from the two contexts and call ``self._llm`` (chat / tool-select /
    structured output); whatever you return, :meth:`_assemble_decision`
    adapts into the framework's decision. Optional hooks ``observation`` /
    ``before_action`` / ``after_action`` refine the cycle further.

    >>> class MyThink(CognitiveWorker):
    ...     async def thinking(self, ota_context, context=None):
    ...         return await self._llm.aselect_tool(
    ...             messages=build_messages(ota_context),
    ...             tools=[t.to_tool() for t in ota_context.tools],
    ...         )
    """

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        verbose: Optional[bool] = None,
    ):
        super().__init__()

        # LLM
        self._llm = llm

        # Log
        self._verbose = verbose

        # Usage stats
        self.spent_tokens = 0
        self.spent_time = 0

    ############################################################################
    # Core methods
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _thinking(self, ota_context: Optional[OTAContext] = None, context: Optional[Context] = None) -> Any:
        """Framework entry for the thinking phase — orchestrates :meth:`thinking`.

        Both contexts are injected by the dispatcher
        (``arun(ota_context=…, context=…)``): ``ota_context`` is the
        small-loop OTA state, ``context`` the free-form knowledge (``None``
        for a pure-reasoning run). Validates the LLM, calls the overridable
        :meth:`thinking` method to interact with the LLM, then *adapts* its
        result — ``(content, tool_calls)``, a structured ``BaseModel``, or
        text — into the framework's decision shape (see
        :meth:`_assemble_decision`). That decision becomes the ``arun()``
        return value — the driver captures it and runs the act phase.

        The seam: ``thinking`` owns *talking to the model*, the framework
        owns *turning the reply into a decision*.
        """
        if self._llm is None:
            raise RuntimeError(
                "CognitiveWorker has no LLM set — pass llm= when constructing "
                "the worker."
            )
        if ota_context is None:
            ota_context = OTAContext()

        result = await self.thinking(ota_context, context)
        return self._assemble_decision(result)

    def _assemble_decision(self, result: Any) -> ThinkResult:
        """Adapt whatever :meth:`thinking` returned into the act phase's decision.

        The single seam between *any* bridgic LLM protocol and
        ``AmphibiousAutoma``'s dispatch: ``thinking`` calls the model however
        it likes and hands back that call's **natural** result; this maps each
        protocol's real return shape (see ``bridgic.llms.*``) onto a decision:

        Every shape collapses to a flat ``ThinkResult`` (``step_content`` +
        ``tool_calls``); a result with NO ``tool_calls`` IS the finish:

        - ``Response``                 [``achat``]
          -> content-only (``step_content`` = the reply text).
        - ``(tool_calls, content)``    [``aselect_tool``]
          -> tool-calling (``step_content`` = content). NOTE the order —
          ``aselect_tool`` returns ``tool_calls`` first.
        - a pydantic ``BaseModel`` / ``dict``  [``astructured_output``]
          -> content-only; the structured value is serialized into
          ``step_content`` (JSON).
        - a ``str``                    [plain text / an accumulated stream]
          -> content-only.

        Tool-call items may be ``ToolCall`` objects (``.name`` / ``.arguments``)
        or ``{"name": ..., "arguments": {...}}`` dicts.
        """
        # chat — a Response (text in .message.content). Checked before
        # BaseModel because Response *is* a BaseModel.
        if isinstance(result, Response):
            return ThinkResult(step_content=result.message.content or "", tool_calls=[])

        # structured output — a pydantic model or a json-schema dict; the
        # typed value is serialized into ``step_content`` (which is text).
        if isinstance(result, BaseModel):
            return ThinkResult(step_content=result.model_dump_json(), tool_calls=[])
        if isinstance(result, dict):
            return ThinkResult(
                step_content=json.dumps(result, ensure_ascii=False, default=str),
                tool_calls=[],
            )

        # plain text / an accumulated stream.
        if isinstance(result, str):
            return ThinkResult(step_content=result, tool_calls=[])

        # tool-select — (tool_calls, content), aselect_tool's native order.
        if not isinstance(result, (tuple, list)):
            raise TypeError(
                f"thinking() returned an unsupported type {type(result).__name__}; "
                "return a Response, (tool_calls, content), a pydantic BaseModel, "
                "a dict, or str."
            )
        tool_calls, content = result
        tool_calls = tool_calls or []
        tool_calls = [
            StepToolCall(
                call_id=self._tool_call_id(call),
                tool=self._tool_call_name(call),
                tool_arguments=[
                    ToolArgument(name=str(name), value=value)
                    for name, value in self._tool_call_args(call).items()
                ],
            )
            for call in tool_calls
        ]
        return ThinkResult(step_content=content or "", tool_calls=tool_calls)

    ############################################################################
    # Internal helpers
    ############################################################################

    @staticmethod
    def _tool_call_id(call: Any) -> Optional[str]:
        """Read a tool call id from common provider/adapter shapes."""
        if isinstance(call, dict):
            for key in ("id", "call_id", "tool_call_id"):
                value = call.get(key)
                if value:
                    return str(value)
            return None

        for attr in ("id", "call_id", "tool_call_id"):
            value = getattr(call, attr, None)
            if value:
                return str(value)
        return None

    @staticmethod
    def _tool_call_name(call: Any) -> str:
        """Read a tool call's name — accepts an object (``.name``) or a dict."""
        return call["name"] if isinstance(call, dict) else call.name

    @staticmethod
    def _tool_call_args(call: Any) -> Dict[str, Any]:
        """Read a tool call's arguments — accepts an object (``.arguments``) or a dict."""
        args = call.get("arguments") if isinstance(call, dict) else call.arguments
        return args or {}

    def _clone(self) -> "CognitiveWorker":
        """Return a fresh worker with the same configuration.

        The ``BaseLlm`` is *shared* across clones — it is stateless per
        call — so the clone is built with ``llm=None`` and the agent sets
        the LLM at runtime. Only config (verbose) is carried over; runtime
        state (tokens, time, GraphAutoma execution state) starts clean.
        Subclasses with extra ``__init__`` params should override.

        Used by ``ThinkUnitDescriptor._clone_worker`` for state isolation
        at every ``yield ThinkUnit(...)``.
        """
        return type(self)(
            llm=None,
            verbose=self._verbose,
        )

    ############################################################################
    # Template Methods
    ############################################################################

    async def observation(self, ota_context: OTAContext, context: Optional[Context] = None) -> Any:
        """Worker-level observation hook. Override to customize.

        Both forms accepted: coroutine (``return _DELEGATE`` /
        ``return value``) or async-generator (yield side-effect calls,
        then ``yield RETURN(value)``).

        Returning ``_DELEGATE`` (or ``None``) hands off to
        ``AmphibiousAutoma.observation()``. Other values become the
        observation directly.

        >>> async def observation(self, ota_context):
        ...     return f"Current state: {ota_context.user_input}"
        """
        return _DELEGATE
    
    async def thinking(self, ota_context: OTAContext, context: Optional[Context] = None) -> Any:
        """Assemble context, call the model, return its result. **Override this.**

        The worker's one job and override point: turn the two contexts into a
        prompt and call ``self._llm`` however the model needs, then return that
        call's **natural** result — :meth:`_assemble_decision` adapts it. Just
        return what the bridgic protocol hands you:

        - ``achat`` -> a ``Response`` (content-only, finished).
        - ``aselect_tool`` -> ``(tool_calls, content)``.
        - ``astructured_output`` -> a pydantic ``BaseModel`` or a ``dict``.
        - ``astream`` -> consume the ``MessageChunk`` deltas yourself (forward
          them to a live callback if you want), then return the accumulated
          ``str`` (or a ``Response``).

        Two-loop inputs: ``ota_context`` is the small-loop OTA context (its
        ``user_input`` + ``ota_record`` round trace form the task, its
        ``tools`` list the action affordances); ``context`` is the free-form
        knowledge context (``None`` for a pure-reasoning run).

        >>> async def thinking(self, ota_context, context=None):
        ...     msgs = my_messages(ota_context, context)
        ...     return await self._llm.aselect_tool(messages=msgs, tools=...)
        """
        raise NotImplementedError(
            "CognitiveWorker.thinking must be overridden: assemble the prompt "
            "from the two contexts, call self._llm, and return "
            "(content, tool_calls), a pydantic BaseModel, or text."
        )

    async def before_action(self, ota_context: OTAContext, context: Optional[Context] = None) -> Any:
        """Worker-level before_action hook. Override to intercept the decision.

        No decision argument — the pending decision is already on the current
        OTA round: read it from ``ota_context.think_result``. Return
        ``_DELEGATE`` / ``None`` to chain to the agent-level hook; return any
        other value (or ``yield RETURN(...)``) to override the decision before
        the act phase runs.
        """
        return _DELEGATE

    async def after_action(self, ota_context: OTAContext, context: Optional[Context] = None) -> Any:
        """Worker-level after_action hook. Override for side-effects.

        No result argument — the action result is already on the current OTA
        round: read it from ``ota_context.action_result``. The return value is
        a control signal: ``_DELEGATE`` / ``None`` chains to the agent-level
        hook; any other value suppresses it. Folding extra fields onto
        ``ota_context`` is the framework's one sanctioned in-place seam (the
        round trace is mutated by design); do not rely on mutating
        user-supplied data elsewhere.
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
        """Execute the thinking phase. The automa runs the observe step first; this round's observation is on the latest ``OTARecord`` (read via ``.obs_result``)."""
        start_time = time.monotonic()
        result = await super().arun(*args, feedback_data=feedback_data, **kwargs)
        self.spent_time += time.monotonic() - start_time
        return result


__all__ = ["CognitiveWorker"]
