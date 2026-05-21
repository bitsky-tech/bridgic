"""
CognitiveWorker — the in-process "think" unit of the amphibious framework.

Each ``arun`` call performs exactly the thinking phase: observation is
injected by ``AmphibiousAutoma._run_think_unit`` before calling, and
action is executed by ``AmphibiousAutoma._run_action_call`` after.

The thinking phase is the single template method
:meth:`CognitiveWorker.thinking`: it talks to the LLM and returns a
``(content, tool_calls)`` pair, which the framework parses into a decision.
The default ``thinking`` uses native function-calling; overriding it is
essentially "write your own LLM call" — the framework only supplies the
hooks and wires the result into the amphibious orchestration.

For *external*-agent delegation (out-of-process CLI), see the symmetric
``AgentWorker`` in ``_agent_worker.py``.
"""

import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel

from bridgic.core.model import BaseLlm
from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import Message
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.utils._console import printer
from bridgic.amphibious._context import CognitiveContext
from bridgic.amphibious._type import (
    StepToolCall,
    ThinkDecision,
    ToolArgument,
    TypedThinkDecision,
)


#############################################################################
# Sentinel
#############################################################################

_DELEGATE = object()  # Worker returns this to delegate observation to Agent


def _tool_call_name(call: Any) -> str:
    """Read a tool call's name — accepts an object (``.name``) or a dict."""
    return call["name"] if isinstance(call, dict) else call.name


def _tool_call_args(call: Any) -> Dict[str, Any]:
    """Read a tool call's arguments — accepts an object (``.arguments``) or a dict."""
    args = call.get("arguments") if isinstance(call, dict) else call.arguments
    return args or {}


#############################################################################
# CognitiveWorker
#############################################################################

class CognitiveWorker(GraphAutoma):
    """Cognitive worker — one observe-think-act cycle.

    Observation and action execution are handled by ``AmphibiousAutoma`` as
    shared infrastructure. The worker owns one thing: the **thinking** step,
    the :meth:`thinking` template method.

    The common case needs no subclass — give a prompt and use the default
    thinking (native function-calling):

    >>> worker = CognitiveWorker.inline("Plan ONE next step.", llm=llm)

    Override :meth:`thinking` to take full control of the LLM interaction —
    a different protocol, plain-text tool-call parsing for a model without
    native function-calling, multiple calls, etc. Optional hooks
    ``observation`` / ``before_action`` / ``after_action`` refine the cycle
    further.

    Class attribute ``output_schema``: if set, the worker produces a typed
    Pydantic instance via structured output instead.
    """

    # Subclasses set this to a Pydantic model to produce typed output directly
    output_schema: Optional[Type[BaseModel]] = None

    # Instruction text the DEFAULT thinking() prepends to the system message.
    # Set it on a subclass (``class X(CognitiveWorker): prompt = "..."``) or
    # via ``CognitiveWorker.inline()``. Ignored when thinking() is overridden.
    prompt: str = ""

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        super().__init__()

        self._llm = llm

        # Instance-level output_schema overrides the class attribute when provided
        if output_schema is not None:
            self.output_schema = output_schema

        # Logging runtime (None = inherit from AmphibiousAutoma)
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Usage stats
        self.spent_tokens = 0
        self.spent_time = 0

    def set_llm(self, llm: BaseLlm) -> None:
        """Set the LLM used for thinking. Replaces any previously set LLM."""
        self._llm = llm

    ############################################################################
    # Core methods
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _thinking(self, context: CognitiveContext) -> Any:
        """Framework entry for the thinking phase — orchestrates :meth:`thinking`.

        Validates the context / LLM, calls the overridable :meth:`thinking`
        method to interact with the LLM, then *parses* its ``(content,
        tool_calls)`` result into the framework's decision shape. That
        decision becomes the ``arun()`` return value — the driver captures
        it and runs the act phase.

        The seam: ``thinking`` owns *talking to the model*, the framework
        owns *turning the reply into a decision*.
        """
        if not isinstance(context, CognitiveContext):
            raise TypeError(
                f"Expected CognitiveContext, got {type(context).__name__}. "
                "CognitiveWorker requires CognitiveContext or its subclass."
            )
        if self._llm is None:
            raise RuntimeError(
                "CognitiveWorker has no LLM set. Either pass llm= in __init__ "
                "or use set_llm() before running."
            )

        # Typed-output workers (output_schema) are a separate structured
        # mode — making it model-agnostic is a planned follow-up.
        if self.output_schema is not None:
            return await self._think_typed_output(context)

        content, tool_calls = await self.thinking(context)
        return self._assemble_decision(content, tool_calls)

    async def _think_typed_output(self, context: CognitiveContext) -> TypedThinkDecision:
        """Structured-output path for ``output_schema`` workers.

        The LLM emits the worker's ``output_schema`` directly; the result is
        wrapped in a ``TypedThinkDecision``. Kept on ``astructured_output``
        for now — unifying it with the native :meth:`thinking` path is a
        planned follow-up.
        """
        messages = self._build_messages(context)
        self._log_prompt("Think", messages)
        result = await self._llm.astructured_output(
            messages=messages,
            constraint=PydanticModel(model=self.output_schema),
        )
        return TypedThinkDecision(finish=True, output=result)

    def _assemble_decision(self, content: Optional[str], tool_calls: List[Any]) -> ThinkDecision:
        """Parse a ``(content, tool_calls)`` result into a ``ThinkDecision``.

        Framework-internal — turns the ``thinking()`` reply into the shape
        the act phase consumes (``step_content`` / ``finish`` / ``output``):

        - text reply   -> ``step_content``
        - tool calls   -> ``output`` (``List[StepToolCall]``)
        - no tool call -> ``finish=True``

        Tool-call items may be objects (``.name`` / ``.arguments``) or
        dicts — so an overridden ``thinking`` can return either form
        without importing framework types.
        """
        output = [
            StepToolCall(
                tool=_tool_call_name(call),
                tool_arguments=[
                    ToolArgument(name=str(name), value=value)
                    for name, value in _tool_call_args(call).items()
                ],
            )
            for call in tool_calls
        ]
        return ThinkDecision(
            step_content=content or "",
            finish=(len(tool_calls) == 0),
            output=output,
        )

    @staticmethod
    def _extract_chat_content(response: Any) -> str:
        """Pull plain text out of a ``BaseLlm.achat`` response."""
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content or ""

    ############################################################################
    # Internal helpers
    ############################################################################

    def _build_messages(self, context: CognitiveContext) -> List[Message]:
        """Assemble the default messages — the worker's ``prompt`` + the context.

        Deliberately adds NO framework-authored instruction or framing text:
        the system message is exactly ``self.prompt``, the user message is
        the context summary. To put anything else in the prompt, write it
        into ``prompt`` or override :meth:`thinking`. Tools are not listed
        here — they reach the LLM through native function-calling.
        """
        summary_dict = context.summary()

        # `tools` is excluded — tools reach the LLM via native function-
        # calling, so repeating them as prompt text would be redundant.
        context_parts = [
            summary_dict[f]
            for f in summary_dict
            if f != 'tools' and summary_dict.get(f)
        ]
        if context.observation is not None:
            context_parts.append(f"Observation:\n{context.observation}")
        context_text = "\n\n".join(context_parts)

        messages: List[Message] = []
        if self.prompt.strip():
            messages.append(Message.from_text(text=self.prompt.strip(), role="system"))
        if context_text:
            messages.append(Message.from_text(text=context_text, role="user"))

        self.spent_tokens += sum(self._count_tokens(m.content) for m in messages)
        return messages

    def _log_prompt(self, stage: str, messages: List[Message]):
        """Log prompts with timestamp and caller location if verbose_prompt is enabled."""
        if not self._verbose_prompt:
            return
        import inspect
        from datetime import datetime
        from os.path import basename

        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                filename = basename(caller.f_code.co_filename)
                lineno = caller.f_lineno
            else:
                filename, lineno = "?", 0
        finally:
            del frame

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        total_tokens = sum(self._count_tokens(m.content) for m in messages)
        for i, msg in enumerate(messages):
            tokens = self._count_tokens(msg.content)
            printer.print(f"[{ts}] [{stage}] ({filename}:{lineno}) Message {i+1} ({msg.role}, {tokens} tokens):", color="cyan")
            printer.print(msg.content, color="gray")
        printer.print(f"[{ts}] [{stage}] ({filename}:{lineno}) Total: {total_tokens} tokens (cumulative: {self.spent_tokens})", color="yellow")

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Rough approximation: ~4 chars per token."""
        return (len(text) + 3) // 4

    ############################################################################
    # Hooks (override to customize the cycle)
    ############################################################################

    async def observation(self, context: CognitiveContext) -> Any:
        """Worker-level observation hook. Override to customize.

        Both forms accepted: coroutine (``return _DELEGATE`` /
        ``return value``) or async-generator (yield side-effect calls,
        then ``yield RETURN(value)``).

        Returning ``_DELEGATE`` (or ``None``) hands off to
        ``AmphibiousAutoma.observation()``. Other values become the
        observation directly.

        >>> async def observation(self, context):
        ...     return f"Current state: {context.goal}"
        """
        return _DELEGATE
    
    async def thinking(self, context: CognitiveContext) -> Tuple[str, List[Any]]:
        """Interact with the LLM and return ``(content, tool_calls)``.

        **This is the worker's one template method — the override point.**

        Returns a ``(content, tool_calls)`` pair:

        - ``content``: the model's text — its reasoning / what it is doing.
        - ``tool_calls``: the tool calls it wants to run — a list of items,
          each an object with ``.name`` / ``.arguments`` *or* a
          ``{"name": ..., "arguments": {...}}`` dict. An empty list means
          "nothing more to do" (the framework then marks it finished).

        ``thinking`` is purely about *talking to the model*. It does NOT
        build the framework's decision object — the framework parses the
        returned pair (see :meth:`_thinking`). Overriding it is essentially
        "write your own LLM call": use ``self._llm`` and ``context``
        however the model needs, and hand back ``(content, tool_calls)``.

        The default implementation below uses native function-calling
        (``aselect_tool``, or ``achat`` when there are no tools). It builds
        the messages from ``self.prompt`` and the context alone — it injects
        no instruction text of its own. Override it freely for a model that
        needs something else.

        >>> async def thinking(self, context):
        ...     # custom model with no native function-calling
        ...     resp = await self._llm.achat(messages=my_messages(context))
        ...     return my_parser(self._extract_chat_content(resp))  # -> (content, tool_calls)
        """
        messages = self._build_messages(context)
        self._log_prompt("Think", messages)

        _, tool_specs = context.get_field('tools')
        tools = [spec.to_tool() for spec in tool_specs] if tool_specs else []
        if tools:
            tool_calls, content = await self._llm.aselect_tool(
                messages=messages, tools=tools
            )
        else:
            response = await self._llm.achat(messages=messages)
            content, tool_calls = self._extract_chat_content(response), []

        return content, tool_calls

    async def before_action(
        self,
        decision_result: Any,
        context: CognitiveContext
    ) -> Any:
        """Worker-level before_action hook. Override to intercept tool calls.

        Return ``_DELEGATE`` / ``None`` to chain to the agent-level hook;
        return any other value (or ``yield RETURN(...)``) to override the
        decision.
        """
        return _DELEGATE

    async def after_action(self, step_result: Any, ctx: "CognitiveContext") -> Any:
        """Worker-level after_action hook. Override for side-effects.

        The return value is a control signal: ``_DELEGATE`` / ``None``
        chains to the agent-level hook; any other value suppresses it.
        Mutate ``ctx`` / ``step_result`` in place for changes to survive.
        """
        return _DELEGATE

    ############################################################################
    # Entry point
    ############################################################################

    @classmethod
    def inline(
        cls,
        prompt: str,
        llm: Optional[BaseLlm] = None,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> "CognitiveWorker":
        """Create a worker that uses the default ``thinking`` with ``prompt``.

        Convenience for the no-subclass case — equivalent to subclassing and
        setting the ``prompt`` class attribute.

        >>> worker = CognitiveWorker.inline("Plan ONE immediate next step", llm=llm)
        """
        worker = cls(
            llm=llm,
            verbose=verbose,
            verbose_prompt=verbose_prompt,
            output_schema=output_schema,
        )
        worker.prompt = prompt
        return worker

    # Alias for inline()
    from_prompt = inline

    async def arun(
        self,
        *args: Any,
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the thinking phase. Observation must be pre-set in context.observation."""
        start_time = time.time()
        result = await super().arun(*args, feedback_data=feedback_data, **kwargs)
        self.spent_time += time.time() - start_time
        return result


__all__ = ["CognitiveWorker"]
