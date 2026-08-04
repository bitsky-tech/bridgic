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

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model.types import ToolCall

from bridgic.amphibious._cognitive_worker import CognitiveWorker, _DELEGATE
from bridgic.amphibious._type import (
    ActionResult,
    ActionStepResult,
    ErrorStrategy,
    RETURN,
    generate_tool_call_id,
)
from bridgic.amphibious._context import OTAContext, Context


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

    # TODO: Refactor this case about standalone runner in future.
    async def arun(
        self,
        *,
        llm: Optional[Any] = None,
        user_input: Any = "",
        ota_context: Optional[OTAContext] = None,
        context: Optional[Context] = None,
        tools: Optional[List[ToolSpec]] = None,
        until: Optional[Union[Callable[..., bool], Callable[..., Awaitable[bool]]]] = None,
        max_attempts: Optional[int] = None,
    ) -> Any:
        """Run this think unit directly outside an ``AmphibiousAutoma``.

        This standalone runner mirrors only the ``CognitiveWorker`` OTA path:
        observe, think, worker ``before_action``, tool action, worker
        ``after_action``. It deliberately does not support framework yield
        primitive dispatch inside hooks.
        """
        run_until = until if until is not None else self._until
        run_max_attempts = (
            max_attempts if max_attempts is not None else self._max_attempts
        )

        worker = self._clone_worker(self._worker_template)
        if worker._llm is None:
            worker._llm = (
                llm if llm is not None
                else getattr(self._worker_template, "_llm", None)
            )
        if worker._llm is None:
            raise RuntimeError(
                "Standalone ThinkUnit requires an LLM. Pass llm=... to "
                "arun(), or set an LLM on the CognitiveWorker template."
            )

        ota_ctx = (
            ota_context if ota_context is not None
            else OTAContext(user_input=user_input)
        )
        if tools is not None:
            ota_ctx.tools = list(tools)
        loop_ctx = context if context is not None else Context()

        async def _invoke_worker_hook(gen_or_coro: Any) -> Any:
            if not inspect.isasyncgen(gen_or_coro):
                return await gen_or_coro

            return_value: Any = None
            try:
                while True:
                    try:
                        item = await gen_or_coro.__anext__()
                    except StopAsyncIteration:
                        break
                    if isinstance(item, RETURN):
                        return_value = item.value
                        break
                    raise RuntimeError(
                        "Standalone ThinkUnit hooks do not dispatch framework "
                        f"yield primitives ({type(item).__name__}). Return a "
                        "value, yield RETURN(value), or run inside "
                        "AmphibiousAutoma for full hook dispatch."
                    )
            finally:
                try:
                    await gen_or_coro.aclose()
                except Exception:
                    pass
            return return_value

        def _matched_tool_calls() -> List[Tuple[ToolCall, ToolSpec]]:
            calls = getattr(ota_ctx.think_result, "tool_calls", None) or []
            matched: List[Tuple[ToolCall, ToolSpec]] = []

            for call in calls:
                tool_spec = next(
                    (s for s in ota_ctx.tools if s.tool_name == call.tool),
                    None,
                )
                if tool_spec is None:
                    continue

                param_types: Dict[str, str] = {}
                param_names: List[str] = []
                if tool_spec.tool_parameters:
                    properties = tool_spec.tool_parameters.get("properties", {})
                    param_names = list(properties.keys())
                    for name, info in properties.items():
                        param_types[name] = info.get("type", "string")

                arguments: Dict[str, Any] = {}
                for arg in call.tool_arguments:
                    value: Any = arg.value
                    param_type = param_types.get(arg.name, "string")
                    if param_type == "integer":
                        try:
                            value = int(value)
                        except (TypeError, ValueError):
                            pass
                    elif param_type == "number":
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            pass
                    elif param_type == "boolean":
                        value = str(value).lower() in ("true", "1", "yes")
                    arguments[arg.name] = value

                if arguments.get("__args__") is not None:
                    args = arguments["__args__"]
                    if isinstance(args, list):
                        arguments = dict(zip(param_names, args))
                    else:
                        arguments = {param_names[0]: args} if param_names else {}

                matched.append((
                    ToolCall(
                        id=getattr(call, "call_id", None) or generate_tool_call_id(),
                        name=call.tool,
                        arguments=arguments,
                    ),
                    tool_spec,
                ))

            return matched

        async def _action_tool_call() -> ActionResult:
            matched = _matched_tool_calls()

            async def _run_one(
                tool_call: ToolCall, tool_spec: ToolSpec,
            ) -> ActionStepResult:
                tool_worker = tool_spec.create_worker()
                sandbox = ConcurrentAutoma()
                worker_key = f"tool_{tool_call.name}_{tool_call.id}"
                sandbox.add_worker(
                    key=worker_key,
                    worker=tool_worker,
                    args_mapping_rule=ArgsMappingRule.UNPACK,
                )
                try:
                    results = await sandbox.arun(InOrder([tool_call.arguments]))
                    result = results[0] if results else None
                    return ActionStepResult(
                        tool_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_arguments=tool_call.arguments,
                        tool_result=result,
                        success=True,
                    )
                except Exception as e:
                    return ActionStepResult(
                        tool_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_arguments=tool_call.arguments,
                        tool_result=None,
                        success=False,
                        error=str(e),
                    )

            step_results = await asyncio.gather(
                *(_run_one(tc, ts) for tc, ts in matched)
            )
            return ActionResult(results=list(step_results))

        async def _run_observe_think_act() -> Tuple[bool, Any]:
            ota_ctx.open_record()

            obs = await _invoke_worker_hook(worker.observation(ota_ctx, loop_ctx))
            if obs is not _DELEGATE and obs is not None:
                ota_ctx.obs_result = obs

            decision = await worker.arun(ota_context=ota_ctx, context=loop_ctx)
            ota_ctx.think_result = decision
            if decision.tool_calls == []:
                ota_ctx.action_result = None
                return True, decision.step_content

            before_ret = await _invoke_worker_hook(
                worker.before_action(ota_ctx, loop_ctx)
            )
            if before_ret is not _DELEGATE and before_ret is not None:
                ota_ctx.think_result = before_ret

            action_result = await _action_tool_call()
            ota_ctx.action_result = action_result

            await _invoke_worker_hook(worker.after_action(ota_ctx, loop_ctx))
            return False, decision.step_content

        result: Any = None
        for _cycle_idx in range(run_max_attempts):
            try:
                finished, result = await _run_observe_think_act()
            except Exception as e:
                if self._on_error == ErrorStrategy.RAISE:
                    raise RuntimeError(
                        "Standalone ThinkUnit failed during "
                        f"observe-think-act cycle: {e}"
                    ) from e
                if self._on_error == ErrorStrategy.IGNORE:
                    finished = False
                elif self._on_error == ErrorStrategy.RETRY:
                    finished = False
                    for attempt in range(self._max_retries + 1):
                        try:
                            finished, result = await _run_observe_think_act()
                            break
                        except Exception as retry_e:
                            if attempt == self._max_retries:
                                raise RuntimeError(
                                    "Standalone ThinkUnit failed after "
                                    f"{self._max_retries + 1} retries: "
                                    f"{retry_e}"
                                ) from retry_e
            else:
                if finished:
                    break
                if run_until is not None:
                    cond_result = run_until(ota_ctx)
                    if inspect.iscoroutine(cond_result):
                        cond_result = await cond_result
                    if cond_result:
                        break

        return result


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
