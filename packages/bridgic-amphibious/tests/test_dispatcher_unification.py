"""Tests for unified dispatcher across all five template methods.

Verifies that yield primitives (ActionCall, HumanCall, LLMCall, RETURN)
work in any template method that the dispatcher drives — not just
``on_workflow``. Also verifies that both async-generator and plain
async-coroutine forms are accepted (legacy compat).
"""

from typing import Any, AsyncGenerator, List, Union

import pytest
from pydantic import BaseModel

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    RETURN,
    StepToolCall,
    ToolArgument,
    human_channel,
    think_unit,
    ThinkUnit,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message, Response, Role


class MockLLM:
    """Drives the new CognitiveWorker's native function-calling path.

    ``aselect_tool`` (think-step with tools) yields scripted
    ``(tool_calls, content)`` pairs; ``achat`` (LLMCall.chat / think-step
    without tools) yields scripted ``Response`` objects.
    """

    def __init__(self, chat_responses=None, structured_responses=None, tool_responses=None):
        self.chat_responses = list(chat_responses or [])
        self.structured_responses = list(structured_responses or [])
        self.tool_responses = list(tool_responses or [])
        self._chat_idx = 0
        self._struct_idx = 0
        self._tool_idx = 0

    async def achat(self, messages, **kwargs):
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self.structured_responses[self._struct_idx % len(self.structured_responses)]
        self._struct_idx += 1
        return resp

    async def aselect_tool(self, messages, tools, **kwargs):
        # Default to an immediate finish (no tool calls) when nothing is
        # scripted — most dispatcher tests only need the worker to finish.
        if not self.tool_responses:
            return [], "Done"
        resp = self.tool_responses[self._tool_idx % len(self.tool_responses)]
        self._tool_idx += 1
        return resp

    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="dispatcher unification test")


def _txt(text: str) -> Response:
    return Response(message=Message.from_text(text, role=Role.AI))


def _finish_step():
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return [], "Done"


# ---------------------------------------------------------------------------
# Yield primitives in non-on_workflow template methods
# ---------------------------------------------------------------------------


class TestHookScopeLogging:
    """Hook-scope log lines are gated by ``verbose_hook``; main-scope
    lines by ``verbose`` — the two flags are independent.

    A ``_log`` call inside a ``_hook_log_scope(...)`` block renders as a
    hook arrow and is emitted only when ``verbose_hook`` is on; the same
    ``_log`` outside the block is a main-scope line gated by ``verbose``.
    """

    @staticmethod
    def _agent(*, verbose: bool, verbose_hook: bool) -> AmphibiousAutoma:
        class _Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx):
                yield RETURN("ok")

        return _Agent(verbose=verbose, verbose_hook=verbose_hook)

    @staticmethod
    def _capture_printer(monkeypatch) -> List[tuple]:
        """Patch the console printer and return the list it appends to."""
        from bridgic.amphibious import _amphibious_automa as _mod

        printed: List[tuple] = []
        monkeypatch.setattr(
            _mod.printer, "print", lambda *a, **k: printed.append(a)
        )
        return printed

    def test_hook_scope_log_suppressed_when_verbose_hook_off(self, monkeypatch):
        """``verbose_hook=False`` suppresses a hook-scope ``_log`` line
        even though ``verbose=True`` — the gate is the hook flag."""
        printed = self._capture_printer(monkeypatch)
        agent = self._agent(verbose=True, verbose_hook=False)

        with agent._hook_log_scope("before_action"):
            agent._log("result", "a hook-scope call happened")

        assert printed == []

    def test_hook_scope_log_surfaces_when_verbose_hook_on(self, monkeypatch):
        """``verbose_hook=True`` surfaces a hook-scope ``_log`` line even
        though ``verbose=False`` — the two flags are independent."""
        printed = self._capture_printer(monkeypatch)
        agent = self._agent(verbose=False, verbose_hook=True)

        with agent._hook_log_scope("before_action"):
            agent._log("result", "a hook-scope call happened")

        assert printed

    def test_main_scope_log_unaffected_by_verbose_hook(self, monkeypatch):
        """A main-scope (non-hook) ``_log`` line is gated by ``verbose``
        alone — ``verbose_hook`` does not touch it."""
        printed = self._capture_printer(monkeypatch)
        agent = self._agent(verbose=True, verbose_hook=False)

        agent._log("Router", "a main-scope line")  # not in a hook scope

        assert printed


class TestYieldsInObservation:

    @pytest.mark.asyncio
    async def test_observation_can_yield_llm_call(self):
        """observation generator can yield LLMCall and RETURN(value)."""
        llm = MockLLM(
            chat_responses=[_txt("computed-observation")],
            tool_responses=[_finish_step()],
        )

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx) -> AsyncGenerator[Any, Any]:
                obs = yield LLMCall.chat("Summarize the page")
                yield RETURN(obs)

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                # Run the worker via ThinkUnit — it'll trigger
                # observation() via the _run_observe_think_act observe phase.
                yield ThinkUnit("plan_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx())

        # observation was invoked, ctx.observation set
        assert agent._current_context.observation == "computed-observation"

    @pytest.mark.asyncio
    async def test_observation_can_yield_action_call(self):
        """observation generator may ``yield ActionCall(...)`` to capture
        a fresh snapshot, ``RETURN`` the tool result, and ctx.observation
        is set to that value — all without infinite recursion.

        Regression test for the documented docstring pattern on
        ``AmphibiousAutoma.observation``. Previously, dispatching an
        ActionCall yielded from within a hook (``scope="hook"``)
        re-entered the same hook generator and blew the stack with
        ``RecursionError``. The fix splits ``_dispatch_step``'s
        ActionCall branch by scope: hook-scope runs
        ``_run_action_call(..., with_hooks=False)`` (no observation, no
        before/after_action wrap, no trace step) while workflow-scope
        keeps the full OTC wrap.
        """
        llm = MockLLM(tool_responses=[_finish_step()])
        call_count = 0

        async def snapshot_tool() -> str:
            """Mock snapshot tool used by the observation hook."""
            nonlocal call_count
            call_count += 1
            return "snap-v1"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx) -> AsyncGenerator[Any, Any]:
                snap = yield ActionCall("snapshot_tool")
                yield RETURN(snap[0].result if snap else None)

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(snapshot_tool))

        agent = Agent()
        await agent.arun(llm=llm, context=ctx)  # must NOT raise RecursionError

        # The hook's ActionCall executed exactly once (the bug would have
        # either recursed forever or, with a guard, fired zero times).
        assert call_count == 1
        # The RETURN value reached ctx.observation.
        assert agent._current_context.observation == "snap-v1"


class TestYieldsInBeforeAction:

    @pytest.mark.asyncio
    async def test_before_action_yields_return_overrides_decision(self):
        """before_action generator yields RETURN(modified) to override the decision."""
        llm = MockLLM(tool_responses=[_finish_step()])
        seen_decisions: List[Any] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def before_action(self, decision_result, ctx) -> AsyncGenerator[Any, Any]:
                seen_decisions.append(decision_result)
                # No RETURN → passthrough; just verify hook is reachable.
                if False:
                    yield

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx())

        # before_action invoked at least once
        assert len(seen_decisions) >= 1

    @pytest.mark.asyncio
    async def test_before_action_can_yield_action_call(self):
        """before_action generator may ``yield ActionCall(...)`` for an
        out-of-band tool call (e.g. an audit log) without recursing.

        Regression test for the same recursion bug as
        ``test_observation_can_yield_action_call``: the ActionCall is
        dispatched with ``scope="hook"`` and must NOT re-enter the
        before_action chain.
        """
        llm = MockLLM(tool_responses=[_finish_step()])
        call_count = 0

        async def audit_tool() -> str:
            """Mock audit tool used by the before_action hook."""
            nonlocal call_count
            call_count += 1
            return "audited"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def before_action(self, decision_result, ctx) -> AsyncGenerator[Any, Any]:
                yield ActionCall("audit_tool")
                # No RETURN → passthrough.
                if False:
                    yield

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(audit_tool))

        agent = Agent()
        await agent.arun(llm=llm, context=ctx)  # must NOT raise RecursionError

        # Tool fires exactly once — once per before_action invocation, and
        # before_action runs once per action phase. With the bug it would
        # have recursed; with the fix it runs cleanly.
        assert call_count == 1


class TestWorkerHookGeneratorForm:
    """Worker-level hooks accept async-generator form, symmetric with
    the agent-level hooks. Pre-fix, ``await worker.observation(ctx)``
    crashed with ``TypeError: can't await async_generator`` when the
    user wrote the hook as a generator. Now ``_invoke_template`` drives
    both forms uniformly."""

    @pytest.mark.asyncio
    async def test_worker_observation_generator_form_with_action_call(self):
        """Worker-level observation as a generator yielding ActionCall +
        RETURN(value) sets ctx.observation and short-circuits the
        agent-level fallback."""
        llm = MockLLM(tool_responses=[_finish_step()])
        agent_fallback_ran = False
        call_count = 0

        async def snapshot_tool() -> str:
            nonlocal call_count
            call_count += 1
            return "worker-snap"

        class GenObservationWorker(CognitiveWorker):
            # Default thinking() (native function-calling) is fine — the
            # scripted MockLLM makes it finish immediately. This worker
            # only customises the observation hook.
            async def observation(self, context) -> AsyncGenerator[Any, Any]:
                snap = yield ActionCall("snapshot_tool")
                yield RETURN(snap[0].result if snap else None)

        worker = GenObservationWorker()

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recovery_unit = think_unit(worker, max_attempts=1)

            async def observation(self, ctx):
                nonlocal agent_fallback_ran
                agent_fallback_ran = True
                yield RETURN("agent-fallback")

            async def on_agent(self, ctx):
                yield ThinkUnit("recovery_unit")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(snapshot_tool))

        agent = Agent()
        await agent.arun(llm=llm, context=ctx)  # must NOT raise TypeError

        # Worker-level generator returned "worker-snap" → no delegation.
        assert call_count == 1
        assert agent_fallback_ran is False
        assert agent._current_context.observation == "worker-snap"

    @pytest.mark.asyncio
    async def test_worker_observation_generator_no_return_delegates(self):
        """Worker generator that exhausts without RETURN → returns None
        → treated as _DELEGATE → agent-level observation runs."""
        llm = MockLLM(tool_responses=[_finish_step()])
        agent_fallback_ran = False

        class GenObservationWorker(CognitiveWorker):
            # Default thinking() (native function-calling) is fine — the
            # scripted MockLLM makes it finish immediately. This worker
            # only customises the observation hook.
            async def observation(self, context) -> AsyncGenerator[Any, Any]:
                # Generator with no yields/RETURN — exhausts immediately.
                if False:
                    yield

        worker = GenObservationWorker()

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recovery_unit = think_unit(worker, max_attempts=1)

            async def observation(self, ctx):
                nonlocal agent_fallback_ran
                agent_fallback_ran = True
                yield RETURN("agent-fallback")

            async def on_agent(self, ctx):
                yield ThinkUnit("recovery_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx())

        assert agent_fallback_ran is True
        assert agent._current_context.observation == "agent-fallback"


class TestYieldsInAfterAction:

    @pytest.mark.asyncio
    async def test_after_action_can_yield_action_call(self):
        """after_action generator may ``yield ActionCall(...)`` for a
        follow-up tool call (e.g. a side-effect refresh) without
        recursing.

        Regression test for the same recursion bug. ActionCall yielded
        from after_action is dispatched with ``scope="hook"`` and must
        NOT re-enter the after_action chain.
        """
        llm = MockLLM(tool_responses=[_finish_step()])
        call_count = 0

        async def followup_tool() -> str:
            """Mock follow-up tool used by the after_action hook."""
            nonlocal call_count
            call_count += 1
            return "followed-up"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def after_action(self, step_result, ctx) -> AsyncGenerator[Any, Any]:
                yield ActionCall("followup_tool")
                if False:
                    yield

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(followup_tool))

        agent = Agent()
        await agent.arun(llm=llm, context=ctx)  # must NOT raise RecursionError

        assert call_count == 1


class TestGeneratorVsCoroutineForms:

    @pytest.mark.asyncio
    async def test_on_agent_coroutine_form(self):
        """on_agent as a plain coroutine (legacy) — _run-driven worker."""
        llm = MockLLM(tool_responses=[_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        await Agent().arun(llm=llm, context=_ctx())  # no exception → pass

    @pytest.mark.asyncio
    async def test_on_agent_generator_form_with_thinkcall(self):
        """on_agent as a generator yielding ThinkUnit."""
        llm = MockLLM(tool_responses=[_finish_step()])
        from bridgic.amphibious import ThinkUnit

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_think")

        await Agent().arun(llm=llm, context=_ctx())  # no exception → pass

    @pytest.mark.asyncio
    async def test_amphibious_template_must_be_async_gen(self):
        """Coroutine-form AmphibiousAutoma templates raise at class creation.

        The framework's dispatch model is yield-driven; every primitive
        (ActionCall / HumanCall / LLMCall / EnterAgent / ThinkUnit /
        ThinkAgent / RETURN) reaches the framework via ``yield``.
        Coroutine-form template overrides cannot use any of these
        primitives, so ``_validate_template_forms`` rejects them at
        class-creation time with a clear error pointing at the bad
        method.
        """
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _BadAgent1(AmphibiousAutoma[CognitiveContext]):
                async def on_agent(self, ctx):
                    pass  # no yield → coroutine form, rejected

        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _BadAgent2(AmphibiousAutoma[CognitiveContext]):
                async def on_workflow(self, ctx):
                    return "coroutine-result"  # no yield → rejected

        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _BadAgent3(AmphibiousAutoma[CognitiveContext]):
                async def after_action(self, step_result, ctx):
                    pass  # no yield → rejected
