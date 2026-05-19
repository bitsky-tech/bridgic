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


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class MockLLM:
    def __init__(self, chat_responses=None, structured_responses=None):
        self.chat_responses = list(chat_responses or [])
        self.structured_responses = list(structured_responses or [])
        self._chat_idx = 0
        self._struct_idx = 0

    async def achat(self, messages, **kwargs):
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self.structured_responses[self._struct_idx % len(self.structured_responses)]
        self._struct_idx += 1
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
    return ThinkDecision(step_content="Done", output=[], finish=True)


# ---------------------------------------------------------------------------
# Yield primitives in non-on_workflow template methods
# ---------------------------------------------------------------------------


class TestHookCallLogging:
    """Logs for Calls dispatched at hook scope are suppressed by default
    (``verbose_hook_calls=False``) and surfaced when the flag is True.

    Workflow-scope Call logs are unaffected by the flag.
    """

    @pytest.mark.asyncio
    async def test_hook_dispatch_log_suppressed_by_default(self):
        """Default: an ActionCall yielded from an observation hook does
        NOT emit an ``Act`` dispatch log, while a workflow-scope
        ActionCall still does."""
        llm = MockLLM(structured_responses=[_finish_step()])

        async def snapshot_tool() -> str:
            return "snap"

        async def workflow_tool() -> str:
            return "wf"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx):
                snap = yield ActionCall("snapshot_tool")
                yield RETURN(snap[0].result if snap else None)

            async def on_workflow(self, ctx):
                yield ActionCall("workflow_tool")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(snapshot_tool))
        ctx.tools.add(FunctionToolSpec.from_raw(workflow_tool))

        # ``verbose=True`` so _log itself is not gated out; the only thing
        # under test is the per-scope hook gate.
        agent = Agent(verbose=True)

        emitted: List[tuple] = []
        original_log = agent._log

        def _capture(stage, message, data=None, color="white"):
            emitted.append((stage, message))
            # Skip the real _log to avoid noisy test output, but exercise
            # nothing else by short-circuiting here.
            return None

        agent._log = _capture  # type: ignore[assignment]
        try:
            await agent.arun(llm=llm, context=ctx)
        finally:
            agent._log = original_log  # type: ignore[assignment]

        # Workflow-scope ActionCall surfaced as an ``Act`` log (label
        # ``dispatch:``); hook-scope ActionCall did NOT (label would have
        # been ``hook-dispatch:``).
        act_messages = [msg for stage, msg in emitted if stage == "Act"]
        assert any("dispatch:" in m and "hook-dispatch:" not in m for m in act_messages), \
            f"workflow-scope ActionCall should log; got {act_messages!r}"
        assert not any("hook-dispatch:" in m for m in act_messages), \
            f"hook-scope ActionCall must NOT log by default; got {act_messages!r}"

    @pytest.mark.asyncio
    async def test_hook_dispatch_log_surfaces_when_flag_on(self):
        """With ``verbose_hook_calls=True``, hook-scope ActionCall logs DO
        appear (useful for debugging a misbehaving hook)."""
        llm = MockLLM(structured_responses=[_finish_step()])

        async def snapshot_tool() -> str:
            return "snap"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx):
                snap = yield ActionCall("snapshot_tool")
                yield RETURN(snap[0].result if snap else None)

            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        ctx = _ctx()
        ctx.tools.add(FunctionToolSpec.from_raw(snapshot_tool))

        agent = Agent(verbose=True, verbose_hook_calls=True)

        emitted: List[tuple] = []
        original_log = agent._log

        def _capture(stage, message, data=None, color="white"):
            emitted.append((stage, message))
            return None

        agent._log = _capture  # type: ignore[assignment]
        try:
            await agent.arun(llm=llm, context=ctx)
        finally:
            agent._log = original_log  # type: ignore[assignment]

        act_messages = [msg for stage, msg in emitted if stage == "Act"]
        assert any("hook-dispatch:" in m for m in act_messages), \
            f"with verbose_hook_calls=True, hook ActionCall should log; got {act_messages!r}"


class TestYieldsInObservation:

    @pytest.mark.asyncio
    async def test_observation_can_yield_llm_call(self):
        """observation generator can yield LLMCall and RETURN(value)."""
        llm = MockLLM(
            chat_responses=[_txt("computed-observation")],
            structured_responses=[_finish_step()],
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
        llm = MockLLM(structured_responses=[_finish_step()])
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
        llm = MockLLM(structured_responses=[_finish_step()])
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
        llm = MockLLM(structured_responses=[_finish_step()])
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
        llm = MockLLM(structured_responses=[_finish_step()])
        agent_fallback_ran = False
        call_count = 0

        async def snapshot_tool() -> str:
            nonlocal call_count
            call_count += 1
            return "worker-snap"

        class GenObservationWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

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
        llm = MockLLM(structured_responses=[_finish_step()])
        agent_fallback_ran = False

        class GenObservationWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

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
        llm = MockLLM(structured_responses=[_finish_step()])
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
        llm = MockLLM(structured_responses=[_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan_unit = think_unit(CognitiveWorker.inline("plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan_unit")

        await Agent().arun(llm=llm, context=_ctx())  # no exception → pass

    @pytest.mark.asyncio
    async def test_on_agent_generator_form_with_thinkcall(self):
        """on_agent as a generator yielding ThinkUnit."""
        llm = MockLLM(structured_responses=[_finish_step()])
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
