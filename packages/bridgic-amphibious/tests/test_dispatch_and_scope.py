"""Unified yield-dispatch: scope matrix, template-form validation,
hook-scope primitives + recursion fix, two-generic context resolution,
and the ``RETURN`` primitive.

Consolidates (post two-loop refactor):

* ``test_scope_rules.py``         — the yield-type ↔ scope matrix (negatives)
* ``test_dispatcher_unification.py`` — hook-scope primitives + recursion fix
* ``test_stub_override_robustness.py`` — template-form validation + worker stubs
* ``test_yield_return.py``        — the ``RETURN`` primitive
* ``test_minimal_context.py``     — two-generic context-class resolution

Scope matrix (``hooks`` = observation / before_action / after_action):

==============  =============  ==========  =======
primitive       on_workflow    on_agent    hooks
==============  =============  ==========  =======
ActionCall      OK             RAISE       OK
HumanCall       OK             RAISE       OK
LLMCall         OK             RAISE       OK
EnterAgent      OK             RAISE       RAISE
ThinkUnit       RAISE          OK          RAISE
==============  =============  ==========  =======

Every test defines its OWN ``AmphibiousAutoma`` / ``Context`` subclass —
``__init_subclass__`` builds per-class registries (``_human_channels`` /
``_declared_tools``) at class-creation time, so a shared subclass would
let one test pollute another.
"""

import types
from typing import Any, AsyncGenerator, Optional, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    Context,
    OTAContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    RETURN,
    ThinkUnit,
    think_unit,
    human_channel,
)
from bridgic.core.model.types import Message, Response, Role, ToolCall


# ---------------------------------------------------------------------------
# Shared LLM stubs (ported from the subsumed files).
# ---------------------------------------------------------------------------


class _DummyLLM:
    """Minimal LLM stub whose ``aselect_tool`` returns no tool calls, so a
    worker's ``thinking()`` resolves to an immediate finish."""

    async def aselect_tool(self, messages, tools, **kwargs):
        return [], "done"

    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class MockLLM:
    """Scripts the native function-calling path.

    ``aselect_tool`` (think-step with tools) yields scripted
    ``(tool_calls, content)`` pairs — an empty ``tool_calls`` list makes the
    worker finish; ``achat`` (LLMCall.chat) yields scripted ``Response``s.
    """

    def __init__(self, chat_responses=None, tool_responses=None):
        self.chat_responses = list(chat_responses or [])
        self.tool_responses = list(tool_responses or [])
        self._chat_idx = 0
        self._tool_idx = 0

    async def achat(self, messages, **kwargs):
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def aselect_tool(self, messages, tools, **kwargs):
        if not self.tool_responses:
            return [], "Done"
        resp = self.tool_responses[self._tool_idx % len(self.tool_responses)]
        self._tool_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class _NoopLLM:
    async def achat(self, messages, **kwargs):  # pragma: no cover - unused
        raise AssertionError("workflow ActionCall path must not call the LLM")


_GOAL = "dispatch-and-scope test"


def _ctx() -> Context:
    """Big-loop (knowledge) context for ``context=``. The goal is seeded into
    the per-run OTA context via the ``user_input=_GOAL`` arun kwarg."""
    return Context()


def _txt(text: str) -> Response:
    return Response(message=Message.from_text(text, role=Role.AI))


def _finish_step():
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return [], "Done"


def _make_agent(name: str, base, namespace: dict):
    """Build a fresh AmphibiousAutoma subclass dynamically.

    ``base`` is a parametrized ``AmphibiousAutoma[...]`` alias, so we go
    through ``types.new_class`` (it performs ``__mro_entries__`` resolution;
    plain ``type()`` does not). Each test gets its OWN subclass — the
    per-class registries are rebuilt at class creation.
    """
    return types.new_class(
        name, (base,), {}, lambda ns: ns.update(namespace)
    )


class _FinishWorker(CognitiveWorker):
    """Finishes immediately (no tool calls)."""

    async def thinking(self, ota_context, context=None):
        return await self._llm.aselect_tool(messages=[], tools=[])


class _PlanWorker(CognitiveWorker):
    """Native-function-calling worker: assembles the OTA tools and calls
    ``aselect_tool`` (the scripted ``MockLLM`` drives the decision)."""

    async def thinking(
        self, ota_context: OTAContext, context: Optional[Context] = None
    ) -> Any:
        return await self._llm.aselect_tool(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )


# ===========================================================================
# Scope matrix — negatives (driven via arun() or the documented _dispatch_step
# entry), plus the atomic-Call positives in workflow/hook scope.
# ===========================================================================


class TestScopeMatrix:
    """Yield-type ↔ scope rules. Negatives raise ``RuntimeError`` at the
    natural dispatch entry; the atomic Calls are legal in workflow + hook."""

    @pytest.mark.asyncio
    async def test_enter_agent_in_on_agent_raises(self):
        """EnterAgent yielded from on_agent → RuntimeError (workflow-only)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield EnterAgent(goal="re-enter")

        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await Agent().arun(llm=_DummyLLM(), context=_ctx(), user_input=_GOAL)

    @pytest.mark.asyncio
    async def test_enter_agent_in_hook_scope_raises(self):
        """EnterAgent dispatched with scope='hook' → RuntimeError."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

        agent = Agent()
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await agent._dispatch_step(EnterAgent(goal="x"), scope="hook")

    @pytest.mark.parametrize("scope", ["hook", "workflow"])
    @pytest.mark.asyncio
    async def test_think_unit_outside_on_agent_raises(self, scope):
        """ThinkUnit dispatched with scope hook/workflow → RuntimeError
        (agent-only)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            think_unit_a = think_unit(_FinishWorker(), max_attempts=1)

        agent = Agent()
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await agent._dispatch_step(ThinkUnit("think_unit_a"), scope=scope)

    @pytest.mark.parametrize(
        "make_call",
        [
            pytest.param(lambda: ActionCall("some_tool", x=1), id="action_call"),
            pytest.param(lambda: HumanCall(prompt="confirm"), id="human_call"),
            pytest.param(lambda: LLMCall.chat("hi"), id="llm_call"),
        ],
    )
    @pytest.mark.asyncio
    async def test_atomic_call_in_agent_scope_raises(self, make_call):
        """ActionCall / HumanCall / LLMCall dispatched with scope='agent'
        → RuntimeError (atomic Calls belong in workflow or hooks)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def feed(self, prompt: str) -> str:
                return "ok"

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

        agent = Agent()
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="not allowed inside on_agent"):
            await agent._dispatch_step(make_call(), scope="agent")

    @pytest.mark.parametrize("scope", ["workflow", "hook"])
    @pytest.mark.asyncio
    async def test_human_call_works_in_workflow_and_hook(self, scope):
        """HumanCall is legal in workflow + hook scope and routes to the
        single registered channel."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def feed(self, prompt: str) -> str:
                return "ok"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

        agent = Agent()
        agent._current_context = _ctx()
        outcome = await agent._dispatch_step(HumanCall(prompt="confirm"), scope=scope)
        assert outcome == "ok"


# ===========================================================================
# Template-form validation — coroutine-form overrides rejected at class
# creation; the ``if False: yield`` no-op stub is accepted.
# ===========================================================================


class TestTemplateFormValidation:
    """``_validate_template_forms`` rejects coroutine-form (no-``yield``)
    overrides of the five drivable template methods at class-creation time."""

    @pytest.mark.parametrize(
        "method_name",
        ["on_agent", "on_workflow", "observation", "before_action", "after_action"],
    )
    def test_coroutine_form_template_rejected(self, method_name):
        """Each of the five template methods, written as a coroutine (no
        ``yield``), raises ``TypeError`` when the class is created."""

        async def _coro_body(self, ota_context, context=None):
            pass  # no yield → coroutine form, rejected

        with pytest.raises(
            TypeError,
            match="must be an ``async def`` function with at least one ``yield``",
        ):
            _make_agent(
                "_BadAgent",
                AmphibiousAutoma[OTAContext, Context],
                {method_name: _coro_body},
            )

    def test_unreachable_yield_stub_accepted(self):
        """``if False: yield`` is the canonical no-op async-gen stub; it
        keeps the async-generator shape and ``_has_workflow()`` stays true."""

        class _NoOpAgent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(
                self, ota_context, context=None
            ) -> AsyncGenerator[Union[ActionCall, HumanCall, EnterAgent], None]:
                if False:  # pragma: no cover
                    yield

        agent = _NoOpAgent()
        assert agent._has_workflow() is True


# ===========================================================================
# Hook-scope primitives + recursion fix.
#
# Regression for the documented hook patterns: a primitive yielded from
# within a hook (scope="hook") must NOT re-enter the hook generator and blow
# the stack with RecursionError. The fix splits _dispatch_step's ActionCall
# branch by scope: hook-scope runs _run_action_call(..., with_hooks=False)
# (no observation, no before/after_action wrap, no trace step).
# ===========================================================================


class TestHookScopePrimitives:

    @pytest.mark.parametrize(
        "hook_name",
        ["observation", "before_action", "after_action"],
    )
    @pytest.mark.asyncio
    async def test_hook_yields_action_call_runs_once_no_recursion(self, hook_name):
        """observation / before_action / after_action may ``yield
        ActionCall(...)`` exactly once, with no RecursionError and without
        leaking the hook's own decision into the act phase.

        ``observation`` fires on the OBSERVE phase, so the worker can finish
        immediately. ``before_action`` / ``after_action`` fire only on an ACT
        cycle, so the worker first calls ``main_tool`` then finishes.
        """
        if hook_name == "observation":
            tool_responses = [_finish_step()]
        else:
            tool_responses = [
                ([ToolCall(id="c0", name="main_tool", arguments={})], "step"),
                _finish_step(),
            ]
        llm = MockLLM(tool_responses=tool_responses)
        call_count = 0

        async def hook_tool() -> str:
            """Out-of-band tool invoked by the hook (snapshot/audit/refresh)."""
            nonlocal call_count
            call_count += 1
            return "hooked"

        async def main_tool() -> str:
            """No-op tool the worker calls on its act cycle."""
            return "ok"

        class _OTA(OTAContext):
            pass

        _OTA.tool(hook_tool)
        _OTA.tool(main_tool)

        async def _hook_body(self, ota_context, context=None):
            yield ActionCall("hook_tool")
            if False:  # no RETURN → passthrough
                yield

        async def _on_agent(self, ota_context, context=None):
            yield ThinkUnit("plan_unit")

        Agent = _make_agent(
            "Agent",
            AmphibiousAutoma[_OTA, Context],
            {
                "plan_unit": think_unit(_PlanWorker()),
                hook_name: _hook_body,
                "on_agent": _on_agent,
            },
        )

        agent = Agent()
        # Must NOT raise RecursionError.
        await agent.arun(llm=llm, context=_ctx(), user_input=_GOAL)

        # The hook-scope ActionCall ran cleanly, exactly once — no re-entry
        # into the hook chain.
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_observation_yields_llm_call_and_return_sets_obs_result(self):
        """observation generator can yield LLMCall then RETURN(value); the
        returned value lands on ``ota_ctx.obs_result``."""
        llm = MockLLM(
            chat_responses=[_txt("computed-observation")],
            tool_responses=[_finish_step()],
        )

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def observation(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                obs = yield LLMCall.chat("Summarize the page")
                yield RETURN(obs)

            plan_unit = think_unit(_PlanWorker())

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("plan_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx(), user_input=_GOAL)

        assert agent._current_ota_context.obs_result == "computed-observation"

    @pytest.mark.asyncio
    async def test_before_action_reads_pending_decision(self):
        """before_action reads the pending decision off
        ``ota_context.think_result`` (payload-free hook); a no-RETURN body
        is a passthrough that does not replace the decision.

        before_action fires only on an ACT cycle, so the worker calls
        ``main_tool`` on its first reply, then finishes.
        """
        llm = MockLLM(tool_responses=[
            ([ToolCall(id="c0", name="main_tool", arguments={})], "step"),
            _finish_step(),
        ])
        seen_decisions: list = []

        async def main_tool() -> str:
            return "ok"

        class _OTA(OTAContext):
            pass

        _OTA.tool(main_tool)

        class Agent(AmphibiousAutoma[_OTA, Context]):
            async def before_action(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                seen_decisions.append(ota_context.think_result)
                if False:  # no RETURN → passthrough
                    yield

            plan_unit = think_unit(_PlanWorker())

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("plan_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx(), user_input=_GOAL)

        # before_action saw the worker's tool-calling decision...
        assert len(seen_decisions) >= 1
        tool_names = [c.tool for c in seen_decisions[0].tool_calls]
        assert tool_names == ["main_tool"]
        # ...and the decision survived the passthrough (the tool executed).
        executed = [r.tool_name for r in agent._current_ota_context.action_result.results]
        assert executed == ["main_tool"]


# ===========================================================================
# Worker-level hook generator form — short-circuits / delegates to the
# agent-level hook depending on whether it RETURNs a value.
# ===========================================================================


class TestWorkerHookGeneratorForm:
    """Worker-level hooks accept async-generator form, symmetric with the
    agent-level hooks. A worker generator that RETURNs short-circuits the
    agent-level fallback; one that exhausts without RETURN delegates."""

    @pytest.mark.asyncio
    async def test_worker_observation_generator_returns_short_circuits(self):
        """Worker observation generator yielding ActionCall + RETURN(value)
        sets ``obs_result`` and short-circuits the agent-level fallback."""
        llm = MockLLM(tool_responses=[_finish_step()])
        agent_fallback_ran = False
        call_count = 0

        async def snapshot_tool() -> str:
            nonlocal call_count
            call_count += 1
            return "worker-snap"

        class _OTA(OTAContext):
            pass

        _OTA.tool(snapshot_tool)

        class GenObservationWorker(CognitiveWorker):
            async def thinking(self, ota_context, context=None) -> Any:
                return await self._llm.aselect_tool(
                    messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
                    tools=[t.to_tool() for t in ota_context.tools],
                )

            async def observation(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                snap = yield ActionCall("snapshot_tool")
                yield RETURN(snap[0].result if snap else None)

        worker = GenObservationWorker()

        class Agent(AmphibiousAutoma[_OTA, Context]):
            recovery_unit = think_unit(worker, max_attempts=1)

            async def observation(self, ota_context, context=None):
                nonlocal agent_fallback_ran
                agent_fallback_ran = True
                yield RETURN("agent-fallback")

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("recovery_unit")

        agent = Agent()
        # Must NOT raise TypeError (can't await async_generator).
        await agent.arun(llm=llm, context=_ctx(), user_input=_GOAL)

        assert call_count == 1
        assert agent_fallback_ran is False
        assert agent._current_ota_context.obs_result == "worker-snap"

    @pytest.mark.asyncio
    async def test_worker_observation_generator_no_return_delegates(self):
        """Worker observation generator that exhausts without RETURN → None
        → treated as _DELEGATE → agent-level observation runs."""
        llm = MockLLM(tool_responses=[_finish_step()])
        agent_fallback_ran = False

        class GenObservationWorker(CognitiveWorker):
            async def thinking(self, ota_context, context=None) -> Any:
                return await self._llm.aselect_tool(
                    messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
                    tools=[t.to_tool() for t in ota_context.tools],
                )

            async def observation(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                if False:  # exhausts immediately, no RETURN
                    yield

        worker = GenObservationWorker()

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            recovery_unit = think_unit(worker, max_attempts=1)

            async def observation(self, ota_context, context=None):
                nonlocal agent_fallback_ran
                agent_fallback_ran = True
                yield RETURN("agent-fallback")

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("recovery_unit")

        agent = Agent()
        await agent.arun(llm=llm, context=_ctx(), user_input=_GOAL)

        assert agent_fallback_ran is True
        assert agent._current_ota_context.obs_result == "agent-fallback"


# ===========================================================================
# Two-generic context-class resolution (``_detect_context_classes``).
#
# AmphibiousAutoma takes two generic args — [OTAContextT, ContextT] —
# resolved at class-creation time and validated by their bounds:
# arg 1 must subclass OTAContext, arg 2 must subclass Context.
# ===========================================================================


class _MyOTA(OTAContext):
    """A small-loop OTA subclass (legal arg 1)."""


class _MyBig(Context):
    """A free-form big-loop context (legal arg 2)."""

    def summary(self) -> str:
        return "my-big"


class _NotAContext:
    """Neither an OTAContext nor a Context — illegal in either position."""


class TestTwoGenericResolution:

    @pytest.mark.parametrize(
        "arg1, arg2, inherit",
        [
            pytest.param(OTAContext, Context, False, id="builtin_classes"),
            pytest.param(_MyOTA, _MyBig, False, id="custom_subclasses"),
            pytest.param(_MyOTA, _MyBig, True, id="inherited_via_subclass"),
        ],
    )
    def test_two_generics_resolve(self, arg1, arg2, inherit):
        """``[OTAContextT, ContextT]`` populates both class slots — for the
        built-in classes, for custom subclasses (resolved by their bounds),
        and through a plain subclass of an already-parametrized agent (whose
        own ``__orig_bases__`` no longer names the generic)."""

        async def _on_workflow(self, ota_ctx, context=None):
            yield RETURN("ok")

        Base = _make_agent(
            "Base", AmphibiousAutoma[arg1, arg2], {"on_workflow": _on_workflow}
        )
        Resolved = type("Child", (Base,), {}) if inherit else Base

        assert Resolved._ota_context_class is arg1
        assert Resolved._context_class is arg2

    @pytest.mark.parametrize(
        "arg1, arg2, match",
        [
            pytest.param(_MyBig, Context, "OTAContext", id="arg1_not_ota"),
            pytest.param(_NotAContext, Context, "OTAContext", id="arg1_not_context"),
            pytest.param(OTAContext, _NotAContext, "Context", id="arg2_not_context"),
        ],
    )
    def test_wrong_generic_arg_raises(self, arg1, arg2, match):
        """Each bound is enforced at class creation: a bad arg 1 (not an
        OTAContext) or bad arg 2 (not a Context) raises ``TypeError``."""

        async def _on_workflow(self, ota_ctx, context=None):
            yield RETURN("never")

        with pytest.raises(TypeError, match=match):
            _make_agent(
                "_BadAgent",
                AmphibiousAutoma[arg1, arg2],
                {"on_workflow": _on_workflow},
            )

    @pytest.mark.asyncio
    async def test_arun_without_context_builds_default_big(self):
        """``arun(context=None)`` is legal: the framework builds the default
        big context (a bare ``Context``) and a fresh OTA, and a workflow runs
        end-to-end."""
        captured: list = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_ctx, context=None) -> AsyncGenerator[
                Union[ActionCall, RETURN], None
            ]:
                result = yield ActionCall("noop")  # unmatched tool -> empty results
                captured.append(result)
                yield RETURN("done")

        agent = Agent()
        answer = await agent.arun(llm=_NoopLLM(), user_input="g")

        assert answer == "done"
        # The ActionCall resolved against an empty toolset → no match → empty
        # tool-result list, no crash.
        assert captured == [[]]
        assert isinstance(agent._current_context, Context)
        assert agent._current_ota_context.user_input == "g"

    @pytest.mark.asyncio
    async def test_arun_rejects_wrong_big_context_type(self):
        """``context=`` must be an instance of the resolved big-context class."""

        class Agent(AmphibiousAutoma[OTAContext, _MyBig]):
            async def on_workflow(self, ota_ctx, context=None) -> AsyncGenerator[Any, None]:
                yield RETURN("ok")

        agent = Agent()
        with pytest.raises(ValueError, match="loop context"):
            await agent.arun(llm=_NoopLLM(), context=Context(), user_input="g")


# ===========================================================================
# RETURN primitive — writes final_answer from on_workflow / on_agent;
# unreachable code after RETURN; None when never RETURNed; runs after
# intermediate yields; dispatcher forwards RETURN.value to its caller.
# ===========================================================================


class TestReturnPrimitive:

    @pytest.mark.asyncio
    async def test_return_in_on_workflow_writes_final_answer(self):
        """yield RETURN(value) at the top of on_workflow → final_answer = value."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield RETURN("explicit-answer")

        agent = Agent()
        await agent.arun(context=_ctx(), user_input=_GOAL)

        assert agent.final_answer == "explicit-answer"

    @pytest.mark.asyncio
    async def test_return_in_on_agent_writes_final_answer(self):
        """yield RETURN(value) in on_agent → final_answer = value."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield RETURN("from-on-agent")

        agent = Agent()
        await agent.arun(llm=MockLLM(), context=_ctx(), user_input=_GOAL)

        assert agent.final_answer == "from-on-agent"

    @pytest.mark.asyncio
    async def test_yields_after_return_are_unreachable(self):
        """Code after yield RETURN never runs (generator is closed)."""
        executed_after = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield RETURN("done")
                executed_after.append("REACHED")  # must never run
                yield ActionCall("never_called")

        await Agent().arun(context=_ctx(), user_input=_GOAL)

        assert executed_after == []

    @pytest.mark.asyncio
    async def test_no_return_means_none_captured(self):
        """Generator that exhausts without RETURN and emits no steps →
        final_answer is None (no explicit RETURN, no auto-capture)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                if False:
                    yield  # empty generator

        agent = Agent()
        await agent.arun(context=_ctx(), user_input=_GOAL)

        assert agent.final_answer is None

    @pytest.mark.asyncio
    async def test_return_after_other_yields(self):
        """RETURN can come after other yields; intermediate yields still run."""
        log = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                log.append(("ask", prompt))
                return "ok"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="step-1")
                log.append(("got", resp))
                yield RETURN("final-result")

        agent = Agent()
        await agent.arun(context=_ctx(), user_input=_GOAL)

        assert log == [("ask", "step-1"), ("got", "ok")]
        assert agent.final_answer == "final-result"


# ===========================================================================
# observation None-preservation — when both worker- and agent-level
# observation produce no value, the prior round's observation_result is
# preserved (a None observation leaves the current round's slot empty,
# it never clobbers an earlier round). Covers the default-stub version too.
# ===========================================================================


class TestObservationNonePreservation:

    @pytest.mark.parametrize("override_agent_observation", [True, False])
    @pytest.mark.asyncio
    async def test_none_observation_preserves_prior_round(self, override_agent_observation):
        """Both worker- and agent-level observation None → the pre-seeded
        round keeps its value; the think cycle's own round is left empty.

        ``override_agent_observation=False`` exercises the base-class default
        observation stub (itself an unreachable-yield async-gen).
        """

        class _TravelOTA(OTAContext):
            pass

        class StubWorker(CognitiveWorker):
            async def thinking(self, ota_context, context=None) -> Any:
                # No tool calls → immediate finish.
                return await self._llm.aselect_tool(messages=[], tools=[])

            async def observation(self, ota_context, context=None):
                pass  # worker coroutine stub — None

        worker = StubWorker()

        body = {
            "main_step": think_unit(worker, max_attempts=1),
        }

        async def _on_agent(self, ota_context, context=None):
            # Pre-seed a round as if a prior after_action had refreshed
            # observation; the ThinkUnit's own cycle opens a new round.
            ota_context.open_record()
            ota_context.obs_result = "from-after-action"
            yield ThinkUnit("main_step")

        body["on_agent"] = _on_agent

        if override_agent_observation:
            async def _observation(self, ota_context, context=None):
                # async-gen stub — exhausts without RETURN → None
                if False:
                    yield

            body["observation"] = _observation

        Agent = _make_agent("Agent", AmphibiousAutoma[_TravelOTA, Context], body)

        agent = Agent()
        await agent.arun(llm=_DummyLLM(), user_input="None observation preserves prior")

        records = agent._current_ota_context.ota_record
        assert records[0].observation_result == "from-after-action"
        assert records[-1].observation_result is None
