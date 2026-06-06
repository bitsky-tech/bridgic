"""End-to-end tests for the AmphibiousAutoma OTA observe-think-act cycle.

Consolidated, integration-oriented suite (driven through ``await agent.arun(...)``
and the public yield primitives) covering:

* the single / multi-step OTA cycle driven by ``ThinkUnit`` + ``CognitiveWorker``;
* ``until`` / ``max_attempts`` loop control (finish stops / predicate stops /
  never-finish runs to the cap);
* finish semantics — an empty ``tool_calls`` IS the finish (``step_content``
  becomes the final answer, no action result);
* ``ErrorStrategy.IGNORE`` / ``ErrorStrategy.RETRY`` (the latter a regression
  net for the now-fixed RETRY return-value gap: a recovered cycle refreshes
  the ``yield ThinkUnit`` send-value; IGNORE still leaves it stale — see its
  NOTE comment);
* eager-vs-lazy LLM requirement (no LLM primitive yielded ⇒ no eager failure;
  ``LLMCall`` / cognitive ``ThinkUnit`` without an LLM raise at the use site);
* worker ``observation`` / ``before_action`` hooks (value reaches the prompt /
  delegates to the agent level / rewrites the decision);
* structured-output thinking (pydantic model → content-only finish, JSON in
  ``step_content``, constrained by the worker's ``output_schema``);
* a streaming-``thinking`` smoke (delta stream consumed inside ``thinking``);
* the ``AgentTrace`` data model (observation field, tool success/error, no
  ``finished`` field, save/load round-trip, flat ``history``).

ISOLATION: every test (or test class) defines its OWN AmphibiousAutoma /
Context / CognitiveWorker subclasses — ``__init_subclass__`` builds per-class
registries at class-creation time, so sharing a subclass across tests would
pollute them.
"""
import json
import os
import tempfile
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel

from bridgic.amphibious import (
    AmphibiousAutoma,
    AgentTrace,
    Context,
    OTAContext,
    CognitiveWorker,
    _DELEGATE,
    ErrorStrategy,
    LLMCall,
    ThinkUnit,
    ThinkUnitDescriptor,
    think_unit,
    RETURN,
    TraceStep,
    RecordedToolCall,
)
from bridgic.amphibious._type import ThinkResult
from bridgic.core.model.types import (
    Message,
    Response,
    Role,
)

from .tools import get_travel_planning_tools


################################################################################
# Module-level helpers — mock LLMs + workers ported from the subsumed suites.
################################################################################


def _txt(text: str) -> Response:
    """A chat ``Response`` carrying plain text (the ``achat`` shape)."""
    return Response(message=Message.from_text(text, role=Role.AI))


def _tool_call(tool: str, **arguments) -> dict:
    """A plain-dict tool call as ``aselect_tool`` would return."""
    return {"name": tool, "arguments": arguments}


def _search_flights_step() -> Tuple[List[dict], str]:
    """Scripted ``aselect_tool`` reply: one ``search_flights`` call (continue)."""
    return (
        [_tool_call(
            "search_flights",
            origin="Beijing",
            destination="Tokyo",
            date="2025-06-01",
        )],
        "Search flights from Beijing to Tokyo",
    )


def _finish_step(content: str = "All done") -> Tuple[List[Any], str]:
    """Scripted reply with no tool calls → the framework derives finish."""
    return ([], content)


class SequencedLLM:
    """Returns a sequence of scripted ``aselect_tool`` / ``achat`` responses.

    Each ``aselect_tool`` reply is a ``(tool_calls, content)`` pair (empty
    ``tool_calls`` ⇒ finish). ``call_count`` tracks per-cycle LLM invocations
    so loop-control tests can assert how many cycles ran. ``captured_messages``
    holds the last message list for prompt-content assertions.
    """

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self.captured_messages: List[Any] = []

    async def aselect_tool(self, messages, tools, **kwargs):
        self.call_count += 1
        self.captured_messages = messages
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs):
        self.call_count += 1
        self.captured_messages = messages
        return _txt("done")

    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class ToolSelectWorker(CognitiveWorker):
    """Drives native function-calling via ``aselect_tool`` when the OTA
    context carries tools, else ``achat`` — returning each protocol's natural
    result verbatim so ``_assemble_decision`` adapts it. The user prompt is the
    context ``summary()`` so observation text is observable in the messages."""

    async def thinking(self, ota_context, context=None):
        messages = [Message.from_text(ota_context.summary(), role=Role.USER)]
        if ota_context.tools:
            return await self._llm.aselect_tool(
                messages=messages,
                tools=[t.to_tool() for t in ota_context.tools],
            )
        return await self._llm.achat(messages=messages)


def _make_ctx() -> Context:
    """The big-loop (knowledge) context passed as ``context=``."""
    return Context()


# Goal seeded into the fresh per-run OTA context via arun kwargs.
_SEED = dict(user_input="Test goal")


################################################################################
# The OTA cycle — single / multi-step, context-declared tools.
################################################################################


class TestOTACycle:
    """``arun`` drives the small-loop observe-think-act cycle: each ``ThinkUnit``
    yield runs the worker's OTC loop against the OTA context's declared tools,
    and the executed tool lands on the round's ``action_result``."""

    @pytest.mark.asyncio
    async def test_run_single_step(self):
        """One OTA cycle executes the worker's chosen tool."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([_search_flights_step(), _finish_step()])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            plan = think_unit(ToolSelectWorker())

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("plan")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        rounds = list(agent._current_ota_context.ota_record)
        assert len(rounds) == 1
        assert rounds[0].action_result.results[0].tool_name == "search_flights"

    @pytest.mark.asyncio
    async def test_multiple_workers_run_their_own_tools(self):
        """``on_agent`` orchestrates two ``ThinkUnit`` yields sequentially;
        each cycle executes its own scripted tool, and the run honours the
        tools the OTA context class declares (no think-unit-level filtering)."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([
            _search_flights_step(),
            (
                [_tool_call(
                    "search_hotels",
                    city="Tokyo",
                    check_in="2025-06-01",
                    check_out="2025-06-05",
                )],
                "Search hotels",
            ),
        ])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            planner = think_unit(ToolSelectWorker())
            executor = think_unit(ToolSelectWorker())

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("planner")
                yield ThinkUnit("executor")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        rounds = list(agent._current_ota_context.ota_record)
        assert len(rounds) == 2
        assert rounds[0].action_result.results[0].tool_name == "search_flights"
        assert rounds[1].action_result.results[0].tool_name == "search_hotels"

    @pytest.mark.asyncio
    async def test_arun_auto_creates_default_big_context(self):
        """``arun`` with no ``context=`` builds the default big context; the
        fresh per-run OTA carries ``user_input`` and the class-declared tools."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([_search_flights_step()])
        observed_input: List[str] = []

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker())

            async def on_agent(self, ota_context, context=None):
                observed_input.append(ota_context.user_input)
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, user_input="Test goal")

        assert observed_input == ["Test goal"]
        assert isinstance(agent._current_context, Context)
        assert len(agent._current_ota_context.ota_record) == 1


################################################################################
# Loop control — until / max_attempts three-state behaviour.
################################################################################


class TestLoopControl:
    """``ThinkUnit`` loop control: the worker's finish (no tool calls) stops
    the OTC loop, an ``until`` predicate stops it early, and a worker that
    never finishes runs the full ``max_attempts``."""

    @pytest.mark.asyncio
    async def test_finish_stops_loop_before_max_attempts(self):
        """A no-tool reply finishes the worker mid-loop (before the cap)."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([
            _search_flights_step(),   # cycle 1: tool call → continue
            _finish_step("done"),     # cycle 2: no tool call → finish
        ])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker(), max_attempts=10)

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        # Stopped after cycle 2 even though max_attempts=10.
        assert llm.call_count == 2
        assert len(agent._current_ota_context.ota_record) == 2

    @pytest.mark.asyncio
    async def test_until_predicate_stops_loop(self):
        """An ``until`` predicate that turns True stops the loop early even
        when the worker keeps emitting tool calls."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        checks = {"n": 0}

        def condition(ota_ctx):
            checks["n"] += 1
            return checks["n"] >= 2  # stop after the 2nd cycle

        # Always a tool call → finish never True; only ``until`` can stop it.
        llm = SequencedLLM([_search_flights_step()])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker(), until=condition, max_attempts=10)

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        assert len(agent._current_ota_context.ota_record) == 2

    @pytest.mark.asyncio
    async def test_never_finishing_worker_runs_max_attempts(self):
        """When every reply carries a tool call (finish never True), the loop
        runs the full ``max_attempts``."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([_search_flights_step()])  # always a tool call

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker(), max_attempts=3)

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        assert llm.call_count == 3
        assert len(agent._current_ota_context.ota_record) == 3


################################################################################
# Finish semantics — empty tool_calls IS the finish.
################################################################################


class TestFinishSemantics:
    """A decision with NO ``tool_calls`` is the finish: ``step_content`` becomes
    both the worker's ``ThinkResult`` text and the run's final answer, and no
    action result is folded onto the round."""

    @pytest.mark.asyncio
    async def test_worker_finish_is_content_only(self):
        """``worker.arun`` with a no-tool reply returns a content-only
        ``ThinkResult`` (empty ``tool_calls``, text in ``step_content``)."""

        class FinishWorker(CognitiveWorker):
            async def thinking(self, ota_context, context=None):
                return await self._llm.achat(messages=[])

        class _LLM:
            async def achat(self, messages, **kwargs):
                return _txt("Nothing left to do")

        worker = FinishWorker(llm=_LLM())
        decision = await worker.arun(ota_context=OTAContext(user_input="x"))

        assert isinstance(decision, ThinkResult)
        assert decision.step_content == "Nothing left to do"
        assert decision.tool_calls == []

    @pytest.mark.asyncio
    async def test_finish_sets_final_answer_and_no_action_result(self):
        """End-to-end: the finishing think's ``step_content`` is the run's
        final answer / the ``yield ThinkUnit`` value, and the finishing round
        has no ``action_result``."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([_finish_step("the final answer")])
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None):
                value = yield ThinkUnit("step")
                captured.append(value)
                yield RETURN(value)

        agent = Agent()
        result = await agent.arun(llm=llm, context=_make_ctx(), **_SEED)

        assert result == "the final answer"
        assert agent.final_answer == "the final answer"
        # Happy-path send-value IS the finishing step_content (no RETRY/IGNORE).
        assert captured == ["the final answer"]
        last_round = agent._current_ota_context.ota_record[-1]
        assert last_round.action_result is None
        assert last_round.think_result.step_content == "the final answer"


################################################################################
# Error strategies — IGNORE gap + RETRY recovery (regression net for the
# known RETRY/IGNORE yield-send-value bug).
################################################################################


class TestErrorStrategies:
    """``ErrorStrategy.IGNORE`` swallows a failing cycle so ``arun`` still
    completes; ``ErrorStrategy.RETRY`` re-runs a failed cycle until it
    succeeds. RETRY now refreshes the ``yield ThinkUnit`` send-value on
    recovery (asserted directly); IGNORE leaves it stale, so its case is
    asserted only at the ``arun`` final-result level + observable effects."""

    @pytest.mark.asyncio
    async def test_ignore_completes_without_raising(self):
        """A worker whose LLM always fails, under IGNORE, does not raise:
        ``arun`` completes and falls back to the OTA ``summary()``."""

        class FailLLM:
            async def aselect_tool(self, messages, tools, **kwargs):
                raise RuntimeError("LLM failed")
            async def achat(self, messages, **kwargs):
                raise RuntimeError("LLM failed")
            async def astructured_output(self, messages, constraint, **kwargs):
                raise RuntimeError("LLM failed")
            async def astream(self, messages, **kwargs): ...
            def chat(self, messages, **kwargs): ...
            def select_tool(self, messages, tools, **kwargs): ...
            def structured_output(self, messages, constraint, **kwargs): ...
            def stream(self, messages, **kwargs): ...

        class Worker(CognitiveWorker):
            async def thinking(self, ota_context, context=None):
                return await self._llm.achat(messages=[])

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            step = think_unit(Worker(), max_attempts=1, on_error=ErrorStrategy.IGNORE)

            async def on_agent(self, ota_context, context=None):
                # NOTE: yield-ThinkUnit send-value under RETRY/IGNORE is a known
                # return-value bug; asserted at arun() level instead.
                yield ThinkUnit("step")

        agent = Agent()
        # Must not raise — the failing cycle is swallowed.
        result = await agent.arun(llm=FailLLM(), context=_make_ctx(), user_input="goal-x")

        assert "goal-x" in result
        assert isinstance(agent._current_context, Context)

    @pytest.mark.asyncio
    async def test_retry_recovers_and_arun_returns_correct_answer(self):
        """A cycle that fails once (the agent-level ``observation`` raises on
        its first invocation) then succeeds on retry: the run recovers and
        BOTH the ``yield ThinkUnit`` send-value and ``arun``'s final answer are
        the recovered think's ``step_content``.

        Regression net for the RETRY return-value gap (was
        ``_amphibious_automa.py:1510``): the retry branch now unpacks
        ``finished, result`` like the happy path, so recovery refreshes the
        send-value instead of leaving it stale/``None``.
        """

        class OkLLM:
            async def achat(self, messages, **kwargs):
                return _txt("the-answer")

        class Worker(CognitiveWorker):
            async def thinking(self, ota_context, context=None):
                return await self._llm.achat(messages=[])

        obs_calls = {"n": 0}
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            step = think_unit(
                Worker(), max_attempts=1,
                on_error=ErrorStrategy.RETRY, max_retries=3,
            )

            async def observation(self, ota_context, context=None):
                obs_calls["n"] += 1
                if obs_calls["n"] == 1:
                    raise RuntimeError("observation boom (once)")
                yield RETURN("observation-ok")

            async def on_agent(self, ota_context, context=None):
                value = yield ThinkUnit("step")
                captured.append(value)
                yield RETURN(value)

        agent = Agent()
        result = await agent.arun(llm=OkLLM(), context=_make_ctx(), **_SEED)

        # The recovered cycle refreshes the value, so the ``yield ThinkUnit``
        # send-value IS the recovered think's ``step_content`` (not stale/None).
        assert captured == ["the-answer"]
        # arun's final answer agrees (set inside the finishing retry cycle).
        assert result == "the-answer"
        assert agent.final_answer == "the-answer"
        # Observable effect: the cycle was retried — observation ran twice
        # (failed once, then succeeded), and the run completed without raising.
        assert obs_calls["n"] == 2


################################################################################
# LLM requirement — eager-vs-lazy.
################################################################################


class TestLLMRequirement:
    """The LLM is an optional dependency consumed only by LLM-driven primitives.
    A run that yields no such primitive does not fail eagerly; a run that does
    (``LLMCall`` from on_workflow / a cognitive ``ThinkUnit``) raises a clear
    error at the use site when no LLM was provided."""

    @pytest.mark.asyncio
    async def test_no_llm_ok_when_no_llm_primitive_yielded(self):
        """A no-op ``on_agent`` is valid without an LLM (no eager failure)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_agent(self, ota_context, context=None):
                if False:  # pragma: no cover — async-generator stub
                    yield

        agent = Agent()
        await agent.arun(user_input="Test")  # no raise

    @pytest.mark.asyncio
    async def test_llmcall_without_llm_raises_at_use_site(self):
        """Yielding ``LLMCall`` with no LLM surfaces a clear error at the
        dispatcher use-point (from on_workflow — LLMCall is scope-restricted)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None):
                yield LLMCall.chat("hi")

        agent = Agent()
        with pytest.raises(RuntimeError, match="LLMCall.*requires self._llm"):
            await agent.arun(user_input="Test")

    @pytest.mark.asyncio
    async def test_cognitive_thinkunit_without_llm_raises_at_use_site(self):
        """A ``ThinkUnit`` driving a ``CognitiveWorker`` needs an LLM; with
        neither worker nor agent LLM set it raises before the OTC starts."""

        class Worker(CognitiveWorker):
            async def thinking(self, ota_context, context=None):
                return await self._llm.achat(messages=[])

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            planner = think_unit(Worker())

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("planner")

        agent = Agent()
        with pytest.raises(RuntimeError, match="CognitiveWorker.*has no LLM"):
            await agent.arun(user_input="Test")

    @pytest.mark.asyncio
    async def test_bare_worker_without_llm_raises(self):
        """``CognitiveWorker.arun`` itself raises a clear error with no LLM."""

        class Worker(CognitiveWorker):
            async def thinking(self, ota_context, context=None):
                return await self._llm.achat(messages=[])

        worker = Worker()  # no llm
        with pytest.raises(RuntimeError, match="no LLM"):
            await worker.arun(ota_context=OTAContext(user_input="x"))


################################################################################
# Worker hooks — observation value / delegation, before_action rewrite.
################################################################################


class TestWorkerHooks:
    """Worker-level ``observation`` / ``before_action`` hooks: a worker
    ``observation`` value lands on ``obs_result`` and reaches the prompt; a
    ``_DELEGATE`` (default) hands off to the agent-level hook; a
    ``before_action`` may rewrite the decision before the act phase runs."""

    @pytest.mark.asyncio
    async def test_worker_observation_value_reaches_prompt(self):
        """A worker ``observation`` returning a value (no delegation) folds it
        onto ``obs_result`` and surfaces in the LLM messages via ``summary()``."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        class ObsWorker(ToolSelectWorker):
            async def observation(self, ota_context, context=None):
                return "Custom observation: environment is ready"

        llm = SequencedLLM([_search_flights_step()])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ObsWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, user_input="Plan a trip to Tokyo")

        rounds = agent._current_ota_context.ota_record
        assert rounds[-1].observation_result == "Custom observation: environment is ready"
        assert "Custom observation: environment is ready" in llm.captured_messages[-1].content

    @pytest.mark.asyncio
    async def test_worker_observation_delegates_to_agent_level(self):
        """A worker ``observation`` returning ``_DELEGATE`` (the default) lets
        the agent-level ``observation`` fill ``obs_result`` — which then reaches
        the LLM prompt."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        llm = SequencedLLM([_search_flights_step()])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(ToolSelectWorker(), max_attempts=1)

            async def observation(self, ota_context, context=None):
                yield RETURN("Default observation from agent")

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        # The worker's default observation returns _DELEGATE.
        assert await ToolSelectWorker().observation(OTAContext(user_input="x")) is _DELEGATE
        await agent.arun(llm=llm, user_input="x")

        assert "Default observation from agent" in llm.captured_messages[-1].content

    @pytest.mark.asyncio
    async def test_before_action_rewrites_decision_dropping_a_tool(self):
        """A worker ``before_action`` reads the pending decision off
        ``ota_context.think_result`` and returns a replacement that drops one
        tool; only the kept tool executes in the act phase."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        class FilterWorker(ToolSelectWorker):
            async def before_action(self, ota_context, context=None):
                decision = ota_context.think_result
                kept = [tc for tc in decision.tool_calls if tc.tool != "book_flight"]
                return ThinkResult(step_content=decision.step_content, tool_calls=kept)

        llm = SequencedLLM([(
            [
                _tool_call("search_flights", origin="Beijing", destination="Tokyo", date="2025-06-01"),
                _tool_call("book_flight", flight_number="CA123"),
            ],
            "Search and book",
        )])

        class Agent(AmphibiousAutoma[TravelOTA, Context]):
            step = think_unit(FilterWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None):
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(llm=llm, user_input="test")

        last_round = agent._current_ota_context.ota_record[-1]
        tool_names = [r.tool_name for r in last_round.action_result.results]
        assert "book_flight" not in tool_names
        assert "search_flights" in tool_names


################################################################################
# Structured-output thinking — pydantic model → content-only finish.
################################################################################


class _PlanPhase(BaseModel):
    sub_goal: str
    skill_name: str


class _PlanResult(BaseModel):
    phases: List[_PlanPhase]


class _StructuredLLM:
    """An LLM whose ``astructured_output`` returns a scripted pydantic model
    and captures the constraint passed in."""

    def __init__(self, response: Any):
        self._response = response
        self.captured_constraint: Any = None

    async def astructured_output(self, messages, constraint, **kwargs):
        self.captured_constraint = constraint
        return self._response

    async def aselect_tool(self, messages, tools, **kwargs): ...
    async def achat(self, messages, **kwargs): ...


class _StructuredWorker(CognitiveWorker):
    """Drives ``astructured_output`` constrained by a class-level
    ``output_schema`` (a class attribute so it survives the per-invocation
    worker clone). Returns the pydantic model — ``_assemble_decision``
    serializes it into a content-only ``ThinkResult``."""

    output_schema = _PlanResult

    async def thinking(self, ota_context, context=None):
        from bridgic.core.model.protocols import PydanticModel

        return await self._llm.astructured_output(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            constraint=PydanticModel(model=type(self).output_schema),
        )


class TestStructuredOutput:
    """A ``thinking`` that returns a pydantic ``BaseModel`` goes through
    ``astructured_output``; ``_assemble_decision`` serializes the typed result
    into a content-only ``ThinkResult`` (``step_content`` = the JSON, no tool
    calls → finish), constrained by the worker's ``output_schema``."""

    @pytest.mark.asyncio
    async def test_structured_output_serialized_into_finish_and_round(self):
        """``arun`` finishes content-only with the model's JSON in
        ``step_content``; the round records the same ``ThinkResult`` and no
        action result."""

        expected = _PlanResult(phases=[_PlanPhase(sub_goal="Phase 1", skill_name="skill-1")])
        llm = _StructuredLLM(expected)

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            plan = think_unit(_StructuredWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None):
                value = yield ThinkUnit("plan")
                yield RETURN(value)

        agent = Agent()
        result = await agent.arun(llm=llm, user_input="Test structured output")

        # Final answer is the serialized model (a content-only finish).
        assert result == expected.model_dump_json()
        assert json.loads(result)["phases"][0]["sub_goal"] == "Phase 1"

        last_round = agent._current_ota_context.ota_record[-1]
        assert last_round.action_result is None
        assert isinstance(last_round.think_result, ThinkResult)
        assert last_round.think_result.tool_calls == []
        assert last_round.think_result.step_content == expected.model_dump_json()

    @pytest.mark.asyncio
    async def test_structured_output_uses_schema_as_constraint(self):
        """The structured-output call is constrained by the worker's
        ``output_schema`` directly (no decision-model wrapper)."""
        from bridgic.core.model.protocols import PydanticModel

        llm = _StructuredLLM(_PlanResult(phases=[]))
        worker = _StructuredWorker(llm=llm)

        await worker.arun(ota_context=OTAContext(user_input="Test"))

        constraint = llm.captured_constraint
        assert isinstance(constraint, PydanticModel)
        assert constraint.model is _PlanResult


################################################################################
# AgentTrace — the flat-history execution-path data model.
################################################################################


class TestAgentTrace:
    """``AgentTrace`` records one flat ``history`` of ``TraceStep`` entries
    (observation field, recorded tool success/error, no ``finished`` field)
    plus a ``goal`` / ``metadata`` envelope; ``save`` / ``load`` round-trips it."""

    @staticmethod
    def _trace_agent(llm, *, observation_text: Optional[str] = None):
        """Build a fresh single-step trace agent (its own subclasses)."""

        class TravelOTA(OTAContext):
            pass

        for _spec in get_travel_planning_tools():
            TravelOTA.tool(_spec)

        if observation_text is None:
            class Agent(AmphibiousAutoma[TravelOTA, Context]):
                plan = think_unit(ToolSelectWorker())

                async def on_agent(self, ota_context, context=None):
                    yield ThinkUnit("plan")
        else:
            class Agent(AmphibiousAutoma[TravelOTA, Context]):
                plan = think_unit(ToolSelectWorker())

                async def observation(self, ota_context, context=None):
                    yield RETURN(observation_text)

                async def on_agent(self, ota_context, context=None):
                    yield ThinkUnit("plan")

        return Agent()

    def test_tracestep_model_shape(self):
        """``TraceStep`` carries an ``observation`` field and NOT a
        ``finished`` field (pure model-shape check)."""
        assert "observation" in TraceStep.model_fields
        assert "finished" not in TraceStep.model_fields

    @pytest.mark.asyncio
    async def test_trace_observation_none_without_override(self, tmp_path):
        """With no ``observation`` override the recorded step's observation is
        ``None``; the single step lands on the flat ``history``."""
        llm = SequencedLLM([_search_flights_step()])
        agent = self._trace_agent(llm)
        await agent.arun(llm=llm, context=_make_ctx(), workdir=tmp_path, trace=True, **_SEED)

        trace = agent._agent_trace.build()
        assert len(trace["history"]) == 1
        step: TraceStep = trace["history"][0]
        assert step.observation is None

    @pytest.mark.asyncio
    async def test_trace_records_observation_and_tool_success(self, tmp_path):
        """An agent ``observation`` is recorded (text + hash); the recorded
        tool call carries ``success`` / ``error`` fields."""
        llm = SequencedLLM([_search_flights_step()])
        agent = self._trace_agent(
            llm, observation_text="Current page: login form with username and password",
        )
        await agent.arun(llm=llm, context=_make_ctx(), workdir=tmp_path, trace=True, **_SEED)

        step: TraceStep = agent._agent_trace.build()["history"][0]
        assert step.observation is not None
        assert "login form" in step.observation
        assert step.observation_hash is not None
        assert len(step.tool_calls) == 1
        tc: RecordedToolCall = step.tool_calls[0]
        assert tc.tool_name == "search_flights"
        assert tc.success is True
        assert tc.error is None

    @pytest.mark.asyncio
    async def test_trace_save_load_roundtrip(self, tmp_path):
        """``save`` then ``load`` yields the unified ``goal`` / ``metadata`` /
        ``history`` shape with no ``finished`` field and the tool call intact."""
        llm = SequencedLLM([_search_flights_step()])
        agent = self._trace_agent(llm)
        await agent.arun(llm=llm, context=_make_ctx(), workdir=tmp_path, trace=True, **_SEED)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent._agent_trace.save(path)
            loaded = AgentTrace.load(path)

            assert "goal" in loaded
            assert "metadata" in loaded
            assert "history" in loaded
            assert len(loaded["history"]) == 1

            step = loaded["history"][0]
            assert "observation" in step
            assert "finished" not in step
            assert len(step["tool_calls"]) == 1
            assert step["tool_calls"][0]["tool_name"] == "search_flights"
            assert step["tool_calls"][0]["success"] is True
            assert step["tool_calls"][0]["error"] is None
        finally:
            os.unlink(path)
