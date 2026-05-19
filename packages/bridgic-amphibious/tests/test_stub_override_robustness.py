"""Robustness tests for stub-form template overrides.

Two layers of conventions:

* **AmphibiousAutoma** templates (``on_agent`` / ``on_workflow`` /
  ``observation`` / ``before_action`` / ``after_action``) must be async
  generators. Coroutine-form overrides (``async def`` with no ``yield``)
  are rejected at class-creation time by
  ``_validate_template_forms``. A stub with no real yields keeps the
  async-gen shape via ``if False: yield``.

* **CognitiveWorker** hooks (``observation`` / ``before_action`` /
  ``after_action``) keep their dual-form contract: a coroutine returning
  ``None`` / ``_DELEGATE`` is the canonical short form for "delegate to
  the agent-level hook". Async-gen worker hooks that exhaust without
  ``RETURN`` are treated identically.
"""

from typing import Any, AsyncGenerator, List, Union
from unittest.mock import MagicMock

import pytest

from bridgic.amphibious import (
    ActionCall,
    EnterAgent,
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    HumanCall,
    RunMode,
    StepToolCall,
    ToolArgument,
    ThinkUnit,
    think_unit,
)
from .tools import get_travel_planning_tools


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class _SeqLLM:
    """Returns a fixed sequence of structured-output responses."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self._idx = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): return MagicMock()
    async def astream(self, messages, **kwargs): return MagicMock()
    def chat(self, messages, **kwargs): return MagicMock()
    def stream(self, messages, **kwargs): return MagicMock()


class _TravelCtx(CognitiveContext):
    def __post_init__(self):
        for t in get_travel_planning_tools():
            self.tools.add(t)


def _search_decision(finish: bool = False) -> ThinkDecision:
    return ThinkDecision(
        step_content="Search for flights",
        output=[
            StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2024-06-01"),
                ],
            )
        ],
        finish=finish,
    )


# ---------------------------------------------------------------------------
# AmphibiousAutoma template form validation
# ---------------------------------------------------------------------------


class TestTemplateFormValidation:
    """``_validate_template_forms`` rejects coroutine-form overrides at
    class-creation time. Stubs must use ``if False: yield`` to keep the
    async-generator shape."""

    def test_pass_body_on_agent_rejected(self):
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _StubAgent(AmphibiousAutoma[CognitiveContext]):
                async def on_agent(self, ctx):
                    pass

    def test_pass_body_on_workflow_rejected(self):
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _StubAgent(AmphibiousAutoma[CognitiveContext]):
                async def on_workflow(self, ctx):
                    pass

    def test_pass_body_observation_rejected(self):
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _StubAgent(AmphibiousAutoma[CognitiveContext]):
                async def observation(self, ctx):
                    pass

    def test_pass_body_before_action_rejected(self):
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _StubAgent(AmphibiousAutoma[CognitiveContext]):
                async def before_action(self, decision_result, ctx):
                    pass

    def test_pass_body_after_action_rejected(self):
        with pytest.raises(TypeError, match="must be an ``async def`` function with at least one ``yield``"):
            class _StubAgent(AmphibiousAutoma[CognitiveContext]):
                async def after_action(self, step_result, ctx):
                    pass

    def test_unreachable_yield_stub_accepted(self):
        """``if False: yield`` is the canonical no-op async-gen stub."""

        class _NoOpAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(
                self, ctx
            ) -> AsyncGenerator[Union[ActionCall, HumanCall, EnterAgent], None]:
                if False:  # pragma: no cover
                    yield

        agent = _NoOpAgent()
        assert agent._has_workflow() is True


# ---------------------------------------------------------------------------
# Worker-level hook stubs (CognitiveWorker keeps dual-form: coroutine OK)
# ---------------------------------------------------------------------------


class TestWorkerHookStubs:
    """``CognitiveWorker`` hooks (``observation`` / ``before_action`` /
    ``after_action``) retain their dual-form contract: a coroutine
    returning ``None`` is identical to returning ``_DELEGATE``."""

    @pytest.mark.asyncio
    async def test_worker_before_action_pass_delegates_to_agent(self):
        """Worker ``before_action`` returning None must chain to agent-level hook."""

        agent_before_calls = []
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def before_action(self, decision_result, context):
                pass  # legacy "delegate" form

        worker = StubWorker()

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            main_step = think_unit(worker, max_attempts=1)

            async def before_action(self, decision_result, ctx) -> AsyncGenerator[Any, Any]:
                agent_before_calls.append(decision_result)
                if False:
                    yield

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Trigger before_action delegation")

        # Agent-level before_action ran exactly once with the original tool list.
        assert len(agent_before_calls) == 1
        decision = agent_before_calls[0]
        assert isinstance(decision, list) and len(decision) == 1
        tool_call, _spec = decision[0]
        assert tool_call.name == "search_flights"

        # The tool actually executed (proves decision_result wasn't replaced with None).
        last_step = agent._current_context.cognitive_history[-1]
        tool_names = [r.tool_name for r in last_step.result.results]
        assert tool_names == ["search_flights"]

    @pytest.mark.asyncio
    async def test_worker_after_action_pass_delegates_to_agent(self):
        """Worker ``after_action`` returning None must still chain to agent-level."""

        agent_after_calls = []
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def after_action(self, step_result, ctx):
                pass  # legacy "delegate" form

        worker = StubWorker()

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            main_step = think_unit(worker, max_attempts=1)

            async def after_action(self, step_result, ctx) -> AsyncGenerator[Any, Any]:
                agent_after_calls.append(step_result)
                if False:
                    yield

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Trigger after_action delegation")

        assert len(agent_after_calls) == 1, (
            "Agent-level after_action should run when worker returns None"
        )

    @pytest.mark.asyncio
    async def test_worker_observation_pass_falls_back_to_agent(self):
        """Worker-level ``observation`` returning None must delegate to agent-level."""

        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def observation(self, context):
                pass  # legacy "delegate" form

        worker = StubWorker()

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            main_step = think_unit(worker, max_attempts=1)

            async def observation(self, ctx) -> AsyncGenerator[Any, Any]:
                from bridgic.amphibious import RETURN
                yield RETURN("agent-level observation")

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Trigger observation delegation")

        assert agent._current_context.observation == "agent-level observation"

    @pytest.mark.asyncio
    async def test_observation_none_at_both_levels_preserves_prior_value(self):
        """When both worker- and agent-level observation produce no value,
        the prior ``ctx.observation`` is preserved — not overwritten.

        Makes the ``after_action``-driven refresh pattern work without
        forcing the user to write a passthrough ``observation`` override
        solely to defeat overwrites.
        """
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def observation(self, context):
                pass  # worker coroutine stub — None

        worker = StubWorker()

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            main_step = think_unit(worker, max_attempts=1)

            async def observation(self, ctx) -> AsyncGenerator[Any, Any]:
                # async-gen stub — exhausts without RETURN → None
                if False:
                    yield

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                # Pre-seed as if a prior after_action had refreshed observation.
                ctx.observation = "from-after-action"
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Both-None observation must preserve prior value")
        assert agent._current_context.observation == "from-after-action"

    @pytest.mark.asyncio
    async def test_default_observation_stub_preserves_prior_value(self):
        """Default (unoverridden) agent observation must not blank ctx.observation.

        The base-class default is itself an unreachable-yield async-gen
        stub; without the "None preserves prior" contract, every workflow
        yield would silently null out any state written by ``after_action``.
        """
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def observation(self, context):
                pass

        worker = StubWorker()

        # NOTE: no observation override on the agent — exercises the default
        # stub method baked into AmphibiousAutoma.
        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            main_step = think_unit(worker, max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                ctx.observation = "from-after-action"
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Default observation stub must preserve")
        assert agent._current_context.observation == "from-after-action"


# ---------------------------------------------------------------------------
# Agent-level action_custom_output stub returning None
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class _PlanOutput(BaseModel):
    note: str


class TestActionCustomOutputStub:
    """``action_custom_output`` is NOT one of the yield-driven templates
    (it's a plain coroutine returning the post-processed output), so
    coroutine-form stubs remain valid for it."""

    @pytest.mark.asyncio
    async def test_action_custom_output_pass_passes_through(self):
        """Agent-level ``action_custom_output`` returning None preserves the typed output."""

        expected = _PlanOutput(note="should be preserved")

        class _PlanWorker(CognitiveWorker):
            output_schema = _PlanOutput

            async def thinking(self):
                return "Produce a plan."

        worker = _PlanWorker()
        decision_model = worker._ThinkDecisionModel
        decision = decision_model(
            step_content="Planning complete",
            output=expected,
            finish=True,
        )

        llm = _SeqLLM([decision])
        worker.set_llm(llm)

        class StubAgent(AmphibiousAutoma[CognitiveContext]):
            main_step = think_unit(worker, max_attempts=1)

            async def action_custom_output(self, decision_result, ctx):
                pass  # stub — preserves the original typed output

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_step")

        agent = StubAgent()
        await agent.arun(llm=llm, goal="Trigger action_custom_output passthrough")

        # The typed _PlanOutput survives the None-returning hook and lands as result.
        last_step = agent._current_context.cognitive_history[-1]
        assert last_step.result is expected
