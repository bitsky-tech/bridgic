"""Robustness tests for AI-generated stub overrides of template methods.

When AI coding tools generate skeleton subclasses, they often emit::

    async def before_action(self, decision_result, ctx): pass
    async def after_action(self, step_result, ctx): pass
    async def on_workflow(self, ctx): pass
    async def observation(self, context): pass
    async def action_custom_output(self, decision_result, ctx): pass

These should NOT crash the framework. The contracts are:

- Worker-level ``before_action`` / ``after_action`` / ``observation`` returning
  ``None`` is treated identically to ``_DELEGATE`` (delegate to the agent-level
  hook).
- Agent-level ``before_action`` and ``action_custom_output`` returning ``None``
  are treated as passthrough (the original input is preserved).
- ``on_workflow`` written as a plain ``async def ... : pass`` (a coroutine, not
  an async generator) is treated as "not overridden" — ``_has_workflow()``
  returns False and ``RunMode.AUTO`` falls back to the agent path.
"""

from typing import Any, AsyncGenerator, List, Union
from unittest.mock import MagicMock

import pytest

from bridgic.amphibious import (
    ActionCall,
    AgentCall,
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    HumanCall,
    RunMode,
    StepToolCall,
    ToolArgument,
)
from .tools import get_travel_planning_tools


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

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
# on_workflow stub (`pass` body) — coroutine, not async generator
# ---------------------------------------------------------------------------

class TestOnWorkflowStub:

    def test_pass_body_makes_has_workflow_return_false(self):
        """`async def on_workflow(...): pass` is a coroutine, not async-gen → not a real override."""

        class StubAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx):  # noqa: D401 — stub
                pass

            async def on_agent(self, ctx):
                pass

        agent = StubAgent(llm=_SeqLLM([]))
        assert agent._has_workflow() is False
        assert agent._has_agent() is True

    def test_real_async_generator_still_detected(self):
        """A real `yield`-bearing override is still recognized as workflow."""

        class WorkflowAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(
                self, ctx
            ) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                if False:  # pragma: no cover
                    yield

        agent = WorkflowAgent(llm=_SeqLLM([]))
        assert agent._has_workflow() is True

    def test_auto_mode_resolves_to_agent_when_workflow_is_stub(self):
        """RunMode.AUTO should pick AGENT when on_workflow is a `pass` stub but on_agent is real."""

        class StubAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx):
                pass

            async def on_agent(self, ctx):
                pass

        agent = StubAgent(llm=_SeqLLM([]))
        assert agent._resolve_mode(RunMode.AUTO) is RunMode.AGENT

    def test_only_stub_workflow_no_agent_raises(self):
        """If both hooks are absent (workflow is a stub, agent not overridden), surface the existing RuntimeError."""

        class OnlyStubAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx):
                pass

        agent = OnlyStubAgent(llm=_SeqLLM([]))
        with pytest.raises(RuntimeError, match="must override on_agent\\(\\) or on_workflow\\(\\)"):
            agent._resolve_mode(RunMode.AUTO)

    @pytest.mark.asyncio
    async def test_arun_does_not_crash_on_stub_workflow(self):
        """End-to-end: `arun()` with a `pass`-body on_workflow runs via on_agent without errors."""

        on_agent_calls = []
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorkflowAgent(AmphibiousAutoma[_TravelCtx]):
            async def on_workflow(self, ctx):  # AI-generated stub
                pass

            async def on_agent(self, ctx):
                on_agent_calls.append(ctx.goal)
                worker = CognitiveWorker.inline("Plan step.", llm=self.llm)
                await self._run(worker, max_attempts=1)

        agent = StubWorkflowAgent(llm=llm)
        await agent.arun(goal="Trigger fallback to on_agent")

        assert on_agent_calls == ["Trigger fallback to on_agent"]


# ---------------------------------------------------------------------------
# Worker-level before_action / after_action stubs returning None
# ---------------------------------------------------------------------------

class TestWorkerHookStubs:

    @pytest.mark.asyncio
    async def test_worker_before_action_pass_delegates_to_agent(self):
        """Worker `before_action` returning None must chain to agent-level hook (not drop decision)."""

        agent_before_calls = []
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def before_action(self, decision_result, context):  # noqa: D401 — stub
                pass

        worker = StubWorker(llm=llm)

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            async def before_action(self, decision_result, ctx):
                agent_before_calls.append(decision_result)
                return decision_result

            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=1)

        agent = StubAgent(llm=llm)
        await agent.arun(goal="Trigger before_action delegation")

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
        """Worker `after_action` returning None must still chain to agent-level after_action."""

        agent_after_calls = []
        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def after_action(self, step_result, ctx):  # noqa: D401 — stub
                pass

        worker = StubWorker(llm=llm)

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            async def after_action(self, step_result, ctx):
                agent_after_calls.append(step_result)

            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=1)

        agent = StubAgent(llm=llm)
        await agent.arun(goal="Trigger after_action delegation")

        assert len(agent_after_calls) == 1, (
            "Agent-level after_action should run when worker returns None"
        )


# ---------------------------------------------------------------------------
# Agent-level before_action stub returning None
# ---------------------------------------------------------------------------

class TestAgentBeforeActionStub:

    @pytest.mark.asyncio
    async def test_agent_before_action_pass_passes_through_decision(self):
        """Agent-level `before_action` returning None preserves the original decision_result."""

        llm = _SeqLLM([_search_decision(finish=True)])
        worker = CognitiveWorker.inline("Plan step.", llm=llm)

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            async def before_action(self, decision_result, ctx):  # noqa: D401 — stub
                pass

            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=1)

        agent = StubAgent(llm=llm)
        await agent.arun(goal="Trigger agent before_action passthrough")

        # The original tool call survives the None-returning hook and executes.
        last_step = agent._current_context.cognitive_history[-1]
        tool_names = [r.tool_name for r in last_step.result.results]
        assert tool_names == ["search_flights"]


# ---------------------------------------------------------------------------
# Worker-level observation stub returning None
# ---------------------------------------------------------------------------

class TestWorkerObservationStub:

    @pytest.mark.asyncio
    async def test_worker_observation_pass_falls_back_to_agent(self):
        """Worker-level `observation` returning None must delegate to agent-level."""

        llm = _SeqLLM([_search_decision(finish=True)])

        class StubWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def observation(self, context):  # noqa: D401 — stub
                pass

        worker = StubWorker(llm=llm)

        class StubAgent(AmphibiousAutoma[_TravelCtx]):
            async def observation(self, ctx):
                return "agent-level observation"

            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=1)

        agent = StubAgent(llm=llm)
        await agent.arun(goal="Trigger observation delegation")

        # Agent-level observation wins instead of the worker silently writing
        # ``None`` into ctx.observation.
        assert agent._current_context.observation == "agent-level observation"


# ---------------------------------------------------------------------------
# Agent-level action_custom_output stub returning None
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class _PlanOutput(BaseModel):
    note: str


class TestActionCustomOutputStub:

    @pytest.mark.asyncio
    async def test_action_custom_output_pass_passes_through(self):
        """Agent-level `action_custom_output` returning None preserves the typed output."""

        expected = _PlanOutput(note="should be preserved")

        class _PlanWorker(CognitiveWorker):
            output_schema = _PlanOutput

            async def thinking(self):
                return "Produce a plan."

        worker = _PlanWorker(llm=_SeqLLM([]))
        decision_model = worker._ThinkDecisionModel
        decision = decision_model(
            step_content="Planning complete",
            output=expected,
            finish=True,
        )

        llm = _SeqLLM([decision])
        worker.set_llm(llm)

        class StubAgent(AmphibiousAutoma[CognitiveContext]):
            async def action_custom_output(self, decision_result, ctx):  # noqa: D401 — stub
                pass

            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=1)

        agent = StubAgent(llm=llm)
        await agent.arun(goal="Trigger action_custom_output passthrough")

        # The typed output survives a None-returning override.
        last_step = agent._current_context.cognitive_history[-1]
        assert last_step.result is expected
        assert isinstance(last_step.result, _PlanOutput)
        assert last_step.result.note == "should be preserved"
