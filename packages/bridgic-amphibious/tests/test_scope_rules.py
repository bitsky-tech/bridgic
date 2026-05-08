"""Tests for the yield-type ↔ scope rules.

Rules:

==============  =============  ==========  =======
primitive       on_workflow    on_agent    hooks
==============  =============  ==========  =======
ActionCall      OK             OK          OK
HumanCall       OK             OK          OK
LLMCall         OK             OK          OK
AgentCall       OK             RAISE       RAISE
ThinkCall       RAISE          OK          RAISE
==============  =============  ==========  =======

``hooks`` = observation / before_action / after_action.
"""

from typing import Any, AsyncGenerator, List, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    AgentCall,
    LLMCall,
    ThinkCall,
    Step,
    StepToolCall,
    ToolArgument,
    think_unit,
)


# ---------------------------------------------------------------------------
# Mock LLM stub
# ---------------------------------------------------------------------------


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class _DummyLLM:
    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs):
        return ThinkDecision(step_content="done", output=[], finish=True)
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="scope-test")


# ---------------------------------------------------------------------------
# AgentCall scope rules
# ---------------------------------------------------------------------------


class TestAgentCallScope:

    @pytest.mark.asyncio
    async def test_agent_call_in_on_workflow_works(self):
        """AgentCall is the canonical workflow→agent transition primitive."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                yield AgentCall(goal="sub")

        # Should not raise.
        await Agent(llm=_DummyLLM()).arun(context=_ctx())

    @pytest.mark.asyncio
    async def test_agent_call_in_on_agent_raises(self):
        """AgentCall yielded from on_agent → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield AgentCall(goal="re-enter")

        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await Agent(llm=_DummyLLM()).arun(context=_ctx())

    @pytest.mark.asyncio
    async def test_agent_call_in_hook_scope_raises(self):
        """AgentCall yielded with scope='hook' (any of observation /
        before_action / after_action) → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await agent._dispatch_call(
                AgentCall(goal="x"),
                agent._current_context,
                scope="hook",
            )


# ---------------------------------------------------------------------------
# ThinkCall scope rules (in addition to those in test_think_call.py)
# ---------------------------------------------------------------------------


class TestThinkCallScope:

    @pytest.mark.asyncio
    async def test_think_call_in_observation_raises(self):
        """ThinkCall yielded from observation hook → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            think_unit_a = think_unit(CognitiveWorker.inline("a"), max_attempts=1)

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await agent._dispatch_call(
                ThinkCall("think_unit_a"),
                agent._current_context,
                scope="hook",
            )

    @pytest.mark.asyncio
    async def test_think_call_in_workflow_via_dispatch(self):
        """ThinkCall yielded with scope='workflow' → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            think_unit_a = think_unit(CognitiveWorker.inline("a"), max_attempts=1)

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await agent._dispatch_call(
                ThinkCall("think_unit_a"),
                agent._current_context,
                scope="workflow",
            )


# ---------------------------------------------------------------------------
# Universal primitives — ActionCall / HumanCall / LLMCall in any scope
# ---------------------------------------------------------------------------


class TestUniversalPrimitivesByScope:

    @pytest.mark.parametrize("scope", ["workflow", "agent", "hook"])
    @pytest.mark.asyncio
    async def test_human_call_works_in_any_scope(self, scope):
        """HumanCall passes scope validation in workflow / agent / hook."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def feed(self, prompt: str) -> str:
                return "ok"

        from bridgic.amphibious import human_channel
        Agent.feed = human_channel(Agent.feed)
        Agent._build_human_channel_registry()

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        result = await agent._dispatch_call(
            HumanCall(prompt="confirm"),
            agent._current_context,
            scope=scope,
        )
        assert result == "ok"
