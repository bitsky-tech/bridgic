"""Tests for the yield-type ↔ scope rules.

Rules:

==============  =============  ==========  =======
primitive       on_workflow    on_agent    hooks
==============  =============  ==========  =======
ActionCall      OK             RAISE       OK
HumanCall       OK             RAISE       OK
LLMCall         OK             RAISE       OK
EnterAgent      OK             RAISE       RAISE
ThinkUnit       RAISE          OK          RAISE
==============  =============  ==========  =======

``hooks`` = observation / before_action / after_action.

on_agent body is reserved for orchestrating ThinkUnit cycles —
deterministic tool / HITL / direct-LLM calls belong in on_workflow
or in worker hooks.
"""

from typing import Any, AsyncGenerator, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    ThinkUnit,
    think_unit,
    human_channel,
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
# EnterAgent scope rules — workflow only
# ---------------------------------------------------------------------------


class TestEnterAgentScope:

    @pytest.mark.asyncio
    async def test_enter_agent_in_on_workflow_works(self):
        """EnterAgent is the canonical workflow→agent transition primitive."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="sub")

        # Should not raise.
        await Agent(llm=_DummyLLM()).arun(context=_ctx())

    @pytest.mark.asyncio
    async def test_enter_agent_in_on_agent_raises(self):
        """EnterAgent yielded from on_agent → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield EnterAgent(goal="re-enter")

        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await Agent(llm=_DummyLLM()).arun(context=_ctx())

    @pytest.mark.asyncio
    async def test_enter_agent_in_hook_scope_raises(self):
        """EnterAgent yielded from any hook → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                if False:
                    yield

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_workflow"):
            await agent._dispatch_call(
                EnterAgent(goal="x"),
                agent._current_context,
                scope="hook",
            )


# ---------------------------------------------------------------------------
# ThinkUnit scope rules — agent only
# ---------------------------------------------------------------------------


class TestThinkUnitScope:

    @pytest.mark.asyncio
    async def test_think_unit_in_hook_raises(self):
        """ThinkUnit yielded with scope='hook' → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            think_unit_a = think_unit(CognitiveWorker.inline("a"), max_attempts=1)

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await agent._dispatch_call(
                ThinkUnit("think_unit_a"),
                agent._current_context,
                scope="hook",
            )

    @pytest.mark.asyncio
    async def test_think_unit_in_workflow_raises(self):
        """ThinkUnit yielded with scope='workflow' → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            think_unit_a = think_unit(CognitiveWorker.inline("a"), max_attempts=1)

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await agent._dispatch_call(
                ThinkUnit("think_unit_a"),
                agent._current_context,
                scope="workflow",
            )


# ---------------------------------------------------------------------------
# Atomic Calls (ActionCall / HumanCall / LLMCall) — workflow + hooks, NOT agent
# ---------------------------------------------------------------------------


class TestActionCallScope:

    @pytest.mark.asyncio
    async def test_action_call_in_agent_raises(self):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            pass

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="not allowed inside on_agent"):
            await agent._dispatch_call(
                ActionCall("some_tool", x=1),
                agent._current_context,
                scope="agent",
            )


class TestHumanCallScope:

    @pytest.mark.asyncio
    async def test_human_call_in_agent_raises(self):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def feed(self, prompt: str) -> str:
                return "ok"

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="not allowed inside on_agent"):
            await agent._dispatch_call(
                HumanCall(prompt="confirm"),
                agent._current_context,
                scope="agent",
            )

    @pytest.mark.parametrize("scope", ["workflow", "hook"])
    @pytest.mark.asyncio
    async def test_human_call_works_in_workflow_and_hook(self, scope):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def feed(self, prompt: str) -> str:
                return "ok"

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        result = await agent._dispatch_call(
            HumanCall(prompt="confirm"),
            agent._current_context,
            scope=scope,
        )
        assert result == "ok"


class TestLLMCallScope:

    @pytest.mark.asyncio
    async def test_llm_call_in_agent_raises(self):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            pass

        agent = Agent(llm=_DummyLLM())
        agent._current_context = _ctx()
        with pytest.raises(RuntimeError, match="not allowed inside on_agent"):
            await agent._dispatch_call(
                LLMCall.chat("hi"),
                agent._current_context,
                scope="agent",
            )
