"""Tests for the ``RETURN`` yield primitive.

Verifies:

* ``yield RETURN(value)`` captures the value and closes the generator
* code after a ``yield RETURN`` is unreachable
* exhausting without ``RETURN`` returns ``None``
* top-level ``on_workflow`` / ``on_agent`` ``RETURN`` writes to ``final_answer``
* the auto-capture path still works when no ``RETURN`` is yielded
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
    RETURN,
    StepToolCall,
    ToolArgument,
    human_channel,
)


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="RETURN test")


class TestReturnPrimitive:

    @pytest.mark.asyncio
    async def test_return_value_writes_to_final_answer(self):
        """yield RETURN(value) at the top of on_workflow → final_answer = value."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                yield RETURN("explicit-answer")

        agent = Agent()
        await agent.arun(context=_ctx())

        assert agent.final_answer == "explicit-answer"

    @pytest.mark.asyncio
    async def test_yields_after_return_are_unreachable(self):
        """Code after yield RETURN never runs (generator is closed)."""
        executed_after = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                yield RETURN("done")
                executed_after.append("REACHED")  # must never run
                yield ActionCall("never_called")

        await Agent().arun(context=_ctx())

        assert executed_after == []

    @pytest.mark.asyncio
    async def test_no_return_means_none_captured(self):
        """Generator that exhausts without RETURN → captured value is None."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                if False:
                    yield  # empty generator

        agent = Agent()
        await agent.arun(context=_ctx())

        # final_answer is auto-captured from history (or summary fallback);
        # the explicit RETURN-override path was NOT taken.
        # We just verify the run completes; final_answer may be None or "" empty,
        # depending on auto-capture behavior with no steps.
        assert agent.final_answer is None  # no steps → no auto-capture either

    @pytest.mark.asyncio
    async def test_return_in_on_agent_writes_to_final_answer(self):
        """yield RETURN(value) in on_agent → final_answer = value."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield RETURN("from-on-agent")

        agent = Agent(llm=MockLLM([]))
        await agent.arun(context=_ctx())

        assert agent.final_answer == "from-on-agent"

    @pytest.mark.asyncio
    async def test_return_after_other_yields(self):
        """RETURN can come after other yields; intermediate yields still execute."""
        log = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                log.append(("ask", prompt))
                return "ok"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="step-1")
                log.append(("got", resp))
                yield RETURN("final-result")

        agent = Agent()
        await agent.arun(context=_ctx())

        assert log == [("ask", "step-1"), ("got", "ok")]
        assert agent.final_answer == "final-result"

    @pytest.mark.asyncio
    async def test_return_value_preserved_through_dispatcher(self):
        """The dispatcher returns the captured RETURN.value to its caller."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall, LLMCall], None
            ]:
                yield RETURN({"kind": "structured", "n": 42})

        agent = Agent()
        agent._current_context = _ctx()
        captured = await agent._dispatch_flow(
            agent.on_workflow(agent._current_context),
            agent._current_context,
        )

        assert captured == {"kind": "structured", "n": 42}
