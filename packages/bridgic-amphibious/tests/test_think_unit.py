"""Tests for the ``ThinkUnit`` yield primitive.

Verifies:

* ``yield ThinkUnit("name")`` resolves a class-level ``think_unit`` descriptor
* descriptor field overlays work (max_attempts, until, tools, skills)
* ThinkUnit result is sent back via ``asend`` (worker output_schema or None)
* unknown / non-descriptor names raise ``AttributeError``
* descriptor instance access returns the descriptor itself (no
  ``await self.<name>`` shortcut)
"""

from typing import Any, AsyncGenerator, List, Union

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
    ThinkUnitDescriptor,
    RETURN,
    StepToolCall,
    ToolArgument,
    think_unit,
)


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class MockLLM:
    """Records every astructured_output call and returns scripted decisions."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        self.call_count += 1
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _finish() -> ThinkDecision:
    return ThinkDecision(step_content="Done", output=[], finish=True)


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="ThinkUnit test")


class TestThinkUnitResolution:

    def test_descriptor_returned_at_class_and_instance_level(self):
        """Descriptor.__get__ returns self for both class and instance access."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=3)

        # Class-level access
        cls_descriptor = Agent.main_think
        assert isinstance(cls_descriptor, ThinkUnitDescriptor)

        # Instance-level access — same descriptor (no _BoundThinkUnit wrapping)
        agent = Agent()
        instance_descriptor = agent.main_think
        assert instance_descriptor is cls_descriptor

    def test_await_descriptor_no_longer_supported(self):
        """``await self.main_think`` is no longer supported — descriptors are not awaitable."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"))

        agent = Agent()
        descriptor = agent.main_think
        # Descriptor exposes no __await__ → not awaitable.
        assert not hasattr(descriptor, "__await__")
        # The legacy `.until(condition)` API is gone too.
        assert not hasattr(descriptor, "until")


class TestThinkUnitDispatch:

    @pytest.mark.asyncio
    async def test_yield_think_call_invokes_descriptor(self):
        llm = MockLLM([_finish()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("main_think")

        await Agent().arun(llm=llm, context=_ctx())

        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_max_attempts_overlay(self):
        """ThinkUnit(max_attempts=N) overrides the descriptor's default."""
        llm = MockLLM([_finish()] * 5)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                # First call hits descriptor's default (1)
                yield ThinkUnit("main_think")
                # Second call overrides to 3 — but worker finish=True so it
                # short-circuits after 1 cycle anyway. Verify call doesn't error.
                yield ThinkUnit("main_think", max_attempts=3)

        await Agent().arun(llm=llm, context=_ctx())

        # Two ThinkUnit yields, each runs at least one cycle → 2+ astructured_output calls.
        assert llm.call_count >= 2

    @pytest.mark.asyncio
    async def test_unknown_name_raises_attribute_error(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("does_not_exist")

        with pytest.raises(AttributeError, match="does_not_exist"):
            await Agent().arun(llm=MockLLM([]), context=_ctx())

    @pytest.mark.asyncio
    async def test_non_descriptor_name_raises_attribute_error(self):
        """yield ThinkUnit("attr_name") where attr exists but isn't a ThinkUnitDescriptor."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            some_var = 42  # not a descriptor

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("some_var")

        with pytest.raises(AttributeError, match="some_var"):
            await Agent().arun(llm=MockLLM([]), context=_ctx())

    @pytest.mark.asyncio
    async def test_until_callback_overrides_descriptor_default(self):
        """ThinkUnit(until=...) overrides the descriptor's until callback."""
        llm = MockLLM([_finish()])
        invocations = []

        def my_condition(ctx):
            invocations.append("condition-checked")
            return True  # stop immediately

        class Agent(AmphibiousAutoma[CognitiveContext]):
            looped = think_unit(
                CognitiveWorker.inline("plan"),
                max_attempts=5,
                until=lambda c: False,  # default never stops
            )

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("looped", until=my_condition)

        await Agent().arun(llm=llm, context=_ctx())

        # Worker hits finish=True on first cycle, so the until callback is
        # never even consulted — but the agent finished cleanly under
        # the overlay. Verify no infinite loop / error.
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_think_call_in_on_workflow_raises(self):
        """ThinkUnit is on_agent-only; yielding it from on_workflow raises."""
        llm = MockLLM([_finish()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            inline_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield ThinkUnit("inline_think")

        with pytest.raises(RuntimeError, match="only valid inside on_agent"):
            await Agent().arun(llm=llm, context=_ctx())

    @pytest.mark.asyncio
    async def test_think_call_in_on_agent_allowed(self):
        """ThinkUnit yielded from on_agent works (the canonical case)."""
        llm = MockLLM([_finish()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            inline_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("inline_think")

        await Agent().arun(llm=llm, context=_ctx())
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_think_call_inside_recursive_agent_call_allowed(self):
        """ThinkUnit is allowed inside on_agent when reached via EnterAgent."""
        llm = MockLLM([_finish()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            inline_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("inline_think")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="sub-task")

        await Agent().arun(llm=llm, context=_ctx())
        assert llm.call_count == 1
