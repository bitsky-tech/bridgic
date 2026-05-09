"""Tests for the unified ``EnterAgent`` semantics (Q4).

After Q4: ``EnterAgent`` defaults to delegating the sub-task to the
agent's own ``on_agent`` strategy (inside a snapshot). The legacy
"ad-hoc worker" behavior is preserved as an escape hatch via the
``worker=`` field.
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
    RETURN,
    StepToolCall,
    ToolArgument,
    think_unit,
    ThinkUnit,
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


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="parent goal")


def _finish() -> ThinkDecision:
    return ThinkDecision(step_content="Done", output=[], finish=True)


# ---------------------------------------------------------------------------
# 1. Default delegation: EnterAgent → on_agent
# ---------------------------------------------------------------------------


class TestEnterAgentDelegation:

    @pytest.mark.asyncio
    async def test_default_delegates_to_on_agent(self):
        """EnterAgent(goal=...) without worker → invokes on_agent in a snapshot."""
        on_agent_invocations = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                # Record the goal seen by each on_agent invocation —
                # demonstrates snapshot scoping.
                on_agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="sub-goal-1")
                yield EnterAgent(goal="sub-goal-2")

        llm = MockLLM([_finish(), _finish()])
        await Agent(llm=llm).arun(context=_ctx())

        # on_agent runs once for each EnterAgent, with the snapshotted goal.
        assert on_agent_invocations == ["sub-goal-1", "sub-goal-2"]

    @pytest.mark.asyncio
    async def test_snapshot_isolates_sub_goal_from_parent(self):
        """The parent's ctx.goal is restored after the EnterAgent returns."""
        observed_goals = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                observed_goals.append(("inside-on_agent", ctx.goal))
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                observed_goals.append(("before-call", ctx.goal))
                yield EnterAgent(goal="sub-task")
                observed_goals.append(("after-call", ctx.goal))

        llm = MockLLM([_finish()])
        await Agent(llm=llm).arun(context=_ctx())

        assert observed_goals == [
            ("before-call", "parent goal"),
            ("inside-on_agent", "sub-task"),
            ("after-call", "parent goal"),
        ]

    @pytest.mark.asyncio
    async def test_no_on_agent_no_worker_raises(self):
        """EnterAgent without worker= and without on_agent override → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="orphan")

        with pytest.raises(RuntimeError, match="requires an on_agent"):
            await Agent().arun(context=_ctx())

    @pytest.mark.asyncio
    async def test_delegation_reuses_think_units(self):
        """on_agent's declared think_units are reusable from EnterAgent."""
        llm = MockLLM([_finish()] * 4)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            sub_think = think_unit(CognitiveWorker.inline("sub"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("sub_think")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="A")
                yield EnterAgent(goal="B")

        await Agent(llm=llm).arun(context=_ctx())

        # Two EnterAgent yields × one ThinkUnit each → at least 2 think cycles
        assert llm.call_count >= 2


# ---------------------------------------------------------------------------
# 2. EnterAgent fields: only context-scoping kwargs are allowed
# ---------------------------------------------------------------------------


class TestEnterAgentSlimFields:

    def test_agent_call_rejects_worker_kwarg(self):
        """EnterAgent no longer accepts ``worker=`` — no ad-hoc escape hatch."""
        with pytest.raises(TypeError):
            EnterAgent(goal="x", worker=CognitiveWorker.inline("foo"))  # type: ignore[call-arg]

    def test_agent_call_rejects_max_attempts_kwarg(self):
        """EnterAgent doesn't control "how to think" — no per-call attempt budget."""
        with pytest.raises(TypeError):
            EnterAgent(goal="x", max_attempts=3)  # type: ignore[call-arg]

    def test_agent_call_accepts_context_scoping_kwargs(self):
        """EnterAgent keeps tools / skills / history for sub-agent context scoping."""
        # No exception means the dataclass accepts these kwargs.
        EnterAgent(goal="x", tools=["t1"], skills=["s1"], history=None)


class TestEnterAgentToolScoping:

    @pytest.mark.asyncio
    async def test_agent_call_tools_filter_visible_to_sub_agent(self):
        """``EnterAgent(tools=[...])`` restricts what the sub-agent's on_agent sees."""
        seen_tool_names: List[List[str]] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                seen_tool_names.append(
                    sorted(t.tool_name for t in ctx.tools.get_all())
                )
                # Empty body — just observe what tools the snapshot exposed.
                if False:
                    yield

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield EnterAgent(goal="scoped", tools=["request_human"])

        llm = MockLLM([_finish()])
        await Agent(llm=llm).arun(context=_ctx())

        # The sub-agent saw only the filtered tool name.
        assert seen_tool_names == [["request_human"]]
