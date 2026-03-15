"""Tests for AgentAutoma: self.run(), error strategies, tool filtering, etc."""
import pytest
from typing import Any, List, Optional

from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    ErrorStrategy,
    ThinkDecision,
)
from bridgic.core.cognitive._cognitive_worker import StepToolCall, ToolArgument
from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(**kwargs):
    """Create a ThinkDecision instance for mock LLM responses."""
    return ThinkDecision(**kwargs)


class MockLLM:
    """Returns a fixed sequence of responses."""

    def __init__(self, responses: List[Any]):
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


def _make_ctx() -> CognitiveContext:
    ctx = CognitiveContext(goal="Test goal")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    return ctx


def _make_search_step():
    return _tr(
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
        finish=False,
    )


def _make_hotel_step(finish=True):
    return _tr(
        step_content="Search for hotels",
        output=[
            StepToolCall(
                tool="search_hotels",
                tool_arguments=[
                    ToolArgument(name="city", value="Tokyo"),
                    ToolArgument(name="check_in", value="2024-06-01"),
                    ToolArgument(name="check_out", value="2024-06-05"),
                ],
            )
        ],
        finish=finish,
    )


# ---------------------------------------------------------------------------
# Tests — self.run() method
# ---------------------------------------------------------------------------

class TestRunMethod:

    @pytest.mark.asyncio
    async def test_run_single_step(self):
        """self.run() executes a single OTA cycle."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])
        worker = CognitiveWorker.inline("Plan step", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(worker, name="search")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 1
        assert steps[0].metadata["action_results"][0]["tool_name"] == "search_flights"

    @pytest.mark.asyncio
    async def test_run_with_until(self):
        """self.run(until=..., max_attempts=...) loops correctly."""
        llm = MockLLM([_make_search_step(), _make_hotel_step(finish=True)])
        worker = CognitiveWorker.inline("Execute step", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(worker, name="execute", max_attempts=5)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        # LLM signals finish=True on step 2, so 2 steps total
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_run_with_condition(self):
        """self.run(until=condition) stops when condition is True."""
        call_count = 0

        def condition(ctx):
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Stop after 2 iterations

        llm = MockLLM([
            _make_search_step(),
            _make_search_step(),
            _make_search_step(),
        ])
        worker = CognitiveWorker.inline("Execute", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(worker, name="loop", until=condition, max_attempts=10)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_run_tool_filtering(self):
        """self.run(tools=[...]) filters visible tools."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Search", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(worker, name="search",
                               tools=["search_flights", "search_hotels"])

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        # Should complete successfully with filtered tools
        steps = result.cognitive_history.get_all()
        assert len(steps) == 1

    @pytest.mark.asyncio
    async def test_run_error_strategy_ignore(self):
        """self.run(on_error=IGNORE) silently ignores errors."""
        class FailLLM:
            async def astructured_output(self, messages, constraint, **kwargs):
                raise RuntimeError("LLM failed")
            async def achat(self, messages, **kwargs): ...
            async def astream(self, messages, **kwargs): ...
            def chat(self, messages, **kwargs): ...
            def stream(self, messages, **kwargs): ...

        llm = FailLLM()
        worker = CognitiveWorker.inline("Fail", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(worker, name="fail", on_error=ErrorStrategy.IGNORE)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        # Should not raise
        result = await agent.arun(context=ctx)
        assert isinstance(result, CognitiveContext)

    @pytest.mark.asyncio
    async def test_run_no_context_raises(self):
        """self.run() outside cognition() raises RuntimeError."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Test", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                pass

        agent = Agent(llm=llm)
        with pytest.raises(RuntimeError, match="no active context"):
            await agent.run(worker, name="test")


# ---------------------------------------------------------------------------
# Tests — AgentAutoma properties and context initialization
# ---------------------------------------------------------------------------

class TestAgentAutomaMisc:

    @pytest.mark.asyncio
    async def test_llm_property(self):
        """Agent.llm property exposes the default LLM."""
        llm = MockLLM([_make_search_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                assert self.llm is not None
                worker = CognitiveWorker.inline("Test", llm=self.llm)
                await self.run(worker, name="test")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

    @pytest.mark.asyncio
    async def test_arun_auto_create_context(self):
        """arun() auto-creates context from kwargs when no context is provided."""
        llm = MockLLM([_make_search_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                assert ctx.goal == "Test goal"
                worker = CognitiveWorker.inline("Test", llm=self.llm)
                await self.run(worker, name="test")

        agent = Agent(llm=llm)
        result = await agent.arun(
            goal="Test goal",
            tools=get_travel_planning_tools(),
        )
        assert isinstance(result, CognitiveContext)
        assert len(result.cognitive_history.get_all()) == 1

    @pytest.mark.asyncio
    async def test_arun_requires_llm(self):
        """arun() raises if no LLM is provided."""
        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                pass

        agent = Agent()
        with pytest.raises(RuntimeError, match="must be initialized with an LLM"):
            await agent.arun(goal="Test")

    @pytest.mark.asyncio
    async def test_multiple_workers_in_cognition(self):
        """cognition() can orchestrate multiple workers sequentially."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                planner = CognitiveWorker.inline("Plan", llm=self.llm)
                executor = CognitiveWorker.inline("Execute", llm=self.llm)
                await self.run(planner, name="plan")
                await self.run(executor, name="execute")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2
        assert steps[0].metadata["action_results"][0]["tool_name"] == "search_flights"
        assert steps[1].metadata["action_results"][0]["tool_name"] == "search_hotels"
