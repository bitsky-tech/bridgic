"""Tests for AmphibiousAutoma: _run(), error strategies, tool filtering, etc."""
import json
import os
import tempfile

import pytest
from typing import Any, List, Optional

from bridgic.amphibious import (
    AmphibiousAutoma,
    AgentTrace,
    CognitiveContext,
    CognitiveWorker,
    ErrorStrategy,
    StepToolCall,
    ToolArgument,
    TraceStep,
    RecordedToolCall,
)
from .tools import get_travel_planning_tools

# Default decision model for mock LLM responses (no policies, no output_schema)
ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


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
# Tests — _run() method
# ---------------------------------------------------------------------------

class TestRunMethod:

    @pytest.mark.asyncio
    async def test_run_single_step(self):
        """_run() executes a single OTA cycle."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])
        worker = CognitiveWorker.inline("Plan step", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 1
        assert steps[0].result.results[0].tool_name == "search_flights"

    @pytest.mark.asyncio
    async def test_run_with_until(self):
        """_run(until=..., max_attempts=...) loops correctly."""
        llm = MockLLM([_make_search_step(), _make_hotel_step(finish=True)])
        worker = CognitiveWorker.inline("Execute step", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=5)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        # LLM signals finish=True on step 2, so 2 steps total
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_run_with_condition(self):
        """_run(until=condition) stops when condition is True."""
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

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker, until=condition, max_attempts=10)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_run_tool_filtering(self):
        """_run(tools=[...]) filters visible tools."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Search", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker,
                               tools=["search_flights", "search_hotels"])

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

        # Should complete successfully with filtered tools
        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 1

    @pytest.mark.asyncio
    async def test_run_error_strategy_ignore(self):
        """_run(on_error=IGNORE) silently ignores errors."""
        class FailLLM:
            async def astructured_output(self, messages, constraint, **kwargs):
                raise RuntimeError("LLM failed")
            async def achat(self, messages, **kwargs): ...
            async def astream(self, messages, **kwargs): ...
            def chat(self, messages, **kwargs): ...
            def stream(self, messages, **kwargs): ...

        llm = FailLLM()
        worker = CognitiveWorker.inline("Fail", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker, on_error=ErrorStrategy.IGNORE)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        # Should not raise
        await agent.arun(context=ctx)
        assert isinstance(agent._current_context, CognitiveContext)

    @pytest.mark.asyncio
    async def test_run_no_context_raises(self):
        """_run() outside on_agent() raises RuntimeError."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Test", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

        agent = Agent(llm=llm)
        with pytest.raises(RuntimeError, match="no active context"):
            await agent._run(worker)


# ---------------------------------------------------------------------------
# Tests — AmphibiousAutoma properties and context initialization
# ---------------------------------------------------------------------------

class TestAmphibiousAutomaMisc:

    @pytest.mark.asyncio
    async def test_llm_property(self):
        """Agent.llm property exposes the default LLM."""
        llm = MockLLM([_make_search_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                assert self.llm is not None
                worker = CognitiveWorker.inline("Test", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

    @pytest.mark.asyncio
    async def test_arun_auto_create_context(self):
        """arun() auto-creates context from kwargs when no context is provided."""
        llm = MockLLM([_make_search_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                assert ctx.goal == "Test goal"
                worker = CognitiveWorker.inline("Test", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(
            goal="Test goal",
            tools=get_travel_planning_tools(),
        )
        assert isinstance(agent._current_context, CognitiveContext)
        assert len(agent._current_context.cognitive_history.get_all()) == 1

    @pytest.mark.asyncio
    async def test_arun_requires_llm(self):
        """arun() raises if no LLM is provided."""
        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

        agent = Agent()
        with pytest.raises(RuntimeError, match="must be initialized with an LLM"):
            await agent.arun(goal="Test")

    @pytest.mark.asyncio
    async def test_multiple_workers_in_on_agent(self):
        """on_agent() can orchestrate multiple workers sequentially."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                planner = CognitiveWorker.inline("Plan", llm=self.llm)
                executor = CognitiveWorker.inline("Execute", llm=self.llm)
                await self._run(planner)
                await self._run(executor)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 2
        assert steps[0].result.results[0].tool_name == "search_flights"
        assert steps[1].result.results[0].tool_name == "search_hotels"


# ---------------------------------------------------------------------------
# Tests — AgentTrace: observation, success/error, finished removal, save/load
# ---------------------------------------------------------------------------

class TestAgentTrace:

    @pytest.mark.asyncio
    async def test_trace_step_has_observation_field(self):
        """Trace steps should have the observation field (None when no observation provided)."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, trace_running=True)

        trace = agent._agent_trace.build()
        assert len(trace["steps"]) == 1
        step: TraceStep = trace["steps"][0]
        # observation field exists on the model
        assert "observation" in TraceStep.model_fields
        # Default CognitiveContext has no observation() override, so it's None
        assert step.observation is None

    @pytest.mark.asyncio
    async def test_trace_records_observation_when_provided(self):
        """Trace steps should record observation text when the agent provides one."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx):
                return "Current page: login form with username and password fields"

            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, trace_running=True)

        trace = agent._agent_trace.build()
        step: TraceStep = trace["steps"][0]
        assert step.observation is not None
        assert "login form" in step.observation
        assert step.observation_hash is not None

    @pytest.mark.asyncio
    async def test_trace_tool_call_success_error(self):
        """RecordedToolCall should carry success and error fields."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, trace_running=True)

        trace = agent._agent_trace.build()
        step: TraceStep = trace["steps"][0]
        assert len(step.tool_calls) == 1
        tc: RecordedToolCall = step.tool_calls[0]
        assert tc.success is True
        assert tc.error is None

    @pytest.mark.asyncio
    async def test_trace_no_finished_field(self):
        """TraceStep should not have a 'finished' field."""
        assert "finished" not in TraceStep.model_fields

    @pytest.mark.asyncio
    async def test_trace_save_load_roundtrip(self):
        """save() then load() should produce equivalent data."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, trace_running=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            agent._agent_trace.save(path)
            loaded = AgentTrace.load(path)

            assert "steps" in loaded
            assert "metadata" in loaded
            assert len(loaded["steps"]) == 1

            step = loaded["steps"][0]
            assert "observation" in step
            assert "finished" not in step
            assert len(step["tool_calls"]) == 1
            assert step["tool_calls"][0]["tool_name"] == "search_flights"
            assert step["tool_calls"][0]["success"] is True
            assert step["tool_calls"][0]["error"] is None
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_trace_records_all_steps(self):
        """All steps are recorded in the flat trace."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, trace_running=True)

        trace = agent._agent_trace.build()
        assert len(trace["steps"]) == 1
        step: TraceStep = trace["steps"][0]
        assert "observation" in TraceStep.model_fields
        assert "finished" not in TraceStep.model_fields
