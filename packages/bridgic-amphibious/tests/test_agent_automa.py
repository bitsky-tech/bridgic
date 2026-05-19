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
    ThinkUnit,
    think_unit,
    RETURN,
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
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan step"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 1
        assert steps[0].result.results[0].tool_name == "search_flights"

    @pytest.mark.asyncio
    async def test_run_with_until(self):
        """_run(until=..., max_attempts=...) loops correctly."""
        llm = MockLLM([_make_search_step(), _make_hotel_step(finish=True)])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=5)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

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
        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(CognitiveWorker.inline("Execute"), until=condition, max_attempts=10)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_run_tool_filtering(self):
        """_run(tools=[...]) filters visible tools."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(
                CognitiveWorker.inline("Search"),
                tools=["search_flights", "search_hotels"],
            )

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

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
        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(CognitiveWorker.inline("Fail"), on_error=ErrorStrategy.IGNORE)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = Agent()
        ctx = _make_ctx()
        # Should not raise
        await agent.arun(llm=llm, context=ctx)
        assert isinstance(agent._current_context, CognitiveContext)

    @pytest.mark.asyncio
    async def test_run_no_context_raises(self):
        """_run_think_unit() outside on_agent() raises RuntimeError."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Test", llm=llm)

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                if False:
                    yield

        agent = Agent()
        with pytest.raises(RuntimeError, match="no active context"):
            await agent._run_think_unit(worker)


# ---------------------------------------------------------------------------
# Tests — AmphibiousAutoma properties and context initialization
# ---------------------------------------------------------------------------

class TestAmphibiousAutomaMisc:

    @pytest.mark.asyncio
    async def test_llm_property(self):
        """Agent.llm property exposes the default LLM."""
        llm = MockLLM([_make_search_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(CognitiveWorker.inline("Test"))

            async def on_agent(self, ctx):
                assert self.llm is not None
                yield ThinkUnit("step")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

    @pytest.mark.asyncio
    async def test_arun_auto_create_context(self):
        """arun() auto-creates context from kwargs when no context is provided."""
        llm = MockLLM([_make_search_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            step = think_unit(CognitiveWorker.inline("Test"))

            async def on_agent(self, ctx):
                assert ctx.goal == "Test goal"
                yield ThinkUnit("step")

        agent = Agent()
        await agent.arun(
            llm=llm,
            goal="Test goal",
            tools=get_travel_planning_tools(),
        )
        assert isinstance(agent._current_context, CognitiveContext)
        assert len(agent._current_context.cognitive_history.get_all()) == 1

    @pytest.mark.asyncio
    async def test_arun_succeeds_without_llm_when_no_llm_primitive_yielded(self):
        """arun() does NOT fail eagerly when llm is missing — the LLM is
        an optional dependency consumed only by LLMCall / ThinkUnit's
        CognitiveWorker path. A no-op on_agent (or one that only uses
        ThinkAgent) is valid without an LLM."""
        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                if False:
                    yield

        agent = Agent()
        await agent.arun(goal="Test")  # no raise

    @pytest.mark.asyncio
    async def test_llmcall_without_llm_raises_at_use_site(self):
        """Yielding LLMCall when no LLM was provided surfaces a clear
        error at the dispatcher use-point, not eagerly at arun().

        ``LLMCall`` is scope-restricted to ``on_workflow`` / hooks /
        ``CognitiveWorker.thinking()`` — never directly inside
        ``on_agent`` — so this test exercises the on_workflow path."""
        from bridgic.amphibious import LLMCall

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx):
                yield LLMCall.chat("hi")

        agent = Agent()
        with pytest.raises(RuntimeError, match="LLMCall.*requires self._llm"):
            await agent.arun(goal="Test")

    @pytest.mark.asyncio
    async def test_thinkunit_cognitive_worker_without_llm_raises_at_use_site(self):
        """ThinkUnit driving a CognitiveWorker requires an LLM; if neither
        the worker nor the agent has one, raise a clear error before OTC
        starts (instead of crashing later with AttributeError)."""
        class Agent(AmphibiousAutoma[CognitiveContext]):
            planner = think_unit(CognitiveWorker.inline("Plan it"))

            async def on_agent(self, ctx):
                yield ThinkUnit("planner")

        agent = Agent()
        with pytest.raises(RuntimeError, match="CognitiveWorker.*has no LLM"):
            await agent.arun(goal="Test")

    @pytest.mark.asyncio
    async def test_multiple_workers_in_on_agent(self):
        """on_agent() can orchestrate multiple workers sequentially."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            planner = think_unit(CognitiveWorker.inline("Plan"))
            executor = think_unit(CognitiveWorker.inline("Execute"))

            async def on_agent(self, ctx):
                yield ThinkUnit("planner")
                yield ThinkUnit("executor")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx)

        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 2
        assert steps[0].result.results[0].tool_name == "search_flights"
        assert steps[1].result.results[0].tool_name == "search_hotels"


# ---------------------------------------------------------------------------
# Tests — AgentTrace: observation, success/error, finished removal, save/load
# ---------------------------------------------------------------------------

class TestAgentTrace:

    @pytest.mark.asyncio
    async def test_trace_step_has_observation_field(self, tmp_path):
        """Trace history entries should have the observation field (None when no observation provided)."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx, workdir=tmp_path, trace=True)

        trace = agent._agent_trace.build()
        assert len(trace["history"]) == 1
        step: TraceStep = trace["history"][0]
        # observation field exists on the model
        assert "observation" in TraceStep.model_fields
        # Default CognitiveContext has no observation() override, so it's None
        assert step.observation is None

    @pytest.mark.asyncio
    async def test_trace_records_observation_when_provided(self, tmp_path):
        """Trace history should record observation text when the agent provides one."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def observation(self, ctx):
                yield RETURN("Current page: login form with username and password fields")

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx, workdir=tmp_path, trace=True)

        trace = agent._agent_trace.build()
        step: TraceStep = trace["history"][0]
        assert step.observation is not None
        assert "login form" in step.observation
        assert step.observation_hash is not None

    @pytest.mark.asyncio
    async def test_trace_tool_call_success_error(self, tmp_path):
        """RecordedToolCall should carry success and error fields."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx, workdir=tmp_path, trace=True)

        trace = agent._agent_trace.build()
        step: TraceStep = trace["history"][0]
        assert len(step.tool_calls) == 1
        tc: RecordedToolCall = step.tool_calls[0]
        assert tc.success is True
        assert tc.error is None

    @pytest.mark.asyncio
    async def test_trace_no_finished_field(self, tmp_path):
        """TraceStep should not have a 'finished' field."""
        assert "finished" not in TraceStep.model_fields

    @pytest.mark.asyncio
    async def test_trace_save_load_roundtrip(self, tmp_path):
        """save() then load() should produce equivalent data with the
        unified ``goal`` / ``metadata`` / ``history`` shape."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx, workdir=tmp_path, trace=True)

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

    @pytest.mark.asyncio
    async def test_trace_records_all_steps(self, tmp_path):
        """All steps are recorded into the flat ``history`` list."""
        llm = MockLLM([_make_search_step()])
        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = Agent()
        ctx = _make_ctx()
        await agent.arun(llm=llm, context=ctx, workdir=tmp_path, trace=True)

        trace = agent._agent_trace.build()
        assert len(trace["history"]) == 1
        step: TraceStep = trace["history"][0]
        assert "observation" in TraceStep.model_fields
        assert "finished" not in TraceStep.model_fields
