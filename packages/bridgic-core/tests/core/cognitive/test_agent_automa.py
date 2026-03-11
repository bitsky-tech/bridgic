"""Tests for the amphibious refactoring: self.run(), execute_plan(), trace, operators, etc."""
import types
import pytest
from typing import Any, List, Optional

from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    think_step,
    ErrorStrategy,
    Workflow,
    TraceStep,
    ExecutionTrace,
    DivergenceDetector,
    DivergenceLevel,
    WorkflowToolCall,
    FlowStep,
    Loop,
    Sequence,
    Branch,
)
from bridgic.core.cognitive._cognitive_worker import StepToolCall, ToolArgument
from bridgic.core.cognitive._amphibious import (
    AmphibiousRunner,
    WorkflowPatcher,
    DivergenceError,
    ExecutionMode,
)
from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(**kwargs):
    defaults = {"step_content": "", "output": [], "finish": False, "details": []}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


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
# Tests — execute_plan() with operators
# ---------------------------------------------------------------------------

class TestExecutePlan:

    @pytest.mark.asyncio
    async def test_execute_step_operator(self):
        """execute_plan(FlowStep(...)) executes a single step."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Plan", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                step = FlowStep(worker=worker, name="plan")
                await self.execute_plan(step)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 1

    @pytest.mark.asyncio
    async def test_execute_loop_operator(self):
        """execute_plan(Loop(...)) loops until finish."""
        llm = MockLLM([_make_search_step(), _make_hotel_step(finish=True)])
        worker = CognitiveWorker.inline("Execute", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                loop = Loop(worker=worker, name="main_loop", max_attempts=5)
                await self.execute_plan(loop)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_execute_sequence_operator(self):
        """execute_plan(Sequence([...])) executes steps in order."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])
        worker1 = CognitiveWorker.inline("Step 1", llm=llm)
        worker2 = CognitiveWorker.inline("Step 2", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                seq = Sequence([
                    FlowStep(worker=worker1, name="step1"),
                    FlowStep(worker=worker2, name="step2"),
                ], name="sequence")
                await self.execute_plan(seq)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_execute_branch_operator(self):
        """execute_plan(Branch(...)) takes the correct branch."""
        llm = MockLLM([_make_search_step()])
        worker_a = CognitiveWorker.inline("Branch A", llm=llm)
        worker_b = CognitiveWorker.inline("Branch B", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                branch = Branch(
                    condition=lambda ctx: "a",
                    branches={
                        "a": FlowStep(worker=worker_a, name="branch_a"),
                        "b": FlowStep(worker=worker_b, name="branch_b"),
                    },
                    name="choice",
                )
                await self.execute_plan(branch)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 1


# ---------------------------------------------------------------------------
# Tests — exception_scope
# ---------------------------------------------------------------------------

class TestExceptionScope:

    @pytest.mark.asyncio
    async def test_exception_scope_marks_steps(self):
        """Steps in exception_scope are marked is_exception_handler=True in trace."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])
        worker = CognitiveWorker.inline("Handle", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                async with self.exception_scope():
                    await self.run(worker, name="handle_error")
                await self.run(worker, name="normal_step")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx, capture_workflow=True)

        ctx_out, workflow = result
        assert isinstance(workflow, Workflow)

        # The exception step should be excluded from workflow blocks
        all_steps = []
        for block in workflow.blocks:
            if hasattr(block, 'steps'):
                all_steps.extend(block.steps)

        # Only non-exception steps should be in the workflow
        non_exception = [s for s in all_steps if not s.get("is_exception_handler", False)]
        assert len(non_exception) >= 1

    @pytest.mark.asyncio
    async def test_exception_scope_restores_flag(self):
        """exception_scope restores _in_exception_scope on exit."""
        llm = MockLLM([_make_search_step()])
        worker = CognitiveWorker.inline("Test", llm=llm)

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                assert not self._in_exception_scope
                async with self.exception_scope():
                    assert self._in_exception_scope
                assert not self._in_exception_scope
                await self.run(worker, name="after")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx)


# ---------------------------------------------------------------------------
# Tests — TraceStep and DivergenceDetector
# ---------------------------------------------------------------------------

class TestTraceAndDivergence:

    def test_trace_step_hash(self):
        """TraceStep computes stable hashes."""
        ts = TraceStep(index=0, name="test")
        ts.observation = "hello world"
        h = ts.compute_observation_hash()
        assert len(h) == 16
        # Same input → same hash
        assert h == TraceStep.compute_hash("hello world")

    def test_trace_step_result_hashes(self):
        """TraceStep computes result hashes."""
        ts = TraceStep(index=0, name="test")
        ts.tool_results = ["result1", "result2"]
        hashes = ts.compute_result_hashes()
        assert len(hashes) == 2
        assert hashes[0] != hashes[1]

    def test_execution_trace_add_step(self):
        """ExecutionTrace.add_step creates and appends steps."""
        trace = ExecutionTrace()
        s1 = trace.add_step("step1", observation="obs1")
        s2 = trace.add_step("step2", observation="obs2")

        assert len(trace.steps) == 2
        assert s1.index == 0
        assert s2.index == 1
        assert s1.observation_hash != ""

    def test_divergence_detector_match(self):
        """DivergenceDetector returns MATCH for identical data."""
        detector = DivergenceDetector()
        recorded = TraceStep(
            index=0, name="test",
            observation="hello",
            tool_calls=[WorkflowToolCall(tool_name="search", tool_arguments={})],
        )
        recorded.compute_observation_hash()
        recorded.tool_results = ["result1"]
        recorded.compute_result_hashes()

        level = detector.check(
            recorded=recorded,
            actual_observation="hello",
            actual_tool_names=["search"],
            actual_results=["result1"],
        )
        assert level == DivergenceLevel.MATCH

    def test_divergence_detector_minor(self):
        """DivergenceDetector returns MINOR for same tools, different data."""
        detector = DivergenceDetector()
        recorded = TraceStep(
            index=0, name="test",
            observation="hello",
            tool_calls=[WorkflowToolCall(tool_name="search", tool_arguments={})],
        )
        recorded.compute_observation_hash()
        recorded.tool_results = ["result1"]
        recorded.compute_result_hashes()

        level = detector.check(
            recorded=recorded,
            actual_observation="world",  # Different observation
            actual_tool_names=["search"],
            actual_results=["result1"],
        )
        assert level == DivergenceLevel.MINOR

    def test_divergence_detector_major(self):
        """DivergenceDetector returns MAJOR for different tool names."""
        detector = DivergenceDetector()
        recorded = TraceStep(
            index=0, name="test",
            tool_calls=[WorkflowToolCall(tool_name="search", tool_arguments={})],
        )
        recorded.compute_observation_hash()

        level = detector.check(
            recorded=recorded,
            actual_observation=None,
            actual_tool_names=["click"],  # Different tool
            actual_results=[],
        )
        assert level == DivergenceLevel.MAJOR


# ---------------------------------------------------------------------------
# Tests — Workflow data model
# ---------------------------------------------------------------------------

class TestWorkflowDataModel:

    def test_workflow_serialization_roundtrip(self):
        """Workflow serializes and deserializes correctly."""
        wf = Workflow()
        wf.add_step_block("step1", tool_calls=[
            WorkflowToolCall(tool_name="search", tool_arguments={"q": "test"})
        ])
        wf.add_loop_block("loop1", max_attempts=5)
        wf.add_linear_trace_block("trace1", steps=[
            {"index": 0, "name": "s1", "tool_calls": []}
        ])

        data = wf.model_dump()
        restored = Workflow(**data)

        assert len(restored.blocks) == 3
        assert restored.blocks[0].type == "step"
        assert restored.blocks[1].type == "loop"
        assert restored.blocks[2].type == "linear_trace"

    def test_workflow_version_starts_at_1(self):
        """New workflow starts at version 1."""
        wf = Workflow()
        assert wf.version == 1


# ---------------------------------------------------------------------------
# Tests — backward compatibility with think_step
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    @pytest.mark.asyncio
    async def test_think_step_still_works(self):
        """Legacy think_step class attribute API still works."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            step = think_step(CognitiveWorker.inline("Plan step"))

            async def cognition(self, ctx):
                await self.step.until(max_attempts=5)

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_think_step_inline_still_works(self):
        """think_step.inline() convenience method still works."""
        llm = MockLLM([_make_search_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            step = think_step.inline("Plan step")

            async def cognition(self, ctx):
                await self.step

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 1

    @pytest.mark.asyncio
    async def test_mixed_old_and_new_api(self):
        """Can mix think_step and self.run() in the same agent."""
        llm = MockLLM([_make_search_step(), _make_hotel_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            legacy_step = think_step(CognitiveWorker.inline("Legacy"))

            async def cognition(self, ctx):
                await self.legacy_step
                worker = CognitiveWorker.inline("New API", llm=self.llm)
                await self.run(worker, name="new_step")

        agent = Agent(llm=llm)
        ctx = _make_ctx()
        result = await agent.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_ctx_init_still_works(self):
        """ctx_init parameter still works for backward compatibility."""
        llm = MockLLM([_make_search_step()])

        class Agent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                await self.run(
                    CognitiveWorker.inline("Test"),
                    name="test",
                )

        from .tools import get_travel_planning_tools
        agent = Agent(
            llm=llm,
            ctx_init={"tools": get_travel_planning_tools()},
        )
        result = await agent.arun(goal="Test goal")

        assert isinstance(result, CognitiveContext)
        steps = result.cognitive_history.get_all()
        assert len(steps) == 1

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


# ---------------------------------------------------------------------------
# Tests — WorkflowPatcher
# ---------------------------------------------------------------------------

class TestWorkflowPatcher:

    def test_guard_patch_from_exception_steps(self):
        """WorkflowPatcher creates guard patches for exception-handler steps."""
        wf = Workflow()
        wf.add_linear_trace_block("main", steps=[
            {"index": 0, "name": "s1", "tool_calls": []},
        ])

        log = [
            TraceStep(
                index=0, name="handle_error",
                is_exception_handler=True,
                tool_calls=[WorkflowToolCall(tool_name="close_popup", tool_arguments={})],
            ),
        ]

        patcher = WorkflowPatcher(wf, log)
        patches = patcher.analyze_and_patch()

        guard_patches = [p for p in patches if p.type == "guard"]
        assert len(guard_patches) >= 1
        assert wf.version == 2  # Version incremented

    def test_no_patches_when_no_divergence(self):
        """WorkflowPatcher produces no patches for normal execution."""
        wf = Workflow()
        wf.add_linear_trace_block("main", steps=[
            {
                "index": 0, "name": "s1",
                "tool_calls": [
                    {"tool_name": "search", "tool_arguments": {"q": "test"}, "tool_result": None}
                ],
            },
        ])

        log = [
            TraceStep(
                index=0, name="s1",
                tool_calls=[WorkflowToolCall(
                    tool_name="search",
                    tool_arguments={"q": "test"},
                )],
            ),
        ]

        patcher = WorkflowPatcher(wf, log)
        patches = patcher.analyze_and_patch()

        assert len(patches) == 0
        assert wf.version == 1  # Not incremented
