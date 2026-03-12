"""Tests for structured workflow capture (sequential/loop phases) and adaptive replay."""
import types
import pytest
from typing import Any, List

from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    Workflow,
)
from bridgic.core.cognitive._agent_automa import _WorkflowBuilder, _count_pattern_occurrences
from bridgic.core.cognitive._workflow import LinearTraceBlock, LoopBlock
from bridgic.core.cognitive._cognitive_worker import StepToolCall, ToolArgument
from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(**kwargs):
    defaults = {"step_content": "", "output": [], "finish": False, "details": []}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class StatefulMockLLM:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self._idx = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self._responses[self._idx]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _make_ctx() -> CognitiveContext:
    ctx = CognitiveContext(goal="Test structured workflow")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    return ctx


def _make_tool_call(tool_name: str, **args) -> StepToolCall:
    return StepToolCall(
        tool=tool_name,
        tool_arguments=[ToolArgument(name=k, value=v) for k, v in args.items()],
    )


# ---------------------------------------------------------------------------
# _WorkflowBuilder unit tests
# ---------------------------------------------------------------------------

class TestWorkflowBuilder:

    def test_empty_builder_has_no_content(self):
        builder = _WorkflowBuilder()
        assert not builder.has_content()

    def test_orphan_steps_have_content(self):
        builder = _WorkflowBuilder()
        builder.record_step({"name": "orphan", "tool_calls": []})
        assert builder.has_content()

    def test_sequential_phase_produces_linear_trace_block(self):
        builder = _WorkflowBuilder()
        builder.begin_phase("sequential", "nav")
        builder.record_step({"name": "s1", "tool_calls": [{"tool_name": "click"}]})
        builder.record_step({"name": "s2", "tool_calls": [{"tool_name": "type"}]})
        builder.end_phase()

        wf = builder.build()
        assert len(wf.blocks) == 1
        block = wf.blocks[0]
        assert isinstance(block, LinearTraceBlock)
        assert block.name == "nav"
        assert len(block.steps) == 2

    def test_loop_phase_produces_loop_block(self):
        builder = _WorkflowBuilder()
        builder.begin_phase("loop", "process")
        # Two iterations of: click → save → close
        for _ in range(2):
            builder.record_step({"name": "s", "tool_calls": [{"tool_name": "click"}]})
            builder.record_step({"name": "s", "tool_calls": [{"tool_name": "save"}]})
            builder.record_step({"name": "s", "tool_calls": [{"tool_name": "close"}]})
        builder.end_phase()

        wf = builder.build()
        assert len(wf.blocks) == 1
        block = wf.blocks[0]
        assert isinstance(block, LoopBlock)
        assert block.name == "process"
        assert block.pattern_template == ["click", "save", "close"]
        assert len(block.iterations) == 2

    def test_mixed_phases(self):
        builder = _WorkflowBuilder()

        # Sequential phase
        builder.begin_phase("sequential", "setup")
        builder.record_step({"name": "s1", "tool_calls": [{"tool_name": "navigate"}]})
        builder.end_phase()

        # Loop phase
        builder.begin_phase("loop", "process")
        for _ in range(3):
            builder.record_step({"name": "s", "tool_calls": [{"tool_name": "click"}]})
            builder.record_step({"name": "s", "tool_calls": [{"tool_name": "save"}]})
        builder.end_phase()

        wf = builder.build()
        assert len(wf.blocks) == 2
        assert isinstance(wf.blocks[0], LinearTraceBlock)
        assert isinstance(wf.blocks[1], LoopBlock)

    def test_orphan_steps_become_pre_block(self):
        builder = _WorkflowBuilder()
        builder.record_step({"name": "orphan", "tool_calls": []})
        builder.begin_phase("sequential", "main")
        builder.record_step({"name": "s1", "tool_calls": []})
        builder.end_phase()

        wf = builder.build()
        assert len(wf.blocks) == 2
        assert wf.blocks[0].name == "pre"
        assert wf.blocks[1].name == "main"

    def test_unclosed_phase_flushed_on_build(self):
        builder = _WorkflowBuilder()
        builder.begin_phase("sequential", "unclosed")
        builder.record_step({"name": "s1", "tool_calls": []})
        # No end_phase() call

        wf = builder.build()
        assert len(wf.blocks) == 1
        assert wf.blocks[0].name == "unclosed"


# ---------------------------------------------------------------------------
# Pattern detection unit tests
# ---------------------------------------------------------------------------

class TestPatternDetection:

    def test_count_pattern_occurrences_basic(self):
        seq = ["a", "b", "c", "a", "b", "c"]
        assert _count_pattern_occurrences(seq, ["a", "b", "c"]) == 2

    def test_count_pattern_with_noise(self):
        seq = ["a", "b", "x", "c", "a", "b", "c"]
        assert _count_pattern_occurrences(seq, ["a", "b", "c"]) == 2

    def test_count_pattern_no_match(self):
        seq = ["a", "b"]
        assert _count_pattern_occurrences(seq, ["x", "y"]) == 0

    def test_detect_loop_pattern_repeating(self):
        steps = []
        for _ in range(3):
            steps.append({"tool_calls": [{"tool_name": "click"}]})
            steps.append({"tool_calls": [{"tool_name": "save"}]})

        pattern, iterations = _WorkflowBuilder._detect_loop_pattern(steps)
        assert pattern == ["click", "save"]
        assert len(iterations) == 3

    def test_detect_loop_pattern_no_repeat(self):
        steps = [
            {"tool_calls": [{"tool_name": "a"}]},
            {"tool_calls": [{"tool_name": "b"}]},
            {"tool_calls": [{"tool_name": "c"}]},
        ]
        pattern, iterations = _WorkflowBuilder._detect_loop_pattern(steps)
        # No repeating pattern → single iteration with full sequence
        assert len(iterations) == 1

    def test_detect_loop_pattern_with_noise_steps(self):
        steps = []
        for _ in range(2):
            steps.append({"tool_calls": [{"tool_name": "click"}]})
            steps.append({"tool_calls": [{"tool_name": "wait_for"}]})  # noise
            steps.append({"tool_calls": [{"tool_name": "save"}]})

        pattern, iterations = _WorkflowBuilder._detect_loop_pattern(steps)
        # Consistent noise becomes part of the detected pattern
        assert pattern == ["click", "wait_for", "save"]
        assert len(iterations) == 2

    def test_detect_loop_pattern_empty(self):
        pattern, iterations = _WorkflowBuilder._detect_loop_pattern([])
        assert pattern == []
        assert iterations == []


# ---------------------------------------------------------------------------
# Integration: sequential() + loop() with AgentAutoma
# ---------------------------------------------------------------------------

class _SimpleWorker(CognitiveWorker):
    async def thinking(self):
        return "Execute one step."


def _make_sequential_loop_agent(llm, seq_responses, loop_responses):
    """Create an agent that uses sequential() then loop()."""
    worker = _SimpleWorker(llm=llm)

    class _StructuredAgent(AgentAutoma[CognitiveContext]):
        async def cognition(self, ctx: CognitiveContext) -> None:
            await self.sequential(worker, name="setup", max_attempts=len(seq_responses))
            await self.loop(worker, name="process", max_attempts=len(loop_responses))

    return _StructuredAgent(llm=llm)


class TestStructuredCapture:

    @pytest.mark.asyncio
    async def test_sequential_loop_produces_structured_workflow(self):
        """sequential() + loop() produces LinearTraceBlock + LoopBlock."""
        # 2 sequential steps + 4 loop steps (2 iterations of click→save)
        seq_steps = [
            _tr(step_content="Navigate", output=[_make_tool_call("search_flights", origin="A", destination="B")], finish=False),
            _tr(step_content="Filter", output=[_make_tool_call("search_hotels", city="B", check_in="2024-01-01", check_out="2024-01-02")], finish=True),
        ]
        loop_steps = [
            _tr(step_content="Click 1", output=[_make_tool_call("search_flights", origin="X", destination="Y")], finish=False),
            _tr(step_content="Save 1", output=[_make_tool_call("search_hotels", city="Y", check_in="2024-02-01", check_out="2024-02-02")], finish=False),
            _tr(step_content="Click 2", output=[_make_tool_call("search_flights", origin="P", destination="Q")], finish=False),
            _tr(step_content="Save 2", output=[_make_tool_call("search_hotels", city="Q", check_in="2024-03-01", check_out="2024-03-02")], finish=True),
        ]

        llm = StatefulMockLLM(seq_steps + loop_steps)
        agent = _make_sequential_loop_agent(llm, seq_steps, loop_steps)
        ctx = _make_ctx()

        result = await agent.arun(context=ctx, capture_workflow=True)
        assert isinstance(result, tuple)
        ctx_out, workflow = result

        assert isinstance(workflow, Workflow)
        assert len(workflow.blocks) == 2

        # First block: LinearTraceBlock from sequential()
        assert isinstance(workflow.blocks[0], LinearTraceBlock)
        assert workflow.blocks[0].name == "setup"
        assert len(workflow.blocks[0].steps) == 2

        # Second block: LoopBlock from loop()
        assert isinstance(workflow.blocks[1], LoopBlock)
        assert workflow.blocks[1].name == "process"

    @pytest.mark.asyncio
    async def test_plain_run_still_produces_flat_workflow(self):
        """Agent using only self.run() still produces a flat LinearTraceBlock (backward compat)."""
        step1 = _tr(
            step_content="Search flights",
            output=[_make_tool_call("search_flights", origin="A", destination="B")],
            finish=False,
        )
        step2 = _tr(
            step_content="Search hotels",
            output=[_make_tool_call("search_hotels", city="B", check_in="2024-01-01", check_out="2024-01-02")],
            finish=True,
        )
        llm = StatefulMockLLM([step1, step2])
        worker = _SimpleWorker(llm=llm)

        class _FlatAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx: CognitiveContext) -> None:
                await self.run(worker, name="step", max_attempts=5)

        agent = _FlatAgent(llm=llm)
        ctx = _make_ctx()

        _, workflow = await agent.arun(context=ctx, capture_workflow=True)

        # Should fall back to flat LinearTraceBlock since no sequential/loop used
        assert len(workflow.blocks) == 1
        assert isinstance(workflow.blocks[0], LinearTraceBlock)
        assert workflow.blocks[0].name == "step"

    @pytest.mark.asyncio
    async def test_loop_block_has_pattern_template(self):
        """LoopBlock from loop() phase has a non-empty pattern_template."""
        # 4 loop steps: 2 iterations of search_flights → search_hotels
        loop_steps = [
            _tr(step_content="s1", output=[_make_tool_call("search_flights", origin="A", destination="B", date="2024-01-01")], finish=False),
            _tr(step_content="s2", output=[_make_tool_call("search_hotels", city="B", check_in="2024-01-01", check_out="2024-01-02")], finish=False),
            _tr(step_content="s3", output=[_make_tool_call("search_flights", origin="C", destination="D", date="2024-02-01")], finish=False),
            _tr(step_content="s4", output=[_make_tool_call("search_hotels", city="D", check_in="2024-02-01", check_out="2024-02-02")], finish=True),
        ]

        llm = StatefulMockLLM(loop_steps)
        worker = _SimpleWorker(llm=llm)

        class _LoopOnlyAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx: CognitiveContext) -> None:
                await self.loop(worker, name="process", max_attempts=10)

        agent = _LoopOnlyAgent(llm=llm)
        ctx = _make_ctx()

        _, workflow = await agent.arun(context=ctx, capture_workflow=True)

        assert len(workflow.blocks) == 1
        block = workflow.blocks[0]
        assert isinstance(block, LoopBlock)
        assert block.pattern_template == ["search_flights", "search_hotels"]
        assert len(block.iterations) == 2

    @pytest.mark.asyncio
    async def test_workflow_metadata_includes_agent_class(self):
        """Structured workflow metadata includes agent_class and steps_count."""
        step = _tr(
            step_content="Done",
            output=[_make_tool_call("search_flights", origin="A", destination="B")],
            finish=True,
        )
        llm = StatefulMockLLM([step])
        worker = _SimpleWorker(llm=llm)

        class _MetaAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx: CognitiveContext) -> None:
                await self.sequential(worker, name="only", max_attempts=1)

        agent = _MetaAgent(llm=llm)
        ctx = _make_ctx()

        _, workflow = await agent.arun(context=ctx, capture_workflow=True)

        assert "agent_class" in workflow.metadata
        assert workflow.metadata["agent_class"] == "_MetaAgent"
        assert "steps_count" in workflow.metadata


# ---------------------------------------------------------------------------
# LoopBlock serialization
# ---------------------------------------------------------------------------

class TestLoopBlockSerialization:

    def test_loop_block_with_pattern_template_roundtrip(self):
        block = LoopBlock(
            name="process",
            pattern_template=["click", "save", "close"],
            iterations=[[{"tool_calls": [{"tool_name": "click"}]}]],
            max_attempts=30,
        )
        data = block.model_dump()
        assert data["pattern_template"] == ["click", "save", "close"]

        restored = LoopBlock(**data)
        assert restored.pattern_template == ["click", "save", "close"]
        assert restored.name == "process"
        assert restored.max_attempts == 30
