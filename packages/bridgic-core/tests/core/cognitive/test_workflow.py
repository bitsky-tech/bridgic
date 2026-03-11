"""Tests for workflow capture and replay (WorkflowStepWorker, capture_workflow)."""
import types
import pytest
from typing import Any, List

from bridgic.core.automa import GraphAutoma
from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    WorkflowToolCall,
    WorkflowStepWorker,
    Workflow,
)
from bridgic.core.cognitive._cognitive_worker import StepToolCall, ToolArgument
from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(**kwargs):
    """Build a SimpleNamespace mock for a LLM structured-output response."""
    defaults = {"step_content": "", "output": [], "finish": False, "details": []}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class StatefulMockLLM:
    """Returns a fixed sequence of responses from astructured_output."""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ctx() -> CognitiveContext:
    ctx = CognitiveContext(goal="Find flights and hotels")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    return ctx


def _make_llm_two_steps() -> StatefulMockLLM:
    """LLM that drives two think steps then stops."""
    step1 = _tr(
        step_content="Search for flights to Tokyo",
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
    step2 = _tr(
        step_content="Search for hotels in Tokyo",
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
        finish=True,
    )
    return StatefulMockLLM([step1, step2])


# ---------------------------------------------------------------------------
# Agent factory (fresh worker per call avoids shared-state cross-test issues)
# ---------------------------------------------------------------------------

class _SimpleWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE immediate next step."


def _make_agent(llm) -> AgentAutoma:
    """Return a fresh AgentAutoma instance with a brand-new worker for each test."""
    worker_instance = _SimpleWorker(llm=llm)

    class _TravelAgent(AgentAutoma[CognitiveContext]):
        async def cognition(self, ctx: CognitiveContext) -> None:
            await self.run(worker_instance, name="step", max_attempts=5)

    return _TravelAgent(llm=llm)


# ---------------------------------------------------------------------------
# Tests — capture_workflow (new Workflow type)
# ---------------------------------------------------------------------------

class TestCaptureWorkflow:

    @pytest.mark.asyncio
    async def test_returns_tuple_when_capture_workflow(self):
        """arun(capture_workflow=True) → (ctx, Workflow)."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        result = await agent.arun(context=ctx, capture_workflow=True)

        assert isinstance(result, tuple), "Expected (ctx, workflow) tuple"
        ctx_out, workflow = result
        assert isinstance(ctx_out, CognitiveContext)
        assert isinstance(workflow, Workflow)

    @pytest.mark.asyncio
    async def test_workflow_has_correct_number_of_blocks(self):
        """Workflow should contain blocks with all recorded steps."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        ctx_out, workflow = await agent.arun(context=ctx, capture_workflow=True)

        # Two steps in history
        history_len = len(ctx_out.cognitive_history.get_all())
        assert history_len == 2

        # Workflow should have at least one block
        assert len(workflow.blocks) >= 1

        # Total steps across all blocks should match
        total_steps = 0
        for block in workflow.blocks:
            if hasattr(block, 'steps'):
                total_steps += len(block.steps)
        assert total_steps == 2

    @pytest.mark.asyncio
    async def test_workflow_blocks_have_recorded_tool_calls(self):
        """Workflow blocks contain the correct recorded tool calls."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        _, workflow = await agent.arun(context=ctx, capture_workflow=True)

        # Get all steps from blocks
        all_steps = []
        for block in workflow.blocks:
            if hasattr(block, 'steps'):
                all_steps.extend(block.steps)

        assert len(all_steps) == 2

        # Step 0 — search_flights
        step0_calls = all_steps[0].get("tool_calls", [])
        assert len(step0_calls) == 1
        assert step0_calls[0]["tool_name"] == "search_flights"
        assert step0_calls[0]["tool_arguments"]["origin"] == "Beijing"

        # Step 1 — search_hotels
        step1_calls = all_steps[1].get("tool_calls", [])
        assert len(step1_calls) == 1
        assert step1_calls[0]["tool_name"] == "search_hotels"
        assert step1_calls[0]["tool_arguments"]["city"] == "Tokyo"

    @pytest.mark.asyncio
    async def test_normal_arun_returns_context_not_tuple(self):
        """Without capture_workflow, arun() returns context directly."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        result = await agent.arun(context=ctx)

        assert isinstance(result, CognitiveContext)

    @pytest.mark.asyncio
    async def test_step_metadata_has_action_results(self):
        """After a live run, each step's metadata contains action_results with tool details."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        ctx_out = await agent.arun(context=ctx)

        steps = ctx_out.cognitive_history.get_all()
        assert len(steps) == 2

        # Step 0 — search_flights
        meta0 = steps[0].metadata
        assert "action_results" in meta0
        assert len(meta0["action_results"]) == 1
        assert meta0["action_results"][0]["tool_name"] == "search_flights"
        assert "tool_arguments" in meta0["action_results"][0]
        assert "tool_result" in meta0["action_results"][0]

        # Step 1 — search_hotels
        meta1 = steps[1].metadata
        assert "action_results" in meta1
        assert meta1["action_results"][0]["tool_name"] == "search_hotels"


# ---------------------------------------------------------------------------
# Tests — new API (self.run)
# ---------------------------------------------------------------------------

class TestNewRunAPI:

    @pytest.mark.asyncio
    async def test_new_api_runs_agent(self):
        """Agent using self.run() completes successfully."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        result = await agent.arun(context=ctx)

        assert isinstance(result, CognitiveContext)
        steps = result.cognitive_history.get_all()
        assert len(steps) == 2
        assert steps[0].metadata["action_results"][0]["tool_name"] == "search_flights"
        assert steps[1].metadata["action_results"][0]["tool_name"] == "search_hotels"

    @pytest.mark.asyncio
    async def test_new_api_capture_workflow(self):
        """New API agent also supports capture_workflow."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        result = await agent.arun(context=ctx, capture_workflow=True)

        assert isinstance(result, tuple)
        ctx_out, workflow = result
        assert isinstance(ctx_out, CognitiveContext)
        assert isinstance(workflow, Workflow)
        assert len(workflow.blocks) >= 1


# ---------------------------------------------------------------------------
# Tests — serialization round-trip
# ---------------------------------------------------------------------------

class TestWorkflowSerialization:

    @pytest.mark.asyncio
    async def test_workflow_model_dump_and_restore(self):
        """Workflow model_dump → Workflow(**data) preserves all blocks."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        ctx = _make_ctx()

        _, workflow = await agent.arun(context=ctx, capture_workflow=True)

        # Serialize
        data = workflow.model_dump()
        assert isinstance(data, dict)

        # Deserialize
        restored = Workflow(**data)
        assert len(restored.blocks) == len(workflow.blocks)

        # Verify step data preserved
        orig_steps = []
        for block in workflow.blocks:
            if hasattr(block, 'steps'):
                orig_steps.extend(block.steps)

        restored_steps = []
        for block in restored.blocks:
            if hasattr(block, 'steps'):
                restored_steps.extend(block.steps)

        assert len(orig_steps) == len(restored_steps)
        assert orig_steps[0]["tool_calls"][0]["tool_name"] == "search_flights"
        assert restored_steps[0]["tool_calls"][0]["tool_name"] == "search_flights"

    def test_workflow_tool_call_model_dump(self):
        """WorkflowToolCall serializes and deserializes via model_dump."""
        tc = WorkflowToolCall(
            tool_name="search_flights",
            tool_arguments={"origin": "Beijing", "destination": "Tokyo"},
            tool_result="3 flights found",
        )
        data = tc.model_dump()
        restored = WorkflowToolCall(**data)
        assert restored.tool_name == tc.tool_name
        assert restored.tool_arguments == tc.tool_arguments
        assert restored.tool_result == tc.tool_result

    def test_workflow_step_worker_dump_load(self):
        """WorkflowStepWorker dump_to_dict / load_from_dict preserves state."""
        tc1 = WorkflowToolCall(
            tool_name="search_flights",
            tool_arguments={"origin": "A", "destination": "B"},
        )
        worker = WorkflowStepWorker(tool_calls=[tc1], step_content="Step content")

        state = worker.dump_to_dict()
        assert state["step_content"] == "Step content"
        assert len(state["tool_calls"]) == 1
        assert state["tool_calls"][0]["tool_name"] == "search_flights"

        restored = WorkflowStepWorker()
        restored.load_from_dict(state)
        assert restored.step_content == "Step content"
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].tool_name == "search_flights"
        assert restored.tool_calls[0].tool_arguments == {"origin": "A", "destination": "B"}


# ---------------------------------------------------------------------------
# Tests — WorkflowStepWorker replay (these still work as before)
# ---------------------------------------------------------------------------

class TestWorkflowReplay:

    @pytest.mark.asyncio
    async def test_replay_step_worker_records_steps(self):
        """WorkflowStepWorker fills ctx.cognitive_history with replayed steps."""
        llm = _make_llm_two_steps()
        agent = _make_agent(llm)
        original_ctx = _make_ctx()

        # Run agent to generate history
        await agent.arun(context=original_ctx)
        history = original_ctx.cognitive_history.get_all()

        # Build a GraphAutoma from history (like the old workflow)
        workflow = GraphAutoma()
        for i, step in enumerate(history):
            raw = step.metadata.get("action_results", [])
            tool_calls = [
                WorkflowToolCall(
                    tool_name=r["tool_name"],
                    tool_arguments=r["tool_arguments"],
                )
                for r in raw
            ]
            step_worker = WorkflowStepWorker(
                tool_calls=tool_calls,
                step_content=step.content,
            )
            from bridgic.core.automa.args import ArgsMappingRule
            workflow.add_worker(
                key=f"step_{i}",
                worker=step_worker,
                dependencies=[] if i == 0 else [f"step_{i - 1}"],
                is_start=(i == 0),
                is_output=(i == len(history) - 1),
                args_mapping_rule=ArgsMappingRule.AS_IS,
            )

        # Replay
        replay_ctx = _make_ctx()
        result = await workflow.arun(context=replay_ctx)

        assert result is replay_ctx
        steps = replay_ctx.cognitive_history.get_all()
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_replay_calls_correct_tools(self):
        """Replay calls the same tools with the same arguments."""
        tc1 = WorkflowToolCall(
            tool_name="search_flights",
            tool_arguments={"origin": "Beijing", "destination": "Tokyo", "date": "2024-06-01"},
        )
        tc2 = WorkflowToolCall(
            tool_name="search_hotels",
            tool_arguments={"city": "Tokyo", "check_in": "2024-06-01", "check_out": "2024-06-05"},
        )
        from bridgic.core.automa.args import ArgsMappingRule

        workflow = GraphAutoma()
        w0 = WorkflowStepWorker(tool_calls=[tc1], step_content="Search flights")
        w1 = WorkflowStepWorker(tool_calls=[tc2], step_content="Search hotels")
        workflow.add_worker(key="step_0", worker=w0, is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
        workflow.add_worker(key="step_1", worker=w1, dependencies=["step_0"], is_output=True, args_mapping_rule=ArgsMappingRule.AS_IS)

        replay_ctx = _make_ctx()
        await workflow.arun(context=replay_ctx)

        steps = replay_ctx.cognitive_history.get_all()
        assert len(steps) == 2

        meta0 = steps[0].metadata
        assert "replayed" in meta0 and meta0["replayed"] is True
        assert "search_flights" in meta0["tool_calls"]
        ar0 = meta0["action_results"][0]
        assert ar0["tool_name"] == "search_flights"
        assert "Tokyo" in ar0["tool_result"]

        meta1 = steps[1].metadata
        assert "search_hotels" in meta1["tool_calls"]
        ar1 = meta1["action_results"][0]
        assert ar1["tool_name"] == "search_hotels"
        assert "Tokyo" in ar1["tool_result"]

    @pytest.mark.asyncio
    async def test_replay_raises_if_tool_missing(self):
        """Replay raises ValueError when a recorded tool is absent from context."""
        tc = WorkflowToolCall(
            tool_name="nonexistent_tool",
            tool_arguments={"x": "y"},
        )
        step_worker = WorkflowStepWorker(tool_calls=[tc], step_content="ghost step")

        workflow = GraphAutoma()
        workflow.add_worker(
            key="step_0",
            worker=step_worker,
            is_start=True,
            is_output=True,
        )

        empty_ctx = CognitiveContext(goal="test")
        with pytest.raises(ValueError, match="Replay: tool 'nonexistent_tool' not found"):
            await workflow.arun(context=empty_ctx)

    @pytest.mark.asyncio
    async def test_replay_step_with_no_tool_calls(self):
        """A step with no tool calls records a step with empty tool_calls in history."""
        step_worker = WorkflowStepWorker(tool_calls=[], step_content="thinking step")
        workflow = GraphAutoma()
        workflow.add_worker(
            key="step_0",
            worker=step_worker,
            is_start=True,
            is_output=True,
        )

        ctx = CognitiveContext(goal="test")
        result = await workflow.arun(context=ctx)

        steps = result.cognitive_history.get_all()
        assert len(steps) == 1
        assert steps[0].content == "thinking step"
        assert steps[0].metadata["tool_calls"] == []
        assert steps[0].metadata["replayed"] is True
