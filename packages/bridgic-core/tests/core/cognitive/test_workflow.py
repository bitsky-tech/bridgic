"""Tests for Workflow module: WorkflowBuilder, loop probe mode, etc."""
import pytest
from typing import Any, List

from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    ThinkDecision,
)
from bridgic.core.cognitive._cognitive_worker import StepToolCall, ToolArgument
from bridgic.core.cognitive._workflow import WorkflowBuilder, Workflow, _detect_iteration_boundary
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


def _make_step_data(tool_name: str) -> dict:
    """Create a minimal step data dict for WorkflowBuilder."""
    return {
        "name": "TestWorker",
        "step_content": f"Step using {tool_name}",
        "tool_calls": [{"tool_name": tool_name, "tool_arguments": {}, "tool_result": "ok"}],
        "finished": False,
        "observation_hash": None,
        "output_type": "tool_calls",
        "structured_output": None,
        "structured_output_class": None,
    }


# ---------------------------------------------------------------------------
# Tests — _detect_iteration_boundary
# ---------------------------------------------------------------------------

class TestDetectIterationBoundary:

    def test_empty_steps(self):
        steps, iters = _detect_iteration_boundary([])
        assert steps == []
        assert iters == 0

    def test_single_step(self):
        steps = [_make_step_data("search_flights")]
        result, iters = _detect_iteration_boundary(steps)
        assert iters == 1

    def test_two_identical_steps(self):
        """Two identical tool signatures → 2 iterations of period 1."""
        steps = [_make_step_data("search_flights"), _make_step_data("search_flights")]
        result, iters = _detect_iteration_boundary(steps)
        assert iters == 2
        assert len(result) == 1

    def test_repeating_pattern_of_two(self):
        """Pattern [A, B, A, B] → 2 iterations of [A, B]."""
        steps = [
            _make_step_data("search_flights"),
            _make_step_data("search_hotels"),
            _make_step_data("search_flights"),
            _make_step_data("search_hotels"),
        ]
        result, iters = _detect_iteration_boundary(steps)
        assert iters == 2
        assert len(result) == 2

    def test_no_clean_repetition(self):
        """[A, B, C] — no clean repetition → 1 iteration."""
        steps = [
            _make_step_data("search_flights"),
            _make_step_data("search_hotels"),
            _make_step_data("book_flight"),
        ]
        result, iters = _detect_iteration_boundary(steps)
        assert iters == 1
        assert len(result) == 3



# ---------------------------------------------------------------------------
# Tests — WorkflowBuilder.is_loop_pattern_confirmed
# ---------------------------------------------------------------------------

class TestIsLoopPatternConfirmed:

    def test_empty_stack_returns_false(self):
        builder = WorkflowBuilder()
        assert builder.is_loop_pattern_confirmed() is False

    def test_sequential_phase_returns_false(self):
        builder = WorkflowBuilder()
        builder.begin_phase("sequential")
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_flights"))
        assert builder.is_loop_pattern_confirmed() is False

    def test_loop_with_one_step_returns_false(self):
        builder = WorkflowBuilder()
        builder.begin_phase("loop")
        builder.record_step(_make_step_data("search_flights"))
        assert builder.is_loop_pattern_confirmed() is False

    def test_loop_with_two_identical_steps_returns_true(self):
        """Two identical tool signatures in a loop → pattern confirmed."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop")
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_flights"))
        assert builder.is_loop_pattern_confirmed() is True

    def test_loop_with_repeating_pair(self):
        """[A, B, A, B] in a loop → pattern confirmed."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop")
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_hotels"))
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_hotels"))
        assert builder.is_loop_pattern_confirmed() is True

    def test_loop_no_repetition_returns_false(self):
        """[A, B, C] in a loop — no clean repetition → not confirmed."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop")
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_hotels"))
        builder.record_step(_make_step_data("book_flight"))
        assert builder.is_loop_pattern_confirmed() is False

    def test_custom_min_iterations(self):
        """min_iterations=3 requires 3 repetitions."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop")
        # 2 repetitions
        builder.record_step(_make_step_data("search_flights"))
        builder.record_step(_make_step_data("search_flights"))
        assert builder.is_loop_pattern_confirmed(min_iterations=3) is False
        # 3 repetitions
        builder.record_step(_make_step_data("search_flights"))
        assert builder.is_loop_pattern_confirmed(min_iterations=3) is True



# ---------------------------------------------------------------------------
# Tests — Integration: probe mode early termination in run()
# ---------------------------------------------------------------------------

class TestLoopProbeMode:

    @pytest.mark.asyncio
    async def test_run_terminates_early_in_loop_probe_mode(self):
        """When capture_workflow=True inside a loop phase, run() stops
        early once KMP detects a repeating pattern."""
        call_count = 0

        # Each call returns the same tool call → identical signature each time
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

        class CountingLLM(MockLLM):
            def __init__(self):
                super().__init__([_make_search_step()])

            async def astructured_output(self, messages, constraint, **kwargs):
                nonlocal call_count
                call_count += 1
                return await super().astructured_output(messages, constraint, **kwargs)

        llm = CountingLLM()

        class ProbeAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                worker = CognitiveWorker.inline("Repeat step", llm=self.llm)
                async with self.loop(goal="process items"):
                    await self.run(worker, max_attempts=80)

        agent = ProbeAgent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, capture_workflow=True)

        # Should have stopped well before 80 iterations.
        # KMP detects pattern after 2 identical steps, so we expect ~2 calls.
        assert call_count <= 5, (
            f"Expected early termination but got {call_count} LLM calls"
        )
        assert call_count >= 2, (
            f"Expected at least 2 calls to confirm pattern, got {call_count}"
        )

    @pytest.mark.asyncio
    async def test_no_early_termination_without_capture(self):
        """Without capture_workflow=True, run() does NOT terminate early."""
        call_count = 0

        def _make_step(finish=False):
            return _tr(
                step_content="Search",
                output=[
                    StepToolCall(
                        tool="search_flights",
                        tool_arguments=[
                            ToolArgument(name="origin", value="A"),
                            ToolArgument(name="destination", value="B"),
                            ToolArgument(name="date", value="2024-01-01"),
                        ],
                    )
                ],
                finish=finish,
            )

        class CountingLLM(MockLLM):
            def __init__(self):
                # 4 non-finish steps, then finish
                super().__init__([
                    _make_step(False),
                    _make_step(False),
                    _make_step(False),
                    _make_step(False),
                    _make_step(True),
                ])

            async def astructured_output(self, messages, constraint, **kwargs):
                nonlocal call_count
                call_count += 1
                return await super().astructured_output(messages, constraint, **kwargs)

        llm = CountingLLM()

        class NormalAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                worker = CognitiveWorker.inline("Repeat step", llm=self.llm)
                await self.run(worker, max_attempts=10)

        agent = NormalAgent(llm=llm)
        ctx = _make_ctx()
        await agent.arun(context=ctx, capture_workflow=False)

        # Without capture, should run until finish=True (5 calls)
        assert call_count == 5


# ---------------------------------------------------------------------------
# Helpers for code-driven loop tests
# ---------------------------------------------------------------------------

def _make_step_data_with_obs(tool_name: str, obs_text: str = "page content", args: dict = None) -> dict:
    """Create a step data dict with observation_text and tool_arguments."""
    return {
        "name": "TestWorker",
        "step_content": f"Step using {tool_name}",
        "tool_calls": [{
            "tool_name": tool_name,
            "tool_arguments": args or {},
            "tool_result": "ok",
        }],
        "finished": False,
        "observation_hash": None,
        "observation_text": obs_text,
        "output_type": "tool_calls",
        "structured_output": None,
        "structured_output_class": None,
    }


# ---------------------------------------------------------------------------
# Tests — build() stores example_iterations
# ---------------------------------------------------------------------------

class TestBuildStoresExampleIterations:

    def test_build_stores_example_iterations(self):
        """Loop with 2 iterations should populate code_slot.example_iterations."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop", goal="process items")
        builder.set_run_config({
            "worker_class": "test.Worker",
            "max_attempts": 10,
        })
        # Iteration 1
        builder.record_step(_make_step_data_with_obs(
            "click_element", obs_text="page with item 1",
            args={"selector": "#item-1"},
        ))
        builder.record_step(_make_step_data_with_obs(
            "save_information", obs_text="detail page 1",
            args={"data": "info1"},
        ))
        # Iteration 2
        builder.record_step(_make_step_data_with_obs(
            "click_element", obs_text="page with item 2",
            args={"selector": "#item-2"},
        ))
        builder.record_step(_make_step_data_with_obs(
            "save_information", obs_text="detail page 2",
            args={"data": "info2"},
        ))
        builder.end_phase()

        wf = builder.build()
        assert len(wf.blocks) == 1
        loop_block = wf.blocks[0]
        assert loop_block.block_type == "loop"
        assert loop_block.code_slot.example_iterations is not None
        assert len(loop_block.code_slot.example_iterations) == 2

        # Check first iteration sample
        iter0 = loop_block.code_slot.example_iterations[0]
        assert len(iter0) == 2
        assert iter0[0]["tool_name"] == "click_element"
        assert iter0[0]["tool_arguments"] == {"selector": "#item-1"}
        assert iter0[0]["observation_text"] == "page with item 1"

        # Check second iteration sample
        iter1 = loop_block.code_slot.example_iterations[1]
        assert iter1[0]["tool_name"] == "click_element"
        assert iter1[0]["tool_arguments"] == {"selector": "#item-2"}

    def test_build_no_examples_for_single_iteration(self):
        """Loop with only 1 iteration should not have example_iterations."""
        builder = WorkflowBuilder()
        builder.begin_phase("loop", goal="one-shot")
        builder.set_run_config({"worker_class": "test.Worker"})
        builder.record_step(_make_step_data_with_obs("click_element"))
        builder.record_step(_make_step_data_with_obs("save_information"))
        builder.record_step(_make_step_data_with_obs("book_flight"))
        builder.end_phase()

        wf = builder.build()
        loop_block = wf.blocks[0]
        assert loop_block.code_slot.example_iterations is None


# ---------------------------------------------------------------------------
# Tests — _execute_code_slot safety
# ---------------------------------------------------------------------------

class TestExecuteCodeSlot:

    def test_safe_code_executes(self):
        """Safe code defining get_items and fill_step_args should work."""
        code = (
            "def get_items(observation):\n"
            "    return [{'id': '1'}, {'id': '2'}]\n"
            "\n"
            "def fill_step_args(step_idx, tool_name, observation, item):\n"
            "    return {'selector': f'#item-{item[\"id\"]}'}\n"
        )
        ns = Workflow._execute_code_slot(code)
        assert callable(ns["get_items"])
        assert callable(ns["fill_step_args"])

        items = ns["get_items"]("some page")
        assert len(items) == 2

        args = ns["fill_step_args"](0, "click", "obs", {"id": "3"})
        assert args == {"selector": "#item-3"}

    def test_blocks_import(self):
        """Code containing __import__ should be blocked."""
        code = "x = __import__('os')"
        with pytest.raises(RuntimeError, match="Forbidden"):
            Workflow._execute_code_slot(code)

    def test_blocks_open(self):
        """Code containing open() should be blocked."""
        code = "f = open('/etc/passwd')"
        with pytest.raises(RuntimeError, match="Forbidden"):
            Workflow._execute_code_slot(code)

    def test_blocks_import_os(self):
        """Code importing os should be blocked."""
        code = "import os\nos.system('rm -rf /')"
        with pytest.raises(RuntimeError, match="Forbidden"):
            Workflow._execute_code_slot(code)

    def test_re_and_json_available(self):
        """re and json modules should be available in the namespace."""
        code = (
            "import_test = re.compile(r'\\d+')\n"
            "parsed = json.loads('{\"a\": 1}')\n"
            "def get_items(obs): return [parsed]\n"
            "def fill_step_args(si, tn, obs, item): return item\n"
        )
        ns = Workflow._execute_code_slot(code)
        items = ns["get_items"]("")
        assert items == [{"a": 1}]


# ---------------------------------------------------------------------------
# Tests — _replay_loop_with_code happy path
# ---------------------------------------------------------------------------

class TestReplayLoopWithCode:

    @pytest.mark.asyncio
    async def test_replay_loop_with_code_happy_path(self):
        """Code-driven loop should complete without extra LLM calls."""
        llm_call_count = 0

        class TrackingLLM(MockLLM):
            def __init__(self):
                super().__init__([])

            async def astructured_output(self, messages, constraint, **kwargs):
                nonlocal llm_call_count
                llm_call_count += 1
                return await super().astructured_output(messages, constraint, **kwargs)

        llm = TrackingLLM()

        # Build a workflow with pre-existing generated code
        from bridgic.core.cognitive._workflow import (
            LoopBlock, LoopCodeSlot, RunConfig, TraceStep, RecordedToolCall,
            StepOutputType,
        )

        code = (
            "def get_items(observation):\n"
            "    return [{'id': '1', 'name': 'Item 1'}, {'id': '2', 'name': 'Item 2'}]\n"
            "\n"
            "def fill_step_args(step_idx, tool_name, observation, item):\n"
            "    if tool_name == 'click_element':\n"
            "        return {'selector': f'#item-{item[\"id\"]}'}\n"
            "    elif tool_name == 'save_information':\n"
            "        return {'data': item['name']}\n"
            "    return {}\n"
        )

        loop_block = LoopBlock(
            goal="process items",
            run_config=RunConfig(
                worker_class="bridgic.core.cognitive._cognitive_worker.CognitiveWorker",
                worker_thinking_prompt="Process items",
                max_attempts=10,
            ),
            body_steps=[
                TraceStep(
                    name="Worker",
                    step_content="Click item",
                    tool_calls=[RecordedToolCall(
                        tool_name="click_element",
                        tool_arguments={"selector": "#item-1"},
                        tool_result="ok",
                    )],
                    output_type=StepOutputType.TOOL_CALLS,
                ),
                TraceStep(
                    name="Worker",
                    step_content="Save info",
                    tool_calls=[RecordedToolCall(
                        tool_name="save_information",
                        tool_arguments={"data": "info1"},
                        tool_result="ok",
                    )],
                    output_type=StepOutputType.TOOL_CALLS,
                ),
            ],
            observed_iterations=2,
            code_slot=LoopCodeSlot(
                slot_id="test-slot",
                description="process items",
                generated_code=code,
            ),
        )

        workflow = Workflow(blocks=[loop_block])

        # Create agent and context
        class TestAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                pass

        agent = TestAgent(llm=llm)
        ctx = _make_ctx()

        # Run replay
        await workflow.arun(agent, ctx)

        # Should have zero LLM calls (all handled by code)
        assert llm_call_count == 0

    @pytest.mark.asyncio
    async def test_replay_loop_code_fallback(self):
        """When generated code fails, should fall back to adaptive fill."""
        from bridgic.core.cognitive._workflow import (
            LoopBlock, LoopCodeSlot, RunConfig, TraceStep, RecordedToolCall,
            StepOutputType, _LoopContinueDecision,
        )

        adaptive_call_count = 0

        class FallbackLLM(MockLLM):
            def __init__(self):
                super().__init__([])

            async def astructured_output(self, messages, constraint, **kwargs):
                nonlocal adaptive_call_count
                adaptive_call_count += 1

                # Check what model is being requested
                model_cls = getattr(constraint, 'model', None)
                if model_cls is _LoopContinueDecision:
                    return _LoopContinueDecision(should_continue=False, reason="done")

                # For tool arg fill, return a dynamic model instance
                if model_cls is not None:
                    return model_cls(selector="#fallback")

                return None

        llm = FallbackLLM()

        # Bad code that will raise
        bad_code = "def get_items(obs): raise ValueError('broken')\ndef fill_step_args(*a): pass"

        loop_block = LoopBlock(
            goal="process items",
            run_config=RunConfig(
                worker_class="bridgic.core.cognitive._cognitive_worker.CognitiveWorker",
                worker_thinking_prompt="Process items",
                max_attempts=2,
            ),
            body_steps=[
                TraceStep(
                    name="Worker",
                    step_content="Click item",
                    tool_calls=[RecordedToolCall(
                        tool_name="click_element",
                        tool_arguments={"selector": "#item-1"},
                        tool_result="ok",
                    )],
                    output_type=StepOutputType.TOOL_CALLS,
                ),
            ],
            observed_iterations=2,
            code_slot=LoopCodeSlot(
                slot_id="test-slot",
                description="process items",
                generated_code=bad_code,
            ),
        )

        workflow = Workflow(blocks=[loop_block])

        class TestAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                pass

        agent = TestAgent(llm=llm)
        ctx = _make_ctx()

        await workflow.arun(agent, ctx)

        # Should have fallen back to adaptive fill, meaning LLM calls happened
        assert adaptive_call_count > 0
