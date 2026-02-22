"""Tests for CognitiveWorker: template methods, type checks, full observe-think-act cycle."""
import os
import pytest
from unittest.mock import MagicMock
from typing import Any, List, Optional, Tuple

from bridgic.core.cognitive import (
    CognitiveContext,
    CognitiveWorker,
    ThinkingMode,
    FastThinkResult,
    DefaultThinkResult,
    ActionStepResult,
    StepToolCall,
    ToolArgument,
)
from bridgic.core.model.types import ToolCall
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self):
        self.structured_output_response = None
        self.select_tool_response = ([], None)
        self.captured_messages: List[Any] = []

    async def astructured_output(self, messages, constraint, **kwargs):
        self.captured_messages = messages
        return self.structured_output_response

    async def aselect_tool(self, messages, tools, **kwargs):
        self.captured_messages = messages
        return self.select_tool_response

    async def achat(self, messages, **kwargs):
        return MagicMock()

    async def astream(self, messages, **kwargs):
        return MagicMock()

    def chat(self, messages, **kwargs):
        return MagicMock()

    def stream(self, messages, **kwargs):
        return MagicMock()


def _make_context() -> CognitiveContext:
    ctx = CognitiveContext(goal="Plan a trip to Tokyo")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    ctx.skills.load_from_directory(SKILLS_DIR)
    return ctx


# ---------------------------------------------------------------------------
# Custom workers
# ---------------------------------------------------------------------------

class _FastWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE immediate next step."


class _DefaultWorker(CognitiveWorker):
    async def thinking(self):
        return "Create a step-by-step plan."


class _PromptCustomWorker(CognitiveWorker):
    """Worker with observation + build_thinking_prompt overrides."""
    async def thinking(self):
        return "Plan ONE step"

    async def observation(self, context):
        return "Custom observation: environment is ready"

    async def build_thinking_prompt(self, think_prompt, tools_description, output_instructions, context_info):
        extra = "EXTRA_INSTRUCTION: Always prefer cheapest option."
        system = f"{think_prompt}\n\n{extra}\n\n{tools_description}\n\n{output_instructions}"
        return system, context_info


class _ActionPipelineWorker(CognitiveWorker):
    """Worker with verify_tools + consequence overrides."""
    async def thinking(self):
        return "Plan ONE step"

    async def verify_tools(self, matched_list, context):
        return [(tc, spec) for tc, spec in matched_list if spec.tool_name != "book_flight"]

    async def consequence(self, action_results):
        return "; ".join(r.tool_result for r in action_results)


class _OverrideSelectToolsWorker(CognitiveWorker):
    """Worker with custom select_tools (DEFAULT mode only)."""
    async def thinking(self):
        return "Plan steps"

    async def select_tools(self, step_content, observation, context):
        return [ToolCall(id="fixed_1", name="search_flights", arguments={
            "origin": "Beijing", "destination": "Tokyo", "date": "2025-06-01"
        })]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveWorker:

    @pytest.mark.asyncio
    async def test_template_method_defaults(self):
        """Default template methods: thinking raises, observation/consequence/verify_tools
        passthrough, build_thinking_prompt assembles, non-context rejected."""
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()

        # thinking() → NotImplementedError
        with pytest.raises(NotImplementedError):
            await worker.thinking()

        # observation() → None
        assert await worker.observation(ctx) is None

        # consequence() → passthrough
        test_results = [
            ActionStepResult(
                tool_id="1", tool_name="search_flights",
                tool_arguments={"origin": "Beijing"}, tool_result="found"
            )
        ]
        assert await worker.consequence(test_results) is test_results

        # verify_tools() → passthrough
        tools = get_travel_planning_tools()
        matched = [(ToolCall(id="1", name="search_flights", arguments={}), tools[0])]
        assert await worker.verify_tools(matched, ctx) is matched

        # build_thinking_prompt() → standard assembly
        system, user = await worker.build_thinking_prompt(
            think_prompt="Plan next step",
            tools_description="Tool A, Tool B",
            output_instructions="Output JSON",
            context_info="Goal: test"
        )
        assert "Plan next step" in system
        assert "Tool A, Tool B" in system
        assert "Output JSON" in system
        assert user == "Goal: test"

        # Type check: rejects non-CognitiveContext
        class SimpleWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan"

        llm2 = MockLLM()
        llm2.structured_output_response = FastThinkResult(
            step_content="", calls=[], details_needed=[]
        )
        w = SimpleWorker(llm=llm2, mode=ThinkingMode.FAST)
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await w.arun(context="not a context")

    @pytest.mark.asyncio
    async def test_override_prompt_customization(self):
        """Override observation + build_thinking_prompt: custom text appears in LLM messages."""
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            step_content="Search flights",
            calls=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
            reasoning="done",
            details_needed=[]
        )
        worker = _PromptCustomWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        # Custom observation appears in user prompt
        user_msg = llm.captured_messages[-1]
        assert "Custom observation: environment is ready" in user_msg.content

        # EXTRA_INSTRUCTION appears in system prompt
        system_msg = llm.captured_messages[0]
        assert "EXTRA_INSTRUCTION: Always prefer cheapest option" in system_msg.content

    @pytest.mark.asyncio
    async def test_override_action_pipeline(self):
        """Override verify_tools + consequence: book_flight filtered, result formatted as string."""
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            step_content="Search and book",
            calls=[
                StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                ),
                StepToolCall(
                    tool="book_flight",
                    tool_arguments=[
                        ToolArgument(name="flight_number", value="CA123"),
                    ]
                ),
            ],
            reasoning="Do both",
            details_needed=[]
        )
        worker = _ActionPipelineWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        last_step = ctx.cognitive_history[-1]
        # verify_tools filtered out book_flight
        assert "book_flight" not in last_step.metadata.get("tool_calls", [])
        assert "search_flights" in last_step.metadata.get("tool_calls", [])
        # consequence formatted as joined string
        assert isinstance(last_step.result, str)
        assert "Found 3 available flights" in last_step.result

    @pytest.mark.asyncio
    async def test_fast_mode_cycle(self):
        """FAST mode: Worker executes one thinking cycle without setting finish."""
        llm = MockLLM()
        ctx = _make_context()
        worker = _FastWorker(llm=llm, mode=ThinkingMode.FAST)

        # Tool execution - Worker executes cycle, doesn't set finish
        llm.structured_output_response = FastThinkResult(
            step_content="Search flights from Beijing to Tokyo",
            calls=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
            reasoning="Search first",
            details_needed=[]
        )
        await worker.arun(context=ctx)

        assert len(ctx.cognitive_history) == 1
        assert ctx.cognitive_history[-1].status is True
        assert "search_flights" in ctx.cognitive_history[-1].metadata.get("tool_calls", [])
        # Worker doesn't set finish (Option A design)
        assert ctx.finish is False

        # Empty calls - Worker still executes, doesn't set finish
        llm.structured_output_response = FastThinkResult(
            step_content="All done",
            calls=[],
            reasoning="Task complete",
            details_needed=[]
        )
        await worker.arun(context=ctx)
        # Worker doesn't set finish even with empty calls
        assert ctx.finish is False
        # But step is recorded
        assert len(ctx.cognitive_history) == 2

    @pytest.mark.asyncio
    async def test_default_mode_cycle(self):
        """DEFAULT mode: Worker executes one thinking cycle without setting finish."""
        llm = MockLLM()
        ctx = _make_context()
        worker = _DefaultWorker(llm=llm, mode=ThinkingMode.DEFAULT)

        # Tool execution with tool selection
        llm.structured_output_response = DefaultThinkResult(
            steps=["Search for available flights from Beijing to Tokyo"],
            reasoning="Start with flights",
            details_needed=[]
        )
        llm.select_tool_response = (
            [ToolCall(id="tc1", name="search_flights", arguments={
                "origin": "Beijing", "destination": "Tokyo", "date": "2025-06-01"
            })],
            None
        )
        await worker.arun(context=ctx)

        assert len(ctx.cognitive_history) == 1
        assert ctx.cognitive_history[-1].status is True
        assert "search_flights" in ctx.cognitive_history[-1].metadata.get("tool_calls", [])
        # Worker doesn't set finish (Option A design)
        assert ctx.finish is False

        # Empty steps - Worker completes without error, doesn't set finish
        llm.structured_output_response = DefaultThinkResult(
            steps=[],
            reasoning="Goal achieved",
            details_needed=[]
        )
        await worker.arun(context=ctx)
        # Worker doesn't set finish even with empty steps
        assert ctx.finish is False

        # Custom select_tools override bypasses LLM tool selection
        llm2 = MockLLM()
        llm2.structured_output_response = DefaultThinkResult(
            steps=["Search for flights to Tokyo"],
            reasoning="Need flights",
            details_needed=[]
        )
        override_worker = _OverrideSelectToolsWorker(llm=llm2, mode=ThinkingMode.DEFAULT)
        ctx2 = _make_context()
        await override_worker.arun(context=ctx2)

        assert "search_flights" in ctx2.cognitive_history[-1].metadata.get("tool_calls", [])
