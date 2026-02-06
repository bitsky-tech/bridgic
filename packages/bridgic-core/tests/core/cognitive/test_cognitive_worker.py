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
# Template method defaults
# ---------------------------------------------------------------------------

class TestThinkingNotImplemented:
    @pytest.mark.asyncio
    async def test_thinking_not_implemented(self):
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        with pytest.raises(NotImplementedError):
            await worker.thinking()


class TestObservationDefault:
    @pytest.mark.asyncio
    async def test_observation_default(self):
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()
        result = await worker.observation(ctx)
        assert result is None


class TestConsequenceDefault:
    @pytest.mark.asyncio
    async def test_consequence_default(self):
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        test_results = [
            ActionStepResult(
                tool_id="1", tool_name="search_flights",
                tool_arguments={"origin": "Beijing"}, tool_result="found"
            )
        ]
        result = await worker.consequence(test_results)
        assert result is test_results


class TestVerifyToolsDefault:
    @pytest.mark.asyncio
    async def test_verify_tools_default(self):
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()
        tools = get_travel_planning_tools()
        matched = [(ToolCall(id="1", name="search_flights", arguments={}), tools[0])]
        result = await worker.verify_tools(matched, ctx)
        assert result is matched


class TestBuildThinkingPromptDefault:
    @pytest.mark.asyncio
    async def test_build_thinking_prompt_default(self):
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
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


# ---------------------------------------------------------------------------
# Template method overrides
# ---------------------------------------------------------------------------

class _OverrideObservationWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE step"

    async def observation(self, context):
        return "Custom observation: environment is ready"


class TestOverrideObservation:
    @pytest.mark.asyncio
    async def test_override_observation(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=True, step_content="Done", calls=[], reasoning="All done",
            details_needed=[]
        )
        worker = _OverrideObservationWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        # The LLM should have received the custom observation in user_prompt
        user_msg = llm.captured_messages[-1]
        assert "Custom observation: environment is ready" in user_msg.content


class _OverrideConsequenceWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE step"

    async def consequence(self, action_results):
        return "; ".join(r.tool_result for r in action_results)


class TestOverrideConsequence:
    @pytest.mark.asyncio
    async def test_override_consequence(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=False,
            step_content="Search flights",
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
        worker = _OverrideConsequenceWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        # History should contain the formatted consequence (joined string), not raw list
        last_step = ctx.cognitive_history[-1]
        assert isinstance(last_step.result, str)
        assert "Found 3 available flights" in last_step.result


class _OverrideVerifyToolsWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE step"

    async def verify_tools(self, matched_list, context):
        return [(tc, spec) for tc, spec in matched_list if spec.tool_name != "book_flight"]


class TestOverrideVerifyTools:
    @pytest.mark.asyncio
    async def test_override_verify_tools(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=False,
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
        worker = _OverrideVerifyToolsWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        last_step = ctx.cognitive_history[-1]
        tool_calls_in_history = last_step.metadata.get("tool_calls", [])
        assert "book_flight" not in tool_calls_in_history
        assert "search_flights" in tool_calls_in_history


class _OverrideBuildThinkingPromptWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE step"

    async def build_thinking_prompt(self, think_prompt, tools_description, output_instructions, context_info):
        extra = "EXTRA_INSTRUCTION: Always prefer cheapest option."
        system = f"{think_prompt}\n\n{extra}\n\n{tools_description}\n\n{output_instructions}"
        return system, context_info


class TestOverrideBuildThinkingPrompt:
    @pytest.mark.asyncio
    async def test_override_build_thinking_prompt(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=True, step_content="Done", calls=[], reasoning="done",
            details_needed=[]
        )
        worker = _OverrideBuildThinkingPromptWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        system_msg = llm.captured_messages[0]
        assert "EXTRA_INSTRUCTION: Always prefer cheapest option" in system_msg.content


class _OverrideSelectToolsWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan steps"

    async def select_tools(self, step_content, context):
        return [ToolCall(id="fixed_1", name="search_flights", arguments={
            "origin": "Beijing", "destination": "Tokyo", "date": "2025-06-01"
        })]


class TestOverrideSelectTools:
    @pytest.mark.asyncio
    async def test_override_select_tools(self):
        llm = MockLLM()
        llm.structured_output_response = DefaultThinkResult(
            finish=False,
            steps=["Search for flights to Tokyo"],
            reasoning="Need flights",
            details_needed=[]
        )
        worker = _OverrideSelectToolsWorker(llm=llm, mode=ThinkingMode.DEFAULT)
        ctx = _make_context()
        await worker.arun(context=ctx)

        last_step = ctx.cognitive_history[-1]
        tool_calls_in_history = last_step.metadata.get("tool_calls", [])
        assert "search_flights" in tool_calls_in_history


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

class TestObservationRejectsNonContext:
    @pytest.mark.asyncio
    async def test_observation_rejects_non_context(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=True, step_content="", calls=[], details_needed=[]
        )

        class SimpleWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan"

        worker = SimpleWorker(llm=llm, mode=ThinkingMode.FAST)
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await worker.arun(context="not a context")


# ---------------------------------------------------------------------------
# Full cycle (mocked LLM)
# ---------------------------------------------------------------------------

class _FastWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE immediate next step."


class _DefaultWorker(CognitiveWorker):
    async def thinking(self):
        return "Create a step-by-step plan."


class TestFastModeFinish:
    @pytest.mark.asyncio
    async def test_fast_mode_finish(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=True, step_content="Task complete", calls=[], reasoning="All done",
            details_needed=[]
        )
        worker = _FastWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)
        assert ctx.finish is True


class TestFastModeToolExecution:
    @pytest.mark.asyncio
    async def test_fast_mode_tool_execution(self):
        llm = MockLLM()
        llm.structured_output_response = FastThinkResult(
            finish=False,
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
        worker = _FastWorker(llm=llm, mode=ThinkingMode.FAST)
        ctx = _make_context()
        await worker.arun(context=ctx)

        assert len(ctx.cognitive_history) > 0
        last_step = ctx.cognitive_history[-1]
        assert last_step.status is True
        assert "search_flights" in last_step.metadata.get("tool_calls", [])


class TestDefaultModeFinish:
    @pytest.mark.asyncio
    async def test_default_mode_finish(self):
        llm = MockLLM()
        llm.structured_output_response = DefaultThinkResult(
            finish=True, steps=[], reasoning="Goal achieved",
            details_needed=[]
        )
        worker = _DefaultWorker(llm=llm, mode=ThinkingMode.DEFAULT)
        ctx = _make_context()
        await worker.arun(context=ctx)
        assert ctx.finish is True


class TestDefaultModeToolExecution:
    @pytest.mark.asyncio
    async def test_default_mode_tool_execution(self):
        llm = MockLLM()
        llm.structured_output_response = DefaultThinkResult(
            finish=False,
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
        worker = _DefaultWorker(llm=llm, mode=ThinkingMode.DEFAULT)
        ctx = _make_context()
        await worker.arun(context=ctx)

        assert len(ctx.cognitive_history) > 0
        last_step = ctx.cognitive_history[-1]
        assert last_step.status is True
        assert "search_flights" in last_step.metadata.get("tool_calls", [])
