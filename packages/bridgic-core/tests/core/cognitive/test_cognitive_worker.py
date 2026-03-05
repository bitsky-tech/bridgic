"""Tests for CognitiveWorker: template methods, type checks, full observe-think-act cycle."""
import os
import pytest
from unittest.mock import MagicMock
from typing import Any, List, Optional, Tuple

from bridgic.core.cognitive import (
    CognitiveContext,
    CognitiveWorker,
    ThinkResult,
    ThinkDecision,
    ActionStepResult,
    StepToolCall,
    ToolArgument,
    _DELEGATE,
    AgentAutoma,
    think_step,
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


class StatefulMockLLM:
    """Mock LLM that returns a sequence of pre-configured responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self._responses[self._idx]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): return MagicMock()
    async def astream(self, messages, **kwargs): return MagicMock()
    def chat(self, messages, **kwargs): return MagicMock()
    def stream(self, messages, **kwargs): return MagicMock()


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

class TravelCtx(CognitiveContext):
    """CognitiveContext pre-loaded with travel planning tools."""
    def __post_init__(self):
        for t in get_travel_planning_tools():
            self.tools.add(t)


def _make_context() -> CognitiveContext:
    ctx = CognitiveContext(goal="Plan a trip to Tokyo")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    ctx.skills.load_from_directory(SKILLS_DIR)
    return ctx


# ---------------------------------------------------------------------------
# Custom workers
# ---------------------------------------------------------------------------

class _SimpleWorker(CognitiveWorker):
    async def thinking(self):
        return "Plan ONE immediate next step."


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveWorker:

    @pytest.mark.asyncio
    async def test_template_method_defaults(self):
        """Default template methods: thinking raises, observation returns _DELEGATE,
        consequence/verify_tools passthrough, build_thinking_prompt assembles, non-context rejected."""
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()

        # thinking() → NotImplementedError
        with pytest.raises(NotImplementedError):
            await worker.thinking()

        # observation() → _DELEGATE (delegate to Agent)
        result = await worker.observation(ctx)
        assert result is _DELEGATE

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
        llm2.structured_output_response = ThinkResult(
            step_content="", calls=[], details_needed=[]
        )
        w = SimpleWorker(llm=llm2)
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await w.arun(context="not a context")

    @pytest.mark.asyncio
    async def test_override_prompt_customization(self):
        """Override observation + build_thinking_prompt: custom text appears in LLM messages."""
        llm = MockLLM()
        llm.structured_output_response = ThinkResult(
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
        worker = _PromptCustomWorker(llm=llm)
        ctx = _make_context()

        # Simulate ThinkStepDescriptor: call observation(), write to ctx.observation, then arun()
        obs = await worker.observation(ctx)
        assert obs == "Custom observation: environment is ready"
        ctx.observation = obs
        await worker.arun(context=ctx)

        # Custom observation appears in user prompt
        user_msg = llm.captured_messages[-1]
        assert "Custom observation: environment is ready" in user_msg.content

        # EXTRA_INSTRUCTION appears in system prompt
        system_msg = llm.captured_messages[0]
        assert "EXTRA_INSTRUCTION: Always prefer cheapest option" in system_msg.content

    @pytest.mark.asyncio
    async def test_thinking_only(self):
        """Worker.arun() only performs thinking, stores decision in _last_decision."""
        llm = MockLLM()
        ctx = _make_context()
        worker = _SimpleWorker(llm=llm)

        # Tool call scenario - decision stored in _last_decision
        llm.structured_output_response = ThinkResult(
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

        # Worker stores decision, does NOT execute tools
        assert worker._last_decision is not None
        assert worker._last_decision.step_content == "Search flights from Beijing to Tokyo"
        assert len(worker._last_decision.calls) == 1
        assert worker._last_decision.calls[0].tool == "search_flights"
        # No history added (action not executed)
        assert len(ctx.cognitive_history) == 0

    @pytest.mark.asyncio
    async def test_cycle_via_agent(self):
        """End-to-end: worker via agent executes full observe-think-act cycle."""
        llm = StatefulMockLLM([
            ThinkResult(
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
            ),
            ThinkResult(
                step_content="All done",
                calls=[],
                reasoning="Task complete",
                details_needed=[]
            ),
        ])
        worker = _SimpleWorker(llm=llm)

        class SimpleAgent(AgentAutoma[TravelCtx]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step  # search_flights
                await self.step  # no tools

        result = await SimpleAgent().arun(goal="Plan a trip to Tokyo")

        assert len(result.cognitive_history) == 2
        # Step 1: search_flights was executed
        assert result.cognitive_history[0].status is True
        assert "search_flights" in result.cognitive_history[0].metadata.get("tool_calls", [])
        assert result.last_step_has_tools is False  # last step had no tools

    @pytest.mark.asyncio
    async def test_override_action_pipeline_via_agent(self):
        """verify_tools + consequence: book_flight filtered, result formatted as string via agent."""
        llm = MockLLM()
        llm.structured_output_response = ThinkResult(
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
        worker = _ActionPipelineWorker(llm=llm)

        class PipelineAgent(AgentAutoma[TravelCtx]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step

        result = await PipelineAgent().arun(goal="test")

        last_step = result.cognitive_history[-1]
        # verify_tools filtered out book_flight
        assert "book_flight" not in last_step.metadata.get("tool_calls", [])
        assert "search_flights" in last_step.metadata.get("tool_calls", [])
        # consequence formatted as joined string
        assert isinstance(last_step.result, str)
        assert "Found 3 available flights" in last_step.result
