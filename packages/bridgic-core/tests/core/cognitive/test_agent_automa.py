"""
Tests for AgentAutoma using real agent patterns from idea_practice.

Replicates ReactAgent, PlanAgent, MixAgent, ReactThenPlanAgent with
StatefulMockLLM to simulate full travel planning workflows (Beijing → Kunming).
"""
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest
from unittest.mock import MagicMock

from bridgic.core.cognitive import (
    AgentAutoma,
    CognitiveContext,
    CognitiveWorker,
    ThinkingMode,
    think_step,
    ErrorStrategy,
    FastThinkResult,
    DefaultThinkResult,
    StepToolCall,
    ToolArgument,
)
from bridgic.core.model.types import ToolCall
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


# ============================================================================
# Stateful Mock LLM
# ============================================================================

class StatefulMockLLM:
    """
    Mock LLM that returns responses from a pre-configured sequence.

    Each call to astructured_output / aselect_tool pops the next response
    from its respective queue.
    """

    def __init__(
        self,
        structured_responses: Optional[List[Any]] = None,
        select_tool_responses: Optional[List[Tuple[List[ToolCall], Optional[str]]]] = None,
    ):
        self._structured_responses = list(structured_responses or [])
        self._select_tool_responses = list(select_tool_responses or [])
        self._structured_idx = 0
        self._select_tool_idx = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self._structured_responses[self._structured_idx]
        self._structured_idx += 1
        return resp

    async def aselect_tool(self, messages, tools, **kwargs):
        resp = self._select_tool_responses[self._select_tool_idx]
        self._select_tool_idx += 1
        return resp

    # Required by BaseLlm ABC (not used in tests but must exist)
    async def achat(self, messages, **kwargs):
        return MagicMock()

    async def astream(self, messages, **kwargs):
        return MagicMock()

    def chat(self, messages, **kwargs):
        return MagicMock()

    def stream(self, messages, **kwargs):
        return MagicMock()


# ============================================================================
# Custom Context (from idea_practice)
# ============================================================================

class TravelPlanningContext(CognitiveContext):
    """Context with travel planning tools and skills pre-loaded."""

    def __post_init__(self):
        for tool_spec in get_travel_planning_tools():
            self.tools.add(tool_spec)

        skills_dir = Path(__file__).parent / "skills"
        if skills_dir.exists():
            self.skills.load_from_directory(str(skills_dir))


# ============================================================================
# Custom Workers (from idea_practice)
# ============================================================================

class ReactThinkingWorker(CognitiveWorker):
    """React-style worker: plan ONE step at a time."""

    async def thinking(self) -> str:
        return (
            "You are a reactive assistant. Your approach is to OBSERVE, THINK, then ACT - one step at a time.\n\n"
            "Rather than planning everything upfront, you respond to the current situation and take "
            "the single most appropriate next action. After each action, you observe the result and "
            "decide what to do next based on the new information.\n\n"
            "Your reactive process:\n"
            "1. Observe the current state - what has happened so far, what information do you have\n"
            "2. Determine the single best next action to move toward the goal\n"
            "3. Execute that one action and wait for the result\n"
            "4. Repeat until the goal is achieved\n\n"
            "Focus only on the immediate next step. Do not plan multiple steps ahead - "
            "the situation may change based on each action's outcome."
        )


class PlanThinkingWorker(CognitiveWorker):
    """Plan-style worker: create a COMPLETE plan."""

    async def thinking(self) -> str:
        return (
            "You are a planning-oriented assistant. Your approach is to THINK BEFORE YOU ACT.\n\n"
            "Before taking any action, create a complete plan covering all steps needed to achieve the goal. "
            "Do not just consider the immediate next step - think through the ENTIRE task from beginning to end.\n\n"
            "Your planning process:\n"
            "1. Understand what the user wants to accomplish\n"
            "2. Assess current progress - what has already been done, if anything\n"
            "3. Identify all remaining work needed to fully complete the goal\n"
            "4. Break down this remaining work into concrete, actionable steps\n\n"
            "Important: If some work has already been completed, build upon that progress. "
            "Only plan the steps that still need to be done.\n\n"
            "For example, if the user wants to book a trip, plan the complete journey: "
            "search flights, book flight, search hotels, book hotel, etc."
        )


# ============================================================================
# Helper: common mock tool calls
# ============================================================================

def _fast_search_flights() -> FastThinkResult:
    return FastThinkResult(
        finish=False,
        step_content="Search flights from Beijing to Kunming",
        calls=[StepToolCall(
            tool="search_flights",
            tool_arguments=[
                ToolArgument(name="origin", value="Beijing"),
                ToolArgument(name="destination", value="Kunming"),
                ToolArgument(name="date", value="2025-06-01"),
            ]
        )],
        reasoning="First search for available flights",
        details_needed=[]
    )


def _fast_book_flight() -> FastThinkResult:
    return FastThinkResult(
        finish=False,
        step_content="Book the cheapest flight MU456",
        calls=[StepToolCall(
            tool="book_flight",
            tool_arguments=[
                ToolArgument(name="flight_number", value="MU456"),
            ]
        )],
        reasoning="Book the most economical flight",
        details_needed=[]
    )


def _fast_search_hotels() -> FastThinkResult:
    return FastThinkResult(
        finish=False,
        step_content="Search hotels in Kunming",
        calls=[StepToolCall(
            tool="search_hotels",
            tool_arguments=[
                ToolArgument(name="city", value="Kunming"),
                ToolArgument(name="check_in", value="2025-06-01"),
                ToolArgument(name="check_out", value="2025-06-03"),
            ]
        )],
        reasoning="Search for accommodation",
        details_needed=[]
    )


def _fast_book_hotel() -> FastThinkResult:
    return FastThinkResult(
        finish=False,
        step_content="Book Comfort Inn Kunming",
        calls=[StepToolCall(
            tool="book_hotel",
            tool_arguments=[
                ToolArgument(name="hotel_name", value="Comfort Inn Kunming"),
                ToolArgument(name="check_in", value="2025-06-01"),
                ToolArgument(name="check_out", value="2025-06-03"),
            ]
        )],
        reasoning="Book the cheapest hotel",
        details_needed=[]
    )


def _fast_finish() -> FastThinkResult:
    return FastThinkResult(
        finish=True,
        step_content="All bookings completed",
        calls=[],
        reasoning="Flight and hotel are booked, task is done",
        details_needed=[]
    )


def _select_tool_call(name: str, arguments: dict) -> Tuple[List[ToolCall], None]:
    return ([ToolCall(id=f"tc_{name}", name=name, arguments=arguments)], None)


# ============================================================================
# Agents (from idea_practice)
# ============================================================================

def _make_react_agent(llm):
    """Create a ReactAgent class with the given LLM."""

    class ReactAgent(AgentAutoma[TravelPlanningContext]):
        react = think_step(
            ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST),
            on_error=ErrorStrategy.RAISE,
        )

        async def cognition(self, ctx: TravelPlanningContext):
            for _ in range(5):
                await self.react
                await ctx.cognitive_history.compress_if_needed()
                if ctx.finish:
                    break

    return ReactAgent


def _make_plan_agent(llm):
    """Create a PlanAgent class with the given LLM."""

    class PlanAgent(AgentAutoma[TravelPlanningContext]):
        plan = think_step(
            PlanThinkingWorker(llm=llm, mode=ThinkingMode.DEFAULT),
        )

        async def cognition(self, ctx: TravelPlanningContext):
            await self.plan

    return PlanAgent


def _make_mix_agent(react_llm, plan_llm):
    """Create a MixAgent class with separate LLMs for react and plan workers."""

    class MixAgent(AgentAutoma[TravelPlanningContext]):
        react = think_step(
            ReactThinkingWorker(llm=react_llm, mode=ThinkingMode.FAST),
        )
        plan = think_step(
            PlanThinkingWorker(llm=plan_llm, mode=ThinkingMode.DEFAULT),
        )

        async def cognition(self, ctx: TravelPlanningContext):
            await self.react
            await self.react
            await self.plan

    return MixAgent


def _make_react_then_plan_agent(react_llm, plan_llm):
    """Create a ReactThenPlanAgent class."""

    class ReactThenPlanAgent(AgentAutoma[TravelPlanningContext]):
        react = think_step(
            ReactThinkingWorker(llm=react_llm, mode=ThinkingMode.FAST),
        )
        plan = think_step(
            PlanThinkingWorker(llm=plan_llm, mode=ThinkingMode.DEFAULT),
        )

        async def cognition(self, ctx: TravelPlanningContext):
            # First try React approach (up to 3 times)
            for _ in range(3):
                await self.react
                if ctx.finish or len(ctx.cognitive_history) >= 3:
                    break

            # If not finished, use Plan
            if not ctx.finish:
                await self.plan

    return ReactThenPlanAgent


# ============================================================================
# Tests: Full Agent Workflows
# ============================================================================

class TestReactAgent:
    """ReactAgent: react loop until finish — simulates full travel planning."""

    @pytest.mark.asyncio
    async def test_react_agent(self):
        llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),   # iteration 1: search flights
            _fast_book_flight(),      # iteration 2: book flight
            _fast_search_hotels(),    # iteration 3: search hotels
            _fast_book_hotel(),       # iteration 4: book hotel
            _fast_finish(),           # iteration 5: finish
        ])

        ReactAgent = _make_react_agent(llm)
        agent = ReactAgent()
        result = await agent.arun(
            goal="I'm currently in Beijing and I want to go to Kunming tomorrow. "
                 "Could you help me plan the route and hotel?"
        )

        assert result.finish is True
        assert len(result.cognitive_history) == 4

        # Verify each step executed the correct tool
        steps = result.cognitive_history.get_all()
        assert "search_flights" in steps[0].metadata["tool_calls"]
        assert "book_flight" in steps[1].metadata["tool_calls"]
        assert "search_hotels" in steps[2].metadata["tool_calls"]
        assert "book_hotel" in steps[3].metadata["tool_calls"]

        # Verify all steps succeeded
        assert all(s.status for s in steps)


class TestPlanAgent:
    """PlanAgent: single plan step — plans and executes all steps at once."""

    @pytest.mark.asyncio
    async def test_plan_agent(self):
        llm = StatefulMockLLM(
            structured_responses=[
                # Plan thinking: return 4 steps
                DefaultThinkResult(
                    finish=False,
                    steps=[
                        "Search for available flights from Beijing to Kunming",
                        "Book the cheapest flight",
                        "Search for hotels in Kunming",
                        "Book the cheapest hotel",
                    ],
                    reasoning="Complete travel planning workflow",
                    details_needed=[]
                ),
            ],
            select_tool_responses=[
                # Tool selection for each step
                _select_tool_call("search_flights", {
                    "origin": "Beijing", "destination": "Kunming", "date": "2025-06-01"
                }),
                _select_tool_call("book_flight", {
                    "flight_number": "MU456"
                }),
                _select_tool_call("search_hotels", {
                    "city": "Kunming", "check_in": "2025-06-01", "check_out": "2025-06-03"
                }),
                _select_tool_call("book_hotel", {
                    "hotel_name": "Comfort Inn Kunming",
                    "check_in": "2025-06-01", "check_out": "2025-06-03"
                }),
            ]
        )

        PlanAgent = _make_plan_agent(llm)
        agent = PlanAgent()
        result = await agent.arun(
            goal="I'm currently in Beijing and I want to go to Kunming. "
                 "Could you help me plan the route?"
        )

        assert len(result.cognitive_history) == 4

        steps = result.cognitive_history.get_all()
        assert "search_flights" in steps[0].metadata["tool_calls"]
        assert "book_flight" in steps[1].metadata["tool_calls"]
        assert "search_hotels" in steps[2].metadata["tool_calls"]
        assert "book_hotel" in steps[3].metadata["tool_calls"]

        assert all(s.status for s in steps)


class TestMixAgent:
    """MixAgent: react twice, then plan — mixing FAST and DEFAULT modes."""

    @pytest.mark.asyncio
    async def test_mix_agent(self):
        # React worker LLM: 2 FAST calls
        react_llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),   # react 1: search flights
            _fast_book_flight(),      # react 2: book flight
        ])

        # Plan worker LLM: 1 DEFAULT thinking + 2 tool selections
        plan_llm = StatefulMockLLM(
            structured_responses=[
                DefaultThinkResult(
                    finish=False,
                    steps=[
                        "Search for hotels in Kunming",
                        "Book the cheapest hotel",
                    ],
                    reasoning="Continue with accommodation after flights are done",
                    details_needed=[]
                ),
            ],
            select_tool_responses=[
                _select_tool_call("search_hotels", {
                    "city": "Kunming", "check_in": "2025-06-01", "check_out": "2025-06-03"
                }),
                _select_tool_call("book_hotel", {
                    "hotel_name": "Comfort Inn Kunming",
                    "check_in": "2025-06-01", "check_out": "2025-06-03"
                }),
            ]
        )

        MixAgent = _make_mix_agent(react_llm, plan_llm)
        agent = MixAgent()
        result = await agent.arun(
            goal="I'm currently in Beijing and I want to go to Kunming. "
                 "Could you help me plan the route?"
        )

        assert len(result.cognitive_history) == 4

        steps = result.cognitive_history.get_all()
        # React steps
        assert "search_flights" in steps[0].metadata["tool_calls"]
        assert "book_flight" in steps[1].metadata["tool_calls"]
        # Plan steps
        assert "search_hotels" in steps[2].metadata["tool_calls"]
        assert "book_hotel" in steps[3].metadata["tool_calls"]

        assert all(s.status for s in steps)


class TestReactThenPlanAgent:
    """ReactThenPlanAgent: react up to 3 times, then plan if not finished."""

    @pytest.mark.asyncio
    async def test_react_then_plan_agent(self):
        # React worker LLM: 3 FAST calls (exits loop when history >= 3)
        react_llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),   # react 1
            _fast_book_flight(),      # react 2
            _fast_search_hotels(),    # react 3 → len(history)=3, break loop
        ])

        # Plan worker LLM: 1 DEFAULT thinking + 1 tool selection
        plan_llm = StatefulMockLLM(
            structured_responses=[
                DefaultThinkResult(
                    finish=False,
                    steps=["Book the cheapest hotel"],
                    reasoning="Remaining step: book accommodation",
                    details_needed=[]
                ),
            ],
            select_tool_responses=[
                _select_tool_call("book_hotel", {
                    "hotel_name": "Comfort Inn Kunming",
                    "check_in": "2025-06-01", "check_out": "2025-06-03"
                }),
            ]
        )

        ReactThenPlanAgent = _make_react_then_plan_agent(react_llm, plan_llm)
        agent = ReactThenPlanAgent()
        result = await agent.arun(
            goal="I'm currently in Beijing and I want to go to Kunming tomorrow. "
                 "Could you help me plan the route and hotel?"
        )

        assert len(result.cognitive_history) == 4

        steps = result.cognitive_history.get_all()
        # React steps
        assert "search_flights" in steps[0].metadata["tool_calls"]
        assert "book_flight" in steps[1].metadata["tool_calls"]
        assert "search_hotels" in steps[2].metadata["tool_calls"]
        # Plan step
        assert "book_hotel" in steps[3].metadata["tool_calls"]

        assert all(s.status for s in steps)


# ============================================================================
# Tests: Tool/Skill Filtering
# ============================================================================

class TestToolFiltering:
    @pytest.mark.asyncio
    async def test_tool_filtering(self):
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                finish=False,
                step_content="Search flights",
                calls=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Kunming"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
                reasoning="Search",
                details_needed=[]
            ),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)

        class TestAgent(AgentAutoma[TravelPlanningContext]):
            filtered_step = think_step(worker, tools=["search_flights"])

            async def cognition(self, ctx):
                await self.filtered_step

        agent = TestAgent()
        result = await agent.arun(
            goal="Search flights only"
        )

        # After step execution, all original tools should be restored
        assert len(result.tools) == len(get_travel_planning_tools())


class TestSkillFiltering:
    @pytest.mark.asyncio
    async def test_skill_filtering(self):
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                finish=True, step_content="Done", calls=[], reasoning="done",
                details_needed=[]
            ),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)

        skills_during = []
        original_execute = worker.arun

        async def patched_arun(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                skills_during.append([s.name for s in ctx.skills.get_all()])
            return await original_execute(*args, **kwargs)

        worker.arun = patched_arun

        class TestAgent(AgentAutoma[TravelPlanningContext]):
            filtered_step = think_step(worker, skills=["travel-planning"])

            async def cognition(self, ctx):
                await self.filtered_step

        agent = TestAgent()
        result = await agent.arun(goal="test")
        original_skills_count = 2  # travel-planning + code-review

        # During execution, only travel-planning should be visible
        assert len(skills_during) > 0
        assert skills_during[0] == ["travel-planning"]

        # After execution, all skills should be restored
        assert len(result.skills) == original_skills_count


# ============================================================================
# Tests: Error Handling Strategies
# ============================================================================

class _FailingWorker(CognitiveWorker):
    """Worker that always raises during thinking."""
    async def thinking(self):
        raise RuntimeError("Worker failed")


class TestErrorStrategyRaise:
    @pytest.mark.asyncio
    async def test_error_strategy_raise(self):
        llm = StatefulMockLLM()
        failing_worker = _FailingWorker(llm=llm, mode=ThinkingMode.FAST)

        class TestAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.RAISE)

            async def cognition(self, ctx):
                await self.step

        agent = TestAgent()
        with pytest.raises(RuntimeError, match="Worker failed"):
            await agent.arun(goal="test")


class TestErrorStrategyIgnore:
    @pytest.mark.asyncio
    async def test_error_strategy_ignore(self):
        llm = StatefulMockLLM()
        failing_worker = _FailingWorker(llm=llm, mode=ThinkingMode.FAST)

        class TestAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.IGNORE)

            async def cognition(self, ctx):
                await self.step

        agent = TestAgent()
        result = await agent.arun(goal="test")
        assert isinstance(result, CognitiveContext)


class TestErrorStrategyRetry:
    @pytest.mark.asyncio
    async def test_error_strategy_retry(self):
        llm = StatefulMockLLM()
        failing_worker = _FailingWorker(llm=llm, mode=ThinkingMode.FAST)

        class TestAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.RETRY, max_retries=2)

            async def cognition(self, ctx):
                await self.step

        agent = TestAgent()
        with pytest.raises(RuntimeError, match="Worker failed"):
            await agent.arun(goal="test")
