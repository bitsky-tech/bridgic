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


class _FailingWorker(CognitiveWorker):
    """Worker that always raises during thinking."""
    async def thinking(self):
        raise RuntimeError("Worker failed")


# ============================================================================
# Helper: common mock tool calls
# ============================================================================

def _fast_search_flights() -> FastThinkResult:
    return FastThinkResult(
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
        step_content="All bookings completed",
        calls=[],
        reasoning="Flight and hotel are booked, task is done",
        details_needed=[]
    )


def _select_tool_call(name: str, arguments: dict) -> Tuple[List[ToolCall], None]:
    return ([ToolCall(id=f"tc_{name}", name=name, arguments=arguments)], None)


def _assert_tool_sequence(result, expected_tools: List[str]):
    """Assert that cognitive_history steps executed the expected tool sequence."""
    steps = result.cognitive_history.get_all()
    assert len(steps) == len(expected_tools)
    for step, tool_name in zip(steps, expected_tools):
        assert tool_name in step.metadata["tool_calls"]
    assert all(s.status for s in steps)


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
            # Option A: Termination controlled by Agent via until()
            # Stop when we've completed 4 steps (search/book flight + hotel)
            await self.react.until(
                lambda c: len(c.cognitive_history) >= 4,
                max_attempts=5
            )
            await ctx.cognitive_history.compress_if_needed()

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
            # Option A: First try React approach (up to 3 times)
            await self.react.until(
                lambda c: len(c.cognitive_history) >= 3,
                max_attempts=3
            )

            # If not enough steps completed, use Plan to finish
            if len(ctx.cognitive_history) < 4:
                await self.plan

    return ReactThenPlanAgent


# ============================================================================
# Tests: Agent Workflows
# ============================================================================

class TestAgentWorkflows:
    """Full agent workflow integration tests — each pattern runs a complete travel planning scenario."""

    @pytest.mark.asyncio
    async def test_react_agent(self):
        """ReactAgent: FAST-mode react loop until finish (5 iterations)."""
        llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
            _fast_search_hotels(),
            _fast_book_hotel(),
            _fast_finish(),
        ])

        ReactAgent = _make_react_agent(llm)
        result = await ReactAgent().arun(
            goal="I'm currently in Beijing and I want to go to Kunming tomorrow. "
                 "Could you help me plan the route and hotel?"
        )

        # Option A: Worker doesn't set finish, Agent controls termination via until()
        # Verify task completion by checking executed steps instead
        _assert_tool_sequence(result, ["search_flights", "book_flight", "search_hotels", "book_hotel"])

    @pytest.mark.asyncio
    async def test_plan_agent(self):
        """PlanAgent: DEFAULT-mode single plan step — plans and executes all steps at once."""
        llm = StatefulMockLLM(
            structured_responses=[
                DefaultThinkResult(
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
                _select_tool_call("search_flights", {
                    "origin": "Beijing", "destination": "Kunming", "date": "2025-06-01"
                }),
                _select_tool_call("book_flight", {"flight_number": "MU456"}),
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
        result = await PlanAgent().arun(
            goal="I'm currently in Beijing and I want to go to Kunming. "
                 "Could you help me plan the route?"
        )

        _assert_tool_sequence(result, ["search_flights", "book_flight", "search_hotels", "book_hotel"])

    @pytest.mark.asyncio
    async def test_mix_agent(self):
        """MixAgent: react twice (FAST), then plan (DEFAULT) — mixing modes."""
        react_llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
        ])

        plan_llm = StatefulMockLLM(
            structured_responses=[
                DefaultThinkResult(
                    steps=["Search for hotels in Kunming", "Book the cheapest hotel"],
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
        result = await MixAgent().arun(
            goal="I'm currently in Beijing and I want to go to Kunming. "
                 "Could you help me plan the route?"
        )

        _assert_tool_sequence(result, ["search_flights", "book_flight", "search_hotels", "book_hotel"])

    @pytest.mark.asyncio
    async def test_react_then_plan_agent(self):
        """ReactThenPlanAgent: react up to 3 times, then plan if not finished."""
        react_llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
            _fast_search_hotels(),
        ])

        plan_llm = StatefulMockLLM(
            structured_responses=[
                DefaultThinkResult(
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
        result = await ReactThenPlanAgent().arun(
            goal="I'm currently in Beijing and I want to go to Kunming tomorrow. "
                 "Could you help me plan the route and hotel?"
        )

        _assert_tool_sequence(result, ["search_flights", "book_flight", "search_hotels", "book_hotel"])


# ============================================================================
# Tests: Tool/Skill Filtering + Error Strategies
# ============================================================================

class TestAgentFeatures:
    """Test think_step features: tool/skill filtering, error strategies."""

    @pytest.mark.asyncio
    async def test_tool_and_skill_filtering(self):
        """think_step tool/skill filtering temporarily hides items, then restores."""
        # --- Tool filtering: only search_flights visible ---
        llm1 = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                step_content="Search flights",
                calls=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Kunming"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
                reasoning="Search", details_needed=[]
            ),
        ])
        worker1 = ReactThinkingWorker(llm=llm1, mode=ThinkingMode.FAST)

        class ToolFilterAgent(AgentAutoma[TravelPlanningContext]):
            filtered_step = think_step(worker1, tools=["search_flights"])
            async def cognition(self, ctx):
                await self.filtered_step

        result = await ToolFilterAgent().arun(goal="Search flights only")
        # After step, all original tools restored
        assert len(result.tools) == len(get_travel_planning_tools())

        # --- Skill filtering: only travel-planning visible during execution ---
        llm2 = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                step_content="Done", calls=[], reasoning="done",
                details_needed=[]
            ),
        ])
        worker2 = ReactThinkingWorker(llm=llm2, mode=ThinkingMode.FAST)

        skills_during = []
        original_execute = worker2.arun
        async def patched_arun(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                skills_during.append([s.name for s in ctx.skills.get_all()])
            return await original_execute(*args, **kwargs)
        worker2.arun = patched_arun

        class SkillFilterAgent(AgentAutoma[TravelPlanningContext]):
            filtered_step = think_step(worker2, skills=["travel-planning"])
            async def cognition(self, ctx):
                await self.filtered_step

        result = await SkillFilterAgent().arun(goal="test")

        # During execution, only travel-planning visible
        assert len(skills_during) > 0
        assert skills_during[0] == ["travel-planning"]
        # After execution, all skills restored
        assert len(result.skills) == 2  # travel-planning + code-review

    @pytest.mark.asyncio
    async def test_error_strategies(self):
        """ErrorStrategy: RAISE re-raises, IGNORE swallows, RETRY exhausts then raises."""
        llm = StatefulMockLLM()
        failing_worker = _FailingWorker(llm=llm, mode=ThinkingMode.FAST)

        # RAISE: exception propagates
        class RaiseAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.RAISE)
            async def cognition(self, ctx):
                await self.step

        with pytest.raises(RuntimeError, match="Worker failed"):
            await RaiseAgent().arun(goal="test")

        # IGNORE: exception swallowed, agent completes
        class IgnoreAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.IGNORE)
            async def cognition(self, ctx):
                await self.step

        result = await IgnoreAgent().arun(goal="test")
        assert isinstance(result, CognitiveContext)

        # RETRY: retries exhausted, then raises
        class RetryAgent(AgentAutoma[CognitiveContext]):
            step = think_step(failing_worker, on_error=ErrorStrategy.RETRY, max_retries=2)
            async def cognition(self, ctx):
                await self.step

        with pytest.raises(RuntimeError, match="Worker failed"):
            await RetryAgent().arun(goal="test")

    @pytest.mark.asyncio
    async def test_until_conditional_repeat(self):
        """test_step.until(condition, max_attempts): repeats until condition met or max reached."""
        # Test 1: Condition met before max_attempts
        llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
            _fast_search_hotels(),
            _fast_finish(),  # Extra in case needed
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)

        class UntilAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx: TravelPlanningContext):
                # Stop after 3 executions
                await self.step.until(lambda ctx: len(ctx.cognitive_history) >= 3, max_attempts=5)

        result = await UntilAgent().arun(goal="test")
        assert len(result.cognitive_history) == 3  # Should stop when condition met

        # Test 2: Max attempts reached before condition
        llm2 = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
            _fast_finish(),  # Extra in case needed
        ])
        worker2 = ReactThinkingWorker(llm=llm2, mode=ThinkingMode.FAST)

        class MaxAttemptsAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker2)

            async def cognition(self, ctx: TravelPlanningContext):
                # Condition will never be true, so should hit max_attempts
                await self.step.until(lambda ctx: False, max_attempts=2)

        result2 = await MaxAttemptsAgent().arun(goal="test")
        assert len(result2.cognitive_history) == 2  # Should execute exactly max_attempts times

    @pytest.mark.asyncio
    async def test_dynamic_tools_and_skills_in_cognition(self):
        """with_tools()/with_skills() and until(tools=..., skills=...): dynamic override in cognition."""
        # --- with_tools: step without tools filter, override at runtime ---
        llm1 = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                step_content="Search flights",
                calls=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Kunming"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
                reasoning="Search", details_needed=[]
            ),
        ])
        worker1 = ReactThinkingWorker(llm=llm1, mode=ThinkingMode.FAST)

        class DynamicToolsAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker1)  # No tools filter at init
            async def cognition(self, ctx):
                await self.step.with_tools(["search_flights"])

        result = await DynamicToolsAgent().arun(goal="test")
        assert len(result.tools) == len(get_travel_planning_tools())  # Restored after

        # --- with_skills + until: dynamic skills in until ---
        llm2 = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
        ])
        worker2 = ReactThinkingWorker(llm=llm2, mode=ThinkingMode.FAST)
        skills_seen = []
        original_arun = worker2.arun
        async def capture_skills(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                skills_seen.append([s.name for s in ctx.skills.get_all()])
            return await original_arun(*args, **kwargs)
        worker2.arun = capture_skills

        class DynamicSkillsAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker2)
            async def cognition(self, ctx):
                await self.step.until(
                    lambda c: True,
                    skills=["travel-planning"],
                    max_attempts=1
                )

        await DynamicSkillsAgent().arun(goal="test")
        assert len(skills_seen) > 0
        assert skills_seen[0] == ["travel-planning"]

    @pytest.mark.asyncio
    async def test_dynamic_think_step_creation(self):
        """Dynamic think_step creation: defined in cognition() using .bind()."""
        llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_book_flight(),
        ])

        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)

        class DynamicStepAgent(AgentAutoma[TravelPlanningContext]):
            async def cognition(self, ctx: TravelPlanningContext):
                # Create step dynamically and bind to agent
                dynamic_step = think_step(worker).bind(self)
                await dynamic_step
                await dynamic_step

        result = await DynamicStepAgent().arun(goal="test")
        assert len(result.cognitive_history) == 2

    @pytest.mark.asyncio
    async def test_chained_override_priority(self):
        """Chained overrides: later override wins (with_tools then with_tools)."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(
                step_content="Done", calls=[], reasoning="", details_needed=[]
            ),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        tools_seen = []

        original_arun = worker.arun
        async def capture_tools(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                tools_seen.append([t.tool_name for t in ctx.tools.get_all()])
            return await original_arun(*args, **kwargs)
        worker.arun = capture_tools

        tools = get_travel_planning_tools()

        class ChainAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx):
                # Chain: with_tools(["search_flights"]) then with_tools(["book_flight"])
                # The last one should win
                await self.step.with_tools(["search_flights"]).with_tools(["book_flight"])

        result = await ChainAgent().arun(goal="test")

        # Should only see book_flight (last override wins)
        assert len(tools_seen) > 0
        assert tools_seen[0] == ["book_flight"]

    @pytest.mark.asyncio
    async def test_until_with_dynamic_tools_override(self):
        """until(tools=...) should override descriptor's tools filter."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="", details_needed=[]),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        tools_seen = []

        original_arun = worker.arun
        async def capture_tools(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                tools_seen.append([t.tool_name for t in ctx.tools.get_all()])
            return await original_arun(*args, **kwargs)
        worker.arun = capture_tools

        class UntilOverrideAgent(AgentAutoma[TravelPlanningContext]):
            # Descriptor filter: ["search_flights"]
            step = think_step(worker, tools=["search_flights"])

            async def cognition(self, ctx):
                # Runtime override via until: ["book_flight"]
                await self.step.until(lambda c: True, max_attempts=1, tools=["book_flight"])

        await UntilOverrideAgent().arun(goal="test")

        # Should see book_flight (until override wins over descriptor filter)
        assert len(tools_seen) > 0
        assert tools_seen[0] == ["book_flight"]

    @pytest.mark.asyncio
    async def test_empty_tools_filter(self):
        """Empty tools filter results in no tools available."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="", details_needed=[]),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        tools_count = []

        original_arun = worker.arun
        async def capture_count(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                tools_count.append(len(ctx.tools))
            return await original_arun(*args, **kwargs)
        worker.arun = capture_count

        class EmptyToolsAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx):
                await self.step.with_tools([])

        await EmptyToolsAgent().arun(goal="test")

        # Should have 0 tools during execution
        assert len(tools_count) > 0
        assert tools_count[0] == 0

    @pytest.mark.asyncio
    async def test_nonexistent_tool_filter(self):
        """Filtering by nonexistent tool name results in empty tool list."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="", details_needed=[]),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        tools_count = []

        original_arun = worker.arun
        async def capture_count(*args, **kwargs):
            ctx = kwargs.get("context", args[0] if args else None)
            if ctx:
                tools_count.append(len(ctx.tools))
            return await original_arun(*args, **kwargs)
        worker.arun = capture_count

        class NonExistentAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx):
                await self.step.with_tools(["nonexistent_tool"])

        await NonExistentAgent().arun(goal="test")

        # Should have 0 tools (nonexistent tool filtered out)
        assert len(tools_count) > 0
        assert tools_count[0] == 0

    @pytest.mark.asyncio
    async def test_until_max_attempts_zero(self):
        """until with max_attempts=0 should not execute the step at all."""
        llm = StatefulMockLLM(structured_responses=[])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        execution_count = [0]

        original_arun = worker.arun
        async def count_executions(*args, **kwargs):
            execution_count[0] += 1
            return await original_arun(*args, **kwargs)
        worker.arun = count_executions

        class ZeroAttemptsAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx):
                # max_attempts=0 means range(0), no iterations
                await self.step.until(lambda c: False, max_attempts=0)

        result = await ZeroAttemptsAgent().arun(goal="test")

        # With max_attempts=0, the loop range(0) produces no iterations
        assert execution_count[0] == 0

    @pytest.mark.asyncio
    async def test_until_condition_immediately_true(self):
        """until with condition that's immediately true executes exactly once."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="", details_needed=[]),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        execution_count = [0]

        original_arun = worker.arun
        async def count_executions(*args, **kwargs):
            execution_count[0] += 1
            return await original_arun(*args, **kwargs)
        worker.arun = count_executions

        class ImmediateTrueAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)

            async def cognition(self, ctx):
                # Condition is always True, should execute once then exit
                await self.step.until(lambda c: True, max_attempts=10)

        result = await ImmediateTrueAgent().arun(goal="test")

        # Should execute only once since condition is true after first execution
        assert execution_count[0] == 1


# ============================================================================
# Tests: ctx_init
# ============================================================================

class TestCtxInit:
    """Test ctx_init parameter for initializing context fields without subclassing."""

    def test_ctx_init_apply_rules(self):
        """ctx_init: Exposure→add each, regular→setattr, unknown→skip, None→noop, type errors."""
        class SimpleAgent(AgentAutoma[CognitiveContext]):
            async def cognition(self, ctx):
                pass

        tools = get_travel_planning_tools()

        # --- Exposure field: items added via add() ---
        agent = SimpleAgent(ctx_init={"tools": tools})
        ctx = CognitiveContext(goal="test")
        agent._apply_ctx_init(ctx)
        assert len(ctx.tools) == len(tools)
        for i, tool in enumerate(tools):
            assert ctx.tools[i].tool_name == tool.tool_name

        # --- Regular field: setattr when type matches ---
        agent = SimpleAgent(ctx_init={"finish": True})
        ctx = CognitiveContext(goal="test")
        assert ctx.finish is False
        agent._apply_ctx_init(ctx)
        assert ctx.finish is True

        # --- Unknown keys: silently skipped ---
        agent = SimpleAgent(ctx_init={"unknown_key": "value", "another": 42})
        ctx = CognitiveContext(goal="test")
        agent._apply_ctx_init(ctx)
        assert len(ctx.tools) == 0
        assert ctx.finish is False

        # --- None ctx_init: no-op ---
        agent = SimpleAgent()
        ctx = CognitiveContext(goal="test")
        agent._apply_ctx_init(ctx)
        assert len(ctx.tools) == 0
        assert ctx.goal == "test"

        # --- Exposure requires list ---
        agent = SimpleAgent(ctx_init={"tools": "not a list"})
        ctx = CognitiveContext(goal="test")
        with pytest.raises(TypeError, match="expected a list for Exposure field"):
            agent._apply_ctx_init(ctx)

        # --- Type mismatch on regular field ---
        agent = SimpleAgent(ctx_init={"goal": 123})
        ctx = CognitiveContext(goal="test")
        with pytest.raises(TypeError, match="ctx_init\\['goal'\\]"):
            agent._apply_ctx_init(ctx)

    @pytest.mark.asyncio
    async def test_ctx_init_integration(self):
        """ctx_init works end-to-end: via arun(goal=...), arun(context=...), and alongside __post_init__."""
        llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
        ])
        worker = ReactThinkingWorker(llm=llm, mode=ThinkingMode.FAST)
        tools = get_travel_planning_tools()

        # 1. Via arun(goal=...) — framework creates context, ctx_init applied
        skill_file = os.path.join(SKILLS_DIR, "travel-planning", "SKILL.md")

        class SimpleAgent(AgentAutoma[CognitiveContext]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step

        result = await SimpleAgent(ctx_init={"tools": tools, "skills": [skill_file]}).arun(goal="test")
        assert len(result.tools) == len(tools)
        assert len(result.skills) == 1
        assert result.skills[0].name == "travel-planning"
        # Option A: Worker doesn't set finish

        # 2. Via arun(context=...) — pre-created context, ctx_init still applied
        ctx = CognitiveContext(goal="test")
        result = await SimpleAgent(ctx_init={"tools": tools}).arun(context=ctx)
        assert len(result.tools) == len(tools)

        # 3. Combined with __post_init__ — both contribute
        class PostInitAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step

        result = await PostInitAgent(ctx_init={"finish": True}).arun(goal="test")
        assert len(result.tools) == len(get_travel_planning_tools())  # from __post_init__
        assert result.finish is True  # from ctx_init


# ============================================================================
# Tests: Agent Runtime (LLM injection, verbose propagation, usage stats)
# ============================================================================

class TestAgentRuntime:
    """Test agent runtime behavior: LLM/verbose injection, override priority, and usage stats."""

    @pytest.mark.asyncio
    async def test_injection_override_and_stats(self):
        """LLM injection, LLM override, verbose inheritance/override, stats tracking/reset."""
        # --- 1. Worker without LLM/verbose inherits from agent, stats accumulated ---
        llm = StatefulMockLLM(structured_responses=[
            _fast_search_flights(),
            _fast_finish(),
        ])
        worker = ReactThinkingWorker(mode=ThinkingMode.FAST)
        assert worker._llm is None
        assert worker._verbose is None

        class InjectAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step
                await self.step

        agent = InjectAgent(llm=llm, verbose=True)
        assert agent.spend_tokens == 0
        assert agent.spend_time == 0.0

        result = await agent.arun(goal="test")
        # Option A: Worker doesn't set finish
        assert worker._llm is llm          # LLM was injected (persists)
        assert worker._verbose is None      # Verbose restored to None after execution
        tokens_first = agent.spend_tokens
        assert tokens_first > 0
        assert agent.spend_time > 0.0

        # --- 2. Worker with own LLM/verbose keeps them; stats reset per run ---
        own_llm = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
        ])
        worker2 = ReactThinkingWorker(llm=own_llm, mode=ThinkingMode.FAST, verbose=False)

        class OverrideAgent(AgentAutoma[TravelPlanningContext]):
            step = think_step(worker2)
            async def cognition(self, ctx):
                await self.step

        agent2 = OverrideAgent(llm=llm, verbose=True)
        result = await agent2.arun(goal="test")
        # Option A: Worker doesn't set finish
        assert worker2._llm is own_llm     # Worker kept its own LLM
        assert worker2._verbose is False    # Worker kept its own verbose=False
        assert agent2.spend_tokens > 0

        # --- 3. Stats reset: re-run with same-shape work, tokens should not double ---
        own_llm2 = StatefulMockLLM(structured_responses=[
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
            FastThinkResult(step_content="Done", calls=[], reasoning="done", details_needed=[]),
        ])
        worker3 = ReactThinkingWorker(llm=own_llm2, mode=ThinkingMode.FAST)

        class StatsAgent(AgentAutoma[CognitiveContext]):
            step = think_step(worker3)
            async def cognition(self, ctx):
                await self.step

        agent3 = StatsAgent()
        await agent3.arun(goal="run1")
        tokens_run1 = agent3.spend_tokens
        await agent3.arun(goal="run2")
        assert agent3.spend_tokens == tokens_run1  # Reset, not accumulated

    @pytest.mark.asyncio
    async def test_no_llm_raises(self):
        """Worker without LLM and agent without LLM raises RuntimeError."""
        worker = ReactThinkingWorker(mode=ThinkingMode.FAST)

        class NoLlmAgent(AgentAutoma[CognitiveContext]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step

        with pytest.raises(RuntimeError, match="no LLM set"):
            await NoLlmAgent().arun(goal="test")
