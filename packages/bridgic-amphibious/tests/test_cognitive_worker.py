"""Tests for CognitiveWorker: template methods, type checks, full observe-think-act cycle."""
import os
import types
import pytest
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Tuple

from bridgic.core.cognitive import (
    CognitiveContext,
    CognitiveWorker,
    ActionStepResult,
    StepToolCall,
    ToolArgument,
    DetailRequest,
    _DELEGATE,
    AmphibiousAutoma,
)
from bridgic.core.model.types import ToolCall
from .tools import get_travel_planning_tools

# Default decision model for mock LLM responses (no policies, no output_schema)
ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def _tr(**kwargs):
    """Create a simple namespace mock for intermediate (non-final) thinking rounds.

    Simulates the dynamic _ThinkResultModel in tests: has step_content, output,
    finish, and optionally details_needed / rehearsal / reflection attributes.
    MockLLM returns this directly, bypassing schema validation.
    """
    defaults = {"step_content": "", "output": [], "finish": False, "details": []}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


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
    """Worker with observation + build_messages overrides."""
    async def thinking(self):
        return "Plan ONE step"

    async def observation(self, context):
        return "Custom observation: environment is ready"

    async def build_messages(self, think_prompt, tools_description, output_instructions, context_info):
        from bridgic.core.model.types import Message
        extra = "EXTRA_INSTRUCTION: Always prefer cheapest option."
        system = f"{think_prompt}\n\n{extra}\n\n{tools_description}\n\n{output_instructions}"
        return [
            Message.from_text(text=system, role="system"),
            Message.from_text(text=context_info, role="user"),
        ]


class _ActionPipelineWorker(CognitiveWorker):
    """Worker with before_action override."""
    async def thinking(self):
        return "Plan ONE step"

    async def before_action(self, matched_list, context):
        return [(tc, spec) for tc, spec in matched_list if spec.tool_name != "book_flight"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveWorker:

    @pytest.mark.asyncio
    async def test_template_method_defaults(self):
        """Default template methods: thinking raises, observation returns _DELEGATE,
        before_action passes through, build_thinking_prompt assembles, non-context rejected."""
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()

        # thinking() → NotImplementedError
        with pytest.raises(NotImplementedError):
            await worker.thinking()

        # observation() → _DELEGATE (delegate to Agent)
        result = await worker.observation(ctx)
        assert result is _DELEGATE

        # before_action() → _DELEGATE (delegate to Agent)
        tools = get_travel_planning_tools()
        matched = [(ToolCall(id="1", name="search_flights", arguments={}), tools[0])]
        assert await worker.before_action(matched, ctx) is _DELEGATE

        # build_messages() → standard assembly: system + optional tools + user
        from bridgic.core.model.types import Message
        messages = await worker.build_messages(
            think_prompt="Plan next step",
            tools_description="Tool A, Tool B",
            output_instructions="Output JSON",
            context_info="Goal: test"
        )
        assert isinstance(messages, list)
        assert messages[0].role == "system"
        assert "Plan next step" in messages[0].content
        assert "Output JSON" in messages[0].content
        # tools_description goes in a separate user message
        assert any("Tool A, Tool B" in m.content for m in messages)
        # context_info in last user message
        assert messages[-1].role == "user"
        assert "Goal: test" in messages[-1].content

        # Type check: rejects non-CognitiveContext
        class SimpleWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan"

        llm2 = MockLLM()
        llm2.structured_output_response = ThinkDecision(step_content="", output=[])
        w = SimpleWorker(llm=llm2)
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await w.arun(context="not a context")

    @pytest.mark.asyncio
    async def test_override_prompt_customization(self):
        """Override observation + build_thinking_prompt: custom text appears in LLM messages."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            output=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )
        worker = _PromptCustomWorker(llm=llm)
        ctx = _make_context()

        # Simulate AmphibiousAutoma._run(): call observation(), write to ctx.observation, then arun()
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
        """Worker.arun() only performs thinking, returns decision directly."""
        llm = MockLLM()
        ctx = _make_context()
        worker = _SimpleWorker(llm=llm)

        # Tool call scenario - decision returned from arun()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights from Beijing to Tokyo",
            output=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )
        decision = await worker.arun(context=ctx)

        # Worker returns decision, does NOT execute tools
        assert decision is not None
        assert decision.step_content == "Search flights from Beijing to Tokyo"
        assert len(decision.output) == 1
        assert decision.output[0].tool == "search_flights"
        # No history added (action not executed)
        assert len(ctx.cognitive_history) == 0

    @pytest.mark.asyncio
    async def test_cycle_via_agent(self):
        """End-to-end: worker via agent executes full observe-think-act cycle."""
        llm = StatefulMockLLM([
            ThinkDecision(
                step_content="Search flights from Beijing to Tokyo",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
            ThinkDecision(step_content="All done", output=[]),
        ])
        worker = _SimpleWorker(llm=llm)

        class SimpleAgent(AmphibiousAutoma[TravelCtx]):
            async def on_agent(self, ctx):
                await self._run(worker)  # search_flights
                await self._run(worker)  # no tools

        agent = SimpleAgent(llm=llm)
        await agent.arun(goal="Plan a trip to Tokyo")

        assert len(agent._current_context.cognitive_history) == 2
        # Step 1: search_flights was executed
        assert agent._current_context.cognitive_history[0].result.results[0].tool_name == "search_flights"

    @pytest.mark.asyncio
    async def test_override_action_pipeline_via_agent(self):
        """before_action: book_flight filtered via agent."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search and book",
            output=[
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
        )
        worker = _ActionPipelineWorker(llm=llm)

        class PipelineAgent(AmphibiousAutoma[TravelCtx]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = PipelineAgent(llm=llm)
        await agent.arun(goal="test")

        last_step = agent._current_context.cognitive_history[-1]
        # before_action filtered out book_flight
        tool_names = [r.tool_name for r in last_step.result.results]
        assert "book_flight" not in tool_names
        assert "search_flights" in tool_names

    @pytest.mark.asyncio
    async def test_rehearsal_policy(self):
        """Rehearsal policy: LLM fills rehearsal (optional) → triggers another round → decision."""
        llm = StatefulMockLLM([
            # Round 0: LLM chooses to fill rehearsal (optional operator)
            _tr(rehearsal="Predicted: search_flights will return 3 flights"),
            # Round 1: Decision (no more operators activated)
            ThinkDecision(
                step_content="Search flights",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
        ])

        worker = CognitiveWorker.from_prompt(
            "Plan ONE step",
            llm=llm,
            enable_rehearsal=True
        )
        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        # Check decision was returned
        assert decision is not None
        assert decision.step_content == "Search flights"

    @pytest.mark.asyncio
    async def test_reflection_policy(self):
        """Reflection policy: LLM fills reflection (optional) → triggers another round → decision."""
        llm = StatefulMockLLM([
            # Round 0: LLM chooses to fill reflection (optional operator)
            _tr(reflection="Information is sufficient, no contradictions found"),
            # Round 1: Decision (no more operators activated)
            ThinkDecision(
                step_content="Search flights",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
        ])

        worker = CognitiveWorker.from_prompt(
            "Plan ONE step",
            llm=llm,
            enable_reflection=True
        )
        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        assert decision is not None
        assert decision.step_content == "Search flights"

    @pytest.mark.asyncio
    async def test_combined_policies(self):
        """Combined policies: LLM fills both rehearsal + reflection → triggers another round → decision."""
        llm = StatefulMockLLM([
            # Round 0: LLM fills both optional operators
            _tr(rehearsal="Will return 3 flights", reflection="Information sufficient"),
            # Round 1: Decision (no more operators)
            ThinkDecision(
                step_content="Search flights",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
        ])

        worker = CognitiveWorker.from_prompt(
            "Plan ONE step",
            llm=llm,
            enable_rehearsal=True,
            enable_reflection=True
        )
        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        assert decision is not None
        assert decision.step_content == "Search flights"

    @pytest.mark.asyncio
    async def test_policies_with_disclosure(self):
        """Policies + disclosure: rehearsal triggers round 2, details_needed triggers round 3 (forced decision)."""
        llm = StatefulMockLLM([
            # Round 0: LLM fills rehearsal operator
            _tr(rehearsal="Need to see skill details first"),
            # Round 1: LLM requests details (single batch disclosure)
            _tr(details=[DetailRequest(field="skills", index=0)]),
            # Round 2: Decision (disclosure_done=True → no details field offered)
            ThinkDecision(
                step_content="Execute skill workflow",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
        ])

        worker = CognitiveWorker.from_prompt(
            "Plan ONE step",
            llm=llm,
            enable_rehearsal=True
        )
        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        assert decision is not None
        assert decision.step_content == "Execute skill workflow"

    @pytest.mark.asyncio
    async def test_observation_enhancement(self):
        """Observation enhancement: worker receives and enhances default observation."""
        class EnhancementWorker(CognitiveWorker):
            async def thinking(self):
                return "Plan ONE step"

            async def observation(self, context, default_observation=None):
                if default_observation is None:
                    return _DELEGATE
                # Enhance with history count
                return f"{default_observation}\n\nSteps completed: {len(context.cognitive_history)}"

        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            output=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )
        worker = EnhancementWorker(llm=llm)

        class EnhancementAgent(AmphibiousAutoma[TravelCtx]):
            async def observation(self, ctx):
                return "Default observation from agent"

            async def on_agent(self, ctx):
                await self._run(worker)

        agent = EnhancementAgent(llm=llm)
        await agent.arun(goal="test")

        # Check that agent-level observation was used (worker delegates via _DELEGATE)
        user_msg = llm.captured_messages[-1]
        assert "Default observation from agent" in user_msg.content

    @pytest.mark.asyncio
    async def test_from_prompt_convenience(self):
        """from_prompt() creates a worker without defining a class."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            output=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )

        worker = CognitiveWorker.from_prompt(
            "Plan ONE immediate next step",
            llm=llm,
            enable_rehearsal=False
        )
        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        assert decision is not None
        assert decision.step_content == "Search flights"

    @pytest.mark.asyncio
    async def test_inline_worker_via_agent(self):
        """CognitiveWorker.inline() creates a worker used with _run() in on_agent."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            output=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )

        worker = CognitiveWorker.inline(
            "Plan ONE step",
            llm=llm,
            enable_rehearsal=False
        )

        class SimpleAgent(AmphibiousAutoma[TravelCtx]):
            async def on_agent(self, ctx):
                await self._run(worker)

        agent = SimpleAgent(llm=llm)
        await agent.arun(goal="test")
        assert len(agent._current_context.cognitive_history) == 1
        assert agent._current_context.cognitive_history[0].content == "Search flights"

    @pytest.mark.asyncio
    async def test_dynamic_model_fields(self):
        """Dynamic ThinkResult model always has details; policy fields added based on flags."""
        # No policies: step_content + calls + details (disclosure is always built-in)
        w_none = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=False,
            enable_reflection=False,
        )
        fields_none = set(w_none._ThinkResultModel.model_fields.keys())
        assert fields_none == {"step_content", "output", "finish", "details"}

        # Rehearsal only: adds rehearsal
        w_reh = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=True,
            enable_reflection=False,
        )
        fields_reh = set(w_reh._ThinkResultModel.model_fields.keys())
        assert fields_reh == {"step_content", "output", "finish", "details", "rehearsal"}

        # Reflection only: adds reflection
        w_ref = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=False,
            enable_reflection=True,
        )
        fields_ref = set(w_ref._ThinkResultModel.model_fields.keys())
        assert fields_ref == {"step_content", "output", "finish", "details", "reflection"}

        # Both policies: all fields present
        w_all = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=True,
            enable_reflection=True,
        )
        fields_all = set(w_all._ThinkResultModel.model_fields.keys())
        assert fields_all == {"step_content", "output", "finish", "details", "rehearsal", "reflection"}

    @pytest.mark.asyncio
    async def test_single_round_batch_disclosure(self):
        """Disclosure is single-round batch: LLM requests details, all expanded, then ThinkDecision forced."""
        llm = StatefulMockLLM([
            # Round 1: Request details (batch of multiple items)
            _tr(details=[
                DetailRequest(field="skills", index=0),
                DetailRequest(field="cognitive_history", index=0),
            ]),
            # Round 2: Forced decision (ThinkDecision model, no details_needed field)
            ThinkDecision(
                step_content="Direct decision after disclosure",
                output=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
        ])

        worker = CognitiveWorker.from_prompt("Plan ONE step", llm=llm)
        # _ThinkResultModel always has details
        assert "details" in worker._ThinkResultModel.model_fields

        ctx = _make_context()
        decision = await worker.arun(context=ctx)

        assert decision is not None
        assert decision.step_content == "Direct decision after disclosure"

    @pytest.mark.asyncio
    async def test_dynamic_model_cache(self):
        """Same policy combination reuses the same model class."""
        w1 = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_rehearsal=True)
        w2 = CognitiveWorker.from_prompt("Different prompt", llm=MockLLM(), enable_rehearsal=True)
        # Same policy config → same cached model class
        assert w1._ThinkResultModel is w2._ThinkResultModel

        w3 = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_reflection=True)
        # Different policy config → different model class
        assert w1._ThinkResultModel is not w3._ThinkResultModel

    def test_prompt_instructions_match_enabled_policies(self):
        """_build_output_instructions describes fields matching enabled policies.

        details is always in normal rounds (acquiring_open=True).
        Policy fields (rehearsal/reflection) appear as optional when their operator is open.
        After acquiring closes (acquiring_open=False), details guidance is removed.
        """
        # Default (no policies): normal round has details, no policy fields
        w_default = CognitiveWorker.from_prompt("Plan", llm=MockLLM())
        instr = w_default._build_output_instructions(acquiring_open=True, rehearsal_open=False, reflection_open=False)
        assert "details" in instr
        assert "rehearsal" not in instr
        assert "reflection" not in instr

        # After acquiring closes: no details
        instr_done = w_default._build_output_instructions(acquiring_open=False, rehearsal_open=False, reflection_open=False)
        assert "details" not in instr_done

        # enable_rehearsal=True: normal round has both details AND rehearsal (optional)
        w_reh = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_rehearsal=True)
        instr = w_reh._build_output_instructions(acquiring_open=True, rehearsal_open=True, reflection_open=False)
        assert "rehearsal" in instr
        assert "details" in instr
        assert "reflection" not in instr

        # enable_reflection=True: normal round has both details AND reflection (optional)
        w_ref = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_reflection=True)
        instr = w_ref._build_output_instructions(acquiring_open=True, rehearsal_open=False, reflection_open=True)
        assert "reflection" in instr
        assert "details" in instr
        assert "rehearsal" not in instr

        # All policies: all fields present in normal round
        w_all = CognitiveWorker.from_prompt(
            "Plan", llm=MockLLM(),
            enable_rehearsal=True, enable_reflection=True
        )
        instr_all = w_all._build_output_instructions(acquiring_open=True, rehearsal_open=True, reflection_open=True)
        assert "details" in instr_all
        assert "rehearsal" in instr_all
        assert "reflection" in instr_all

        # After acquiring closes with policies still open: still no details
        instr_done_all = w_all._build_output_instructions(acquiring_open=False, rehearsal_open=True, reflection_open=True)
        assert "details" not in instr_done_all

        # No progressive pressure (removed): no "Detail Budget" regardless of round count
        instr_a = w_default._build_output_instructions(acquiring_open=True, rehearsal_open=False, reflection_open=False)
        assert "Detail Budget" not in instr_a
        assert "LIMIT REACHED" not in instr_a


# ---------------------------------------------------------------------------
# Tests for Feature 3: CognitiveWorker.output_type
# ---------------------------------------------------------------------------

from pydantic import BaseModel
from typing import List as _List


class _PlanPhase(BaseModel):
    sub_goal: str
    skill_name: str


class _PlanResult(BaseModel):
    phases: _List[_PlanPhase]


class _PlannerWorker(CognitiveWorker):
    """Worker with output_schema set — returns _PlanResult directly."""
    output_schema = _PlanResult

    async def thinking(self) -> str:
        return "Produce a phased execution plan."


class TestOutputType:

    @pytest.mark.asyncio
    async def test_output_schema_returns_typed_instance(self):
        """When output_schema is set, arun() returns a ThinkDecision wrapping the typed output."""
        expected_output = _PlanResult(phases=[
            _PlanPhase(sub_goal="Step A", skill_name="skill-a"),
        ])
        worker = _PlannerWorker(llm=MockLLM())
        decision_model = worker._ThinkDecisionModel
        decision_instance = decision_model(
            step_content="Planning complete", output=expected_output, finish=False
        )

        llm = MockLLM()
        llm.structured_output_response = decision_instance
        worker = _PlannerWorker(llm=llm)
        ctx = CognitiveContext(goal="Test")
        ctx.observation = None

        # worker.arun() returns the ThinkDecision wrapper
        result = await worker.arun(context=ctx)

        assert isinstance(result, decision_model)
        assert result.output is expected_output
        assert isinstance(result.output, _PlanResult)
        assert result.output.phases[0].sub_goal == "Step A"

    @pytest.mark.asyncio
    async def test_output_schema_skips_action(self):
        """When output_schema is set, the typed output is stored in context history."""
        expected_output = _PlanResult(phases=[
            _PlanPhase(sub_goal="Phase 1", skill_name="skill-1"),
        ])

        planner_worker = _PlannerWorker(llm=MockLLM())

        class _TrackingAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(planner_worker)

        llm = MockLLM()
        planner_worker.set_llm(llm)
        decision_model = planner_worker._ThinkDecisionModel
        llm.structured_output_response = decision_model(
            step_content="Done", output=expected_output, finish=False
        )

        agent = _TrackingAgent(llm=llm)
        await agent.arun(goal="Test output_schema")

        # The typed output is stored as the step result in history
        last_step = agent._current_context.cognitive_history[-1]
        assert last_step.result is expected_output
        assert isinstance(last_step.result, _PlanResult)

    @pytest.mark.asyncio
    async def test_output_schema_uses_schema_constraint(self):
        """LLM is called with ThinkDecision model (wrapping output_schema) as constraint."""
        from bridgic.core.model.protocols import PydanticModel

        captured_constraint = []
        worker_ref = []

        class _CapturingLLM(MockLLM):
            async def astructured_output(self, messages, constraint, **kwargs):
                captured_constraint.append(constraint)
                # Return a proper ThinkDecision instance with the output wrapped
                w = worker_ref[0]
                dm = w._ThinkDecisionModel
                return dm(step_content="", output=_PlanResult(phases=[]), finish=False)

        worker = _PlannerWorker(llm=_CapturingLLM())
        worker_ref.append(worker)
        ctx = CognitiveContext(goal="Test")
        ctx.observation = None

        await worker.arun(context=ctx)

        assert len(captured_constraint) == 1
        assert isinstance(captured_constraint[0], PydanticModel)
        # Constraint is the ThinkDecision model (wrapping _PlanResult in 'output' field)
        model = captured_constraint[0].model
        assert 'output' in model.model_fields
        assert 'finish' in model.model_fields
        assert 'step_content' in model.model_fields


class TestFinishSignal:
    """Tests for finish=True signal stopping the loop early via _run(max_attempts=...)."""

    @pytest.mark.asyncio
    async def test_finish_true_stops_run_loop(self):
        """When LLM sets finish=True, _run(max_attempts=...) stops after that round."""

        call_idx = [0]

        class _FinishLLM(MockLLM):
            async def astructured_output(self, messages, constraint, **kwargs):
                idx = call_idx[0]
                call_idx[0] += 1
                # Return a decision; finish=True on second call
                return ThinkDecision(
                    step_content="done" if idx == 1 else "step",
                    output=[],
                    finish=(idx == 1),
                )

        worker = CognitiveWorker.inline("Plan one step.")

        class _SimpleAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=10)

        agent = _SimpleAgent(llm=_FinishLLM())
        await agent.arun(goal="Test finish signal")

        assert call_idx[0] == 2, f"Expected 2 LLM calls, got {call_idx[0]}"

    @pytest.mark.asyncio
    async def test_finish_false_continues_loop(self):
        """When finish is always False and no condition, loop runs max_attempts times."""
        call_idx = [0]

        class _NeverFinishLLM(MockLLM):
            async def astructured_output(self, messages, constraint, **kwargs):
                call_idx[0] += 1
                return ThinkDecision(step_content="step", output=[], finish=False)

        worker = CognitiveWorker.inline("Plan one step.")

        class _LoopAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(worker, max_attempts=3)

        agent = _LoopAgent(llm=_NeverFinishLLM())
        await agent.arun(goal="Test no finish")

        assert call_idx[0] == 3, f"Expected 3 LLM calls, got {call_idx[0]}"


class TestActionDefensive:
    """Tests for _action() handling of output_schema decisions."""

    @pytest.mark.asyncio
    async def test_action_custom_output_stores_result(self):
        """_action() stores custom output in step when decision is not a tool-call list."""
        from pydantic import BaseModel

        class _MySchema(BaseModel):
            value: str

        worker = CognitiveWorker.inline("Plan.", output_schema=_MySchema)

        class _SchemaAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx): pass

        agent = _SchemaAgent(llm=MockLLM())
        ctx = CognitiveContext(goal="Test")
        agent._current_context = ctx

        # Create an output_schema decision where output is _MySchema, not a list
        schema_decision = type('SchemaDecision', (), {
            'output': _MySchema(value="result"),
            'step_content': "done",
        })()

        await agent._action(schema_decision, ctx, _worker=worker)
        last_step = ctx.cognitive_history._items[-1]
        assert last_step.content == "done"
        assert last_step.result.value == "result"


class TestSkillRevealPersistence:
    """Tests that skill _revealed state persists across until() iterations via write-back."""

    @pytest.mark.asyncio
    async def test_revealed_state_persists_across_iterations(self):
        """Skills revealed in iteration N are available in iteration N+1 via index remapping."""
        from bridgic.core.cognitive import CognitiveSkills

        ctx = _make_context()  # already loads skills from SKILLS_DIR

        # Pre-reveal skill[0] on original_skills
        ctx.get_details("skills", 0)
        assert 0 in ctx.skills._revealed

        # Simulate what _run() does when tools filter is active:
        # create filtered_skills with only skill[0], copy reveals, then write back.
        original_skills = ctx.skills
        filtered_skills = CognitiveSkills()
        orig_to_filtered: Dict[int, int] = {}
        filtered_to_orig: Dict[int, int] = {}

        for orig_idx, skill in enumerate(original_skills.get_all()):
            new_idx = len(filtered_skills)
            filtered_skills.add(skill)
            orig_to_filtered[orig_idx] = new_idx
            filtered_to_orig[new_idx] = orig_idx

        # Forward-copy reveals
        for orig_idx, detail in original_skills._revealed.items():
            if orig_idx in orig_to_filtered:
                filtered_skills._revealed[orig_to_filtered[orig_idx]] = detail

        ctx.skills = filtered_skills

        # Inside iteration: skill[0] should be visible as revealed in filtered
        assert 0 in ctx.skills._revealed

        # Simulate revealing skill[1] inside the iteration
        ctx.get_details("skills", 1)
        assert 1 in ctx.skills._revealed

        # Write back: both 0 and 1 should propagate to original
        for filtered_idx, detail in filtered_skills._revealed.items():
            orig_idx = filtered_to_orig.get(filtered_idx)
            if orig_idx is not None:
                original_skills._revealed[orig_idx] = detail
        ctx.skills = original_skills

        # After write-back: both 0 and 1 in original_skills._revealed
        assert 0 in ctx.skills._revealed
        assert 1 in ctx.skills._revealed
