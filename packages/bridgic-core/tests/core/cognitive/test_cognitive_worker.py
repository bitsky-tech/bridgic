"""Tests for CognitiveWorker: template methods, type checks, full observe-think-act cycle."""
import os
import types
import pytest
from unittest.mock import MagicMock
from typing import Any, List, Optional, Tuple

from bridgic.core.cognitive import (
    CognitiveContext,
    CognitiveWorker,
    ThinkDecision,
    ActionStepResult,
    StepToolCall,
    ToolArgument,
    DetailRequest,
    _DELEGATE,
    AgentAutoma,
    think_step,
)
from bridgic.core.model.types import ToolCall
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def _tr(**kwargs):
    """Create a simple namespace mock for intermediate (non-final) thinking rounds.

    Simulates the dynamic _ThinkResultModel in tests: has step_content, calls,
    and optionally details_needed / rehearsal / reflection attributes.
    MockLLM returns this directly, bypassing schema validation.
    """
    defaults = {"step_content": "", "calls": [], "details_needed": []}
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
    """Worker with before_action + after_action overrides."""
    async def thinking(self):
        return "Plan ONE step"

    async def before_action(self, matched_list, context):
        return [(tc, spec) for tc, spec in matched_list if spec.tool_name != "book_flight"]

    async def after_action(self, action_results, context):
        return "; ".join(r.tool_result for r in action_results)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveWorker:

    @pytest.mark.asyncio
    async def test_template_method_defaults(self):
        """Default template methods: thinking raises, observation returns _DELEGATE,
        before_action/after_action pass through, build_thinking_prompt assembles, non-context rejected."""
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()

        # thinking() → NotImplementedError
        with pytest.raises(NotImplementedError):
            await worker.thinking()

        # observation() → _DELEGATE (delegate to Agent)
        result = await worker.observation(ctx)
        assert result is _DELEGATE

        # before_action() → passthrough
        tools = get_travel_planning_tools()
        matched = [(ToolCall(id="1", name="search_flights", arguments={}), tools[0])]
        assert await worker.before_action(matched, ctx) is matched

        # after_action() → passthrough (returns raw list)
        test_results = [
            ActionStepResult(
                tool_id="1", tool_name="search_flights",
                tool_arguments={"origin": "Beijing"}, tool_result="found"
            )
        ]
        assert await worker.after_action(test_results, ctx) is test_results

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
        llm2.structured_output_response = ThinkDecision(step_content="", calls=[])
        w = SimpleWorker(llm=llm2)
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await w.arun(context="not a context")

    @pytest.mark.asyncio
    async def test_override_prompt_customization(self):
        """Override observation + build_thinking_prompt: custom text appears in LLM messages."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            calls=[StepToolCall(
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
        """Worker.arun() only performs thinking, returns decision directly."""
        llm = MockLLM()
        ctx = _make_context()
        worker = _SimpleWorker(llm=llm)

        # Tool call scenario - decision returned from arun()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights from Beijing to Tokyo",
            calls=[StepToolCall(
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
        assert len(decision.calls) == 1
        assert decision.calls[0].tool == "search_flights"
        # No history added (action not executed)
        assert len(ctx.cognitive_history) == 0

    @pytest.mark.asyncio
    async def test_cycle_via_agent(self):
        """End-to-end: worker via agent executes full observe-think-act cycle."""
        llm = StatefulMockLLM([
            ThinkDecision(
                step_content="Search flights from Beijing to Tokyo",
                calls=[StepToolCall(
                    tool="search_flights",
                    tool_arguments=[
                        ToolArgument(name="origin", value="Beijing"),
                        ToolArgument(name="destination", value="Tokyo"),
                        ToolArgument(name="date", value="2025-06-01"),
                    ]
                )],
            ),
            ThinkDecision(step_content="All done", calls=[]),
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
        """before_action + after_action: book_flight filtered, result formatted as string via agent."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
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
        )
        worker = _ActionPipelineWorker(llm=llm)

        class PipelineAgent(AgentAutoma[TravelCtx]):
            step = think_step(worker)
            async def cognition(self, ctx):
                await self.step

        result = await PipelineAgent().arun(goal="test")

        last_step = result.cognitive_history[-1]
        # before_action filtered out book_flight
        assert "book_flight" not in last_step.metadata.get("tool_calls", [])
        assert "search_flights" in last_step.metadata.get("tool_calls", [])
        # after_action formatted as joined string
        assert isinstance(last_step.result, str)
        assert "Found 3 available flights" in last_step.result

    @pytest.mark.asyncio
    async def test_rehearsal_policy(self):
        """Rehearsal policy: LLM outputs rehearsal content, then makes decision."""
        llm = StatefulMockLLM([
            # Round 0: Policy round (rehearsal)
            _tr(rehearsal="Predicted: search_flights will return 3 flights"),
            # Round 1: Decision
            ThinkDecision(
                step_content="Search flights",
                calls=[StepToolCall(
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
        """Reflection policy: LLM outputs reflection content, then makes decision."""
        llm = StatefulMockLLM([
            # Round 0: Policy round (reflection)
            _tr(reflection="Information is sufficient, no contradictions found"),
            # Round 1: Decision
            ThinkDecision(
                step_content="Search flights",
                calls=[StepToolCall(
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
        """Combined policies: rehearsal + reflection in same round."""
        llm = StatefulMockLLM([
            # Round 0: Policy round (both)
            _tr(rehearsal="Will return 3 flights", reflection="Information sufficient"),
            # Round 1: Decision
            ThinkDecision(
                step_content="Search flights",
                calls=[StepToolCall(
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
        """Policies + single-round batch disclosure: policy round → detail batch → forced decision."""
        llm = StatefulMockLLM([
            # Round 0: Policy round
            _tr(rehearsal="Need to see skill details first"),
            # Round 1: Request details (single batch)
            _tr(details_needed=[DetailRequest(field="skills", index=0)]),
            # Round 2: Decision (using ThinkDecision model — disclosure_done=True)
            ThinkDecision(
                step_content="Execute skill workflow",
                calls=[StepToolCall(
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
            calls=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )
        worker = EnhancementWorker(llm=llm)

        class EnhancementAgent(AgentAutoma[TravelCtx]):
            step = think_step(worker)

            async def observation(self, ctx):
                return "Default observation from agent"

            async def cognition(self, ctx):
                await self.step

        result = await EnhancementAgent().arun(goal="test")

        # Check that enhanced observation was used
        user_msg = llm.captured_messages[-1]
        assert "Default observation from agent" in user_msg.content
        assert "Steps completed: 0" in user_msg.content

    @pytest.mark.asyncio
    async def test_from_prompt_convenience(self):
        """from_prompt() creates a worker without defining a class."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            calls=[StepToolCall(
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
    async def test_think_step_from_prompt(self):
        """think_step.from_prompt() creates a step without defining a worker class."""
        llm = MockLLM()
        llm.structured_output_response = ThinkDecision(
            step_content="Search flights",
            calls=[StepToolCall(
                tool="search_flights",
                tool_arguments=[
                    ToolArgument(name="origin", value="Beijing"),
                    ToolArgument(name="destination", value="Tokyo"),
                    ToolArgument(name="date", value="2025-06-01"),
                ]
            )],
        )

        class SimpleAgent(AgentAutoma[TravelCtx]):
            step = think_step.from_prompt(
                "Plan ONE step",
                llm=llm,
                enable_rehearsal=False
            )

            async def cognition(self, ctx):
                await self.step

        result = await SimpleAgent().arun(goal="test")
        assert len(result.cognitive_history) == 1
        assert result.cognitive_history[0].content == "Search flights"

    @pytest.mark.asyncio
    async def test_dynamic_model_fields(self):
        """Dynamic ThinkResult model always has details_needed; policy fields added based on flags."""
        # No policies: step_content + calls + details_needed (disclosure is always built-in)
        w_none = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=False,
            enable_reflection=False,
        )
        fields_none = set(w_none._ThinkResultModel.model_fields.keys())
        assert fields_none == {"step_content", "calls", "details_needed"}

        # Rehearsal only: adds rehearsal
        w_reh = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=True,
            enable_reflection=False,
        )
        fields_reh = set(w_reh._ThinkResultModel.model_fields.keys())
        assert fields_reh == {"step_content", "calls", "details_needed", "rehearsal"}

        # Reflection only: adds reflection
        w_ref = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=False,
            enable_reflection=True,
        )
        fields_ref = set(w_ref._ThinkResultModel.model_fields.keys())
        assert fields_ref == {"step_content", "calls", "details_needed", "reflection"}

        # Both policies: all fields present
        w_all = CognitiveWorker.from_prompt(
            "Plan",
            llm=MockLLM(),
            enable_rehearsal=True,
            enable_reflection=True,
        )
        fields_all = set(w_all._ThinkResultModel.model_fields.keys())
        assert fields_all == {"step_content", "calls", "details_needed", "rehearsal", "reflection"}

    @pytest.mark.asyncio
    async def test_single_round_batch_disclosure(self):
        """Disclosure is single-round batch: LLM requests details, all expanded, then ThinkDecision forced."""
        llm = StatefulMockLLM([
            # Round 1: Request details (batch of multiple items)
            _tr(details_needed=[
                DetailRequest(field="skills", index=0),
                DetailRequest(field="cognitive_history", index=0),
            ]),
            # Round 2: Forced decision (ThinkDecision model, no details_needed field)
            ThinkDecision(
                step_content="Direct decision after disclosure",
                calls=[StepToolCall(
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
        # _ThinkResultModel always has details_needed
        assert "details_needed" in worker._ThinkResultModel.model_fields

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

        details_needed is always in non-policy rounds (disclosure is built-in).
        Policy fields (rehearsal/reflection) appear only in policy rounds.
        After disclosure (disclosure_done=True), details_needed guidance is removed.
        """
        ctx = _make_context()

        # Default (no policies): non-policy round has details_needed, no policy fields
        w_default = CognitiveWorker.from_prompt("Plan", llm=MockLLM())
        instr = w_default._build_output_instructions(is_policy_round=False, disclosure_done=False, context=ctx)
        assert "details_needed" in instr
        assert "rehearsal" not in instr
        assert "reflection" not in instr

        # After disclosure done: no details_needed (ThinkDecision model is used)
        instr_done = w_default._build_output_instructions(is_policy_round=False, disclosure_done=True, context=ctx)
        assert "details_needed" not in instr_done

        # enable_rehearsal=True: policy round has rehearsal, not details_needed
        w_reh = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_rehearsal=True)
        instr = w_reh._build_output_instructions(is_policy_round=True, disclosure_done=False, context=ctx)
        assert "rehearsal" in instr
        assert "details_needed" not in instr
        assert "reflection" not in instr

        # enable_reflection=True: policy round has reflection, not others
        w_ref = CognitiveWorker.from_prompt("Plan", llm=MockLLM(), enable_reflection=True)
        instr = w_ref._build_output_instructions(is_policy_round=True, disclosure_done=False, context=ctx)
        assert "reflection" in instr
        assert "rehearsal" not in instr
        assert "details_needed" not in instr

        # All policies: non-policy round has details_needed (not policy fields)
        w_all = CognitiveWorker.from_prompt(
            "Plan", llm=MockLLM(),
            enable_rehearsal=True, enable_reflection=True
        )
        instr_detail = w_all._build_output_instructions(is_policy_round=False, disclosure_done=False, context=ctx)
        assert "details_needed" in instr_detail
        assert "rehearsal" not in instr_detail
        assert "reflection" not in instr_detail

        # Policy round has both policy fields, not details_needed
        instr_policy = w_all._build_output_instructions(is_policy_round=True, disclosure_done=False, context=ctx)
        assert "rehearsal" in instr_policy
        assert "reflection" in instr_policy
        assert "details_needed" not in instr_policy

        # No progressive pressure (removed): no "Detail Budget" regardless of round count
        instr_a = w_default._build_output_instructions(is_policy_round=False, disclosure_done=False, context=ctx)
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


class _MockAgentForOutputType(AgentAutoma[CognitiveContext]):
    """Minimal agent wrapping _PlannerWorker for output_type testing."""
    plan_step = think_step(_PlannerWorker())

    async def cognition(self, ctx: CognitiveContext) -> None:
        pass  # not used directly in these tests


class TestOutputType:

    @pytest.mark.asyncio
    async def test_output_schema_returns_typed_instance(self):
        """When output_schema is set, arun() returns the typed instance directly."""
        expected = _PlanResult(phases=[
            _PlanPhase(sub_goal="Step A", skill_name="skill-a"),
        ])
        llm = MockLLM()
        llm.structured_output_response = expected

        worker = _PlannerWorker(llm=llm)
        ctx = CognitiveContext(goal="Test")
        ctx.observation = None

        result = await worker.arun(context=ctx)

        assert result is expected
        assert isinstance(result, _PlanResult)
        assert result.phases[0].sub_goal == "Step A"

    @pytest.mark.asyncio
    async def test_output_schema_skips_action(self):
        """When output_schema is set, agent.action() is NOT called; await step returns typed instance."""
        expected = _PlanResult(phases=[
            _PlanPhase(sub_goal="Phase 1", skill_name="skill-1"),
        ])
        llm = MockLLM()
        llm.structured_output_response = expected

        # Track whether action() was called
        action_called = []

        class _TrackingAgent(_MockAgentForOutputType):
            async def action(self, decision, ctx, *, _worker):
                action_called.append(decision)

        agent = _TrackingAgent(llm=llm)
        ctx = CognitiveContext(goal="Test output_schema")
        agent._current_context = ctx

        # Inject llm into the plan_step worker
        agent.plan_step.worker.set_llm(llm)

        result = await agent.plan_step._execute_once(agent)

        # action() must NOT have been called
        assert len(action_called) == 0

        # The typed result must be returned
        assert result is expected
        assert isinstance(result, _PlanResult)

    @pytest.mark.asyncio
    async def test_output_schema_uses_schema_constraint(self):
        """LLM is called with the output_schema model as constraint (not ThinkDecision)."""
        from bridgic.core.model.protocols import PydanticModel

        captured_constraint = []

        class _CapturingLLM(MockLLM):
            async def astructured_output(self, messages, constraint, **kwargs):
                captured_constraint.append(constraint)
                return _PlanResult(phases=[])

        worker = _PlannerWorker(llm=_CapturingLLM())
        ctx = CognitiveContext(goal="Test")
        ctx.observation = None

        await worker.arun(context=ctx)

        assert len(captured_constraint) == 1
        assert isinstance(captured_constraint[0], PydanticModel)
        assert captured_constraint[0].model is _PlanResult

