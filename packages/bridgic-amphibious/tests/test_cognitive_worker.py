"""Tests for CognitiveWorker: the ``thinking`` template method, type
checks, and the full observe-think-act cycle.

After the refactor, ``CognitiveWorker`` owns exactly one template method:
``thinking(self, context) -> (content, tool_calls)``. The default
implementation talks to the LLM via native function-calling
(``aselect_tool`` when the context carries tools, ``achat`` otherwise);
the framework parses the returned pair into a ``ThinkDecision`` whose
``finish`` flag is *derived* from the absence of tool calls. Workers with
an ``output_schema`` take the structured-output path instead.
"""
import os
import pytest
from typing import Any, List, Optional, Tuple

from bridgic.amphibious import (
    CognitiveContext,
    CognitiveWorker,
    StepToolCall,
    ToolArgument,
    _DELEGATE,
    AmphibiousAutoma,
    ThinkUnit,
    think_unit,
)
from bridgic.amphibious._type import ThinkDecision
from bridgic.core.model.types import Message, Response, Role, ToolCall
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


def _txt(text: str) -> Response:
    """Build a chat ``Response`` carrying plain text (for ``achat``)."""
    return Response(message=Message.from_text(text, role=Role.AI))


def _tool_call(tool: str, **arguments) -> dict:
    """Build a plain-dict tool call as ``aselect_tool`` would return."""
    return {"name": tool, "arguments": arguments}


class MockLLM:
    """Drives the new CognitiveWorker's native function-calling path.

    ``aselect_tool`` returns the scripted ``(tool_calls, content)`` pair;
    ``achat`` returns the scripted ``Response``. ``astructured_output``
    returns the scripted instance for ``output_schema`` workers.
    ``captured_messages`` holds the last message list sent to the LLM.
    """

    def __init__(self):
        self.select_tool_response: Tuple[List[Any], str] = ([], "")
        self.chat_response: Response = _txt("")
        self.structured_output_response: Any = None
        self.captured_messages: List[Any] = []
        self.captured_constraint: Any = None

    async def aselect_tool(self, messages, tools, **kwargs):
        self.captured_messages = messages
        return self.select_tool_response

    async def achat(self, messages, **kwargs):
        self.captured_messages = messages
        return self.chat_response

    async def astructured_output(self, messages, constraint, **kwargs):
        self.captured_messages = messages
        self.captured_constraint = constraint
        return self.structured_output_response

    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class StatefulMockLLM:
    """Returns a sequence of scripted ``aselect_tool`` responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0

    async def aselect_tool(self, messages, tools, **kwargs):
        self.call_count += 1
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs):
        self.call_count += 1
        return _txt("Done")

    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


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


# Scripted ``aselect_tool`` replies reused across tests.
def _search_flights_step():
    """One ``search_flights`` tool call (worker continues)."""
    return (
        [_tool_call(
            "search_flights",
            origin="Beijing",
            destination="Tokyo",
            date="2025-06-01",
        )],
        "Search flights from Beijing to Tokyo",
    )


def _finish_step(content: str = "All done"):
    """No tool calls → the framework derives ``finish=True``."""
    return ([], content)


# ---------------------------------------------------------------------------
# Custom workers
# ---------------------------------------------------------------------------


class _PromptCustomWorker(CognitiveWorker):
    """Worker that customizes ``prompt`` and the ``observation`` hook.

    Both surfaces should reach the LLM through the default ``thinking``:
    ``prompt`` becomes the system message, the observation lands in the
    user message.
    """
    prompt = (
        "Plan ONE step\n\n"
        "EXTRA_INSTRUCTION: Always prefer cheapest option."
    )

    async def observation(self, context):
        return "Custom observation: environment is ready"


class _ActionPipelineWorker(CognitiveWorker):
    """Worker whose ``before_action`` hook filters out ``book_flight``.

    The hook receives the matched ``[(ToolCall, ToolSpec), ...]`` list —
    returning a filtered list overrides which tools the act phase runs.
    """
    prompt = "Plan ONE step"

    async def before_action(self, decision_result, context):
        return [
            (tc, spec) for tc, spec in decision_result
            if spec.tool_name != "book_flight"
        ]


# ---------------------------------------------------------------------------
# Tests — template method + defaults
# ---------------------------------------------------------------------------


class TestCognitiveWorker:

    @pytest.mark.asyncio
    async def test_template_method_defaults(self):
        """Default hooks delegate; default ``thinking`` talks to the LLM
        and returns ``(content, tool_calls)``; non-context input is
        rejected by ``arun``."""
        llm = MockLLM()
        worker = CognitiveWorker(llm=llm)
        ctx = _make_context()

        # observation() → _DELEGATE (delegate to the agent-level hook)
        assert await worker.observation(ctx) is _DELEGATE

        # before_action() → _DELEGATE (delegate to the agent-level hook)
        tools = get_travel_planning_tools()
        matched = [(ToolCall(id="1", name="search_flights", arguments={}), tools[0])]
        assert await worker.before_action(matched, ctx) is _DELEGATE

        # Default thinking(): the context carries tools, so it routes to
        # aselect_tool and returns the (content, tool_calls) pair verbatim.
        llm.select_tool_response = (
            [_tool_call("search_flights", origin="Beijing")],
            "thinking about flights",
        )
        content, tool_calls = await worker.thinking(ctx)
        assert content == "thinking about flights"
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search_flights"

        # Type check: arun() rejects a non-CognitiveContext.
        with pytest.raises(TypeError, match="Expected CognitiveContext"):
            await worker.arun(context="not a context")

    @pytest.mark.asyncio
    async def test_default_thinking_without_tools_uses_chat(self):
        """When the context carries no tools, default ``thinking`` routes
        to ``achat`` and returns an empty tool-call list."""
        llm = MockLLM()
        llm.chat_response = _txt("plain reasoning, nothing to call")
        worker = CognitiveWorker(llm=llm)
        ctx = CognitiveContext(goal="No tools here")  # no tools added

        content, tool_calls = await worker.thinking(ctx)
        assert content == "plain reasoning, nothing to call"
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_override_prompt_customization(self):
        """A custom ``prompt`` and ``observation`` both surface in the
        messages the default ``thinking`` sends to the LLM."""
        llm = MockLLM()
        llm.select_tool_response = (
            [_tool_call(
                "search_flights",
                origin="Beijing",
                destination="Tokyo",
                date="2025-06-01",
            )],
            "Search flights",
        )
        worker = _PromptCustomWorker(llm=llm)
        ctx = _make_context()

        # Simulate AmphibiousAutoma: run observation(), write it to
        # ctx.observation, then run the worker's thinking phase.
        obs = await worker.observation(ctx)
        assert obs == "Custom observation: environment is ready"
        ctx.observation = obs
        await worker.arun(context=ctx)

        # The custom prompt (system message) carries EXTRA_INSTRUCTION.
        system_msg = llm.captured_messages[0]
        assert system_msg.role == "system"
        assert "EXTRA_INSTRUCTION: Always prefer cheapest option" in system_msg.content

        # The custom observation reaches the user message.
        user_msg = llm.captured_messages[-1]
        assert "Custom observation: environment is ready" in user_msg.content

    @pytest.mark.asyncio
    async def test_thinking_only(self):
        """``worker.arun()`` performs only thinking and returns the
        ``ThinkDecision`` — it does not execute tools."""
        llm = MockLLM()
        llm.select_tool_response = _search_flights_step()
        ctx = _make_context()
        worker = CognitiveWorker(llm=llm)

        decision = await worker.arun(context=ctx)

        assert isinstance(decision, ThinkDecision)
        assert decision.step_content == "Search flights from Beijing to Tokyo"
        assert len(decision.output) == 1
        assert decision.output[0].tool == "search_flights"
        # A tool call was emitted, so finish is False.
        assert decision.finish is False
        # No history added — the worker does not execute the action.
        assert len(ctx.cognitive_history) == 0

    @pytest.mark.asyncio
    async def test_finish_derived_from_no_tool_calls(self):
        """A reply with no tool calls yields ``finish=True``; the text
        becomes ``step_content``."""
        llm = MockLLM()
        llm.select_tool_response = _finish_step("Nothing left to do")
        ctx = _make_context()
        worker = CognitiveWorker(llm=llm)

        decision = await worker.arun(context=ctx)

        assert isinstance(decision, ThinkDecision)
        assert decision.finish is True
        assert decision.step_content == "Nothing left to do"
        assert decision.output == []

    @pytest.mark.asyncio
    async def test_cycle_via_agent(self):
        """End-to-end: a worker driven by an agent runs the full
        observe-think-act cycle for each ThinkUnit."""
        llm = StatefulMockLLM([
            _search_flights_step(),       # cycle 1: search_flights
            _finish_step("All done"),     # cycle 2: no tools → finish
        ])
        worker = CognitiveWorker.inline("Plan ONE immediate next step.")

        class SimpleAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")  # search_flights
                yield ThinkUnit("step")  # no tools

        agent = SimpleAgent()
        await agent.arun(llm=llm, goal="Plan a trip to Tokyo")

        assert len(agent._current_context.cognitive_history) == 2
        # Step 1 actually executed search_flights.
        assert (
            agent._current_context.cognitive_history[0]
            .result.results[0].tool_name == "search_flights"
        )

    @pytest.mark.asyncio
    async def test_override_before_action_pipeline_via_agent(self):
        """The ``before_action`` hook filters ``book_flight`` out of the
        matched tool calls before the act phase runs them."""
        llm = MockLLM()
        llm.select_tool_response = (
            [
                _tool_call(
                    "search_flights",
                    origin="Beijing",
                    destination="Tokyo",
                    date="2025-06-01",
                ),
                _tool_call("book_flight", flight_number="CA123"),
            ],
            "Search and book",
        )
        worker = _ActionPipelineWorker()

        class PipelineAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = PipelineAgent()
        await agent.arun(llm=llm, goal="test")

        last_step = agent._current_context.cognitive_history[-1]
        tool_names = [r.tool_name for r in last_step.result.results]
        assert "book_flight" not in tool_names
        assert "search_flights" in tool_names

    @pytest.mark.asyncio
    async def test_observation_delegates_to_agent_level(self):
        """A worker whose ``observation`` returns ``_DELEGATE`` (the
        default) lets the agent-level observation fill ``ctx.observation``,
        which then reaches the LLM prompt."""
        llm = MockLLM()
        llm.select_tool_response = _search_flights_step()
        worker = CognitiveWorker.inline("Plan ONE step")  # default observation

        class EnhancementAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker, max_attempts=1)

            async def observation(self, ctx):
                from bridgic.amphibious import RETURN
                yield RETURN("Default observation from agent")

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = EnhancementAgent()
        await agent.arun(llm=llm, goal="test")

        user_msg = llm.captured_messages[-1]
        assert "Default observation from agent" in user_msg.content

    @pytest.mark.asyncio
    async def test_from_prompt_convenience(self):
        """``from_prompt()`` builds a worker without defining a subclass."""
        llm = MockLLM()
        llm.select_tool_response = _search_flights_step()

        worker = CognitiveWorker.from_prompt(
            "Plan ONE immediate next step", llm=llm
        )
        assert worker.prompt == "Plan ONE immediate next step"

        ctx = _make_context()
        decision = await worker.arun(context=ctx)
        assert decision.step_content == "Search flights from Beijing to Tokyo"

    @pytest.mark.asyncio
    async def test_inline_worker_via_agent(self):
        """``CognitiveWorker.inline()`` builds a worker usable as a
        think_unit inside ``on_agent``."""
        llm = MockLLM()
        llm.select_tool_response = _search_flights_step()

        worker = CognitiveWorker.inline("Plan ONE step", llm=llm)

        class SimpleAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = SimpleAgent()
        await agent.arun(llm=llm, goal="test")

        assert len(agent._current_context.cognitive_history) == 1
        assert (
            agent._current_context.cognitive_history[0].content
            == "Search flights from Beijing to Tokyo"
        )

    @pytest.mark.asyncio
    async def test_inline_passes_verbose_and_prompt_to_instance(self):
        """``inline()`` forwards verbose flags and stores the prompt as an
        instance attribute (no ``build_messages`` method exists)."""
        worker = CognitiveWorker.inline(
            "Plan", llm=MockLLM(), verbose=True, verbose_prompt=True
        )
        assert worker.prompt == "Plan"
        assert worker._verbose is True
        assert worker._verbose_prompt is True
        # The refactor removed the public ``build_messages`` method.
        assert not hasattr(CognitiveWorker, "build_messages")


# ---------------------------------------------------------------------------
# Tests — output_schema workers (structured-output path)
# ---------------------------------------------------------------------------

from pydantic import BaseModel
from typing import List as _List


class _PlanPhase(BaseModel):
    sub_goal: str
    skill_name: str


class _PlanResult(BaseModel):
    phases: _List[_PlanPhase]


class TestOutputType:
    """``output_schema`` workers go through ``_think_typed_output``, which
    calls ``astructured_output`` and wraps the typed result in a
    ``TypedThinkDecision``."""

    @pytest.mark.asyncio
    async def test_output_schema_returns_typed_decision(self):
        """``arun()`` returns a decision wrapping the typed output."""
        expected = _PlanResult(phases=[
            _PlanPhase(sub_goal="Step A", skill_name="skill-a"),
        ])
        llm = MockLLM()
        llm.structured_output_response = expected
        worker = CognitiveWorker.inline("Plan.", llm=llm, output_schema=_PlanResult)

        ctx = CognitiveContext(goal="Test")
        result = await worker.arun(context=ctx)

        # finish is always True for the typed-output path.
        assert result.finish is True
        assert result.output is expected
        assert isinstance(result.output, _PlanResult)
        assert result.output.phases[0].sub_goal == "Step A"

    @pytest.mark.asyncio
    async def test_output_schema_stored_in_history(self):
        """The typed output lands as the step result in the agent's
        cognitive history."""
        expected = _PlanResult(phases=[
            _PlanPhase(sub_goal="Phase 1", skill_name="skill-1"),
        ])
        llm = MockLLM()
        llm.structured_output_response = expected
        planner = CognitiveWorker.inline("Plan.", llm=llm, output_schema=_PlanResult)

        class _TrackingAgent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(planner)

            async def on_agent(self, ctx):
                yield ThinkUnit("plan")

        agent = _TrackingAgent()
        await agent.arun(llm=llm, goal="Test output_schema")

        last_step = agent._current_context.cognitive_history[-1]
        assert last_step.result is expected
        assert isinstance(last_step.result, _PlanResult)

    @pytest.mark.asyncio
    async def test_output_schema_uses_schema_as_constraint(self):
        """The structured-output call is constrained by the worker's
        ``output_schema`` directly (no decision-model wrapper)."""
        from bridgic.core.model.protocols import PydanticModel

        llm = MockLLM()
        llm.structured_output_response = _PlanResult(phases=[])
        worker = CognitiveWorker.inline("Plan.", llm=llm, output_schema=_PlanResult)

        ctx = CognitiveContext(goal="Test")
        await worker.arun(context=ctx)

        constraint = llm.captured_constraint
        assert isinstance(constraint, PydanticModel)
        # The constraint model IS the output schema itself.
        assert constraint.model is _PlanResult


# ---------------------------------------------------------------------------
# Tests — finish signal stops the OTC loop early
# ---------------------------------------------------------------------------


class TestFinishSignal:
    """A reply with no tool calls finishes the worker; ``think_unit`` with
    ``max_attempts`` stops the loop after that cycle."""

    @pytest.mark.asyncio
    async def test_finish_stops_run_loop(self):
        """The OTC loop stops once the worker emits a no-tool reply."""
        llm = StatefulMockLLM([
            _search_flights_step(),    # cycle 1: tool call → continue
            _finish_step("done"),      # cycle 2: no tool call → finish
        ])
        worker = CognitiveWorker.inline("Plan one step.")

        class _SimpleAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker, max_attempts=10)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = _SimpleAgent()
        await agent.arun(llm=llm, goal="Test finish signal")

        # Stopped after cycle 2 even though max_attempts=10.
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_never_finish_runs_max_attempts(self):
        """When every reply carries a tool call (finish never True), the
        loop runs the full ``max_attempts``."""
        llm = StatefulMockLLM([_search_flights_step()])  # always a tool call
        worker = CognitiveWorker.inline("Plan one step.")

        class _LoopAgent(AmphibiousAutoma[TravelCtx]):
            step = think_unit(worker, max_attempts=3)

            async def on_agent(self, ctx):
                yield ThinkUnit("step")

        agent = _LoopAgent()
        await agent.arun(llm=llm, goal="Test no finish")

        assert llm.call_count == 3


# ---------------------------------------------------------------------------
# Tests — action phase handling of custom (output_schema) decisions
# ---------------------------------------------------------------------------


class TestActionCustomOutput:
    """The act phase stores a non-tool-call ``output`` (typed output)
    directly as the step result."""

    @pytest.mark.asyncio
    async def test_action_custom_output_stores_result(self):
        """``_run_action_call`` stores custom output in the step when the
        decision's ``output`` is not a tool-call list."""
        class _MySchema(BaseModel):
            value: str

        worker = CognitiveWorker.inline("Plan.", output_schema=_MySchema)

        class _SchemaAgent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                if False:
                    yield

        agent = _SchemaAgent()
        ctx = CognitiveContext(goal="Test")
        agent._current_context = ctx

        # A typed-output decision: ``output`` is a model instance, not a list.
        schema_decision = type("SchemaDecision", (), {
            "output": _MySchema(value="result"),
            "step_content": "done",
        })()

        await agent._run_action_call(schema_decision, ctx, _worker=worker)
        last_step = ctx.cognitive_history._items[-1]
        assert last_step.content == "done"
        assert last_step.result.value == "result"
