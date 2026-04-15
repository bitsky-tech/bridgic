"""Tests for human-in-the-loop support in AmphibiousAutoma.

Covers all three HITL entry points:
1. Code-level: await self.request_human(prompt) in on_agent()
2. Workflow yield: feedback = yield HumanCall(prompt=...) in on_workflow()
3. LLM tool: request_human_tool(agent) as a FunctionToolSpec
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Union

import pytest

from bridgic.core.automa.interaction import Event, Feedback, FeedbackSender
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    AgentCall,
    HUMAN_INPUT_EVENT_TYPE,
    StepToolCall,
    ToolArgument,
)
from bridgic.amphibious.builtin_tools import request_human_tool

from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


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


def _make_finish_step():
    return ThinkDecision(
        step_content="Done",
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
        finish=True,
    )


def _auto_respond_handler(response: str = "yes"):
    """Create an event handler that immediately sends a Feedback response."""

    def handler(event: Event, feedback_sender: FeedbackSender):
        feedback_sender.send(Feedback(data=response))

    return handler


def _async_respond_handler(response: str = "approved", delay: float = 0.05):
    """Create an event handler that responds asynchronously after a short delay."""

    def handler(event: Event, feedback_sender: FeedbackSender):
        async def _respond():
            await asyncio.sleep(delay)
            feedback_sender.send(Feedback(data=response))

        asyncio.ensure_future(_respond())

    return handler


# ---------------------------------------------------------------------------
# Tests — HUMAN_INPUT_EVENT_TYPE constant
# ---------------------------------------------------------------------------

class TestHumanInputEventType:

    def test_constant_value(self):
        assert HUMAN_INPUT_EVENT_TYPE == "HUMAN_INPUT_REQUEST"

    def test_constant_is_string(self):
        assert isinstance(HUMAN_INPUT_EVENT_TYPE, str)


# ---------------------------------------------------------------------------
# Tests — Event handler registration
# ---------------------------------------------------------------------------

class TestEventHandlerRegistration:

    def test_handler_registered_on_init(self):
        """AmphibiousAutoma.__init__ registers a HUMAN_INPUT_REQUEST handler."""
        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

        agent = Agent(llm=llm)
        # The framework stores handlers in _event_handlers dict
        assert HUMAN_INPUT_EVENT_TYPE in agent._event_handlers


# ---------------------------------------------------------------------------
# Tests — Entry 1: request_human() in on_agent()
# ---------------------------------------------------------------------------

class TestRequestHuman:

    @pytest.mark.asyncio
    async def test_request_human_returns_response(self):
        """request_human() suspends and returns the human's response string."""
        collected_response = None

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                nonlocal collected_response
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                collected_response = await self.request_human("Continue?")

        llm = MockLLM([_make_finish_step()])
        agent = Agent(llm=llm)
        # Override the default handler with one that auto-responds
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("yes please"),
        )
        await agent.arun(context=_make_ctx())

        assert collected_response == "yes please"

    @pytest.mark.asyncio
    async def test_request_human_async_response(self):
        """request_human() works with an async handler that responds after a delay."""
        collected_response = None

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                nonlocal collected_response
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                collected_response = await self.request_human("Proceed?")

        llm = MockLLM([_make_finish_step()])
        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _async_respond_handler("go ahead", delay=0.05),
        )
        await agent.arun(context=_make_ctx())

        assert collected_response == "go ahead"

    @pytest.mark.asyncio
    async def test_request_human_event_contains_prompt(self):
        """The Event posted by request_human() carries the prompt in its data."""
        captured_event = None

        def capture_handler(event: Event, feedback_sender: FeedbackSender):
            nonlocal captured_event
            captured_event = event
            feedback_sender.send(Feedback(data="ok"))

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                await self.request_human("What is your name?")

        llm = MockLLM([_make_finish_step()])
        agent = Agent(llm=llm)
        agent.register_event_handler(HUMAN_INPUT_EVENT_TYPE, capture_handler)
        await agent.arun(context=_make_ctx())

        assert captured_event is not None
        assert captured_event.event_type == HUMAN_INPUT_EVENT_TYPE
        assert captured_event.data["prompt"] == "What is your name?"

    @pytest.mark.asyncio
    async def test_request_human_multiple_calls(self):
        """Multiple request_human() calls in sequence each get independent responses."""
        responses = []
        call_count = 0

        def counting_handler(event: Event, feedback_sender: FeedbackSender):
            nonlocal call_count
            call_count += 1
            feedback_sender.send(Feedback(data=f"answer-{call_count}"))

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                r1 = await self.request_human("Q1")
                r2 = await self.request_human("Q2")
                responses.extend([r1, r2])

        llm = MockLLM([_make_finish_step()])
        agent = Agent(llm=llm)
        agent.register_event_handler(HUMAN_INPUT_EVENT_TYPE, counting_handler)
        await agent.arun(context=_make_ctx())

        assert responses == ["answer-1", "answer-2"]


# ---------------------------------------------------------------------------
# Tests — Entry 2: HumanCall in on_workflow()
# ---------------------------------------------------------------------------

class TestHumanCallWorkflow:

    @pytest.mark.asyncio
    async def test_human_call_basic(self):
        """yield HumanCall() pauses workflow and returns human response via asend()."""
        collected_feedback = None

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                nonlocal collected_feedback
                collected_feedback = yield HumanCall(prompt="Confirm?")

        llm = MockLLM([])
        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("confirmed"),
        )
        await agent.arun(context=_make_ctx())

        assert collected_feedback == "confirmed"

    @pytest.mark.asyncio
    async def test_human_call_with_action_calls(self):
        """HumanCall works correctly interleaved with ActionCall steps."""
        workflow_trace = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                result = yield ActionCall("search_flights", origin="Beijing", destination="Tokyo", date="2024-06-01")
                workflow_trace.append(("action", result))

                feedback = yield HumanCall(prompt="Book this flight?")
                workflow_trace.append(("human", feedback))

                if feedback == "yes":
                    result2 = yield ActionCall("book_flight", flight_number="CA123")
                    workflow_trace.append(("action", result2))

        llm = MockLLM([])
        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("yes"),
        )
        await agent.arun(context=_make_ctx())

        assert len(workflow_trace) == 3
        assert workflow_trace[0][0] == "action"
        assert workflow_trace[1] == ("human", "yes")
        assert workflow_trace[2][0] == "action"

    @pytest.mark.asyncio
    async def test_human_call_negative_response_skips_action(self):
        """Workflow can branch based on human response."""
        workflow_trace = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                feedback = yield HumanCall(prompt="Proceed?")
                workflow_trace.append(feedback)
                if feedback == "no":
                    return  # Early exit

                yield ActionCall("search_flights", origin="A", destination="B", date="2024-01-01")
                workflow_trace.append("should_not_reach")

        llm = MockLLM([])
        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("no"),
        )
        await agent.arun(context=_make_ctx())

        assert workflow_trace == ["no"]

    @pytest.mark.asyncio
    async def test_human_call_multiple_prompts(self):
        """Multiple HumanCall yields in sequence each get independent responses."""
        responses = []
        call_count = 0

        def counting_handler(event: Event, feedback_sender: FeedbackSender):
            nonlocal call_count
            call_count += 1
            feedback_sender.send(Feedback(data=f"reply-{call_count}"))

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                r1 = yield HumanCall(prompt="Q1")
                r2 = yield HumanCall(prompt="Q2")
                r3 = yield HumanCall(prompt="Q3")
                responses.extend([r1, r2, r3])

        llm = MockLLM([])
        agent = Agent(llm=llm)
        agent.register_event_handler(HUMAN_INPUT_EVENT_TYPE, counting_handler)
        await agent.arun(context=_make_ctx())

        assert responses == ["reply-1", "reply-2", "reply-3"]

    @pytest.mark.asyncio
    async def test_human_call_empty_prompt(self):
        """HumanCall with default empty prompt still works."""
        collected = None

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                nonlocal collected
                collected = yield HumanCall()

        llm = MockLLM([])
        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("default reply"),
        )
        await agent.arun(context=_make_ctx())

        assert collected == "default reply"


# ---------------------------------------------------------------------------
# Tests — Entry 3: request_human_tool (LLM tool)
# ---------------------------------------------------------------------------

class TestHumanRequestTool:

    def test_is_function_tool_spec(self):
        """request_human_tool is a FunctionToolSpec instance (not a factory)."""
        assert isinstance(request_human_tool, FunctionToolSpec)

    def test_tool_name_is_request_human(self):
        """The tool uses 'request_human' as its name."""
        assert request_human_tool.tool_name == "request_human"

    def test_tool_has_prompt_parameter(self):
        """The tool schema includes a 'prompt' parameter."""
        params = request_human_tool.tool_parameters

        assert "properties" in params
        assert "prompt" in params["properties"]

    @pytest.mark.asyncio
    async def test_tool_calls_request_human(self):
        """End-to-end: LLM autonomously calls request_human tool via arun(tools=[request_human_tool])."""

        # Step 1: LLM decides to call request_human tool
        request_human_step = ThinkDecision(
            step_content="I need to ask the user for confirmation",
            output=[
                StepToolCall(
                    tool="request_human",
                    tool_arguments=[
                        ToolArgument(name="prompt", value="Should I proceed?"),
                    ],
                )
            ],
            finish=False,
        )

        # Step 2: LLM finishes after receiving human response
        finish_step = ThinkDecision(
            step_content="User confirmed, task complete.",
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
            finish=True,
        )

        llm = MockLLM([request_human_step, finish_step])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker, max_attempts=5)

        agent = Agent(llm=llm, verbose=True)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("yes, go ahead"),
        )
        await agent.arun(
            goal="Complete a task that requires human confirmation",
            tools=[request_human_tool, *get_travel_planning_tools()],
        )

        # Verify the agent ran both steps (request_human + search_flights)
        steps = agent._current_context.cognitive_history.get_all()
        assert len(steps) == 2
        # First step called request_human
        assert steps[0].result.results[0].tool_name == "request_human"
        assert steps[0].result.results[0].tool_result == "yes, go ahead"
        # Second step proceeded after human confirmation
        assert steps[1].result.results[0].tool_name == "search_flights"

    @pytest.mark.asyncio
    async def test_tool_outside_arun_raises(self):
        """Calling request_human outside of arun() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="only be called during agent execution"):
            await request_human_tool._func(prompt="hello")


# ---------------------------------------------------------------------------
# Tests — human_input() template method override
# ---------------------------------------------------------------------------

class TestHumanInputTemplateOverride:

    @pytest.mark.asyncio
    async def test_override_human_input(self):
        """Subclass can override human_input() to customize response source."""
        collected = None

        class CustomAgent(AmphibiousAutoma[CognitiveContext]):
            async def human_input(self, data: Dict[str, Any]) -> str:
                return f"custom: {data['prompt']}"

            async def on_agent(self, ctx):
                nonlocal collected
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                collected = await self.request_human("test prompt")

        llm = MockLLM([_make_finish_step()])
        agent = CustomAgent(llm=llm)
        await agent.arun(context=_make_ctx())

        assert collected == "custom: test prompt"

    @pytest.mark.asyncio
    async def test_override_human_input_in_workflow(self):
        """Overridden human_input() is also used by HumanCall in workflow mode."""
        collected = None

        class CustomAgent(AmphibiousAutoma[CognitiveContext]):
            async def human_input(self, data: Dict[str, Any]) -> str:
                return f"auto-{data['prompt']}"

            async def on_agent(self, ctx):
                pass

            async def on_workflow(self, ctx) -> AsyncGenerator[Union[ActionCall, HumanCall, AgentCall], None]:
                nonlocal collected
                collected = yield HumanCall(prompt="confirm")

        llm = MockLLM([])
        agent = CustomAgent(llm=llm)
        await agent.arun(context=_make_ctx())

        assert collected == "auto-confirm"


# ---------------------------------------------------------------------------
# Tests — ContextVar concurrency safety
# ---------------------------------------------------------------------------

class TestContextVarConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_agents_isolated(self):
        """Two agents running via asyncio.gather get independent ContextVar values."""
        results_a = []
        results_b = []

        class AgentA(AmphibiousAutoma[CognitiveContext]):
            async def human_input(self, data: Dict[str, Any]) -> str:
                # Small delay to interleave with AgentB
                await asyncio.sleep(0.05)
                return "response-from-A"

            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                r = await self.request_human("question from A")
                results_a.append(r)

        class AgentB(AmphibiousAutoma[CognitiveContext]):
            async def human_input(self, data: Dict[str, Any]) -> str:
                return "response-from-B"

            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                r = await self.request_human("question from B")
                results_b.append(r)

        llm_a = MockLLM([_make_finish_step()])
        llm_b = MockLLM([_make_finish_step()])
        agent_a = AgentA(llm=llm_a)
        agent_b = AgentB(llm=llm_b)

        # Run both agents concurrently — each should see its own ContextVar
        await asyncio.gather(
            agent_a.arun(context=_make_ctx()),
            agent_b.arun(context=_make_ctx()),
        )

        assert results_a == ["response-from-A"]
        assert results_b == ["response-from-B"]

    @pytest.mark.asyncio
    async def test_concurrent_agents_tool_isolated(self):
        """The shared request_human_tool resolves to the correct agent in concurrent runs."""
        tool_results = {}

        class AgentWithTool(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                result = await request_human_tool._func(prompt="who am i?")
                tool_results[self.name] = result

        # Each agent has its own human_input override via event handler
        llm_1 = MockLLM([_make_finish_step()])
        llm_2 = MockLLM([_make_finish_step()])
        agent_1 = AgentWithTool(name="agent-1", llm=llm_1)
        agent_2 = AgentWithTool(name="agent-2", llm=llm_2)

        agent_1.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("you are agent-1"),
        )
        agent_2.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("you are agent-2"),
        )

        await asyncio.gather(
            agent_1.arun(context=_make_ctx()),
            agent_2.arun(context=_make_ctx()),
        )

        assert tool_results["agent-1"] == "you are agent-1"
        assert tool_results["agent-2"] == "you are agent-2"

    @pytest.mark.asyncio
    async def test_contextvar_cleaned_after_arun(self):
        """ContextVar is reset after arun() completes — no leaking between runs."""
        from bridgic.amphibious.builtin_tools.human.request_human import current_agent

        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(context=_make_ctx())

        # After arun() finishes, current_agent should be cleared
        assert current_agent.get(None) is None

    @pytest.mark.asyncio
    async def test_sequential_agents_no_leaking(self):
        """Running agents sequentially — second agent doesn't see the first agent's ContextVar."""
        seen_agents = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)
                # Use the tool to verify which agent the ContextVar resolves to
                from bridgic.amphibious.builtin_tools.human.request_human import current_agent
                seen_agents.append(current_agent.get(None))

        llm = MockLLM([_make_finish_step()])
        agent_a = Agent(name="first", llm=llm)
        agent_b = Agent(name="second", llm=MockLLM([_make_finish_step()]))

        await agent_a.arun(context=_make_ctx())
        await agent_b.arun(context=_make_ctx())

        # Each run sees its own agent, not the previous one
        assert seen_agents[0].name == "first"
        assert seen_agents[1].name == "second"


# ---------------------------------------------------------------------------
# Tests — Built-in tool auto-injection
# ---------------------------------------------------------------------------

class TestBuiltinToolInjection:
    """Verify that framework-level built-in tools (e.g. request_human) are
    auto-injected into every agent's context.tools, including the
    on_workflow → on_agent fallback paths."""

    @pytest.mark.asyncio
    async def test_builtin_injected_when_no_tools_passed(self):
        """arun() without a `tools=` kwarg still exposes request_human to the agent."""
        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(goal="Test builtin injection")

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert "request_human" in tool_names

    @pytest.mark.asyncio
    async def test_builtin_injected_dedupe_on_explicit(self):
        """Explicitly passing request_human_tool does not produce a duplicate entry."""
        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("Plan", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(goal="Test dedupe", tools=[request_human_tool])

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert tool_names.count("request_human") == 1

    @pytest.mark.asyncio
    async def test_builtin_injected_in_workflow_mode(self):
        """Pure WORKFLOW mode (no LLM) still receives the auto-injected builtin."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall], None
            ]:
                if False:
                    yield  # make this an async generator without yielding anything

        agent = Agent()  # No LLM required for pure WORKFLOW mode
        await agent.arun(goal="Test workflow-mode injection")

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert "request_human" in tool_names

    @pytest.mark.asyncio
    async def test_builtin_visible_in_workflow_step_fallback(self):
        """
        When a workflow step fails, the step-level fallback path
        (snapshot(goal=fallback_goal) + _run(worker)) must still see the
        auto-injected request_human tool.
        """

        async def always_fails() -> str:
            """Tool that always raises to force a fallback."""
            raise RuntimeError("simulated step failure")

        always_fails_tool = FunctionToolSpec.from_raw(always_fails)

        # Fallback worker decides to call request_human, then finishes.
        request_human_step = ThinkDecision(
            step_content="Ask the human how to recover",
            output=[
                StepToolCall(
                    tool="request_human",
                    tool_arguments=[
                        ToolArgument(name="prompt", value="what now?"),
                    ],
                )
            ],
            finish=False,
        )
        finish_step = ThinkDecision(
            step_content="Recovered",
            output=[],
            finish=True,
        )
        llm = MockLLM([request_human_step, finish_step])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline(
                    "Recover from the failed step.", llm=self.llm,
                )
                await self._run(worker, max_attempts=5)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall], None
            ]:
                yield ActionCall("always_fails")

        # Step-level fallback runs under a fresh CognitiveHistory snapshot,
        # so we capture request_human invocations via a recording event handler
        # instead of inspecting cognitive_history after the run.
        captured_prompts: List[str] = []

        def recording_handler(event: Event, feedback_sender: FeedbackSender):
            captured_prompts.append(event.data.get("prompt", ""))
            feedback_sender.send(Feedback(data="retry later"))

        agent = Agent(llm=llm)
        agent.register_event_handler(HUMAN_INPUT_EVENT_TYPE, recording_handler)
        # max_consecutive_fallbacks=5 forces the step-level fallback path
        # (line ~1201 in _amphibious_automa.py), not the full agent fallback.
        await agent.arun(
            goal="Trigger step-level fallback",
            tools=[always_fails_tool],
            max_consecutive_fallbacks=5,
        )

        # The fallback worker must have successfully called request_human,
        # proving the auto-injected builtin is visible inside the snapshot.
        assert captured_prompts == ["what now?"]

    @pytest.mark.asyncio
    async def test_builtin_visible_in_full_agent_fallback(self):
        """
        When consecutive workflow failures exceed max_consecutive_fallbacks,
        execution flows into self.on_agent(ctx) directly. The auto-injected
        request_human tool must still be visible there.
        """

        async def always_fails() -> str:
            raise RuntimeError("simulated failure")

        always_fails_tool = FunctionToolSpec.from_raw(always_fails)

        request_human_step = ThinkDecision(
            step_content="Ask for rescue",
            output=[
                StepToolCall(
                    tool="request_human",
                    tool_arguments=[
                        ToolArgument(name="prompt", value="help?"),
                    ],
                )
            ],
            finish=False,
        )
        finish_step = ThinkDecision(
            step_content="Got human help, done.",
            output=[],
            finish=True,
        )
        llm = MockLLM([request_human_step, finish_step])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline(
                    "Full-mode recovery.", llm=self.llm,
                )
                await self._run(worker, max_attempts=5)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, AgentCall], None
            ]:
                yield ActionCall("always_fails")

        agent = Agent(llm=llm)
        agent.register_event_handler(
            HUMAN_INPUT_EVENT_TYPE,
            _auto_respond_handler("here is help"),
        )
        # max_consecutive_fallbacks=1 means the first failure immediately
        # routes to self.on_agent(ctx) (the full agent fallback path at
        # line ~1181 in _amphibious_automa.py).
        await agent.arun(
            goal="Trigger full agent fallback",
            tools=[always_fails_tool],
            max_consecutive_fallbacks=1,
        )

        steps = agent._current_context.cognitive_history.get_all()
        request_human_calls = [
            r
            for step in steps
            if step.result is not None and hasattr(step.result, "results")
            for r in step.result.results
            if r.tool_name == "request_human"
        ]
        assert len(request_human_calls) == 1
        assert request_human_calls[0].tool_result == "here is help"
        assert request_human_calls[0].success is True
