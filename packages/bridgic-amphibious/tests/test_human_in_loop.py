"""Tests for human-in-the-loop support after the yield-only / channel refactor.

Covers the post-refactor surface:

1. ``yield HumanCall(channel=..., prompt=...)`` from ``on_workflow``
2. ``@human_channel`` decorator + class-level registry
3. ``request_human_tool`` (LLM tool) routing to the default channel
4. Channel routing rules: zero channels → stdin fallback, one channel
   → implicit default, multiple → explicit name required
5. ContextVar isolation across concurrent agents
6. Built-in tool auto-injection (request_human stays a tool)
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Union

import pytest

from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    human_channel,
    ThinkUnit,
    think_unit,
)
from bridgic.amphibious.builtin_tools import request_human_tool

from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """Drives the new CognitiveWorker's native function-calling path.

    Each scripted response is a ``(tool_calls, content)`` pair returned by
    ``aselect_tool``; an empty ``tool_calls`` list makes the worker finish.
    """

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self._idx = 0

    async def aselect_tool(self, messages, tools, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _make_ctx() -> CognitiveContext:
    ctx = CognitiveContext(goal="Test goal")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    return ctx


def _make_finish_step():
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return ([], "Done")


def _tool_call_step(tool: str, content: str, **arguments):
    """Scripted ``aselect_tool`` reply with one tool call (worker continues)."""
    return ([{"name": tool, "arguments": arguments}], content)


# ---------------------------------------------------------------------------
# Tests — yield HumanCall in on_workflow with @human_channel handlers
# ---------------------------------------------------------------------------


class TestHumanCallInWorkflow:

    @pytest.mark.asyncio
    async def test_human_call_with_one_channel_uses_it_as_default(self):
        """One @human_channel registered → HumanCall(channel=None) routes to it."""
        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def feishu(self, prompt: str) -> str:
                captured.append(prompt)
                return "approved"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="Confirm the plan?")
                captured.append(("response", resp))

        await Agent().arun(llm=MockLLM([]), context=_make_ctx())

        assert captured == ["Confirm the plan?", ("response", "approved")]

    @pytest.mark.asyncio
    async def test_human_call_explicit_channel_routing(self):
        """HumanCall(channel='name') routes to that specific channel."""
        feishu_seen, terminal_seen = [], []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                feishu_seen.append(prompt)
                return "feishu-reply"

            @human_channel("terminal")
            async def via_terminal(self, prompt: str) -> str:
                terminal_seen.append(prompt)
                return "terminal-reply"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                a = yield HumanCall(channel="feishu", prompt="Q1")
                b = yield HumanCall(channel="terminal", prompt="Q2")
                assert a == "feishu-reply"
                assert b == "terminal-reply"

        await Agent().arun(llm=MockLLM([]), context=_make_ctx())

        assert feishu_seen == ["Q1"]
        assert terminal_seen == ["Q2"]

    @pytest.mark.asyncio
    async def test_human_call_multiple_channels_no_default_raises(self):
        """2+ channels registered + HumanCall(channel=None) → RuntimeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("a")
            async def a(self, prompt: str) -> str:
                return "a"

            @human_channel("b")
            async def b(self, prompt: str) -> str:
                return "b"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield HumanCall(prompt="ambiguous")

        with pytest.raises(RuntimeError, match="ambiguous"):
            await Agent().arun(llm=MockLLM([]), context=_make_ctx())

    @pytest.mark.asyncio
    async def test_human_call_unknown_channel_raises(self):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("a")
            async def a(self, prompt: str) -> str:
                return "a"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield HumanCall(channel="nonexistent", prompt="?")

        with pytest.raises(RuntimeError, match="Unknown human channel"):
            await Agent().arun(llm=MockLLM([]), context=_make_ctx())

    @pytest.mark.asyncio
    async def test_human_call_interleaved_with_actioncalls(self):
        """Channel-based HumanCall composes correctly with ActionCall steps."""
        trace = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                return "yes"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                r = yield ActionCall(
                    "search_flights",
                    origin="Beijing",
                    destination="Tokyo",
                    date="2024-06-01",
                )
                trace.append(("action", r[0].tool_name))
                a = yield HumanCall(prompt="Book?")
                trace.append(("human", a))
                if a == "yes":
                    r2 = yield ActionCall("book_flight", flight_number="CA123")
                    trace.append(("action", r2[0].tool_name))

        await Agent().arun(llm=MockLLM([]), context=_make_ctx())

        assert trace == [
            ("action", "search_flights"),
            ("human", "yes"),
            ("action", "book_flight"),
        ]


# ---------------------------------------------------------------------------
# Tests — request_human_tool (LLM tool) end-to-end
# ---------------------------------------------------------------------------


class TestRequestHumanTool:

    def test_is_function_tool_spec(self):
        assert isinstance(request_human_tool, FunctionToolSpec)

    def test_tool_name_is_request_human(self):
        assert request_human_tool.tool_name == "request_human"

    def test_tool_has_prompt_parameter(self):
        params = request_human_tool.tool_parameters
        assert "properties" in params
        assert "prompt" in params["properties"]

    @pytest.mark.asyncio
    async def test_tool_routes_to_default_channel(self):
        """LLM-invoked request_human routes via the @human_channel registry."""
        seen_prompts: List[str] = []

        request_step = _tool_call_step(
            "request_human", "Need confirmation", prompt="Proceed?"
        )
        finish_step = _make_finish_step()
        llm = MockLLM([request_step, finish_step])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                seen_prompts.append(prompt)
                return "yes, go ahead"

            plan = think_unit(CognitiveWorker.inline("Plan"), max_attempts=5)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        await Agent().arun(llm=llm, goal="Trigger request_human via tool",
            tools=[request_human_tool, *get_travel_planning_tools()],
        )

        assert seen_prompts == ["Proceed?"]

    @pytest.mark.asyncio
    async def test_tool_outside_arun_raises(self):
        with pytest.raises(RuntimeError, match="only be called during agent execution"):
            await request_human_tool._func(prompt="hello")


# ---------------------------------------------------------------------------
# Tests — Channel override via subclassing
# ---------------------------------------------------------------------------


class TestChannelOverride:

    @pytest.mark.asyncio
    async def test_subclass_adds_channel(self):
        """Subclass adds a @human_channel method; HumanCall routes to it."""
        captured = []

        class Custom(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                captured.append(("stdin", prompt))
                return f"custom:{prompt}"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="hi")
                captured.append(("got", resp))

        await Custom().arun(llm=MockLLM([]), context=_make_ctx())

        assert captured == [("stdin", "hi"), ("got", "custom:hi")]

    @pytest.mark.asyncio
    async def test_subclass_overrides_inherited_channel_by_same_name(self):
        """A subclass's @human_channel('x') replaces the parent's @human_channel('x')."""
        captured = []

        class Base(AmphibiousAutoma[CognitiveContext]):
            @human_channel("primary")
            async def parent_impl(self, prompt: str) -> str:
                return "from-parent"

        class Child(Base):
            @human_channel("primary")
            async def child_impl(self, prompt: str) -> str:
                captured.append(prompt)
                return "from-child"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="who answers?")
                captured.append(resp)

        await Child().arun(llm=MockLLM([]), context=_make_ctx())

        assert captured == ["who answers?", "from-child"]


# ---------------------------------------------------------------------------
# Tests — ContextVar isolation across concurrent agents
# ---------------------------------------------------------------------------


class TestContextVarConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_agents_isolated(self):
        """Two agents running via asyncio.gather see only their own channels."""
        results_a, results_b = [], []

        class AgentA(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                await asyncio.sleep(0.05)  # interleave with B
                return "from-A"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                r = yield HumanCall(prompt="A?")
                results_a.append(r)

        class AgentB(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                return "from-B"

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                r = yield HumanCall(prompt="B?")
                results_b.append(r)

        agent_a = AgentA()
        agent_b = AgentB()
        await asyncio.gather(
            agent_a.arun(llm=MockLLM([]), context=_make_ctx()),
            agent_b.arun(llm=MockLLM([]), context=_make_ctx()),
        )

        assert results_a == ["from-A"]
        assert results_b == ["from-B"]

    @pytest.mark.asyncio
    async def test_request_human_tool_isolated_across_agents(self):
        """The shared request_human_tool resolves to the correct agent."""
        tool_results: Dict[str, str] = {}

        class AgentWithTool(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                return f"reply-from-{self.name}"

            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")
                result = await request_human_tool._func(prompt="who?")
                tool_results[self.name] = result

        llm_1 = MockLLM([_make_finish_step()])
        llm_2 = MockLLM([_make_finish_step()])
        agent_1 = AgentWithTool(name="agent-1")
        agent_2 = AgentWithTool(name="agent-2")

        await asyncio.gather(
            agent_1.arun(llm=llm_1, context=_make_ctx()),
            agent_2.arun(llm=llm_2, context=_make_ctx()),
        )

        assert tool_results["agent-1"] == "reply-from-agent-1"
        assert tool_results["agent-2"] == "reply-from-agent-2"

    @pytest.mark.asyncio
    async def test_contextvar_cleared_after_arun(self):
        """current_agent ContextVar is reset after arun() completes."""
        from bridgic.amphibious.builtin_tools import current_agent

        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        await Agent().arun(llm=llm, context=_make_ctx())

        assert current_agent.get(None) is None


# ---------------------------------------------------------------------------
# Tests — Built-in tool auto-injection
# ---------------------------------------------------------------------------


class TestBuiltinToolInjection:

    @pytest.mark.asyncio
    async def test_builtin_injected_when_no_tools_passed(self):
        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        agent = Agent()
        await agent.arun(llm=llm, goal="Test builtin injection")

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert "request_human" in tool_names

    @pytest.mark.asyncio
    async def test_builtin_injected_dedupe_on_explicit(self):
        llm = MockLLM([_make_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            plan = think_unit(CognitiveWorker.inline("Plan"))

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        agent = Agent()
        await agent.arun(llm=llm, goal="Test dedupe", tools=[request_human_tool])

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert tool_names.count("request_human") == 1

    @pytest.mark.asyncio
    async def test_builtin_injected_in_workflow_mode(self):
        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                if False:
                    yield

        agent = Agent()
        await agent.arun(goal="Test workflow-mode injection")

        tool_names = [t.tool_name for t in agent._current_context.tools.get_all()]
        assert "request_human" in tool_names

    @pytest.mark.asyncio
    async def test_request_human_tool_uses_default_channel_in_full_fallback(self):
        """When AMPHIFLOW falls back to on_agent, the @human_channel is still resolvable."""

        async def always_fails() -> str:
            raise RuntimeError("simulated failure")

        always_fails_tool = FunctionToolSpec.from_raw(always_fails)

        request_step = _tool_call_step(
            "request_human", "Ask for rescue", prompt="help?"
        )
        finish_step = ([], "Got help")
        llm = MockLLM([request_step, finish_step])

        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                captured.append(prompt)
                return "here is help"

            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=5)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield ActionCall("always_fails")

        await Agent().arun(llm=llm, goal="Trigger full fallback path",
            tools=[always_fails_tool],
            max_consecutive_fallbacks=1,)

        assert captured == ["help?"]
