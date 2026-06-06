"""Human-in-the-loop integration tests (consolidated).

Drives HITL end-to-end through ``await agent.arun(...)`` round-trips and the
LLM-callable ``request_human`` tool, covering the post-"two-loop"-refactor
surface:

1. ``@human_channel`` registry (built per class at ``__init_subclass__`` time)
   + ``HumanCall(prompt=, channel=)`` routing from ``on_workflow``:
   one channel → implicit default; multiple → explicit name required
   (ambiguous / unknown both raise ``RuntimeError``).
2. ``request_human`` tool: LLM-selected inside a ``ThinkUnit`` resolves the
   channel via the ``current_agent`` ContextVar at call time; calling it
   outside agent execution raises.
3. ``request_human`` is a *declared* (opt-in) tool — a context carries it only
   if it calls ``Ctx.tool(request_human_tool)``; it is never auto-injected.
4. ContextVar concurrency isolation across ``asyncio.gather``; ``current_agent``
   resets to ``None`` after each ``arun``.
5. The zero-channel stdin fallback (the one unit seam with no e2e equivalent).

ISOLATION: ``@human_channel`` handlers and ``Context._declared_tools`` are
class-creation-time registries — every test defines its OWN ``AmphibiousAutoma``
subclass (and its OWN context subclass where a declared tool is needed) so
registries never cross-pollinate between tests.
"""

import asyncio
from typing import Any, AsyncGenerator, List, Optional, Union

import pytest

from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message, Role
from bridgic.amphibious import (
    AmphibiousAutoma,
    Context,
    OTAContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    human_channel,
    ThinkUnit,
    think_unit,
)
from bridgic.amphibious.builtin_tools import request_human_tool, current_agent

from .tools import get_travel_planning_tools


# ---------------------------------------------------------------------------
# Helpers (ported from the subsumed test files)
# ---------------------------------------------------------------------------


class MockLLM:
    """Drives CognitiveWorker's native function-calling path.

    Each scripted response is a ``(tool_calls, content)`` pair returned by
    ``aselect_tool``; an empty ``tool_calls`` list makes the worker finish.
    """

    def __init__(self, responses: Optional[List[Any]] = None):
        self._responses = list(responses or [])
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


class PlanWorker(CognitiveWorker):
    """Minimal tool-selecting worker.

    ``thinking`` calls ``aselect_tool`` with the OTA context's tools; the
    scripted ``MockLLM`` decides whether it finishes (empty tool calls) or
    invokes a tool such as ``request_human``.
    """

    async def thinking(
        self, ota_context: OTAContext, context: Optional[Context] = None
    ) -> Any:
        return await self._llm.aselect_tool(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )


def _make_ctx() -> Context:
    """Big-loop (knowledge) context for ``context=``; the goal is seeded into
    the per-run OTA context via the ``**_seed`` arun kwargs."""
    return Context()


# Goal seeded into the fresh per-run OTA context via arun kwargs. Tools are not
# passed to arun — each context declares the tools it carries.
_seed = dict(user_input="Test goal")


def _finish_step():
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return ([], "Done")


def _tool_call_step(tool: str, content: str, **arguments):
    """Scripted ``aselect_tool`` reply with one tool call (worker continues)."""
    return ([{"name": tool, "arguments": arguments}], content)


# ---------------------------------------------------------------------------
# HumanCall routing from on_workflow (end-to-end via arun)
# ---------------------------------------------------------------------------


class TestHumanCallInWorkflow:

    @pytest.mark.asyncio
    async def test_one_channel_implicit_default(self):
        """One @human_channel registered → HumanCall(channel=None) routes to it
        and the handler's reply is asend-ed back into on_workflow."""
        captured = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def feishu(self, prompt: str) -> str:
                captured.append(prompt)
                return "approved"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="Confirm the plan?")
                captured.append(("response", resp))

        await Agent().arun(llm=MockLLM(), context=_make_ctx(), **_seed)

        assert captured == ["Confirm the plan?", ("response", "approved")]

    @pytest.mark.asyncio
    async def test_explicit_channel_routing(self):
        """With multiple channels, HumanCall(channel='name') routes to that
        specific handler."""
        feishu_seen, terminal_seen = [], []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                feishu_seen.append(prompt)
                return "feishu-reply"

            @human_channel("terminal")
            async def via_terminal(self, prompt: str) -> str:
                terminal_seen.append(prompt)
                return "terminal-reply"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                a = yield HumanCall(channel="feishu", prompt="Q1")
                b = yield HumanCall(channel="terminal", prompt="Q2")
                assert a == "feishu-reply"
                assert b == "terminal-reply"

        await Agent().arun(llm=MockLLM(), context=_make_ctx(), **_seed)

        assert feishu_seen == ["Q1"]
        assert terminal_seen == ["Q2"]

    @pytest.mark.parametrize(
        "channel, match",
        [
            (None, "ambiguous"),  # 2+ channels + no explicit name
            ("nonexistent", "Unknown human channel"),  # name not in registry
        ],
    )
    @pytest.mark.asyncio
    async def test_channel_resolution_errors(self, channel, match):
        """Two channels registered: omitting the name is ambiguous and an
        unknown name is rejected — both raise RuntimeError (covers the old
        ``_run_human_call`` unit-layer cases via the e2e path)."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel("a")
            async def a(self, prompt: str) -> str:
                return "a"

            @human_channel("b")
            async def b(self, prompt: str) -> str:
                return "b"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield HumanCall(channel=channel, prompt="?")

        with pytest.raises(RuntimeError, match=match):
            await Agent().arun(llm=MockLLM(), context=_make_ctx(), **_seed)

    @pytest.mark.asyncio
    async def test_human_call_interleaved_with_actioncalls(self):
        """Channel-based HumanCall composes with ActionCall steps
        (action → human yes → action)."""
        trace = []

        class TravelOTAContext(OTAContext):
            """Small-loop context carrying the travel-planning tools."""

        for _spec in get_travel_planning_tools():
            TravelOTAContext.tool(_spec)

        class Agent(AmphibiousAutoma[TravelOTAContext, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                return "yes"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
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

        await Agent().arun(llm=MockLLM(), context=_make_ctx(), **_seed)

        assert trace == [
            ("action", "search_flights"),
            ("human", "yes"),
            ("action", "book_flight"),
        ]


# ---------------------------------------------------------------------------
# request_human tool (LLM-callable inside a ThinkUnit)
# ---------------------------------------------------------------------------


class TestRequestHumanTool:

    @pytest.mark.asyncio
    async def test_tool_routes_via_registry(self):
        """LLM-invoked request_human resolves the @human_channel registry at
        call time: with a single channel it routes implicitly; with multiple
        an explicit channel name selects the handler."""
        single_seen: List[str] = []
        multi_seen: List[Any] = []

        class SingleCtx(OTAContext):
            pass

        SingleCtx.tool(request_human_tool)

        class MultiCtx(OTAContext):
            pass

        MultiCtx.tool(request_human_tool)

        class SingleAgent(AmphibiousAutoma[SingleCtx, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                single_seen.append(prompt)
                return "sole-reply"

            plan = think_unit(PlanWorker(), max_attempts=5)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        class MultiAgent(AmphibiousAutoma[MultiCtx, Context]):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                multi_seen.append(("feishu", prompt))
                return "feishu-reply"

            @human_channel("slack")
            async def via_slack(self, prompt: str) -> str:
                multi_seen.append(("slack", prompt))
                return "slack-reply"

            plan = think_unit(PlanWorker(), max_attempts=5)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("plan")

        # Single channel: LLM omits `channel`, routed implicitly.
        await SingleAgent().arun(
            llm=MockLLM(
                [
                    _tool_call_step("request_human", "ask", prompt="Proceed?"),
                    _finish_step(),
                ]
            ),
            user_input="trigger request_human (single)",
        )
        # Multiple channels: LLM names the channel explicitly.
        await MultiAgent().arun(
            llm=MockLLM(
                [
                    _tool_call_step(
                        "request_human", "ask", prompt="Pick?", channel="slack"
                    ),
                    _finish_step(),
                ]
            ),
            user_input="trigger request_human (multi)",
        )

        assert single_seen == ["Proceed?"]
        assert multi_seen == [("slack", "Pick?")]

    @pytest.mark.asyncio
    async def test_tool_outside_arun_raises(self):
        """No current_agent ContextVar set → calling the tool raises."""
        with pytest.raises(
            RuntimeError, match="only be called during agent execution"
        ):
            await request_human_tool._func(prompt="hello")

    @pytest.mark.parametrize(
        "field, required",
        [
            ("prompt", True),  # the question is mandatory
            ("channel", False),  # optional routing hint, surfaced to the LLM
        ],
    )
    def test_tool_schema_advertises_params(self, field, required):
        """The exported request_human spec is a FunctionToolSpec named
        ``request_human`` whose schema advertises ``prompt`` (required) and an
        optional ``channel`` so the LLM can route explicitly."""
        assert isinstance(request_human_tool, FunctionToolSpec)
        assert request_human_tool.tool_name == "request_human"

        params = request_human_tool.to_tool().parameters
        props = params.get("properties", {})
        assert field in props
        assert (field in params.get("required", [])) is required


# ---------------------------------------------------------------------------
# @human_channel registry mechanics (built at __init_subclass__ time)
# ---------------------------------------------------------------------------


class TestChannelRegistry:

    def test_subclass_inherits_parent_channels(self):
        """Bare + named decorator forms register a handler, and a subclass
        inherits its parent's channels while adding its own."""

        class Parent(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:  # bare → method name
                return "p"

            @human_channel("parent_chan")  # named → explicit key
            async def parent_method(self, prompt: str) -> str:
                return "p2"

        class Child(Parent):
            @human_channel("child_chan")
            async def child_method(self, prompt: str) -> str:
                return "c"

        assert Parent._human_channels == {
            "stdin": "stdin",
            "parent_chan": "parent_method",
        }
        assert Child._human_channels == {
            "stdin": "stdin",
            "parent_chan": "parent_method",
            "child_chan": "child_method",
        }

    @pytest.mark.asyncio
    async def test_subclass_override_by_same_name_wins(self):
        """A child's @human_channel('x') replaces the parent's same-named one
        (routing reaches the child impl) and the parent registry is unaffected."""
        captured = []

        class Base(AmphibiousAutoma[OTAContext, Context]):
            @human_channel("primary")
            async def parent_impl(self, prompt: str) -> str:
                return "from-parent"

        class Child(Base):
            @human_channel("primary")
            async def child_impl(self, prompt: str) -> str:
                captured.append(prompt)
                return "from-child"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                resp = yield HumanCall(prompt="who answers?")
                captured.append(resp)

        await Child().arun(llm=MockLLM(), context=_make_ctx(), **_seed)

        assert captured == ["who answers?", "from-child"]
        # Registry mappings are distinct per class.
        assert Child._human_channels["primary"] == "child_impl"
        assert Base._human_channels["primary"] == "parent_impl"


# ---------------------------------------------------------------------------
# Zero-channel stdin fallback — the one unit seam with no e2e equivalent
# ---------------------------------------------------------------------------


class TestStdinFallback:

    @pytest.mark.asyncio
    async def test_zero_channels_falls_back_to_stdin(self, monkeypatch):
        """No @human_channel registered → ``_run_human_call`` reads from
        ``builtins.input`` (in a thread executor) and formats the prompt."""

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            pass

        agent = Agent()
        captured: list = []

        def fake_input(prompt):
            captured.append(prompt)
            return "stdin-reply"

        monkeypatch.setattr("builtins.input", fake_input)

        result = await agent._run_human_call(HumanCall(prompt="question?"))

        assert result == "stdin-reply"
        # The closure formats the prompt as "\n[HumanInput] question?\n> ".
        assert len(captured) == 1
        assert "question?" in captured[0]


# ---------------------------------------------------------------------------
# ContextVar concurrency isolation
# ---------------------------------------------------------------------------


class TestContextVarConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_agents_isolated_and_reset(self):
        """Two agents under asyncio.gather each see only their OWN
        @human_channel (resolved via the per-Task current_agent ContextVar);
        after both finish the ContextVar is reset to None."""
        results_a, results_b = [], []

        class AgentA(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                await asyncio.sleep(0.05)  # interleave with B
                return "from-A"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                r = yield HumanCall(prompt="A?")
                results_a.append(r)

        class AgentB(AmphibiousAutoma[OTAContext, Context]):
            @human_channel
            async def stdin(self, prompt: str) -> str:
                return "from-B"

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                r = yield HumanCall(prompt="B?")
                results_b.append(r)

        await asyncio.gather(
            AgentA().arun(llm=MockLLM(), context=_make_ctx(), **_seed),
            AgentB().arun(llm=MockLLM(), context=_make_ctx(), **_seed),
        )

        # Each agent routed to its own channel, not the other's.
        assert results_a == ["from-A"]
        assert results_b == ["from-B"]
        # ContextVar (no default) is cleared once arun's finally runs.
        assert current_agent.get(None) is None


# ---------------------------------------------------------------------------
# Declared (opt-in) request_human tool — never auto-injected
# ---------------------------------------------------------------------------


class TestRequestHumanToolDeclaration:

    @pytest.mark.asyncio
    async def test_declared_tool_present_in_workflow_mode(self):
        """A context that declares request_human carries it in ctx.tools even
        in a pure-workflow run (no on_agent / ThinkUnit). Confirms the tool is
        a declared built-in, not something the framework auto-injects."""

        class HumanToolOTAContext(OTAContext):
            pass

        HumanToolOTAContext.tool(request_human_tool)

        class Agent(AmphibiousAutoma[HumanToolOTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                if False:
                    yield

        agent = Agent()
        await agent.arun(user_input="Test workflow-mode declaration")

        tool_names = [t.tool_name for t in agent._current_ota_context.tools]
        assert "request_human" in tool_names
