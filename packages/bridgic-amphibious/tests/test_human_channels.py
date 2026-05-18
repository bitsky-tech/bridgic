"""Tests for the ``@human_channel`` decorator + class-level registry.

Focuses on the registry-building mechanics (``__init_subclass__``):

* Bare and named decorator forms both register a method
* Subclass channels are inherited from parent
* Subclass override (same channel name) replaces parent
* Base ``AmphibiousAutoma`` has an empty registry
* Multiple channels in one class are all registered
"""

from typing import Any, AsyncGenerator, List, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    RETURN,
    RunMode,
    ThinkUnit,
    human_channel,
    think_unit,
)


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="channel registry test")


class MockLLM:
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def achat(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class TestRegistryBuild:

    def test_base_class_has_empty_registry(self):
        assert AmphibiousAutoma._human_channels == {}

    def test_bare_decorator_uses_method_name(self):

        class A(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def my_method(self, prompt: str) -> str:
                return "x"

        assert A._human_channels == {"my_method": "my_method"}

    def test_named_decorator_uses_explicit_name(self):

        class A(AmphibiousAutoma[CognitiveContext]):
            @human_channel("custom_name")
            async def my_method(self, prompt: str) -> str:
                return "x"

        assert A._human_channels == {"custom_name": "my_method"}

    def test_multiple_channels(self):

        class A(AmphibiousAutoma[CognitiveContext]):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return "f"

            @human_channel("terminal")
            async def via_terminal(self, prompt: str) -> str:
                return "t"

            @human_channel
            async def stdin(self, prompt: str) -> str:
                return "s"

        assert A._human_channels == {
            "feishu": "via_feishu",
            "terminal": "via_terminal",
            "stdin": "stdin",
        }

    def test_subclass_inherits_parent_channels(self):

        class Parent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("parent_chan")
            async def parent_method(self, prompt: str) -> str:
                return "p"

        class Child(Parent):
            @human_channel("child_chan")
            async def child_method(self, prompt: str) -> str:
                return "c"

        assert Child._human_channels == {
            "parent_chan": "parent_method",
            "child_chan": "child_method",
        }

    def test_subclass_overrides_parent_channel_by_name(self):
        """Same channel-name in child wins over parent's registration."""

        class Parent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("shared")
            async def parent_impl(self, prompt: str) -> str:
                return "from-parent"

        class Child(Parent):
            @human_channel("shared")
            async def child_impl(self, prompt: str) -> str:
                return "from-child"

        # Child's registry maps "shared" to its own method, not parent's
        assert Child._human_channels["shared"] == "child_impl"
        # Parent registry is unaffected
        assert Parent._human_channels["shared"] == "parent_impl"


class TestChannelDispatchBehavior:

    @pytest.mark.asyncio
    async def test_zero_channels_falls_back_to_stdin_helper(self, monkeypatch):
        """No @human_channel registered → _run_human_call uses stdin.

        The framework's stdin fallback is a closure inside
        ``_run_human_call`` that goes through ``input()`` in a thread
        executor. Stub the actual stdin layer by patching
        ``builtins.input``.
        """

        class Agent(AmphibiousAutoma[CognitiveContext]):
            pass

        agent = Agent(llm=MockLLM())
        captured: list = []

        def fake_input(prompt):
            captured.append(prompt)
            return "stdin-reply"

        monkeypatch.setattr("builtins.input", fake_input)

        result = await agent._run_human_call("question?")

        assert result == "stdin-reply"
        # The closure formats the prompt as "\n[HumanInput] question?\n> ".
        assert len(captured) == 1
        assert "question?" in captured[0]

    @pytest.mark.asyncio
    async def test_explicit_channel_works(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("a")
            async def channel_a(self, prompt: str) -> str:
                return f"A:{prompt}"

            @human_channel("b")
            async def channel_b(self, prompt: str) -> str:
                return f"B:{prompt}"

        agent = Agent()
        a = await agent._run_human_call("Q", channel="a")
        b = await agent._run_human_call("Q", channel="b")

        assert a == "A:Q"
        assert b == "B:Q"

    @pytest.mark.asyncio
    async def test_default_resolution_with_two_channels_raises(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("a")
            async def channel_a(self, prompt: str) -> str:
                return "a"

            @human_channel("b")
            async def channel_b(self, prompt: str) -> str:
                return "b"

        agent = Agent()
        with pytest.raises(RuntimeError, match="ambiguous"):
            await agent._run_human_call("Q")

    @pytest.mark.asyncio
    async def test_unknown_channel_raises(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("real")
            async def real_channel(self, prompt: str) -> str:
                return "ok"

        agent = Agent()
        with pytest.raises(RuntimeError, match="Unknown human channel"):
            await agent._run_human_call("Q", channel="fake")


class TestRequestHumanToolChannelParam:
    """The agent-facing ``request_human`` tool must accept a ``channel``
    argument and route through ``@human_channel`` registry the same way
    workflow-side ``HumanCall(channel=...)`` does.
    """

    @pytest.mark.asyncio
    async def test_tool_routes_to_explicit_channel_with_multiple_registered(self):
        """LLM (or any direct caller) can pick a specific channel by name."""
        from bridgic.amphibious.builtin_tools._agent_state import current_agent
        from bridgic.amphibious.builtin_tools.human.request_human import request_human

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return f"feishu:{prompt}"

            @human_channel("slack")
            async def via_slack(self, prompt: str) -> str:
                return f"slack:{prompt}"

        agent = Agent()
        token = current_agent.set(agent)
        try:
            feishu_resp = await request_human("hello", channel="feishu")
            slack_resp = await request_human("hello", channel="slack")
        finally:
            current_agent.reset(token)

        assert feishu_resp == "feishu:hello"
        assert slack_resp == "slack:hello"

    @pytest.mark.asyncio
    async def test_tool_uses_implicit_default_when_single_channel(self):
        """No channel arg + exactly one registered → that one is used."""
        from bridgic.amphibious.builtin_tools._agent_state import current_agent
        from bridgic.amphibious.builtin_tools.human.request_human import request_human

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("only_one")
            async def only_one(self, prompt: str) -> str:
                return f"sole:{prompt}"

        agent = Agent()
        token = current_agent.set(agent)
        try:
            resp = await request_human("ping")
        finally:
            current_agent.reset(token)

        assert resp == "sole:ping"

    @pytest.mark.asyncio
    async def test_tool_raises_when_ambiguous_and_channel_not_given(self):
        """Multiple channels registered + no channel arg → ambiguity error."""
        from bridgic.amphibious.builtin_tools._agent_state import current_agent
        from bridgic.amphibious.builtin_tools.human.request_human import request_human

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("a")
            async def chan_a(self, prompt: str) -> str:
                return "a"

            @human_channel("b")
            async def chan_b(self, prompt: str) -> str:
                return "b"

        agent = Agent()
        token = current_agent.set(agent)
        try:
            with pytest.raises(RuntimeError, match="ambiguous"):
                await request_human("Q")
        finally:
            current_agent.reset(token)

    @pytest.mark.asyncio
    async def test_tool_raises_on_unknown_channel(self):
        """Channel name not in registry → clear error."""
        from bridgic.amphibious.builtin_tools._agent_state import current_agent
        from bridgic.amphibious.builtin_tools.human.request_human import request_human

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("real")
            async def real(self, prompt: str) -> str:
                return "ok"

        agent = Agent()
        token = current_agent.set(agent)
        try:
            with pytest.raises(RuntimeError, match="Unknown human channel"):
                await request_human("Q", channel="fake")
        finally:
            current_agent.reset(token)

    def test_tool_schema_advertises_channel_param(self):
        """The exported ``request_human_tool`` must include ``channel`` in
        its JSON schema so the LLM can see it.
        """
        from bridgic.amphibious.builtin_tools import request_human_tool

        params = request_human_tool.to_tool().parameters
        props = params.get("properties", {})
        assert "prompt" in props
        assert "channel" in props, (
            "request_human tool must expose `channel` to the LLM so it can "
            "route to a specific @human_channel when multiple are registered."
        )
        # channel must be optional (not in `required`)
        required = params.get("required", [])
        assert "channel" not in required


# ---------------------------------------------------------------------------
# Dynamic request_human spec — schema reflects the agent's registered
# @human_channel keys (enum constraint + description listing).
# ---------------------------------------------------------------------------


class _ToolSpecCapturer(AmphibiousAutoma[CognitiveContext]):
    """Captures the live ``ToolSpec`` instances injected into ``ctx.tools``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.captured_specs: dict = {}

    async def on_workflow(self, ctx):
        self.captured_specs = {t.tool_name: t for t in ctx.tools.get_all()}
        if False:  # pragma: no cover — async generator marker only
            yield


class TestDynamicRequestHumanSpecFactory:
    """Unit tests for ``build_request_human_tool``."""

    def test_returns_static_spec_when_no_channels(self):
        from bridgic.amphibious.builtin_tools.human.request_human import (
            build_request_human_tool,
            request_human_tool,
        )

        assert build_request_human_tool(None) is request_human_tool
        assert build_request_human_tool([]) is request_human_tool
        # All-empty / falsy entries are also treated as "no channels".
        assert build_request_human_tool(["", None]) is request_human_tool  # type: ignore[list-item]

    def test_constrains_channel_param_to_enum(self):
        from bridgic.amphibious.builtin_tools.human.request_human import (
            build_request_human_tool,
        )

        spec = build_request_human_tool(["feishu", "slack"])
        params = spec.tool_parameters
        assert params is not None
        enum = params["properties"]["channel"]["enum"]
        assert enum == ["feishu", "slack"]

    def test_description_lists_channel_names(self):
        from bridgic.amphibious.builtin_tools.human.request_human import (
            build_request_human_tool,
        )

        spec = build_request_human_tool(["feishu", "slack"])
        assert spec.tool_description is not None
        assert "feishu" in spec.tool_description
        assert "slack" in spec.tool_description

    def test_channel_names_are_sorted_and_deduped(self):
        from bridgic.amphibious.builtin_tools.human.request_human import (
            build_request_human_tool,
        )

        spec = build_request_human_tool(["zeta", "alpha", "mu", "alpha"])
        enum = spec.tool_parameters["properties"]["channel"]["enum"]
        assert enum == ["alpha", "mu", "zeta"]

    def test_returns_independent_specs(self):
        """Two calls with the same input must return distinct specs so
        mutations on one never bleed into another."""
        from bridgic.amphibious.builtin_tools.human.request_human import (
            build_request_human_tool,
        )

        a = build_request_human_tool(["x", "y"])
        b = build_request_human_tool(["x", "y"])
        assert a is not b
        a.tool_parameters["properties"]["channel"]["enum"].append("dirty")
        assert b.tool_parameters["properties"]["channel"]["enum"] == ["x", "y"]


class TestDynamicRequestHumanSpecInjection:
    """Integration tests for the ``arun()`` injection path — the tool spec
    the agent actually sees in ``ctx.tools`` must be specialised to the
    agent class's ``@human_channel`` registry.
    """

    @pytest.mark.asyncio
    async def test_multi_channel_agent_gets_specialised_spec(self):
        class TwoChannelAgent(_ToolSpecCapturer):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return "f"

            @human_channel("slack")
            async def via_slack(self, prompt: str) -> str:
                return "s"

        agent = TwoChannelAgent()
        await agent.arun(goal="x")

        spec = agent.captured_specs["request_human"]
        enum = spec.tool_parameters["properties"]["channel"]["enum"]
        assert enum == ["feishu", "slack"]
        assert "feishu" in spec.tool_description
        assert "slack" in spec.tool_description

    @pytest.mark.asyncio
    async def test_single_channel_agent_also_gets_specialised_spec(self):
        """Even with one channel, the LLM should see its actual name."""

        class SingleChannelAgent(_ToolSpecCapturer):
            @human_channel("my_ui")
            async def via_my_ui(self, prompt: str) -> str:
                return "u"

        agent = SingleChannelAgent()
        await agent.arun(goal="x")

        spec = agent.captured_specs["request_human"]
        enum = spec.tool_parameters["properties"]["channel"]["enum"]
        assert enum == ["my_ui"]
        assert "my_ui" in spec.tool_description

    @pytest.mark.asyncio
    async def test_zero_channel_agent_gets_generic_spec(self):
        """No channels → fall back to the shared static spec (identity)."""
        from bridgic.amphibious.builtin_tools import request_human_tool

        class NoChannelAgent(_ToolSpecCapturer):
            pass

        agent = NoChannelAgent()
        await agent.arun(goal="x")

        assert agent.captured_specs["request_human"] is request_human_tool

    @pytest.mark.asyncio
    async def test_two_agent_classes_do_not_share_state(self):
        """Each subclass gets its own specialised spec — no global pollution."""

        class AgentA(_ToolSpecCapturer):
            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return "f"

            @human_channel("slack")
            async def via_slack(self, prompt: str) -> str:
                return "s"

        class AgentB(_ToolSpecCapturer):
            @human_channel("email")
            async def via_email(self, prompt: str) -> str:
                return "e"

            @human_channel("sms")
            async def via_sms(self, prompt: str) -> str:
                return "m"

        a, b = AgentA(), AgentB()
        await a.arun(goal="x")
        await b.arun(goal="x")

        a_enum = a.captured_specs["request_human"].tool_parameters[
            "properties"
        ]["channel"]["enum"]
        b_enum = b.captured_specs["request_human"].tool_parameters[
            "properties"
        ]["channel"]["enum"]
        assert a_enum == ["feishu", "slack"]
        assert b_enum == ["email", "sms"]

    @pytest.mark.asyncio
    async def test_inject_runs_fresh_each_arun_call(self):
        """Two successive ``arun()`` calls on the same instance must each
        receive a freshly-built spec (not a stale, shared one)."""

        class Agent(_ToolSpecCapturer):
            @human_channel("a")
            async def via_a(self, prompt: str) -> str:
                return "a"

            @human_channel("b")
            async def via_b(self, prompt: str) -> str:
                return "b"

        agent = Agent()
        await agent.arun(goal="run1")
        spec1 = agent.captured_specs["request_human"]
        await agent.arun(goal="run2")
        spec2 = agent.captured_specs["request_human"]

        # Different objects, both correctly specialised
        assert spec1 is not spec2
        assert spec1.tool_parameters["properties"]["channel"]["enum"] == ["a", "b"]
        assert spec2.tool_parameters["properties"]["channel"]["enum"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_end_to_end_on_agent_thinkunit_routes_through_injected_spec(self):
        """End-to-end: ``on_agent`` → ``ThinkUnit`` → ``WorkerRunner``
        finds the injected ``request_human`` spec in ``ctx.tools``, and
        invoking its underlying function with ``channel="feishu"`` reaches
        the feishu handler — not slack — proving the inject + dispatch
        chain is correct in the actual agent reasoning path.
        """
        captured: dict = {}

        class _Probe:
            """WorkerRunner Protocol: pulls the injected request_human
            spec out of ``ctx.tools`` and exercises its underlying func
            the same way the LLM tool-call wrapper would.
            """

            async def run(self, agent, ctx) -> None:
                spec = next(
                    t for t in ctx.tools.get_all() if t.tool_name == "request_human"
                )
                captured["enum"] = spec.tool_parameters["properties"]["channel"][
                    "enum"
                ]
                captured["desc"] = spec.tool_description
                captured["feishu_response"] = await spec._func(
                    prompt="ping", channel="feishu"
                )
                captured["slack_response"] = await spec._func(
                    prompt="ping", channel="slack"
                )

        class _AgentClass(AmphibiousAutoma[CognitiveContext]):
            worker = think_unit(_Probe())

            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return f"feishu-answer-to-{prompt}"

            @human_channel("slack")
            async def via_slack(self, prompt: str) -> str:
                return f"slack-answer-to-{prompt}"

            async def on_agent(self, ctx):
                yield ThinkUnit("worker")

        agent = _AgentClass(llm=MockLLM())
        await agent.arun(goal="test", mode=RunMode.AGENT)

        assert captured["enum"] == ["feishu", "slack"]
        assert "feishu" in captured["desc"] and "slack" in captured["desc"]
        # Routing: channel="feishu" hits the feishu handler, not slack.
        assert captured["feishu_response"] == "feishu-answer-to-ping"
        assert captured["slack_response"] == "slack-answer-to-ping"

    @pytest.mark.asyncio
    async def test_end_to_end_unknown_channel_raises_through_inject_path(self):
        """If the LLM (or anything else) bypasses schema validation and
        passes an unregistered channel name to the injected spec's
        underlying func, the dispatcher's ``Unknown human channel``
        guard still fires — proving the runtime safety net is in place.
        """

        class _Probe:
            async def run(self, agent, ctx) -> None:
                spec = next(
                    t for t in ctx.tools.get_all() if t.tool_name == "request_human"
                )
                with pytest.raises(RuntimeError, match="Unknown human channel"):
                    await spec._func(prompt="ping", channel="not_registered")

        class _AgentClass(AmphibiousAutoma[CognitiveContext]):
            worker = think_unit(_Probe())

            @human_channel("feishu")
            async def via_feishu(self, prompt: str) -> str:
                return "f"

            async def on_agent(self, ctx):
                yield ThinkUnit("worker")

        agent = _AgentClass(llm=MockLLM())
        await agent.arun(goal="test", mode=RunMode.AGENT)
