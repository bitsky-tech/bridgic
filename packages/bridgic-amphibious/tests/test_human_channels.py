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
    AgentCall,
    LLMCall,
    human_channel,
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
    async def test_zero_channels_falls_back_to_stdin_helper(self):
        """No @human_channel registered → _dispatch_human_channel uses stdin."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            pass

        agent = Agent(llm=MockLLM())
        # Stub out _stdin_human_fallback to avoid actual stdin in tests.
        captured = []

        async def fake_stdin(prompt):
            captured.append(prompt)
            return "stdin-reply"

        agent._stdin_human_fallback = fake_stdin

        result = await agent._dispatch_human_channel("question?")

        assert result == "stdin-reply"
        assert captured == ["question?"]

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
        a = await agent._dispatch_human_channel("Q", channel="a")
        b = await agent._dispatch_human_channel("Q", channel="b")

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
            await agent._dispatch_human_channel("Q")

    @pytest.mark.asyncio
    async def test_unknown_channel_raises(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel("real")
            async def real_channel(self, prompt: str) -> str:
                return "ok"

        agent = Agent()
        with pytest.raises(RuntimeError, match="Unknown human channel"):
            await agent._dispatch_human_channel("Q", channel="fake")
