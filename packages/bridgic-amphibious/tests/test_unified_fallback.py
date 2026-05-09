"""Tests for unified fallback across atomic Calls (ActionCall / HumanCall / LLMCall).

In AMPHIFLOW mode, all three atomic Call types share the same two-tier
fallback semantics:

* **Step-level fallback**: when ``consecutive_failures < threshold``,
  the dispatcher snapshots the context with a fallback goal AND a
  fresh ``resolve_step_fallback`` tool, then runs ``on_agent``. The
  agent can call ``resolve_step_fallback(...)`` to set the value the
  workflow's failed yield should receive; if it doesn't, a benign
  default is used. After ``on_agent`` exhausts, the workflow generator
  resumes at the next instruction with that value.
* **Full fallback**: when ``consecutive_failures >= threshold`` (or the
  workflow generator raises an internal exception), the workflow
  generator is closed and ``on_agent`` runs with the original context;
  the workflow does not resume.
"""

from typing import Any, AsyncGenerator, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    RETURN,
    RunMode,
    StepToolCall,
    ToolArgument,
    ThinkUnit,
    human_channel,
    think_unit,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


def _finish_step():
    return ThinkDecision(step_content="Recovered", output=[], finish=True)


class _MockLLM:
    def __init__(self, structured_responses=None, chat_responses=None):
        self.structured_responses = list(structured_responses or [])
        self.chat_responses = list(chat_responses or [])
        self._struct_idx = 0
        self._chat_idx = 0
        self.chat_call_count = 0

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self.structured_responses[self._struct_idx % len(self.structured_responses)]
        self._struct_idx += 1
        return resp

    async def achat(self, messages, **kwargs):
        self.chat_call_count += 1
        if not self.chat_responses:
            raise RuntimeError("LLM unavailable")
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="unified-fallback")


# ---------------------------------------------------------------------------
# LLMCall fallback
# ---------------------------------------------------------------------------


class TestLLMCallFallback:

    @pytest.mark.asyncio
    async def test_llm_call_step_level_fallback_resumes_workflow(self):
        """LLMCall.chat raises → on_agent runs with fallback goal → workflow resumes."""

        agent_invocations = []
        # _MockLLM with empty chat_responses raises on every achat() call.
        # But _MockLLM serves both LLMCall (achat) and worker structured output.
        llm = _MockLLM(structured_responses=[_finish_step()])

        post_fallback_steps = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("Recover.", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                # First step: LLMCall fails (no chat responses).
                yield LLMCall.chat("Will fail")
                # If we get here, step-level fallback resumed us.
                post_fallback_steps.append("after-fallback")

        # threshold=2 means first failure triggers step-level (not full).
        await Agent(llm=llm).arun(
            goal="trigger LLMCall step-level fallback",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        assert len(agent_invocations) == 1, (
            "on_agent should have been invoked exactly once for step-level fallback"
        )
        assert "[Workflow fallback]" in agent_invocations[0], (
            f"Fallback goal should be set; got: {agent_invocations[0]!r}"
        )
        assert post_fallback_steps == ["after-fallback"], (
            "workflow should have resumed after step-level fallback"
        )

    @pytest.mark.asyncio
    async def test_llm_call_threshold_breach_full_fallback(self):
        """Two LLMCall failures in a row with threshold=1 → full fallback."""

        agent_invocations = []
        llm = _MockLLM(structured_responses=[_finish_step()])
        post_fallback_steps = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("Recover.", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                yield LLMCall.chat("Will fail")
                # Should not be reached: threshold=1 → first failure = full fallback.
                post_fallback_steps.append("UNREACHABLE")

        await Agent(llm=llm).arun(
            goal="trigger LLMCall full fallback",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=1,
        )

        assert len(agent_invocations) == 1
        assert post_fallback_steps == [], (
            "workflow should NOT resume after full fallback"
        )


# ---------------------------------------------------------------------------
# HumanCall fallback
# ---------------------------------------------------------------------------


class TestHumanCallFallback:

    @pytest.mark.asyncio
    async def test_human_call_step_level_fallback_resumes_workflow(self):
        """HumanCall channel raises → on_agent runs with fallback goal → workflow resumes."""

        agent_invocations = []
        llm = _MockLLM(structured_responses=[_finish_step()])
        post_fallback_steps = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def broken(self, prompt: str) -> str:
                raise RuntimeError("channel broken")

            async def on_agent(self, ctx):
                agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("Recover.", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                yield HumanCall(prompt="confirm?")
                post_fallback_steps.append("after-fallback")

        await Agent(llm=llm).arun(
            goal="trigger HumanCall step-level fallback",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        assert len(agent_invocations) == 1
        assert "[Workflow fallback]" in agent_invocations[0]
        assert post_fallback_steps == ["after-fallback"]

    @pytest.mark.asyncio
    async def test_human_call_threshold_breach_full_fallback(self):
        """HumanCall failure with threshold=1 → full fallback."""

        agent_invocations = []
        llm = _MockLLM(structured_responses=[_finish_step()])
        post_fallback_steps = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            @human_channel
            async def broken(self, prompt: str) -> str:
                raise RuntimeError("channel broken")

            async def on_agent(self, ctx):
                agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("Recover.", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                yield HumanCall(prompt="confirm?")
                post_fallback_steps.append("UNREACHABLE")

        await Agent(llm=llm).arun(
            goal="trigger HumanCall full fallback",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=1,
        )

        assert len(agent_invocations) == 1
        assert post_fallback_steps == []


# ---------------------------------------------------------------------------
# Mixed: success after step-level fallback resets the counter
# ---------------------------------------------------------------------------


class TestFallbackCounterReset:

    @pytest.mark.asyncio
    async def test_successful_call_after_fallback_resets_counter(self):
        """A successful atomic Call after a fallback resets consecutive_failures.

        Workflow yields four LLMCalls: fail, succeed, fail, succeed. With
        threshold=2, the second fail-after-success would breach if the
        counter were NOT reset (would be 2 = threshold). Because it IS
        reset, the second fail is just step-level fallback.
        """
        from bridgic.core.model.types import Message, Response, Role

        class _AlternatingLLM:
            def __init__(self):
                self.chat_count = 0

            async def astructured_output(self, messages, constraint, **kwargs):
                return _finish_step()

            async def achat(self, messages, **kwargs):
                self.chat_count += 1
                if self.chat_count in {1, 3}:
                    raise RuntimeError(f"fail on attempt {self.chat_count}")
                return Response(message=Message.from_text(f"ok-{self.chat_count}", role=Role.AI))

            async def astream(self, messages, **kwargs): ...
            def chat(self, messages, **kwargs): ...
            def stream(self, messages, **kwargs): ...

        llm = _AlternatingLLM()
        agent_invocations = []
        results = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                agent_invocations.append(ctx.goal)
                worker = CognitiveWorker.inline("Recover.", llm=self.llm)
                await self._run(worker, max_attempts=1)

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                r1 = yield LLMCall.chat("c1")  # chat 1 fails → fallback
                results.append(("r1", r1))
                r2 = yield LLMCall.chat("c2")  # chat 2 succeeds → counter reset
                results.append(("r2", r2))
                r3 = yield LLMCall.chat("c3")  # chat 3 fails → fallback (NOT full)
                results.append(("r3", r3))
                r4 = yield LLMCall.chat("c4")  # chat 4 succeeds
                results.append(("r4", r4))

        await Agent(llm=llm).arun(
            goal="counter-reset test",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        # Two step-level fallbacks should have happened (not one full fallback).
        assert len(agent_invocations) == 2, (
            f"expected 2 step-level fallbacks; got {len(agent_invocations)}"
        )
        assert len(results) == 4, (
            "workflow should have completed all four steps "
            f"(fallback + success + fallback + success); got {len(results)}"
        )
        # The fallback agent in this test does not call resolve_step_fallback,
        # so the slot keeps its benign default. For LLMCall(chat) that's "".
        assert results[0][1] == ""    # step-level fallback default for chat
        assert results[1][1] == "ok-2"
        assert results[2][1] == ""    # step-level fallback default for chat
        assert results[3][1] == "ok-4"


# ---------------------------------------------------------------------------
# Slot mechanism — agent calls resolve_step_fallback to feed value back
# ---------------------------------------------------------------------------


class TestSlotMechanism:

    @pytest.mark.asyncio
    async def test_action_call_resolve_via_tool_flows_to_workflow(self):
        """Fallback agent calls resolve_step_fallback → workflow's yield
        receives a ToolResult wrapping the agent's submitted value."""

        async def always_fails():
            raise RuntimeError("boom")

        # Worker sees the resolve_step_fallback tool in its tools surface
        # and decides to call it. Then on next cycle, finishes.
        recover_step = ThinkDecision(
            step_content="Submitting recovered value",
            output=[StepToolCall(
                tool="resolve_step_fallback",
                tool_arguments=[ToolArgument(name="result", value="recovered_data")],
            )],
            finish=False,
        )
        finish_step = ThinkDecision(step_content="Done", output=[], finish=True)
        llm = _MockLLM(structured_responses=[recover_step, finish_step])

        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=5)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                data = yield ActionCall("always_fails")
                captured.append(data)

        await Agent(llm=llm).arun(
            goal="test slot recovery",
            tools=[FunctionToolSpec.from_raw(always_fails)],
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        assert len(captured) == 1
        result_list = captured[0]
        assert isinstance(result_list, list) and len(result_list) == 1
        rec = result_list[0]
        assert rec.tool_name == "always_fails"
        assert rec.success is True
        assert rec.result == "recovered_data"

    @pytest.mark.asyncio
    async def test_action_call_no_resolve_uses_default_slot(self):
        """If fallback agent never calls resolve_step_fallback, workflow
        receives the benign default (one ToolResult with result=None)."""

        async def always_fails():
            raise RuntimeError("boom")

        finish_step = ThinkDecision(step_content="Gave up", output=[], finish=True)
        llm = _MockLLM(structured_responses=[finish_step])

        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=5)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                data = yield ActionCall("always_fails", arg1="x")
                captured.append(data)

        await Agent(llm=llm).arun(
            goal="default slot test",
            tools=[FunctionToolSpec.from_raw(always_fails)],
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        # Default slot: one ToolResult with result=None for the failed call.
        assert len(captured) == 1
        result_list = captured[0]
        assert isinstance(result_list, list) and len(result_list) == 1
        rec = result_list[0]
        assert rec.tool_name == "always_fails"
        assert rec.tool_arguments == {"arg1": "x"}
        assert rec.success is True
        assert rec.result is None

    @pytest.mark.asyncio
    async def test_human_call_resolve_via_tool_flows_to_workflow(self):
        """HumanCall fallback: agent submits a response via resolve tool."""

        recover_step = ThinkDecision(
            step_content="Submitting human response",
            output=[StepToolCall(
                tool="resolve_step_fallback",
                tool_arguments=[ToolArgument(name="response", value="yes please")],
            )],
            finish=False,
        )
        finish_step = ThinkDecision(step_content="Done", output=[], finish=True)
        llm = _MockLLM(structured_responses=[recover_step, finish_step])

        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=5)

            @human_channel
            async def broken(self, prompt: str) -> str:
                raise RuntimeError("channel broken")

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                feedback = yield HumanCall(prompt="confirm?")
                captured.append(feedback)

        await Agent(llm=llm).arun(
            goal="human resolve test",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        assert captured == ["yes please"]

    @pytest.mark.asyncio
    async def test_resolve_tool_only_present_during_fallback(self):
        """The resolve_step_fallback tool is only injected for the duration
        of the fallback's snapshot; user tools are restored on exit."""

        async def always_fails():
            raise RuntimeError("boom")

        async def real_user_tool() -> str:
            return "user-tool-output"

        observed_tool_names_during_recovery = []

        # Worker decision: call real_user_tool first (record what tools are
        # visible to the LLM), then resolve, then finish.
        check_step = ThinkDecision(
            step_content="Inspecting tools",
            output=[StepToolCall(
                tool="real_user_tool",
                tool_arguments=[],
            )],
            finish=False,
        )
        recover_step = ThinkDecision(
            step_content="Submitting",
            output=[StepToolCall(
                tool="resolve_step_fallback",
                tool_arguments=[ToolArgument(name="result", value="recovered")],
            )],
            finish=False,
        )
        finish_step = ThinkDecision(step_content="Done", output=[], finish=True)
        llm = _MockLLM(structured_responses=[check_step, recover_step, finish_step])

        captured = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=5)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                # Record the tool names visible inside fallback agent.
                observed_tool_names_during_recovery.extend(
                    [t.tool_name for t in ctx.tools.get_all()]
                )
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                data = yield ActionCall("always_fails")
                captured.append(data)
                # After workflow resumes, the fallback tool should be GONE.
                captured.append([t.tool_name for t in ctx.tools.get_all()])

        await Agent(llm=llm).arun(
            goal="tool scope test",
            tools=[
                FunctionToolSpec.from_raw(always_fails),
                FunctionToolSpec.from_raw(real_user_tool),
            ],
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        # During fallback, resolve_step_fallback is in the tool surface
        # alongside user tools.
        assert "resolve_step_fallback" in observed_tool_names_during_recovery
        assert "real_user_tool" in observed_tool_names_during_recovery
        # After workflow resumes (snapshot rolled back), the fallback tool
        # is gone but user tools remain.
        post_resume_tools = captured[1]
        assert "resolve_step_fallback" not in post_resume_tools
        assert "real_user_tool" in post_resume_tools
