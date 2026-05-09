"""Tests for unified fallback across atomic Calls (ActionCall / HumanCall / LLMCall).

In AMPHIFLOW mode, all three atomic Call types share the same two-tier
fallback semantics:

* **Step-level fallback**: when ``consecutive_failures < threshold``,
  the dispatcher snapshots the context with a fallback goal and runs
  ``on_agent`` to handle just this step. After ``on_agent`` exhausts,
  the workflow generator resumes at the next instruction.
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
    human_channel,
)


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
        assert results[0][1] is None  # step-level fallback returns None
        assert results[1][1] == "ok-2"
        assert results[2][1] is None  # step-level fallback returns None
        assert results[3][1] == "ok-4"
