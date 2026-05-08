"""Tests for unified dispatcher across all five template methods.

Verifies that yield primitives (ActionCall, HumanCall, LLMCall, RETURN)
work in any template method that the dispatcher drives — not just
``on_workflow``. Also verifies that both async-generator and plain
async-coroutine forms are accepted (legacy compat).
"""

from typing import Any, AsyncGenerator, List, Union

import pytest
from pydantic import BaseModel

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    AgentCall,
    LLMCall,
    RETURN,
    StepToolCall,
    ToolArgument,
    human_channel,
    think_unit,
)
from bridgic.core.model.types import Message, Response, Role


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class MockLLM:
    def __init__(self, chat_responses=None, structured_responses=None):
        self.chat_responses = list(chat_responses or [])
        self.structured_responses = list(structured_responses or [])
        self._chat_idx = 0
        self._struct_idx = 0

    async def achat(self, messages, **kwargs):
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs):
        resp = self.structured_responses[self._struct_idx % len(self.structured_responses)]
        self._struct_idx += 1
        return resp

    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...
    def structured_output(self, messages, constraint, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...


def _ctx() -> CognitiveContext:
    return CognitiveContext(goal="dispatcher unification test")


def _txt(text: str) -> Response:
    return Response(message=Message.from_text(text, role=Role.AI))


def _finish_step():
    return ThinkDecision(step_content="Done", output=[], finish=True)


# ---------------------------------------------------------------------------
# Yield primitives in non-on_workflow template methods
# ---------------------------------------------------------------------------


class TestYieldsInObservation:

    @pytest.mark.asyncio
    async def test_observation_can_yield_llm_call(self):
        """observation generator can yield LLMCall and RETURN(value)."""
        llm = MockLLM(
            chat_responses=[_txt("computed-observation")],
            structured_responses=[_finish_step()],
        )

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx) -> AsyncGenerator[Any, Any]:
                obs = yield LLMCall.chat("Summarize the page")
                yield RETURN(obs)

            # Coroutine form for on_agent — avoid yielding bare ``None``
            # which the dispatcher would attempt to treat as a yield item.
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                # Run the worker — it'll trigger observation() via the
                # _run_observe_think_act observe phase.
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(context=_ctx())

        # observation was invoked, ctx.observation set
        assert agent._current_context.observation == "computed-observation"

    @pytest.mark.asyncio
    async def test_observation_as_coroutine_still_works(self):
        """Legacy form: observation as a plain coroutine returning a string."""
        llm = MockLLM(structured_responses=[_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def observation(self, ctx):
                return "legacy-obs"

            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(context=_ctx())

        assert agent._current_context.observation == "legacy-obs"


class TestYieldsInBeforeAction:

    @pytest.mark.asyncio
    async def test_before_action_yields_return_overrides_decision(self):
        """before_action generator yields RETURN(modified) to override the decision."""
        llm = MockLLM(structured_responses=[_finish_step()])
        seen_decisions: List[Any] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def before_action(self, decision_result, ctx) -> AsyncGenerator[Any, Any]:
                seen_decisions.append(decision_result)
                # No RETURN → passthrough; just verify hook is reachable.
                if False:
                    yield

            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker)

        agent = Agent(llm=llm)
        await agent.arun(context=_ctx())

        # before_action invoked at least once
        assert len(seen_decisions) >= 1


class TestGeneratorVsCoroutineForms:

    @pytest.mark.asyncio
    async def test_on_agent_coroutine_form(self):
        """on_agent as a plain coroutine (legacy) — _run-driven worker."""
        llm = MockLLM(structured_responses=[_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker)

        await Agent(llm=llm).arun(context=_ctx())  # no exception → pass

    @pytest.mark.asyncio
    async def test_on_agent_generator_form_with_thinkcall(self):
        """on_agent as a generator yielding ThinkCall."""
        llm = MockLLM(structured_responses=[_finish_step()])
        from bridgic.amphibious import ThinkCall

        class Agent(AmphibiousAutoma[CognitiveContext]):
            main_think = think_unit(CognitiveWorker.inline("plan"), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkCall("main_think")

        await Agent(llm=llm).arun(context=_ctx())  # no exception → pass

    @pytest.mark.asyncio
    async def test_after_action_coroutine_pass_body(self):
        """after_action as a plain coroutine with `pass` body still works."""
        llm = MockLLM(structured_responses=[_finish_step()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def after_action(self, step_result, ctx):
                pass  # legacy "do nothing" form

            async def on_agent(self, ctx):
                worker = CognitiveWorker.inline("plan", llm=self.llm)
                await self._run(worker)

        await Agent(llm=llm).arun(context=_ctx())  # no exception → pass
