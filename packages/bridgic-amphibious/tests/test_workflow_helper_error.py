"""Tests for generator-internal exceptions raised inside ``on_workflow``.

When user code between yields (helpers, conditionals, parameter prep) raises,
the framework cannot resume the generator (asend() would raise StopIteration).
The expected behavior is mode-dependent:

- WORKFLOW (pure, no on_agent override): re-raise the original exception
- AMPHIFLOW (on_agent overridden): hand the remaining task to on_agent(ctx)
- AMPHIFLOW without on_agent override: explicit RuntimeError

These tests cover the exception path added in ``_run_workflow``'s inner try
around ``gen.__anext__()`` / ``gen.asend()``.
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
    StepToolCall,
    ToolArgument,
    ThinkUnit,
    think_unit,
)


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class _MockLLM:
    """Returns a fixed sequence of structured-output responses."""

    def __init__(self, responses):
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


def _finish_decision() -> ThinkDecision:
    return ThinkDecision(
        step_content="Recovered",
        output=[],
        finish=True,
    )


class TestWorkflowGeneratorError:

    @pytest.mark.asyncio
    async def test_helper_raises_in_workflow_mode_propagates(self):
        """Pure WORKFLOW mode: a helper exception inside on_workflow propagates as-is."""

        sentinel_message = "boom-from-helper"

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                # Generator raises before any yield — equivalent to a helper
                # call between yields blowing up.
                raise ValueError(sentinel_message)
                yield  # pragma: no cover

        agent = Agent()  # No on_agent override → resolved as RunMode.WORKFLOW

        with pytest.raises(ValueError, match=sentinel_message):
            await agent.arun(goal="Trigger helper failure")

    @pytest.mark.asyncio
    async def test_helper_raises_in_amphiflow_falls_back_to_on_agent(self):
        """AMPHIFLOW mode: a helper exception routes execution to on_agent(ctx)."""

        on_agent_calls = []
        llm = _MockLLM([_finish_decision()])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                on_agent_calls.append(ctx.goal)
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                raise RuntimeError("simulated helper failure")
                yield  # pragma: no cover

        agent = Agent()
        # arun should not raise — fallback handles it.
        await agent.arun(llm=llm, goal="Trigger amphiflow fallback")

        assert len(on_agent_calls) == 1, (
            f"on_agent should have been invoked exactly once for the fallback, "
            f"got {len(on_agent_calls)} invocations"
        )

    @pytest.mark.asyncio
    async def test_amphiflow_without_agent_override_raises_at_routing(self):
        """No on_agent override + AMPHIFLOW forced: rejected at ``_resolve_mode``.

        ``_resolve_mode`` validates that an explicit ``RunMode.AMPHIFLOW``
        requires both ``on_agent`` and ``on_workflow`` to be overridden;
        the run is rejected at the routing boundary before any driver
        runs, with an error that names the missing template(s).
        """
        from bridgic.amphibious import RunMode

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                raise KeyError("missing key")
                yield  # pragma: no cover

        agent = Agent()

        with pytest.raises(
            RuntimeError,
            match=r"requested mode=RunMode\.AMPHIFLOW but does not override on_agent\(\)",
        ):
            await agent.arun(
                goal="Trigger amphiflow without agent",
                mode=RunMode.AMPHIFLOW,
            )
