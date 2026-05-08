"""Tests for the ``WorkerRunner`` Protocol abstraction (Q3).

Verifies:

* A custom (non-CognitiveWorker) class satisfying the ``WorkerRunner``
  Protocol can be plugged into ``think_unit`` / ``ThinkCall``
* The framework calls ``run(agent, ctx)`` once and skips the
  observe-think-act cycle
* ``until`` / ``max_attempts`` / ``tools`` overlays are ignored on the
  WorkerRunner path (the runner manages its own loop)
* Passing a non-CognitiveWorker non-WorkerRunner raises ``TypeError``
"""

from typing import Any, AsyncGenerator, List, Union

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    WorkerRunner,
    ActionCall,
    HumanCall,
    AgentCall,
    LLMCall,
    ThinkCall,
    Step,
    StepToolCall,
    ToolArgument,
    think_unit,
)


# ---------------------------------------------------------------------------
# Test fixtures: external WorkerRunner implementations
# ---------------------------------------------------------------------------


class RecordingExternalWorker:
    """A ``WorkerRunner`` that records every ``run`` invocation.

    Demonstrates the integration pattern for external agent runtimes:
    no inheritance from ``CognitiveWorker``, no observe-think-act
    decomposition — just ``async def run(agent, ctx)`` driving its own
    work and mutating ctx.cognitive_history if it wants visibility.
    """

    def __init__(self, label: str = "external"):
        self.label = label
        # Capture goal at run time as a string, not the ctx reference,
        # because ctx is mutated back when the snapshot exits.
        self.observed_goals: List[str] = []
        self.invocations: int = 0

    async def run(self, agent, ctx) -> None:
        self.invocations += 1
        self.observed_goals.append(ctx.goal)
        # Mimic an external runtime that produces a single "step" record.
        ctx.add_info(Step(
            content=f"[{self.label}] handled goal: {ctx.goal}",
            result=None,
            metadata={"runner": self.label},
        ))


class CountingExternalWorker:
    """Counts how many times its ``run`` was invoked."""

    def __init__(self):
        self.invocations = 0

    async def run(self, agent, ctx) -> None:
        self.invocations += 1


class _NoRunObject:
    """Placeholder that is neither a CognitiveWorker nor a WorkerRunner."""

    def some_method(self) -> None: ...


class _DummyLLM:
    """No-op LLM stub; satisfies arun()'s AGENT-mode precondition.

    Tests in this file never trigger any real LLM operations because
    the WorkerRunner path bypasses observe-think-act entirely.
    """

    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkerRunnerProtocol:

    def test_external_runner_satisfies_protocol(self):
        runner = RecordingExternalWorker()
        assert isinstance(runner, WorkerRunner)

    def test_object_without_run_does_not_satisfy_protocol(self):
        assert not isinstance(_NoRunObject(), WorkerRunner)


class TestThinkCallWithWorkerRunner:

    @pytest.mark.asyncio
    async def test_external_runner_invoked_once_per_thinkcall(self):
        """yield ThinkCall(name) where descriptor wraps a WorkerRunner →
        runner.run(agent, ctx) called once."""
        runner = CountingExternalWorker()

        class Agent(AmphibiousAutoma[CognitiveContext]):
            external = think_unit(runner)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkCall("external")
                yield ThinkCall("external")
                yield ThinkCall("external")

        await Agent(llm=_DummyLLM()).arun(context=CognitiveContext(goal="Q3 test"))

        assert runner.invocations == 3

    @pytest.mark.asyncio
    async def test_external_runner_receives_agent_and_ctx(self):
        """Verify the (agent, ctx) arguments handed to run()."""
        recorder = RecordingExternalWorker(label="rec")

        class Agent(AmphibiousAutoma[CognitiveContext]):
            external = think_unit(recorder)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkCall("external")

        agent = Agent(llm=_DummyLLM())
        ctx = CognitiveContext(goal="hello-runner")
        await agent.arun(context=ctx)

        # The runner saw the active goal at dispatch time.
        assert recorder.observed_goals == ["hello-runner"]
        # The runner mutated history; the framework saw it.
        steps = ctx.cognitive_history.get_all()
        assert any("[rec] handled goal: hello-runner" in s.content for s in steps)

    @pytest.mark.asyncio
    async def test_overlays_ignored_on_worker_runner_path(self):
        """until / max_attempts / tools / skills don't loop a WorkerRunner."""
        runner = CountingExternalWorker()

        class Agent(AmphibiousAutoma[CognitiveContext]):
            # max_attempts=99 and a permissive until — would loop a CognitiveWorker
            external = think_unit(
                runner,
                max_attempts=99,
                until=lambda ctx: False,  # never stop
            )

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkCall("external")

        await Agent(llm=_DummyLLM()).arun(context=CognitiveContext(goal="overlay-ignore"))

        # WorkerRunner is invoked exactly once per ThinkCall — overlays ignored.
        assert runner.invocations == 1


class TestRunDispatchType:

    @pytest.mark.asyncio
    async def test_non_runner_non_cognitive_worker_raises_typeerror(self):
        """Passing a random object to _run is rejected with a clear TypeError."""

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx):
                await self._run(_NoRunObject())  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="CognitiveWorker or a WorkerRunner"):
            await Agent(llm=_DummyLLM()).arun(context=CognitiveContext(goal="bad-arg"))
