"""Tests for ``arun(trace=..., workdir=...)`` and per-run artifacts.

``trace`` and ``workdir`` are orthogonal:

* ``trace=True``  ⇒ in-memory ``AgentTrace`` is created (recording).
* ``workdir=path`` ⇒ ``<workdir>/runs/<run_id>/`` is materialised so
                    the run dir exists (currently used only as the
                    ``AgentTrace`` persistence target).
* Both           ⇒ ``AgentTrace`` also persists ``<run>/trace.json``.

ThinkAgent delegate state lives entirely in-memory now (no per-delegate
artifact directory); the trace step records the delegate outcome
metadata inside the unified ``AgentTrace`` instead.
"""

import json
from pathlib import Path
from typing import Any, AsyncGenerator, List

import pytest

from bridgic.amphibious import (
    AgentRequest,
    AgentResult,
    AgentWorker,
    AmphibiousAutoma,
    BaseAgent,
    CognitiveContext,
    RETURN,
    ThinkAgent,
    think_agent,
)


class _DummyLLM:
    """No-op LLM stub."""

    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


class _WorkflowAgent(AmphibiousAutoma[CognitiveContext]):
    """A minimal workflow-only agent — no LLM needed."""

    async def on_workflow(self, ctx) -> AsyncGenerator[Any, Any]:
        yield RETURN("done")


# ---------------------------------------------------------------------------
# No-workdir baseline
# ---------------------------------------------------------------------------


class TestArunWithoutWorkdir:
    """Default behaviour: no workdir → no run dir, no artifacts written."""

    @pytest.mark.asyncio
    async def test_arun_without_workdir_creates_nothing(self, tmp_path):
        await _WorkflowAgent().arun(goal="x")
        # Nothing should be created under tmp_path by the framework itself.
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_current_run_dir_none_after_arun(self):
        agent = _WorkflowAgent()
        await agent.arun(goal="x")
        assert agent._current_run_dir is None


# ---------------------------------------------------------------------------
# Workdir-driven run dir + artifacts
# ---------------------------------------------------------------------------


class TestArunWithWorkdir:
    """``workdir=path`` alone materialises ``<workdir>/runs/<id>/`` so
    delegates have a home — but does NOT write ``trace.json`` unless
    ``trace=True`` is also set."""

    @pytest.mark.asyncio
    async def test_workdir_creates_run_subdirectory(self, tmp_path):
        await _WorkflowAgent().arun(goal="x", workdir=tmp_path)
        runs_dir = tmp_path / "runs"
        assert runs_dir.is_dir()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        # run-id is timestamp-prefixed + 4-hex suffix.
        assert run_dirs[0].name.count("-") >= 1

    @pytest.mark.asyncio
    async def test_workdir_alone_does_not_write_trace_json(self, tmp_path):
        """``workdir`` without ``trace=True`` → run dir exists (for
        delegate artifacts) but no ``trace.json`` is written."""
        await _WorkflowAgent().arun(goal="hello", workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        files = {p.name for p in run_dir.iterdir() if p.is_file()}
        assert files == set()  # empty: no delegates fired, no trace

    @pytest.mark.asyncio
    async def test_run_directory_contains_only_trace_json(self, tmp_path):
        """trace=True + workdir → single unified ``trace.json`` artifact
        (no legacy meta / ctx_initial / ctx_final)."""
        await _WorkflowAgent().arun(goal="hello", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        files = {p.name for p in run_dir.iterdir() if p.is_file()}
        assert files == {"trace.json"}

    @pytest.mark.asyncio
    async def test_trace_metadata_has_expected_fields(self, tmp_path):
        """Metadata previously split across ``meta.json`` is now part of
        ``trace.json``'s ``metadata`` block."""
        await _WorkflowAgent().arun(goal="hello", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        meta = trace["metadata"]
        assert "agent_class" in meta
        assert "mode" in meta
        assert "run_id" in meta
        assert "start_time" in meta
        assert "end_time" in meta
        assert "spent_time" in meta
        assert "cost_time" in meta
        assert meta["mode"] == "workflow"

    @pytest.mark.asyncio
    async def test_trace_captures_goal(self, tmp_path):
        """The original arun goal lands in the top-level ``goal`` field."""
        await _WorkflowAgent().arun(
            goal="hello-goal", trace=True, workdir=tmp_path,
        )
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        assert trace["goal"] == "hello-goal"

    @pytest.mark.asyncio
    async def test_workdir_accepts_string_path(self, tmp_path):
        await _WorkflowAgent().arun(goal="x", workdir=str(tmp_path))
        assert (tmp_path / "runs").exists()

    @pytest.mark.asyncio
    async def test_run_dir_cleared_after_arun(self, tmp_path):
        agent = _WorkflowAgent()
        await agent.arun(goal="x", workdir=tmp_path)
        # ``_current_run_dir`` is per-run state; cleared on exit.
        assert agent._current_run_dir is None

    @pytest.mark.asyncio
    async def test_trace_json_has_unified_shape(self, tmp_path):
        """Unified shape: goal + metadata + history at top level."""
        await _WorkflowAgent().arun(goal="x", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        assert set(trace.keys()) == {"goal", "metadata", "history"}


# ---------------------------------------------------------------------------
# Trace activation — the trace / workdir flags
# ---------------------------------------------------------------------------


class TestTraceActivation:
    """``trace`` activates ``AgentTrace``; ``workdir`` only controls
    persistence destination. Four combos make up the truth table."""

    @pytest.mark.asyncio
    async def test_neither_flag_leaves_trace_none(self):
        """trace=False, workdir=None → no AgentTrace at all."""
        agent = _WorkflowAgent()
        await agent.arun(goal="x")
        assert agent._agent_trace is None

    @pytest.mark.asyncio
    async def test_workdir_alone_does_not_activate_trace(self, tmp_path):
        """trace=False, workdir=path → run dir exists (for delegates),
        but no AgentTrace is created."""
        agent = _WorkflowAgent()
        await agent.arun(goal="x", workdir=tmp_path)
        assert agent._agent_trace is None
        assert agent._current_run_dir is None  # reset in finally

    @pytest.mark.asyncio
    async def test_trace_alone_keeps_in_memory_trace(self):
        """trace=True, workdir=None → AgentTrace in memory, no disk."""
        agent = _WorkflowAgent()
        await agent.arun(goal="x", trace=True)
        assert agent._agent_trace is not None

    @pytest.mark.asyncio
    async def test_trace_plus_workdir_persists_and_keeps_object(self, tmp_path):
        """trace=True + workdir=path → AgentTrace persists to disk and
        survives on ``self._agent_trace`` for post-run inspection."""
        agent = _WorkflowAgent()
        await agent.arun(goal="x", trace=True, workdir=tmp_path)
        assert agent._agent_trace is not None
        run_dir = next((tmp_path / "runs").iterdir())
        assert (run_dir / "trace.json").is_file()


# ---------------------------------------------------------------------------
# ThinkAgent trace recording — trace.json contains a step per yield
# ---------------------------------------------------------------------------


class _NoopAgent(BaseAgent):
    """A ``BaseAgent`` stub — ``run()`` never spawns a subprocess."""

    async def run(self, request: AgentRequest) -> AgentResult:  # pragma: no cover
        return AgentResult(output="ok", exit_code=0, completion="agent_done")


def _noop_worker() -> AgentWorker:
    """A default AgentWorker whose body is patched out in these tests."""
    return AgentWorker(_NoopAgent())


@pytest.fixture
def mock_run_body(monkeypatch):
    """Stub out ``_run_think_agent_body`` so no subprocess is spawned.

    Stashes a deterministic ``AgentResult`` on the worker so the
    ``_record_think_agent`` envelope writes a complete trace step.
    """
    captured: dict = {"workers": []}

    async def fake_body(self_agent, worker):
        captured["workers"].append(worker)
        # The body returns an AgentResult — the parent unwraps .output
        # and folds exit_code / completion into the trace step.
        return AgentResult(output="ok", exit_code=0, completion="agent_done")

    monkeypatch.setattr(AmphibiousAutoma, "_run_think_agent_body", fake_body)
    return captured


class TestThinkAgentTraceRecording:
    """When ThinkAgent fires under an active trace, the dispatcher records
    a ``THINK_AGENT`` step so ``trace.json`` reflects what happened."""

    @pytest.mark.asyncio
    async def test_trace_contains_think_agent_step(
        self, tmp_path, mock_run_body,
    ):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_noop_worker())

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="the-goal")

        await A().arun(goal="x", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        think_agent_steps = [
            s for s in trace["history"] if s.get("name") == "think_agent"
        ]
        assert len(think_agent_steps) == 1
        step = think_agent_steps[0]
        assert step["output_type"] == "think_agent"
        assert step["think_agent_name"] == "do_thing"
        assert step["structured_output"]["goal"] == "the-goal"
        assert step["structured_output"]["result"] == "ok"
        # Worker outcome metadata folded into the trace step.
        assert step["structured_output"]["exit_code"] == 0
        assert step["structured_output"]["completion_signal"] == "agent_done"

    @pytest.mark.asyncio
    async def test_trace_records_one_step_per_yield(
        self, tmp_path, mock_run_body,
    ):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_noop_worker())

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="g1")
                yield ThinkAgent("do_thing", goal="g2")

        await A().arun(goal="x", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        think_agent_steps = [
            s for s in trace["history"] if s.get("name") == "think_agent"
        ]
        goals = [s["structured_output"]["goal"] for s in think_agent_steps]
        assert goals == ["g1", "g2"]
