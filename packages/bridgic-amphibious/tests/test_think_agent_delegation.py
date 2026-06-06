"""Integration tests for external-agent delegation.

This file consolidates the ``ThinkAgent`` delegation path, the
``EnterAgent`` sub-run isolation path, and the supporting
``AgentWorker`` / ``BaseAgent`` / MCP machinery into one
integration-oriented suite. Everything is driven through
``await agent.arun(...)`` where it can be; the external coding-agent
run body is always **stubbed** (``_run_think_agent_body`` monkeypatched)
so no real CLI is ever spawned.

Scope covered:

* ``yield ThinkAgent(...)`` — per-yield worker clone (fresh, shares the
  BaseAgent identity), per-yield ``goal`` → sub-run ``user_input``,
  ``expose_tools`` narrowing of the sub-run toolset, unknown-name error.
* Factory / worker / agent type guards (merged).
* Decision channel round-trips (worker emits, framework executes).
* ``ClaudeCodeAgent`` constructor defaults + ``_write_mcp_config``.
* MCP binding derivation + ``_build_handler`` signature synthesis
  (parametrized).
* ``EnterAgent`` fresh sub-run delegation to ``on_agent``, parent
  isolation, slim-field kwarg rejection (parametrized), forced WORKFLOW
  mode.
* ``THINK_AGENT`` trace recording (one step per yield, with metadata).

Each test (or class) defines its OWN ``AmphibiousAutoma`` / context
subclass — never shared — so per-class registries (channels, declared
tools) cannot bleed across tests.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, List, Optional

import pytest

from bridgic.amphibious import (
    AgentRequest,
    AgentResult,
    AgentWorker,
    AmphibiousAutoma,
    BaseAgent,
    ClaudeCodeAgent,
    CognitiveWorker,
    Context,
    EnterAgent,
    OTAContext,
    RETURN,
    RunMode,
    Step,
    ThinkAgent,
    ThinkUnit,
    think_agent,
    think_unit,
)
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.types import Message, Role


# ---------------------------------------------------------------------------
# Shared local fixtures (deliberately tiny — anchor everything on a stub
# BaseAgent so the real CLI path is never exercised).
# ---------------------------------------------------------------------------


class _NoopAgent(BaseAgent):
    """A ``BaseAgent`` stub — ``run()`` returns a canned result, never
    spawns a subprocess. The BASE for workers in tests that don't go
    through the dispatcher (decision channel / binding helpers)."""

    def __init__(self, *, output: str = "noop-result") -> None:
        self.output = output

    async def run(self, request: AgentRequest) -> AgentResult:  # pragma: no cover
        return AgentResult(output=self.output, exit_code=0, completion="agent_done")


def _worker() -> AgentWorker:
    """A default AgentWorker anchored on a stub BaseAgent."""
    return AgentWorker(_NoopAgent())


async def _sample_tool(query: str) -> str:
    """A sample non-builtin tool used to validate binding derivation."""
    return f"echo:{query}"


_sample_tool_spec = FunctionToolSpec.from_raw(_sample_tool)

_BUILTIN_NAMES = {t.tool_name for t in ALL_BUILTIN_TOOLS}


class _ScriptedLLM:
    """Scripts the native function-calling path used by EnterAgent's
    in-process think unit. Each scripted response is a ``(tool_calls,
    content)`` pair; an empty ``tool_calls`` list makes the worker
    finish. ``call_count`` tracks per-cycle LLM invocations."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0

    async def aselect_tool(self, messages, tools, **kwargs):
        self.call_count += 1
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


def _finish():
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return [], "Done"


@pytest.fixture
def mock_run_body(monkeypatch):
    """Replace ``_run_think_agent_body`` with a recording stub.

    By the time the body would run, the dispatcher has already built the
    fresh sub-run OTA context (``user_input`` = the per-yield goal,
    ``tools`` = the ``expose_tools``-filtered parent toolset), so the stub
    captures ``ctx.user_input`` / ``ctx.tools`` to show what the worker
    would have seen. It returns an ``AgentResult`` — the parent unwraps
    ``.output`` for the yield value and the trace step reads its metadata.
    """
    captured: dict = {"workers": [], "ctx_goals": [], "ctx_tools": []}

    async def fake_body(self_agent, worker):
        captured["workers"].append(worker)
        ctx = self_agent._current_ota_context
        captured["ctx_goals"].append(ctx.user_input)
        captured["ctx_tools"].append([t.tool_name for t in ctx.tools])
        return AgentResult(
            output=f"mocked-{len(captured['workers'])}",
            exit_code=0,
            completion="agent_done",
        )

    monkeypatch.setattr(AmphibiousAutoma, "_run_think_agent_body", fake_body)
    return captured


# ---------------------------------------------------------------------------
# Type guards — factory / worker / agent (merged from the granular cases)
# ---------------------------------------------------------------------------


class TestDelegationTypeGuards:
    """``think_agent`` requires an ``AgentWorker``; ``AgentWorker``
    requires a ``BaseAgent``."""

    def test_think_agent_rejects_non_agent_worker(self):
        class NotAWorker:
            pass

        with pytest.raises(TypeError, match="requires an AgentWorker instance"):
            think_agent(NotAWorker())  # type: ignore[arg-type]

    def test_agent_worker_rejects_non_base_agent(self):
        with pytest.raises(TypeError, match="requires a BaseAgent instance"):
            AgentWorker(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AgentWorker default thinking — message assembly (goal + contract)
# ---------------------------------------------------------------------------


class TestAgentWorkerThinking:
    """The default ``thinking()`` assembles a usable message with the
    goal and the COMPLETION CONTRACT; subclasses can extend it."""

    @pytest.mark.asyncio
    async def test_default_thinking_assembles_goal_and_is_overridable(self):
        # Default: goal + completion contract, usable with no subclass.
        base_msg = await _worker().thinking(OTAContext(user_input="Audit the config"))
        assert "Audit the config" in base_msg
        assert "COMPLETION CONTRACT" in base_msg

        # A subclass can extend the assembled message via ``super()``.
        class _Custom(AgentWorker):
            async def thinking(self, ota_ctx, big_ctx=None):
                base = await super().thinking(ota_ctx, big_ctx)
                return base + "\n\nEXTRA: be terse."

        ext_msg = await _Custom(_NoopAgent()).thinking(OTAContext(user_input="x"))
        assert ext_msg.endswith("EXTRA: be terse.")
        assert "COMPLETION CONTRACT" in ext_msg


# ---------------------------------------------------------------------------
# Decision channel — AgentWorker emits, the framework executes
# ---------------------------------------------------------------------------


class TestDecisionChannel:
    """``AgentWorker._emit_decision`` surfaces each external-agent tool
    call as a decision onto ``_decision_channel`` and relays back whatever
    the framework's consumer resolves the future with — the worker never
    executes anything itself."""

    @pytest.mark.asyncio
    async def test_emit_decision_round_trips_through_channel(self):
        worker = AgentWorker(_NoopAgent())
        channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = channel
        captured: dict = {}

        async def _consumer():
            decision, result_future = await channel.get()
            captured["decision"] = decision
            # Stand in for the framework's _run_action_call: resolve with a
            # Step carrying the tool result.
            result_future.set_result(Step(result="ECHOED"))

        consumer = asyncio.create_task(_consumer())
        result = await worker._emit_decision("echo", {"msg": "hi"})
        await consumer

        # A decision was surfaced (produced, not executed)...
        assert captured["decision"] is not None
        # ...and the consumer's Step was relayed back, unwrapped to the
        # raw tool result the external agent expects.
        assert result == "ECHOED"

    @pytest.mark.asyncio
    async def test_emit_decision_handles_colliding_arg_names(self):
        """A bridged tool may name its own parameter ``description`` or
        ``tool_name``; those must not collide with ActionCall's
        constructor. Regression for the ``**kwargs`` splat bug."""
        worker = AgentWorker(_NoopAgent())
        channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = channel
        captured: dict = {}

        async def _consumer():
            decision, result_future = await channel.get()
            captured["decision"] = decision
            result_future.set_result(Step(result="OK"))

        consumer = asyncio.create_task(_consumer())
        result = await worker._emit_decision(
            "save_note", {"description": "a note", "tool_name": "x"},
        )
        await consumer

        assert result == "OK"
        arg_names = {
            a.name for a in captured["decision"].tool_calls[0].tool_arguments
        }
        assert arg_names == {"description", "tool_name"}

    @pytest.mark.asyncio
    async def test_emit_decision_propagates_consumer_exception(self):
        """If the framework's executor raises, the worker's await on the
        result future re-raises — the external agent sees the error."""
        worker = AgentWorker(_NoopAgent())
        channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = channel

        async def _consumer():
            _decision, result_future = await channel.get()
            result_future.set_exception(RuntimeError("tool blew up"))

        consumer = asyncio.create_task(_consumer())
        with pytest.raises(RuntimeError, match="tool blew up"):
            await worker._emit_decision("echo", {})
        await consumer

    @pytest.mark.asyncio
    async def test_emit_decision_without_channel_raises(self):
        """No channel wired → the worker wasn't driven by the framework;
        emitting a decision is a hard error."""
        worker = AgentWorker(_NoopAgent())
        assert worker._decision_channel is None
        with pytest.raises(RuntimeError, match="no decision channel"):
            await worker._emit_decision("echo", {})


# ---------------------------------------------------------------------------
# ClaudeCodeAgent — shipped concrete BaseAgent
# ---------------------------------------------------------------------------


class TestClaudeCodeAgent:
    def test_constructor_defaults(self):
        agent = ClaudeCodeAgent()
        assert agent.bin == "claude"
        assert agent.permission_mode == "bypassPermissions"
        assert agent.completion_timeout == 180.0
        assert set(agent.allowed_builtin_tools) == {
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        }
        assert isinstance(agent, BaseAgent)

    def test_write_mcp_config_serializes_servers(self, tmp_path):
        servers = {"amphi-bridge": {"type": "http", "url": "http://127.0.0.1:9/mcp"}}
        path = ClaudeCodeAgent._write_mcp_config(tmp_path, servers)
        assert path == tmp_path / "mcp_config.json"
        assert json.loads(path.read_text()) == {"mcpServers": servers}


# ---------------------------------------------------------------------------
# ThinkAgent dispatch — descriptor resolution, per-yield clone + snapshot,
# all driven through ``arun`` with the body stubbed.
# ---------------------------------------------------------------------------


class TestThinkAgentDispatch:

    @pytest.mark.asyncio
    async def test_dispatcher_drives_worker_and_returns_result(self, mock_run_body):
        class A(AmphibiousAutoma[OTAContext, Context]):
            do_thing = think_agent(_worker())

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                result = yield ThinkAgent("do_thing", goal="hello")
                yield RETURN(result)

        out = await A().arun(user_input="run")
        assert len(mock_run_body["workers"]) == 1
        # AgentResult.output flows out as the yield value and the arun return.
        assert out == "mocked-1"

    @pytest.mark.asyncio
    async def test_dispatcher_clones_worker_per_yield_sharing_agent(self, mock_run_body):
        """Two ThinkAgent yields → two distinct clones, neither the
        template, both sharing the same BaseAgent identity."""
        agent_obj = _NoopAgent()
        worker_template = AgentWorker(agent_obj)

        class A(AmphibiousAutoma[OTAContext, Context]):
            do_thing = think_agent(worker_template)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="a")
                yield ThinkAgent("do_thing", goal="b")

        await A().arun(user_input="run")
        workers = mock_run_body["workers"]
        assert len(workers) == 2
        assert all(w is not worker_template for w in workers)
        assert workers[0] is not workers[1]
        # The BaseAgent is shared (stateless per call), not deep-copied.
        assert all(w._agent is agent_obj for w in workers)

    @pytest.mark.asyncio
    async def test_dispatcher_snapshots_goal_onto_ctx(self, mock_run_body):
        """Per-yield ``goal`` becomes the fresh sub-run context's
        ``user_input``; omitting it inherits the parent run's input."""
        class A(AmphibiousAutoma[OTAContext, Context]):
            do_thing = think_agent(_worker())

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="step1")
                yield ThinkAgent("do_thing", goal="step2")
                yield ThinkAgent("do_thing")  # no override → inherit user_input

        await A().arun(user_input="run")
        assert mock_run_body["ctx_goals"] == ["step1", "step2", "run"]

    @pytest.mark.asyncio
    async def test_dispatcher_snapshots_expose_tools_onto_ctx(self, mock_run_body):
        """``expose_tools`` narrows the fresh sub-run context's ``tools``,
        at both descriptor level and per-yield level. Tools are declared on
        the OTA context class (``Context.tool``)."""
        async def alpha_tool(x: str) -> str:
            """alpha"""
            return x

        async def beta_tool(x: str) -> str:
            """beta"""
            return x

        class _ToolsOTA(OTAContext):
            pass

        _ToolsOTA.tool(FunctionToolSpec.from_raw(alpha_tool))
        _ToolsOTA.tool(FunctionToolSpec.from_raw(beta_tool))

        class A(AmphibiousAutoma[_ToolsOTA, Context]):
            do_thing = think_agent(_worker(), expose_tools=["alpha_tool"])

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing")  # descriptor-level filter
                yield ThinkAgent("do_thing", expose_tools=["beta_tool"])

        await A().arun(user_input="run")

        seen = [
            [n for n in names if n not in _BUILTIN_NAMES]
            for names in mock_run_body["ctx_tools"]
        ]
        assert seen == [["alpha_tool"], ["beta_tool"]]

    @pytest.mark.asyncio
    async def test_unknown_think_agent_name_raises_attribute_error(self, mock_run_body):
        class A(AmphibiousAutoma[OTAContext, Context]):
            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("nonexistent")

        with pytest.raises(AttributeError, match="does not match any"):
            await A().arun(user_input="run")


# ---------------------------------------------------------------------------
# MCP binding derivation + handler signature synthesis (parametrized)
# ---------------------------------------------------------------------------


class TestMCPBinding:
    """``_build_bindings_from_ctx`` exposes every non-builtin tool in
    ``ctx.tools`` (stripping builtins), carrying its JSON-Schema;
    ``_build_handler`` materialises a function whose signature mirrors the
    binding's parameters."""

    def test_default_exposes_user_tools_with_schema_and_strips_builtins(self):
        worker = _worker()
        ctx = OTAContext(user_input="t")
        ctx.tools.append(_sample_tool_spec)
        for spec in ALL_BUILTIN_TOOLS:
            ctx.tools.append(spec)

        bindings = worker._build_bindings_from_ctx(ctx, _BUILTIN_NAMES)
        names = [b.name for b in bindings]
        # User tool exposed, builtins stripped.
        assert "_sample_tool" in names
        assert not (set(names) & _BUILTIN_NAMES)
        # The binding carries the tool's JSON-schema.
        sample = next(b for b in bindings if b.name == "_sample_tool")
        params = sample.parameters
        assert params["type"] == "object"
        assert "query" in params.get("properties", {})
        assert "query" in params.get("required", [])

    @pytest.mark.parametrize(
        "parameters, checks",
        [
            # required param → no default, typed
            (
                {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
                {"q": {"has_default": False, "annotation": str}},
            ),
            # optional param → typed default of the right shape
            (
                {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": [],
                },
                {"n": {"has_default": True, "default": 0, "annotation": int}},
            ),
            # required ordered before optional regardless of property order
            (
                {
                    "type": "object",
                    "properties": {
                        "opt": {"type": "string"},
                        "req": {"type": "string"},
                    },
                    "required": ["req"],
                },
                {"_order": ["req", "opt"]},
            ),
        ],
        ids=["required-no-default", "optional-typed-default", "required-first"],
    )
    def test_build_handler_signature(self, parameters, checks):
        import inspect

        from bridgic.amphibious._mcp_host import MCPToolBinding, _build_handler

        async def _noop(name, args):  # pragma: no cover — never invoked
            return None

        binding = MCPToolBinding(name="probe", description="probe", parameters=parameters)
        sig = inspect.signature(_build_handler(binding, _noop))

        order = checks.pop("_order", None)
        if order is not None:
            names = list(sig.parameters.keys())
            assert names.index(order[0]) < names.index(order[1])
        for pname, expect in checks.items():
            param = sig.parameters[pname]
            if expect["has_default"]:
                assert param.default == expect["default"]
            else:
                assert param.default is param.empty
            assert param.annotation is expect["annotation"]


# ---------------------------------------------------------------------------
# EnterAgent — fresh sub-run delegation to on_agent + isolation
# ---------------------------------------------------------------------------


class _PlanCognitiveWorker(CognitiveWorker):
    async def thinking(
        self, ota_context: OTAContext, context: Optional[Context] = None,
    ) -> Any:
        return await self._llm.aselect_tool(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )


class TestEnterAgentDelegation:
    """``EnterAgent(goal=...)`` suspends ``on_workflow`` and runs
    ``on_agent`` in a fresh sub-run (the goal seeds the sub-run's
    ``user_input``); the parent OTA context is restored, never mutated."""

    @pytest.mark.asyncio
    async def test_default_delegates_to_on_agent_per_yield(self):
        on_agent_goals: List[str] = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            plan = think_unit(_PlanCognitiveWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                on_agent_goals.append(ota_context.user_input)
                yield ThinkUnit("plan")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield EnterAgent(goal="sub-goal-1")
                yield EnterAgent(goal="sub-goal-2")

        await Agent().arun(llm=_ScriptedLLM([_finish(), _finish()]), context=Context(), user_input="parent")
        assert on_agent_goals == ["sub-goal-1", "sub-goal-2"]

    @pytest.mark.asyncio
    async def test_sub_run_isolates_goal_from_parent(self):
        observed: List[tuple] = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            plan = think_unit(_PlanCognitiveWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                observed.append(("inside", ota_context.user_input))
                yield ThinkUnit("plan")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                observed.append(("before", ota_context.user_input))
                yield EnterAgent(goal="sub-task")
                observed.append(("after", ota_context.user_input))

        await Agent().arun(llm=_ScriptedLLM([_finish()]), context=Context(), user_input="parent goal")
        assert observed == [
            ("before", "parent goal"),
            ("inside", "sub-task"),
            ("after", "parent goal"),
        ]

    @pytest.mark.asyncio
    async def test_no_on_agent_override_raises(self):
        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield EnterAgent(goal="orphan")

        with pytest.raises(RuntimeError, match="requires an on_agent"):
            await Agent().arun(context=Context(), user_input="parent")

    @pytest.mark.asyncio
    async def test_enter_agent_in_forced_workflow_mode(self):
        """Forced ``mode=WORKFLOW`` + EnterAgent dispatches through the
        recursive ``_invoke_template`` branch (no state machine)."""
        on_agent_goals: List[str] = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            plan = think_unit(_PlanCognitiveWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                on_agent_goals.append(ota_context.user_input)
                yield ThinkUnit("plan")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield EnterAgent(goal="forced-workflow-sub")

        await Agent().arun(llm=_ScriptedLLM([_finish()]), context=Context(), mode=RunMode.WORKFLOW)
        assert on_agent_goals == ["forced-workflow-sub"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"worker": object()},
            {"max_attempts": 3},
            {"tools": ["t1"]},
            {"skills": ["s1"]},
            {"history": None},
        ],
        ids=["worker", "max_attempts", "tools", "skills", "history"],
    )
    def test_enter_agent_rejects_legacy_kwargs(self, kwargs):
        """``EnterAgent`` carries ONLY ``goal=`` — every removed scoping /
        attempt-budget field is rejected at construction."""
        with pytest.raises(TypeError):
            EnterAgent(goal="x", **kwargs)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# THINK_AGENT trace recording — one step per yield, with metadata
# ---------------------------------------------------------------------------


class TestThinkAgentTrace:
    """When a ThinkAgent fires under an active trace, the dispatcher
    records a ``THINK_AGENT`` step into ``trace.json`` — one per yield,
    carrying name + goal + result + exit_code/completion_signal."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "goals",
        [["the-goal"], ["g1", "g2"]],
        ids=["single-yield", "two-yields"],
    )
    async def test_trace_records_one_think_agent_step_per_yield(
        self, tmp_path, mock_run_body, goals,
    ):
        class A(AmphibiousAutoma[OTAContext, Context]):
            do_thing = think_agent(_worker())

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                for g in goals:
                    yield ThinkAgent("do_thing", goal=g)

        await A().arun(user_input="x", trace=True, workdir=tmp_path)
        run_dir = next((tmp_path / "runs").iterdir())
        trace = json.loads((run_dir / "trace.json").read_text())
        steps = [s for s in trace["history"] if s.get("name") == "think_agent"]

        assert [s["structured_output"]["goal"] for s in steps] == goals
        # Each step carries the full delegate-outcome envelope.
        for s in steps:
            assert s["output_type"] == "think_agent"
            assert s["think_agent_name"] == "do_thing"
            assert s["structured_output"]["result"].startswith("mocked-")
            assert s["structured_output"]["exit_code"] == 0
            assert s["structured_output"]["completion_signal"] == "agent_done"
