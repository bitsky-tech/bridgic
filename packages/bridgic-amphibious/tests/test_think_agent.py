"""Tests for the ``ThinkAgent`` primitive, ``AgentWorker``, and ``BaseAgent``.

Covers:

* ``think_agent(worker, ...)`` factory and ``ThinkAgentDescriptor``
  shape (wraps an ``AgentWorker`` instance).
* ``AgentWorker`` — concrete worker, peer of ``CognitiveWorker``,
  anchored on a ``BaseAgent``; default ``thinking()`` message assembly;
  state-isolated ``_clone()``.
* ``BaseAgent`` / ``ClaudeCodeAgent`` — the external coding-agent
  abstraction and its shipped ``claude code`` driver.
* Dispatcher branch in ``_dispatch_step`` (scope enforcement, descriptor
  resolution, per-yield ``goal`` / ``expose_tools`` snapshot onto ctx).
* Auto-derivation of MCP tool bindings from ``ctx.tools``.
* ``_build_handler`` signature precision.

Integration with a real ``claude`` subprocess is out of scope — that
path lives in the standalone demos. Here ``_run_think_agent_body`` (or
``BaseAgent.run``) is stubbed so the tests are deterministic.
"""

import asyncio
from typing import Any, AsyncGenerator, List, Optional

import pytest

from bridgic.amphibious import (
    AgentRequest,
    AgentResult,
    AgentWorker,
    AmphibiousAutoma,
    BaseAgent,
    ClaudeCodeAgent,
    CognitiveContext,
    RETURN,
    Step,
    ThinkAgent,
    ThinkAgentDescriptor,
    think_agent,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _NoopAgent(BaseAgent):
    """A ``BaseAgent`` stub — ``run()`` returns a canned result, never
    spawns a subprocess. Used as the BASE for workers in tests that
    don't exercise the real CLI path."""

    def __init__(self, *, output: str = "noop-result") -> None:
        self.output = output

    async def run(self, request: AgentRequest) -> AgentResult:  # pragma: no cover
        return AgentResult(
            output=self.output, exit_code=0, completion="agent_done",
        )


def _worker() -> AgentWorker:
    """A default AgentWorker for descriptor / dispatch tests."""
    return AgentWorker(_NoopAgent())


async def _sample_tool(query: str) -> str:
    """A sample non-builtin tool used to validate auto-derivation."""
    return f"echo:{query}"


_sample_tool_spec = FunctionToolSpec.from_raw(_sample_tool)


# ---------------------------------------------------------------------------
# Factory + descriptor
# ---------------------------------------------------------------------------


class TestThinkAgentFactory:
    """``think_agent(worker, ...)`` returns a properly-populated descriptor."""

    def test_factory_returns_descriptor(self):
        d = think_agent(_worker())
        assert isinstance(d, ThinkAgentDescriptor)

    def test_factory_stores_worker_template(self):
        worker = _worker()
        d = think_agent(worker)
        assert d._worker_template is worker

    def test_factory_stores_expose_tools(self):
        d = think_agent(_worker(), expose_tools=["a", "b"])
        assert d._expose_tools == ["a", "b"]

    def test_factory_default_expose_tools_is_none(self):
        d = think_agent(_worker())
        assert d._expose_tools is None

    def test_factory_rejects_non_agent_worker(self):
        class NotAWorker:
            pass

        with pytest.raises(TypeError, match="requires an AgentWorker instance"):
            think_agent(NotAWorker())  # type: ignore[arg-type]

    def test_descriptor_get_is_idempotent_on_class_and_instance(self):
        class A(AmphibiousAutoma[CognitiveContext]):
            x = think_agent(_worker())

            async def on_agent(self, ctx):
                if False:  # pragma: no cover
                    yield

        assert isinstance(A.x, ThinkAgentDescriptor)
        a = A()
        assert a.x is A.x  # __get__ returns self


# ---------------------------------------------------------------------------
# AgentWorker — concrete worker anchored on a BaseAgent
# ---------------------------------------------------------------------------


class TestAgentWorker:
    """``AgentWorker`` is concrete, peer of ``CognitiveWorker``, holding
    a ``BaseAgent`` the way ``CognitiveWorker`` holds a ``BaseLlm``."""

    def test_requires_a_base_agent(self):
        with pytest.raises(TypeError, match="requires a BaseAgent instance"):
            AgentWorker(object())  # type: ignore[arg-type]

    def test_holds_the_agent(self):
        agent = _NoopAgent()
        worker = AgentWorker(agent)
        assert worker._agent is agent

    def test_init_is_minimal(self):
        """AgentWorker.__init__ takes only the BASE + framework-generic
        logging knobs. No ``goal`` / ``tools`` / ``skills`` / ``history``
        params — those are context data, read straight from ``context``."""
        import inspect

        params = list(inspect.signature(AgentWorker.__init__).parameters.keys())
        assert params == ["self", "agent", "verbose", "verbose_prompt"]

    def test_clone_is_fresh_but_shares_agent(self):
        """``_clone()`` gives a fresh worker; the BaseAgent is shared
        (stateless per call, like a BaseLlm across CognitiveWorker clones)."""
        agent = _NoopAgent()
        worker = AgentWorker(agent, verbose=True)
        clone = worker._clone()
        assert clone is not worker
        assert clone._agent is agent          # shared, not deep-copied
        assert clone._verbose is True

    @pytest.mark.asyncio
    async def test_default_thinking_includes_goal(self):
        """The default ``thinking()`` assembles a message carrying the
        context goal — usable with no subclassing."""
        worker = _worker()
        ctx = CognitiveContext(goal="Audit the config file")
        message = await worker.thinking(ctx)
        assert "Audit the config file" in message
        assert "COMPLETION CONTRACT" in message

    @pytest.mark.asyncio
    async def test_thinking_is_overridable(self):
        class Custom(AgentWorker):
            async def thinking(self, context):
                base = await super().thinking(context)
                return base + "\n\nEXTRA: be terse."

        worker = Custom(_NoopAgent())
        ctx = CognitiveContext(goal="x")
        message = await worker.thinking(ctx)
        assert message.endswith("EXTRA: be terse.")


# ---------------------------------------------------------------------------
# Decision channel — AgentWorker emits, framework executes
# ---------------------------------------------------------------------------


class TestDecisionChannel:
    """``AgentWorker._emit_decision`` surfaces each CLI tool call as a
    decision onto ``_decision_channel`` and relays back whatever the
    framework's consumer resolves the future with — the worker never
    executes anything itself (no ``_run_action_call`` in AgentWorker)."""

    @pytest.mark.asyncio
    async def test_emit_decision_round_trips_through_channel(self):
        worker = AgentWorker(_NoopAgent())
        channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = channel

        captured: dict = {}

        async def _consumer():
            decision, result_future = await channel.get()
            captured["decision"] = decision
            # Stand in for the framework's _run_action_call: resolve the
            # future with a Step carrying the tool result.
            result_future.set_result(Step(content="done", result="ECHOED"))

        consumer = asyncio.create_task(_consumer())
        result = await worker._emit_decision("echo", {"msg": "hi"})
        await consumer

        # A decision was surfaced (the worker produced it, did not execute)...
        assert captured["decision"] is not None
        # ...and the consumer's Step was relayed back, unwrapped to the
        # raw tool result the external agent expects.
        assert result == "ECHOED"

    @pytest.mark.asyncio
    async def test_emit_decision_handles_colliding_arg_names(self):
        """A bridged tool may name its own parameter ``description`` or
        ``tool_name`` — those must not collide with ActionCall's
        constructor signature. Regression for the ``**kwargs`` splat bug.
        """
        worker = AgentWorker(_NoopAgent())
        channel: asyncio.Queue = asyncio.Queue()
        worker._decision_channel = channel

        captured: dict = {}

        async def _consumer():
            decision, result_future = await channel.get()
            captured["decision"] = decision
            result_future.set_result(Step(content="done", result="OK"))

        consumer = asyncio.create_task(_consumer())
        # Both keys collide with ActionCall.__init__'s own kw-params.
        result = await worker._emit_decision(
            "save_note", {"description": "a note", "tool_name": "x"},
        )
        await consumer

        assert result == "OK"
        # Both colliding-named args survived as tool arguments.
        arg_names = {
            a.name for a in captured["decision"].output[0].tool_arguments
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
        """No channel wired → the worker wasn't driven by
        ``_run_think_agent``; emitting a decision is a hard error."""
        worker = AgentWorker(_NoopAgent())
        assert worker._decision_channel is None
        with pytest.raises(RuntimeError, match="no decision channel"):
            await worker._emit_decision("echo", {})


# ---------------------------------------------------------------------------
# BaseAgent / ClaudeCodeAgent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    """``BaseAgent`` is abstract — ``run()`` must be overridden."""

    @pytest.mark.asyncio
    async def test_run_raises_by_default(self):
        with pytest.raises(NotImplementedError, match="run"):
            await BaseAgent().run(None)  # type: ignore[arg-type]


class TestClaudeCodeAgent:
    """The shipped concrete ``BaseAgent`` for ``claude code``."""

    def test_constructor_defaults(self):
        agent = ClaudeCodeAgent()
        assert agent.bin == "claude"
        assert agent.permission_mode == "bypassPermissions"
        assert agent.completion_timeout == 180.0
        assert set(agent.allowed_builtin_tools) == {
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        }

    def test_constructor_overrides(self):
        agent = ClaudeCodeAgent(
            bin="/usr/local/bin/claude",
            allowed_builtin_tools=["Read", "Grep"],
            permission_mode="acceptEdits",
            completion_timeout=42.0,
        )
        assert agent.bin == "/usr/local/bin/claude"
        assert agent.allowed_builtin_tools == ["Read", "Grep"]
        assert agent.permission_mode == "acceptEdits"
        assert agent.completion_timeout == 42.0

    def test_is_a_base_agent(self):
        assert isinstance(ClaudeCodeAgent(), BaseAgent)

    def test_write_mcp_config_serializes_servers(self, tmp_path):
        servers = {"amphi-bridge": {"type": "http", "url": "http://127.0.0.1:9/mcp"}}
        path = ClaudeCodeAgent._write_mcp_config(tmp_path, servers)
        import json

        assert path == tmp_path / "mcp_config.json"
        data = json.loads(path.read_text())
        assert data == {"mcpServers": servers}


# ---------------------------------------------------------------------------
# Dispatcher branch (body stubbed so no subprocess is spawned)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_run_body(monkeypatch):
    """Replace ``_run_think_agent_body`` with a recording stub.

    By the time the body would run, the dispatcher has already wrapped
    the context in a ``snapshot(goal=..., tools=...)``, so capturing
    ``ctx.goal`` / ``ctx.tools`` inside the stub shows what the worker
    would have seen. The stub stashes an ``AgentResult`` on the worker
    (the envelope record reads it) and returns the output string.
    """
    captured: dict = {"workers": [], "ctx_goals": [], "ctx_tools": []}

    async def fake_body(self_agent, worker):
        captured["workers"].append(worker)
        ctx = self_agent._current_context
        captured["ctx_goals"].append(ctx.goal)
        captured["ctx_tools"].append([t.tool_name for t in ctx.tools.get_all()])
        # The body returns an AgentResult — the parent unwraps .output.
        return AgentResult(
            output=f"mocked-{len(captured['workers'])}",
            exit_code=0,
            completion="agent_done",
        )

    monkeypatch.setattr(AmphibiousAutoma, "_run_think_agent_body", fake_body)
    return captured


class TestThinkAgentDispatch:
    """Verify the dispatcher branch resolves descriptor + clones worker."""

    @pytest.mark.asyncio
    async def test_dispatcher_drives_worker_and_returns_result(self, mock_run_body):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_worker())

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                result = yield ThinkAgent("do_thing", goal="hello")
                yield RETURN(result)

        out = await A().arun(goal="run")
        assert len(mock_run_body["workers"]) == 1
        assert out == "mocked-1"

    @pytest.mark.asyncio
    async def test_dispatcher_clones_worker_per_yield(self, mock_run_body):
        worker_template = _worker()

        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(worker_template)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="a")
                yield ThinkAgent("do_thing", goal="b")

        await A().arun(goal="run")
        workers = mock_run_body["workers"]
        assert len(workers) == 2
        assert all(w is not worker_template for w in workers)
        assert workers[0] is not workers[1]

    @pytest.mark.asyncio
    async def test_dispatcher_snapshots_goal_onto_ctx(self, mock_run_body):
        """Per-yield ``goal`` flows through ``ctx.goal`` (snapshot); the
        original ``ctx.goal`` is restored after each delegation."""
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_worker())

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="step1")
                yield ThinkAgent("do_thing", goal="step2")
                yield ThinkAgent("do_thing")  # no override → keep ctx.goal

        await A().arun(goal="run")
        assert mock_run_body["ctx_goals"] == ["step1", "step2", "run"]

    @pytest.mark.asyncio
    async def test_dispatcher_snapshots_expose_tools_onto_ctx(self, mock_run_body):
        """``expose_tools`` filters ``ctx.tools`` via snapshot."""
        async def alpha_tool(x: str) -> str:
            """alpha"""
            return x

        async def beta_tool(x: str) -> str:
            """beta"""
            return x

        alpha_spec = FunctionToolSpec.from_raw(alpha_tool)
        beta_spec = FunctionToolSpec.from_raw(beta_tool)

        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_worker(), expose_tools=["alpha_tool"])

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing")  # descriptor-level filter
                yield ThinkAgent("do_thing", expose_tools=["beta_tool"])

        ctx = CognitiveContext(goal="run")
        ctx.tools.add(alpha_spec)
        ctx.tools.add(beta_spec)
        await A().arun(context=ctx)

        from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS
        builtin_names = {t.tool_name for t in ALL_BUILTIN_TOOLS}
        seen = [
            [n for n in names if n not in builtin_names]
            for names in mock_run_body["ctx_tools"]
        ]
        assert seen == [["alpha_tool"], ["beta_tool"]]

    @pytest.mark.asyncio
    async def test_think_agent_unknown_name_raises_attribute_error(self, mock_run_body):
        class A(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("nonexistent")

        with pytest.raises(AttributeError, match="does not match any"):
            await A().arun(goal="run")

    @pytest.mark.asyncio
    async def test_think_agent_outside_on_agent_raises_runtime_error(self, mock_run_body):
        """ThinkAgent yielded from on_workflow should fail scope validation."""

        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(_worker())

            async def on_workflow(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="x")

        with pytest.raises(RuntimeError, match="only valid inside"):
            await A().arun(goal="run")


# ---------------------------------------------------------------------------
# Auto-derivation of MCP tool bindings from ctx.tools
# ---------------------------------------------------------------------------


class TestBindingDerivation:
    """``_build_bindings_from_ctx`` exposes every non-builtin tool in
    ``ctx.tools``. The ``expose_tools`` whitelist is applied upstream
    (the dispatcher snapshots ``ctx.tools`` before invoking the worker),
    so this helper just strips builtins."""

    def _build_worker_and_ctx(self, *, tools: Optional[List] = None):
        from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS

        worker = _worker()
        ctx = CognitiveContext(goal="t")
        for spec in tools or []:
            ctx.tools.add(spec)
        for spec in ALL_BUILTIN_TOOLS:
            ctx.tools.add(spec)
        builtin_names = {t.tool_name for t in ALL_BUILTIN_TOOLS}
        return worker, ctx, builtin_names

    def test_default_exposes_only_user_tools(self):
        worker, ctx, builtin_names = self._build_worker_and_ctx(
            tools=[_sample_tool_spec],
        )
        bindings = worker._build_bindings_from_ctx(ctx, builtin_names)
        names = [b.name for b in bindings]
        assert "_sample_tool" in names
        assert not (set(names) & builtin_names)

    def test_binding_carries_tool_schema(self):
        worker, ctx, builtin_names = self._build_worker_and_ctx(
            tools=[_sample_tool_spec],
        )
        bindings = worker._build_bindings_from_ctx(ctx, builtin_names)
        sample = next(b for b in bindings if b.name == "_sample_tool")
        params = sample.parameters
        assert params["type"] == "object"
        assert "query" in params.get("properties", {})
        assert "query" in params.get("required", [])


# ---------------------------------------------------------------------------
# MCPHost._build_handler — signature precision (required vs optional, types)
# ---------------------------------------------------------------------------


class TestMCPHandlerSignature:
    """``_build_handler`` materialises a Python function whose signature
    mirrors the binding's JSON-Schema: required params have no default,
    optional params get a typed default, ``type`` is propagated."""

    def _build(self, parameters):
        import inspect as _inspect

        from bridgic.amphibious._mcp_host import MCPToolBinding, _build_handler

        async def _noop(name, args):  # pragma: no cover — never invoked here
            return None

        binding = MCPToolBinding(
            name="probe", description="probe", parameters=parameters,
        )
        handler = _build_handler(binding, _noop)
        sig = _inspect.signature(handler)
        return handler, sig

    def test_required_param_has_no_default(self):
        _, sig = self._build({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        })
        param = sig.parameters["q"]
        assert param.default is param.empty
        assert param.annotation is str

    def test_optional_param_has_typed_default(self):
        _, sig = self._build({
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": [],
        })
        param = sig.parameters["n"]
        assert param.default == 0
        assert param.annotation is int

    def test_required_ordered_before_optional(self):
        """Python forbids non-default args after default args, so the
        signature must put required first regardless of the JSON-Schema
        property order."""
        _, sig = self._build({
            "type": "object",
            "properties": {
                "opt": {"type": "string"},
                "req": {"type": "string"},
            },
            "required": ["req"],
        })
        param_names = list(sig.parameters.keys())
        assert param_names.index("req") < param_names.index("opt")
