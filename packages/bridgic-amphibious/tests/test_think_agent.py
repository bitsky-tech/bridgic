"""Tests for the ``ThinkAgent`` primitive.

Covers (in order):

* ``think_agent(...)`` factory and ``ThinkAgentDescriptor`` shape.
* Dispatcher branch in ``_dispatch_step`` (scope enforcement, descriptor
  resolution, ``asend()`` of the runtime's return value back into the
  on_agent generator).
* Item-level overlay vs descriptor-level defaults resolution inside
  ``_ThinkAgentRuntime``.
* Auto-derivation of MCP tool bindings from ``ctx.tools`` (excludes
  framework built-ins, applies ``expose_tools`` whitelist).

Integration with a real ``claude`` subprocess is intentionally out of
scope for this file — that path lives in the standalone demos under
``aphiloop_test/``. Here the runtime is mocked so the tests are
deterministic, fast, and runnable in any environment that has the
``fastmcp`` / ``mcp`` deps installed (per the package's pyproject).
"""

from typing import Any, AsyncGenerator, List

import pytest

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    RETURN,
    ThinkAgent,
    ThinkAgentDescriptor,
    ThinkUnit,
    think_agent,
)
from bridgic.amphibious._think_agent import _ThinkAgentRuntime
from bridgic.core.agentic.tool_specs import FunctionToolSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _DummyLLM:
    """No-op LLM stub; satisfies arun()'s AGENT-mode precondition."""

    async def achat(self, messages, **kwargs): ...
    async def astructured_output(self, messages, constraint, **kwargs): ...
    async def astream(self, messages, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...


async def _sample_tool(query: str) -> str:
    """A sample non-builtin tool used to validate auto-derivation."""
    return f"echo:{query}"


_sample_tool_spec = FunctionToolSpec.from_raw(_sample_tool)


# ---------------------------------------------------------------------------
# Factory + descriptor
# ---------------------------------------------------------------------------


class TestThinkAgentFactory:
    """``think_agent(...)`` returns a properly-populated descriptor."""

    def test_factory_returns_descriptor(self):
        d = think_agent()
        assert isinstance(d, ThinkAgentDescriptor)

    def test_default_external_agent_is_claude(self):
        d = think_agent()
        assert d._external_agent == "claude"

    def test_default_builtin_tools_full_set(self):
        d = think_agent()
        assert set(d._allowed_builtin_tools) == {
            "Read", "Write", "Edit", "Bash", "Glob", "Grep"
        }

    def test_explicit_builtin_tools_whitelist(self):
        d = think_agent(allowed_builtin_tools=["Read", "Bash"])
        assert d._allowed_builtin_tools == ["Read", "Bash"]

    def test_explicit_empty_builtin_tools_whitelist(self):
        d = think_agent(allowed_builtin_tools=[])
        assert d._allowed_builtin_tools == []

    def test_unsupported_external_agent_raises(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            think_agent(external_agent="cursor")

    def test_descriptor_get_is_idempotent_on_class_and_instance(self):
        class A(AmphibiousAutoma[CognitiveContext]):
            x = think_agent()

            async def on_agent(self, ctx):
                if False:  # pragma: no cover
                    yield

        assert isinstance(A.x, ThinkAgentDescriptor)
        a = A()
        assert a.x is A.x  # __get__ returns self


# ---------------------------------------------------------------------------
# Dispatcher branch (runtime mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runtime(monkeypatch):
    """Replace ``_ThinkAgentRuntime.run`` with a recording stub.

    Returns the dict that the stub populates on each invocation so tests
    can assert on what the dispatcher passed in.
    """
    captured: dict = {
        "invocations": [],
    }

    async def fake_run(self_runtime, agent, ctx):
        captured["invocations"].append({
            "descriptor": self_runtime.descriptor,
            "goal": self_runtime.goal,
            "permission_mode": self_runtime.permission_mode,
            "allowed_builtin_tools": list(self_runtime.allowed_builtin_tools),
            "expose_tools_filter": (
                list(self_runtime.expose_tools_filter)
                if self_runtime.expose_tools_filter is not None
                else None
            ),
            "agent": agent,
            "ctx": ctx,
        })
        return f"mocked-result-{len(captured['invocations'])}"

    monkeypatch.setattr(_ThinkAgentRuntime, "run", fake_run)
    return captured


class TestThinkAgentDispatch:
    """Verify the dispatcher branch wires runtime correctly."""

    @pytest.mark.asyncio
    async def test_dispatcher_invokes_runtime_and_returns_result(self, mock_runtime):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent()

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                result = yield ThinkAgent("do_thing", goal="hello")
                yield RETURN(result)

        out = await A().arun(llm=_DummyLLM(), goal="run")
        assert mock_runtime["invocations"] == [
            mock_runtime["invocations"][0]
        ]  # exactly one invocation
        assert mock_runtime["invocations"][0]["goal"] == "hello"
        assert out == "mocked-result-1"

    @pytest.mark.asyncio
    async def test_dispatcher_passes_descriptor_to_runtime(self, mock_runtime):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent(permission_mode="acceptEdits")

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="x")

        await A().arun(llm=_DummyLLM(), goal="run")
        descriptor = mock_runtime["invocations"][0]["descriptor"]
        assert isinstance(descriptor, ThinkAgentDescriptor)
        assert descriptor is A.do_thing

    @pytest.mark.asyncio
    async def test_multiple_think_agent_yields_in_one_arun(self, mock_runtime):
        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent()

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="step1")
                yield ThinkAgent("do_thing", goal="step2")
                yield ThinkAgent("do_thing", goal="step3")

        await A().arun(llm=_DummyLLM(), goal="run")
        goals = [inv["goal"] for inv in mock_runtime["invocations"]]
        assert goals == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_think_agent_unknown_name_raises_attribute_error(self, mock_runtime):
        class A(AmphibiousAutoma[CognitiveContext]):
            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("nonexistent")

        with pytest.raises(AttributeError, match="does not match any"):
            await A().arun(llm=_DummyLLM(), goal="run")

    @pytest.mark.asyncio
    async def test_think_agent_outside_on_agent_raises_runtime_error(self, mock_runtime):
        """ThinkAgent yielded from on_workflow should fail scope validation."""

        class A(AmphibiousAutoma[CognitiveContext]):
            do_thing = think_agent()

            async def on_workflow(self, ctx) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="x")

        with pytest.raises(RuntimeError, match="only valid inside"):
            await A().arun(llm=_DummyLLM(), goal="run")


# ---------------------------------------------------------------------------
# Item-level overlay vs descriptor-level defaults
# ---------------------------------------------------------------------------


class TestThinkAgentOverlay:
    """Item fields override descriptor fields; ``None`` falls through."""

    def test_item_permission_mode_beats_descriptor(self):
        d = ThinkAgentDescriptor(permission_mode="bypassPermissions")
        item = ThinkAgent("x", permission_mode="acceptEdits")
        rt = _ThinkAgentRuntime(d, item)
        assert rt.permission_mode == "acceptEdits"

    def test_item_allowed_builtin_tools_beats_descriptor(self):
        d = ThinkAgentDescriptor(allowed_builtin_tools=["Read"])
        item = ThinkAgent("x", allowed_builtin_tools=["Bash"])
        rt = _ThinkAgentRuntime(d, item)
        assert rt.allowed_builtin_tools == ["Bash"]

    def test_item_expose_tools_beats_descriptor(self):
        d = ThinkAgentDescriptor(expose_tools=["a"])
        item = ThinkAgent("x", expose_tools=["b", "c"])
        rt = _ThinkAgentRuntime(d, item)
        assert rt.expose_tools_filter == ["b", "c"]

    def test_descriptor_used_when_item_field_is_none(self):
        d = ThinkAgentDescriptor(
            permission_mode="acceptEdits",
            allowed_builtin_tools=["Read", "Write"],
        )
        item = ThinkAgent("x")
        rt = _ThinkAgentRuntime(d, item)
        assert rt.permission_mode == "acceptEdits"
        assert rt.allowed_builtin_tools == ["Read", "Write"]

    def test_goal_carried_from_item(self):
        d = ThinkAgentDescriptor()
        item = ThinkAgent("x", goal="the goal")
        rt = _ThinkAgentRuntime(d, item)
        # At construction time, ``self.goal`` is the item goal; the
        # ``ctx.goal`` fallback only applies when the explicit goal is
        # absent (and is resolved at ``run()`` time — see _resolve_goal).
        assert rt.goal == "the goal"
        assert rt._explicit_goal == "the goal"

    def test_empty_goal_when_item_goal_is_none(self):
        d = ThinkAgentDescriptor()
        item = ThinkAgent("x")
        rt = _ThinkAgentRuntime(d, item)
        assert rt.goal == ""
        assert rt._explicit_goal is None

    def test_resolve_goal_falls_back_to_ctx_goal_when_not_explicit(self):
        """If the yield omits ``goal``, the runtime should pick up
        ``ctx.goal`` (the AMPHIFLOW step-level fallback prompt path)."""
        d = ThinkAgentDescriptor()
        item = ThinkAgent("x")
        rt = _ThinkAgentRuntime(d, item)
        ctx = CognitiveContext(goal="from-ctx")
        assert rt._resolve_goal(ctx) == "from-ctx"

    def test_resolve_goal_explicit_beats_ctx_goal(self):
        d = ThinkAgentDescriptor()
        item = ThinkAgent("x", goal="explicit")
        rt = _ThinkAgentRuntime(d, item)
        ctx = CognitiveContext(goal="from-ctx")
        assert rt._resolve_goal(ctx) == "explicit"

    def test_resolve_goal_empty_string_when_neither_set(self):
        d = ThinkAgentDescriptor()
        item = ThinkAgent("x")
        rt = _ThinkAgentRuntime(d, item)
        ctx = CognitiveContext()  # no goal
        assert rt._resolve_goal(ctx) == ""


# ---------------------------------------------------------------------------
# Auto-derivation of MCP tool bindings from ctx.tools
# ---------------------------------------------------------------------------


class TestExposeToolsAutoDerivation:
    """``expose_tools=None`` should expose every non-builtin ctx tool;
    a filter list restricts to those names."""

    def _build_runtime_and_ctx(
        self,
        *,
        expose_filter: List[str] | None = None,
        tools: List = None,
    ):
        d = ThinkAgentDescriptor(expose_tools=expose_filter)
        item = ThinkAgent("x")
        rt = _ThinkAgentRuntime(d, item)

        # Build a CognitiveContext with both user tools and builtins. arun()
        # would normally inject builtins for us; here we mimic that subset
        # so the test runs without an arun call.
        from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS

        ctx = CognitiveContext(goal="t")
        for spec in tools or []:
            ctx.tools.add(spec)
        for spec in ALL_BUILTIN_TOOLS:
            ctx.tools.add(spec)
        builtin_names = {t.tool_name for t in ALL_BUILTIN_TOOLS}
        return rt, ctx, builtin_names

    def test_default_exposes_only_user_tools(self):
        rt, ctx, builtin_names = self._build_runtime_and_ctx(
            tools=[_sample_tool_spec],
        )
        bindings = rt._build_bindings_from_ctx(ctx, builtin_names)
        names = [b.name for b in bindings]
        # User tool is exposed.
        assert "_sample_tool" in names
        # No builtin leaked through.
        assert not (set(names) & builtin_names)

    def test_expose_filter_whitelist(self):
        # Two user tools; whitelist only one.
        async def other_tool(x: str) -> str:  # noqa: D401
            """another tool"""
            return x

        other_spec = FunctionToolSpec.from_raw(other_tool)
        rt, ctx, builtin_names = self._build_runtime_and_ctx(
            expose_filter=["other_tool"],
            tools=[_sample_tool_spec, other_spec],
        )
        bindings = rt._build_bindings_from_ctx(ctx, builtin_names)
        names = [b.name for b in bindings]
        assert names == ["other_tool"]

    def test_binding_carries_tool_schema(self):
        rt, ctx, builtin_names = self._build_runtime_and_ctx(
            tools=[_sample_tool_spec],
        )
        bindings = rt._build_bindings_from_ctx(ctx, builtin_names)
        sample = next(b for b in bindings if b.name == "_sample_tool")
        params = sample.parameters
        # JSON Schema shape sourced from the bridgic ToolSpec.
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

    def test_type_mapping_for_primitives(self):
        _, sig = self._build({
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
            },
            "required": ["s", "i", "f", "b"],
        })
        assert sig.parameters["s"].annotation is str
        assert sig.parameters["i"].annotation is int
        assert sig.parameters["f"].annotation is float
        assert sig.parameters["b"].annotation is bool

    def test_unknown_type_falls_back_to_str(self):
        _, sig = self._build({
            "type": "object",
            "properties": {"x": {"type": "unrecognised"}},
            "required": ["x"],
        })
        assert sig.parameters["x"].annotation is str

    def test_empty_schema_produces_zero_arg_handler(self):
        _, sig = self._build({"type": "object", "properties": {}})
        assert len(sig.parameters) == 0

    def test_handler_dispatches_to_callback_with_args_dict(self):
        """Smoke: invoking the generated handler funnels args through
        the ``on_tool_call`` callback as a flat dict keyed by the
        original schema property names."""
        import asyncio
        import inspect as _inspect

        from bridgic.amphibious._mcp_host import MCPToolBinding, _build_handler

        seen: List[dict] = []

        async def _cb(name, args):
            seen.append({"name": name, "args": args})
            return f"got:{args.get('q')}"

        binding = MCPToolBinding(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        handler = _build_handler(binding, _cb)
        result = asyncio.run(handler(q="hello"))
        assert seen == [{"name": "echo", "args": {"q": "hello"}}]
        assert result == "got:hello"
