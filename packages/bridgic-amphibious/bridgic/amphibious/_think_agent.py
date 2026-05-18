"""ThinkAgent — declarative external-agent think primitive.

Layout (mirrors ``_think_unit.py``): ``ThinkAgentDescriptor`` + factory
``think_agent(...)`` + ``_ThinkAgentRuntime``. The runtime materialises a
per-call workdir, snapshots ctx + trace to files, boots an in-process
FastMCP server exposing project tools + an ``agent_done`` signal, spawns
``claude -p`` as a subprocess wired to that server, and awaits
completion.

``fastmcp`` / ``uvicorn`` are imported lazily — users who never use
``ThinkAgent`` pay zero install / import cost. Requires the ``claude``
CLI on PATH.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from bridgic.amphibious._run_dir import (
    ensure_run_dir,
    make_run_id,
    next_delegate_subdir,
)
from bridgic.amphibious._type import (
    ActionCall,
    ActionResult,
    Step,
    ThinkAgent,
)

if TYPE_CHECKING:
    from bridgic.amphibious._amphibious_automa import AmphibiousAutoma
    from bridgic.amphibious._context import CognitiveContext


# ----------------------------------------------------------------------
# Descriptor + factory
# ----------------------------------------------------------------------


class ThinkAgentDescriptor:
    """Class-level marker for a declared ``think_agent``.

    Invocation goes through ``yield ThinkAgent("name", ...)`` inside
    ``on_agent``; the dispatcher resolves the name, picks up the
    descriptor, and hands it to ``_ThinkAgentRuntime``.
    """

    DEFAULT_BUILTIN_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")

    def __init__(
        self,
        *,
        external_agent: str = "claude",
        allowed_builtin_tools: Optional[List[str]] = None,
        permission_mode: str = "bypassPermissions",
        expose_tools: Optional[List[str]] = None,
        completion_timeout: float = 180.0,
        claude_bin: str = "claude",
        workdir: Optional[Path] = None,
        mcp_server_name: str = "amphi-bridge",
    ) -> None:
        if external_agent != "claude":
            raise NotImplementedError(
                f"external_agent={external_agent!r} is not yet supported. "
                "Only 'claude' is wired up in this release."
            )
        self._external_agent = external_agent
        self._allowed_builtin_tools = (
            list(allowed_builtin_tools)
            if allowed_builtin_tools is not None
            else list(self.DEFAULT_BUILTIN_TOOLS)
        )
        self._permission_mode = permission_mode
        self._expose_tools = expose_tools
        self._completion_timeout = completion_timeout
        self._claude_bin = claude_bin
        self._workdir = workdir
        self._mcp_server_name = mcp_server_name

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "ThinkAgentDescriptor":
        return self


def think_agent(
    *,
    external_agent: str = "claude",
    allowed_builtin_tools: Optional[List[str]] = None,
    permission_mode: str = "bypassPermissions",
    expose_tools: Optional[List[str]] = None,
    completion_timeout: float = 180.0,
    claude_bin: str = "claude",
    workdir: Optional[Path] = None,
    mcp_server_name: str = "amphi-bridge",
) -> ThinkAgentDescriptor:
    """Declare a think-agent unit, invoked via ``yield ThinkAgent(name, ...)``.

    Mirrors ``think_unit(...)`` but binds an external agent runtime
    instead of a ``CognitiveWorker`` / ``WorkerRunner``. Today only
    ``external_agent="claude"`` is wired up.

    ``allowed_builtin_tools`` is the claude built-in whitelist (defaults
    to all six: Read/Write/Edit/Bash/Glob/Grep). ``expose_tools``
    whitelists which project tools to expose via MCP (``None`` exposes
    every non-builtin tool from ``ctx.tools``; builtin tools are never
    re-exposed — claude has its own). ``workdir=None`` defaults to
    ``cwd / ".bridgic"``. ``mcp_server_name`` shows up in claude's tool
    names as ``mcp__<server>__<tool>``.

    >>> class MyAutoma(AmphibiousAutoma[MyContext]):
    ...     write_article = think_agent(
    ...         allowed_builtin_tools=["Read", "Write", "Edit", "Bash"],
    ...     )
    ...     async def on_agent(self, ctx):
    ...         result = yield ThinkAgent("write_article", goal="...")
    ...         yield RETURN(result)
    """
    return ThinkAgentDescriptor(
        external_agent=external_agent,
        allowed_builtin_tools=allowed_builtin_tools,
        permission_mode=permission_mode,
        expose_tools=expose_tools,
        completion_timeout=completion_timeout,
        claude_bin=claude_bin,
        workdir=workdir,
        mcp_server_name=mcp_server_name,
    )


# ----------------------------------------------------------------------
# Runtime — does the actual delegation work
# ----------------------------------------------------------------------


class _ThinkAgentRuntime:
    """Per-invocation runtime; instantiated by ``_dispatch_step`` and
    used once via ``await runtime.run(agent, ctx)``.
    """

    def __init__(
        self,
        descriptor: ThinkAgentDescriptor,
        item: ThinkAgent,
    ) -> None:
        self.descriptor = descriptor
        # Resolve overlays (item-level beats descriptor-level beats default)
        self.allowed_builtin_tools: List[str] = (
            list(item.allowed_builtin_tools)
            if item.allowed_builtin_tools is not None
            else list(descriptor._allowed_builtin_tools)
        )
        self.permission_mode: str = (
            item.permission_mode if item.permission_mode is not None
            else descriptor._permission_mode
        )
        self.expose_tools_filter: Optional[List[str]] = (
            list(item.expose_tools) if item.expose_tools is not None
            else (list(descriptor._expose_tools) if descriptor._expose_tools is not None else None)
        )
        # Goal is resolved late (in ``run``) so we can fall back to
        # ``ctx.goal`` — important under AMPHIFLOW step-level fallback,
        # where the framework synthesises a goal-shaped ctx.goal carrying
        # the failure context (step intent / failed call / error /
        # ``resolve_step_fallback`` instructions). Letting the goal
        # default to ctx.goal in that scope means
        # ``yield ThinkAgent("fixer")`` is enough.
        self._explicit_goal: Optional[str] = item.goal
        self.goal: str = item.goal or ""
        self.completion_timeout: float = descriptor._completion_timeout
        self.claude_bin: str = descriptor._claude_bin
        self.mcp_server_name: str = descriptor._mcp_server_name
        self.workdir_base: Optional[Path] = descriptor._workdir

    async def run(
        self,
        agent: "AmphibiousAutoma",
        ctx: "CognitiveContext",
    ) -> Optional[str]:
        """Execute the delegate. Returns the agent_done result string."""
        # Lazy imports — pulled in only when ThinkAgent actually fires.
        from bridgic.amphibious._mcp_host import MCPHost, MCPToolBinding
        from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS

        builtin_names = {t.tool_name for t in ALL_BUILTIN_TOOLS}

        # Resolve the effective goal: explicit > ctx.goal > "".
        self.goal = self._resolve_goal(ctx)

        # 1. Workdir layout — prefer the agent's active arun-level run
        # directory so all delegates from one arun share a parent dir.
        workdir = self._resolve_delegate_workdir(agent)

        # 2. Snapshot ctx + trace prefix to files
        (workdir / "ctx_snapshot.json").write_text(
            self._serialize_ctx(ctx),
            encoding="utf-8",
        )
        (workdir / "trace_prefix.json").write_text(
            self._serialize_trace(getattr(agent, "_agent_trace", None)),
            encoding="utf-8",
        )
        (workdir / "goal.txt").write_text(self.goal, encoding="utf-8")

        # 3. Derive MCP bindings from ctx.tools (auto-exclude builtins)
        bindings = self._build_bindings_from_ctx(ctx, builtin_names)

        # 4. Build the MCP host
        agent_done_future: asyncio.Future[str] = (
            asyncio.get_event_loop().create_future()
        )

        async def on_tool_call(tool_name: str, args: Dict[str, Any]) -> Any:
            return await self._dispatch_project_tool(agent, ctx, tool_name, args)

        def on_agent_done(result: str) -> None:
            if not agent_done_future.done():
                agent_done_future.set_result(result)

        host = MCPHost(
            server_name=self.mcp_server_name,
            bindings=bindings,
            on_tool_call=on_tool_call,
            on_agent_done=on_agent_done,
        )
        await host.start()

        try:
            mcp_config_path = workdir / "mcp_config.json"
            mcp_config_path.write_text(
                json.dumps({
                    "mcpServers": {
                        host.server_name: {"type": "http", "url": host.url},
                    },
                }, indent=2),
                encoding="utf-8",
            )

            # 5. Spawn claude
            allowed_mcp_tools = [
                f"mcp__{host.server_name}__{b.name}" for b in bindings
            ] + [f"mcp__{host.server_name}__agent_done"]
            allowed_tools_arg = ",".join(
                self.allowed_builtin_tools + allowed_mcp_tools
            )

            cmd = [
                self.claude_bin, "-p",
                "--mcp-config", str(mcp_config_path),
                "--strict-mcp-config",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--permission-mode", self.permission_mode,
                "--allowedTools", allowed_tools_arg,
                "--no-session-persistence",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
            )

            assert proc.stdin is not None
            initial_message = self._build_initial_prompt(
                workdir=workdir,
                bindings=bindings,
                server_name=host.server_name,
            )
            proc.stdin.write(
                (json.dumps({
                    "type": "user",
                    "message": {"role": "user", "content": initial_message},
                }) + "\n").encode("utf-8")
            )
            await proc.stdin.drain()

            # 6. Concurrent log capture
            stdout_task = asyncio.create_task(_drain_stream(proc.stdout))
            stderr_task = asyncio.create_task(_drain_stream(proc.stderr))

            # 7. Wait for completion: agent_done OR claude exit OR timeout
            proc_wait_task = asyncio.create_task(proc.wait())
            result_value: Optional[str] = None
            try:
                await asyncio.wait(
                    [agent_done_future, proc_wait_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.completion_timeout,
                )
                if agent_done_future.done():
                    result_value = agent_done_future.result()
            finally:
                if proc.returncode is None:
                    try:
                        proc.terminate()
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()

                (workdir / "claude_stdout.jsonl").write_text(
                    await stdout_task, encoding="utf-8",
                )
                (workdir / "claude_stderr.log").write_text(
                    await stderr_task, encoding="utf-8",
                )
        finally:
            await host.stop()

        # 8. Record summary into ctx.cognitive_history
        ctx.add_info(Step(
            content=f"[think_agent={self.descriptor._external_agent}] goal complete",
            result=result_value,
            metadata={
                "external_agent": self.descriptor._external_agent,
                "workdir": str(workdir),
            },
            status=True,
        ))
        return result_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_goal(self, ctx: "CognitiveContext") -> str:
        """Resolve the effective goal for this delegate.

        Resolution order:

        1. ``item.goal`` if the yield supplied one explicitly.
        2. ``ctx.goal`` — important under AMPHIFLOW step-level fallback,
           where the framework synthesises a recovery-shaped goal carrying
           the failed-step context. Letting the goal default to ctx.goal
           lets users write ``yield ThinkAgent("fixer")`` without
           threading the failure prompt through manually.
        3. Empty string as a last resort (so prompts stay well-formed).
        """
        if self._explicit_goal is not None:
            return self._explicit_goal
        return getattr(ctx, "goal", None) or ""

    def _resolve_delegate_workdir(self, agent: "AmphibiousAutoma") -> Path:
        """Pick the delegate's per-invocation directory.

        Resolution order:

        1. **arun-level run dir** — if the parent automa was invoked
           with ``arun(workdir=...)`` the framework set
           ``agent._current_run_dir``; place the delegate under
           ``<run_dir>/delegates/<n>/``. This is the recommended path
           and keeps every artifact from one arun under one parent.
        2. **descriptor-level workdir** — for users who didn't pass
           ``workdir`` to ``arun`` but did set one on the
           ``think_agent(...)`` descriptor; behaves like a private mini
           run directory.
        3. **cwd fallback** — ``Path.cwd() / ".bridgic"``; same shape as
           path (2) but defaulted.
        """
        agent_run_dir = getattr(agent, "_current_run_dir", None)
        if agent_run_dir is not None:
            return next_delegate_subdir(agent_run_dir)

        base = (
            self.workdir_base
            if self.workdir_base is not None
            else Path.cwd() / ".bridgic"
        )
        run_dir = ensure_run_dir(base, _ensure_process_run_id())
        return next_delegate_subdir(run_dir)

    @staticmethod
    def _serialize_ctx(ctx: Any) -> str:
        try:
            tools = [getattr(t, "tool_name", None) for t in ctx.tools.get_all()]
        except Exception:
            tools = []
        try:
            history = [
                {
                    "content": s.content,
                    "result": str(s.result) if s.result is not None else None,
                }
                for s in ctx.cognitive_history.get_all()
            ]
        except Exception:
            history = []
        return json.dumps(
            {
                "goal": getattr(ctx, "goal", None),
                "tools": tools,
                "history": history,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _serialize_trace(trace: Any) -> str:
        if trace is None:
            return json.dumps({"steps": [], "metadata": {}}, indent=2)
        try:
            data = trace.build(metadata={})
            if hasattr(trace, "_to_serializable"):
                data = trace._to_serializable(data)
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    def _build_bindings_from_ctx(
        self,
        ctx: Any,
        builtin_names: set,
    ) -> List[Any]:
        from bridgic.amphibious._mcp_host import MCPToolBinding

        bindings: List[MCPToolBinding] = []
        filter_set = (
            set(self.expose_tools_filter)
            if self.expose_tools_filter is not None
            else None
        )
        for tool in ctx.tools.get_all():
            name = tool.tool_name
            if name in builtin_names:
                continue
            if filter_set is not None and name not in filter_set:
                continue
            bindings.append(MCPToolBinding(
                name=name,
                description=tool.tool_description or name,
                parameters=tool.tool_parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ))
        return bindings

    def _build_initial_prompt(
        self,
        *,
        workdir: Path,
        bindings: List[Any],
        server_name: str,
    ) -> str:
        tool_lines = "\n".join(
            f"  - mcp__{server_name}__{b.name}: {b.description.splitlines()[0] if b.description else ''}"
            for b in bindings
        )
        done_tool = f"mcp__{server_name}__agent_done"
        return (
            f"You are running as the external agent layer of an "
            f"AmphibiousAutoma. Your goal:\n\n"
            f"{self.goal}\n\n"
            f"CONTEXT FILES (read on demand):\n"
            f"  - ctx_snapshot: {workdir / 'ctx_snapshot.json'} (parent automa's ctx state)\n"
            f"  - trace_prefix: {workdir / 'trace_prefix.json'} (trace steps before you were called)\n\n"
            f"PROJECT TOOLS (call via MCP — results flow back into the parent's hook pipeline):\n"
            f"{tool_lines if tool_lines else '  (none exposed for this delegate)'}\n\n"
            f"COMPLETION CONTRACT:\n"
            f"  When the goal is fully complete, call {done_tool} with `result` set to "
            f"the final answer / summary string. After that you may finish; the parent "
            f"automa will resume with the value you passed."
        )

    async def _dispatch_project_tool(
        self,
        agent: "AmphibiousAutoma",
        ctx: "CognitiveContext",
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Route a claude-originated tool call through agent._action().

        This is the architectural payoff: every external-agent tool call
        is processed by the same pipeline that workflow-side ``yield
        ActionCall`` uses, so observation / before_action / after_action
        hooks all fire and the call lands in ``ctx.cognitive_history``
        automatically.
        """
        action_call = ActionCall(
            tool_name=tool_name,
            description=f"[think_agent] {tool_name}",
            **args,
        )
        # Mirror the workflow-scope ActionCall semantics in _dispatch_step:
        # run observation, then _action (which itself runs before/after_action).
        try:
            obs = await agent._invoke_template(agent.observation(ctx), ctx)
            if obs is not None:
                ctx.observation = obs
        except Exception:
            pass

        step = await agent._action(action_call.decision, ctx, _worker=None)
        return _extract_tool_result(step)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


_PROCESS_RUN_ID: Optional[str] = None


def _ensure_process_run_id() -> str:
    """Stable per-process run id used when ``arun(workdir=...)`` is not set.

    When the parent automa is invoked with ``arun(workdir=...)`` the
    framework's per-arun run-id (under ``agent._current_run_dir``) is
    used instead. This module-level id is only the fallback when the
    user goes through ``think_agent(workdir=...)`` directly without an
    arun-level workdir — useful for ad-hoc / test scaffolds.
    """
    global _PROCESS_RUN_ID
    if _PROCESS_RUN_ID is None:
        _PROCESS_RUN_ID = make_run_id()
    return _PROCESS_RUN_ID


async def _drain_stream(stream: Optional[asyncio.StreamReader]) -> str:
    if stream is None:
        return ""
    chunks: List[str] = []
    while True:
        line = await stream.readline()
        if not line:
            break
        chunks.append(line.decode("utf-8", errors="replace"))
    return "".join(chunks)


def _extract_tool_result(step: Optional[Step]) -> Any:
    if step is None:
        return None
    inner = step.result
    if isinstance(inner, ActionResult) and inner.results:
        first = inner.results[0]
        return first.tool_result
    return inner


__all__ = [
    "ThinkAgentDescriptor",
    "think_agent",
]
