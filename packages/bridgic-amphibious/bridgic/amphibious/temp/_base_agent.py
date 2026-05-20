"""
BaseAgent — the external coding-agent abstraction.

This is the **BASE** (anchor type) of ``AgentWorker``, mirroring the role
``BaseLlm`` plays for ``CognitiveWorker``:

* ``CognitiveWorker`` holds a ``BaseLlm`` and only ever calls its
  interface (``astructured_output`` …); it never embeds provider
  internals.
* ``AgentWorker`` holds a ``BaseAgent`` and only ever calls
  ``agent.run(request)``; it never embeds CLI internals.

``BaseAgent``'s purpose is narrower than ``BaseLlm``'s, though.
``bridgic-llms`` is a *full* LLM abstraction (protocol-based, carries
every upper-layer usage shape). ``BaseAgent`` is a leaner **driver /
adapter**: its whole job is to integrate one external coding-agent CLI
(claude code / codex / openclaw / …) and be reusable from inside the
framework. A subclass owns exactly two things — how to spawn its CLI,
and how to detect completion + extract the result.

Layout:

* ``AgentRequest`` — everything one delegation needs, assembled by
  ``AgentWorker`` and consumed by ``BaseAgent.run``.
* ``AgentResult`` — the outcome of one ``run``.
* ``BaseAgent`` — abstract base; one required override (``run``), one
  provided helper (``_run_subprocess``).
* ``ClaudeCodeAgent`` — the concrete ``claude code`` implementation
  shipped with the framework.

``fastmcp`` / ``uvicorn`` are not imported here — ``BaseAgent`` only
deals with the subprocess; the in-process MCP host lives on the
``AgentWorker`` side.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


################################################################################################################
# Request / Result value objects
################################################################################################################


@dataclass
class AgentRequest:
    """One external-agent delegation, fully specified.

    Assembled by ``AgentWorker`` (which organizes the context) and
    consumed by ``BaseAgent.run``. Mirrors the ``messages`` +
    ``constraint`` pair that ``CognitiveWorker`` hands to
    ``BaseLlm.astructured_output`` — the worker prepares it, the base
    type executes it.

    Fields
    ------
    message : str
        The full prompt / task handed to the external agent.
    cwd : Path
        Working directory for the CLI subprocess (an ephemeral tempdir
        materialised by ``AgentWorker``).
    mcp_servers : Dict[str, Dict[str, Any]]
        MCP servers to wire into the CLI, ``{server_name: {"type":
        "http", "url": ...}}``. The concrete agent decides how to feed
        this to its CLI (claude writes an ``mcp_config.json``).
    allowed_tools : List[str]
        The MCP-bridged tool names (already ``mcp__<server>__<tool>``
        prefixed, plus the ``agent_done`` signal) the agent should be
        permitted to call. The concrete agent merges these with its own
        built-in tool allow-list.
    done_signal : Optional[asyncio.Future[str]]
        Resolves with the result string when the agent calls the MCP
        ``agent_done`` tool. ``BaseAgent.run`` races it against process
        exit to decide completion.

    Note there is no ``goal`` / ``skills`` / ``history`` field — those
    are context data; ``AgentWorker`` bakes them into ``message`` (via
    its ``thinking()`` template). ``AgentRequest`` only carries the
    non-context runtime wiring (cwd / mcp servers / allow-list /
    completion future).
    """

    message: str
    cwd: Path
    mcp_servers: Dict[str, Dict[str, Any]]
    allowed_tools: List[str] = field(default_factory=list)
    done_signal: Optional["asyncio.Future[str]"] = None


@dataclass
class AgentResult:
    """Outcome of one ``BaseAgent.run``.

    Fields
    ------
    output : Optional[str]
        The final answer — the string the agent passed to
        ``agent_done(result=...)``, or ``None`` if it exited without
        signalling.
    exit_code : Optional[int]
        The CLI subprocess's return code (``None`` if force-killed
        before one was set).
    completion : str
        How the run ended — one of ``"agent_done"`` (clean MCP signal),
        ``"process_exit"`` (CLI exited on its own), ``"timeout"``.
    """

    output: Optional[str]
    exit_code: Optional[int]
    completion: str


################################################################################################################
# BaseAgent — abstract base
################################################################################################################


class BaseAgent:
    """Abstract driver for one external coding-agent CLI.

    Subclass per CLI (``ClaudeCodeAgent``, ``CodexAgent``, …). The only
    required override is :meth:`run`; :meth:`_run_subprocess` is a
    provided helper that handles the generic spawn / drain / wait
    mechanics so subclasses focus on argv assembly + result extraction.

    Constructor params carry the agent's *static identity* (which
    binary, default model, default permission policy). The *per-call*
    inputs (cwd, mcp servers, message, …) arrive through
    :class:`AgentRequest`.
    """

    async def run(self, request: AgentRequest) -> AgentResult:
        """Drive the CLI for one delegation and return its result.

        Subclasses MUST override. A typical implementation:

        1. Translate ``request.mcp_servers`` into whatever the CLI
           expects (claude writes an ``mcp_config.json`` into
           ``request.cwd``).
        2. Assemble argv + the initial stdin payload.
        3. ``await self._run_subprocess(argv, ...)``.
        4. Wrap the outcome in an :class:`AgentResult`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run(request) -> AgentResult."
        )

    # ------------------------------------------------------------------
    # Provided helper — generic subprocess orchestration
    # ------------------------------------------------------------------

    async def _run_subprocess(
        self,
        argv: List[str],
        *,
        stdin_payload: Optional[bytes] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 180.0,
        done_signal: Optional["asyncio.Future[str]"] = None,
    ) -> Tuple[Optional[str], Optional[int], str]:
        """Spawn the CLI subprocess; wait for completion.

        Waits for the FIRST of:

        * ``done_signal`` resolving (the CLI called the MCP
          ``agent_done`` tool).
        * The subprocess exiting on its own.
        * ``timeout`` elapsing.

        Returns ``(output, exit_code, completion)`` — see
        :class:`AgentResult` for the field meanings.

        stdout / stderr are drained to keep the subprocess from blocking
        on a full pipe; their content is intentionally discarded —
        anything the agent wants to surface to the parent must go
        through the MCP bridge (``agent_done`` or a bridged tool call),
        keeping ``AgentTrace`` the single source of truth.
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )

        # Push the initial prompt payload (if any) onto stdin.
        if stdin_payload is not None and proc.stdin is not None:
            proc.stdin.write(stdin_payload)
            await proc.stdin.drain()

        # Drain stdout / stderr concurrently — content discarded.
        stdout_task = asyncio.create_task(_consume_stream(proc.stdout))
        stderr_task = asyncio.create_task(_consume_stream(proc.stderr))
        proc_wait_task = asyncio.create_task(proc.wait())

        # Race: done_signal vs process exit (vs timeout).
        wait_for = [proc_wait_task]
        if done_signal is not None:
            wait_for.append(done_signal)

        completion: str = "process_exit"
        output: Optional[str] = None
        try:
            await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
            if done_signal is not None and done_signal.done():
                output = done_signal.result()
                completion = "agent_done"
            elif proc_wait_task.done():
                completion = "process_exit"
            else:
                completion = "timeout"
        finally:
            if proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            exit_code = proc.returncode
            await stdout_task
            await stderr_task

        return output, exit_code, completion


################################################################################################################
# ClaudeCodeAgent — concrete ``claude code`` driver
################################################################################################################


class ClaudeCodeAgent(BaseAgent):
    """``claude -p`` driver — the concrete ``BaseAgent`` shipped with the
    framework.

    Sends the prompt as a stream-json ``user`` message on stdin. The
    tool allow-list combines the agent's own built-ins (Read / Write /
    Edit / Bash / Glob / Grep by default) with the MCP-bridged tools
    from ``request.allowed_tools``.

    Constructor params are claude-specific static config:

    * ``bin`` — the ``claude`` binary (path or name on ``PATH``).
    * ``allowed_builtin_tools`` — claude's own built-in tools to permit.
    * ``permission_mode`` — claude's ``--permission-mode``.
    * ``completion_timeout`` — seconds to wait before force-terminating.

    >>> agent = ClaudeCodeAgent(allowed_builtin_tools=["Read", "Grep"])
    >>> reviewer = think_agent(AgentWorker(agent))
    """

    DEFAULT_BUILTIN_TOOLS: Tuple[str, ...] = (
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    )

    def __init__(
        self,
        *,
        bin: str = "claude",
        allowed_builtin_tools: Optional[List[str]] = None,
        permission_mode: str = "bypassPermissions",
        completion_timeout: float = 180.0,
    ) -> None:
        self.bin = bin
        self.allowed_builtin_tools: List[str] = (
            list(allowed_builtin_tools)
            if allowed_builtin_tools is not None
            else list(self.DEFAULT_BUILTIN_TOOLS)
        )
        self.permission_mode = permission_mode
        self.completion_timeout = completion_timeout

    async def run(self, request: AgentRequest) -> AgentResult:
        """Build the ``claude -p`` invocation and drive it."""
        ########################
        # 1. Materialise the MCP config claude needs on disk
        ########################
        mcp_config_path = self._write_mcp_config(request.cwd, request.mcp_servers)

        ########################
        # 2. Assemble argv
        ########################
        allowed = ",".join(
            list(self.allowed_builtin_tools) + list(request.allowed_tools)
        )
        argv = [
            self.bin, "-p",
            "--mcp-config", str(mcp_config_path),
            "--strict-mcp-config",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", self.permission_mode,
            "--allowedTools", allowed,
            "--no-session-persistence",
        ]

        ########################
        # 3. Assemble the stdin payload (stream-json user message)
        ########################
        payload = (
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": request.message},
            }) + "\n"
        ).encode("utf-8")

        ########################
        # 4. Spawn + wait
        ########################
        output, exit_code, completion = await self._run_subprocess(
            argv,
            stdin_payload=payload,
            cwd=request.cwd,
            timeout=self.completion_timeout,
            done_signal=request.done_signal,
        )
        return AgentResult(
            output=output, exit_code=exit_code, completion=completion,
        )

    @staticmethod
    def _write_mcp_config(
        cwd: Path, mcp_servers: Dict[str, Dict[str, Any]],
    ) -> Path:
        """Write claude's ``mcp_config.json`` into ``cwd`` and return its path."""
        mcp_config_path = cwd / "mcp_config.json"
        mcp_config_path.write_text(
            json.dumps({"mcpServers": dict(mcp_servers)}, indent=2),
            encoding="utf-8",
        )
        return mcp_config_path


################################################################################################################
# Module-level helpers
################################################################################################################


async def _consume_stream(stream: Optional["asyncio.StreamReader"]) -> None:
    """Read a subprocess stream to EOF, discarding the content.

    Keeps the subprocess from blocking on a full pipe without
    accumulating anything in memory.
    """
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            return


__all__ = [
    "AgentRequest",
    "AgentResult",
    "BaseAgent",
    "ClaudeCodeAgent",
]
