"""Bash built-in tool — execute shell commands.

Stateless: does not depend on the running ``AmphibiousAutoma`` instance.
Captures stdout, stderr and exit code. Output is truncated past
``MAX_OUTPUT_BYTES`` to keep tool results LLM-friendly.

Errors propagate as exceptions; ``AmphibiousAutoma._action`` converts
them into ``ActionStepResult(success=False, error=...)`` for the LLM.
A non-zero exit code is NOT an exception — the LLM receives the full
stdout/stderr/exit_code envelope and decides what to do.
"""

import asyncio

from bridgic.core.agentic.tool_specs import FunctionToolSpec


# Default 2 minutes; max 10 minutes — matches Claude Code's terminal tool.
DEFAULT_TIMEOUT_MS: int = 120_000
MAX_TIMEOUT_MS: int = 600_000

# Truncate captured output past this length so a chatty command does not
# blow up the LLM's context window.
MAX_OUTPUT_BYTES: int = 30_000


async def bash(
    command: str,
    timeout: int = DEFAULT_TIMEOUT_MS,
    cwd: str = "",
) -> str:
    """Execute a shell command and return its captured output.

    Runs the command via the user's default shell. Captures stdout,
    stderr and the exit code, then formats them into a single string
    response. Use this for tasks like listing files, running tests,
    building projects or invoking ``git``.

    Parameters
    ----------
    command : str
        The shell command to execute. Multiple commands can be chained
        with ``&&``, ``||`` or ``;``.
    timeout : int
        Maximum duration in milliseconds before the process is killed.
        Default 120000 (2 minutes); maximum 600000 (10 minutes).
    cwd : str
        Working directory for the command. Empty string means inherit
        the parent process's current working directory.
    """
    if not command or not command.strip():
        raise ValueError("command is required")

    timeout_ms = max(1, min(int(timeout), MAX_TIMEOUT_MS))
    timeout_s = timeout_ms / 1000.0

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd or None,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError as exc:
        # Best-effort cleanup so we don't leak the killed process. The
        # raise below is what actually surfaces the failure to the LLM.
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"Command timed out after {timeout_ms}ms and was killed."
        ) from exc

    stdout = _decode_truncate(stdout_bytes)
    stderr = _decode_truncate(stderr_bytes)
    exit_code = proc.returncode if proc.returncode is not None else -1

    parts = []
    if stdout:
        parts.append(f"<stdout>\n{stdout}\n</stdout>")
    if stderr:
        parts.append(f"<stderr>\n{stderr}\n</stderr>")
    parts.append(f"<exit_code>{exit_code}</exit_code>")
    return "\n".join(parts)


def _decode_truncate(data: bytes) -> str:
    """Decode bytes as UTF-8 (replacing errors) and truncate past MAX_OUTPUT_BYTES."""
    text = data.decode("utf-8", errors="replace")
    if len(text) <= MAX_OUTPUT_BYTES:
        return text
    head = text[:MAX_OUTPUT_BYTES]
    return f"{head}\n... [truncated, {len(text) - MAX_OUTPUT_BYTES} more chars]"


bash_tool: FunctionToolSpec = FunctionToolSpec.from_raw(bash)
