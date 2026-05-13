"""Bash built-in tool — execute shell commands.

Stateless: does not depend on the running ``AmphibiousAutoma`` instance.
Returns the captured ``stdout`` of the command verbatim, with no
decoration / envelope / tags. Downstream consumers (LLM tool dispatch
or workflow ``yield ActionCall``) get the raw shell output and can
parse it directly.

Failure handling follows the ``subprocess.check_output`` convention:
a non-zero exit code raises ``RuntimeError`` carrying ``stderr`` and
the exit code. The framework (``AmphibiousAutoma._action``) converts
this exception into ``ActionStepResult(success=False, error=...)`` for
either the LLM or the workflow, so callers never have to parse tags
to detect failure.

Output past ``MAX_OUTPUT_BYTES`` is truncated with a clear marker so a
chatty command does not blow up the LLM's context window.
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
    """Execute a shell command and return its captured ``stdout``.

    Runs the command via the user's default shell. On success (exit
    code 0) returns the ``stdout`` exactly as the shell produced it —
    no tags, no envelope, no decoration. Use this for tasks like
    listing files, running tests, building projects, or invoking
    ``git``.

    Parameters
    ----------
    command : str
        The shell command to execute. Multiple commands can be chained
        with ``&&``, ``||`` or ``;``. To merge stderr into stdout
        (e.g. when a tool writes its useful output to stderr), append
        ``2>&1``.
    timeout : int
        Maximum duration in milliseconds before the process is killed.
        Default 120000 (2 minutes); maximum 600000 (10 minutes).
    cwd : str
        Working directory for the command. Empty string means inherit
        the parent process's current working directory.

    Returns
    -------
    str
        Raw ``stdout`` from the command. Output longer than
        ``MAX_OUTPUT_BYTES`` is truncated with a ``[truncated, N more
        chars]`` marker.

    Raises
    ------
    ValueError
        ``command`` is empty or whitespace-only.
    TimeoutError
        The command exceeded ``timeout`` and was killed.
    RuntimeError
        The command exited with a non-zero status. The message
        includes the exit code and any captured ``stderr`` (falling
        back to ``stdout`` when ``stderr`` is empty).
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
        # raise below is what actually surfaces the failure.
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

    if exit_code != 0:
        # subprocess.check_output convention: non-zero exit is an error.
        # The framework wraps the raised exception into an
        # ``ActionStepResult(success=False, error=...)`` for the caller.
        detail = stderr.strip() or stdout.strip() or "(no output captured)"
        raise RuntimeError(
            f"Command failed with exit code {exit_code}: {detail}"
        )

    return stdout


def _decode_truncate(data: bytes) -> str:
    """Decode bytes as UTF-8 (replacing errors) and truncate past MAX_OUTPUT_BYTES."""
    text = data.decode("utf-8", errors="replace")
    if len(text) <= MAX_OUTPUT_BYTES:
        return text
    head = text[:MAX_OUTPUT_BYTES]
    return f"{head}\n... [truncated, {len(text) - MAX_OUTPUT_BYTES} more chars]"


bash_tool: FunctionToolSpec = FunctionToolSpec.from_raw(bash)
