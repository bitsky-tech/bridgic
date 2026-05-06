"""Shared helpers for the filesystem built-in tools.

The read-tracker invariant
--------------------------
``read_file`` records the file's mtime on the running agent's
``_read_tracker`` dict. ``write_file`` (for existing files) and
``edit_file`` then verify the file was read first AND has not been
modified externally since that read. This mirrors Claude-Code's
``Read``-before-``Edit``/``Write`` safety check.

The shared agent reference comes from the ``current_agent`` ContextVar
set by ``AmphibiousAutoma.arun()``. If a tool somehow runs outside an
agent context (e.g. a unit test calling the function directly without
an agent), the tracker calls become no-ops so the function still works
in isolation.

Failure mode: helpers raise on validation errors; ``AmphibiousAutoma``'s
``_action`` path catches per-tool exceptions and surfaces them to the
LLM via ``ActionStepResult(success=False, error=str(exc))``.
"""

import os

from bridgic.amphibious.builtin_tools._agent_state import current_agent


def resolve_abs_path(file_path: str) -> str:
    """Validate and absolutize a file path.

    Raises
    ------
    ValueError
        If ``file_path`` is empty or not an absolute path.
    """
    if not file_path or not file_path.strip():
        raise ValueError("file_path is required")
    if not os.path.isabs(file_path):
        raise ValueError(f"file_path must be an absolute path: {file_path}")
    return os.path.abspath(file_path)


def track_read(abs_path: str) -> None:
    """Record the file's current mtime on the running agent's tracker.

    Best-effort: silently swallows ``OSError`` (e.g. file vanished
    between read and stat) so that a successful read result is not
    masked by a flaky stat. A subsequent ``edit_file``/``write_file``
    call would then re-issue the "must read first" error, which is the
    correct outcome — we cannot reason about staleness without an mtime.
    """
    agent = current_agent.get(None)
    if agent is None:
        return
    tracker = getattr(agent, "_read_tracker", None)
    if tracker is None:
        return
    try:
        tracker[abs_path] = os.stat(abs_path).st_mtime
    except OSError:
        pass


def check_read_before_modify(abs_path: str) -> None:
    """Verify the file was read first AND has not changed since the read.

    No-op when no agent is bound (running outside ``arun()``) — keeps
    the tools callable in plain unit-test contexts while still
    enforcing the invariant during real agent runs.

    Raises
    ------
    RuntimeError
        If the file has not been read in this agent run, or if the
        file's mtime has advanced since the last read.
    """
    agent = current_agent.get(None)
    if agent is None:
        return
    tracker = getattr(agent, "_read_tracker", None)
    if tracker is None:
        return
    if abs_path not in tracker:
        raise RuntimeError(
            f"You must use the read_file tool to read {abs_path} "
            f"at least once in the conversation before modifying it."
        )
    current_mtime = os.stat(abs_path).st_mtime
    if current_mtime > tracker[abs_path]:
        raise RuntimeError(
            f"File has been modified externally since the last "
            f"read_file call: {abs_path}. Re-read it before modifying."
        )
