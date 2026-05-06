"""``read_file`` built-in tool.

Returns the file's content with ``cat -n`` style line numbers. Records
the file's mtime on the running agent's read-tracker so ``write_file``
and ``edit_file`` can enforce the read-before-modify invariant.

Errors propagate as exceptions; ``AmphibiousAutoma._action`` converts
them into ``ActionStepResult(success=False, error=...)`` for the LLM.
"""

import os

from bridgic.core.agentic.tool_specs import FunctionToolSpec

from ._shared import resolve_abs_path, track_read


# Mirror Claude-Code's defaults so the LLM behaves consistently.
DEFAULT_MAX_LINES: int = 2000
MAX_LINE_LENGTH: int = 2000
MAX_FILE_BYTES: int = 5 * 1024 * 1024  # 5 MB


async def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 0,
) -> str:
    """Read a file from the local filesystem.

    Returns the content prefixed with line numbers in ``cat -n`` format
    so subsequent ``edit_file`` calls can produce uniquely-locatable
    ``old_string`` snippets. Calling this tool also marks the file as
    "safe to modify" for the read-before-modify invariant.

    Parameters
    ----------
    file_path : str
        Absolute path to the file. Relative paths are rejected.
    offset : int
        1-based line number to start reading from. ``0`` means start at
        the first line.
    limit : int
        Maximum number of lines to return. ``0`` means use the default
        cap of 2000 lines.
    """
    abs_path = resolve_abs_path(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File does not exist: {abs_path}")
    if not os.path.isfile(abs_path):
        raise ValueError(f"Not a regular file: {abs_path}")

    size = os.path.getsize(abs_path)
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"File is too large ({size} bytes); maximum is "
            f"{MAX_FILE_BYTES} bytes."
        )

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    track_read(abs_path)

    if not lines:
        return "(File exists but is empty.)"

    start = max(offset - 1, 0) if offset > 0 else 0
    if start >= len(lines):
        return f"(Offset {offset} is past the end of the file [{len(lines)} lines].)"
    span = limit if limit > 0 else DEFAULT_MAX_LINES
    end = min(start + span, len(lines))

    rendered = []
    for idx, line in enumerate(lines[start:end], start=start + 1):
        if line.endswith("\n"):
            line = line[:-1]
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "...[line truncated]"
        rendered.append(f"{idx:6d}\t{line}")

    if end < len(lines):
        rendered.append(
            f"... [{len(lines) - end} more lines; pass offset/limit to read further]"
        )

    return "\n".join(rendered)


read_file_tool: FunctionToolSpec = FunctionToolSpec.from_raw(read_file)
