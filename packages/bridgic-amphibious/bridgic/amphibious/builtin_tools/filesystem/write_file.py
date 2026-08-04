"""``write_file`` built-in tool.

Creates a new file or overwrites an existing one. Overwriting an
existing file requires a prior ``read_file`` call (and that the file
has not changed since), preventing the LLM from blindly clobbering
content it has not seen.
"""

import os

from bridgic.core.agentic.tool_specs import FunctionToolSpec

from ._shared import resolve_abs_path, check_read_before_modify, track_read


async def write_file(file_path: str, content: str) -> str:
    """Write ``content`` to ``file_path``, creating or overwriting it.

    For existing files, the read-before-modify invariant applies: you
    must first call ``read_file`` on the path. Creating new files has
    no such requirement.

    Parameters
    ----------
    file_path : str
        Absolute path to the file. Parent directories must already exist.
    content : str
        The full content to write. Existing content is replaced.
    """
    abs_path = resolve_abs_path(file_path)

    exists = os.path.exists(abs_path)
    if exists:
        if not os.path.isfile(abs_path):
            raise ValueError(f"Not a regular file: {abs_path}")
        check_read_before_modify(abs_path)

    parent = os.path.dirname(abs_path)
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    track_read(abs_path)

    action = "Overwrote" if exists else "Created"
    byte_count = len(content.encode("utf-8"))
    return f"{action} {abs_path} ({byte_count} bytes)."


write_file_tool: FunctionToolSpec = FunctionToolSpec.from_raw(write_file)
