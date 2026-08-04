"""``edit_file`` built-in tool — exact string replacement.

Mirrors Claude-Code's ``Edit`` tool: requires that ``old_string`` is
unique in the file (or that ``replace_all`` is set), and enforces the
read-before-modify invariant.
"""

import os

from bridgic.core.agentic.tool_specs import FunctionToolSpec

from ._shared import resolve_abs_path, check_read_before_modify, track_read


async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace ``old_string`` with ``new_string`` in ``file_path``.

    By default, ``old_string`` must occur exactly once — otherwise the
    edit is refused so the LLM can supply a more uniquely-identifying
    snippet. Set ``replace_all=True`` to substitute every occurrence
    (e.g. for a rename refactor).

    The read-before-modify invariant is enforced: the file must have
    been read with ``read_file`` first, and must not have changed since.

    Parameters
    ----------
    file_path : str
        Absolute path to the file.
    old_string : str
        The exact substring to replace. Must be unique unless
        ``replace_all`` is True.
    new_string : str
        The replacement substring. May be empty to delete ``old_string``.
    replace_all : bool
        If True, replace every occurrence; otherwise require uniqueness.
    """
    abs_path = resolve_abs_path(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File does not exist: {abs_path}")
    if not os.path.isfile(abs_path):
        raise ValueError(f"Not a regular file: {abs_path}")
    if not old_string:
        raise ValueError("old_string must not be empty.")
    if old_string == new_string:
        raise ValueError("old_string and new_string are identical — nothing to do.")

    check_read_before_modify(abs_path)

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    occurrences = content.count(old_string)
    if occurrences == 0:
        raise ValueError(
            f"old_string not found in {abs_path}. The Read output "
            f"includes line-number prefixes which must NOT be part of "
            f"old_string — only the actual file content."
        )
    if occurrences > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {occurrences} times in {abs_path}. "
            f"Provide a more uniquely-identifying snippet, or set "
            f"replace_all=True."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replaced = occurrences
    else:
        new_content = content.replace(old_string, new_string, 1)
        replaced = 1

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    track_read(abs_path)

    return (
        f"Edited {abs_path}: replaced {replaced} "
        f"occurrence{'s' if replaced != 1 else ''} of old_string."
    )


edit_file_tool: FunctionToolSpec = FunctionToolSpec.from_raw(edit_file)
