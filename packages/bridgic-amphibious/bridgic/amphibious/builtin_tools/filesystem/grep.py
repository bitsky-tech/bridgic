"""``grep`` built-in tool — regex content search across files.

Stateless: pure-Python via the standard ``re`` module so the framework
stays dependency-free. Comparable in feature surface to Claude-Code's
``Grep`` (output_mode, glob filter, head_limit) without trying to be a
ripgrep replacement.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from bridgic.core.agentic.tool_specs import FunctionToolSpec


# Bound how much we'll scan so a careless `.` regex against a large
# tree doesn't hang the agent.
MAX_FILES_SCANNED: int = 5_000
MAX_RESULTS: int = 200


async def grep(
    pattern: str,
    path: str = "",
    glob: str = "",
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
    head_limit: int = 0,
) -> str:
    """Search file contents using a regular expression.

    Parameters
    ----------
    pattern : str
        Regex pattern to search for (Python ``re`` flavour).
    path : str
        Absolute directory to search in. Empty string means the
        process's current working directory.
    glob : str
        Optional glob filter applied to file paths (e.g. ``"**/*.py"``).
        Empty string means scan all regular files recursively.
    output_mode : str
        Output format. One of:
        - ``"files_with_matches"`` (default): one matching path per line.
        - ``"count"``: ``path:N`` where ``N`` is the match count.
        - ``"content"``: ``path:lineno:line`` for every matching line.
    case_insensitive : bool
        If True, match case-insensitively.
    head_limit : int
        Maximum number of result lines to emit. ``0`` means use the
        default cap of 200.
    """
    if not pattern:
        raise ValueError("pattern is required")

    search_root = path or os.getcwd()
    if not os.path.isabs(search_root):
        raise ValueError(f"path must be an absolute directory: {search_root}")
    if not os.path.isdir(search_root):
        raise NotADirectoryError(f"Search path is not a directory: {search_root}")
    if output_mode not in ("files_with_matches", "count", "content"):
        raise ValueError(
            f"Unknown output_mode: {output_mode!r}. "
            f"Use 'files_with_matches', 'count' or 'content'."
        )

    flags = re.MULTILINE | (re.IGNORECASE if case_insensitive else 0)
    regex = re.compile(pattern, flags)  # raises re.error on invalid pattern

    candidates = _iter_files(Path(search_root), glob)
    matches_per_file: Dict[str, List[Tuple[int, str]]] = {}

    for file_path in candidates:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    if regex.search(line):
                        matches_per_file.setdefault(str(file_path), []).append(
                            (lineno, line.rstrip("\n"))
                        )
        except OSError:
            # A single unreadable file (perms, vanished, etc.) shouldn't
            # abort the whole search — skip and continue.
            continue

    if not matches_per_file:
        return "(No matches found.)"

    cap = head_limit if head_limit > 0 else MAX_RESULTS
    return _render(matches_per_file, output_mode, cap)


def _iter_files(root: Path, glob_filter: str):
    """Yield up to ``MAX_FILES_SCANNED`` regular files under ``root``.

    Skips entries inside hidden directories (anything starting with a
    dot, e.g. ``.git``, ``.venv``) so a sloppy regex doesn't drown in
    repository metadata.
    """
    pattern = glob_filter or "**/*"
    count = 0
    for entry in root.glob(pattern):
        if not entry.is_file():
            continue
        try:
            rel = entry.relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield entry
        count += 1
        if count >= MAX_FILES_SCANNED:
            break


def _render(
    matches_per_file: Dict[str, List[Tuple[int, str]]],
    output_mode: str,
    cap: int,
) -> str:
    """Format the match map according to ``output_mode``, truncating to ``cap``."""
    if output_mode == "files_with_matches":
        rows = list(matches_per_file.keys())
    elif output_mode == "count":
        rows = [f"{p}:{len(m)}" for p, m in matches_per_file.items()]
    else:  # "content" — already validated upstream
        rows = [
            f"{p}:{lineno}:{line}"
            for p, hits in matches_per_file.items()
            for lineno, line in hits
        ]

    truncated = rows[:cap]
    if len(rows) > cap:
        truncated.append(
            f"... [{len(rows) - cap} more results truncated; "
            f"narrow the pattern or raise head_limit]"
        )
    return "\n".join(truncated)


grep_tool: FunctionToolSpec = FunctionToolSpec.from_raw(grep)
