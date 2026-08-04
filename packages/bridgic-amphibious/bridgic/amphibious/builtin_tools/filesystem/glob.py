"""``glob`` built-in tool — fast file pattern matching.

Stateless: returns paths matching a glob pattern, sorted by mtime
descending so the freshest matches surface first. Pure-Python via
``pathlib.Path.glob`` — no third-party dependency.
"""

import os
from pathlib import Path

from bridgic.core.agentic.tool_specs import FunctionToolSpec


# Cap the result list so a sloppy ``**/*`` does not flood the LLM.
MAX_RESULTS: int = 100


async def glob(pattern: str, path: str = "") -> str:
    """Find files whose path matches ``pattern``.

    Supports recursive globs like ``**/*.py`` and standard wildcards.
    Returns matching file paths sorted by mtime (most recently modified
    first) — handy for "what did I just touch?" queries.

    Parameters
    ----------
    pattern : str
        Glob pattern relative to ``path`` (e.g. ``"**/*.py"``,
        ``"src/**/*.ts"``).
    path : str
        Absolute directory to search in. Empty string means use the
        process's current working directory.
    """
    if not pattern or not pattern.strip():
        raise ValueError("pattern is required")

    search_root = path or os.getcwd()
    if not os.path.isabs(search_root):
        raise ValueError(f"path must be an absolute directory: {search_root}")
    if not os.path.isdir(search_root):
        raise NotADirectoryError(f"Search path is not a directory: {search_root}")

    matches = list(Path(search_root).glob(pattern))
    files = [m for m in matches if m.is_file()]
    if not files:
        return "(No files matched.)"

    files.sort(key=_safe_mtime, reverse=True)

    rendered = [str(f) for f in files[:MAX_RESULTS]]
    if len(files) > MAX_RESULTS:
        rendered.append(
            f"... [{len(files) - MAX_RESULTS} more matches truncated; "
            f"narrow the pattern to see them]"
        )
    return "\n".join(rendered)


def _safe_mtime(p: Path) -> float:
    """Return ``p``'s mtime, or ``-inf`` if stat fails (so it sorts last)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return float("-inf")


glob_tool: FunctionToolSpec = FunctionToolSpec.from_raw(glob)
