"""Helpers for the per-arun run-directory layout.

When ``AmphibiousAutoma.arun(workdir=...)`` is set, the framework lays
out a run directory under ``<workdir>/runs/<run_id>/``. ``AgentTrace``
owns persistence of the single unified ``trace.json`` (goal + metadata
+ history). ``_ThinkAgentRuntime`` places per-ThinkAgent subdirs at
``<run>/delegates/<n>/`` carrying claude stdout / stderr / mcp_config /
goal.txt / ctx_snapshot.json / trace_prefix.json.

This module is intentionally pure — no Automa coupling — so it can be
reused from both ``_amphibious_automa.py`` (the run-dir producer) and
``_think_agent.py`` (which places delegate subdirs underneath the
active run dir).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path


def make_run_id() -> str:
    """Generate a stable per-arun id (UTC timestamp + 4-byte hex suffix)."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def ensure_run_dir(workdir: Path, run_id: str) -> Path:
    """Create and return ``<workdir>/runs/<run_id>/``."""
    run_dir = workdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def next_delegate_subdir(run_dir: Path) -> Path:
    """Return the next available ``delegates/<NNN>/`` path inside ``run_dir``.

    The path is created on disk; numbering starts at 001 and is the
    successor of the highest existing numeric subdir, so multiple
    ThinkAgent invocations within one arun fall into sibling subdirs in
    yield order.
    """
    base = run_dir / "delegates"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    next_idx = (int(existing[-1]) + 1) if existing else 1
    subdir = base / f"{next_idx:03d}"
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


__all__ = [
    "make_run_id",
    "ensure_run_dir",
    "next_delegate_subdir",
]
