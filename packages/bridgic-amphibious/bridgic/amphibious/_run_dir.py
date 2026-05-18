"""Helpers for the per-arun run-directory layout.

When ``AmphibiousAutoma.arun(workdir=...)`` is set, the framework lays
out a run directory under ``<workdir>/runs/<run_id>/`` and progressively
writes artifacts there:

* ``meta.json``         — arun parameters + resolved mode + start/end timing.
* ``ctx_initial.json``  — context snapshot at arun entry.
* ``ctx_final.json``    — context snapshot at arun exit (success or failure).
* ``trace.json``        — full AgentTrace dump (always captured when
                          workdir is set; ``trace_running=True`` is an
                          orthogonal in-memory-only flag).
* ``delegates/<n>/``    — per-ThinkAgent workdir; populated by
                          ``_ThinkAgentRuntime`` when ``ThinkAgent``
                          fires inside this run.

This module is intentionally pure — no Automa coupling — so it can be
reused from both ``_amphibious_automa.py`` (the producer of the run
dir) and ``_think_agent.py`` (which places delegate subdirs underneath
the active run dir).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def make_run_id() -> str:
    """Generate a stable per-arun id (UTC timestamp + 4-byte hex suffix)."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def serialize_ctx(ctx: Any) -> str:
    """Best-effort JSON dump of the most useful ctx fields.

    Defensive against partially-initialised contexts: every field is
    extracted under a try/except so a malformed Exposure cannot prevent
    a snapshot from being written.

    Captures: goal, observation, tool names, and a flattened cognitive
    history. Skills / per-field LayeredExposure ``_revealed`` state is
    omitted — adding it later is a straightforward extension when a
    consumer needs it.
    """
    try:
        tools = [getattr(t, "tool_name", None) for t in ctx.tools.get_all()]
    except Exception:
        tools = []
    try:
        history = [
            {
                "content": s.content,
                "result": str(s.result) if s.result is not None else None,
                "metadata": s.metadata if isinstance(s.metadata, dict) else None,
            }
            for s in ctx.cognitive_history.get_all()
        ]
    except Exception:
        history = []
    return json.dumps(
        {
            "goal": getattr(ctx, "goal", None),
            "observation": getattr(ctx, "observation", None),
            "tools": tools,
            "history": history,
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )


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
    "serialize_ctx",
    "ensure_run_dir",
    "next_delegate_subdir",
]
