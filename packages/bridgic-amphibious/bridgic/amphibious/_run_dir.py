"""Helpers for the per-arun run-directory layout.

When ``AmphibiousAutoma.arun(workdir=...)`` is set, the framework lays
out a run directory under ``<workdir>/runs/<run_id>/``. ``AgentTrace``
owns persistence of the single unified ``trace.json`` (goal + metadata
+ history) inside it — that one file is the run directory's only
artifact.

This module is intentionally pure — no Automa coupling — so it stays
trivially testable and reusable by ``_amphibious_automa.py`` (the
run-dir producer).
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


__all__ = [
    "make_run_id",
    "ensure_run_dir",
]
