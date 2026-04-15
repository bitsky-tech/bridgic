"""
Project scaffolding for Amphibious Automa projects.

Creates the standard directory structure for an amphibious automa project:

    <project_name>/
    ├── task.md                # Task description (input)
    ├── config.py              # LLM configuration
    ├── tools.py               # Tool definitions
    ├── workers.py             # Context, data models, custom workers
    ├── agents.py              # Amphibious agent code
    ├── skills/                # Amphibious skills (SKILL.md files)
    ├── result/                # Trace and analysis results
    └── log/                   # Runtime logs

Usage
-----
CLI::

    bridgic-amphibious create -n my_project
    bridgic-amphibious create -n my_project --task "Navigate to example.com and extract data"

Python API::

    from bridgic.amphibious.scaffold import create_project
    create_project("my_project", task="Navigate to example.com and extract data")
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────────────────────────────

_TASK_MD = """\
# Task Description

{task}
"""

_CONFIG_PY = '''\
"""LLM configuration for this project."""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
'''

_TOOLS_PY = '''\
"""Tool definitions for this project.

Define your tools here and export them for use in agents.py.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
'''

_WORKERS_PY = '''\
"""Context, data models, and custom workers for this project.

Define your custom CognitiveContext subclass, data models, and any
CognitiveWorker subclasses here.
"""

from typing import Optional

from pydantic import Field

from bridgic.amphibious import CognitiveContext


class ProjectContext(CognitiveContext):
    """Execution context for this project.

    Add custom fields as needed:
    - Runtime fields (hidden from LLM): use json_schema_extra={"display": False}
    - LLM-visible state: use EntireExposure or LayeredExposure subclasses
    """
    pass
'''

_AGENTS_PY = '''\
"""Amphibious agent code.

Define your AmphibiousAutoma subclass with on_agent() and on_workflow() methods here.
"""

# TODO: Implement your agent class here
'''


# ──────────────────────────────────────────────────────────────────────
# Scaffolding API
# ──────────────────────────────────────────────────────────────────────

def create_project(
    name: str,
    base_dir: Optional[str] = None,
    task: Optional[str] = None,
) -> Path:
    """Create a new amphibious automa project with standard directory structure.

    Parameters
    ----------
    name : str
        Project directory name.
    base_dir : str, optional
        Parent directory. Defaults to current working directory.
    task : str, optional
        Initial task description for task.md.

    Returns
    -------
    Path
        Path to the created project directory.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    project_dir = base / name

    if project_dir.exists():
        raise FileExistsError(f"Directory already exists: {project_dir}")

    # Create directory structure
    project_dir.mkdir(parents=True)
    (project_dir / "skills").mkdir()
    (project_dir / "result").mkdir()
    (project_dir / "log").mkdir()

    # Write template files
    task_text = task or "Describe your task here."
    _write(project_dir / "task.md", _TASK_MD.format(task=task_text))
    _write(project_dir / "config.py", _CONFIG_PY)
    _write(project_dir / "tools.py", _TOOLS_PY)
    _write(project_dir / "workers.py", _WORKERS_PY)
    _write(project_dir / "agents.py", _AGENTS_PY)

    return project_dir


def _write(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point: bridgic-amphibious <command> [options]
# ──────────────────────────────────────────────────────────────────────

def _print_tree(project_dir: Path) -> None:
    """Print the project directory tree."""
    print(f"\nProject structure:")
    for root, dirs, files in os.walk(project_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        level = len(Path(root).relative_to(project_dir).parts)
        indent = "  " * level
        print(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            print(f"{indent}  {f}")


def _cmd_create(args) -> None:
    """Handle the 'create' subcommand."""
    try:
        project_dir = create_project(args.name, args.base_dir, args.task)
        print(f"Created project: {project_dir}")
        _print_tree(project_dir)
    except FileExistsError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


def cli() -> None:
    """CLI entry point for ``bridgic-amphibious``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="bridgic-amphibious",
        description="Bridgic Amphibious — dual-mode agent framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              bridgic-amphibious create -n my_project
              bridgic-amphibious create -n my_project --task "Navigate to example.com"
        """),
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # ── create ──
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new Amphibious Automa project",
        description="Scaffold a project with standard directory structure (task.md, config.py, tools.py, workers.py, agents.py, etc.).",
    )
    create_parser.add_argument(
        "-n", "--name", required=True,
        help="Project directory name",
    )
    create_parser.add_argument(
        "--base-dir", default=None,
        help="Parent directory (default: current directory)",
    )
    create_parser.add_argument(
        "--task", default=None,
        help="Initial task description for task.md",
    )

    args = parser.parse_args()

    if args.command == "create":
        _cmd_create(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
