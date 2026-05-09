"""
Project scaffolding for Amphibious Automa projects.

Generates a single ``amphi.py`` file in the target directory (default: cwd)
containing a stub :class:`AmphibiousAutoma` subclass. Runtime concerns
(LLM credentials, entry-point script, etc.) are intentionally left to the
caller — the scaffold only seeds the agent definition.

Usage
-----
CLI::

    bridgic-amphibious create
    bridgic-amphibious create --task "Navigate to example.com and extract data"

Python API::

    from bridgic.amphibious.scaffold import create_project
    create_project(task="Navigate to example.com and extract data")
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Optional


_AMPHI_FILENAME = "amphi.py"

_AMPHI_PY = '''\
{task_comment}from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    think_unit,
    ActionCall,
    EnterAgent,
    HumanCall,
    LLMCall,
    ThinkUnit,
    RETURN,
)


# Custom context: extend with your own fields (Pydantic BaseModel under the hood).
class AmphiContext(CognitiveContext):
    pass


class Amphi(AmphibiousAutoma[AmphiContext]):
    # Think unit — one observe-think-act cycle driven by an LLM. Invoked
    # from on_agent via ``yield ThinkUnit("main_think")``.
    main_think = think_unit(
        CognitiveWorker.inline("Decide and execute the next step."),
        max_attempts=10,
    )

    # Agent mode: LLM-driven cognitive flow. Only ThinkUnit (named
    # think_units) and RETURN are allowed here — deterministic tool /
    # HITL / LLM calls belong in on_workflow or in worker hooks. Yield
    # RETURN(answer) to set the final answer; otherwise the framework
    # auto-captures from the finishing think step's step_content.
    async def on_agent(self, ctx: AmphiContext):
        yield ThinkUnit("main_think")
        # TODO

    # Workflow mode: developer-defined deterministic steps. Implementing
    # both on_agent and on_workflow enables AMPHIFLOW (workflow-led, with
    # automatic agent fallback on atomic-Call failure). EnterAgent is the
    # explicit mode-switch primitive: workflow suspends, on_agent runs,
    # workflow resumes when on_agent's generator exhausts.
    async def on_workflow(self, ctx: AmphiContext):
        # result = yield ActionCall("tool_name", arg="value")
        # feedback = yield HumanCall(prompt="Confirm?")
        # text = yield LLMCall.chat("Summarize the run")
        # yield EnterAgent(goal="Switch to agent mode for this sub-task")
        # yield RETURN("final answer")
        if False:
            yield  # makes this a proper async generator
        # TODO
'''


def create_project(
    base_dir: Optional[str] = None,
    task: Optional[str] = None,
) -> Path:
    """Generate ``amphi.py`` in the target directory.

    Parameters
    ----------
    base_dir : str, optional
        Target directory for the generated file. Defaults to the current
        working directory.
    task : str, optional
        Task description, injected as a top-of-file ``# Task: ...`` comment.
        Omitted when not provided.

    Returns
    -------
    Path
        Path to the generated ``amphi.py``.

    Raises
    ------
    FileExistsError
        If ``amphi.py`` already exists in the target directory.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    target = base / _AMPHI_FILENAME

    if target.exists():
        raise FileExistsError(f"File already exists: {target}")

    base.mkdir(parents=True, exist_ok=True)

    task_comment = f"# Task: {task}\n\n" if task else ""
    target.write_text(_AMPHI_PY.format(task_comment=task_comment), encoding="utf-8")

    return target


def _cmd_create(args) -> None:
    try:
        path = create_project(args.base_dir, args.task)
        print(f"Created: {path}")
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
              bridgic-amphibious create
              bridgic-amphibious create --task "Navigate to example.com"
        """),
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    create_parser = subparsers.add_parser(
        "create",
        help="Generate amphi.py in the current directory",
        description="Generate a single amphi.py stub in the target directory.",
    )
    create_parser.add_argument(
        "--base-dir", default=None,
        help="Target directory (default: current directory)",
    )
    create_parser.add_argument(
        "--task", default=None,
        help="Task description, injected as a top-of-file comment",
    )

    args = parser.parse_args()

    if args.command == "create":
        _cmd_create(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
