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

import textwrap
from pathlib import Path
from typing import Optional


_AMPHI_FILENAME = "amphi.py"

_AMPHI_PY = '''\
{task_comment}from bridgic.amphibious import (
    AmphibiousAutoma,
    OTAContext,
    Context,
    CognitiveWorker,
    think_unit,
    ActionCall,
    EnterAgent,
    HumanCall,
    LLMCall,
    ThinkUnit,
    RETURN,
    # Built-in tool specs — declared on the OTA context via OTAContext.tool
    # (below); nothing is auto-injected by the framework anymore.
    bash_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    glob_tool,
    grep_tool,
    request_human_tool,
)
from bridgic.core.model.types import Message, Role
from typing import Optional


# Small-loop context (framework-owned): this run's user_input + OTA round
# trace + tools. Subclass to fold custom per-run fields onto the current
# round from the before_action / after_action hooks (OTARecord is extra="allow").
class AmphiOTAContext(OTAContext):
    pass


# This is how tools are declared now — no auto-injection. The OTA context owns
# the tools it carries; register the framework builtins this run wants (plus
# any of your own) via OTAContext.tool. The small loop carries exactly what is
# declared here.
for _t in (bash_tool, read_file_tool, write_file_tool, edit_file_tool, glob_tool, grep_tool, request_human_tool):
    AmphiOTAContext.tool(_t)


# Big-loop context (free-form, optional): cross-turn knowledge (skills /
# memory / conversation). Override summary() to render it into the prompt;
# tools are an OTA-loop concern and live on the OTA context, not here.
# Use the base Context directly as the second generic when no custom
# big-loop state is needed; AmphibiousAutoma still requires both generics.
class AmphiBigContext(Context):
    pass


# Think worker: subclass CognitiveWorker and implement thinking() — assemble a
# prompt from the contexts and call the model; whatever you return, the
# framework adapts into a decision. Here: native tool-select over the tools the
# OTA context declares (the OTA loop owns the toolset).
class MainThink(CognitiveWorker):
    async def thinking(self, ota_context, context=None):
        messages = [Message.from_text(ota_context.summary(), role=Role.USER)]
        return await self._llm.aselect_tool(
            messages=messages,
            tools=[t.to_tool() for t in ota_context.tools],
        )


class Amphi(AmphibiousAutoma[AmphiOTAContext, AmphiBigContext]):
    # Think unit — one observe-think-act cycle driven by an LLM. Invoked
    # from on_agent via ``yield ThinkUnit("main_think")``.
    main_think = think_unit(MainThink(), max_attempts=10)

    # Agent mode: LLM-driven cognitive flow. Only ThinkUnit (named
    # think_units) and RETURN are allowed here — deterministic tool /
    # HITL / LLM calls belong in on_workflow or in worker hooks. Yield
    # RETURN(answer) to set the final answer; otherwise the framework
    # auto-captures from the finishing think step's step_content. Every
    # overridable template method takes the same pair: the per-run small-loop
    # ``ota_context`` and the optional big-loop ``context``.
    async def on_agent(self, ota_context: AmphiOTAContext, context: Optional[AmphiBigContext] = None):
        yield ThinkUnit("main_think")
        # TODO

    # Workflow mode: developer-defined deterministic steps. Implementing
    # both on_agent and on_workflow enables AMPHIFLOW (workflow-led, with
    # automatic agent fallback on atomic-Call failure). EnterAgent is the
    # explicit mode-switch primitive: workflow suspends, on_agent runs,
    # workflow resumes when on_agent's generator exhausts.
    async def on_workflow(self, ota_context: AmphiOTAContext, context: Optional[AmphiBigContext] = None):
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
