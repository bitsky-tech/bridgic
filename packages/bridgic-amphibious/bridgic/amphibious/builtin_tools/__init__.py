"""Built-in tools for AmphibiousAutoma.

These are pre-packaged ``FunctionToolSpec`` instances auto-injected by
``AmphibiousAutoma.arun()`` so the LLM can use them in any mode
(AGENT, WORKFLOW fallback, AMPHIFLOW) with no extra wiring.

Categories
----------
* ``human``       — request_human (human-in-the-loop)
* ``shell``       — bash (shell command execution)
* ``filesystem``  — read_file / write_file / edit_file / glob / grep

Selective injection
-------------------
Subclasses of ``AmphibiousAutoma`` can opt out of specific tools by
declaring a class-level ``builtin_tools`` attribute (a ``frozenset`` of
tool names to keep), or by passing ``arun(builtin_tools=[...])`` for
runtime control. ``None`` (the default) means "inject all".
"""

from ._agent_state import current_agent
from .human import request_human_tool
from .shell import bash_tool
from .filesystem import (
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    glob_tool,
    grep_tool,
)

# All built-in tools, in display order. ``AmphibiousAutoma`` reads this
# tuple to decide what to auto-inject (subject to the ``builtin_tools``
# filter). Adding a new built-in tool means importing its ToolSpec here
# and appending to this tuple — no other framework wiring needed.
ALL_BUILTIN_TOOLS = (
    request_human_tool,
    bash_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    glob_tool,
    grep_tool,
)

__all__ = [
    "current_agent",
    "ALL_BUILTIN_TOOLS",
    # Human
    "request_human_tool",
    # Shell
    "bash_tool",
    # Filesystem
    "read_file_tool",
    "write_file_tool",
    "edit_file_tool",
    "glob_tool",
    "grep_tool",
]
