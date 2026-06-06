"""Built-in tools for Amphibious agents.

Pre-packaged ``FunctionToolSpec`` instances the framework ships for
convenience. Nothing is auto-injected — an OTA context carries a built-in only
if you declare it on the context class via ``OTAContext.tool`` (decorator or
call), e.g. ``MyOTAContext.tool(bash_tool)``. Import ``ALL_BUILTIN_TOOLS`` to
declare the whole set at once.

Categories
----------
* ``human``       — request_human (HITL; resolves the running agent's
                    ``@human_channel`` registry at call time)
* ``shell``       — bash (shell command execution)
* ``filesystem``  — read_file / write_file / edit_file / glob / grep
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

# All built-in tool specs, in display order. Import this to declare the whole
# set on an OTA context at once (e.g. ``for t in ALL_BUILTIN_TOOLS: MyOTACtx.tool(t)``).
# Adding a new built-in means importing its ToolSpec here and appending it.
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
