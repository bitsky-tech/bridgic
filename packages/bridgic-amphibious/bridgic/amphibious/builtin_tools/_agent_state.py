"""Shared per-agent runtime state for built-in tools.

The ``current_agent`` ContextVar is set by ``AmphibiousAutoma.arun()`` at
task entry, allowing built-in tools (which run as plain async functions)
to access the running agent for shared state — e.g. the read-before-edit
tracker used by the filesystem tools, or the ``@human_channel`` registry
that ``request_human`` routes through via ``agent._run_human_call``.

Because each ``asyncio.Task`` gets its own ContextVar value, concurrent
agents never interfere with each other.
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridgic.amphibious._amphibious_automa import AmphibiousAutoma


current_agent: ContextVar["AmphibiousAutoma"] = ContextVar("current_agent")
