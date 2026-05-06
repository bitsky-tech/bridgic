"""Human-in-the-loop built-in tools.

Every ``AmphibiousAutoma`` agent automatically receives ``request_human``
in its ``context.tools`` during ``arun()`` (subject to the
``builtin_tools`` filter). The LLM can call it in any mode (AGENT,
WORKFLOW fallback, AMPHIFLOW) with no extra wiring::

    await agent.arun(goal="...", tools=[search_tool])

Importing and passing ``request_human_tool`` explicitly still works —
the injection step deduplicates by tool name::

    from bridgic.amphibious.builtin_tools import request_human_tool

    await agent.arun(goal="...", tools=[search_tool, request_human_tool])
"""

from .request_human import request_human_tool

__all__ = ["request_human_tool"]
