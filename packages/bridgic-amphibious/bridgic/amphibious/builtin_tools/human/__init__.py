"""
Human built-in tools for human-in-the-loop support.

Every ``AmphibiousAutoma`` agent automatically receives the built-in
``request_human`` tool in its ``context.tools`` during ``arun()``. The
LLM can call it in any mode (AGENT, WORKFLOW fallback, AMPHIFLOW) with
no extra wiring::

    # No need to pass request_human_tool — it is already available as `request_human`.
    await agent.arun(goal="...", tools=[search_tool])

If you want to be explicit, importing and passing ``request_human_tool``
still works — the injection step deduplicates by tool name::

    from bridgic.amphibious.builtin_tools import request_human_tool

    await agent.arun(goal="...", tools=[search_tool, request_human_tool])  # also fine
"""

from .request_human import request_human_tool, current_agent

__all__ = ["request_human_tool", "current_agent"]
