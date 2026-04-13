"""
Human built-in tools for human-in-the-loop support.

Usage::

    from bridgic.amphibious.builtin_tools import human_request_tool

    await agent.arun(goal="...", tools=[search_tool, human_request_tool])
"""

from .request_human import human_request_tool, current_agent

__all__ = ["human_request_tool", "current_agent"]
