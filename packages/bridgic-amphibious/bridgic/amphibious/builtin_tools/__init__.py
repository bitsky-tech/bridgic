"""
Built-in tools for AmphibiousAutoma.

These are pre-packaged ToolSpec factories that can be injected into
an agent's tool set to enable framework-level capabilities (e.g.,
requesting human input during LLM-driven execution).
"""

from .human import human_request_tool, current_agent

__all__ = ["human_request_tool", "current_agent"]
