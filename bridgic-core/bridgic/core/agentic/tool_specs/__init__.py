"""
The Tool Specs module provides definitions and implementations of tool specifications.

This module contains various tool specification classes that support transforming 
"tool ingredients" such as Python functions and Automa workflows into LLM-callable tools, 
enabling callable objects to be seamlessly used in agentic systems.
"""

from ._base_tool_spec import ToolSpec
from ._function_tool_spec import FunctionToolSpec
from ._automa_tool_spec import AutomaToolSpec, as_tool


__all__ = [
    "ToolSpec",
    "FunctionToolSpec",
    "AutomaToolSpec",
    "as_tool",
]