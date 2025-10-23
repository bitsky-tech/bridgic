"""
The Agentic Types module defines foundational data structures for agentic systems.

This module defines several important type definitions, such as `ToolSpec` and 
`ChatMessage`, which are designed to be "model-neutral" as much as possible, 
allowing developers to build agentic systems using different models.
"""

from bridgic.core.agentic.types._chat_message import *

__all__ = [
    "Function",
    "FunctionToolCall",
    "FunctionToolSpec",
    "ToolSpec",
    "ToolMessage",
    "ChatMessage",
    "SystemMessage",
    "UserTextMessage",
    "AssistantTextMessage",
]