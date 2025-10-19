"""
The Model Protocols module defines high-level interface protocols for model interaction.

This module contains several important interface protocol definitions to provide 
capabilities needed in real-world application development, such as tool selection and 
structured output. These interfaces have clear input and output definitions and are 
"model-neutral", aiming to reduce the details developers need to consider when 
implementing features, thereby improving development efficiency.
"""

from bridgic.core.model.protocols._tool_selection import ToolSelection
from bridgic.core.model.protocols._structured_output import (
    Constraint,
    PydanticModel,
    JsonSchema,
    EbnfGrammar,
    LarkGrammar,
    Regex,
    Choice,
    RegexPattern,
    StructuredOutput,
)

__all__ = [
    "ToolSelection",
    "StructuredOutput",
    "Constraint",
    "PydanticModel",
    "JsonSchema",
    "EbnfGrammar",
    "LarkGrammar",
    "Regex",
    "RegexPattern",
    "Choice",
]