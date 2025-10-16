"""
Protocol definitions for LLM provider capabilities.

This module defines protocols that encapsulate different model calling capabilities,
enabling better utilization of model abilities through standardized interfaces.
These protocols provide a unified abstraction layer for working with various language
model providers while maintaining type safety and clear capability contracts.

The protocols abstract key model capabilities:

- **Tool Selection**: Models that can intelligently select and parameterize tools
- **Structured Output**: Models that can generate outputs in specific formats

By implementing these protocols, LLM providers can expose their advanced capabilities
in a consistent manner, allowing applications to leverage model-specific features
without being tightly coupled to particular implementations. This enables the creation
of flexible, provider-agnostic systems that can automatically adapt to the capabilities
of different model backends.
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