"""
The Args module provides Arguments Mapping and Arguments Injection mechanisms in Bridgic.
"""

from bridgic.core.types._common import ArgsMappingRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext, JSON_SCHEMA_IGNORE_ARG_TYPES

__all__ = [
    "ArgsMappingRule",
    "From",
    "System",
    "RuntimeContext",
    "JSON_SCHEMA_IGNORE_ARG_TYPES",
]