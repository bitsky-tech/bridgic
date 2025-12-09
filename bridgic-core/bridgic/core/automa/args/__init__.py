"""
The Args module provides Arguments Mapping and Arguments Injection mechanisms in Bridgic.
"""

from bridgic.core.types._common import ArgsMappingRule, ResultDispatchingRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext
from bridgic.core.automa.args._args_binding import InOrder

__all__ = [
    "ArgsMappingRule",
    "ResultDispatchingRule",
    "From",
    "System",
    "RuntimeContext",
    "InOrder",
]