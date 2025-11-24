"""
The Args module provides Arguments Mapping and Arguments Injection mechanisms in Bridgic.
"""

from bridgic.core.types._common import ArgsMappingRule, ResultDispatchRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext
from bridgic.core.automa.args._args_binding import Distribute

__all__ = [
    "ArgsMappingRule",
    "ResultDispatchRule",
    "From",
    "System",
    "RuntimeContext",
    "Distribute"
]