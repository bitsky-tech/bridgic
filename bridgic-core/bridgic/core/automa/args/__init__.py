"""
The Args module provides Arguments Mapping and Arguments Injection mechanisms in Bridgic.
"""

from bridgic.core.types._common import ArgsMappingRule, ResultDispatchingRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext
from bridgic.core.automa.args._args_binding import (
    InOrder, 
    set_func_signature, 
    set_method_signature, 
    override_func_signature, 
    safely_map_args
)

__all__ = [
    "ArgsMappingRule",
    "ResultDispatchingRule",
    "From",
    "System",
    "RuntimeContext",
    "InOrder",
    "set_func_signature",
    "set_method_signature",
    "override_func_signature",
    "safely_map_args"
]