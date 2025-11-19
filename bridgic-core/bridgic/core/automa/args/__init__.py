"""
The Args module provides Arguments Mapping and Arguments Injection mechanisms in Bridgic.
"""

from bridgic.core.types._common import ArgsMappingRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext
from bridgic.core.automa.args._args_binding import Distribute, set_func_signature, set_method_signature, override_func_signature

__all__ = [
    "ArgsMappingRule",
    "From",
    "System",
    "RuntimeContext",
    "Distribute",
    "set_func_signature",
    "set_method_signature",
    "override_func_signature",
]