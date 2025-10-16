"""
The Args module provides parameter mapping and injection mechanisms in Automa.

This module provides arguments mapping rules and descriptors for implementing flexible 
argument passing, system parameter injection, and runtime context access in an Automa.
"""

from bridgic.core.types._common import ArgsMappingRule
from bridgic.core.automa.args._args_descriptor import From, System, RuntimeContext

__all__ = [
    "ArgsMappingRule",
    "From",
    "System",
    "RuntimeContext",
]