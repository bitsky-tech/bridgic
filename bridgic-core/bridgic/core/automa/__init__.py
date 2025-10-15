"""
This module contains the core automa classes and functions.
"""

from bridgic.core.automa.automa import Automa, _InteractionAndFeedback, _InteractionEventException
from bridgic.core.automa.graph_automa import GraphAutoma
from bridgic.core.automa.arguments_descriptor import From, RuntimeContext, System
from bridgic.core.automa.worker_decorator import worker
from bridgic.core.types.common import ArgsMappingRule
from bridgic.core.types.error import *

__all__ = [
    "Automa",
    "_InteractionAndFeedback",
    "_InteractionEventException",
    "GraphAutoma",
    "worker",
    "AutomaCompilationError",
    "AutomaDeclarationError",
    "WorkerSignatureError",
    "WorkerArgsMappingError",
    "AutomaRuntimeError",
    "AutomaDataInjectionError",
    "ArgsMappingRule",
    "From",
    "RuntimeContext",
    "System",
]