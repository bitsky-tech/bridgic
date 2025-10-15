"""
This module contains the core automa classes and functions.
"""

from bridgic.core.automa._automa import Automa, Snapshot
from bridgic.core.automa._graph_automa import GraphAutoma
from bridgic.core.automa.worker._worker_decorator import worker
from bridgic.core.types._error import *

__all__ = [
    "Automa",
    "GraphAutoma",
    "Snapshot",
    "worker",
    "WorkerSignatureError",
    "WorkerArgsMappingError",
    "WorkerArgsInjectionError",
    "WorkerRuntimeError",
    "AutomaCompilationError",
    "AutomaDeclarationError",
    "AutomaRuntimeError",
]