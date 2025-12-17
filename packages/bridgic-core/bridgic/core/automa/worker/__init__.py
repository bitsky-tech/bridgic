"""
The Worker module defines the Worker concept and its related implementations in Automa.

This module provides the core abstractions and implementations of Worker, including:

- **Worker**: The base class for all workers, which is the basic execution unit in Automa, 
  defining the execution interface (`arun()` and `run()` methods) for nodes
- **CallableWorker**: A worker implementation for wrapping callable objects (functions or methods)
- **WorkerCallback**: A callback interface during worker execution, supporting validation, 
  monitoring, and log collection before and after execution
- **WorkerCallbackBuilder**: A builder for constructing and configuring worker callbacks
"""

from ._worker import Worker
from ._callable_worker import CallableWorker
from ._worker_callback import WorkerCallback, WorkerCallbackBuilder, T_WorkerCallback

__all__ = [
    "Worker",
    "CallableWorker",
    "WorkerCallback",
    "WorkerCallbackBuilder",
    "T_WorkerCallback",
]