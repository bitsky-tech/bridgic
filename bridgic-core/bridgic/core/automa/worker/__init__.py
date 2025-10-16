"""
The Worker module defines work nodes in Automa.

This module contains the base classes of workers, which will be used within Automa.
Workers are the basic execution units in Automa, with each work node typically 
corresponding to a function (which can be synchronous or asynchronous) responsible 
for executing specific business logic.
"""

from ._worker import Worker
from ._callable_worker import CallableWorker

__all__ = [
    "Worker",
    "CallableWorker",
]