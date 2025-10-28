"""
Constants module for Bridgic Core.

This module contains all constants used throughout the Bridgic framework.
"""

from ._logging import *

__all__ = [
    # Logger names
    "ROOT_LOGGER_NAME",
    "EVENT_LOGGER_NAME", 
    "TRACE_LOGGER_NAME",
    "WORKER_LOGGER_NAME",
    "AUTOMA_LOGGER_NAME",
    "MODEL_LOGGER_NAME",
    # Source prefixes
    "DEFAULT_WORKER_SOURCE_PREFIX",
    "DEFAULT_AUTOMA_SOURCE_PREFIX",
    "DEFAULT_MODEL_SOURCE_PREFIX",
    "DEFAULT_CACHE_SOURCE_PREFIX",
    # Status values
    "STATUS_SUCCESS",
    "STATUS_ERROR",
    "STATUS_START",
    "STATUS_END",
]
