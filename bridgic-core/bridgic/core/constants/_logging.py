"""
Logging constants for Bridgic Core.

This module defines the logger names and other constants used throughout the Bridgic framework.
"""

ROOT_LOGGER_NAME = "bridgic_core"
"""Logger name used for root logger"""

EVENT_LOGGER_NAME = "bridgic_core.events"
"""Logger name used for structured event logging"""

TRACE_LOGGER_NAME = "bridgic_core.trace"
"""Logger name used for developer intended trace logging. The content and format of this log should not be depended upon."""

WORKER_LOGGER_NAME = "bridgic_core.workers"
"""Logger name used for worker-related logging"""

AUTOMA_LOGGER_NAME = "bridgic_core.automa"
"""Logger name used for automa-related logging"""

MODEL_LOGGER_NAME = "bridgic_core.model"
"""Logger name used for model-related logging"""

# Default source prefixes
DEFAULT_WORKER_SOURCE_PREFIX = "Worker"
"""Default prefix for worker source names"""

DEFAULT_AUTOMA_SOURCE_PREFIX = "Automa"
"""Default prefix for automa source names"""

DEFAULT_MODEL_SOURCE_PREFIX = "Model"
"""Default prefix for model source names"""

DEFAULT_CACHE_SOURCE_PREFIX = "Cache"
"""Default prefix for cache source names"""

# Status values
STATUS_SUCCESS = "success"
"""Success status value"""

STATUS_ERROR = "error"
"""Error status value"""

STATUS_START = "start"
"""Start status value"""

STATUS_END = "end"
"""End status value"""
