"""
Configuration management module for Bridgic.
"""

from ._global_setting import GlobalSetting

# Import WorkerCallbackBuilder to resolve forward references in GlobalSetting.
# This must be done after GlobalSetting is imported but before it's used.
from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder

# Rebuild the model to resolve forward references
GlobalSetting.model_rebuild()

__all__ = [
    "GlobalSetting",
]

