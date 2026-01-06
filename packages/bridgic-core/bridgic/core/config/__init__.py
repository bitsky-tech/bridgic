"""
Configuration management module for Bridgic.
"""

from ._global_setting import GlobalSetting
from ._http_client_config import (
    HttpClientConfig,
    HttpClientTimeoutConfig,
    HttpClientAuthConfig,
    create_http_client_from_config,
)

# Import WorkerCallbackBuilder to resolve forward references in GlobalSetting.
# This must be done after GlobalSetting is imported but before it's used.
from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder

# Rebuild the model to resolve forward references
GlobalSetting.model_rebuild()

__all__ = [
    "GlobalSetting",
    "HttpClientConfig",
    "HttpClientTimeoutConfig",
    "HttpClientAuthConfig",
    "create_http_client_from_config",
]

